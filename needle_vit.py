import needle as ndl
from needle.autograd import Tensor
from needle.nn.nn_basic import Module, Linear, Dropout, LayerNorm1d, Sequential, ReLU
from needle.nn.nn_conv import Conv as Conv2d
from needle.nn.nn_transformer import MultiHeadAttention
from needle import ops


class MLPBlock(Module):
	def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0, device=None, dtype="float32"):
		super().__init__()
		self.fc1 = Linear(in_features, hidden_features, device=device, dtype=dtype)
		self.act = ReLU()
		self.drop1 = Dropout(dropout)
		self.fc2 = Linear(hidden_features, in_features, device=device, dtype=dtype)
		self.drop2 = Dropout(dropout)

	def forward(self, x: Tensor) -> Tensor:
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop1(x)
		x = self.fc2(x)
		x = self.drop2(x)
		return x


class EncoderBlock(Module):
	def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float = 0.0, device=None, dtype="float32"):
		super().__init__()
		self.ln_1 = LayerNorm1d(embed_dim)
		self.self_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, device=device, dtype=dtype)
		self.attn_out_proj = Linear(embed_dim, embed_dim, device=device, dtype=dtype)
		self.dropout = Dropout(dropout)
		self.ln_2 = LayerNorm1d(embed_dim)
		self.mlp = MLPBlock(embed_dim, mlp_hidden_dim, dropout=dropout, device=device, dtype=dtype)

	def forward(self, x: Tensor) -> Tensor:
		# x: (B, L, C)
		h = self.ln_1(x)
		attn = self.self_attention(h, h, h)
		attn = self.attn_out_proj(attn)
		attn = self.dropout(attn)
		x = x + attn
		h2 = self.ln_2(x)
		mlp_out = self.mlp(h2)
		x = x + mlp_out
		return x


class Encoder(Module):
	def __init__(self, num_layers: int, embed_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float = 0.0, device=None, dtype="float32"):
		super().__init__()
		self.dropout = Dropout(dropout)
		self.layers = Sequential(*[
			EncoderBlock(embed_dim, num_heads, mlp_hidden_dim, dropout=dropout, device=device, dtype=dtype)
			for _ in range(num_layers)
		])
		self.ln = LayerNorm1d(embed_dim)

	def forward(self, x: Tensor) -> Tensor:
		x = self.dropout(x)
		x = self.layers(x)
		x = self.ln(x)
		return x


class VisionTransformer(Module):
	def __init__(self, image_size: int = 224, patch_size: int = 16,
				 in_channels: int = 3, embed_dim: int = 768,
				 num_layers: int = 12, num_heads: int = 12,
				 mlp_hidden_dim: int = 3072, num_classes: int = 1000,
				 dropout: float = 0.0, device=None, dtype="float32"):
		super().__init__()
		self.image_size = image_size
		self.patch_size = patch_size
		self.conv_proj = Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), device=device, dtype=dtype)
		# class token as Parameter shape (1,1,C)
		self.class_token = ndl.init.zeros((1, 1, embed_dim), device=device, dtype=dtype)
		# positional embedding for (1 + N_patches)
		num_patches = (image_size // patch_size) * (image_size // patch_size)
		self.encoder = Encoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, device=device, dtype=dtype)
		self.pos_embedding = ndl.init.zeros((1, num_patches + 1, embed_dim), device=device, dtype=dtype)
		self.heads = Sequential(Linear(embed_dim, num_classes, device=device, dtype=dtype))

	def forward(self, x: Tensor) -> Tensor:
		# x: (B, C, H, W)
		b, c, h, w = x.shape
		# patch projection: (B, C, H, W) -> (B, embed_dim, H/ps, W/ps)
		x = self.conv_proj(x)
		# flatten patches: (B, C, H', W') -> (B, C, H'*W')
		b2, c2, hp, wp = x.shape
		x = ops.reshape(x, (b2, c2, hp * wp))
		# (B, C, L) -> (B, L, C)
		x = ops.transpose(x, (0, 2, 1))
		# prepend class token
		cls = ops.broadcast_to(self.class_token, (b, 1, c2))
		x = ops.concatenate([cls, x], axis=1)
		# add positional embedding
		x = x + self.pos_embedding
		# encoder
		x = self.encoder(x)
		# take [CLS] token (index 0 along seq dim)
		cls_vec = ops.slice(x, [(0, b, 1), (0, 1, 1), (0, c2, 1)])
		cls_vec = ops.squeeze(cls_vec, axes=(1,))
		# head
		out = self.heads(cls_vec)
		return out


# Torch→Needle layer映射参考（vit_layers.txt对应）：
# - Conv2d(3, 768, kernel_size=16, stride=16) -> needle.nn.nn_conv.Conv(in_channels=3, out_channels=768, kernel_size=(16,16), stride=(16,16))
# - LayerNorm((768,), eps=1e-6) -> needle.nn.nn_basic.LayerNorm1d(768)
# - MultiheadAttention(out_proj=Linear(768,768)) -> needle.nn.nn_transformer.MultiHeadAttention(embed_dim=768, num_heads=12) + Linear(768,768)
# - Dropout(p=0.0) -> needle.nn.nn_basic.Dropout(0.0)
# - GELU -> 使用 needle.nn.nn_basic.ReLU 近似或自定义激活；此实现采用 ReLU
# - MLPBlock: Linear(768->3072) + GELU + Dropout + Linear(3072->768) + Dropout -> MLPBlock(embedding=768, hidden=3072)
# - EncoderBlock: ln_1 + self_attention + out_proj + dropout + residual + ln_2 + mlp + residual -> EncoderBlock
# - Encoder(layers=12, ln) -> Encoder(num_layers=12) + LayerNorm1d
# - heads: Sequential(Linear(768->1000)) -> Sequential(Linear)

