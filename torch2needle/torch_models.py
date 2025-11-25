import torch
import torch.nn as nn

class TorchMLP_v1(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=5, output_dim=1):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net1(x)


class TorchMLP_v2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=5, output_dim=1):
        super().__init__()
        self.net = nn.Linear(output_dim, output_dim)
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.net3 = nn.Linear(output_dim, output_dim)


    def forward(self, x):
        x = self.net(self.net1(x) + self.net2(x) + self.net1(x))
        x = x - self.net3(x)
        return x


class TorchTransformerEncoderLayer(nn.Module):
    """Simple model with a single TransformerEncoderLayer"""
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x):
        return self.layer(x)


class TorchTransformerModel(nn.Module):
    """Simple Transformer model"""
    def __init__(self, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1, vocab_size=1000, max_seq_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        self.transformer_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Embedding
        x = self.embedding(x) + self.pos_encoder(positions)
        
        # Transformer layers
        x = self.transformer_layers(x)
        
        # Output projection
        x = self.fc_out(x)
        return x


class TorchMultiheadAttentionModel(nn.Module):
    """Simple model with MultiheadAttention"""
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_output

