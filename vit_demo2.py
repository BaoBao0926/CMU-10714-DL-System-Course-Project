import torch
import numpy as np
from pytorch_pretrained_vit import ViT
import needle as ndl 

# 不引入 needle 顶层模块，避免循环 import！
from needle.autograd import Tensor

# Torch2Needle
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping


# ==============================================================  
# Step 0: 加载 PyTorch ViT  
# ==============================================================  
print("\n===== Step 0: 加载 PyTorch ViT 模型 =====")

vit = ViT('B_16_imagenet1k', pretrained=True)
vit.eval()

print(vit)


# ==============================================================  
# Step 1: 构造可转换子模型（Conv2d + AvgPool + Flatten + Linear）  
# ==============================================================  
print("\n===== Step 1: 构造可转换子模型（Conv2d patch + AvgPool + Flatten + FC） =====")

class MiniViT(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.patch = vit.patch_embedding
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc = vit.fc

    def forward(self, x):
        x = self.patch(x)        # (N,768,H,W)
        x = self.pool(x)         # (N,768,1,1)
        x = self.flatten(x)      # (N,768)
        x = self.fc(x)           # (N,1000)
        return x


torch_model = MiniViT(vit).eval()
print(torch_model)


# ==============================================================  
# Step 2: PyTorch 推理  
# ==============================================================  
print("\n===== Step 2: PyTorch 推理 =====")

input_shape = (1, 3, 384, 384)
test_input = torch.randn(*input_shape)

with torch.no_grad():
    torch_output = torch_model(test_input)

print("PyTorch 输出形状：", torch_output.shape)


# ==============================================================  
# Step 3: 转换 Needle 模型  
# ==============================================================  
print("\n===== Step 3: 转换到 Needle =====")

# device: ndl.cpu(), ndl.cuda()
device = ndl.cpu() 
dtype = "float32"

needle_model, trace_log, torch_mapping_needle = torch2needle_fx(
    torch_model,
    device=device,   # 字符串即可
    dtype=dtype
)

print("\nNeedle 模型结构：")
print(needle_model)


# ==============================================================  
# Step 4: 加载权重  
# ==============================================================  
print("\n===== Step 4: 加载权重 =====")

load_torch_weights_by_mapping(
    torch_mapping_needle,
    verbose=True,
    device=device,
    dtype=dtype
)

needle_model.eval()


# ==============================================================  
# Step 5: Needle 推理  
# ==============================================================  
print("\n===== Step 5: 验证 Needle 输出 =====")

needle_input = Tensor(test_input.numpy(), device=device, dtype=dtype)
needle_output = needle_model(needle_input)

diff = np.abs(torch_output.detach().numpy() - needle_output.numpy())
max_diff = diff.max()

print("\n最大误差：", max_diff)

if max_diff < 1e-5:
    print("\n🎉🎉🎉 转换正确！")
else:
    print("\n❌ 存在转换误差！")
