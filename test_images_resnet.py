"""
测试 20 张图片：Torch ResNet18 vs Needle ResNet18
"""

import os
from PIL import Image
import numpy as np

import torch
from torchvision import models, transforms

import needle as ndl
from needle.autograd import Tensor

# Torch2Needle 工具
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping


# ======================================================
# Step 1 — Torch 模型
# ======================================================
print("\n===== 加载 PyTorch ResNet18 =====")

torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch_model.eval()

# Torch 图像预处理（ResNet 标准）
preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# ======================================================
# Step 2 — Needle 模型
# ======================================================
print("\n===== 转换为 Needle 模型 =====")

device = ndl.cuda()
dtype = "float32"

needle_model, trace_log, mapping = torch2needle_fx(
    torch_model,
    device=device,
    dtype=dtype
)

print("加载 Torch 权重 → Needle")
load_torch_weights_by_mapping(mapping, verbose=True, device=device, dtype=dtype)
needle_model.eval()


# ======================================================
# Step 3 — 加载 images/ 中的 20 张图片
# ======================================================
print("\n===== 加载 images 文件夹下 20 张图片 =====")

IMAGE_DIR = "./images"

images = []
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"Found {len(image_files)} images.")

for fname in image_files:
    path = os.path.join(IMAGE_DIR, fname)
    try:
        img = Image.open(path).convert("RGB")
        images.append(img)
    except:
        print(f"❌ Failed to load {fname}")

print(f"Loaded {len(images)} images.\n")

if len(images) == 0:
    raise RuntimeError("No images loaded!")


# ======================================================
# Step 4 — 定义推理函数（Torch / Needle）
# ======================================================
def torch_predict(img):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = torch_model(x)
    prob = torch.softmax(out[0], dim=0)
    pred = prob.argmax().item()
    return pred

def needle_predict(img):
    x = preprocess(img).unsqueeze(0)             # → torch tensor
    x_np = x.numpy().astype("float32")           # → numpy
    x_ndl = Tensor(x_np, device=device)          # → needle Tensor

    out = needle_model(x_ndl)
    prob = out.numpy()[0]
    pred = np.argmax(prob)
    return pred


# ======================================================
# Step 5 — 对比 20 张图片
# ======================================================
print("===== 开始推理 =====")

torch_correct = 0
needle_correct = 0

for i, img in enumerate(images):
    torch_pred = torch_predict(img)
    needle_pred = needle_predict(img)

    print(f"[{i+1:02d}] Torch={torch_pred:4d} | Needle={needle_pred:4d}")

    if torch_pred == needle_pred:
        needle_correct += 1
    torch_correct += 1   # torch 自己就是 ground truth baseline（比较模型差异）


# ======================================================
# Step 6 — 输出结果
# ======================================================
torch_acc = torch_correct / len(images) * 100
needle_acc = needle_correct / len(images) * 100

print("\n================ 结果总结 ================")
print(f"Torch accuracy:   {torch_acc:.1f}%  ({torch_correct}/{len(images)})")
print(f"Needle accuracy:  {needle_acc:.1f}%  ({needle_correct}/{len(images)})")
print(f"\nAccuracy diff: {abs(torch_acc - needle_acc):.1f}%")
print("==========================================\n")

print("🎉 测试完成！")
