import torch
import numpy as np
import needle as ndl
from needle import Tensor

from tiny_cifar_loader import get_20_cifar10_loader
from test_torch_model import ResNetConv18   # 你已有
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping


def evaluate_both_models():

    # ==============================
    # 1. 创建 Torch 模型
    # ==============================
    torch_model = ResNetConv18(num_classes=10).eval()

    # ==============================
    # 2. 转 Needle 模型
    # ==============================
    device = ndl.cpu()
    dtype = "float32"

    needle_model, trace_log, mapping = torch2needle_fx(
        torch_model,
        device=device,
        dtype=dtype
    )

    # 加载权重
    load_torch_weights_by_mapping(
        mapping, verbose=False,
        device=device, dtype=dtype
    )
    needle_model.eval()


    # ==============================
    # 3. 从 CIFAR10 取 20 张图测试
    # ==============================
    loader = get_20_cifar10_loader()

    torch_correct = 0
    needle_correct = 0

    for img, label in loader:

        # Torch 推理
        with torch.no_grad():
            t_out = torch_model(img)
            t_pred = t_out.argmax(dim=1).item()

        # Needle 推理
        nd_img = Tensor(img.numpy(), device=device, dtype=dtype)
        nd_out = needle_model(nd_img)
        n_pred = np.argmax(nd_out.numpy(), axis=1)[0]

        # accuracy 统计
        if t_pred == label.item():
            torch_correct += 1
        if n_pred == label.item():
            needle_correct += 1

    # ==============================
    # 打印结果
    # ==============================
    print(f"\nTorch accuracy:  {torch_correct}/20 = {torch_correct/20*100:.1f}%")
    print(f"Needle accuracy: {needle_correct}/20 = {needle_correct/20*100:.1f}%")
    print("\nAccuracy diff:", abs(torch_correct - needle_correct))


if __name__ == "__main__":
    evaluate_both_models()
