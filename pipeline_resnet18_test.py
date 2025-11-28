import torch
import numpy as np
import needle as ndl
from needle import Tensor
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping

from test_torch_model import ResNetConv18   # 你已经有这个模型定义


# ===============================
# Step 1: 准备 Torch ResNetConv18
# ===============================
def test_resnet18_pipeline():

    # 构建模型
    torch_model = ResNetConv18(num_classes=10).eval()

    # 构造 32×32 CIFAR10 输入
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        torch_out = torch_model(x)


    # ===============================
    # Step 2: 转换 Needle 模型
    # ===============================
    device = ndl.cuda()
    dtype = "float32"

    needle_model, trace_log, mapping = torch2needle_fx(
        torch_model,
        device=device,
        dtype=dtype
    )

    # ===============================
    # Step 3: 加载权重
    # ===============================
    load_torch_weights_by_mapping(
        mapping, verbose=True,
        device=device, dtype=dtype
    )

    needle_model.eval()

    # ===============================
    # Step 4: Needle 推理
    # ===============================
    nd_x = Tensor(x.numpy(), device=device, dtype=dtype)
    nd_out = needle_model(nd_x)

    # ===============================
    # Step 5: 误差比较
    # ===============================
    diff = np.abs(torch_out.numpy() - nd_out.numpy())
    print("\nMax diff =", diff.max())

    return diff.max()


if __name__ == "__main__":
    d = test_resnet18_pipeline()
    print("Final diff =", d)
