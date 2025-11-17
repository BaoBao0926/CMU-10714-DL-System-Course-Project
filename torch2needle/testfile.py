import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.fx import symbolic_trace
from torch_models import TorchMLP_v2

from torch2needle_converter import torch2needle_fx
from operator_fusion import fuse_operators  # 导入算子融合功能


torch_model = TorchMLP_v2()

# 查看 PyTorch 计算图
traced = symbolic_trace(torch_model)
print("="*80)
print("PyTorch 计算图:")
print("="*80)
print(traced.graph)
print()

# 转换为 Needle 模型
print("="*80)
print("转换为 Needle 模型:")
print("="*80)
needle_model, trace_log, torch_mapping = torch2needle_fx(torch_model)
print(needle_model)
print()

# 执行算子融合
print("="*80)
print("执行算子融合:")
print("="*80)
fused_model = fuse_operators(needle_model, verbose=True)
print()
print("融合后的模型结构:")
print(fused_model)