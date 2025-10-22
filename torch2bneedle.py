from needle.nn.nn_basic import Identity, Linear, Flatten, ReLU, Sequential, BatchNorm1d, LayerNorm1d, Dropout, Residual, SoftmaxLoss
from needle import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import needle.init as init
import random

class TorchMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def torch2needle(torch_model):
    """
    Convert a torch.nn.Module (Sequential-like) into a needle.nn.Sequential model.
    ONLY build the structure, no weights are copied.
    """
    layer_map = []
    for name, layer in torch_model.named_children():
        if isinstance(layer, nn.Sequential):
            # 递归转换
            layer_map.append(torch2needle(layer))
        elif isinstance(layer, nn.Linear):
            layer_map.append(Linear(layer.in_features, layer.out_features))
        elif isinstance(layer, nn.ReLU):
            layer_map.append(ReLU())
        elif isinstance(layer, nn.Flatten):
            layer_map.append(Flatten())
        elif isinstance(layer, nn.Dropout):
            layer_map.append(Dropout(layer.p))
        elif isinstance(layer, nn.BatchNorm1d):
            layer_map.append(BatchNorm1d(layer.num_features))
        elif isinstance(layer, nn.LayerNorm):
            layer_map.append(LayerNorm1d(layer.normalized_shape[0]))
        else:
            print(f"[Warning] Unsupported layer {name} ({layer.__class__.__name__}), replaced by Identity().")
            layer_map.append(Identity())

    # needle 的 Sequential 接收 *modules
    return Sequential(*layer_map)


def print_needle_model_detailed(model, indent=0):
    prefix = " " * indent
    if hasattr(model, "modules"):
        print(f"{prefix}{model.__class__.__name__}(")
        for m in model.modules:
            print_needle_model_detailed(m, indent + 2)
        print(f"{prefix})")
    else:
        print(f"{prefix}{model.__class__.__name__}", end="")
        if hasattr(model, "weight"):
            print(f"  weight shape: {getattr(model.weight, 'shape', None)}", end="")
        if hasattr(model, "bias") and model.bias is not None:
            print(f", bias shape: {getattr(model.bias, 'shape', None)}", end="")
        if hasattr(model, "in_features"):
            print(f", in_features: {model.in_features}", end="")
        if hasattr(model, "out_features"):
            print(f", out_features: {model.out_features}", end="")
        if hasattr(model, "p"):
            print(f", p: {model.p}", end="")
        print()

def flatten_needle_layers(model):
    layers = []
    if hasattr(model, "modules"):
        for m in model.modules:
            if isinstance(m, Sequential):
                layers.extend(flatten_needle_layers(m))
            else:
                layers.append(m)
    return layers


def load_weight_from_torch(needle_model, torch_model):
    torch_layers = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    needle_layers = flatten_needle_layers(needle_model)
    needle_layers = [m for m in needle_layers if isinstance(m, Linear)]

    print(f"Found {len(torch_layers)} torch Linear layers, {len(needle_layers)} needle Linear layers")

    for t_layer, n_layer in zip(torch_layers, needle_layers):
        print("Loading Linear layer weights...")
        # ✅ 转置 torch 权重
        w = t_layer.weight.detach().numpy().T.copy()
        n_layer.weight = Tensor(w)
        if t_layer.bias is not None:
            n_layer.bias = Tensor(t_layer.bias.detach().numpy().copy())


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# 1. 构建模型
torch_model = TorchMLP()
needle_model = torch2needle(torch_model)

# 2. 加载权重
load_weight_from_torch(needle_model, torch_model)

needle_model.eval()

# 3.1 torch 前向
x_torch = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], dtype=torch.float32)
y_torch = torch_model(x_torch)
print("Torch output:", y_torch.detach().numpy())

# 3.2 nedle 前向
x_needle = Tensor(np.array([[1,2,3,4,5,6,7,8,9,10]], dtype=np.float32))
y_needle = needle_model(x_needle)
print_needle_model_detailed(needle_model)

print("Needle output:", y_needle.data)

