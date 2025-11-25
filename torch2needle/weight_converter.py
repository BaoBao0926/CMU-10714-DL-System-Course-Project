import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import needle as ndl
from needle import backend_ndarray as nd
from needle.nn.nn_basic import Linear, BatchNorm1d, LayerNorm1d, BatchNorm2d, ReLU
from needle.nn.nn_conv import Conv

# Try to import optional pooling layers
try:
    from needle.nn.nn_conv import MaxPool2d, AdaptiveAvgPool2d
except ImportError:
    MaxPool2d = None
    AdaptiveAvgPool2d = None

# 尝试导入 torch_models 和 torch2needle_converter（仅用于测试）
try:
    from torch_models import *
    from torch2needle_converter import *
except ImportError:
    pass


def load_torch_weights_by_mapping(layer_mapping, verbose=True, device=ndl.cpu(), dtype="float32"):
    """
    according to torch→needle mapping copy weights
    """

    copied = 0
    skipped = 0
    
    for torch_layer, needle_layer in layer_mapping.items():
        try:
            # Jump Sequential (it has no weights itself)
            if isinstance(torch_layer, nn.Sequential):
                skipped += 1
                continue
            # === Linear ===
            if isinstance(torch_layer, nn.Linear) and isinstance(needle_layer, Linear):
                needle_layer.weight = ndl.Tensor(np.reshape(
                    torch_layer.weight.detach().T.cpu().numpy().astype(np.float32),
                    (torch_layer.in_features, torch_layer.out_features)
                ),device=device,dtype=dtype)
                if torch_layer.bias is not None:
                    needle_layer.bias = ndl.Tensor(np.reshape(
                        torch_layer.bias.detach().cpu().numpy().astype(np.float32),
                        (torch_layer.out_features,)
                    ),device=device,dtype=dtype)
                if verbose: print(f"[✔] Copied Linear({torch_layer.in_features}, {torch_layer.out_features})")
                copied += 1
            # === BatchNorm1d ===
            elif isinstance(torch_layer, nn.BatchNorm1d) and isinstance(needle_layer, BatchNorm1d):
                needle_layer.weight = ndl.Tensor(np.array(torch_layer.weight.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                needle_layer.bias = ndl.Tensor(np.array(torch_layer.bias.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                needle_layer.running_mean = ndl.Tensor(torch_layer.running_mean.detach().cpu().numpy().astype(np.float32),device=device,dtype=dtype)
                needle_layer.running_var = ndl.Tensor(torch_layer.running_var.detach().cpu().numpy().astype(np.float32),device=device,dtype=dtype)
                if verbose: print(f"[✔] Copied BatchNorm1d({torch_layer.num_features})")
                copied += 1
            # === LayerNorm ===
            elif isinstance(torch_layer, nn.LayerNorm) and isinstance(needle_layer, LayerNorm1d):
                needle_layer.weight = ndl.Tensor(np.array(torch_layer.weight.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                needle_layer.bias = ndl.Tensor(np.array(torch_layer.bias.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                if verbose: print(f"[✔] Copied LayerNorm1d({torch_layer.normalized_shape})")
                copied += 1
            
            # === Conv2d ===
            elif isinstance(torch_layer, nn.Conv2d) and isinstance(needle_layer, Conv):
                # PyTorch: (out_channels, in_channels, kernel_height, kernel_width)
                # Needle: (kernel_height, kernel_width, in_channels, out_channels)
                torch_weight = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
                # 转置: (out_ch, in_ch, kh, kw) -> (kh, kw, in_ch, out_ch)
                needle_weight = np.transpose(torch_weight, (2, 3, 1, 0))
                needle_layer.weight = ndl.Tensor(needle_weight, device=device, dtype=dtype)
                
                if torch_layer.bias is not None:
                    needle_layer.bias = ndl.Tensor(
                        torch_layer.bias.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                if verbose: 
                    print(f"[✔] Copied Conv2d({torch_layer.in_channels}, {torch_layer.out_channels}, kernel_size={torch_layer.kernel_size})")
                copied += 1
            
            # === BatchNorm2d ===
            elif isinstance(torch_layer, nn.BatchNorm2d) and isinstance(needle_layer, BatchNorm2d):
                needle_layer.weight = ndl.Tensor(np.array(torch_layer.weight.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                needle_layer.bias = ndl.Tensor(np.array(torch_layer.bias.detach().cpu().numpy().astype(np.float32)),device=device,dtype=dtype)
                needle_layer.running_mean = ndl.Tensor(torch_layer.running_mean.detach().cpu().numpy().astype(np.float32),device=device,dtype=dtype)
                needle_layer.running_var = ndl.Tensor(torch_layer.running_var.detach().cpu().numpy().astype(np.float32),device=device,dtype=dtype)
                if verbose: print(f"[✔] Copied BatchNorm2d({torch_layer.num_features})")
                copied += 1
            
            # === ReLU (no weights to copy) ===
            elif isinstance(torch_layer, nn.ReLU) and isinstance(needle_layer, ReLU):
                # ReLU has no learnable parameters
                if verbose: print(f"[✔] ReLU (no weights)")
                copied += 1
            
            # === MaxPool2d (no weights to copy) ===
            elif MaxPool2d is not None and isinstance(torch_layer, nn.MaxPool2d) and isinstance(needle_layer, MaxPool2d):
                # MaxPool2d has no learnable parameters
                if verbose: print(f"[✔] MaxPool2d (no weights)")
                copied += 1
            
            # === AdaptiveAvgPool2d (no weights to copy) ===
            elif AdaptiveAvgPool2d is not None and isinstance(torch_layer, nn.AdaptiveAvgPool2d) and isinstance(needle_layer, AdaptiveAvgPool2d):
                # AdaptiveAvgPool2d has no learnable parameters
                if verbose: print(f"[✔] AdaptiveAvgPool2d (no weights)")
                copied += 1

            else:
                if verbose:
                    print(f"[⚠] Unsupported mapping: {type(torch_layer).__name__} → {type(needle_layer).__name__}")
                skipped += 1

        except Exception as e:
            print(f"[❌] Error copying {type(torch_layer).__name__} → {type(needle_layer).__name__}: {e}")

    if verbose:
        print(f"\n✅ Successfully copied {copied}/{len(layer_mapping)} layers ({skipped} skipped).")
    return layer_mapping


if __name__ == "__main__":
    input_tensor = torch.tensor([[1.,2.,3.]])
    print("=== Torch → Needle Weight Converter Test ===")
    torch_model = TorchMLP_v1()
    torch_model.eval()
    torch_output = torch_model(input_tensor)
    print("torch output is: ", torch_output)

    # get needle_model and load weight to needle model
    needle_model, trace_log, torch_mapping_needle = torch2needle_fx(torch_model)
    device = ndl.cpu_numpy()
    dtype = "float32"
    load_torch_weights_by_mapping(torch_mapping_needle, device=device, dtype=dtype)
    print("\n✅ Weight transfer complete!")

    needle_input = Tensor([[1.,2.,3.]], device=device, dtype=dtype)
    needle_output = needle_model(needle_input)
    print("needle output is: ", needle_output)


