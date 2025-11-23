import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import needle as ndl
from needle import backend_ndarray as nd
from needle.nn.nn_basic import Linear, BatchNorm1d, LayerNorm1d

# 尝试导入 torch_models 和 torch2needle_converter（仅用于测试）
try:
    from torch_models import *
    from torch2needle_converter import *
except ImportError:
    pass


def load_torch_weights_by_mapping(layer_mapping, verbose=True,device=ndl.cpu(),dtype="float32"):
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
            
            # TODO: we should add more layers here in the future

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
    load_torch_weights_by_mapping(torch_mapping_needle)
    print("\n✅ Weight transfer complete!")

    needle_input = Tensor([[1.,2.,3.]])
    needle_output = needle_model(needle_input)
    print("needle output is: ", needle_output)


