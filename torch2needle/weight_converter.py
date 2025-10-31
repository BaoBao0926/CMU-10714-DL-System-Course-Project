import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from needle import Tensor
from needle.nn.nn_basic import Linear, BatchNorm1d, LayerNorm1d
from torch_models import *
from torch2needle_converter import *


def load_torch_weights_by_mapping(layer_mapping, verbose=True):
    """
    根据 torch→needle 映射表复制权重
    """
    copied = 0
    for torch_layer, needle_layer in layer_mapping.items():
        try:
            # === Linear ===
            if isinstance(torch_layer, nn.Linear) and isinstance(needle_layer, Linear):
                needle_layer.weight.cached_data = np.reshape(
                    torch_layer.weight.detach().T.cpu().numpy().astype(np.float32),
                    (torch_layer.in_features, torch_layer.out_features)
                )
                if torch_layer.bias is not None:
                    needle_layer.bias.cached_data = np.reshape(
                        torch_layer.bias.detach().cpu().numpy().astype(np.float32),
                        (torch_layer.out_features,)
                    )
                copied += 1

            # === BatchNorm1d ===
            elif isinstance(torch_layer, nn.BatchNorm1d) and isinstance(needle_layer, BatchNorm1d):
                needle_layer.weight.cached_data = Tensor(torch_layer.weight.detach().numpy())
                needle_layer.bias.cached_data = Tensor(torch_layer.bias.detach().numpy())
                needle_layer.running_mean = Tensor(torch_layer.running_mean.detach().numpy())
                needle_layer.running_var = Tensor(torch_layer.running_var.detach().numpy())
                if verbose: print(f"[✔] Copied BatchNorm1d({torch_layer.num_features})")
                copied += 1

            # === LayerNorm ===
            elif isinstance(torch_layer, nn.LayerNorm) and isinstance(needle_layer, LayerNorm1d):
                needle_layer.weight.cached_data = Tensor(torch_layer.weight.detach().numpy())
                needle_layer.bias.cached_data = Tensor(torch_layer.bias.detach().numpy())
                if verbose: print(f"[✔] Copied LayerNorm1d({torch_layer.normalized_shape})")
                copied += 1

            else:
                if verbose:
                    print(f"[skip] Unsupported mapping: {type(torch_layer).__name__} → {type(needle_layer).__name__}")

        except Exception as e:
            print(f"[error] {type(torch_layer).__name__} → {type(needle_layer).__name__}: {e}")

    if verbose:
        print(f"\n✅ Successfully copied {copied}/{len(layer_mapping)} layers.")
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


