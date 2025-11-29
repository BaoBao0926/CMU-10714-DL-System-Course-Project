import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import needle as ndl
from needle import backend_ndarray as nd
from needle.nn.nn_basic import Linear, BatchNorm1d, LayerNorm1d, BatchNorm2d, ReLU
from needle.nn.nn_conv import Conv, ConvTranspose2d

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
            
            # ConvTranspose2d
            elif isinstance(torch_layer, nn.ConvTranspose2d) and isinstance(needle_layer, ConvTranspose2d):
                # PyTorch: (in_channels, out_channels, kernel_height, kernel_width)
                # Needle: (kernel_height, kernel_width, out_channels, in_channels)
                torch_weight = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
                # 转置: (in_ch, out_ch, kh, kw) -> (kh, kw, cin,cout)
                needle_weight = np.transpose(torch_weight, (2, 3, 0, 1))
                needle_layer.weight = ndl.Tensor(needle_weight, device=device, dtype=dtype)
                if torch_layer.bias is not None:
                    needle_layer.bias = ndl.Tensor(
                        torch_layer.bias.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                if verbose:
                    print(f"[✔] Copied ConvTranspose2d({torch_layer.in_channels}, {torch_layer.out_channels}, kernel_size={torch_layer.kernel_size})")
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
            
            # === TransformerEncoderLayer/DecoderLayer → TransformerLayer === may not work too well, since our needle implementation is different from torch implementation
            # Our code can supoport torch- Transformer defined in the local
            elif isinstance(torch_layer, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Import TransformerLayer
                from needle.nn.nn_transformer import TransformerLayer
                if isinstance(needle_layer, TransformerLayer):
                    # Copy attention layer weights (q, k, v projections and out projection)
                    # PyTorch stores these in self_attn.in_proj_weight and out_proj
                    torch_attn = torch_layer.self_attn
                    d_model = torch_attn.embed_dim
                    
                    # Split in_proj_weight into q, k, v (if not using separate projections)
                    if hasattr(torch_attn, 'in_proj_weight') and torch_attn.in_proj_weight is not None:
                        in_proj = torch_attn.in_proj_weight.detach().cpu().numpy().astype(np.float32)
                        # Shape: (3*embed_dim, embed_dim) -> split into q, k, v
                        q_weight, k_weight, v_weight = np.split(in_proj, 3, axis=0)
                        
                        # Transpose for Needle format
                        needle_layer.attn.q_projection.weight = ndl.Tensor(q_weight.T, device=device, dtype=dtype)
                        needle_layer.attn.k_projection.weight = ndl.Tensor(k_weight.T, device=device, dtype=dtype)
                        needle_layer.attn.v_projection.weight = ndl.Tensor(v_weight.T, device=device, dtype=dtype)
                        
                        # Copy biases if present (PyTorch has in_proj_bias)
                        if hasattr(torch_attn, 'in_proj_bias') and torch_attn.in_proj_bias is not None:
                            in_proj_bias = torch_attn.in_proj_bias.detach().cpu().numpy().astype(np.float32)
                            q_bias, k_bias, v_bias = np.split(in_proj_bias, 3, axis=0)
                            # Note: Needle's projections are created with bias=False by default
                            # We need to handle this - for now we'll just skip or add manually
                            # Since Needle projections have bias=False, we skip this
                    
                    # Copy out projection
                    out_proj_weight = torch_attn.out_proj.weight.detach().cpu().numpy().astype(np.float32)
                    needle_layer.attn.out_projection.weight = ndl.Tensor(out_proj_weight.T, device=device, dtype=dtype)
                    # Note: Needle's out_projection is created with bias=False, but PyTorch has bias
                    # This is a known limitation - we skip the bias for now
                    
                    # Copy feed-forward weights
                    linear1_weight = torch_layer.linear1.weight.detach().cpu().numpy().astype(np.float32)
                    needle_layer.linear_in.weight = ndl.Tensor(linear1_weight.T, device=device, dtype=dtype)
                    if torch_layer.linear1.bias is not None:
                        needle_layer.linear_in.bias = ndl.Tensor(
                            torch_layer.linear1.bias.detach().cpu().numpy().astype(np.float32),
                            device=device, dtype=dtype
                        )
                    
                    linear2_weight = torch_layer.linear2.weight.detach().cpu().numpy().astype(np.float32)
                    needle_layer.linear_out.weight = ndl.Tensor(linear2_weight.T, device=device, dtype=dtype)
                    if torch_layer.linear2.bias is not None:
                        needle_layer.linear_out.bias = ndl.Tensor(
                            torch_layer.linear2.bias.detach().cpu().numpy().astype(np.float32),
                            device=device, dtype=dtype
                        )
                    
                    # Copy layer norm weights
                    # prenorm_q, prenorm_k, prenorm_v all correspond to norm1 (for self-attention)
                    norm1_weight = ndl.Tensor(
                        torch_layer.norm1.weight.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                    norm1_bias = ndl.Tensor(
                        torch_layer.norm1.bias.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                    needle_layer.attn.prenorm_q.weight = norm1_weight
                    needle_layer.attn.prenorm_q.bias = norm1_bias
                    needle_layer.attn.prenorm_k.weight = norm1_weight
                    needle_layer.attn.prenorm_k.bias = norm1_bias
                    needle_layer.attn.prenorm_v.weight = norm1_weight
                    needle_layer.attn.prenorm_v.bias = norm1_bias
                    
                    # prenorm corresponds to norm2 (for FFN)
                    needle_layer.prenorm.weight = ndl.Tensor(
                        torch_layer.norm2.weight.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                    needle_layer.prenorm.bias = ndl.Tensor(
                        torch_layer.norm2.bias.detach().cpu().numpy().astype(np.float32),
                        device=device, dtype=dtype
                    )
                    
                    if verbose: print(f"[✔] Copied TransformerEncoderLayer")
                    copied += 1
                else:
                    if verbose:
                        print(f"[⚠] Unsupported mapping: {type(torch_layer).__name__} → {type(needle_layer).__name__}")
                    skipped += 1

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


