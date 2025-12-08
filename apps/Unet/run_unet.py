import sys
import torch
import numpy as np
import argparse
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch→Needle pipeline for ResNet with optional fusion.")
    parser.add_argument("--backend", type=str, default="nd",choices=["nd","hip"],)
    parser.add_argument("--n_channels",type=int,default=3)
    parser.add_argument("--n_classes",type=int,default=1)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "hip"],
                        help="Backend device for Needle.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--height", type=int, default=224, help="Input height.")
    parser.add_argument("--width", type=int, default=224, help="Input width.")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Needle tensor dtype.")
    parser.add_argument("--weights", type=str, default="DEFAULT",
                        help="Torchvision weights enum name or 'NONE'.")
    return parser.parse_args()

def resolve_backend(backend_str):
    import os
    if backend_str == "nd":
        os.environ["NEEDLE_BACKEND"] = "nd"
    elif backend_str == "hip":
        os.environ["NEEDLE_BACKEND"] = "hip"
    else:
        raise ValueError(f"Unsupported backend: {backend_str}")

def import_needle_modules():
    import needle as ndl
    from needle import Tensor
    from needle.nn import Sequential
    # torch2needle transformation tool and fusion tool
    from torch2needle.torch2needle_converter import torch2needle_fx
    from torch2needle.weight_converter import load_torch_weights_by_mapping
    from operator_fusion.operator_fusion import OperatorFusion
    return ndl, Tensor,Sequential,torch2needle_fx,load_torch_weights_by_mapping,OperatorFusion
    
def _run_pipeline_test(torch_model, input_shape,device,dtype,
                       Tensor, ndl, Sequential, torch2needle_fx, load_torch_weights_by_mapping, OperatorFusion):
    """run complete PyTorch → Needle → weight mapping → operator fusion pipeline test"""
    
    # Step 1: prepare pytorch model
    print("\n【Step 1】PyTorch model Prepare")
    torch_model.eval()
    
    # 准备测试数据
    test_input = torch.randn(*input_shape)
    with torch.no_grad():
        torch_output = torch_model(test_input)
    
    # print(f"PyTorch model architecture:")
    # print(torch_model)
    print(f"PyTorch model input shape: {test_input.shape}")
    print(f"PyTorch model output shape: {torch_output.shape}")
    
    # Step 2: Transfer to needle model
    print("\n【Step 2】Transfer to needle model")
    needle_model, trace_log, torch_mapping_needle = torch2needle_fx(torch_model,device,dtype)
    
    print(f"Needle model type: {type(needle_model).__name__}")
    print(f"Needle model structure:")
    print(needle_model)
    
    # Step 3: load weight to needle model
    print("\n【Step 3】Load weight to needle model")
    load_torch_weights_by_mapping(torch_mapping_needle, verbose=True,device=device,dtype=dtype)
    
    # set to eval mode
    needle_model.eval()
    
    # Step 4: Verify converted model
    print("\n【Step 4】Verify converted model")
    needle_input = Tensor(test_input.detach().numpy(),device=device,dtype=dtype)
    needle_output_before = needle_model(needle_input)
    
    diff_before = np.abs(torch_output.detach().numpy() - needle_output_before.numpy())
    max_diff_before = np.max(diff_before)
    print(f"Max difference after conversion: {max_diff_before:.2e}")
    
    if max_diff_before < 1e-5:
        print("✅ Conversion is correct!")
    else:
        print("❌ Conversion has error!")
        return False
    
    # Step 5: Perform operator fusion
    print("\n【Step 5】Perform operator fusion")
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_model(needle_model)
    
    # Set fused model to eval
    fused_model.eval()
    
    print(f"\nFuse report:")
    fusion_engine.print_fusion_report()
    
    print(f"\nFused model:")
    print(fused_model)
    
    # Step 6: Verify correctness of fused model
    print("\n【Step 6】Verify conversion of fused model with torch model")
    needle_output_after = fused_model(needle_input)
    
    diff_after = np.abs(torch_output.detach().numpy() - needle_output_after.numpy())
    max_diff_after = np.max(diff_after)
    print(f"Max difference between fused and torch model: {max_diff_after:.2e}")
    
    if max_diff_after < 1e-5:
        print("✅ Fusion correct!")
    else:
        print("❌ Fusion has error!")
        return False
    
    # Step 7: Compared fused model and non-fused model
    print("\n【Step 7】Compared fused model and non-fused model")
    diff_fusion = np.abs(needle_output_before.numpy() - needle_output_after.numpy())
    max_diff_fusion = np.max(diff_fusion)
    print(f"Max difference before and after fused: {max_diff_fusion:.2e}")
    
    if max_diff_fusion < 1e-6:
        print("✅ Fusion produces no difference")
    else:
        print("⚠️ Fusion produces little difference")
    
    print("\n" + "=" * 80)
    print("✅ Test passed!")
    print("=" * 80)
    
    return True

def resolve_device(ndl,device_str):
    if device_str == "cpu":
        return ndl.cpu()
    elif device_str == "cuda":
        return ndl.cuda()
    elif device_str == "hip":
        return ndl.hip()
    else:
        raise ValueError(f"Unsupported device: {device_str}")


if __name__ == "__main__":
    all_passed = True
    args = parse_args()
    # set backend
    resolve_backend(args.backend)
    # import needle modules based on backend setting
    ndl, Tensor, Sequential, torch2needle_fx, load_torch_weights_by_mapping, OperatorFusion = import_needle_modules()
    device = resolve_device(ndl,args.device)
    dtype = args.dtype
    model = UNet(n_channels=args.n_channels,n_classes=args.n_classes)
    input_shape = (args.batch, 3, args.height, args.width)
    print("\n\n" + "=" * 80)
    print(f"test: Unet model, device={args.device}, input={input_shape}, dtype={dtype}")
    print("=" * 80)

    all_passed &= _run_pipeline_test(model,input_shape,device,dtype,
                                     Tensor,ndl,Sequential,torch2needle_fx, load_torch_weights_by_mapping, OperatorFusion)
    # 总结
    print("\n\n" + "=" * 80)
    if all_passed:
        print("🎉 all test passed!")
    else:
        print("❌ part of tests failed!")
    print("=" * 80)
    
    sys.exit(0 if all_passed else 1)