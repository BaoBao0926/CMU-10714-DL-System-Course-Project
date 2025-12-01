import sys
import torch
import numpy as np
import needle as ndl
from needle import Tensor
from needle.nn import Sequential

# 导入转换和融合工具
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping
from operator_fusion.operator_fusion import OperatorFusion
from torchvision import models

def _run_pipeline_test(torch_model, input_shape,device=ndl.cpu(),dtype="fl"):
    """运行完整的 PyTorch → Needle → 权重加载 → 算子融合 流程测试"""
    
    # Step 1: 创建 PyTorch 模型
    print("\n【Step 1】Prepare Pytorch Model")
    torch_model.eval()
    
    # 准备测试数据
    test_input = torch.randn(*input_shape)
    with torch.no_grad():
        torch_output = torch_model(test_input)
    
    print(f"PyTorch Model Architecture:")
    print(torch_model)
    print(f"PyTorch Input shape: {test_input.shape}")
    print(f"PyTorch Output shape: {torch_output.shape}")
    
    # Step 2: 转换为 Needle 模型
    print("\n【Step 2】Transform to Needle Model")
    needle_model, trace_log, torch_mapping_needle = torch2needle_fx(torch_model,device,dtype)
    
    print(f"Needle Model Type: {type(needle_model).__name__}")
    print(f"Needle Model Architecture:")
    print(needle_model)
    
    # Step 3: 加载权重
    print("\n【Step 3】Load Weights into Needle Model")
    load_torch_weights_by_mapping(torch_mapping_needle, verbose=True,device=device,dtype=dtype)
    
    # 设置为 eval 模式
    needle_model.eval()
    
    # Step 4: 验证转换后的模型输出
    print("\n【Step 4】Validate Converted Needle Model")
    needle_input = Tensor(test_input.detach().numpy(),device=device,dtype=dtype)
    needle_output_before = needle_model(needle_input)
    
    diff_before = np.abs(torch_output.detach().numpy() - needle_output_before.numpy())
    max_diff_before = np.max(diff_before)
    print(f"Maximum difference between Needle Model and Torch Model: {max_diff_before:.2e}")
    
    if max_diff_before < 1e-5:
        print("✅ Conversion success!")
    else:
        print("❌ Conversion has error!")
        return False
    
    # Step 5: 检查模型是否可融合
    print("\n【Step 5】Check Model for Fusion")
   # print(f"模型类型: {type(needle_model).__name__}")
    
    if isinstance(needle_model, Sequential):
        print("✅ 模型是 Sequential，直接支持融合")
    else:
        print(f"✅ Model is {type(needle_model).__name__}, assuming it supports fusion")
    
    # Step 6: 执行算子融合
    print("\n【Step 6】Try Operator Fusion")
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_model(needle_model)
    
    # 设置融合后模型为 eval 模式
    fused_model.eval()
    
    print(f"\nFusion report:")
    fusion_engine.print_fusion_report()
    
    print(f"\nFused model:")
    print(fused_model)
    
    # Step 7: 验证融合后模型的正确性
    print("\n【Step 7】Validate Fused Needle Model with Torch Model")
    needle_output_after = fused_model(needle_input)
    
    diff_after = np.abs(torch_output.detach().numpy() - needle_output_after.numpy())
    max_diff_after = np.max(diff_after)
    print(f"Maximum difference between fused model and torch model: {max_diff_after:.2e}")
    
    if max_diff_after < 1e-5:
        print("✅ fusion correct!")
    else:
        print("❌ fusion has error!")
        return False
    
    # Step 8: 对比融合前后输出
    print("\n【Step 8】Compare Outputs Before and After Fusion")
    diff_fusion = np.abs(needle_output_before.numpy() - needle_output_after.numpy())
    max_diff_fusion = np.max(diff_fusion)
    print(f"Maximum difference between fused model and non-fused model: {max_diff_fusion:.2e}")
    
    if max_diff_fusion < 1e-6:
        print("✅ fusion has no significant effect on output!")
    else:
        print("⚠️  fusion changed the output a bit!")
    
    print("\n" + "=" * 80)
    print("✅ test passed!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    all_passed = True
    #device = ndl.cpu() # this is correct, it is ndl.cpu() not ndl.numpy_cpu()\
    device = ndl.cuda() 


    dtype = "float32"
    
    # # 测试 3: ResNet18 模型
    print("\n\n" + "=" * 80)
    model = models.resnet101(models.ResNet101_Weights.DEFAULT)
    print("Test ResNet101 Model")
    print("=" * 80)
    all_passed &= _run_pipeline_test(model,(1,3,224,224),device=device,dtype=dtype)
    # 总结
    print("\n\n" + "=" * 80)
    if all_passed:
        print("🎉 All test passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 80)
    
    sys.exit(0 if all_passed else 1)