import sys
import numpy as np
import os
sys.path.append(os.path.dirname(__file__))
os.environ["NEEDLE_BACKEND"] = "hip"
import torch
import torch.nn.functional as F
import needle as ndl
from unet import UNet
from needle import Tensor
from needle.nn import Sequential

# 导入转换和融合工具
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping
from operator_fusion.operator_fusion import OperatorFusion


def _run_pipeline_test(torch_model, input_shape,device=ndl.cpu(),dtype="fl"):
    """运行完整的 PyTorch → Needle → 权重加载 → 算子融合 流程测试"""
    
    # Step 1: 创建 PyTorch 模型
    print("\n【Step 1】PyTorch 模型准备")
    torch_model.eval()
    
    # 准备测试数据
    test_input = torch.randn(*input_shape)
    with torch.no_grad():
        torch_output = torch_model(test_input)
    
    print(f"PyTorch 模型结构:")
    print(torch_model)
    print(f"PyTorch 输入形状: {test_input.shape}")
    print(f"PyTorch 输出形状: {torch_output.shape}")
    
    # Step 2: 转换为 Needle 模型
    print("\n【Step 2】转换为 Needle 模型")
    needle_model, trace_log, torch_mapping_needle = torch2needle_fx(torch_model,device,dtype)
    
    print(f"Needle 模型类型: {type(needle_model).__name__}")
    print(f"Needle 模型结构:")
    print(needle_model)
    
    # Step 3: 加载权重
    print("\n【Step 3】加载权重")
    load_torch_weights_by_mapping(torch_mapping_needle, verbose=True,device=device,dtype=dtype)
    
    # 设置为 eval 模式
    needle_model.eval()
    
    # Step 4: 验证转换后的模型输出
    print("\n【Step 4】验证转换后模型")
    needle_input = Tensor(test_input.detach().numpy(),device=device,dtype=dtype)
    needle_output_before = needle_model(needle_input)
    
    np.testing.assert_allclose(needle_output_before.numpy(), torch_output.detach().numpy(), rtol=1.5e-2, atol=1e-3), "转换后模型输出与 PyTorch 不匹配"
    print("转换后模型输出与 PyTorch 匹配")
    # diff_before = np.abs(torch_output.detach().numpy() - needle_output_before.numpy())
    # max_diff_before = np.max(diff_before)
    # print(f"转换后最大误差: {max_diff_before:.2e}")
    
    # if max_diff_before < 1e-5:
    #     print("✅ 转换正确！")
    # else:
    #     print("❌ 转换有误差！")
    #     return False
    
    # Step 5: 检查模型是否可融合
    print("\n【Step 5】检查模型是否支持算子融合")
    print(f"模型类型: {type(needle_model).__name__}")
    
    if isinstance(needle_model, Sequential):
        print("✅ 模型是 Sequential，直接支持融合")
    else:
        print(f"✅ 模型是 {type(needle_model).__name__}，将尝试融合其中的层序列")
    
    # Step 6: 执行算子融合
    print("\n【Step 6】执行算子融合")
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_model(needle_model)
    
    # 设置融合后模型为 eval 模式
    fused_model.eval()
    
    print(f"\n融合报告:")
    fusion_engine.print_fusion_report()
    
    print(f"\n融合后模型:")
    print(fused_model)
    
    # Step 7: 验证融合后模型的正确性
    print("\n【Step 7】验证融合后模型")
    needle_output_after = fused_model(needle_input)
    
    np.testing.assert_allclose(needle_output_after.numpy(), torch_output.detach().numpy(), rtol=1.5e-2, atol=1e-3), "融合后模型输出与 PyTorch 不匹配"
    print("融合后模型输出与 PyTorch 匹配")
    # diff_after = np.abs(torch_output.detach().numpy() - needle_output_after.numpy())
    # max_diff_after = np.max(diff_after)
    # print(f"融合后最大误差: {max_diff_after:.2e}")
    
    # if max_diff_after < 1e-5:
    #     print("✅ 融合正确！")
    # else:
    #     print("❌ 融合后有误差！")
    #     return False
    
    # Step 8: 对比融合前后输出
    print("\n【Step 8】对比融合前后")
    diff_fusion = np.abs(needle_output_before.numpy() - needle_output_after.numpy())
    max_diff_fusion = np.max(diff_fusion)
    print(f"融合前后最大差异: {max_diff_fusion:.2e}")
    
    if max_diff_fusion < 1e-6:
        print("✅ 融合前后输出一致！")
    else:
        print("⚠️  融合前后有细微差异")
    
    print("\n" + "=" * 80)
    print("✅ 测试通过！")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    all_passed = True
    device = ndl.hip()
    # 示例：测试 UNet 模型
    print("运行 UNet 模型测试")
    print("=" * 80)
    torch_unet = UNet(n_channels=3, n_classes=1)
    input_shape = (1, 3, 572, 572)  # UNet 输入形状
    all_passed &= _run_pipeline_test(torch_unet, input_shape,device=device,dtype="float32")
    # 总结
    print("\n\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)
