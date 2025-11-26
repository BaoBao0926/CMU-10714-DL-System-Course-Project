"""
Test script for Conv+BatchNorm2d+ReLU fusion
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from needle.autograd import Tensor
from needle.nn.nn_basic import Sequential, ReLU, BatchNorm2d
from needle.nn.nn_conv import Conv
from operator_fusion.operator_fusion import OperatorFusion
from operator_fusion.fused_layer import ConvBatchNorm2dReLU


def test_conv_bn_relu_fusion_pattern():
    """Test that Conv+BN+ReLU pattern is correctly recognized and fused"""
    print("\n=== Test 1: Fusion Pattern Recognition ===")
    
    # Create a simple model with Conv+BN+ReLU pattern
    model = Sequential(
        Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1),
        BatchNorm2d(16),
        ReLU()
    )
    
    print(f"Original model has {len(model.modules)} modules:")
    for i, m in enumerate(model.modules):
        print(f"  {i}: {type(m).__name__}")
    
    # Apply fusion
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_sequential(model)
    
    print(f"\nFused model has {len(fused_model.modules)} modules:")
    for i, m in enumerate(fused_model.modules):
        print(f"  {i}: {type(m).__name__}")
    
    # Verify fusion occurred
    assert len(fused_model.modules) == 1, f"Expected 1 module after fusion, got {len(fused_model.modules)}"
    assert isinstance(fused_model.modules[0], ConvBatchNorm2dReLU), \
        f"Expected ConvBatchNorm2dReLU, got {type(fused_model.modules[0]).__name__}"
    
    print("✓ Fusion pattern recognition successful!")


def test_conv_bn_relu_forward():
    """Test that fused module produces same output as unfused sequence"""
    print("\n=== Test 2: Forward Pass Correctness ===")
    
    # Create input tensor (batch=2, channels=3, height=8, width=8)
    np.random.seed(42)
    x_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    x = Tensor(x_data)
    
    # Create unfused model
    conv = Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1)
    bn = BatchNorm2d(16)
    relu = ReLU()
    unfused_model = Sequential(conv, bn, relu)
    
    # Set to eval mode for deterministic behavior
    unfused_model.eval()
    
    # Forward pass through unfused model
    out_unfused = unfused_model(x)
    
    # Create fused model with same parameters
    fused_model = ConvBatchNorm2dReLU(
        in_channels=3, out_channels=16, kernel_size=3, stride=1,
        eps=bn.eps, momentum=bn.momentum
    )
    fused_model.eval()
    
    # Copy parameters from unfused to fused
    fused_model.weight = conv.weight
    if conv.bias is not None:
        fused_model.conv_bias = conv.bias
    fused_model.bn_weight = bn.weight
    fused_model.bn_bias = bn.bias
    fused_model.running_mean = bn.running_mean
    fused_model.running_var = bn.running_var
    
    # Forward pass through fused model
    out_fused = fused_model(x)
    
    # Compare outputs
    print(f"Unfused output shape: {out_unfused.shape}")
    print(f"Fused output shape: {out_fused.shape}")
    
    diff = np.abs(out_unfused.numpy() - out_fused.numpy())
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Allow small numerical differences due to floating point arithmetic
    assert max_diff < 1e-4, f"Outputs differ too much: max_diff={max_diff}"
    
    print("✓ Forward pass correctness verified!")


def test_fusion_with_multiple_blocks():
    """Test fusion of multiple Conv+BN+ReLU blocks"""
    print("\n=== Test 3: Multiple Block Fusion ===")
    
    # Create model with multiple Conv+BN+ReLU blocks
    model = Sequential(
        Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1),
        BatchNorm2d(16),
        ReLU(),
        Conv(in_channels=16, out_channels=32, kernel_size=3, stride=1),
        BatchNorm2d(32),
        ReLU(),
        Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        BatchNorm2d(64),
        ReLU()
    )
    
    print(f"Original model has {len(model.modules)} modules")
    
    # Apply fusion
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_sequential(model)
    
    print(f"Fused model has {len(fused_model.modules)} modules")
    
    # Verify all three blocks were fused
    assert len(fused_model.modules) == 3, f"Expected 3 fused modules, got {len(fused_model.modules)}"
    for i, module in enumerate(fused_model.modules):
        assert isinstance(module, ConvBatchNorm2dReLU), \
            f"Module {i} is {type(module).__name__}, expected ConvBatchNorm2dReLU"
    
    print("✓ Multiple block fusion successful!")


def test_mixed_fusion():
    """Test fusion with both Conv and Linear fusion patterns"""
    print("\n=== Test 4: Mixed Fusion Patterns ===")
    
    from needle.nn.nn_basic import Linear, Flatten
    
    # Create model with both Conv+BN+ReLU and Linear+ReLU patterns
    model = Sequential(
        Conv(in_channels=3, out_channels=16, kernel_size=3, stride=1),
        BatchNorm2d(16),
        ReLU(),
        Flatten(),
        Linear(in_features=16*8*8, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=10),
        ReLU()
    )
    
    print(f"Original model has {len(model.modules)} modules:")
    for i, m in enumerate(model.modules):
        print(f"  {i}: {type(m).__name__}")
    
    # Apply fusion
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_sequential(model)
    
    print(f"\nFused model has {len(fused_model.modules)} modules:")
    for i, m in enumerate(fused_model.modules):
        print(f"  {i}: {type(m).__name__}")
    
    # Verify fusion occurred (Conv+BN+ReLU -> 1, Flatten -> 1, Linear+ReLU -> 1, Linear+ReLU -> 1)
    assert len(fused_model.modules) == 4, f"Expected 4 modules after fusion, got {len(fused_model.modules)}"
    
    print("✓ Mixed fusion patterns successful!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Conv+BatchNorm2d+ReLU Fusion")
    print("=" * 60)
    
    try:
        test_conv_bn_relu_fusion_pattern()
        test_conv_bn_relu_forward()
        test_fusion_with_multiple_blocks()
        test_mixed_fusion()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
