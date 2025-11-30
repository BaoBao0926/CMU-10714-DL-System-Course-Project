import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from needle.autograd import Tensor
from needle.ops.ops_mathematic import relu, matmul, broadcast_to, summation, reshape
from needle.ops.ops_fused import (
    fused_linear_relu,
    fused_batchnorm_relu,
    fused_linear_batchnorm,
    fused_linear_batchnorm_relu,
    fused_multihead_attention,
)
import needle.init as init
from needle.nn.nn_basic import (
    Module, Linear, ReLU, Sequential, BatchNorm1d, BatchNorm2d,
    LayerNorm1d, Dropout, Identity, Flatten, Parameter
)
from needle.nn.nn_conv import Conv
from typing import Any, List, Tuple, Optional

try:
    from fused_layer import (
        LinearReLU, LinearBatchNorm, BatchNormReLU, LinearBatchNormReLU,
        ConvBatchNorm2dReLU, FusedMultiHeadAttention,ConvBatchNorm2d
    )
except ImportError:
    from operator_fusion.fused_layer import (
        LinearReLU, LinearBatchNorm, BatchNormReLU, LinearBatchNormReLU,
        ConvBatchNorm2dReLU, FusedMultiHeadAttention,ConvBatchNorm2d
    )


# ============================================================================
# Fusion Pattern Recognition and Application
# ============================================================================


class FusionPattern:
    """
    融合模式的基类
    每个具体的融合模式都需要实现 match() 和 fuse() 方法
    """
    # how many layers do fusion layer has been consumed
    consumed_count:int
    def match(self, modules: List[Module], start_idx: int) -> bool:
        """
        检查从 start_idx 开始的模块序列是否匹配该融合模式
        
        Args:
            modules: 模块列表
            start_idx: 开始检查的索引位置
            
        Returns:
            bool: 是否匹配
        """
        raise NotImplementedError
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        """
        将匹配的模块序列融合成一个模块
        
        Args:
            modules: 模块列表
            start_idx: 开始融合的索引位置
            
        Returns:
            Tuple[Module, int]: (融合后的模块, 融合的模块数量)
        """
        raise NotImplementedError


class LinearReLUPattern(FusionPattern):
    """Linear + ReLU fusion pattern"""
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 1 >= len(modules):
            return False
        return isinstance(modules[start_idx], Linear) and isinstance(modules[start_idx + 1], ReLU)
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        linear = modules[start_idx]
        # 创建融合模块并复制权重
        fused = LinearReLU(linear.in_features, linear.out_features, bias=linear.bias is not None)
        fused.weight = linear.weight
        if linear.bias is not None:
            fused.bias = linear.bias
        self.consumed_count = 2
        return fused, 2  # 融合了 2 个模块


class LinearBatchNormPattern(FusionPattern):
    """Linear + BatchNorm1d fusion pattern"""
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 1 >= len(modules):
            return False
        if not isinstance(modules[start_idx], Linear):
            return False
        if not isinstance(modules[start_idx + 1], BatchNorm1d):
            return False
        # 检查维度是否匹配
        linear = modules[start_idx]
        bn = modules[start_idx + 1]
        return linear.out_features == bn.dim
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        linear = modules[start_idx]
        bn = modules[start_idx + 1]
        
        # 创建融合模块并复制参数
        fused = LinearBatchNorm(linear.in_features, linear.out_features, 
                               eps=bn.eps, momentum=bn.momentum, 
                               bias=linear.bias is not None)
        fused.weight = linear.weight
        if linear.bias is not None:
            fused.linear_bias = linear.bias
        fused.bn_weight = bn.weight
        fused.bn_bias = bn.bias
        fused.running_mean = bn.running_mean
        fused.running_var = bn.running_var
        self.consumed_count = 2
        return fused, 2


class BatchNormReLUPattern(FusionPattern):
    """BatchNorm1d + ReLU 融合模式"""
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 1 >= len(modules):
            return False
        return isinstance(modules[start_idx], BatchNorm1d) and isinstance(modules[start_idx + 1], ReLU)
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        bn = modules[start_idx]
        
        # 创建融合模块并复制参数
        fused = BatchNormReLU(bn.dim, eps=bn.eps, momentum=bn.momentum)
        fused.weight = bn.weight
        fused.bias = bn.bias
        fused.running_mean = bn.running_mean
        fused.running_var = bn.running_var
        self.consumed_count = 2
        return fused, 2


class LinearBatchNormReLUPattern(FusionPattern):
    """Linear + BatchNorm1d + ReLU 融合模式"""
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 2 >= len(modules):
            return False
        if not isinstance(modules[start_idx], Linear):
            return False
        if not isinstance(modules[start_idx + 1], BatchNorm1d):
            return False
        if not isinstance(modules[start_idx + 2], ReLU):
            return False
        # 检查维度是否匹配
        linear = modules[start_idx]
        bn = modules[start_idx + 1]
        return linear.out_features == bn.dim
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        linear = modules[start_idx]
        bn = modules[start_idx + 1]
        
        # 创建融合模块并复制参数
        fused = LinearBatchNormReLU(linear.in_features, linear.out_features, 
                                    eps=bn.eps, momentum=bn.momentum, 
                                    bias=linear.bias is not None)
        fused.weight = linear.weight
        if linear.bias is not None:
            fused.linear_bias = linear.bias
        fused.bn_weight = bn.weight
        fused.bn_bias = bn.bias
        fused.running_mean = bn.running_mean
        fused.running_var = bn.running_var
        self.consumed_count = 3
        return fused, 3


class ConvBatchNorm2dReLUPattern(FusionPattern):
    """Conv + BatchNorm2d + ReLU fusion pattern
    
    This is the fundamental building block in modern CNNs:
    - ResNet: Basic and Bottleneck blocks
    - MobileNet: Depthwise separable convolutions
    - EfficientNet: MBConv blocks
    - Most modern vision architectures
    """
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 2 >= len(modules):
            return False
        if not isinstance(modules[start_idx], Conv):
            return False
        if not isinstance(modules[start_idx + 1], BatchNorm2d):
            return False
        if not isinstance(modules[start_idx + 2], ReLU):
            return False
        # Check dimension matching
        conv = modules[start_idx]
        bn = modules[start_idx + 1]
        return conv.out_channels == bn.dim
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        conv = modules[start_idx]
        bn = modules[start_idx + 1]
        
        # Create fused module and copy parameters
        fused = ConvBatchNorm2dReLU(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, bias=conv.bias is not None,
            eps=bn.eps, momentum=bn.momentum
        )
        fused.weight = conv.weight
        if conv.bias is not None:
            fused.conv_bias = conv.bias
        fused.bn_weight = bn.weight
        fused.bn_bias = bn.bias
        fused.running_mean = bn.running_mean
        fused.running_var = bn.running_var
        self.consumed_count = 3
        return fused, 3

class ConvBatchNorm2dPattern(FusionPattern):
    """Conv + BatchNorm2d fusion pattern
    
    This pattern fuses a Conv layer followed by a BatchNorm2d layer.
    It is commonly used in convolutional neural networks to improve
    performance by reducing the number of separate operations.
    """
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        if start_idx + 1 >= len(modules):
            return False
        if not isinstance(modules[start_idx], Conv):
            return False
        if not isinstance(modules[start_idx + 1], BatchNorm2d):
            return False
        # Check dimension matching
        conv = modules[start_idx]
        bn = modules[start_idx + 1]
        return conv.out_channels == bn.dim
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        conv = modules[start_idx]
        bn = modules[start_idx + 1]
        
        # Create fused module and copy parameters
        fused = ConvBatchNorm2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, bias=conv.bias is not None,
            eps=bn.eps, momentum=bn.momentum
        )
        fused.weight = conv.weight
        if conv.bias is not None:
            fused.conv_bias = conv.bias
        fused.bn_weight = bn.weight
        fused.bn_bias = bn.bias
        fused.running_mean = bn.running_mean
        fused.running_var = bn.running_var
        self.consumed_count = 2
        return fused, 2

class MultiHeadAttentionPattern(FusionPattern):
    """
    Multi-Head Attention fusion pattern (FlashAttention-style)
    
    Fuses the MultiHeadAttention module into a single optimized operation.
    This is beneficial for:
    - Transformer models
    
    Unlike other patterns that fuse sequential layers, this pattern
    replaces a single complex module with an optimized version.
    """
    
    def match(self, modules: List[Module], start_idx: int) -> bool:
        """Match standalone MultiHeadAttention modules"""
        if start_idx >= len(modules):
            return False
        
        # Import here to avoid circular dependency
        try:
            from needle.nn.nn_transformer import MultiHeadAttention
        except ImportError:
            return False
        
        return isinstance(modules[start_idx], MultiHeadAttention)
    
    def fuse(self, modules: List[Module], start_idx: int) -> Tuple[Module, int]:
        """
        Replace MultiHeadAttention with FusedMultiHeadAttention.
        
        Follows the same pattern as other fusion operations:
        - ConvBatchNorm2dReLUPattern.fuse() creates ConvBatchNorm2dReLU
        - LinearBatchNormReLUPattern.fuse() creates LinearBatchNormReLU
        - MultiHeadAttentionPattern.fuse() creates FusedMultiHeadAttention
        """
        from needle.nn.nn_transformer import MultiHeadAttention
        
        mha = modules[start_idx]
        
        # Extract parameters from the original module
        # Extract embed_dim and num_heads (with fallback to default if not set)
        embed_dim = mha.embed_dim if hasattr(mha, 'embed_dim') and mha.embed_dim is not None else 512
        num_heads = mha.num_heads if hasattr(mha, 'num_heads') and mha.num_heads is not None else 8
        
        # Extract dropout probability
        dropout_prob = 0.0
        if hasattr(mha, 'dropout') and hasattr(mha.dropout, 'p'):
            dropout_prob = mha.dropout.p
        
        # Create fused module (same pattern as ConvBatchNorm2dReLU, LinearBatchNormReLU)
        fused = FusedMultiHeadAttention(
            embed_dim=embed_dim,  # Now uses mha.embed_dim
            num_heads=num_heads,  # Now uses mha.num_heads
            dropout=dropout_prob,
            causal=mha.causal if hasattr(mha, 'causal') else False,
            device=mha.device if hasattr(mha, 'device') else None,
            dtype=mha.dtype if hasattr(mha, 'dtype') else "float32"
        )
        
        # Copy parameters if they exist
        # In the current MultiHeadAttention implementation, there are no QKV projection weights
        # because projections are done in AttentionLayer, not MultiHeadAttention itself
        # When MultiHeadAttention is refactored to include projections, you would copy them here:
        # if hasattr(mha, 'q_proj_weight'):
        #     fused.q_proj_weight = mha.q_proj_weight
        # if hasattr(mha, 'k_proj_weight'):
        #     fused.k_proj_weight = mha.k_proj_weight
        # if hasattr(mha, 'v_proj_weight'):
        #     fused.v_proj_weight = mha.v_proj_weight
        self.consumed_count = 1
        return fused, 1

