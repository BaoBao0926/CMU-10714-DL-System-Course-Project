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
)
import needle.init as init
from needle.nn.nn_basic import (
    Module, Linear, ReLU, Sequential, BatchNorm1d, 
    LayerNorm1d, Dropout, Identity, Flatten, Parameter
)
from typing import Any, List, Tuple, Optional

try:
    from fused_layer import (
        LinearReLU, LinearBatchNorm, BatchNormReLU, LinearBatchNormReLU,
    )
except ImportError:
    from operator_fusion.fused_layer import (
        LinearReLU, LinearBatchNorm, BatchNormReLU, LinearBatchNormReLU,
    )


# ============================================================================
# Fusion Pattern Recognition and Application
# ============================================================================


class FusionPattern:
    """
    融合模式的基类
    每个具体的融合模式都需要实现 match() 和 fuse() 方法
    """
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
        
        return fused, 3
