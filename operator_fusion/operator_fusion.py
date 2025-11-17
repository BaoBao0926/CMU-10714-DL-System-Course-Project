"""
算子融合模块 (Operator Fusion Module)
用于优化 Needle 模型，将连续的算子融合成单个算子以提高执行效率

常见融合模式:
1. Linear + ReLU -> LinearReLU
2. Linear + BatchNorm1d -> LinearBatchNorm
3. BatchNorm1d + ReLU -> BatchNormReLU
4. 更多融合模式...
"""

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

from operator_fusion.fusion_pattrern import (
    FusionPattern, LinearReLUPattern, LinearBatchNormPattern, BatchNormReLUPattern, LinearBatchNormReLUPattern
)


# ============================================================================
# Operator Fusion Engine
# ============================================================================

class OperatorFusion:
    """
    Operator Fusion EngineL: This is sued to identify and fuse operator patterns in models.
    To be simple, it iterates through Sequential modules, and check whether there are layers that can be fused together
    """
    
    def __init__(self, patterns: Optional[List[FusionPattern]] = None):
        """
        Initialize the fusion engine
        Args:
            patterns: List of fusion patterns. If None, default patterns are used
        """
        if patterns is None:
            self.patterns = patterns = [
                LinearBatchNormReLUPattern(), 
                LinearBatchNormPattern(),
                LinearReLUPattern(),
                BatchNormReLUPattern(),
            ]
        else:
            self.patterns = patterns
        
        self.fusion_count = 0  # the number of fusions performed
        self.fusion_log = []   # the fusion log
    
    def fuse_sequential(self, sequential: Sequential) -> Sequential:
        """
        对 Sequential 模块执行算子融合
        
        Args:
            sequential: 待融合的 Sequential 模块
            
        Returns:
            Sequential: 融合后的 Sequential 模块
        """
        if not isinstance(sequential, Sequential):
            raise TypeError(f"Expected Sequential module, got {type(sequential)}")
        
        modules = list(sequential.modules)
        fused_modules = []
        i = 0
        
        while i < len(modules):
            # 递归处理嵌套的 Sequential
            if isinstance(modules[i], Sequential):
                fused_modules.append(self.fuse_sequential(modules[i]))
                i += 1
                continue
            
            # 尝试匹配融合模式
            matched = False
            for pattern in self.patterns:
                if pattern.match(modules, i):
                    # 执行融合
                    fused_module, consumed = pattern.fuse(modules, i)
                    fused_modules.append(fused_module)
                    
                    # 记录融合信息
                    self.fusion_count += 1
                    pattern_name = pattern.__class__.__name__.replace("Pattern", "")
                    original_modules = [type(m).__name__ for m in modules[i:i+consumed]]
                    self.fusion_log.append({
                        "pattern": pattern_name,
                        "original": " -> ".join(original_modules),
                        "fused": type(fused_module).__name__,
                        "position": i
                    })
                    
                    i += consumed
                    matched = True
                    break
            
            # 如果没有匹配任何模式，保持原模块
            if not matched:
                fused_modules.append(modules[i])
                i += 1
        
        return Sequential(*fused_modules)
    
    def fuse_model(self, model: Module) -> Module:
        """
        对整个模型执行算子融合
        
        Args:
            model: 待融合的模型
            
        Returns:
            Module: 融合后的模型
        """
        self.fusion_count = 0
        self.fusion_log = []
        
        # 如果模型本身就是 Sequential，直接融合
        if isinstance(model, Sequential):
            return self.fuse_sequential(model)
        
        # 否则递归处理模型中的所有 Sequential 子模块
        for attr_name in dir(model):
            if attr_name.startswith('_'):
                continue
            attr = getattr(model, attr_name, None)
            if isinstance(attr, Sequential):
                fused = self.fuse_sequential(attr)
                setattr(model, attr_name, fused)
            elif isinstance(attr, Module) and not isinstance(attr, (Linear, ReLU, BatchNorm1d, LayerNorm1d, Dropout, Identity, Flatten)):
                # 递归处理复杂子模块
                self.fuse_model(attr)
        
        return model
    
    def print_fusion_report(self):
        """打印融合报告"""
        print(f"\n{'='*60}")
        print(f"算子融合报告 (Operator Fusion Report)")
        print(f"{'='*60}")
        print(f"总融合次数: {self.fusion_count}")
        print(f"{'-'*60}")
        
        if self.fusion_log:
            print(f"{'位置':<8} {'融合模式':<25} {'原始算子':<20}")
            print(f"{'-'*60}")
            for log in self.fusion_log:
                print(f"{log['position']:<8} {log['pattern']:<25} {log['original']:<20}")
        else:
            print("未发现可融合的算子模式")
        
        print(f"{'='*60}\n")
    
    def get_fusion_stats(self) -> dict:
        """
        获取融合统计信息
        
        Returns:
            dict: 包含融合统计信息的字典
        """
        pattern_counts = {}
        for log in self.fusion_log:
            pattern = log['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            "total_fusions": self.fusion_count,
            "pattern_counts": pattern_counts,
            "fusion_log": self.fusion_log
        }


# ============================================================================
# Main function to start operator fusion
# ============================================================================

def fuse_operators(model: Module, verbose: bool = True) -> Module:
    """
    对模型执行算子融合的便捷函数
    
    Args:
        model: 待融合的 Needle 模型
        verbose: 是否打印融合报告
        
    Returns:
        Module: 融合后的模型
    
    Example:
        >>> from torch2needle_converter import torch2needle_fx
        >>> from operator_fusion import fuse_operators
        >>> 
        >>> # 转换 PyTorch 模型为 Needle 模型
        >>> needle_model, _, _ = torch2needle_fx(torch_model)
        >>> 
        >>> # 执行算子融合
        >>> fused_model = fuse_operators(needle_model, verbose=True)
    """
    # Pattern can be increamental added
    patterns = [
        LinearBatchNormReLUPattern(), 
        LinearBatchNormPattern(),
        LinearReLUPattern(),
        BatchNormReLUPattern(),
    ]
    fusion_engine = OperatorFusion(patterns=patterns)
    fused_model = fusion_engine.fuse_model(model)
    
    if verbose:
        fusion_engine.print_fusion_report()
    
    return fused_model


def get_fusion_stats(model: Module) -> dict:
    """
    获取模型融合统计信息（不修改模型）
    
    Args:
        model: Needle 模型
        
    Returns:
        dict: 融合统计信息
    """
    fusion_engine = OperatorFusion()
    # 在临时副本上执行融合以获取统计信息
    import copy
    temp_model = copy.deepcopy(model)
    fusion_engine.fuse_model(temp_model)
    
    return fusion_engine.get_fusion_stats()


if __name__ == "__main__":
    print("算子融合模块测试")
    print("="*60)
    
    # 创建一个简单的测试模型
    test_model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 30),
        BatchNorm1d(30),
        ReLU(),
        Linear(30, 10)
    )
    
    print("原始模型:")
    print(test_model)
    print()
    
    # 执行融合
    fused_model = fuse_operators(test_model, verbose=True)
    
    print("融合后模型:")
    print(fused_model)
