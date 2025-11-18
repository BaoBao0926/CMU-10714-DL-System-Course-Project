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

try:
    from fusion_pattrern import (
        FusionPattern, LinearReLUPattern, LinearBatchNormPattern, BatchNormReLUPattern, LinearBatchNormReLUPattern
    )
except ImportError:
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
        fuse operator on Sequential module
        
        Args:
            sequential: Sequential module to be fused
            
        Returns:
            Sequential: Fused Sequential module
        """
        if not isinstance(sequential, Sequential):
            raise TypeError(f"Expected Sequential module, got {type(sequential)}")
        
        modules = list(sequential.modules)
        fused_modules = []
        i = 0
        
        # the basic algorithm is: iterate through the modules, at each position try to match any fusion pattern
        while i < len(modules):
            # iterate nested Sequential modules recursively
            if isinstance(modules[i], Sequential):
                fused_modules.append(self.fuse_sequential(modules[i]))
                i += 1
                continue
            
            # try to match fusion patterns
            matched = False
            for pattern in self.patterns:
                # iterate our all fusion patterns
                if pattern.match(modules, i):
                    # perform fusion
                    fused_module, consumed = pattern.fuse(modules, i)
                    fused_modules.append(fused_module)
                    
                    # record fusion information
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
            
            # if no pattern matched, keep the original module
            if not matched:
                fused_modules.append(modules[i])
                i += 1
        
        return Sequential(*fused_modules)
    
    def fuse_model(self, model: Module) -> Module:
        """
        fuse operators on the entire model
        
        Args:
            model: Model to be fused
            
        Returns:
            Module: Fused model
        """
        self.fusion_count = 0
        self.fusion_log = []
        
        # If the model itself is a Sequential, fuse directly
        if isinstance(model, Sequential):
            return self.fuse_sequential(model)
        
        # Check if it is an FXGraphExecutor (by checking for layer_N attributes)
        layer_attrs = sorted([name for name in dir(model) if name.startswith('layer_') and name.split('_')[1].isdigit()])
        if layer_attrs:
            # FXGraphExecutor case: extract all layer_N, fuse them, and replace back
            layers = [getattr(model, attr) for attr in layer_attrs]
            temp_sequential = Sequential(*layers)
            fused_sequential = self.fuse_sequential(temp_sequential)
            
            # Replace back to FXGraphExecutor
            for i, fused_layer in enumerate(fused_sequential.modules):
                attr_name = f"layer_{i}"
                if hasattr(model, attr_name):
                    delattr(model, attr_name)
                setattr(model, attr_name, fused_layer)
                # Update _layer_to_name if it exists
                if hasattr(model, '_layer_to_name'):
                    model._layer_to_name[id(fused_layer)] = attr_name
            
            # Clean up extra layer attributes
            for i in range(len(fused_sequential.modules), len(layers)):
                attr_name = f"layer_{i}"
                if hasattr(model, attr_name):
                    delattr(model, attr_name)
            
            return model
        
        # Otherwise, recursively process all Sequential submodules and complex modules in the model
        for attr_name in dir(model):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(model, attr_name, None)
            except:
                continue
                
            if attr is None or not isinstance(attr, Module):
                continue
                
            if isinstance(attr, Sequential):
                fused = self.fuse_sequential(attr)
                setattr(model, attr_name, fused)
            elif not isinstance(attr, (Linear, ReLU, BatchNorm1d, LayerNorm1d, Dropout, Identity, Flatten)):
                # Recursively process complex submodules
                self.fuse_model(attr)
        
        return model
    
    def print_fusion_report(self):
        """Print fusion report"""
        print(f"\n{'='*60}")
        print(f"Operator Fusion Report")
        print(f"{'='*60}")
        print(f"Total fusions: {self.fusion_count}")
        print(f"{'-'*60}")
        
        if self.fusion_log:
            print(f"{'Position':<8} {'Fusion Pattern':<25} {'Original Operators':<20}")
            print(f"{'-'*60}")
            for log in self.fusion_log:
                print(f"{log['position']:<8} {log['pattern']:<25} {log['original']:<20}")
        else:
            print("No fusion patterns found")
        
        print(f"{'='*60}\n")
    
    def get_fusion_stats(self) -> dict:
        """
        Get fusion statistics
        
        Returns:
            dict: Dictionary containing fusion statistics
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
    Convenient function to perform operator fusion on a model
    
    Args:
        model: Needle model to be fused
        verbose: Whether to print the fusion report
        
    Returns:
        Module: Fused model
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
    Get fusion statistics of the model (without modifying the model)
    
    Args:
        model: Needle model
        
    Returns:
        dict: Dictionary containing fusion statistics
    """
    fusion_engine = OperatorFusion()
    # Perform fusion on a temporary copy to get statistics
    import copy
    temp_model = copy.deepcopy(model)
    fusion_engine.fuse_model(temp_model)
    
    return fusion_engine.get_fusion_stats()


if __name__ == "__main__":
    print("算子融合模块测试")
    print("="*60)
    
    # Create a simple test model
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
    
    # Perform fusion
    fused_model = fuse_operators(test_model, verbose=True)
    
    print("融合后模型:")
    print(fused_model)
