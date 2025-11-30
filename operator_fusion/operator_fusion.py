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
    fused_conv_batchnorm2d_relu,
)
import needle.init as init
from needle.nn.nn_basic import (
    Module, Linear, ReLU, Sequential, BatchNorm1d, BatchNorm2d,
    LayerNorm1d, Dropout, Identity, Flatten, Parameter
)
from needle.nn.nn_conv import Conv
from typing import Any, List, Tuple, Optional

try:
    from fusion_pattrern import (
        FusionPattern, LinearReLUPattern, LinearBatchNormPattern, BatchNormReLUPattern, 
        LinearBatchNormReLUPattern, ConvBatchNorm2dReLUPattern
    )
except ImportError:
    from operator_fusion.fusion_pattrern import (
        FusionPattern, LinearReLUPattern, LinearBatchNormPattern, BatchNormReLUPattern, 
        LinearBatchNormReLUPattern, ConvBatchNorm2dReLUPattern
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
                ConvBatchNorm2dReLUPattern(),  # Conv+BN+ReLU (ResNet pattern)
                LinearBatchNormReLUPattern(),  # Linear+BN+ReLU
                LinearBatchNormPattern(),      # Linear+BN
                LinearReLUPattern(),           # Linear+ReLU
                BatchNormReLUPattern(),        # BN+ReLU
            ]
        else:
            self.patterns = patterns
        
        self.fusion_count = 0  # the number of fusions performed
        self.fusion_log = []   # the fusion log
    
    def fuse_sequential(self, sequential: Sequential,train=True) -> Sequential:
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
        fusion_info = [] # record fusion information
        i = 0
        
        # the basic algorithm is: iterate through the modules, at each position try to match any fusion pattern
        while i < len(modules):
            # iterate nested Sequential modules recursively
            if isinstance(modules[i], Sequential):
                fused_child, _ = self.fuse_sequential(modules[i],train)
                fused_modules.append(fused_child)
                fusion_info.append({
                    "type":"nested",
                    "module": fused_child,
                    "consumed": 1,
                })
                i += 1
                continue
            
            # try to match fusion patterns
            matched = False
            for pattern in self.patterns:
                # iterate our all fusion patterns
                if pattern.match(modules, i):
                    # perform fusion
                    fused_module, consumed = pattern.fuse(modules, i)
                    if hasattr(fused_module, 'training'):
                        fused_module.training = train
                    fused_modules.append(fused_module)
                    # record fusion information
                    # first update this record to fusion_info
                    fusion_info.append({
                        "type":"fused",
                        "pattern": pattern.__class__.__name__,
                        "consumed": consumed,
                        "fused_module": fused_module,  
                    })

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
                fusion_info.append({
                    "type":"original",
                    "module": modules[i],
                    "consumed": 1
                })
                i += 1
        
        return Sequential(*fused_modules),fusion_info
    
    def fuse_model(self, model: Module) -> Module:
        self.fusion_count = 0
        self.fusion_log = []

        #If the model itself is a Sequential, fuse directly
        if isinstance(model, Sequential):
            fused_sequential,fusion_info = self.fuse_sequential(model,model.training)
            self._update_fx_graph_mapping(model,model.modules, fused_sequential,fusion_info)
            self._refresh_use_counts(model)
            return fused_sequential
        
        # 检查是否是 FXGraphExecutor（通过检查是否有 layer_N 属性）
        layer_attrs = sorted([name for name in dir(model) if name.startswith('layer_') and name.split('_')[1].isdigit()],
                             key=lambda x: int(x.split('_')[1]))
        if layer_attrs:
            # FXGraphExecutor 情况：提取所有 layer_N，融合它们，并替换回去
            layers = [getattr(model, attr) for attr in layer_attrs]
            
            # 创建临时 Sequential 进行融合
            temp_sequential = Sequential(*layers)
            fused_sequential,fusion_info = self.fuse_sequential(temp_sequential,model.training)
            
            # important update: need to update original FX graph mapping
            # otherwise, the fused layers won't be recognized in the FX graph execution!!
            self._update_fx_graph_mapping(model, layers, fusion_info)
            self._refresh_use_counts(model)
            
            # 替换回 FXGraphExecutor
            for i, fused_layer in enumerate(fused_sequential.modules):
                attr_name = f"layer_{i}"
                if hasattr(model, attr_name):
                    delattr(model, attr_name)
                setattr(model, attr_name, fused_layer)
                # 更新 _layer_to_name 映射
                if hasattr(model, '_layer_to_name'):
                    model._layer_to_name[id(fused_layer)] = attr_name
            
            # 清理多余的 layer 属性
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
            # TODO:this line may cause issues if there are custom modules that can be fused
            elif not isinstance(attr, (Linear, ReLU, BatchNorm1d, LayerNorm1d, Dropout, Identity, Flatten,Conv,BatchNorm2d)):
                # Recursively process complex submodules
                self.fuse_model(attr)
        
        return model
    
    def _update_fx_graph_mapping(self, model, original_layers, fusion_info):
        """
        更新 FXGraphExecutor 的节点到层映射关系，确保融合后的层能正确执行
        """
        if not hasattr(model, 'node_to_layer'):
            return
        
        # 构建节点名称到融合层的映射
        node_mapping_updates = {}
        
        fused_idx = 0
        orig_idx = 0
        
        for info in fusion_info:
            if info["type"] == "fused":
                # 融合层：将多个原始层映射到同一个融合层
                consumed = info["consumed"]
                fused_module = info["fused_module"]
                
                for i in range(consumed): 
                    if orig_idx + i < len(original_layers):
                        # 找到使用这个原始层的所有节点
                        for node_name, layer in model.node_to_layer.items():
                            if layer is original_layers[orig_idx + i]:
                                if i == 0:  # 第一个层映射到融合层
                                    node_mapping_updates[node_name] = fused_module
                                else:  # 其他被融合的层映射到Identity #TODO: Identity may be removed afterward
                                    node_mapping_updates[node_name] = Identity()
                                break
                
                orig_idx += consumed
                fused_idx += 1
                
            elif info["type"] == "original":
                # 原始层：一对一映射
                if orig_idx < len(original_layers):
                    original_module = info["module"]
                    # 找到使用这个原始层的所有节点
                    for node_name, layer in model.node_to_layer.items():
                        if layer is original_layers[orig_idx]:
                            node_mapping_updates[node_name] = original_module
                            break
                    orig_idx += 1
                    fused_idx += 1
                    
            elif info["type"] == "nested":
                # 嵌套 Sequential：暂时按一对一处理，如果需要可以递归处理
                if orig_idx < len(original_layers):
                    nested_module = info["module"]
                    for node_name, layer in model.node_to_layer.items():
                        if layer is original_layers[orig_idx]:
                            node_mapping_updates[node_name] = nested_module
                    orig_idx += 1
                    fused_idx += 1
        
        # 应用更新
        model.node_to_layer.update(node_mapping_updates)
    
    def _refresh_use_counts(self, model):
        """
        融合后重新计算 FXGraphExecutor 的 _use_count 字段
        """
        if not hasattr(model,'fx_graph') or not hasattr(model,'_use_count'):
            return
        new_counts = {}
        for node in model.fx_graph.nodes:
            if node.name not in new_counts:
                new_counts[node.name] = 0
            for input_node in node.all_input_nodes:
                new_counts[input_node.name] = new_counts.get(input_node.name,0) + 1
        model._use_count = new_counts

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
    def _ensure_mapping_consistency(self, model):
        """
        A function to ensure that the FXGraphExecutor's node_to_layer mapping is consistent with the model's current layers
        after fusion. Use Just for test & debug.
        """
        if not hasattr(model, 'node_to_layer'):
            return
        
        # 构建当前层属性到层的映射
        current_layers = {}
        for attr_name in dir(model):
            if attr_name.startswith('layer_'):
                layer = getattr(model, attr_name)
                current_layers[attr_name] = layer
        
        # 检查 node_to_layer 中的层是否都在当前层属性中
        for node_name, layer in model.node_to_layer.items():
            layer_found = any(id(layer) == id(current_layer) for current_layer in current_layers.values())
            if not layer_found:
                print(f"Warning: Layer for node {node_name} not found in current model attributes")


# ============================================================================
# Main function to start operator fusion
# ============================================================================

def fuse_model_with_mapping(self, model: Module) -> Module:
    """
    执行融合并确保映射正确更新
    """
    # 执行常规融合
    fused_model = self.fuse_model(model)
    
    # 确保 FXGraphExecutor 的映射已更新
    if hasattr(fused_model, 'node_to_layer'):
        self._ensure_mapping_consistency(fused_model)
    
    return fused_model

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
    # patterns = [
    #     LinearBatchNormReLUPattern(), 
    #     LinearBatchNormPattern(),
    #     LinearReLUPattern(),
    #     BatchNormReLUPattern(),
    # ]
    fusion_engine = OperatorFusion()
    #fused_model = fusion_engine.fuse_model(model)
    fused_model = fuse_model_with_mapping(fusion_engine, model)
    
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
    # test_model = Sequential(
    #     Linear(10, 20),
    #     ReLU(),
    #     Linear(20, 30),
    #     BatchNorm1d(30),
    #     ReLU(),
    #     Linear(30, 10)
    # )
    # convolutional test model
    test_model = Sequential(
        Conv(3, 16, kernel_size=3, stride=1,
             bias=True),
        BatchNorm2d(16),
        ReLU(),
        Conv(16, 32, kernel_size=3, stride=1,
             bias=True),
        BatchNorm2d(32),
        ReLU(),
    )
    print("原始模型")
    print(test_model)
    print()

    # Perform fusion
    fused_model = fuse_operators(test_model, verbose=True)
    print("融合后模型")
    print(fused_model)
    # test forward pass of fused model and original model
    x = Tensor(init.rand(5,3,32,32))
    original_output = test_model(x)
    fused_output = fused_model(x)
    print()
    print("原始模型输出:")
    print(original_output)
    print("融合后模型输出:")
    print(fused_output)
    print("输出差异 (原始 - 融合):")
    print((original_output.numpy() - fused_output.numpy()).sum())  # should be close
    # #test_model.eval()  # Set to eval mode for BatchNorm
    # print("原始模型:")
    # print(test_model)
    # print()
    
    # # Perform fusion
    # fused_model = fuse_operators(test_model, verbose=True)
    
    # print("融合后模型:")
    # print(fused_model)

    # # test forward pass of fused model and original model
    # x = Tensor(init.rand(5,10))
    # original_output = test_model(x)
    # fused_output = fused_model(x)
    # print()
    # print("原始模型输出:")
    # print(original_output)
    # print("融合后模型输出:")
    # print(fused_output)
    # print("输出差异 (原始 - 融合):")
    # print((original_output.numpy() - fused_output.numpy()).sum())  # should be close to 0
