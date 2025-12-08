import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from needle.autograd import Tensor
from needle.ops.ops_mathematic import relu, matmul, broadcast_to, summation, reshape
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
        LinearBatchNormReLUPattern, ConvBatchNorm2dReLUPattern, ConvBatchNorm2dPattern
    )
except ImportError:
    from operator_fusion.fusion_pattrern import (
        FusionPattern, LinearReLUPattern, LinearBatchNormPattern, BatchNormReLUPattern, 
        LinearBatchNormReLUPattern, ConvBatchNorm2dReLUPattern, ConvBatchNorm2dPattern
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
                ConvBatchNorm2dPattern(),  # Conv+BN
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
        
        # sorted layer attributes in FXGraphExecutor
        layer_attrs = sorted([name for name in dir(model) if name.startswith('layer_') and name.split('_')[1].isdigit()],
                             key=lambda x: int(x.split('_')[1]))
        if layer_attrs:
            # extracts all layer attributes
            layers = [getattr(model, attr) for attr in layer_attrs]
            
            # use a Sequential module to include all layers
            temp_sequential = Sequential(*layers)
            fused_sequential,fusion_info = self.fuse_sequential(temp_sequential,model.training)
            
            # important update: need to update original FX graph mapping
            # otherwise, the fused layers won't be recognized in the FX graph execution!!
            self._update_fx_graph_mapping(model, layers, fusion_info)
            self._refresh_use_counts(model)
            
            # replace back to FXGraphExecutor
            for i, fused_layer in enumerate(fused_sequential.modules):
                attr_name = f"layer_{i}"
                if hasattr(model, attr_name):
                    delattr(model, attr_name)
                setattr(model, attr_name, fused_layer)
                # update _layer_to_name mapping
                if hasattr(model, '_layer_to_name'):
                    model._layer_to_name[id(fused_layer)] = attr_name
            
            # clear redundant layer attributes
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
        Update FXGraphExecutor node to layer mapping
        """
        if not hasattr(model, 'node_to_layer'):
            return
        
        node_mapping_updates = {}
        
        fused_idx = 0
        orig_idx = 0
        
        for info in fusion_info:
            if info["type"] == "fused":
                # fused layer: map multiple original layers to the same fused layer
                consumed = info["consumed"]
                fused_module = info["fused_module"]
                
                for i in range(consumed): 
                    if orig_idx + i < len(original_layers):
                        # find all nodes that use this original layer
                        for node_name, layer in model.node_to_layer.items():
                            if layer is original_layers[orig_idx + i]:
                                if i == 0:  # first matched layer maps to fused layer
                                    node_mapping_updates[node_name] = fused_module
                                else:  # other layers in original fused layer changes to identity() #TODO: Identity may be removed afterward
                                    node_mapping_updates[node_name] = Identity()
                                break
                
                orig_idx += consumed
                fused_idx += 1
                
            elif info["type"] == "original":
                # original layer: one-to-one mapping
                if orig_idx < len(original_layers):
                    original_module = info["module"]
                    # find the node that uses this original layer
                    for node_name, layer in model.node_to_layer.items():
                        if layer is original_layers[orig_idx]:
                            node_mapping_updates[node_name] = original_module
                            break
                    orig_idx += 1
                    fused_idx += 1
                    
            elif info["type"] == "nested":
                # nested Sequential module: map original layers to nested module
                if orig_idx < len(original_layers):
                    nested_module = info["module"]
                    for node_name, layer in model.node_to_layer.items():
                        if layer is original_layers[orig_idx]:
                            node_mapping_updates[node_name] = nested_module
                    orig_idx += 1
                    fused_idx += 1
        
        # update the node_to_layer mapping
        model.node_to_layer.update(node_mapping_updates)
    
    def _refresh_use_counts(self, model):
        """
        refresh the _use_count attribute in FXGraphExecutor after fusion
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

