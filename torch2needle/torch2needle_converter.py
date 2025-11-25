import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from needle.nn.nn_basic import Module, Identity, Linear, Flatten, ReLU, Sequential, BatchNorm1d, LayerNorm1d, Dropout, BatchNorm2d
from needle.nn.nn_basic import ADD, SUB
from needle.nn.nn_conv import Conv, MaxPool2d, AdaptiveAvgPool2d
from needle.nn.nn_transformer import MultiHeadAttention, AttentionLayer, TransformerLayer, Transformer
from needle.nn.nn_sequence import Embedding
import needle as nd

# 尝试导入 utils，如果失败则跳过（用于独立导入时）
try:
    from utils import print_trace_grouped
except ImportError:
    from torch2needle.utils import print_trace_grouped

# 尝试导入 torch_models，如果失败则跳过（用于独立导入时）
try:
    from torch_models import *
except ImportError:
    pass

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import operator


# Torch2Needle, starter function
def torch2needle_fx(torch_model, device=nd.cpu(), dtype="float32"):
    traced = symbolic_trace(torch_model)
    print(traced)
    named_modules = dict(traced.named_modules())
    node_to_layer = {}

    # this is used for weight converter
    torch_mapping_needle = {}

    # 转换所有节点，而不仅仅是从输出节点开始
    trace_log = []
    for node in traced.graph.nodes:
        if node.name not in node_to_layer:
            _, trace_log = convert_node(node, named_modules, node_to_layer, torch_mapping_needle, 
                                       depth=0, parent=None, trace_log=trace_log, device=device, dtype=dtype)

    # add sequential modules to current torch-mapping-needle mapping relation
    populate_sequential_mapping(torch_model, named_modules, torch_mapping_needle)

    # construct executable Needle model
    needle_model = build_executable_model(torch_model, traced.graph, node_to_layer, torch_mapping_needle)

    return needle_model, trace_log, torch_mapping_needle


def build_executable_model(torch_model, fx_graph, node_to_layer, torch_mapping_needle):
    """
    构建可执行的 Needle 模型
    """
    # 特殊情况：如果模型就是一个 Sequential，直接返回它
    if isinstance(torch_model, nn.Sequential):
        return build_model_structure(torch_model, {}, torch_mapping_needle)
    
    # 特殊情况：如果模型只有一个 Sequential 子模块
    children = list(torch_model.children())
    if len(children) == 1 and isinstance(children[0], nn.Sequential):
        return build_model_structure(torch_model, {}, torch_mapping_needle)
    
    # 一般情况：创建一个执行 FX 图的包装器
    class FXGraphExecutor(Module):
        def __init__(self, fx_graph, node_to_layer):
            super().__init__()
            self.fx_graph = fx_graph
            self.node_to_layer = node_to_layer
            
            # 创建从 layer 到 node_name 的反向映射，用于打印时显示引用
            # 使用 _ 前缀使其在打印时被忽略
            self._layer_to_name = {}
            
            # 去重：只为每个唯一的 layer 对象创建一个属性
            seen_layers = {}  # id(layer) -> layer_name
            layer_counter = 0
            
            for node_name, layer in node_to_layer.items():
                if isinstance(layer, Identity):
                    continue
                    
                layer_id = id(layer)
                if layer_id not in seen_layers: # 如果这里只创建一个seen_layers会有问题吗？
                    # 首次遇到这个 layer，创建属性
                    layer_name = f"layer_{layer_counter}"
                    setattr(self, layer_name, layer)
                    seen_layers[layer_id] = layer_name
                    self._layer_to_name[layer_id] = layer_name
                    layer_counter += 1
        
        def __repr__(self):
            """自定义打印格式，显示 ADD/SUB 的 left/right 引用"""
            lines = ["FXGraphExecutor("]
            
            for attr_name in sorted(dir(self)):
                if attr_name.startswith('layer_'):
                    layer = getattr(self, attr_name)
                    layer_repr = self._format_layer(layer, indent=1)
                    lines.append(f"  ({attr_name}): {layer_repr}")
            
            lines.append(")")
            return "\n".join(lines)
        
        def _format_layer(self, layer, indent=0):
            """格式化单个层的表示，处理 ADD/SUB 的引用"""
            pad = "  " * indent
            
            if isinstance(layer, (ADD, SUB)):
                # 对于 ADD/SUB，显示 left/right 是哪个 layer
                op_name = type(layer).__name__
                left_ref = self._layer_to_name.get(id(layer.left), repr(layer.left))
                right_ref = self._layer_to_name.get(id(layer.right), repr(layer.right))
                
                return f"{op_name}(left={left_ref}, right={right_ref})"
            else:
                # 其他层直接使用其默认表示
                return repr(layer).replace('\n', '\n' + pad)
        
        def forward(self, *args):
            # 执行 FX 图
            env = {}
            for node in self.fx_graph.nodes:
                if node.op == "placeholder":
                    # 输入节点
                    env[node.name] = args[0] if len(args) == 1 else args[len(env)]
                elif node.op == "call_module":
                    # 调用模块
                    needle_layer = self.node_to_layer[node.name]
                    inputs = [env[arg.name] for arg in node.all_input_nodes]
                    env[node.name] = needle_layer(inputs[0])  # 模块调用只接受单个输入
                elif node.op == "call_function":
                    # 调用函数 - 直接对张量进行操作
                    # 获取输入张量
                    inputs = []
                    for arg in node.args:
                        if isinstance(arg, torch.fx.Node):
                            inputs.append(env[arg.name])
                    
                    # 根据操作类型执行
                    if node.target == operator.add:
                        env[node.name] = inputs[0] + inputs[1]
                    elif node.target == operator.sub:
                        env[node.name] = inputs[0] - inputs[1]
                    elif node.target == operator.mul:
                        env[node.name] = inputs[0] * inputs[1]
                    elif node.target == operator.truediv:
                        env[node.name] = inputs[0] / inputs[1]
                    else:
                        # 对于其他函数，尝试调用存储的模块
                        needle_layer = self.node_to_layer.get(node.name, Identity())
                        env[node.name] = needle_layer(inputs[0]) if inputs else needle_layer()
                
                elif node.op == "call_method":
                    # 调用方法 - 例如 tensor.mean(), tensor.size() 等
                    # 获取输入（方法调用的对象）
                    if len(node.all_input_nodes) > 0:
                        target_tensor = env[node.all_input_nodes[0].name]
                        method_name = node.target
                        
                        # 处理常见方法
                        if method_name == "mean":
                            # 提取 dim 参数
                            kwargs = node.kwargs if hasattr(node, 'kwargs') else {}
                            dim = kwargs.get('dim', None)
                            if dim is not None:
                                import needle.ops as ops
                                env[node.name] = ops.summation(target_tensor, axes=(dim,)) / target_tensor.shape[dim]
                            else:
                                # 全局 mean
                                import needle.ops as ops
                                total = 1
                                for s in target_tensor.shape:
                                    total *= s
                                env[node.name] = ops.summation(target_tensor) / total
                        elif method_name == "size":
                            # size() 方法 - 返回形状信息，在这里我们跳过
                            env[node.name] = target_tensor
                        elif method_name == "unsqueeze":
                            # unsqueeze - 扩展维度
                            env[node.name] = target_tensor
                        else:
                            # 其他方法，直接传递
                            env[node.name] = target_tensor
                    else:
                        env[node.name] = Identity()
                        
                elif node.op == "output":
                    # 输出节点
                    real_output = node.args
                    while isinstance(real_output, (tuple, list)) and len(real_output) == 1:
                        real_output = real_output[0]
                    if isinstance(real_output, torch.fx.Node):
                        return env[real_output.name]
                    elif isinstance(real_output, (tuple, list)):
                        return tuple(env[n.name] for n in real_output)
            return None
    
    return FXGraphExecutor(fx_graph, node_to_layer)


def populate_sequential_mapping(torch_model, named_modules, torch_mapping_needle):
    """
    iterate PyTorch model, add all Sequential modules to torch_mapping_needle
    so that weight_converter can see the complete mapping table (although Sequential itself will not load weights)
    """
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Sequential) and module not in torch_mapping_needle:
            # Construct the corresponding Needle Sequential
            needle_modules = []
            for sub_module in module:
                if sub_module in torch_mapping_needle:
                    needle_modules.append(torch_mapping_needle[sub_module])
                else:
                    # If the submodule has not been converted yet, skip (theoretically should not happen)
                    pass
            
            if needle_modules:  # Only create Sequential if there are submodules
                needle_seq = Sequential(*needle_modules)
                torch_mapping_needle[module] = needle_seq


def build_model_structure(torch_model, named_modules, torch_mapping_needle):
    """
    according to Pytroch model structure, build Needle model
    keep Sequential etc structure consistent with original model
    """
    # if the whole model is a Sequential
    if isinstance(torch_model, nn.Sequential):
        needle_modules = []
        for torch_module in torch_model:
            if torch_module in torch_mapping_needle:
                needle_modules.append(torch_mapping_needle[torch_module])
            else:
                # Recursively convert submodules
                needle_modules.append(build_model_structure(torch_module, named_modules, torch_mapping_needle))
        needle_seq = Sequential(*needle_modules)
        # Add the Sequential itself to the mapping
        torch_mapping_needle[torch_model] = needle_seq
        return needle_seq
    
    # If there is only one child module, directly return the converted module
    children_list = list(torch_model.children())
    if len(children_list) == 1:
        child = children_list[0]
        if child in torch_mapping_needle:
            return torch_mapping_needle[child]
        elif isinstance(child, nn.Sequential):
            # Convert Sequential
            needle_modules = []
            for sub_module in child:
                if sub_module in torch_mapping_needle:
                    needle_modules.append(torch_mapping_needle[sub_module])
                else:
                    needle_sub = convert_layer(sub_module)
                    torch_mapping_needle[sub_module] = needle_sub
                    needle_modules.append(needle_sub)
            needle_seq = Sequential(*needle_modules)
            torch_mapping_needle[child] = needle_seq
            return needle_seq
        else:
            # Single ordinary layer
            needle_module = convert_layer(child)
            torch_mapping_needle[child] = needle_module
            return needle_module
    
    # If the model has multiple named child modules
    # Create a wrapper module to maintain structure
    class NeedleModelWrapper(Module):
        def __init__(self, graph_module):
            super().__init__()
            self.graph_module = graph_module
            # Iterate over all child modules of the PyTorch model
            for name, torch_module in torch_model.named_children():
                if torch_module in torch_mapping_needle:
                    # Directly use the already converted module
                    setattr(self, name, torch_mapping_needle[torch_module])
                elif isinstance(torch_module, nn.Sequential):
                    # Sequential needs to be recursively converted
                    needle_seq_modules = []
                    for sub_module in torch_module:
                        if sub_module in torch_mapping_needle:
                            needle_seq_modules.append(torch_mapping_needle[sub_module])
                        else:
                            # Convert and add to mapping
                            needle_sub = convert_layer(sub_module)
                            torch_mapping_needle[sub_module] = needle_sub
                            needle_seq_modules.append(needle_sub)
                    needle_sequential = Sequential(*needle_seq_modules)
                    # Add the Sequential itself to the mapping
                    torch_mapping_needle[torch_module] = needle_sequential
                    setattr(self, name, needle_sequential)
                else:
                    # Other types of modules: convert and add to mapping
                    needle_module = convert_layer(torch_module)
                    torch_mapping_needle[torch_module] = needle_module
                    setattr(self, name, needle_module)
        
        def forward(self, x):
            # This forward cannot be generally implemented because different models have different computation graphs
            # Actually, for multi-module models, FX graph execution should be used
            # Here, raise an error to indicate that the wrapper's forward should not be called directly
            raise NotImplementedError(
                "NeedleModelWrapper is a structural wrapper and should not be called directly. "
                "The converted model should use the FX graph execution path."
            )
    
    return NeedleModelWrapper(None)

#### here is the complete mapping logic ####
#### just map each torch layer to needle one and initialize in needle ####

def convert_layer(layer,device,dtype):
    """transform a single PyTorch layer into Needle layer"""
    if isinstance(layer, nn.Linear):
        bias = False
        if layer.bias is not None:
            bias = True
        return Linear(layer.in_features, layer.out_features,bias=bias,device=device,dtype=dtype)
    elif isinstance(layer, nn.ReLU):
        return ReLU()
    elif isinstance(layer, nn.Flatten):
        return Flatten()
    elif isinstance(layer, nn.Dropout):
        return Dropout(layer.p)
    elif isinstance(layer, nn.BatchNorm1d):
        return BatchNorm1d(layer.num_features,eps=layer.eps,momentum=layer.momentum,device=device,dtype=dtype)
    elif isinstance(layer, nn.BatchNorm2d):
        return BatchNorm2d(layer.num_features,eps=layer.eps,momentum=layer.momentum,device=device,dtype=dtype)
    elif isinstance(layer, nn.LayerNorm):
        return LayerNorm1d(layer.normalized_shape[0],device=device,dtype=dtype)
    elif isinstance(layer, nn.Identity):
        return Identity()
    elif isinstance(layer, nn.Sequential):
        return Sequential(*[convert_layer(m) for m in layer])
    elif isinstance(layer, nn.Conv2d):
        bias = False
        if layer.bias is not None:
            bias = True
        return Conv(in_channels=layer.in_channels,
                          out_channels=layer.out_channels,
                          kernel_size=layer.kernel_size,
                          stride=layer.stride,
                          bias=bias,
                          device=device,
                          dtype=dtype)
    elif isinstance(layer, nn.MaxPool2d):
        return MaxPool2d(kernel_size=layer.kernel_size,
                               stride=layer.stride,
                               padding=layer.padding)
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        return AdaptiveAvgPool2d(output_size=layer.output_size)
    
    # Transformer components-may not work too well, since our needle implementation is different from torch implementation
    elif isinstance(layer, nn.MultiheadAttention):
        # PyTorch MultiheadAttention parameters
        embed_dim = layer.embed_dim
        num_heads = layer.num_heads
        dropout = layer.dropout
        # Note: PyTorch's MultiheadAttention has batch_first parameter
        # but our Needle implementation has causal parameter
        # We'll use default causal=False for now
        return MultiHeadAttention(
            dropout=dropout,
            causal=False,
            device=device,
            dtype=dtype
        )
    
    elif isinstance(layer, nn.Embedding):
        return Embedding(
            layer.num_embeddings,
            layer.embedding_dim,
            device=device,
            dtype=dtype
        )
    
    elif isinstance(layer, nn.TransformerEncoderLayer):
        # PyTorch TransformerEncoderLayer parameters
        d_model = layer.self_attn.embed_dim
        nhead = layer.self_attn.num_heads
        dim_feedforward = layer.linear1.out_features
        dropout = layer.dropout.p if hasattr(layer.dropout, 'p') else 0.0
        
        # Calculate dim_head from d_model and nhead
        dim_head = d_model // nhead
        
        return TransformerLayer(
            q_features=d_model,
            num_head=nhead,
            dim_head=dim_head,
            hidden_size=dim_feedforward,
            dropout=dropout,
            causal=False,  # Encoder layers are typically not causal
            device=device,
            dtype=dtype
        )
    
    elif isinstance(layer, nn.TransformerDecoderLayer):
        # PyTorch TransformerDecoderLayer parameters
        d_model = layer.self_attn.embed_dim
        nhead = layer.self_attn.num_heads
        dim_feedforward = layer.linear1.out_features
        dropout = layer.dropout.p if hasattr(layer.dropout, 'p') else 0.0
        
        # Calculate dim_head from d_model and nhead
        dim_head = d_model // nhead
        
        return TransformerLayer(
            q_features=d_model,
            num_head=nhead,
            dim_head=dim_head,
            hidden_size=dim_feedforward,
            dropout=dropout,
            causal=True,  # Decoder layers are typically causal
            device=device,
            dtype=dtype
        )
    
    elif isinstance(layer, nn.TransformerEncoder):
        # PyTorch TransformerEncoder is a stack of TransformerEncoderLayer
        # We need to convert each layer individually and store the mapping
        # This should NOT be called directly - the convert_layer is typically called
        # from convert_node which handles the mapping
        # For now, just convert each layer
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            needle_layers = []
            for torch_enc_layer in layer.layers:
                # Convert each TransformerEncoderLayer
                needle_layer = convert_layer(torch_enc_layer, device, dtype)
                needle_layers.append(needle_layer)
            return Sequential(*needle_layers)
        else:
            print("[Warning] TransformerEncoder has no layers")
            return Identity()
    
    elif isinstance(layer, nn.TransformerDecoder):
        # PyTorch TransformerDecoder is a stack of TransformerDecoderLayer
        # Extract parameters from the first layer
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            first_layer = layer.layers[0]
            d_model = first_layer.self_attn.embed_dim
            nhead = first_layer.self_attn.num_heads
            dim_feedforward = first_layer.linear1.out_features
            dropout = first_layer.dropout.p if hasattr(first_layer.dropout, 'p') else 0.0
            num_layers = len(layer.layers)
            
            # Calculate dim_head
            dim_head = d_model // nhead
            
            # Create a Sequential of TransformerLayer instances
            needle_layers = []
            for _ in range(num_layers):
                needle_layers.append(
                    TransformerLayer(
                        q_features=d_model,
                        num_head=nhead,
                        dim_head=dim_head,
                        hidden_size=dim_feedforward,
                        dropout=dropout,
                        causal=True,  # Decoder is causal
                        device=device,
                        dtype=dtype
                    )
                )
            return Sequential(*needle_layers)
        else:
            print("[Warning] TransformerDecoder has no layers")
            return Identity()
    
    elif isinstance(layer, nn.Transformer):
        # PyTorch Transformer parameters
        d_model = layer.d_model
        nhead = layer.nhead
        num_encoder_layers = layer.num_encoder_layers
        num_decoder_layers = layer.num_decoder_layers
        dim_feedforward = layer.encoder.layers[0].linear1.out_features if hasattr(layer, 'encoder') else 2048
        dropout = layer.dropout
        
        # Calculate dim_head
        dim_head = d_model // nhead
        
        # For simplicity, we'll treat it as a decoder-only transformer (causal=True)
        # and use total number of layers
        num_layers = num_encoder_layers + num_decoder_layers
        
        return Transformer(
            embedding_size=d_model,
            hidden_size=dim_feedforward,
            num_layers=num_layers,
            num_head=nhead,
            dim_head=dim_head,
            dropout=dropout,
            causal=True,
            device=device,
            dtype=dtype,
            batch_first=True,  # Assuming batch_first
            sequence_len=2048  # Default sequence length
        )
    
    else:
        print(f"[Warning] Unsupported layer: {layer.__class__.__name__}, replaced with Identity().")
        raise NotImplementedError

# convert ADD/SUB and so on
def convert_function_node(node, named_modules, node_to_layer, torch_mapping_needle, depth, trace_log,device,dtype):
    """
    deal with call_function nodes:
      - operator.add
      - operator.sub)
      - operator.mul
      - operator.truediv
      - flatten (torch.flatten)
      - Identity
    return: (needle_module, note_string)
    """
    op = node.target


    if op == operator.add:
        left, trace_log = convert_node(node.args[0], named_modules, node_to_layer, torch_mapping_needle,
                                       depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
        right, trace_log = convert_node(node.args[1], named_modules, node_to_layer, torch_mapping_needle,
                                        depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
        module = ADD(left, right)
        note = "operator.add"
    elif op == operator.sub:
        left, trace_log = convert_node(node.args[0], named_modules, node_to_layer, torch_mapping_needle,
                                       depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
        right, trace_log = convert_node(node.args[1], named_modules, node_to_layer, torch_mapping_needle,
                                        depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
        module = SUB(left, right)
        note = "operator.sub"
    # elif op == operator.mul:
    #     left, trace_log = convert_node(node.args[0], named_modules, node_to_layer,
    #                                    depth + 1, parent=node.name, trace_log=trace_log)
    #     right, trace_log = convert_node(node.args[1], named_modules, node_to_layer,
    #                                     depth + 1, parent=node.name, trace_log=trace_log)
    #     module = MUL(left, right)  # 你可以定义一个 MUL 模块类似 ADD
    #     note = "operator.mul"
    # elif op == operator.truediv:
    #     left, trace_log = convert_node(node.args[0], named_modules, node_to_layer,
    #                                    depth + 1, parent=node.name, trace_log=trace_log)
    #     right, trace_log = convert_node(node.args[1], named_modules, node_to_layer,
    #                                     depth + 1, parent=node.name, trace_log=trace_log)
    #     module = DIV(left, right)  # 可定义 DIV 模块
    #     note = "operator.truediv"

    elif op == torch.flatten:
        module = Flatten()
        note = "torch.flatten"
    
    elif op == operator.getitem:
        # Handle tuple/list indexing (e.g., for multi-output operations)
        # Convert the input node first
        input_node, trace_log = convert_node(node.args[0], named_modules, node_to_layer, torch_mapping_needle,
                                             depth + 1, parent=node.name, trace_log=trace_log, device=device, dtype=dtype)
        # For getitem, we just pass through the input module
        # The actual indexing will be handled at runtime
        module = Identity()
        note = "operator.getitem (pass-through)"
    
    elif op == getattr or (hasattr(__builtins__, 'getattr') and op == __builtins__.getattr):
        # Handle getattr operations (e.g., accessing module attributes)
        module = Identity()
        note = "getattr (pass-through)"
    
    elif hasattr(torch, 'arange') and op == torch.arange:
        # Handle torch.arange - used for position encoding
        module = Identity()
        note = "torch.arange (pass-through)"
    
    elif hasattr(torch.Tensor, 'size') and str(op) == 'size':
        # Handle tensor.size() calls
        module = Identity()
        note = "tensor.size (pass-through)"
    
    elif hasattr(torch.Tensor, 'unsqueeze') and str(op) == 'unsqueeze':
        # Handle tensor.unsqueeze() calls
        module = Identity()
        note = "tensor.unsqueeze (pass-through)"

    # unimplemented operator
    else:
        print(f"[Warning] Unsupported function op: {op}")
        raise NotImplementedError

    return module, note, trace_log

# main Torch2Needle converter
def convert_node(node, named_modules, node_to_layer, torch_mapping_needle, depth=0, parent=None, trace_log=None,device=nd.cpu(), dtype="float32"):
    if trace_log is None:
        trace_log = []

    indent = "  " * depth
    entry = {
        "name": node.name,
        "op": node.op,
        "target": str(node.target),
        "depth": depth,
        "inputs": [n.name for n in node.all_input_nodes],
        "parent": parent,
        "module_type": None,
        "needle_type": None,
        "note": "",
    }

    if node.name in node_to_layer:
        entry["needle_type"] = type(node_to_layer[node.name]).__name__
        entry["note"] = "cache hit"
        trace_log.append(entry)
        return node_to_layer[node.name], trace_log

    module = None

    # === 1️.placeholder ===
    if node.op == "placeholder":
        module = Identity()
        entry["module_type"],  entry["needle_type"] = "placeholder", "Identity"

    # === 2.call_module ===
    elif node.op == "call_module":
        target = node.target
        torch_layer = named_modules[target]
        entry["module_type"] = type(torch_layer).__name__

        # ensure all input nodes are converted first (but not combined)
        for arg in node.all_input_nodes:
            if arg.name not in node_to_layer and arg.op != "placeholder":
                _, trace_log = convert_node(arg, named_modules, node_to_layer, torch_mapping_needle,
                                           depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)

        # check if this PyTorch layer has been converted before (reuse the same Needle module)
        if torch_layer in torch_mapping_needle:
            module = torch_mapping_needle[torch_layer]
            entry["note"] = "reusing existing needle module"
        else:
            # transform current layer
            module = convert_layer(torch_layer,device,dtype)
            torch_mapping_needle[torch_layer] = module # record mapping relation from torch model to needle model
            
            # Special handling for TransformerEncoder/Decoder: map individual layers
            if isinstance(torch_layer, (nn.TransformerEncoder, nn.TransformerDecoder)):
                if hasattr(torch_layer, 'layers') and isinstance(module, Sequential):
                    # Map each PyTorch encoder/decoder layer to corresponding Needle layer
                    for torch_sublayer, needle_sublayer in zip(torch_layer.layers, module.modules):
                        torch_mapping_needle[torch_sublayer] = needle_sublayer
        
        entry["needle_type"] = type(module).__name__

    # === 3.call_function ===
    elif node.op == "call_function":
        entry["module_type"] = "function"
        module, note, trace_log = convert_function_node(
            node, named_modules, node_to_layer, torch_mapping_needle, depth, trace_log,device,dtype
        )
        entry["needle_type"] = type(module).__name__
        entry["note"] = note

    # === 4.call_method ===
    elif node.op == "call_method":
        entry["module_type"] = "method"
        # Handle tensor method calls like .mean(), .size(), etc.
        # These are typically pass-through as they don't have learnable parameters
        module = Identity()
        entry["needle_type"] = "Identity"
        entry["note"] = f"method call: {node.target} (pass-through)"

    # === 5️.output ===
    elif node.op == "output":
        entry["module_type"] = "output"
        real_output = node.args
        while isinstance(real_output, (tuple, list)) and len(real_output) == 1:
            real_output = real_output[0]
        if isinstance(real_output, torch.fx.Node):
            module, trace_log = convert_node(real_output, named_modules, node_to_layer, torch_mapping_needle,
                                             depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
            entry["needle_type"] = type(module).__name__
        elif isinstance(real_output, (tuple, list)) and all(isinstance(n, torch.fx.Node) for n in real_output):
            # 多输出情况：只转换节点，不创建 Sequential
            # 返回最后一个节点的模块（通常是主输出）
            for n in real_output:
                module, trace_log = convert_node(n, named_modules, node_to_layer, torch_mapping_needle,
                                                depth + 1, parent=node.name, trace_log=trace_log,device=device,dtype=dtype)
            # module 已经是最后一个节点的模块
            entry["needle_type"] = type(module).__name__
        else:
            module = Identity()
            entry["needle_type"] = "Identity"
            entry["note"] = f"unrecognized output args {real_output}"

    else:
        module = Identity()
        entry["needle_type"] = "Identity"
        entry["note"] = f"unknown node type {node.op}"

    node_to_layer[node.name] = module
    trace_log.append(entry)
    return module, trace_log


if __name__ == "__main__":
    print("this is test for torch2needle converter")
    torch_model = TorchMLP_v2()
    needle_model, trace_log, torch_mapping_needle = torch2needle_fx(torch_model)
    print("======== Torch Structure =========")
    print(torch_model)
    print("======== Needle Model Structure ========")
    print(needle_model)
    print("======== Needle print_trace_grouped ========")
    print_trace_grouped(trace_log)
    print("======== Torch Mapping Needle ========\r")
    print(torch_mapping_needle)
    print(len(torch_mapping_needle))

