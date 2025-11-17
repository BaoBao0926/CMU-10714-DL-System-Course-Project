"""
我现在想实现CMU的DL system的一个功能，给一个torch model，直接根据计算图转成needl的mdoel
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from needle.nn.nn_basic import Module, Identity, Linear, Flatten, ReLU, Sequential, BatchNorm1d, LayerNorm1d, Dropout, Residual, SoftmaxLoss
from needle.nn.nn_basic import ADD, SUB
from needle import Tensor
import needle
from utils import print_trace_grouped
from torch_models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import needle.init as init
import random
from torch.fx import symbolic_trace
import operator


# Torch2Needle, starter function
def torch2needle_fx(torch_model):
    traced = symbolic_trace(torch_model)
    print(traced)
    named_modules = dict(traced.named_modules())
    node_to_layer = {}

    # this is used for weight converter
    torch_mapping_needle = {}

    output_node = list(traced.graph.nodes)[-1]
    _, trace_log = convert_node(output_node, named_modules, node_to_layer, torch_mapping_needle)

    # 补充 Sequential 模块到 mapping（用于权重加载时的完整性）
    populate_sequential_mapping(torch_model, named_modules, torch_mapping_needle)

    # 构建可执行的 Needle 模型
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
            
            # 将所有层作为属性存储，以便 parameters() 可以找到它们
            for i, (node_name, layer) in enumerate(node_to_layer.items()):
                if not isinstance(layer, Identity):
                    setattr(self, f"layer_{i}", layer)
        
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
    遍历 PyTorch 模型，将所有 Sequential 模块添加到 torch_mapping_needle
    这样 weight_converter 可以看到完整的映射表（虽然 Sequential 本身不会被加载权重）
    """
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Sequential) and module not in torch_mapping_needle:
            # 构建对应的 Needle Sequential
            needle_modules = []
            for sub_module in module:
                if sub_module in torch_mapping_needle:
                    needle_modules.append(torch_mapping_needle[sub_module])
                else:
                    # 如果子模块还没转换，跳过（理论上不应该发生）
                    pass
            
            if needle_modules:  # 只有当有子模块时才创建 Sequential
                needle_seq = Sequential(*needle_modules)
                torch_mapping_needle[module] = needle_seq


def build_model_structure(torch_model, named_modules, torch_mapping_needle):
    """
    根据 PyTorch 模型的原始结构构建 Needle 模型
    保持 Sequential 等结构与原始模型一致
    """
    # 如果整个模型就是一个 Sequential
    if isinstance(torch_model, nn.Sequential):
        needle_modules = []
        for torch_module in torch_model:
            if torch_module in torch_mapping_needle:
                needle_modules.append(torch_mapping_needle[torch_module])
            else:
                # 递归转换子模块
                needle_modules.append(build_model_structure(torch_module, named_modules, torch_mapping_needle))
        needle_seq = Sequential(*needle_modules)
        # 将 Sequential 本身也添加到映射中
        torch_mapping_needle[torch_model] = needle_seq
        return needle_seq
    
    # 如果只有一个子模块，直接返回转换后的模块
    children_list = list(torch_model.children())
    if len(children_list) == 1:
        child = children_list[0]
        if child in torch_mapping_needle:
            return torch_mapping_needle[child]
        elif isinstance(child, nn.Sequential):
            # 转换 Sequential
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
            # 单个普通层
            needle_module = convert_layer(child)
            torch_mapping_needle[child] = needle_module
            return needle_module
    
    # 如果模型有多个命名的子模块
    # 创建一个包装模块来保持结构
    class NeedleModelWrapper(Module):
        def __init__(self, graph_module):
            super().__init__()
            self.graph_module = graph_module
            # 遍历 PyTorch 模型的所有子模块
            for name, torch_module in torch_model.named_children():
                if torch_module in torch_mapping_needle:
                    # 直接使用已转换的模块
                    setattr(self, name, torch_mapping_needle[torch_module])
                elif isinstance(torch_module, nn.Sequential):
                    # Sequential 需要递归转换
                    needle_seq_modules = []
                    for sub_module in torch_module:
                        if sub_module in torch_mapping_needle:
                            needle_seq_modules.append(torch_mapping_needle[sub_module])
                        else:
                            # 转换并添加到映射
                            needle_sub = convert_layer(sub_module)
                            torch_mapping_needle[sub_module] = needle_sub
                            needle_seq_modules.append(needle_sub)
                    needle_sequential = Sequential(*needle_seq_modules)
                    # 将 Sequential 本身也添加到映射中
                    torch_mapping_needle[torch_module] = needle_sequential
                    setattr(self, name, needle_sequential)
                else:
                    # 其他类型的模块：转换并添加到映射
                    needle_module = convert_layer(torch_module)
                    torch_mapping_needle[torch_module] = needle_module
                    setattr(self, name, needle_module)
        
        def forward(self, x):
            # 这个 forward 不能通用实现，因为不同模型有不同的计算图
            # 实际上对于多模块的模型，应该使用 FX 图来执行
            # 这里抛出错误，提示用户不应该直接调用包装器的 forward
            raise NotImplementedError(
                "NeedleModelWrapper is a structural wrapper and should not be called directly. "
                "The converted model should use the FX graph execution path."
            )
    
    return NeedleModelWrapper(None)

# convert Linear Layers/CNN layers and so on
def convert_layer(layer):
    """把单个 PyTorch 层转成 Needle 层"""
    if isinstance(layer, nn.Linear):
        return Linear(layer.in_features, layer.out_features)
    elif isinstance(layer, nn.ReLU):
        return ReLU()
    elif isinstance(layer, nn.Flatten):
        return Flatten()
    elif isinstance(layer, nn.Dropout):
        return Dropout(layer.p)
    elif isinstance(layer, nn.BatchNorm1d):
        return BatchNorm1d(layer.num_features)
    elif isinstance(layer, nn.LayerNorm):
        return LayerNorm1d(layer.normalized_shape[0])
    elif isinstance(layer, nn.Identity):
        return Identity()
    elif isinstance(layer, nn.Sequential):
        return Sequential(*[convert_layer(m) for m in layer])
    else:
        print(f"[Warning] Unsupported layer: {layer.__class__.__name__}, replaced with Identity().")
        raise NotImplementedError

# convert ADD/SUB and so on
def convert_function_node(node, named_modules, node_to_layer, torch_mapping_needle, depth, trace_log):
    """
    处理 FX 中的 call_function 类型节点，支持常见算子：
      - 加法 (operator.add)
      - 减法 (operator.sub)
      - 乘法 (operator.mul)
      - 除法 (operator.truediv)
      - flatten (torch.flatten)
      - 其它默认 Identity
    返回: (needle_module, note_string)
    """
    op = node.target


    if op == operator.add:
        left, trace_log = convert_node(node.args[0], named_modules, node_to_layer, torch_mapping_needle,
                                       depth + 1, parent=node.name, trace_log=trace_log)
        right, trace_log = convert_node(node.args[1], named_modules, node_to_layer, torch_mapping_needle,
                                        depth + 1, parent=node.name, trace_log=trace_log)
        module = ADD(left, right)
        note = "operator.add"
    elif op == operator.sub:
        left, trace_log = convert_node(node.args[0], named_modules, node_to_layer, torch_mapping_needle,
                                       depth + 1, parent=node.name, trace_log=trace_log)
        right, trace_log = convert_node(node.args[1], named_modules, node_to_layer, torch_mapping_needle,
                                        depth + 1, parent=node.name, trace_log=trace_log)
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

    # unimplemented operator
    else:
        print(f"[Warning] Unsupported function op: {op}")
        raise NotImplementedError

    return module, note, trace_log

# main Torch2Needle converter
def convert_node(node, named_modules, node_to_layer, torch_mapping_needle, depth=0, parent=None, trace_log=None):
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

        # 先确保所有输入节点都已转换（但不组合它们）
        for arg in node.all_input_nodes:
            if arg.name not in node_to_layer and arg.op != "placeholder":
                _, trace_log = convert_node(arg, named_modules, node_to_layer, torch_mapping_needle,
                                           depth + 1, parent=node.name, trace_log=trace_log)

        # 检查是否已经转换过这个 PyTorch 层（重用同一个 Needle 模块）
        if torch_layer in torch_mapping_needle:
            module = torch_mapping_needle[torch_layer]
            entry["note"] = "reusing existing needle module"
        else:
            # 转换当前层
            module = convert_layer(torch_layer)
            torch_mapping_needle[torch_layer] = module
        
        entry["needle_type"] = type(module).__name__

    # === 3.call_function ===
    elif node.op == "call_function":
        entry["module_type"] = "function"
        module, note, trace_log = convert_function_node(
            node, named_modules, node_to_layer, torch_mapping_needle, depth, trace_log
        )
        entry["needle_type"] = type(module).__name__
        entry["note"] = note

    # === 4️.output ===
    elif node.op == "output":
        entry["module_type"] = "output"
        real_output = node.args
        while isinstance(real_output, (tuple, list)) and len(real_output) == 1:
            real_output = real_output[0]
        if isinstance(real_output, torch.fx.Node):
            module, trace_log = convert_node(real_output, named_modules, node_to_layer, torch_mapping_needle,
                                             depth + 1, parent=node.name, trace_log=trace_log)
            entry["needle_type"] = type(module).__name__
        elif isinstance(real_output, (tuple, list)) and all(isinstance(n, torch.fx.Node) for n in real_output):
            # 多输出情况：只转换节点，不创建 Sequential
            # 返回最后一个节点的模块（通常是主输出）
            for n in real_output:
                module, trace_log = convert_node(n, named_modules, node_to_layer, torch_mapping_needle,
                                                depth + 1, parent=node.name, trace_log=trace_log)
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
    print("======== Torch Mapping Needle ========")
    print(torch_mapping_needle)
    print(len(torch_mapping_needle))

