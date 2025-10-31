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
    named_modules = dict(traced.named_modules())
    node_to_layer = {}

    # this is used for weight converter
    torch_mapping_needle = {}

    output_node = list(traced.graph.nodes)[-1]
    needle_model, trace_log = convert_node(output_node, named_modules, node_to_layer, torch_mapping_needle)

    return needle_model, trace_log, torch_mapping_needle

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

        # 递归构建输入的 Needle 模块（如果不是 placeholder）
        submodules = []
        for arg in node.all_input_nodes:
            if arg.op != "placeholder":  # 非输入层
                sub_module, trace_log = convert_node(arg, named_modules, node_to_layer, torch_mapping_needle,
                                                     depth + 1, parent=node.name, trace_log=trace_log)
                submodules.append(sub_module)

        # 转换当前层
        module = convert_layer(torch_layer)
        entry["needle_type"] = type(module).__name__

        torch_mapping_needle[torch_layer] = module

        # ✅ 如果有输入子模块，则把它们组合成 Sequential
        if submodules:
            full = Sequential(*submodules, module)
            module = full
            entry["needle_type"] = f"Sequential({len(submodules)}+1)"


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
            modules = []
            for n in real_output:
                sub_module, trace_log = convert_node(n, named_modules, node_to_layer, torch_mapping_needle,
                                                     depth + 1, parent=node.name, trace_log=trace_log)
                modules.append(sub_module)
            module = Sequential(*modules)
            entry["needle_type"] = "Sequential"
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

