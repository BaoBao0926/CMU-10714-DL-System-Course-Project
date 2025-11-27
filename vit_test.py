import torch
import numpy as np
from pytorch_pretrained_vit import ViT
import needle as ndl 

# 不引入 needle 顶层模块，避免循环 import！
from needle.autograd import Tensor

# Torch2Needle
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping