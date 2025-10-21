import os
import sys

# 设置环境变量
os.environ["PYTHONPATH"] = "./python"
os.environ["NEEDLE_BACKEND"] = "nd"

# 可选：手动把 ./python 加入模块搜索路径
sys.path.insert(0, "./python")

# 检查是否生效
print("PYTHONPATH:", os.environ["PYTHONPATH"])
print("NEEDLE_BACKEND:", os.environ["NEEDLE_BACKEND"])



from needle.nn.nn_basic import Module, Tensor, Parameter, init, matmul, broadcast_to, reshape, relu
from needle.nn.nn_basic import *



class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.nn = Linear(5, 1)
        self.relu = ReLU()

    def forward(self, x):
        x = self.nn(x)
        x = self.relu(x)
        return x

model = SimpleNN()
input_data = Tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype="float32")
output = model(input_data)
print("Model output:", output)