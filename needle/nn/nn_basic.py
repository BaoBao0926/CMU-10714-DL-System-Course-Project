"""The module.
"""
from math import sqrt
from typing import Any
from needle.autograd import Tensor
from needle.ops.ops_mathematic import *
from needle.ops.ops_logarithmic import *
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self, indent=0) -> str:
        """Pretty print for nested Needle modules (clean PyTorch-style)."""
        pad = "  " * indent
        child_lines = []

        for name, value in self.__dict__.items():
            # 跳过私有属性和参数
            if name.startswith('_') or isinstance(value, Parameter):
                continue
            # 递归打印子模块
            if isinstance(value, Module):
                sub_repr = value.__repr__(indent + 1)
                child_lines.append(f"{pad}  ({name}): {sub_repr}")
            # 递归打印模块列表或元组
            elif isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(x, Module) for x in value):
                for i, x in enumerate(value):
                    sub_repr = x.__repr__(indent + 1)
                    child_lines.append(f"{pad}  ({name}.{i}): {sub_repr}")

        # === Base Case ===
        if not child_lines:
            # 构建参数信息用于显示
            params = []
            if hasattr(self, 'in_features') and hasattr(self, 'out_features'):
                params.append(f"in_features={self.in_features}, out_features={self.out_features}")
                if hasattr(self, 'bias'):
                    params.append(f"bias={self.bias is not None}")
            elif hasattr(self, 'dim'):
                params.append(f"dim={self.dim}")
                if hasattr(self, 'eps'):
                    params.append(f"eps={self.eps}")
            elif hasattr(self, 'p') and hasattr(self, '__class__') and 'Dropout' in self.__class__.__name__:
                params.append(f"p={self.p}")
            
            if params:
                return f"{self.__class__.__name__}({', '.join(params)})"
            else:
                return f"{self.__class__.__name__}()"

        # === Recursive Case ===
        inner = "\n".join(child_lines)
        return f"{self.__class__.__name__}(\n{inner}\n{pad})"


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        W = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(W)
        if bias:
            bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(bias.reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
      if self.bias != None:
        out = matmul(X, self.weight)
        return out + broadcast_to(self.bias, out.shape)
      else:
        return matmul(X, self.weight)
    
    def __repr__(self, indent=0) -> str:
        pad = " " * indent
        return f"{pad}Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
      shape = list(X.shape)
      t_shape = 1
      for i in shape[1:]:
        t_shape *= i
      return reshape(X, (shape[0], t_shape))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
          x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
      y_one_hot = init.one_hot(logits.shape[1], y)  # shape (batch, num_classes)
      z_y = summation(logits * y_one_hot, axes=1)  # shape (batch,)
      LSE = logsumexp(logits, axes=1)
      loss = LSE - z_y
      return summation(loss) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
      super().__init__()
      self.dim = dim
      self.eps = eps
      self.momentum = momentum
      
      self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))  # (C)
      self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))   # (C)

      self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype, requires_grad=False)  # (C)
      self.running_var  = Tensor(init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype, requires_grad=False)  # (C)

    def forward(self, x: Tensor) -> Tensor:
      N, C = x.shape
      if self.training:
          # get mean/var in this batch
          mean = summation(x, axes=0, keepdims=True) / N
          mean_broadcast = broadcast_to(mean, x.shape)
          var  = summation((x - mean_broadcast) ** 2, axes=0, keepdims=True) / N
          std_broadcast = broadcast_to((var+self.eps)**0.5, x.shape)

          # update runing_mean/var
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * reshape(mean, self.dim).detach()
          self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * reshape(var, self.dim).detach()
          x_hat = (x - mean_broadcast) / std_broadcast
      else: # test
          test_mean = broadcast_to(reshape(self.running_mean, (1, self.dim)), x.shape)
          test_std = broadcast_to(reshape((self.running_var + self.eps)** 0.5, (1, self.dim)), x.shape)
          x_hat = (x - test_mean) / test_std

      # out = broadcast_to(self.weight, x_hat.shape) * x_hat + broadcast_to(self.bias, x_hat.shape)
      weight_broadcast = broadcast_to(reshape(self.weight, (1, self.dim)), x_hat.shape)  # (N, C)
      bias_broadcast   = broadcast_to(reshape(self.bias, (1, self.dim)), x_hat.shape)    # (N, C)
      out = weight_broadcast * x_hat + bias_broadcast
      return out


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        """
        dim - number of channels
        eps - a value added to the denominator for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        W = init.ones(1, dim, device=device, dtype=dtype)  # [1, C]
        self.weight = Parameter(W)
        bias = init.zeros(1, dim, device=device, dtype=dtype) # [1, C]
        self.bias = Parameter(bias.reshape((1, dim)))

    def forward(self, x: Tensor) -> Tensor:
      #  X [N, C]
      N, C = x.shape
      mean = summation(x, axes=1, keepdims=True) / self.dim
      var = summation((x - broadcast_to(mean, x.shape)) ** 2, axes=1, keepdims=True) / self.dim
      
      std = (var + self.eps) ** 0.5
      mean_broadcast = broadcast_to(mean, x.shape)
      std_broadcast = broadcast_to(std, x.shape)
      normalized_x = (x - mean_broadcast) / std_broadcast

      
      result = broadcast_to(self.weight, x.shape) * normalized_x + broadcast_to(self.bias, x.shape)

      return result


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
      if self.training:
        mask = (init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype))
        return (x * mask) / (1 - self.p)
      else:
        return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
      return x + self.fn(x)


class ADD(Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return self.left(x) + self.right(x)

class SUB(Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x: Tensor) -> Tensor:
        return self.left(x) - self.right(x)