"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
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


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,requires_grad=True,device=device,dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features,1,requires_grad=True,device=device,dtype=dtype).reshape((1,out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X.matmul(self.weight)
        if hasattr(self,'bias'):
            bias = self.bias
            # if len(output.shape)==2:
            #     bias = bias.reshape((1,-1))
            bias = bias.broadcast_to(output.shape) # add bias term should broadcast
            output += bias
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.gelu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[1],y,device=logits.device)
        log_sum_exp = ops.logsumexp(logits,axes=1)
        logits_y = (logits * one_hot).sum(axes=1)
        #print(logits_y.shape)
        return (log_sum_exp - logits_y).sum() / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        # running_mean and running_var in batchnorm are not parameters!!
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, *_ = x.shape
        if self.training:
            mean = x.sum(axes=0) / batch
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            #mean = mean.reshape((1,-1))
            var = ((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=0) / batch
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            #var = var.reshape((1,-1))
        else:
            mean = self.running_mean
            #mean = mean.reshape((1,-1))
            var = self.running_var
            #var = var.reshape((1,-1))
        x_hat = (x - mean.broadcast_to(x.shape)) / ((var.broadcast_to(x.shape) + self.eps) ** 0.5)
        # weight = self.weight.reshape((1,-1))
        # bias = self.bias.reshape((1,-1))
        weight = self.weight
        bias = self.bias
        return x_hat * weight.broadcast_to(x.shape) + bias.broadcast_to(x.shape)
        # if self.training:
        #   mean = x.sum(axes=0)/x.shape[0]
        #   self.running_mean = (1-self.momentum)*self.running_mean.data + (self.momentum*mean).data # update running mean should not records the inputs of running mean
        #   # expand mean's shape
        #   mean = mean.reshape((1,-1))
        #   mean = mean.broadcast_to(x.shape) # implicit broadcast will not calculate the gradient, should change to explicit broadcast!!!
        #   var = ops.power_scalar(x-mean,2).sum(axes=0)/x.shape[0]
        #   self.running_var = (1-self.momentum)*self.running_var.data + (self.momentum*var).data # update running var should not records the inputs of running var
        #   var = var.reshape((1,-1))
        #   var = var.broadcast_to(x.shape)
        # else:
        #   #print("I am here! HAHAHAHA")
        #   mean = self.running_mean.reshape((1,-1)).broadcast_to(x.shape) 
        #   var = self.running_var.reshape((1,-1)).broadcast_to(x.shape) 
        # std = ops.power_scalar(var+self.eps,0.5)
        # norm_term = (x-mean) / std
        # #norm_term = (x-mean) / ops.power_scalar(var+self.eps,0.5)
        # weight = self.weight.reshape((1,-1)).broadcast_to(x.shape)
        # bias = self.bias.reshape((1,-1)).broadcast_to(x.shape)
        # return weight * norm_term + bias
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        reduce_shape = x.shape[:-1] + (1,)
        mean = x.sum(axes=-1).reshape(reduce_shape).broadcast_to(x.shape) / self.dim
        var = (x-mean)**2
        var = var.sum(axes=-1).reshape(reduce_shape).broadcast_to(x.shape) / self.dim
        x = (x-mean) / ((var+self.eps)**0.5)
        return self.weight.broadcast_to(x.shape) * x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if(self.training):
          # mask = Tensor((np.random.random(x.shape) > self.p))
          mask = init.rand(*x.shape)
          mask = Tensor(mask.cached_data < (1-self.p),device=x.device,dtype=x.dtype)
          return x * mask * (1/(1-self.p))
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
        
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