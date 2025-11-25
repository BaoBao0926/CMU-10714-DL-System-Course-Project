from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=1,keepdims=True)
        exp_sum_z = array_api.sum(array_api.exp(Z - max_z),axis=1,keepdims=True)
        log_sum_exp = (array_api.log(exp_sum_z) + max_z)
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0].cached_data
        #print(z,z.shape)
        max_z = z.max(axis=1,keepdims=True)
        exp_sum_z = array_api.sum(array_api.exp(z - max_z),axis=1,keepdims=True)
        softmax = (array_api.exp(z-max_z) / exp_sum_z)
        #print(exp_sum_z.shape,out_grad.shape)
        if(len(out_grad.shape) < len(exp_sum_z.shape)): # should broadcast to a shape that can make right element-wise multiply with z
          out_grad = out_grad.reshape(exp_sum_z.shape)
        out_grad_sum = out_grad.sum(axes=1).reshape(exp_sum_z.shape) # summation of out_grad also should keep original dimension
        return out_grad - out_grad_sum* Tensor(softmax,device=node.device)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(self.axes,keepdims=True)
        max_z_bc = max_z.broadcast_to(Z.shape)
        exp_sum_z = array_api.sum(array_api.exp(Z - max_z_bc),axis=self.axes,keepdims=True)

        return_value =  (array_api.log(exp_sum_z) + max_z)
        # get rid of reduced axes
        final_shape = []
        for i,dim in enumerate(return_value.shape):
            if(i==self.axes):
                continue
            final_shape.append(dim)
        return array_api.reshape(return_value,final_shape)
        # if(return_value.shape == ()):
        #     return return_value.astype(array_api.float32)
        # else:
        #     return return_value
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        lgse = logsumexp(Z,axes=self.axes)
        if self.axes is not None:
            new_shape = list(Z.shape)
            axes = [self.axes] if isinstance(self.axes,int) else list(self.axes)
            for ax in axes:
                new_shape[ax] = 1 # the reduced axis is set to 1
            # both out_grad and lgse should be reshape and explicitly broadcast to input's shape
            lgse = lgse.reshape(new_shape).broadcast_to(Z.shape)
            out_grad = out_grad.reshape(new_shape).broadcast_to(Z.shape)
        sfx = exp(Z-lgse)
        return out_grad * sfx
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)