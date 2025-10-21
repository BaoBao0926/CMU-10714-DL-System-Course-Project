from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):

    def compute(self, Z: NDArray) -> NDArray:
      max_z = array_api.max(Z, axis=1, keepdims=True)
      result = array_api.log(array_api.sum(array_api.exp(Z-max_z), axis=1, keepdims=True)) + max_z
      return Z-result

    def gradient(self, out_grad: Tensor, node: Tensor):
      Z = node.inputs[0]
      LSE = logsumexp(Z, axes=1)  # shape [N,]
      shape = list(Z.shape)
      shape[1] = 1
      LSE_ = reshape(LSE, shape)  # shaoe [N, 1]
      softmax_Z = exp(Z - LSE_)  # shape: [N, C]
      # calculate sum of out_grad
      sum_out_grad = summation(out_grad, axes=1, keepdims=True)  # shape [N,1]
      # gradient = out_grad - softmax * sum(out_grad)
      return out_grad - softmax_Z * sum_out_grad


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # we need delete additional dimension by squeeze()
        result = array_api.log(array_api.sum(array_api.exp(Z-max_z), axis=self.axes)) + array_api.squeeze(max_z, axis=self.axes)
        return result


    def gradient(self, out_grad: Tensor, node: Tensor):
      """
      Because LogSumExp() can delete some dimension. So, the first step is the restore the original dimension
      """
      
      Z = node.inputs[0]  # input
      y = node       

      # frist find the reduced dimension
      z_shape = Z.shape
      n = len(z_shape)
      if self.axes is None:
          reduced_axes = tuple(range(n))
      elif isinstance(self.axes, int):
          reduced_axes = (self.axes,)
      else:
          reduced_axes = tuple(self.axes)
      reduced_axes = tuple(ax % n for ax in reduced_axes) # transform negative number into positive number
      # recover to original shape
      grad_shape = tuple(1 if i in reduced_axes else z_shape[i] for i in range(n))

      reshaped_out_grad = reshape(out_grad, grad_shape)     # reshape out_grad into grad_shape 
      out_grad_b = broadcast_to(reshaped_out_grad, Z.shape)   
      y_b = broadcast_to(reshape(y, grad_shape), Z.shape)   
      softmax_Z = exp(Z - y_b)                  # stable softmax
      return out_grad_b * softmax_Z


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)