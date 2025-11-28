"""AMD Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..init import zeros
from ..backend_ndarray import ndarray_backend_hip as hip_backend
from ..backend_ndarray import hip

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *
from .ops_mathematic import dilate, undilate, flip


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A: NDArray, B: NDArray):
        ### BEGIN YOUR SOLUTION
        ## HIP backend convolution use NCHW format##
        N,Cin,H,W = A.shape
        Cout,_,K,_ = B.shape
        P = self.padding
        S = self.stride
        H_out = (H +2*P - K) // S +1
        W_out = (W +2*P - K) // S +1
        out = NDArray.make((N,Cout,H_out,W_out),device=A.device)
        hip_backend.conv(A.compact()._handle,B.compact()._handle,out._handle,N,Cin,H,W,Cout,K,K,S,P)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        weight = node.inputs[1]
        X = node.inputs[0]
        S = self.stride
        P = self.padding
        ##we may need dilate for strided conv##
        if S>1:
            out_grad = dilate(out_grad,axes=(1,2),dilation=S-1)
        # else:
        #     out_grad_dilated = out_grad
        #######################################
        # X.grad
        W_flipped = flip(weight,axes=(0,1))
        #W_flipped = W_flipped.transpose((2,3)) # cannot use transpose, use permute instead! 
        W_flipped = Tensor(W_flipped.cached_data.permute((0,1,3,2)),device=W_flipped.device)
        X_grad = conv(out_grad,W_flipped,stride=1,padding=weight.shape[0]-1-P)
        # W.grad
        # pad X to compute W_grad, X_padded is a NDArray
        if P:
            X_padded = X.cached_data.pad(axes=((0,0),(P,P),(P,P),(0,0)))
        else:
            X_padded = X.cached_data
        X_permuted = Tensor(X_padded.permute((3,1,2,0)),device=X_padded.device).detach()
        out_grad_permuted = Tensor(out_grad.cached_data.permute((1,2,0,3)),device=out_grad.device).detach()
        # dilate outgrad if S>1
        W_grad = conv(X_permuted,out_grad_permuted,stride=1,padding=0)
        W_grad = Tensor(W_grad.cached_data.permute((1,2,0,3)),device=W_grad.device)
        # if stride>1, the conv op produce a "dilated" kernel, we should undo the dilated op
        # if S>1:
        #     W_grad = undilate(W_grad,axes=(0,1),dilation=S-1)
        return X_grad,W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

class BatchNorm2d(TensorOp):
    def __init__(self, channel:int, weight:Optional[Tensor]=None, bias:Optional[Tensor]=None,
                 running_mean:Optional[Tensor]=None, running_var:Optional[Tensor]=None,
                 eps: float = 1e-5, momentum: float = 0.1):
        self.dim = channel
        self.eps = eps
        self.momentum = momentum
        self.weight = weight.cached_data 
        self.bias = bias.cached_data
        self.running_mean = running_mean.cached_data
        self.running_var = running_var.cached_data
    def compute(self, x: NDArray) -> NDArray:
        batch,_,h,w = x.shape
        out = NDArray.make((batch,self.dim,h,w),device=x.device)
        hip_backend.batchnorm2d(x.compact()._handle,
                                out._handle,
                                self.weight.compact()._handle,
                                self.bias.compact()._handle,
                                self.running_mean.compact()._handle,
                                self.running_var.compact()._handle,
                                batch,self.dim,h,w,
                                self.eps)
        return out
    def gradient(self, out_grad: Tensor, node: Tensor) -> List[Tensor]:
        ### implement BatchNorm2d gradient for AMD backend if needed ###
        raise NotImplementedError()

def batchnorm2d(x: Tensor, channel:int, weight:Optional[Tensor]=None, bias:Optional[Tensor]=None,
                running_mean:Optional[Tensor]=None, running_var:Optional[Tensor]=None,
                eps: float = 1e-5, momentum: float = 0.1) -> Tensor:
    return BatchNorm2d(channel, weight, bias, running_mean, running_var, eps, momentum)(x)


class ConvBatchnorm2dRelu(TensorOp):
    def __init__(self, out_channel:int,stride: Optional[int] = 1, padding: Optional[int] = 0,
                bias:Optional[Tensor]=None,
                scale:Optional[Tensor]=None, shift:Optional[Tensor]=None,
                running_mean:Optional[Tensor]=None, running_var:Optional[Tensor]=None,
                eps: float = 1e-5, momentum: float = 0.1):
        self.stride = stride
        self.padding = padding
        self.out_channel = out_channel
        self.eps = eps
        self.momentum = momentum
        self.bias = bias.cached_data if bias is not None else zeros(out_channel,device=hip()).cached_data
        self.scale = scale.cached_data
        self.shift = shift.cached_data
        self.running_mean = running_mean.cached_data
        self.running_var = running_var.cached_data
    def compute(self, A: NDArray, B: NDArray) -> NDArray:
        N,Cin,H,W = A.shape
        Cout,_,K,_ = B.shape
        P = self.padding
        S = self.stride
        H_out = (H +2*P - K) // S +1
        W_out = (W +2*P - K) // S +1
        out = NDArray.make((N,Cout,H_out,W_out),device=A.device)
        hip_backend.convbn2drelu(A.compact()._handle,out._handle,
                                 B.compact()._handle,self.bias.compact()._handle,
                                 self.scale.compact()._handle,self.shift.compact()._handle,
                                 self.running_mean.compact()._handle,
                                 self.running_var.compact()._handle,
                                 N,Cin,H,W,Cout,K,K,S,P,self.eps)
        return out
    def gradient(self, out_grad: Tensor, node: Tensor) -> List[Tensor]:
        ### implement ConvBatchnorm2dRelu gradient for AMD backend if needed ###
        raise NotImplementedError()

def conv_batchnorm2d_relu(X: Tensor, W: Tensor, out_channel:int, stride: Optional[int] = 1, padding: Optional[int] = 0,
                          bias:Optional[Tensor]=None,
                          scale:Optional[Tensor]=None, shift:Optional[Tensor]=None,
                          running_mean:Optional[Tensor]=None, running_var:Optional[Tensor]=None,
                          eps: float = 1e-5, momentum: float = 0.1) -> Tensor:
    return ConvBatchnorm2dRelu(out_channel,stride, padding, bias, scale, shift,
                               running_mean, running_var, eps, momentum)(X, W)