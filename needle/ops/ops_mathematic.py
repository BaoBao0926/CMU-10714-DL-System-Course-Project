"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import math

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        #raise NotImplementedError()
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_inputs0 = node.inputs[0].cached_data
        node_inputs1 = node.inputs[1].cached_data
        ln_x = array_api.log(node_inputs0)
        pow_x_1 = node_inputs0 ** (node_inputs1 - 1)
        pow_x = node_inputs0 ** node_inputs1
        # pow_x_1 = array_api.pow(node_inputs0,node_inputs1 - 1)
        # pow_x = array_api.pow(node_inputs0,node_inputs1)
        return out_grad * (node_inputs1 * pow_x_1), out_grad * (pow_x * ln_x)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #return out_grad * (self.scalar * array_api.pow(node.inputs[0],self.scalar-1))
        return out_grad * (self.scalar *node.inputs[0] ** (self.scalar-1))
        #raise NotImplementedError()
        ### END YOUR SOLUTION



def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[1], out_grad * (-node.inputs[0].cached_data / (node.inputs[1].cached_data ** 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # transpose should be implemented by recalculating new strides
        ndim = len(a.shape)
        if self.axes is None: # reverse all axes
            axes = list(range(ndim))
            axes = axes[:-2] + [axes[-1]] + [axes[-2]]
        elif len(self.axes)==2: # reverse given two axes
            axes = list(range(ndim))
            swap_axis = axes[self.axes[0]]
            axes[self.axes[0]] = axes[self.axes[1]]
            axes[self.axes[1]] = swap_axis
        elif len(self.axes)==ndim: # permute axes
            axes = self.axes
        # if you want the axes length not limited to 2
        # ndim = len(a.shape)
        # if self.axes is None:
        #     axes = list(reversed(range(ndim)))
        else:
            # list specified axes on the front, then the remaining axes, this feature may not likely to use here 
            axes = list(range(ndim))
            specified_axes = list(self.axes)
            remaining_axes = [i for i in range(ndim) if i not in specified_axes]
            axes = specified_axes+remaining_axes
        
        new_shape = tuple([a.shape[i] for i in axes])
        new_strides = tuple([a.strides[i] for i in axes])
        return a.make(shape=new_shape,strides=new_strides,
                      device=a.device,handle=a._handle,offset=a._offset)
        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return transpose(out_grad,axes=None)
        elif len(self.axes) == len(node.inputs[0].shape):
            original_axes = self.axes
            inverse_axes = [original_axes.index(i) for i in range(len(original_axes))] # rearrange to original axis arrangement
            return transpose(out_grad, axes=tuple(inverse_axes))
        else:
            return transpose(out_grad, axes=self.axes)
        #return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
    
        new_shape = list(self.shape)
        # support -1 in shape and deduce dimemsion of -1 means
        if -1 in new_shape:
            total_elements = array_api.prod(a.shape)
            idx = new_shape.index(-1) # find the location of -1
            known_size = 1
            for i,dim in enumerate(new_shape):
                if i!=idx:
                    assert dim>0,"must have only one negative dimension"
                    known_size *=dim
            if total_elements % known_size !=0:
                raise ValueError("cannot inference dimemsion of -1")
            new_shape[idx] = total_elements // known_size
                           
        return a.compact().reshape(tuple(new_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        input_shape = a.shape
        out_shape = out_grad.shape
        input_shape_ = (1,) * (len(out_shape) - len(input_shape)) + input_shape
        axes = tuple(i for i, (a, b) in enumerate(zip(input_shape_, out_shape)) if a == 1 and b > 1)
        grad = out_grad
        if axes:
            for ax in sorted(axes,reverse=True):
                grad = grad.sum(axes=(ax,))
        return grad.reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        else:
            # normalize negative axes:
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            axes = tuple(ax if ax>=0 else ax+len(a.shape) for ax in axes)
            return a.sum(axis = axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        axes = axes or tuple(range(len(ori_shape)))
        # Normalize negative axes
        axes = tuple(ax if ax >= 0 else ax + len(ori_shape) for ax in axes)
        shape = [size if i not in axes else 1 for i, size in enumerate(ori_shape)]
        grad = out_grad.reshape(shape)
        return grad.broadcast_to(ori_shape)
    
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        if grad_a.shape != a.shape:
            axes = tuple(i for i, (ga, aa) in enumerate(zip(grad_a.shape, a.shape)) if ga != aa)
            for ax in sorted(axes,reverse=True):
                grad_a = grad_a.sum(ax)
        if grad_b.shape != b.shape:
            axes = tuple(i for i, (gb, bb) in enumerate(zip(grad_b.shape, b.shape)) if gb != bb)
            for ax in sorted(axes,reverse=True):
                grad_b = grad_b.sum(ax)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1) * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0] ** -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #relu_grad = array_api.full(shape=node.inputs[0].shape,fill_value=0)
        relu_grad = node.inputs[0].cached_data > 0
        #relu_grad[node.inputs[0].cached_data > 0] = 1
        return out_grad * Tensor(relu_grad,device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-tanh(node.inputs[0])**2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tupleLen = len(args)
        new_shape = args[0].shape[:self.axis] + (tupleLen,) +args[0].shape[self.axis:]
        
        new_array = array_api.empty(shape=new_shape,device=args[0].device) # should be the same device as args[0]!
        for i in range(tupleLen):
            new_slice = [slice(None)] * len(new_shape)
            new_slice[self.axis] = i
            new_array[tuple(new_slice)] = args[i]
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_tuple = split(out_grad,self.axis)
        return out_grad_tuple
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        tupleLen = A.shape[self.axis]
        ndarray_tuple = []
        new_shape = A.shape[:self.axis] + A.shape[self.axis+1:]
        for i in range(tupleLen):
            new_slice = [slice(None)] * len(A.shape)
            new_slice[self.axis] = i
            new_array = A[tuple(new_slice)].compact().reshape(new_shape)
            ndarray_tuple.append(new_array)
        return tuple(ndarray_tuple)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not self.dilation:
            return a
        max_dim = max(self.axes)
        # ensure there is no index out of bound error
        if(a.ndim<=max_dim):
            return a
            # new_shape = list(a.shape) + [1] * (max_dim - a.ndim + 1)
            # a = a.reshape(new_shape)
        #new_shape = []
        a_shape = list(a.shape)
        for ax in self.axes:
            a_shape[ax] = a_shape[ax] * (self.dilation+1)
        # for i in range(a.ndim):
        #     if(i in self.axes):
        #         new_shape.append(a.shape[i] * (self.dilation+1))
        #     else:
        #         new_shape.append(a.shape[i])
        a_dilate = array_api.full(shape=a_shape,fill_value=0,device=a.device)
        slices = [slice(None)] * len(a_shape)
        for axis in self.axes:
            slices[axis] = slice(None,None,self.dilation+1)
        a_dilate[tuple(slices)] = a
        return a_dilate
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not self.dilation:
            return a
        a_shape = a.shape
        #print(a.shape)
        # new_shape = []
        # for i in range(a.ndim):
        #     if(i in self.axes):
        #         if (a_shape[i] == self.dilation+1):
        #             continue
        #         else:
        #             new_shape.append(a_shape[i] // (self.dilation+1))
        #     else:
        #         new_shape.append(a_shape[i])
        slices = [slice(None)] * len(a_shape)
        for axis in self.axes:
            if(axis >= a.ndim):
                continue
            slices[axis] = slice(0,a.shape[axis],self.dilation+1)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N,H,W,Cin = A.shape
        K,_,_,Cout = B.shape
        P = self.padding
        S = self.stride
        if P:
            A = A.pad(axes=((0,0),(P,P),(P,P),(0,0)))
            N, H_pad, W_pad, Cin = A.shape
        else:
            H_pad,W_pad = H,W
        Ns,Hs,Ws,Cs = A.strides
        # H_out = (H_pad - K) // S + 1
        # W_out = (W_pad - K) // S + 1
        H_out = (H_pad - K+1) // S 
        W_out = (W_pad - K+1) // S 
        inner_dim = K * K *Cin
        outer_dim = N * H_out * W_out
        new_shape = (N, H_out,W_out, K, K, Cin)
        new_strides = (Ns,Hs*S,Ws*S,Hs,Ws,Cs)
        A = A.make(shape=new_shape,strides=new_strides,handle=A._handle, device=A.device, offset=A._offset).compact()
        # A.shape -> (N * ((H-K)//S+1) * ((W-K)//S+1) , (K*K*C))
        A = A.reshape((outer_dim,inner_dim))
        # B.shape -> (K*K*C)*Cout
        B = B.reshape((inner_dim,Cout))
        result =  A @ B
        return result.reshape((N,H_out,W_out,Cout))
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

class MaxPool2d(TensorOp):
    def __init__(self,kernel_size:Optional[int] = 3,stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def compute(self, x):
        batch_size, channels, in_height, in_width = x.shape
        pad = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        # add padding
        if self.padding > 0:
            x = x.pad(((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_value=float('-inf'))
        else:
            x = x
        # compute output height and width, then create new view of ndarray
        # The same logic as Conv op
        out_height = (in_height + 2 * pad - kernel_size) // stride + 1
        out_width = (in_width + 2 * pad - kernel_size) // stride + 1
        shape_6d = (batch_size, channels, out_height, out_width, kernel_size, kernel_size)
        strides_6d = (
            x.strides[0],  # batch
            x.strides[1],  # channel  
            stride * x.strides[2],  # height
            stride * x.strides[3],  # width
            x.strides[2],  # kernel height
            x.strides[3]   # kernel width
        )
        
        windows_6d = x.as_strided( 
            shape=shape_6d, 
            strides=strides_6d
        )
        
        windows_5d = windows_6d.compact().reshape((batch_size, channels, out_height, out_width, kernel_size * kernel_size))
        # find the maximum value of the last axis
        output = windows_5d.max(axis=len(windows_5d.shape) - 1)
        
        return output
    def gradient(self, out_grad, node):
        ## if needed, implement backward of maxpool
        raise NotImplementedError()
    
def max_pool2d(x,kernel_size=3,stride=1,padding=0):
    return MaxPool2d(kernel_size,stride,padding)(x)

class AdaptiveAvgPool2d(TensorOp):
    def __init__(self,output_size:Optional[Tuple]=(1,1)):
        self.output_size = output_size
    def compute(self, x):
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = self.output_size
        assert (in_height % out_height == 0) and (in_width % out_width == 0),"cannot apply since input size cannot be divided evenly by output size"

        # 计算池化核大小
        kernel_h = in_height // out_height
        kernel_w = in_width // out_width
        
        # 重塑张量并计算平均值
        new_shape = (batch_size,channels,out_height,kernel_h,out_width,kernel_w)
        new_strides = (
            x.strides[0],
            x.strides[1],
            kernel_h * x.strides[2],
            x.strides[2],
            kernel_w * x.strides[3],
            x.strides[3]
        )

        window_6d = x.as_strided(new_shape,new_strides)
        window_6d = window_6d.compact().permute((0,1,2,4,3,5))
        window_5d = window_6d.compact().reshape((batch_size, channels, out_height, out_width, kernel_h*kernel_w))
        output = array_api.mean(window_5d, axis=len(window_5d.shape)-1)
        return output
    def gradient(self, out_grad, node):
        ## if needed, implement backward of AdaptiveAveragepool2d
        raise NotImplementedError()

def adaptive_avg_pool2d(x,output_size=(1,1)):
    return AdaptiveAvgPool2d(output_size=output_size)(x)

class GELU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return 0.5 * a * (1 + array_api.tanh(array_api.sqrt(2 / math.pi) * (a + 0.044715 * (a ** 3))))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        tanh_out = array_api.tanh(array_api.sqrt(2 / math.pi) * (x.cached_data + 0.044715 * (x.cached_data ** 3)))
        sech2 = 1 - tanh_out ** 2
        coeff = array_api.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * (x.cached_data ** 2))
        gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x.cached_data * sech2 * coeff
        return out_grad * Tensor(gelu_grad,device=out_grad.device)
        ### END YOUR SOLUTION

def gelu(a):
    return GELU()(a)
