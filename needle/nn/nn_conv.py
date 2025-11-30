"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from needle.needle_profiling import profile_operation
from ..backend_ndarray import hip


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding=None, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = kernel_size //2 if padding is None else padding
        shape = (kernel_size,kernel_size,in_channels,out_channels)
        ### BEGIN YOUR SOLUTION
        weight = init.kaiming_uniform(fan_in=in_channels * (kernel_size**2),fan_out=out_channels * (kernel_size**2),
                                           shape=shape,device=device,dtype=dtype,requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = None
        if bias:
            bound = 1/np.sqrt(in_channels * kernel_size**2)
            bias_data = init.rand(out_channels,low=-bound,
                                high=bound,device=device,dtype=dtype,requires_grad=True)
            self.bias = Parameter(bias_data)
        ### END YOUR SOLUTION

    @profile_operation
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((0,2,3,1)) if x.device != hip() else x# N,C,H,W -> N,H,W,C
        x = ops.conv(x,self.weight,stride=self.stride,padding=self.kernel_size//2)
        if self.bias is not None:
            #bias_shape = tuple(1 if (dim!=self.out_channels) else self.out_channels for dim in res.shape)
            bias = self.bias.reshape((1,self.out_channels,1,1)).broadcast_to(x.shape) if x.device == hip() else self.bias.broadcast_to(x.shape)
            x += bias
        return x.transpose((0,3,1,2)) if x.device != hip() else x
        ### END YOUR SOLUTION

class MaxPool2d(Module):
    """
    Multi-channel Maxpool 2d layer
    """
    def __init__(self,kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
    @profile_operation
    def forward(self,x:Tensor)->Tensor:
        return ops.max_pool2d(x,self.kernel_size,self.stride,self.pad)

class AdaptiveAvgPool2d(Module):
    """
    Multi-channel AdaptiveAvgPool 2d layer
    """
    def __init__(self,output_size=(1,1)):
        super().__init__()
        self.output_size = output_size
    @profile_operation
    def forward(self,x:Tensor)->Tensor:
        return ops.adaptive_avg_pool2d(x,self.output_size)

class ConvTranspose2d(Module):
    """
    Simplified ConvTranspose2d
    Supports: NCHW input, square kernel, stride>=1, padding (same semantics as PyTorch: padding=P).
    No groups, no dilation (other than stride-induced), no output_padding.
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True,device=None,dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        ## store weight as (K,K,C_in,C_out) for conv operation ##
        shape = (kernel_size,kernel_size,in_channels,out_channels)
        weight = init.kaiming_uniform(fan_in=in_channels * (kernel_size**2),fan_out=out_channels * (kernel_size**2),
                                           shape=shape,device=device,dtype=dtype,requires_grad=True)
        self.weight = Parameter(weight)
        self.bias = None
        if bias:
            bound = 1/np.sqrt(in_channels * kernel_size**2)
            bias_data = init.rand(out_channels,low=-bound,
                                high=bound,device=device,dtype=dtype,requires_grad=True)
            self.bias = Parameter(bias_data)
    @profile_operation
    def forward(self,x:Tensor)->Tensor:
        # x: N,C_in,H,W
        if self.stride>1: # 为了行为和nn.ConvTranspose2d一样，dilate不应该扩张最后一行和最后一列
            x = ops.dilate(x,axes=(2,3),dilation=self.stride-1)
            Hp = (x.shape[2] - (self.stride -1))
            Wp = (x.shape[3] - (self.stride -1))
            x = ops.tensor_slice(x,[
                (0,x.shape[0],1),
                (0,x.shape[1],1),
                (0,Hp,1),
                (0,Wp,1)
            ])
        W = ops.flip(self.weight,axes=(0,1))
        x = x.transpose((0,2,3,1)) if x.device != hip() else x # N,H,W,C_in
        pad = self.kernel_size - self.pad - 1
        x = ops.conv(x,W,stride=1,padding=pad)
        if self.bias is not None:
            bias = self.bias.reshape((1,self.out_channels,1,1)).broadcast_to(x.shape) if x.device == hip() else self.bias.broadcast_to(x.shape)
            x += bias
        return x.transpose((0,3,1,2)) if x.device != hip() else x # N,C_out,H,W
