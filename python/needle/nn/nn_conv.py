"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
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

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((0,2,3,1)) # N,C,H,W -> N,H,W,C
        x = ops.conv(x,self.weight,stride=self.stride,padding=self.kernel_size//2)
        if self.bias is not None:
            #bias_shape = tuple(1 if (dim!=self.out_channels) else self.out_channels for dim in res.shape)
            bias = self.bias.broadcast_to(x.shape)
            x += bias
        return x.transpose((0,3,1,2))
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
    def forward(self,x:Tensor)->Tensor:
        return ops.max_pool2d(x,self.kernel_size,self.stride,self.pad)

class AdaptiveAvgPool2d(Module):
    """
    Multi-channel AdaptiveAvgPool 2d layer
    """
    def __init__(self,output_size=(1,1)):
        super().__init__()
        self.output_size = output_size
    def forward(self,x:Tensor)->Tensor:
        return ops.adaptive_avg_pool2d(x,self.output_size)