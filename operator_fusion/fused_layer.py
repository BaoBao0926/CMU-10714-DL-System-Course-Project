import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from needle.autograd import Tensor
from needle.ops.ops_mathematic import relu, matmul, broadcast_to, summation, reshape
from needle.ops.ops_fused import (
    fused_linear_relu,
    fused_batchnorm_relu,
    fused_linear_batchnorm,
    fused_linear_batchnorm_relu,
)
import needle.init as init
from needle.nn.nn_basic import (
    Module, Linear, ReLU, Sequential, BatchNorm1d, 
    LayerNorm1d, Dropout, Identity, Flatten, Parameter
)
from typing import Any, List, Tuple, Optional


# ============================================================================
# Fused Operator Definitions
# ============================================================================



class LinearReLU(Module):
    """
    fuse Linear + ReLU layer
    combine linear layer and Relu into a single layer
    
    Uses the fused_linear_relu operation which combines matmul + bias + relu
    into a single operation for better memory efficiency.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        W = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(W)
        if bias:
            bias_tensor = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(bias_tensor.reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        """Perform fused Linear + ReLU operation using fused op"""
        return fused_linear_relu(X, self.weight, self.bias)


class LinearBatchNorm(Module):
    """
    Fuse Linear + BatchNorm1d layer
    Can reduce memory access and improve cache utilization during training
    
    Uses the fused_linear_batchnorm operation for better performance.
    """
    def __init__(self, in_features: int, out_features: int, 
                 eps: float = 1e-5, momentum: float = 0.1,
                 bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Linear layer parameters
        W = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(W)
        if bias:
            bias_tensor = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.linear_bias = Parameter(bias_tensor.reshape((1, out_features)))
        else:
            self.linear_bias = None
        
        # BatchNorm layer parameters
        self.bn_weight = Parameter(init.ones(out_features, device=device, dtype=dtype))
        self.bn_bias = Parameter(init.zeros(out_features, device=device, dtype=dtype))
        
        # Running statistics (not trainable)
        self.running_mean = Tensor(init.zeros(out_features, device=device, dtype=dtype), 
                                   device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(init.ones(out_features, device=device, dtype=dtype), 
                                  device=device, dtype=dtype, requires_grad=False)

    def forward(self, X: Tensor) -> Tensor:
        """Perform fused Linear + BatchNorm operation using fused op"""
        # Use fused operation for forward pass
        out = fused_linear_batchnorm(
            X, self.weight, self.linear_bias,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            self.eps, self.momentum, self.training
        )
        
        # Update running statistics during training
        # Note: The fused op computes the stats, but we need to update running stats here
        if self.training:
            # Recompute mean and var for running stats update
            # (In a fully optimized version, the fused op would return these)
            linear_out = matmul(X, self.weight)
            if self.linear_bias is not None:
                linear_out = linear_out + broadcast_to(self.linear_bias, linear_out.shape)
            
            N = linear_out.shape[0]
            mean = summation(linear_out, axes=0, keepdims=True) / N
            mean_broadcast = broadcast_to(mean, linear_out.shape)
            var = summation((linear_out - mean_broadcast) ** 2, axes=0, keepdims=True) / N
            
            # Update running stats
            self.running_mean = ((1 - self.momentum) * self.running_mean + 
                               self.momentum * reshape(mean, self.out_features).detach())
            self.running_var = ((1 - self.momentum) * self.running_var + 
                              self.momentum * reshape(var, self.out_features).detach())
        
        return out


class BatchNormReLU(Module):
    """
    Fuse BatchNorm1d + ReLU layer
    Commonly used in architectures like ResNet
    
    Uses the fused_batchnorm_relu operation for better performance.
    """
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, 
                 device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        # BatchNorm parameters
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        
        # Running statistics
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype), 
                                   device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype), 
                                  device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Perform fused BatchNorm + ReLU operation using fused op"""
        # Use fused operation
        out = fused_batchnorm_relu(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.eps, self.momentum, self.training
        )
        
        # Update running statistics during training
        if self.training:
            N = x.shape[0]
            mean = summation(x, axes=0, keepdims=True) / N
            mean_broadcast = broadcast_to(mean, x.shape)
            var = summation((x - mean_broadcast) ** 2, axes=0, keepdims=True) / N
            
            # Update running stats
            self.running_mean = ((1 - self.momentum) * self.running_mean + 
                               self.momentum * reshape(mean, self.dim).detach())
            self.running_var = ((1 - self.momentum) * self.running_var + 
                              self.momentum * reshape(var, self.dim).detach())
        
        return out


class LinearBatchNormReLU(Module):
    """
    Fuse Linear + BatchNorm1d + ReLU layer
    Three-layer fusion to further reduce intermediate variables and memory access
    
    Uses the fused_linear_batchnorm_relu operation for maximum performance.
    """
    def __init__(self, in_features: int, out_features: int, 
                 eps: float = 1e-5, momentum: float = 0.1,
                 bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Linear layer parameters
        W = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(W)
        if bias:
            bias_tensor = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.linear_bias = Parameter(bias_tensor.reshape((1, out_features)))
        else:
            self.linear_bias = None
        
        # BatchNorm layer parameters
        self.bn_weight = Parameter(init.ones(out_features, device=device, dtype=dtype))
        self.bn_bias = Parameter(init.zeros(out_features, device=device, dtype=dtype))
        
        # Running statistics
        self.running_mean = Tensor(init.zeros(out_features, device=device, dtype=dtype), 
                                   device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(init.ones(out_features, device=device, dtype=dtype), 
                                  device=device, dtype=dtype, requires_grad=False)

    def forward(self, X: Tensor) -> Tensor:
        """Perform fused Linear + BatchNorm + ReLU operation using fused op"""
        # Use fused operation for forward pass
        out = fused_linear_batchnorm_relu(
            X, self.weight, self.linear_bias,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            self.eps, self.momentum, self.training
        )
        
        # Update running statistics during training
        if self.training:
            # Recompute mean and var for running stats update
            linear_out = matmul(X, self.weight)
            if self.linear_bias is not None:
                linear_out = linear_out + broadcast_to(self.linear_bias, linear_out.shape)
            
            N = linear_out.shape[0]
            mean = summation(linear_out, axes=0, keepdims=True) / N
            mean_broadcast = broadcast_to(mean, linear_out.shape)
            var = summation((linear_out - mean_broadcast) ** 2, axes=0, keepdims=True) / N
            
            # Update running stats
            self.running_mean = ((1 - self.momentum) * self.running_mean + 
                               self.momentum * reshape(mean, self.out_features).detach())
            self.running_var = ((1 - self.momentum) * self.running_var + 
                              self.momentum * reshape(var, self.out_features).detach())
        
        return out
