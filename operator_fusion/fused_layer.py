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
    fused_conv_batchnorm2d_relu,
    fused_multihead_attention,
)
import needle.init as init
from needle.nn.nn_basic import (
    Module, Linear, ReLU, Sequential, BatchNorm1d, BatchNorm2d,
    LayerNorm1d, Dropout, Identity, Flatten, Parameter
)
from needle.nn.nn_conv import Conv
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
        self.training = True

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


class ConvBatchNorm2dReLU(Module):
    """
    Fused Conv2d + BatchNorm2d + ReLU layer
    
    This is the most common building block in modern CNNs:
    - ResNet: Every conv block uses Conv-BN-ReLU
    - MobileNet: Depthwise separable convs use this pattern
    - EfficientNet: Most layers follow this structure
    
    Fusion benefits:
    1. Eliminates intermediate feature map storage (saves 2x memory)
    2. Reduces memory bandwidth by ~3x
    3. Enables single-kernel execution on GPU (2-3x speedup potential)
    4. Critical for efficient inference on mobile/edge devices
    
    Uses the fused_conv_batchnorm2d_relu operation for optimal performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = True,
                 eps: float = 1e-5, momentum: float = 0.1,
                 device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.momentum = momentum
        self.padding = kernel_size // 2  # same padding as Conv module
        
        # Conv layer parameters
        import numpy as np
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        weight = init.kaiming_uniform(
            fan_in=in_channels * (kernel_size**2),
            fan_out=out_channels * (kernel_size**2),
            shape=shape, device=device, dtype=dtype, requires_grad=True
        )
        self.weight = Parameter(weight)
        
        self.conv_bias = None
        if bias:
            bound = 1 / np.sqrt(in_channels * kernel_size**2)
            bias_data = init.rand(
                out_channels, low=-bound, high=bound,
                device=device, dtype=dtype, requires_grad=True
            )
            self.conv_bias = Parameter(bias_data)
        
        # BatchNorm2d parameters
        self.bn_weight = Parameter(init.ones(out_channels, device=device, dtype=dtype))
        self.bn_bias = Parameter(init.zeros(out_channels, device=device, dtype=dtype))
        
        # Running statistics (not trainable)
        self.running_mean = Tensor(init.zeros(out_channels, device=device, dtype=dtype),
                                   device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(init.ones(out_channels, device=device, dtype=dtype),
                                  device=device, dtype=dtype, requires_grad=False)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Perform fused Conv2d + BatchNorm2d + ReLU operation using fused op"""
        # Use fused operation for forward pass
        out = fused_conv_batchnorm2d_relu(
            x, self.weight, self.out_channels,
            self.conv_bias,self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            stride=self.stride, padding=self.padding,
            eps=self.eps, momentum=self.momentum, training=self.training
        )
        
        # Update running statistics during training
        if self.training:
            # Need to recompute stats for running mean/var update
            # Convert to NHWC for conv
            x_nhwc = x.transpose((0, 2, 3, 1))
            from needle.ops.ops_mathematic import conv as conv_op
            conv_out = conv_op(x_nhwc, self.weight, stride=self.stride, padding=self.padding)
            if self.conv_bias is not None:
                conv_out = conv_out + broadcast_to(self.conv_bias, conv_out.shape)
            
            # Convert to NCHW and reshape for BN stats
            conv_out_nchw = conv_out.transpose((0, 3, 1, 2))
            N, C, H, W = conv_out_nchw.shape
            conv_out_reshaped = conv_out_nchw.transpose((1, 2)).transpose((2, 3)).reshape((N * H * W, C))
            
            # Compute mean and variance
            batch_size = N * H * W
            mean = conv_out_reshaped.sum(axes=0) / batch_size
            mean_broadcast = broadcast_to(mean, conv_out_reshaped.shape)
            var = ((conv_out_reshaped - mean_broadcast) ** 2).sum(axes=0) / batch_size
            
            # Update running stats
            self.running_mean = ((1 - self.momentum) * self.running_mean +
                               self.momentum * mean.detach())
            self.running_var = ((1 - self.momentum) * self.running_var +
                              self.momentum * var.detach())
        
        return out


class FusedMultiHeadAttention(Module):
    """
    Fused Multi-Head Attention module inspired by FlashAttention.
    
    This module fuses the entire multi-head attention computation into a single
    optimized operation, combining:
    - Q, K, V projections (handled separately)
    - Scaled dot-product attention
    - Causal masking (if enabled)
    - Softmax normalization
    - Dropout
    - Output aggregation
    
    Benefits:
    - Reduced memory footprint (no intermediate attention matrix storage)
    - Faster computation through kernel fusion
    - Better numerical stability
    - Essential for long-sequence transformers
    
    This is commonly used in:
    - GPT-style autoregressive models (with causal=True)
    - BERT-style bidirectional models (with causal=False)
    - Vision Transformers
    - Any transformer-based architecture
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32"
    ) -> None:
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        self.dropout_prob = dropout
        self.causal = causal
        
        # QKV projection weights (combined for efficiency)
        # Standard practice: use separate projections for Q, K, V
        self.q_proj_weight = Parameter(
            init.kaiming_uniform(embed_dim, embed_dim, device=device, dtype=dtype)
        )
        self.k_proj_weight = Parameter(
            init.kaiming_uniform(embed_dim, embed_dim, device=device, dtype=dtype)
        )
        self.v_proj_weight = Parameter(
            init.kaiming_uniform(embed_dim, embed_dim, device=device, dtype=dtype)
        )
        
        if bias:
            self.q_proj_bias = Parameter(
                init.zeros(embed_dim, device=device, dtype=dtype).reshape((1, embed_dim))
            )
            self.k_proj_bias = Parameter(
                init.zeros(embed_dim, device=device, dtype=dtype).reshape((1, embed_dim))
            )
            self.v_proj_bias = Parameter(
                init.zeros(embed_dim, device=device, dtype=dtype).reshape((1, embed_dim))
            )
        else:
            self.q_proj_bias = None
            self.k_proj_bias = None
            self.v_proj_bias = None
        
        # Output projection
        self.out_proj_weight = Parameter(
            init.kaiming_uniform(embed_dim, embed_dim, device=device, dtype=dtype)
        )
        if bias:
            self.out_proj_bias = Parameter(
                init.zeros(embed_dim, device=device, dtype=dtype).reshape((1, embed_dim))
            )
        else:
            self.out_proj_bias = None
        self.training = True
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of fused multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len_k, embed_dim). If None, uses query (self-attention)
            value: Value tensor of shape (batch_size, seq_len_k, embed_dim). If None, uses query (self-attention)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Project Q, K, V
        q = query.reshape((batch_size * seq_len_q, self.embed_dim))
        k = key.reshape((batch_size * seq_len_k, self.embed_dim))
        v = value.reshape((batch_size * seq_len_k, self.embed_dim))
        
        q = matmul(q, self.q_proj_weight)
        k = matmul(k, self.k_proj_weight)
        v = matmul(v, self.v_proj_weight)
        
        if self.q_proj_bias is not None:
            q = q + broadcast_to(self.q_proj_bias, q.shape)
            k = k + broadcast_to(self.k_proj_bias, k.shape)
            v = v + broadcast_to(self.v_proj_bias, v.shape)
        
        # Reshape to (batch, seq_len, num_heads, dim_head) then transpose to (batch, num_heads, seq_len, dim_head)
        q = q.reshape((batch_size, seq_len_q, self.num_heads, self.dim_head)).transpose((1, 2))
        k = k.reshape((batch_size, seq_len_k, self.num_heads, self.dim_head)).transpose((1, 2))
        v = v.reshape((batch_size, seq_len_k, self.num_heads, self.dim_head)).transpose((1, 2))
        
        # Call fused multihead attention (implements full algorithm in ops_fused.py)
        attn_output = fused_multihead_attention(
            q, k, v,
            dim_head=self.dim_head,
            dropout_prob=self.dropout_prob,
            causal=self.causal,
            training=self.training
        )
        
        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose((1, 2)).reshape((batch_size * seq_len_q, self.embed_dim))
        
        # Output projection
        output = matmul(attn_output, self.out_proj_weight)
        if self.out_proj_bias is not None:
            output = output + broadcast_to(self.out_proj_bias, output.shape)
        
        output = output.reshape((batch_size, seq_len_q, self.embed_dim))
        
        return output


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
        self.training = True

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
        self.training = True

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
            #mean =summation(linear_out, axes=0, keepdims=True) / N
            mean = linear_out.sum(axes=0) / N
            mean_broadcast = broadcast_to(mean, linear_out.shape)
            var = ((linear_out - mean_broadcast) ** 2).sum(axes=0) / N 
            #var = summation((linear_out - mean_broadcast) ** 2, axes=0, keepdims=True) / N
            
            # Update running stats
            self.running_mean = ((1 - self.momentum) * self.running_mean + 
                               self.momentum * mean.detach())
            self.running_var = ((1 - self.momentum) * self.running_var + 
                              self.momentum * var.detach())
        
        return out
