import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in+fan_out))
    return rand(fan_in,fan_out,low=-a,high=a,**kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in+fan_out))
    return randn(fan_in,fan_out,std=std,**kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3/fan_in)
    return rand(fan_in,fan_out,low=-bound,high=bound,**kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # fan_in = input_size * receptive field size(kxk)
    if shape is not None and len(shape)==4:# assume shape in (K,K,Cin,Cout)
        recep_field_size = 1
        for dim in shape[:2]: 
            recep_field_size *= dim
        fan_in = shape[2] * recep_field_size
        fan_out = shape[3] * recep_field_size
    bound = math.sqrt(2) * math.sqrt(3/fan_in)
    if shape is not None:
        return rand(*shape,low=-bound,high=bound,**kwargs)
    else:
        return rand(fan_in,fan_out,low=-bound,high=bound,**kwargs)

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(fan_in,fan_out,std=std,**kwargs)
    ### END YOUR SOLUTION