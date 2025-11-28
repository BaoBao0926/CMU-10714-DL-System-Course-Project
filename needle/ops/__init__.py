from .ops_mathematic import *
from .ops_logarithmic import *
from .ops_tuple import *
from .ops_fused import *

# 根据后端动态导入或覆盖算子
def _setup_backend_ops():
    """根据当前后端选择算子实现"""
    from ..backend_selection import BACKEND
    
    if BACKEND == "hip":
        # 从 ops_hip 导入并覆盖全局符号
        try:
            #from .ops_hip import *
            # 显式列出要覆盖的算子
            from . import ops_hip 
            globals().update({
                "conv": ops_hip.conv,
                "batchnorm2d": ops_hip.batchnorm2d,
                "conv_batchnorm2d_relu": ops_hip.conv_batchnorm2d_relu,
            })
            print("[Needle] Using HIP-optimized operators")
        except ImportError as e:
            print(f"[Needle] Warning: Failed to load HIP ops: {e}")
            print("[Needle] Falling back to default ops")
    # 其他后端可以继续使用默认的 ops_mathematic

_setup_backend_ops()