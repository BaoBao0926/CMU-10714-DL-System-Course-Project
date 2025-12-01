"""Logic for backend selection"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd":
    print("Using needle backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        hip,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device

    NDArray = array_api.ndarray
elif BACKEND == "hip":
    print("Using HIP backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        hip,
        cpu,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
