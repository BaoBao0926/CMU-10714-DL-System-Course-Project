#!/bin/bash
echo "Start transfer CUDA code to HIP..."

# get python path
PYTHON_INCLUDES=$(python3-config --includes | cut -d' ' -f1 | sed 's/-I//')

hipify-clang src/ndarray_backend_cuda.cu \
  --cuda-path=/usr/local/cuda-12.8 \
  -o src/ndarray_backend_hip.cpp \
  -I /root/10714-project/.venv/lib/python3.12/site-packages/pybind11/include \
  -I $PYTHON_INCLUDES

if [ $? -eq 0 ]; then
    echo "✅ Transfer successful"
else
    echo "❌ Transfer failed!"
fi