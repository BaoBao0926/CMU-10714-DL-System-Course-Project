.PHONY: lib, pybind, clean, format, all

all: lib

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE) ndarray_backend_cpu || true
	@cd build; $(MAKE) ndarray_backend_hip || true # 先编译hip再编译cuda,如果到只有Nvidia GPU的版本
	@cd build; $(MAKE) ndarray_backend_cuda || true
	@echo ""
	@echo "Build Summary:"
	@if [ -f needle/backend_ndarray/ndarray_backend_hip.cpython-*.so ]; then \
	  echo "✓ HIP backend compiled"; \
	  echo "   Hint: export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\$$LD_LIBRARY_PATH"; \
	fi
	@if [ -f needle/backend_ndarray/ndarray_backend_cpu.cpython-*.so ]; then \
	  echo "✓ CPU backend compiled"; \
	fi
	@if [ -f needle/backend_ndarray/ndarray_backend_cuda.cpython-*.so ]; then \
	  echo "✓ CUDA backend compiled"; \
	fi
	@if [ ! -f needle/backend_ndarray/ndarray_backend_hip.cpython-*.so ] && [ ! -f needle/backend_ndarray/ndarray_backend_cpu.cpython-*.so ]; then \
	  echo "✗ Build failed - no backends compiled"; \
	  exit 1; \
	fi

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so