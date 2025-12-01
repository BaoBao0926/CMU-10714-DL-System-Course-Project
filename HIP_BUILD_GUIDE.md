# AMD GPU HIP 后端编译指南

## 概述
本指南介绍如何修改 CMakeLists.txt 和相关编译配置文件，以编译可在 AMD GPU 上运行的 HIP 后端代码。

## 系统要求
- **ROCm 环境**: `/opt/rocm` (或通过 `ROCM_HOME` 环境变量指定)
- **编译器**: amdclang/amdclang++ (包含在 ROCm 中)
- **CMake**: 3.5 或更高版本
- **Python 开发库**: python3-dev
- **Pybind11**: 用于 Python 绑定

## 修改说明

### 1. CMakeLists.txt 主要修改

#### 添加 HIP 编译器配置
```cmake
cmake_minimum_required(VERSION 3.5)

# 设置 ROCm 路径
if(DEFINED ENV{ROCM_HOME})
  set(ROCM_HOME $ENV{ROCM_HOME})
else()
  set(ROCM_HOME "/opt/rocm")
endif()

# 设置 HIP 编译器为 amdclang++（不能使用 hipcc 包装器）
if(NOT CMAKE_HIP_COMPILER)
  set(CMAKE_HIP_COMPILER "${ROCM_HOME}/bin/amdclang++")
endif()

# 启用 HIP 语言
enable_language(HIP)

project(needle C CXX HIP)
```

**关键点**:
- `CMAKE_HIP_COMPILER` 必须设置为实际的编译器（amdclang++），不能是 hipcc 包装器
- 必须在 `project()` 命令之前调用 `enable_language(HIP)`
- ROCm 路径应该可配置

#### HIP 后端库配置
```cmake
find_package(HIP REQUIRED)
message(STATUS "Found HIP, building HIP backend for AMD GPU")

include_directories(SYSTEM ${HIP_INCLUDE_DIRS})

# 设置 HIP 编译标志
set(HIP_HIPCC_FLAGS "-std=c++14 -O2")
set(CMAKE_HIP_FLAGS "${HIP_HIPCC_FLAGS} ${CMAKE_HIP_FLAGS}")

# 检测 AMD GPU 架构（可选自动检测）
execute_process(COMMAND "rocm-smi" "--showid" ERROR_QUIET RESULT_VARIABLE AMD_RET OUTPUT_QUIET)
if(AMD_RET EQUAL "0")
  message(STATUS "AMD GPU detected, HIP will auto-detect architecture")
else()
  message(STATUS "No AMD GPU detected, using gfx900 as default")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx900")
endif()

# 创建 HIP 模块
add_library(ndarray_backend_hip MODULE src/ndarray_backend_hip.cpp)
set_source_files_properties(src/ndarray_backend_hip.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(ndarray_backend_hip PUBLIC ${LINKER_LIBS} hip::host hip::device)
pybind11_extension(ndarray_backend_hip)
pybind11_strip(ndarray_backend_hip)

set_target_properties(ndarray_backend_hip
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
  HIP_VISIBILITY_PRESET "hidden"
  PREFIX ""
)
```

## 编译方法

### 快速编译（推荐）
```bash
chmod +x build_hip.sh
./build_hip.sh
```

### 完全配置+编译
```bash
chmod +x setup_hip_build.sh
./setup_hip_build.sh
```

### 手动 CMake 编译
```bash
mkdir -p build
cd build

export ROCM_HOME=/opt/rocm
export CC=$ROCM_HOME/bin/amdclang
export CXX=$ROCM_HOME/bin/amdclang++

cmake \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DROCM_HOME=$ROCM_HOME \
  ..

make -j$(nproc)
```

### Make 命令
```bash
# 编译所有后端
make lib

# 只编译 HIP 后端
make -f Makefile.hip hip

# 只编译 CPU 后端
make -f Makefile.hip cpu

# 清理构建
make clean
```

## 常见问题解决

### 问题 1: "CMAKE_HIP_COMPILER is set to the hipcc wrapper"
**原因**: CMake 不支持 hipcc 包装器作为编译器
**解决方案**: 使用 amdclang++ 而不是 hipcc
```cmake
set(CMAKE_HIP_COMPILER "${ROCM_HOME}/bin/amdclang++")
```

### 问题 2: "C compiler not found" 或 "CXX compiler not found"
**原因**: 未设置编译器环境变量或使用了系统的 clang
**解决方案**: 显式设置 ROCm 提供的编译器
```bash
export CC=/opt/rocm/bin/amdclang
export CXX=/opt/rocm/bin/amdclang++
```

### 问题 3: "HIP header files not found"
**原因**: 编译时未包含 HIP 头文件路径
**解决方案**: 确保 HIP 库被正确找到
```cmake
find_package(HIP REQUIRED)
include_directories(SYSTEM ${HIP_INCLUDE_DIRS})
```

### 问题 4: GPU 架构不匹配
**原因**: 为不同的 GPU 使用了错误的架构标志
**解决方案**: 使用 `rocm-smi` 查询 GPU 信息，或使用自动检测

常见 GPU 架构:
- gfx900: Radeon Instinct MI25
- gfx906: Radeon Instinct MI50/MI60
- gfx908: Radeon Instinct MI100
- gfx90a: Radeon Instinct MI210/MI250
- gfx940: MI300X (本项目中的 GPU)

### 问题 5: Shell 脚本中的语法错误 `[: unexpected operator`
**原因**: 在某些 shell 中使用 `==` 进行字符串比较而不是 `=`
**解决方案**: 使用 `=` 而不是 `==` 进行字符串比较
```bash
# 错误
if [ "$1" == "help" ]; then

# 正确
if [ "$1" = "help" ]; then
```

## 验证编译

成功编译后应该生成:
```
needle/backend_ndarray/ndarray_backend_hip.cpython-312-x86_64-linux-gnu.so
```

验证库文件:
```bash
ls -lh needle/backend_ndarray/ndarray_backend_hip*.so
```

输出示例:
```
-rwxr-xr-x 1 root root 285K Nov 26 00:10 /root/10714/project/needle/backend_ndarray/ndarray_backend_hip.cpython-312-x86_64-linux-gnu.so
✓ HIP backend compiled successfully!
```

## 环境变量配置

编译前需要设置以下环境变量:

```bash
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
export CC=$ROCM_HOME/bin/amdclang
export CXX=$ROCM_HOME/bin/amdclang++
```

## Python 中使用 HIP 后端

```python
import needle as nd

# 查询可用的后端
print(nd.backend_ndarray.__file__)

# 设置为 HIP 后端
nd.backend_selection.set_backend("hip")
```

## 调试编译

如果编译失败，可以启用详细输出:
```bash
cd build
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1
```

查看详细的 HIP 编译选项:
```bash
hipcc --help
amdclang++ -help | grep offload
```

## 性能优化提示

1. **启用优化**:
   ```cmake
   set(HIP_HIPCC_FLAGS "-std=c++14 -O3 -march=native")
   ```

2. **调整工作组大小** (在 HIP 内核中):
   - 通常为 64 的倍数（AMD GPU 的 warp 大小）
   - 参考 `src/ndarray_backend_hip.cpp` 中的 `BASE_THREAD_NUM`

3. **使用内存合并** (shared memory tiling):
   - 参考代码中的 `MMKernel` 实现

## 相关文件

- `CMakeLists.txt`: 主编译配置文件（已修改）
- `build_hip.sh`: 简化的 HIP 编译脚本
- `setup_hip_build.sh`: 完整的 HIP 环境配置和检查脚本（已修改）
- `Makefile.hip`: 便捷的 Make 命令集
- `src/ndarray_backend_hip.cpp`: HIP 后端实现

## 参考资源

- [ROCm 官方文档](https://rocmdocs.amd.com/)
- [HIP 编程指南](https://github.com/ROCm/HIP)
- [CMake HIP 支持](https://cmake.org/cmake/help/latest/module/FindHIP.html)

## 总结

主要修改包括:

1. **CMakeLists.txt**:
   - 在 `cmake_minimum_required` 之后立即设置 ROCm 路径和 HIP 编译器
   - 在 `project()` 前调用 `enable_language(HIP)`
   - 使用 `find_package(HIP REQUIRED)` 而不是可选的 `find_package(HIP)`
   - 链接 `hip::host` 和 `hip::device` 库

2. **setup_hip_build.sh**:
   - 修复 shell 脚本中的条件判断语法（`=` 而不是 `==`）
   - 设置正确的 ROCm 编译器路径

3. **build_hip.sh** (新增):
   - 简化的编译脚本，自动设置所有环境变量

4. **Makefile.hip** (新增):
   - 方便的 Make 目标以单独编译 HIP 或 CPU 后端cmake
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_FLAGS "-O2 ${CMAKE_HIP_FLAGS}")
```

#### 3. HIP后端编译块 (新增)
```cmake
###################
### HIP BACKEND ###
###################
find_package(HIP)
if(HIP_FOUND)
  message(STATUS "Found HIP, building HIP backend for AMD GPU")

  include_directories(SYSTEM ${HIP_INCLUDE_DIRS})

  # 检测AMD GPU架构
  execute_process(COMMAND "rocm-smi" "--showid" ERROR_QUIET RESULT_VARIABLE AMD_RET)
  if(AMD_RET EQUAL "0")
    message(STATUS "AMD GPU detected, using AUTO architecture detection")
    set(HIP_HIPCC_FLAGS "-std=c++14 -O2 ${HIP_HIPCC_FLAGS}")
  else()
    message(STATUS "No AMD GPU detected, using gfx900 as default (Fiji/MI25)")
    set(HIP_HIPCC_FLAGS "-std=c++14 -O2 --offload-arch=gfx900 ${HIP_HIPCC_FLAGS}")
  endif()

  # 添加HIP模块
  add_library(ndarray_backend_hip MODULE src/ndarray_backend_hip.cpp)
  set_source_files_properties(src/ndarray_backend_hip.cpp PROPERTIES LANGUAGE HIP)
  
  target_link_libraries(ndarray_backend_hip PUBLIC ${LINKER_LIBS} hip::host)
  pybind11_extension(ndarray_backend_hip)
  pybind11_strip(ndarray_backend_hip)

  # 输出到ffi文件夹
  set_target_properties(ndarray_backend_hip
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
    HIP_VISIBILITY_PRESET "hidden"
  )

endif()
```

### Makefile 修改内容

#### 1. 更新format命令以支持.cpp文件
```makefile
format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu src/*.cpp
```

#### 2. 更新clean命令
```makefile
clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so needle/backend_ndarray/ndarray_backend*.so
```

## 编译流程 (Build Process)

### 方式1：使用Makefile（推荐）
```bash
# 编译所有后端（CPU、CUDA、HIP）
make lib

# 清理构建文件
make clean

# 代码格式化
make format
```

### 方式2：使用CMake手动编译
```bash
# 创建构建目录
mkdir -p build
cd build

# 运行CMake
cmake ..

# 编译
make -j$(nproc)
```

### 方式3：使用CMake指定HIP编译器
```bash
# 如果CMake未自动检测到HIP
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=clang -DCMAKE_C_COMPILER=clang ..
make -j$(nproc)
```

## GPU架构支持 (GPU Architecture Support)

### 常见AMD GPU架构

| 架构代码 | GPU型号 | 用途 |
|---------|--------|------|
| gfx900 | Fiji, MI25, Radeon VII | 数据中心、工作站 |
| gfx906 | MI50, MI60 | 高性能计算 |
| gfx908 | MI100 | AI/ML训练 |
| gfx90a | MI210, MI250 | 高性能AI |
| gfx1030 | RX 6800 | 消费级GPU |
| gfx1100 | RDNA 3 | 新一代消费级 |

### 自动检测
- 如果系统有AMD GPU，CMake会自动检测
- 如果没有GPU但想交叉编译，手动指定架构

### 手动指定架构
编辑CMakeLists.txt，修改HIP编译标志：
```cmake
set(HIP_HIPCC_FLAGS "-std=c++14 -O2 --offload-arch=gfx906 ${HIP_HIPCC_FLAGS}")
```

## 编译输出 (Build Output)

编译成功后，生成的库文件位置：
```
needle/backend_ndarray/ndarray_backend_hip.so
```

### 验证编译
```bash
# 检查库文件是否存在
ls -la needle/backend_ndarray/ndarray_backend_hip.so

# 查看库文件信息
file needle/backend_ndarray/ndarray_backend_hip.so
```

## 故障排查 (Troubleshooting)

### 1. CMake找不到HIP
```bash
# 确保ROCm在PATH中
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 重新运行CMake
cd build
rm -rf CMakeCache.txt CMakeFiles
cmake ..
```

### 2. hipcc编译错误
```bash
# 检查hipcc版本
hipcc --version

# 尝试使用verbose模式调试
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make
```

### 3. GPU不被识别
```bash
# 检查GPU驱动
rocm-smi

# 如果无输出，检查驱动安装
sudo apt-get install rocm-dkms
```

### 4. 编译目标架构不匹配
```bash
# 查看GPU架构
rocm-smi --showid

# 更新CMakeLists.txt中的--offload-arch标志
```

## 运行测试 (Testing)

### Python中使用HIP后端
```python
import needle as nd

# 设置后端为HIP
nd.backend_selection.set_backend_type("hip")

# 创建数组
a = nd.array([1, 2, 3], device=nd.hip())

# 验证
print(a)
```

### 运行完整测试
```bash
# 如果有测试脚本
python tests/test_hip_backend.py

# 或运行现有测试（它会自动选择可用的后端）
python manual_test_hw4.py
```

## 环境变量配置 (Environment Variables)

在`~/.bashrc`或`~/.zshrc`中添加：
```bash
# ROCm环境变量
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# HIP特定变量
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
```

## 性能优化 (Performance Tips)

1. **使用正确的GPU架构标志** - 确保`--offload-arch`与实际GPU匹配
2. **启用编译器优化** - `-O2`或`-O3`标志已在CMakeLists.txt中设置
3. **共享内存优化** - `src/ndarray_backend_hip.cpp`中的共享内存配置可根据GPU调整
4. **线程块大小** - 在HIP代码中调整`BASE_THREAD_NUM`以匹配GPU特性

## 参考资源 (References)

- [ROCm官方文档](https://rocmdocs.amd.com/)
- [HIP编程指南](https://rocmdocs.amd.com/en/docs/deploy/linux/index.html)
- [CMake HIP支持](https://cmake.org/cmake/help/latest/module/FindHIP.html)
- [HIP to CUDA迁移指南](https://rocmdocs.amd.com/en/docs/deploy/linux/index.html)

## 总结 (Summary)

通过以上修改，您的项目现在支持：
- ✅ CPU后端（原有）
- ✅ CUDA后端（原有）
- ✅ HIP后端（新增，用于AMD GPU）

编译时会自动检测可用的编译工具链，编译相应的后端。
