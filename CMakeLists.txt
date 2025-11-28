cmake_minimum_required(VERSION 3.5)

# ============================================================================
# HIP/ROCm Configuration - Auto-detect and setup
# ============================================================================

# Try to detect ROCm/HIP installation
if(DEFINED ENV{ROCM_HOME})
  set(ROCM_HOME $ENV{ROCM_HOME})
else()
  set(ROCM_HOME "/opt/rocm")
endif()

# Check if HIP is available
if(EXISTS "${ROCM_HOME}/bin/hipcc" AND EXISTS "${ROCM_HOME}/bin/amdclang++")
  set(HIP_AVAILABLE TRUE)
  message(STATUS "ROCm found at: ${ROCM_HOME}")
else()
  set(HIP_AVAILABLE FALSE)
  message(WARNING "ROCm/HIP not found at ${ROCM_HOME}. HIP backend will not be compiled.")
endif()

# If HIP is available, setup compiler before project() declaration
if(HIP_AVAILABLE)
  # Set HIP compiler to clang (not hipcc wrapper)
  if(NOT CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${ROCM_HOME}/bin/amdclang++")
  endif()
  
  # Use ROCm compilers for C/CXX
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${ROCM_HOME}/bin/amdclang")
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${ROCM_HOME}/bin/amdclang++")
  endif()
  
  # Set CMAKE_PREFIX_PATH for HIP packages
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_HOME}/lib/cmake")
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_HOME}/hip/cmake")
  
  # Enable HIP language
  enable_language(HIP)
  set(LANGUAGES_TO_ENABLE "C;CXX;HIP")
else()
  set(LANGUAGES_TO_ENABLE "C;CXX")
endif()

project(needle ${LANGUAGES_TO_ENABLE})
cmake_policy(SET CMP0146 OLD)

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_HIP_STANDARD 14)
  set(CMAKE_HIP_FLAGS "-O2 ${CMAKE_HIP_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_HIP_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

else()
  message(WARNING "CUDA not found or not compatible. CUDA backend will not be compiled.")
endif()

###################
### HIP BACKEND ###
###################
if(HIP_AVAILABLE)
  find_package(HIP REQUIRED)
  message(STATUS "Found HIP, building HIP backend for AMD GPU")

  # Try to find MIOpen
  find_path(MIOPEN_INCLUDE_DIR miopen/miopen.h PATHS ${ROCM_HOME}/include /opt/rocm/include)
  find_library(MIOPEN_LIBRARY NAMES MIOpen miopen PATHS ${ROCM_HOME}/lib ${ROCM_HOME}/lib64 /opt/rocm/lib /opt/rocm/lib64)
  if(NOT MIOPEN_LIBRARY)
    message(FATAL_ERROR "MIOpen library not found. Expected in ${ROCM_HOME}/lib or ${ROCM_HOME}/lib64")
  endif()
  message(STATUS "MIOpen include: ${MIOPEN_INCLUDE_DIR}")
  message(STATUS "MIOpen library: ${MIOPEN_LIBRARY}")

  # Create HIP module (single target definition)
  add_library(ndarray_backend_hip MODULE src/ndarray_backend_hip.cpp)
  set_source_files_properties(src/ndarray_backend_hip.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(ndarray_backend_hip PRIVATE ${HIP_INCLUDE_DIRS} ${MIOPEN_INCLUDE_DIR})
  
  # Link HIP and MIOpen
  target_link_libraries(ndarray_backend_hip PRIVATE hip::host hip::device ${MIOPEN_LIBRARY} ${LINKER_LIBS})

  # Ensure runtime can find libMIOpen.so
  # Ensure runtime can find libMIOpen.so via rpath
  set_target_properties(ndarray_backend_hip PROPERTIES
    BUILD_RPATH "${ROCM_HOME}/lib;${ROCM_HOME}/lib64;/opt/rocm/lib;/opt/rocm/lib64"
    INSTALL_RPATH "${ROCM_HOME}/lib;${ROCM_HOME}/lib64;/opt/rocm/lib;/opt/rocm/lib64"
  )

  include_directories(SYSTEM ${HIP_INCLUDE_DIRS})

  # Set HIP compiler flags
  set(HIP_HIPCC_FLAGS "-std=c++14 -O2")
  set(CMAKE_HIP_FLAGS "${HIP_HIPCC_FLAGS} ${CMAKE_HIP_FLAGS}")

  # Auto-detect AMD GPU architecture
  execute_process(COMMAND "${ROCM_HOME}/bin/rocm-smi" "--showid" ERROR_QUIET RESULT_VARIABLE AMD_RET OUTPUT_QUIET)
  if(AMD_RET EQUAL "0")
    message(STATUS "AMD GPU detected, HIP will auto-detect architecture")
  else()
    message(STATUS "No AMD GPU detected, using gfx900 as default (Fiji/MI25)")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx900")
  endif()

  # PyBind11 properties
  pybind11_extension(ndarray_backend_hip)
  pybind11_strip(ndarray_backend_hip)

  # Output to backend folder
  set_target_properties(ndarray_backend_hip
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
    HIP_VISIBILITY_PRESET "hidden"
    PREFIX ""
  )
  
  message(STATUS "✓ HIP backend will be compiled")
else()
  message(STATUS "✗ HIP backend will NOT be compiled (ROCm not found)")
endif()