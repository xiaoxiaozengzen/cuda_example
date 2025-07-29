cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(vectoradd)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_COMPILER $ENV{CUDAToolkit_ROOT}/bin/nvcc)
message(STATUS "CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA compiler not found")
endif()

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_DIR: ${CUDAToolkit_DIR}")
message(STATUS "ENV{CUDAToolkit_ROOT}: $ENV{CUDAToolkit_ROOT}")
message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
message(STATUS "CUDAToolkit_FOUND ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIRS: ${CUDAToolkit_LIBRARY_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")

add_executable(vectoradd add.cu)
target_include_directories(vectoradd PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(vectoradd PRIVATE CUDA::cudart)