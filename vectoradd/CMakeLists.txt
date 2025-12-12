cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(vectoradd)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_COMPILER $ENV{CUDAToolkit_ROOT}/bin/nvcc)
if(EXISTS ${CMAKE_CUDA_COMPILER})
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA compiler not found")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(vectoradd add.cu)
target_include_directories(vectoradd PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(vectoradd PRIVATE CUDA::cudart)