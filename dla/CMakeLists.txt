cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
project(dla LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_COMPILER $ENV{CUDAToolkit_ROOT}/bin/nvcc)
message(STATUS "CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
if(EXISTS ${CMAKE_CUDA_COMPILER})
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA compiler not found")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(tensorrt REQUIRED)

set(DLA_BUILD_EXAMPLE build_engine)
add_executable(${DLA_BUILD_EXAMPLE} ${DLA_BUILD_EXAMPLE}.cpp)
target_include_directories(${DLA_BUILD_EXAMPLE} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${tensorrt_INCLUDE_DIRS})
target_link_libraries(${DLA_BUILD_EXAMPLE} PRIVATE CUDA::cudart ${tensorrt_LIBRARIES})

set(DLA_RUN_EXAMPLE run_inference)
add_executable(${DLA_RUN_EXAMPLE} ${DLA_RUN_EXAMPLE}.cpp)
target_include_directories(${DLA_RUN_EXAMPLE} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${tensorrt_INCLUDE_DIRS})
target_link_libraries(${DLA_RUN_EXAMPLE} PRIVATE CUDA::cudart ${tensorrt_LIBRARIES})