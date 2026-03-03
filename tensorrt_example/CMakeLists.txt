cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(tensorrt_example LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_COMPILER $ENV{CUDAToolkit_ROOT}/bin/nvcc)
if(EXISTS ${CMAKE_CUDA_COMPILER})
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA compiler not found")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(tensorrt REQUIRED)
message(STATUS "Found TensorRT: ${tensorrt_DIR}")
message(STATUS "TensorRT include dirs: ${tensorrt_INCLUDE_DIRS}")
message(STATUS "TensorRT libraries: ${tensorrt_LIBRARIES}")

add_executable(load_engine load_engine.cpp)
target_include_directories(
    load_engine PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${tensorrt_INCLUDE_DIRS}
)
target_link_libraries(
    load_engine PRIVATE 
    CUDA::cudart 
    ${tensorrt_LIBRARIES}
)