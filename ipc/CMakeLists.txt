cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
project(ipc LANGUAGES CXX CUDA)

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

set(IPC_CLIENT ipc_client)
add_executable(${IPC_CLIENT} ${IPC_CLIENT}.cpp)
target_include_directories(${IPC_CLIENT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${IPC_CLIENT} PRIVATE CUDA::cudart)

set(IPC_SERVER ipc_server)
add_executable(${IPC_SERVER} ${IPC_SERVER}.cpp)
target_include_directories(${IPC_SERVER} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${IPC_SERVER} PRIVATE CUDA::cudart)