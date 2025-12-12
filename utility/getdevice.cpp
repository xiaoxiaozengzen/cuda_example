#include <iostream>

#include <cuda_runtime.h>
#include "cuda.h"

/**
 * typedef __device_builtin__ enum cudaError cudaError_t;
 */

void get_device_property(int device) {
    /**
     * struct cudaDeviceProp {
     *     char name[256];                   // GPU 名称
     *     cudaUUID_t uuid;                 // GPU 唯一标识符
     *     size_t totalGlobalMem;           // 全局内存总大小（字节）
     *     size_t sharedMemPerBlock;        // 每个块的共享内存大小（字节）
     *     int regsPerBlock;                // 每个块的寄存器数量
     *     int warpSize;                    // 每个 warp 的线程数量
     *     size_t memPitch;                 // 内存对齐大小（字节）
     *     int maxThreadsPerBlock;          // 每个块的最大线程数量
     *     int maxThreadsDim[3];            // 每个块在每个维度上的最大线程数量
     *     int maxGridSize[3];              // 网格在每个维度上的最大块数量
     *     int clockRate;                   // 时钟频率（千赫兹）
     *     size_t totalConstMem;            // 常量内存总大小（字节）
     *     int major;                       // 计算能力主版本号
     *     int minor;                       // 计算能力次版本号
     *     size_t textureAlignment;         // 纹理对齐大小（字节）
     *     int deviceOverlap;               // 设备重叠支持
     *     ... // 其他属性
     * };
     */
    struct cudaDeviceProp devProp;
    cudaError_t error = cudaGetDeviceProperties(&devProp, device);
    if(error != cudaError::cudaSuccess) {
        std::cerr << "Failed to get device properties for device " << device << ", error code: " << error << std::endl;
        return;
    }
    std::cout << "CUresult::CUDA_SUCCESS: " << static_cast<int>(CUresult::CUDA_SUCCESS) << std::endl;
    std::cout << "error code: " << static_cast<int>(error) << std::endl;
    std::cout << "Device " << device << ": " 
              << "\n  Name: " << devProp.name
              << "\n  Compute capability: " << devProp.major << "." << devProp.minor
              << "\n  Total global memory: " << static_cast<float>(devProp.totalGlobalMem) / (1024 * 1024) << " MB"
              << "\n  Shared memory per block: " << static_cast<float>(devProp.sharedMemPerBlock) / 1024 << " KB"
              << "\n  Registers per block: " << devProp.regsPerBlock
              << "\n  Warp size: " << devProp.warpSize
              << "\n  Memory pitch: " << devProp.memPitch
              << "\n  Max threads per block: " << devProp.maxThreadsPerBlock
              << "\n  Max threads dimensions: [" << devProp.maxThreadsDim[0] << ", "
              << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << "]"
              << "\n  Max grid size: [" << devProp.maxGridSize[0] << ", "
              << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << "]"
              << "\n  Clock rate: " << static_cast<float>(devProp.clockRate) / 1000 << " MHz"
              << "\n  Total constant memory: " << static_cast<float>(devProp.totalConstMem) / 1024 << " KB"
              << std::endl;
    int driverVersion = 0;
    int runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
}

void get_device_attribute(int device) {
    int attribute;
    /**
     * enum CUdevice_attribute {
     *     CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1, // 每个块的最大线程数量
     *     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,      // 块在 X 维度上的最大尺寸
     *     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,      // 块在 Y 维度上的最大尺寸
     *     CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,      // 块在 Z 维度上的最大尺寸
     *     CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,       // 网格在 X 维度上的最大尺寸
     *     CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,       // 网格在 Y 维度上的最大尺寸
     *     ... // 其他属性
     * };
     * ::cudaDevAttrMaxThreadsPerBlock = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
     * ::cudaDevAttrMaxBlockDimX = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
     */
    cudaError_t error = cudaDeviceGetAttribute(&attribute, cudaDevAttrMaxThreadsPerBlock, device);
    if(error != cudaSuccess) {
        std::cerr << "Failed to get device attribute for device " << device << ", error code: " << error << std::endl;
        return;
    }
    std::cout << "Device " << device << " max threads per block: " << attribute << std::endl;

}

int main(int argc, char** argv)
{    
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device; 
    cudaGetDevice(&device);

    get_device_property(device);
    get_device_attribute(device);

    return 0;
}