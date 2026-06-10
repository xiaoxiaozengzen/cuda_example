#include <iostream>
#include <vector>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << " at line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err)      \
                      << " (" << err << ")" << std::endl;                 \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

void vec_add_example() {
    const int N = 20;
    int* host_first = nullptr;
    int* host_second = nullptr;
    int* host_result = nullptr;
    int* device_first = nullptr;
    int* device_second = nullptr;
    int* device_result = nullptr;

    host_first = new int[N];
    host_second = new int[N];
    host_result = new int[N];  
    std::cerr << "Host memory allocated for " << N << " integers." << std::endl;  
    CHECK_CUDA(cudaMalloc(&device_first, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_second, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_result, N*sizeof(int)));

    std::cerr << "Device memory allocated for " << N << " integers." << std::endl;
    for(int i=0;i<N;i++) {
      host_first[i] = i + 1;
      host_second[i] = 2*(i + 1);
    }
    CHECK_CUDA(cudaMemcpy(device_first, host_first, N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_second, host_second, N*sizeof(int), cudaMemcpyHostToDevice));

    dim3 gridDim(2);   // Number of blocks in the grid
    dim3 blockDim(10); // Number of threads in each block
    std::cout << "gridDim.x: "<< gridDim.x << ", gridDim.y: " << gridDim.y << ", gridDim.z: " << gridDim.z << std::endl;
    std::cout << "blockDim.x: " << blockDim.x << ", blockDim.y: " << blockDim.y << ", blockDim.z: " << blockDim.z << std::endl;
    std::cout << "Total threads: " << (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) << std::endl;
    {
        /**
         * @brief 开启一个NVTX范围
         * @param message const char* 类型的消息字符串，表示范围的名称
         */
        nvtxRangePush("Vector Addition Kernel");
        VecAdd<<<gridDim, blockDim>>>(device_first, device_second, device_result);

        /**
         * @brief 结束当前的NVTX范围
         */
        nvtxRangePop();

    }

    CHECK_CUDA(cudaMemcpy(host_result, device_result, N*sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "host_first: " << std::endl;
    for(int i=0;i<N;i++) {
        std::cout << std::setw(3) << host_first[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "host_second: " << std::endl;
    for(int i=0;i<N;i++) {
        std::cout << std::setw(3) << host_second[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Result of vector addition:" << std::endl;
    for(int i=0;i<N;i++) {
        std::cout << std::setw(3) << host_result[i] << " ";
    }
    std::cout << std::endl;

    delete[] host_first;
    delete[] host_second;
    delete[] host_result;
    CHECK_CUDA(cudaFree(device_first));
    CHECK_CUDA(cudaFree(device_second));
    CHECK_CUDA(cudaFree(device_result));
}

__global__ void VecAdd1(int* A, int* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + 1;
}

void loop_multi_test() {
    const int N = 20;
    const int loop_count = 40;
    int* host = nullptr;
    int* host_result = nullptr;
    int* device = nullptr;
    int* device_result = nullptr;

    host = new int[N];
    host_result = new int[N];   
    CHECK_CUDA(cudaMalloc(&device, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_result, N*sizeof(int)));

    std::cerr << "Device memory allocated for " << N << " integers." << std::endl;
    for(int i=0;i<N;i++) {
      host[i] = i + 1;
    }
    CHECK_CUDA(cudaMemcpy(device, host, N*sizeof(int), cudaMemcpyHostToDevice));

    dim3 gridDim(2);   // Number of blocks in the grid
    dim3 blockDim(10); // Number of threads in each block
    std::cout << "gridDim.x: "<< gridDim.x << ", gridDim.y: " << gridDim.y << ", gridDim.z: " << gridDim.z << std::endl;
    std::cout << "blockDim.x: " << blockDim.x << ", blockDim.y: " << blockDim.y << ", blockDim.z: " << blockDim.z << std::endl;
    std::cout << "Total threads: " << (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) << std::endl;

    {
        nvtxRangePush("Loop Multi Test");
        for(int i=0; i<loop_count; i++) {
            nvtxRangePush(("Iteration " + std::to_string(i)).c_str());
            VecAdd1<<<gridDim, blockDim>>>(device, device_result);
            nvtxRangePop();
        }
        nvtxRangePop();
    }

    CHECK_CUDA(cudaMemcpy(host_result, device_result, N*sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "host: " << std::endl;
    for(int i=0;i<N;i++) {
        std::cout << std::setw(3) << host[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Result of vector addition:" << std::endl;
    for(int i=0;i<N;i++) {
        std::cout << std::setw(3) << host_result[i] << " ";
    }
    std::cout << std::endl;

    delete[] host;
    delete[] host_result;
    CHECK_CUDA(cudaFree(device));
    CHECK_CUDA(cudaFree(device_result));
}

int main(){
    std::cout << "================ Vector Addition Example ================" << std::endl;
    vec_add_example();
    std::cout << "================ Loop Multi Test ================" << std::endl;
    loop_multi_test();

    return 0;
}