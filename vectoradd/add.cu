#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda.h>

using namespace std;

/**
 * @brief CUDA kernel function
 * <<<Dg, Db, Ns, S>>>：
 * Dg: 栅格的维数和大小，则总的block数为 Dg.x * Dg.y
 * Db: 每个block的维数和大小，则每个block中的线程数为 Db.x * Db.y * Db.z
 * Ns: 类型是size_t，指定在共享内存中动态分配的字节数，默认为0
 * S: cudaStream_t 指定在该流上执行内核，默认为0，即默认流
 * 
 * @note
 * blockIdx.x: 当前block在x维度的索引
 * blockDim.x: 每个block在x维度的线程数
 * threadIdx.x: 当前线程在所属block中x维度的索引
 * gridDim.x: 当前栅格在x维度的block数
 * 
 * block_id = blockIdx.x + blockIdx.y * gridDim.x
 * threadsPerBlock = blockDim.x * blockDim.y * blockDim.z
 * local_thread_id = threadIdx.y * blockDim.x + threadIdx.x
 * global_id = block_id * threadsPerBlock + local_thread_id
 *           = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.y * blockDim.x + threadIdx.x)
 */
__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.y * blockDim.x + threadIdx.x);
    C[i] = A[i] + B[i];
}

/**
 * struct dim3 {
 *     unsigned int x, y, z;
 *     dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1) : x(x), y(y), z(z) {}
 *     dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
 * };
 */

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
    cudaMalloc(&device_first, N*sizeof(int));
    cudaMalloc(&device_second, N*sizeof(int));
    cudaMalloc(&device_result, N*sizeof(int));

    for(int i=0;i<N;i++) {
      host_first[i] = i + 1;
      host_second[i] = 2*(i + 1);
    }
    cudaMemcpy(device_first, host_first, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_second, host_second, N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim(2, 1, 1);   // Number of blocks in the grid
    dim3 blockDim(5, 2, 1); // Number of threads in each block
    std::cout << "gridDim.x: "<< gridDim.x << ", gridDim.y: " << gridDim.y << ", gridDim.z: " << gridDim.z << std::endl;
    std::cout << "blockDim.x: " << blockDim.x << ", blockDim.y: " << blockDim.y << ", blockDim.z: " << blockDim.z << std::endl;
    std::cout << "Total threads: " << (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) << std::endl;
    VecAdd<<<gridDim, blockDim>>>(device_first, device_second, device_result);

    cudaMemcpy(host_result, device_result, N*sizeof(int), cudaMemcpyDeviceToHost);
    cout << "host_first: " << endl;
    for(int i=0;i<N;i++) {
        cout << std::setw(3) << host_first[i] << " ";
    }
    cout << endl;
    cout << "host_second: " << endl;
    for(int i=0;i<N;i++) {
        cout << std::setw(3) << host_second[i] << " ";
    }
    cout << endl;
    cout << "Result of vector addition:" << endl;
    for(int i=0;i<N;i++) {
        cout << std::setw(3) << host_result[i] << " ";
    }
    cout << endl;

    delete[] host_first;
    delete[] host_second;
    delete[] host_result;
    cudaFree(device_first);
    cudaFree(device_second);
    cudaFree(device_result);
}

int main(){
    std::cout << "================ Vector Addition Example ================" << std::endl;
    vec_add_example();

    return 0;
}