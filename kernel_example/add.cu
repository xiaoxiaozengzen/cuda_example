#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda.h>

using namespace std;

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
 * local_thread_id = threadIdx.x + threadIdx.y * blockDim.x
 * global_id = block_id * threadsPerBlock + local_thread_id
 *           = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.x + threadIdx.y * blockDim.x)
 *
 * @note kernel函数中只要使用*去解引用指针，则必须保证指针指向的是设备内存，
 *       否则会出现非法访问错误（Illegal memory access）。
 *       因此，在kernel函数中，所有指针参数都必须是设备内存的地址。
 */
__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.x + threadIdx.y * blockDim.x);
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
    CHECK_CUDA(cudaMalloc(&device_first, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_second, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_result, N*sizeof(int)));

    for(int i=0;i<N;i++) {
      host_first[i] = i + 1;
      host_second[i] = 2*(i + 1);
    }
    CHECK_CUDA(cudaMemcpy(device_first, host_first, N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_second, host_second, N*sizeof(int), cudaMemcpyHostToDevice));

    dim3 gridDim(2, 1, 1);   // Number of blocks in the grid
    dim3 blockDim(5, 2, 1); // Number of threads in each block
    std::cout << "gridDim.x: "<< gridDim.x << ", gridDim.y: " << gridDim.y << ", gridDim.z: " << gridDim.z << std::endl;
    std::cout << "blockDim.x: " << blockDim.x << ", blockDim.y: " << blockDim.y << ", blockDim.z: " << blockDim.z << std::endl;
    std::cout << "Total threads: " << (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) << std::endl;
    VecAdd<<<gridDim, blockDim>>>(device_first, device_second, device_result);

    CHECK_CUDA(cudaMemcpy(host_result, device_result, N*sizeof(int), cudaMemcpyDeviceToHost));
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
    CHECK_CUDA(cudaFree(device_first));
    CHECK_CUDA(cudaFree(device_second));
    CHECK_CUDA(cudaFree(device_result));
}

__global__ void atomicAdd_kernel(int* data, int* old_data) {
    int old = atomicAdd(data, 1);
    if (old_data) {
        *old_data = old;
    }
}

void atomicAdd_example() {
    int h_src_data = 10;
    int h_old_data = 0;
    int* d_src_data = nullptr;
    int* d_old_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src_data, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_old_data, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_src_data, &h_src_data, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_old_data, &h_old_data, sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "Before atomicAdd, src_data: " << h_src_data << ", old_data: " << h_old_data << std::endl;
    atomicAdd_kernel<<<1, 1>>>(d_src_data, d_old_data);
    CHECK_CUDA(cudaMemcpy(&h_src_data, d_src_data, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_old_data, d_old_data, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "After atomicAdd, src_data: " << h_src_data << ", old_data: " << h_old_data << std::endl;
    CHECK_CUDA(cudaFree(d_src_data));
    CHECK_CUDA(cudaFree(d_old_data));
}

__global__ void num_test_kernel(int* counter) {
    atomicAdd(counter, 1);
}

void exec_num_test() {
    int h_counter = 0;
    int* d_counter = nullptr;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice));
    dim3 gridDim(5, 2, 1);   // Number of blocks in the grid
    dim3 blockDim(10, 3, 1); // Number of threads in each block
    num_test_kernel<<<gridDim, blockDim>>>(d_counter);
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    int expected_count = (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z);
    std::cout << "Total increments: " << h_counter << " (expected: " << expected_count << ")" << std::endl;
    CHECK_CUDA(cudaFree(d_counter));
}

int main(){
    std::cout << "================ Vector Addition Example ================" << std::endl;
    vec_add_example();
    std::cout << "================ Atomic Add Example ================" << std::endl;
    atomicAdd_example();
    std::cout << "================ Num Test Example ================" << std::endl;
    exec_num_test();

    return 0;
}