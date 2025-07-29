#include <iostream>
#include <cuda.h>

using namespace std;

/**
 * @brief CUDA kernel function
 * 
 * @note 执行的时候使用 <<<N,M>>> 来指定 N 个块，每个块有 M 个线程。每个线程都有唯一的 threadIdx.x。
 */
__global__ void VecAdd(int* A, int* B, int* C, int* device_count)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
    atomicAdd(device_count, 1);
}

__global__ void Veclist(int * D, int* device_count)
{
    int i = blockIdx.x * 10 + threadIdx.x;
    D[i] = blockIdx.x * 10 + threadIdx.x;
    atomicAdd(device_count, 1);
}

int main(){
    const int N = 10;
    int *a, *b, *c = nullptr;
    int * temp = new int[N];

    int * host_count = nullptr;
    int * device_count = nullptr;
    host_count = new int(0);
    cudaMalloc(&device_count, sizeof(int));
    cudaMemcpy(device_count, host_count, sizeof(int), cudaMemcpyHostToDevice);

    // malloc DEVICE memory for a, b, c
    cudaMalloc(&a, N*sizeof(int));
    cudaMalloc(&b, N*sizeof(int));
    cudaMalloc(&c, N*sizeof(int));

    // set a's values: a[i] = i
    for(int i=0;i<N;i++) {
      temp[i] = i + 1 ;
    }
    cudaMemcpy(a, temp, N*sizeof(int), cudaMemcpyHostToDevice);

    // set b's values: b[i] = 2*i
    for(int i=0;i<N;i++) {
      temp[i] = 2*(i + 1);
    }
    cudaMemcpy(b, temp, N*sizeof(int), cudaMemcpyHostToDevice);

    // add executed in parallel
    VecAdd<<<1,10>>>(a, b, c, device_count);

    cudaMemcpy(temp, c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Result of vector addition:" << endl;
    for(int i=0;i<N;i++) {
        cout << temp[i] << " ";
    }
    cout << endl;

    cudaMemcpy(host_count, device_count, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Number of additions performed: " << *host_count << endl;

    // free HOST & DEVICE memory
    delete [] temp;
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete host_count;
    cudaFree(device_count);


    /****************************************** */
    int * D = nullptr;
    cudaMalloc(&D, N*sizeof(int)* 3);
    cudaMemset(D, 0, N*sizeof(int) * 3);
    int * d_data = new int[N * 3];
    for(int i=0;i<N*3;i++) {
        d_data[i] = 0;
    }

    int * d_count = new int(0);
    int * device_count2 = nullptr;
    cudaMalloc(&device_count2, sizeof(int));
    cudaMemcpy(device_count2, d_count, sizeof(int), cudaMemcpyHostToDevice);

    Veclist<<<3,10>>>(D, device_count2);
    cudaMemcpy(d_data, D, N*sizeof(int)*3, cudaMemcpyDeviceToHost);
    cout << "Result of vector list:" << endl;
    for(int i=0;i<N*3;i++) {
        cout << d_data[i] << " ";
    }
    cout << endl;
    cudaMemcpy(d_count, device_count2, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Number of additions performed: " << *d_count << endl;

    delete [] d_data;
    delete d_count;
    cudaFree(D);
    cudaFree(device_count2);
}