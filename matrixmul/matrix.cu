#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

/**
 * Minimal example to apply sigmoid activation on a tensor using cuDNN.
 **/
int main(int argc, char** argv)
{    
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    return 0;
}