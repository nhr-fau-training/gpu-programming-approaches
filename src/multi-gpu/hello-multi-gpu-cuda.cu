#include <cstdio>
#include <cstdlib>

#include "../cuda-util.h"


__global__ void hello(int device, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        printf("Hello from GPU %d, thread %ld\n", device, i0);
    }
}


int main(int argc, char *argv[]) {
    size_t nx = argc > 1 ? atoi(argv[1]) : 4;

    int numDevices;
    checkCudaError(cudaGetDeviceCount(&numDevices));
    printf("Found %d GPU(s)\n", numDevices);

    // launch on all GPUs - kernel launches are asynchronous, so the GPUs work concurrently
    for (int d = 0; d < numDevices; ++d) {
        checkCudaError(cudaSetDevice(d));
        hello<<<1, nx>>>(d, nx);
    }

    // wait for all GPUs
    for (int d = 0; d < numDevices; ++d) {
        checkCudaError(cudaSetDevice(d));
        checkCudaError(cudaDeviceSynchronize(), true);
    }

    return 0;
}
