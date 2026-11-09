#include "increase-cuda.h"

#include "../cuda-util.h"


__global__ void increaseKernel(double *data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] += 1;
    }
}


void increaseCuda(double *d_data, size_t nx) {
    const size_t blockSize = 256;
    const size_t numBlocks = (nx + blockSize - 1) / blockSize;

    increaseKernel<<<numBlocks, blockSize>>>(d_data, nx);

    // other approaches use their own streams - wait for completion before handing the data back
    checkCudaError(cudaDeviceSynchronize(), true);
}
