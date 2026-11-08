#include "increase-util.h"

#include "../cuda-util.h"


__global__ void increase(double *data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] += 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *data;
    checkCudaError(cudaMallocManaged((void **)&data, sizeof(double) * nx));

    // init
    initIncrease(data, nx);

    cudaMemLocation deviceLocation{cudaMemLocationTypeDevice, 0};
    checkCudaError(cudaMemPrefetchAsync(data, sizeof(double) * nx, deviceLocation, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    cudaMemLocation hostLocation{cudaMemLocationTypeHost, 0};
    checkCudaError(cudaMemPrefetchAsync(data, sizeof(double) * nx, hostLocation, 0));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(data));

    return 0;
}
