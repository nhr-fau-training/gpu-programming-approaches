#include "increase-util.h"

#include "../cuda-util.h"


inline void init(double *data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}


__global__ void increase(double *data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] += 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto startGlobal = std::chrono::steady_clock::now();

    double *data;
    checkCudaError(cudaMallocManaged((void **)&data, sizeof(double) * nx));

    // init
    init(data, nx);

    cudaMemLocation deviceLocation{cudaMemLocationTypeDevice, 0};
    checkCudaError(cudaMemPrefetchAsync(data, sizeof(double) * nx, deviceLocation, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto stopWork = std::chrono::steady_clock::now();

    printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

    cudaMemLocation hostLocation{cudaMemLocationTypeHost, 0};
    checkCudaError(cudaMemPrefetchAsync(data, sizeof(double) * nx, hostLocation, 0));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(data));

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
