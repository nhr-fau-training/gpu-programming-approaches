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
    checkCudaError(cudaMallocHost((void **)&data, sizeof(double) * nx));

    double *d_data;
    checkCudaError(cudaMalloc((void **)&d_data, sizeof(double) * nx));

    // init
    init(data, nx);

    checkCudaError(cudaMemcpy(d_data, data, sizeof(double) * nx, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto stopWork = std::chrono::steady_clock::now();

    printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

    checkCudaError(cudaMemcpy(data, d_data, sizeof(double) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_data));

    checkCudaError(cudaFreeHost(data));

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
