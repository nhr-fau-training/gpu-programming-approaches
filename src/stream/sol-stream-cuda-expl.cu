#include "stream-util.h"

#include "../cuda-util.h"


__global__ void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        dest[i0] = src[i0] + 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *dest;
    checkCudaError(cudaMallocHost((void **)&dest, sizeof(double) * nx));
    double *src;
    checkCudaError(cudaMallocHost((void **)&src, sizeof(double) * nx));

    double *d_dest;
    checkCudaError(cudaMalloc((void **)&d_dest, sizeof(double) * nx));
    double *d_src;
    checkCudaError(cudaMalloc((void **)&d_src, sizeof(double) * nx));

    // init
    initStream(dest, src, nx);

    checkCudaError(cudaMemcpy(d_dest, dest, sizeof(double) * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_src, src, sizeof(double) * nx, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream<<<ceilingDivide(nx, 256), 256>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream<<<ceilingDivide(nx, 256), 256>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(double) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(src, d_src, sizeof(double) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_dest));
    checkCudaError(cudaFree(d_src));

    checkCudaError(cudaFreeHost(dest));
    checkCudaError(cudaFreeHost(src));

    return 0;
}
