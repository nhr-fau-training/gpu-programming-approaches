#include "matrix-add-util.h"

#include "../cuda-util.h"


__global__ void matrixAdd(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, size_t nx, size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < nx && i1 < ny) {
        c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *a;
    checkCudaError(cudaMallocManaged((void **)&a, sizeof(double) * nx * ny));
    double *b;
    checkCudaError(cudaMallocManaged((void **)&b, sizeof(double) * nx * ny));
    double *c;
    checkCudaError(cudaMallocManaged((void **)&c, sizeof(double) * nx * ny));

    // init
    initMatrixAdd(a, b, c, nx, ny);

    cudaMemLocation deviceLocation{cudaMemLocationTypeDevice, 0};
    checkCudaError(cudaMemPrefetchAsync(a, sizeof(double) * nx * ny, deviceLocation, 0));
    checkCudaError(cudaMemPrefetchAsync(b, sizeof(double) * nx * ny, deviceLocation, 0));
    checkCudaError(cudaMemPrefetchAsync(c, sizeof(double) * nx * ny, deviceLocation, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(a, b, c, nx, ny);
        std::swap(c, a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(a, b, c, nx, ny);
        std::swap(c, a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    cudaMemLocation hostLocation{cudaMemLocationTypeHost, 0};
    checkCudaError(cudaMemPrefetchAsync(a, sizeof(double) * nx * ny, hostLocation, 0));
    checkCudaError(cudaMemPrefetchAsync(b, sizeof(double) * nx * ny, hostLocation, 0));
    checkCudaError(cudaMemPrefetchAsync(c, sizeof(double) * nx * ny, hostLocation, 0));

    // check solution
    checkSolutionMatrixAdd(a, b, c, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(a));
    checkCudaError(cudaFree(b));
    checkCudaError(cudaFree(c));

    return 0;
}
