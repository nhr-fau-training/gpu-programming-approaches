#include "stencil-2d-util.h"

#include "../cuda-util.h"


__global__ void stencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *u;
    checkCudaError(cudaMallocHost((void **)&u, sizeof(double) * nx * ny));
    double *uNew;
    checkCudaError(cudaMallocHost((void **)&uNew, sizeof(double) * nx * ny));

    double *d_u;
    checkCudaError(cudaMalloc((void **)&d_u, sizeof(double) * nx * ny));
    double *d_uNew;
    checkCudaError(cudaMalloc((void **)&d_uNew, sizeof(double) * nx * ny));

    // init
    initStencil2D(u, uNew, nx, ny);

    checkCudaError(cudaMemcpy(d_u, u, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_uNew, uNew, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    const dim3 blockSize(16, 16);
    const dim3 numBlocks(ceilingDivide(nx - 1, blockSize.x), ceilingDivide(ny - 1, blockSize.y));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(d_u, d_uNew, nx, ny);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(d_u, d_uNew, nx, ny);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double), 7);

    checkCudaError(cudaMemcpy(u, d_u, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(uNew, d_uNew, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_u));
    checkCudaError(cudaFree(d_uNew));

    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(uNew));

    return 0;
}
