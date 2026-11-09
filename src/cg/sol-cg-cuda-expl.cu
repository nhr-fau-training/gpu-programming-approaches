#include <cub/cub.cuh>

#include "cg-util.h"

#include "../cuda-util.h"


constexpr unsigned int blockSizeX = 16;
constexpr unsigned int blockSizeY = 16;


__global__ void computeResidual(const double *const __restrict__ rhs, const double *const __restrict__ u, double *__restrict__ res, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        res[j * nx + i] = rhs[j * nx + i] - (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
    }
}

__global__ void initP(const double *const __restrict__ res, double *__restrict__ p, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        p[j * nx + i] = res[j * nx + i];
    }
}

__global__ void computeAp(const double *const __restrict__ p, double *__restrict__ ap, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        ap[j * nx + i] = 4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
    }
}

__global__ void updateSolution(const double alpha, const double *const __restrict__ p, double *__restrict__ u, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        u[j * nx + i] += alpha * p[j * nx + i];
    }
}

__global__ void updateResidual(const double alpha, const double *const __restrict__ ap, double *__restrict__ res, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        res[j * nx + i] -= alpha * ap[j * nx + i];
    }
}

__global__ void updateP(const double beta, const double *const __restrict__ res, double *__restrict__ p, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
    }
}

__global__ void dot(const double *const __restrict__ a, const double *const __restrict__ b, double *__restrict__ sum, const size_t nx, const size_t ny) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    // block reduction over the 2D thread block, see reductions.ipynb
    typedef cub::BlockReduce<double, blockSizeX, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, blockSizeY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    double elem = 0;
    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        elem = a[j * nx + i] * b[j * nx + i];
    }

    // all threads of the block must participate
    const double blockSum = BlockReduce(tempStorage).Sum(elem);

    // one atomic per block
    if (0 == threadIdx.x && 0 == threadIdx.y) {
        atomicAdd(sum, blockSum);
    }
}


inline double dotProduct(const double *const __restrict__ a, const double *const __restrict__ b, double *__restrict__ d_sum, const size_t nx, const size_t ny,
                         const dim3 &numBlocks, const dim3 &blockSize) {
    checkCudaError(cudaMemset(d_sum, 0, sizeof(double)));
    dot<<<numBlocks, blockSize>>>(a, b, d_sum, nx, ny);

    // cudaMemcpy waits for the kernel to finish
    double sum;
    checkCudaError(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    return sum;
}


inline size_t conjugateGradient(const double *const __restrict__ rhs, double *__restrict__ u, double *__restrict__ res,
                                double *__restrict__ p, double *__restrict__ ap, double *__restrict__ d_sum,
                                const size_t nx, const size_t ny, const size_t maxIt) {
    const dim3 blockSize(blockSizeX, blockSizeY);
    const dim3 numBlocks(ceilingDivide(nx - 1, blockSize.x), ceilingDivide(ny - 1, blockSize.y));

    // initialization
    double initResSq = 0;

    // compute initial residual
    computeResidual<<<numBlocks, blockSize>>>(rhs, u, res, nx, ny);

    // compute residual norm
    initResSq = dotProduct(res, res, d_sum, nx, ny, numBlocks, blockSize);

    // init p
    initP<<<numBlocks, blockSize>>>(res, p, nx, ny);

    double curResSq = initResSq;

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        // compute A * p
        computeAp<<<numBlocks, blockSize>>>(p, ap, nx, ny);

        double alphaNominator = curResSq;
        double alphaDenominator = dotProduct(p, ap, d_sum, nx, ny, numBlocks, blockSize);
        double alpha = alphaNominator / alphaDenominator;

        // update solution
        updateSolution<<<numBlocks, blockSize>>>(alpha, p, u, nx, ny);

        // update residual
        updateResidual<<<numBlocks, blockSize>>>(alpha, ap, res, nx, ny);

        // compute residual norm
        double nextResSq = dotProduct(res, res, d_sum, nx, ny, numBlocks, blockSize);

        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        double beta = nextResSq / curResSq;
        curResSq = nextResSq;

        // update p
        updateP<<<numBlocks, blockSize>>>(beta, res, p, nx, ny);
    }

    return maxIt;
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *u;
    checkCudaError(cudaMallocHost((void **)&u, sizeof(double) * nx * ny));
    double *rhs;
    checkCudaError(cudaMallocHost((void **)&rhs, sizeof(double) * nx * ny));

    double *res;
    checkCudaError(cudaMallocHost((void **)&res, sizeof(double) * nx * ny));
    double *p;
    checkCudaError(cudaMallocHost((void **)&p, sizeof(double) * nx * ny));
    double *ap;
    checkCudaError(cudaMallocHost((void **)&ap, sizeof(double) * nx * ny));

    double *d_u;
    checkCudaError(cudaMalloc((void **)&d_u, sizeof(double) * nx * ny));
    double *d_rhs;
    checkCudaError(cudaMalloc((void **)&d_rhs, sizeof(double) * nx * ny));

    double *d_res;
    checkCudaError(cudaMalloc((void **)&d_res, sizeof(double) * nx * ny));
    double *d_p;
    checkCudaError(cudaMalloc((void **)&d_p, sizeof(double) * nx * ny));
    double *d_ap;
    checkCudaError(cudaMalloc((void **)&d_ap, sizeof(double) * nx * ny));

    // result of the dot products
    double *d_sum;
    checkCudaError(cudaMalloc((void **)&d_sum, sizeof(double)));

    // init
    initConjugateGradient(u, rhs, nx, ny);

    memset(res, 0, nx * ny * sizeof(double));
    memset(p, 0, nx * ny * sizeof(double));
    memset(ap, 0, nx * ny * sizeof(double));

    checkCudaError(cudaMemcpy(d_u, u, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_rhs, rhs, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    checkCudaError(cudaMemcpy(d_res, res, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_p, p, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_ap, ap, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    // warm-up
    nItWarmUp = conjugateGradient(d_rhs, d_u, d_res, d_p, d_ap, d_sum, nx, ny, nItWarmUp);
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(d_rhs, d_u, d_res, d_p, d_ap, d_sum, nx, ny, nIt);
    checkCudaError(cudaDeviceSynchronize(), true);
    std::cout << "  CG steps:      " << nIt << std::endl;

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, 8 * sizeof(double), 15);

    checkCudaError(cudaMemcpy(u, d_u, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionConjugateGradient(u, rhs, nx, ny);

    checkCudaError(cudaFree(d_u));
    checkCudaError(cudaFree(d_rhs));

    checkCudaError(cudaFree(d_res));
    checkCudaError(cudaFree(d_p));
    checkCudaError(cudaFree(d_ap));

    checkCudaError(cudaFree(d_sum));

    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(rhs));

    checkCudaError(cudaFreeHost(res));
    checkCudaError(cudaFreeHost(p));
    checkCudaError(cudaFreeHost(ap));

    return 0;
}
