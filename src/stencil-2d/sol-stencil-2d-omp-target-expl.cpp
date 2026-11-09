#include "stencil-2d-util.h"


inline void stencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny) {
#pragma omp target teams distribute parallel for collapse(2)
    for (size_t i1 = 1; i1 < ny - 1; ++i1) {
        for (size_t i0 = 1; i0 < nx - 1; ++i0) {
            uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
        }
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *u;
    u = new double[nx * ny];
    double *uNew;
    uNew = new double[nx * ny];

    // init
    initStencil2D(u, uNew, nx, ny);

#pragma omp target enter data map(to : u[0 : nx * ny])
#pragma omp target enter data map(to : uNew[0 : nx * ny])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d(u, uNew, nx, ny);
        std::swap(u, uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d(u, uNew, nx, ny);
        std::swap(u, uNew);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double), 7);

#pragma omp target exit data map(from : u[0 : nx * ny])
#pragma omp target exit data map(from : uNew[0 : nx * ny])

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    delete[] u;
    delete[] uNew;

    return 0;
}
