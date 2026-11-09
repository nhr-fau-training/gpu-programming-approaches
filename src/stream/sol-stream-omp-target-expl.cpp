#include "stream-util.h"


inline void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
#pragma omp target teams distribute parallel for
    for (size_t i0 = 0; i0 < nx; ++i0) {
        dest[i0] = src[i0] + 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *dest;
    dest = new double[nx];
    double *src;
    src = new double[nx];

    // init
    initStream(dest, src, nx);

#pragma omp target enter data map(to : dest[0 : nx])
#pragma omp target enter data map(to : src[0 : nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

#pragma omp target exit data map(from : dest[0 : nx])
#pragma omp target exit data map(from : src[0 : nx])

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    delete[] dest;
    delete[] src;

    return 0;
}
