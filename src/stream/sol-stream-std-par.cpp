#include <algorithm>
#include <execution>

#include "stream-util.h"


inline void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    std::for_each(std::execution::par_unseq, src, src + nx, //
                  [=](const double &src_item) {             //
                      const size_t i0 = &src_item - src;
                      dest[i0] = src[i0] + 1;
                  });
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

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    delete[] dest;
    delete[] src;

    return 0;
}
