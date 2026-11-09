#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/tabulate.h>


#include "stream-util.h"


inline void stream(const thrust::device_vector<double> &src, thrust::device_vector<double> &dest, size_t nx) {
    const double *src_ptr = thrust::raw_pointer_cast(src.data());
    thrust::tabulate(thrust::device, dest.begin(), dest.end(),
                     [=] __host__ __device__ (size_t i0) {
                         return src_ptr[i0] + 1;
                     });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    thrust::host_vector<double> dest(nx);
    thrust::device_vector<double> d_dest(nx);
    thrust::host_vector<double> src(nx);
    thrust::device_vector<double> d_src(nx);

    // init
    initStream(dest.data(), src.data(), nx);

    thrust::copy(dest.begin(), dest.end(), d_dest.begin());
    thrust::copy(src.begin(), src.end(), d_src.begin());

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    thrust::copy(d_dest.begin(), d_dest.end(), dest.begin());
    thrust::copy(d_src.begin(), d_src.end(), src.begin());

    // check solution
    checkSolutionStream(dest.data(), src.data(), nx, nIt + nItWarmUp);

    return 0;
}
