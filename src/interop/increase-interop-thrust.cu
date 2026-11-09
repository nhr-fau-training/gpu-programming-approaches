#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

#include "../increase/increase-util.h"

#include "increase-cuda.h"


inline void init(thrust::host_vector<double> &data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}


inline void increase(thrust::device_vector<double> &data, size_t nx) {
    double *data_ptr = thrust::raw_pointer_cast(data.data());
    thrust::tabulate(thrust::device, data.begin(), data.end(),
                     [=] __host__ __device__ (size_t i0) {
                         return data_ptr[i0] + 1;
                     });
}


inline void increaseInterop(thrust::device_vector<double> &data, size_t nx) {
    // the raw pointer behind a device_vector is a regular CUDA device pointer
    increaseCuda(thrust::raw_pointer_cast(data.data()), nx);
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto startGlobal = std::chrono::steady_clock::now();

    thrust::host_vector<double> data(nx);
    thrust::device_vector<double> d_data(nx);

    // init
    init(data, nx);

    thrust::copy(data.begin(), data.end(), d_data.begin());

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        if (0 == i % 2)
            increase(d_data, nx);
        else
            increaseInterop(d_data, nx);
    }

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        if (0 == i % 2)
            increase(d_data, nx);
        else
            increaseInterop(d_data, nx);
    }

    auto stopWork = std::chrono::steady_clock::now();

    printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

    thrust::copy(d_data.begin(), d_data.end(), data.begin());

    // check solution
    checkSolutionIncrease(data.data(), nx, nIt + nItWarmUp);

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
