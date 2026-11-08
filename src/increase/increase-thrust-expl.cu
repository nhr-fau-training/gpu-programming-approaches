#include <algorithm>
#include <execution>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/for_each.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>


#include "increase-util.h"


inline void init(thrust::host_vector<double> &data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}


inline void increase(thrust::device_vector<double> &data, size_t nx) {
    // // version 1: thrust transform
    // thrust::transform(thrust::device, data.begin(), data.end(), data.begin(),
    //                   [=] __host__ __device__ (double data_elem) {
    //                       return data_elem + 1;
    //                   });

    // // version 2: thrust for_each + counting iterator
    // double *data_ptr = thrust::raw_pointer_cast(data.data());
    // thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx),
    //                  [=] __host__ __device__ (size_t i0) {
    //                      data_ptr[i0] += 1;
    //                  });

    // version 3: thrust tabulate
    double *data_ptr = thrust::raw_pointer_cast(data.data());
    thrust::tabulate(thrust::device, data.begin(), data.end(),
                     [=] __host__ __device__ (size_t i0) {
                         return data_ptr[i0] + 1;
                     });
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
        increase(d_data, nx);
    }

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(d_data, nx);
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
