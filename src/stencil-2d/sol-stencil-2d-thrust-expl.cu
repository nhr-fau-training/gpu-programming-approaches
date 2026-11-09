#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>


#include "stencil-2d-util.h"


inline void stencil2d(const thrust::device_vector<double> &u, thrust::device_vector<double> &uNew, size_t nx, size_t ny) {
    const double *u_ptr = thrust::raw_pointer_cast(u.data());
    double *uNew_ptr = thrust::raw_pointer_cast(uNew.data());
    thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx * ny),
                     [=] __host__ __device__ (size_t idx) {
                         const size_t i0 = idx % nx;
                         const size_t i1 = idx / nx;
                         if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
                             uNew_ptr[i0 + i1 * nx] = 0.25 * u_ptr[i0 + i1 * nx + 1] + 0.25 * u_ptr[i0 + i1 * nx - 1] + 0.25 * u_ptr[i0 + nx * (i1 + 1)] + 0.25 * u_ptr[i0 + nx * (i1 - 1)];
                         }
                     });
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    thrust::host_vector<double> u(nx * ny);
    thrust::device_vector<double> d_u(nx * ny);
    thrust::host_vector<double> uNew(nx * ny);
    thrust::device_vector<double> d_uNew(nx * ny);

    // init
    initStencil2D(u.data(), uNew.data(), nx, ny);

    thrust::copy(u.begin(), u.end(), d_u.begin());
    thrust::copy(uNew.begin(), uNew.end(), d_uNew.begin());

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d(d_u, d_uNew, nx, ny);
        std::swap(d_u, d_uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d(d_u, d_uNew, nx, ny);
        std::swap(d_u, d_uNew);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double), 7);

    thrust::copy(d_u.begin(), d_u.end(), u.begin());
    thrust::copy(d_uNew.begin(), d_uNew.end(), uNew.begin());

    // check solution
    checkSolutionStencil2D(u.data(), uNew.data(), nx, ny, nIt + nItWarmUp);

    return 0;
}
