#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>


#include "cg-util.h"


// all algorithms iterate over the flat index of the full grid and recover the indices (i, j) from it;
// boundary elements are skipped (for_each) or contribute zero (transform_reduce)


inline size_t conjugateGradient(const thrust::device_vector<double> &rhs_vec, thrust::device_vector<double> &u_vec, thrust::device_vector<double> &res_vec,
                                thrust::device_vector<double> &p_vec, thrust::device_vector<double> &ap_vec,
                                const size_t nx, const size_t ny, const size_t maxIt) {
    const double *rhs = thrust::raw_pointer_cast(rhs_vec.data());
    double *u = thrust::raw_pointer_cast(u_vec.data());
    double *res = thrust::raw_pointer_cast(res_vec.data());
    double *p = thrust::raw_pointer_cast(p_vec.data());
    double *ap = thrust::raw_pointer_cast(ap_vec.data());

    const auto first = thrust::make_counting_iterator<size_t>(0);
    const auto last = thrust::make_counting_iterator<size_t>(nx * ny);

    // initialization
    double initResSq = 0;

    // compute initial residual
    thrust::for_each(thrust::device, first, last,
                     [=] __host__ __device__ (size_t idx) {
                         const size_t i = idx % nx;
                         const size_t j = idx / nx;
                         if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                             res[j * nx + i] = rhs[j * nx + i] - (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
                         }
                     });

    // compute residual norm
    initResSq = thrust::transform_reduce(thrust::device, first, last,
                                         [=] __host__ __device__ (size_t idx) {
                                             const size_t i = idx % nx;
                                             const size_t j = idx / nx;
                                             if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                                 return res[j * nx + i] * res[j * nx + i];
                                             }
                                             return 0.;
                                         },
                                         0., thrust::plus<double>());

    // init p
    thrust::for_each(thrust::device, first, last,
                     [=] __host__ __device__ (size_t idx) {
                         const size_t i = idx % nx;
                         const size_t j = idx / nx;
                         if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                             p[j * nx + i] = res[j * nx + i];
                         }
                     });

    double curResSq = initResSq;

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        // compute A * p
        thrust::for_each(thrust::device, first, last,
                         [=] __host__ __device__ (size_t idx) {
                             const size_t i = idx % nx;
                             const size_t j = idx / nx;
                             if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                 ap[j * nx + i] = 4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
                             }
                         });
        double alphaNominator = curResSq;
        double alphaDenominator = thrust::transform_reduce(thrust::device, first, last,
                                                           [=] __host__ __device__ (size_t idx) {
                                                               const size_t i = idx % nx;
                                                               const size_t j = idx / nx;
                                                               if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                                                   return p[j * nx + i] * ap[j * nx + i];
                                                               }
                                                               return 0.;
                                                           },
                                                           0., thrust::plus<double>());
        double alpha = alphaNominator / alphaDenominator;

        // update solution
        thrust::for_each(thrust::device, first, last,
                         [=] __host__ __device__ (size_t idx) {
                             const size_t i = idx % nx;
                             const size_t j = idx / nx;
                             if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                 u[j * nx + i] += alpha * p[j * nx + i];
                             }
                         });

        // update residual
        thrust::for_each(thrust::device, first, last,
                         [=] __host__ __device__ (size_t idx) {
                             const size_t i = idx % nx;
                             const size_t j = idx / nx;
                             if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                 res[j * nx + i] -= alpha * ap[j * nx + i];
                             }
                         });

        // compute residual norm
        double nextResSq = thrust::transform_reduce(thrust::device, first, last,
                                                    [=] __host__ __device__ (size_t idx) {
                                                        const size_t i = idx % nx;
                                                        const size_t j = idx / nx;
                                                        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                                            return res[j * nx + i] * res[j * nx + i];
                                                        }
                                                        return 0.;
                                                    },
                                                    0., thrust::plus<double>());

        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        double beta = nextResSq / curResSq;
        curResSq = nextResSq;

        // update p
        thrust::for_each(thrust::device, first, last,
                         [=] __host__ __device__ (size_t idx) {
                             const size_t i = idx % nx;
                             const size_t j = idx / nx;
                             if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                                 p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
                             }
                         });
    }

    return maxIt;
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    thrust::host_vector<double> u(nx * ny);
    thrust::device_vector<double> d_u(nx * ny);
    thrust::host_vector<double> rhs(nx * ny);
    thrust::device_vector<double> d_rhs(nx * ny);

    // device_vector initializes its elements to zero, which replaces the memset of the base version
    thrust::device_vector<double> d_res(nx * ny);
    thrust::device_vector<double> d_p(nx * ny);
    thrust::device_vector<double> d_ap(nx * ny);

    // init
    initConjugateGradient(u.data(), rhs.data(), nx, ny);

    thrust::copy(u.begin(), u.end(), d_u.begin());
    thrust::copy(rhs.begin(), rhs.end(), d_rhs.begin());

    // warm-up
    nItWarmUp = conjugateGradient(d_rhs, d_u, d_res, d_p, d_ap, nx, ny, nItWarmUp);

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(d_rhs, d_u, d_res, d_p, d_ap, nx, ny, nIt);
    std::cout << "  CG steps:      " << nIt << std::endl;

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, 8 * sizeof(double), 15);

    thrust::copy(d_u.begin(), d_u.end(), u.begin());

    // check solution
    checkSolutionConjugateGradient(u.data(), rhs.data(), nx, ny);

    return 0;
}
