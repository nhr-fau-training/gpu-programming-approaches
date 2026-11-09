#include "cg-util.h"

#include "../sycl-util.h"


// 2D launch configuration covering all inner points, as in the stencil-2d solution
inline sycl::nd_range<2> innerRange(const size_t nx, const size_t ny) {
    return sycl::nd_range<2>(sycl::range<2>(ceilToMultipleOf(ny - 1, 16), ceilToMultipleOf(nx - 1, 16)), sycl::range<2>(16, 16));
}

// dot product of a and b over all inner points
inline double dot(sycl::queue &q, const double *const __restrict__ a, const double *const __restrict__ b, double *__restrict__ d_sum,
                  const size_t nx, const size_t ny) {
    q.memset(d_sum, 0, sizeof(double));

    q.submit([&](sycl::handler &h) {
        auto reduce = sycl::reduction(d_sum, sycl::plus<double>());
        h.parallel_for(sycl::range<2>(ny, nx), reduce, [=](auto item, auto &acc) {
            const size_t i = item[1];
            const size_t j = item[0];

            if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                acc += a[j * nx + i] * b[j * nx + i];
            }
        });
    });

    double sum;
    q.memcpy(&sum, d_sum, sizeof(double));
    q.wait();

    return sum;
}


inline size_t conjugateGradient(sycl::queue &q, const double *const __restrict__ rhs, double *__restrict__ u, double *__restrict__ res,
                                double *__restrict__ p, double *__restrict__ ap, double *__restrict__ d_sum,
                                const size_t nx, const size_t ny, const size_t maxIt) {
    // initialization
    double initResSq = 0;

    // compute initial residual
    q.submit([&](sycl::handler &h) {
        h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
            const size_t i = item.get_global_id(1);
            const size_t j = item.get_global_id(0);

            if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                res[j * nx + i] = rhs[j * nx + i] - (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
            }
        });
    });

    // compute residual norm
    initResSq = dot(q, res, res, d_sum, nx, ny);

    // init p
    q.submit([&](sycl::handler &h) {
        h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
            const size_t i = item.get_global_id(1);
            const size_t j = item.get_global_id(0);

            if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                p[j * nx + i] = res[j * nx + i];
            }
        });
    });

    double curResSq = initResSq;

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        // compute A * p
        q.submit([&](sycl::handler &h) {
            h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
                const size_t i = item.get_global_id(1);
                const size_t j = item.get_global_id(0);

                if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                    ap[j * nx + i] = 4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
                }
            });
        });
        double alphaNominator = curResSq;
        double alphaDenominator = dot(q, p, ap, d_sum, nx, ny);
        double alpha = alphaNominator / alphaDenominator;

        // update solution
        q.submit([&](sycl::handler &h) {
            h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
                const size_t i = item.get_global_id(1);
                const size_t j = item.get_global_id(0);

                if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                    u[j * nx + i] += alpha * p[j * nx + i];
                }
            });
        });

        // update residual
        q.submit([&](sycl::handler &h) {
            h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
                const size_t i = item.get_global_id(1);
                const size_t j = item.get_global_id(0);

                if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                    res[j * nx + i] -= alpha * ap[j * nx + i];
                }
            });
        });

        // compute residual norm
        double nextResSq = dot(q, res, res, d_sum, nx, ny);

        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        double beta = nextResSq / curResSq;
        curResSq = nextResSq;

        // update p
        q.submit([&](sycl::handler &h) {
            h.parallel_for(innerRange(nx, ny), [=](sycl::nd_item<2> item) {
                const size_t i = item.get_global_id(1);
                const size_t j = item.get_global_id(0);

                if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
                    p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
                }
            });
        });
    }

    return maxIt;
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    double *u;
    u = sycl::malloc_host<double>(nx * ny, q);
    double *rhs;
    rhs = sycl::malloc_host<double>(nx * ny, q);

    double *d_u;
    d_u = sycl::malloc_device<double>(nx * ny, q);
    double *d_rhs;
    d_rhs = sycl::malloc_device<double>(nx * ny, q);

    double *d_res;
    d_res = sycl::malloc_device<double>(nx * ny, q);
    double *d_p;
    d_p = sycl::malloc_device<double>(nx * ny, q);
    double *d_ap;
    d_ap = sycl::malloc_device<double>(nx * ny, q);

    // result of the dot products
    double *d_sum;
    d_sum = sycl::malloc_device<double>(1, q);

    // init
    initConjugateGradient(u, rhs, nx, ny);

    q.memcpy(d_u, u, sizeof(double) * nx * ny);
    q.memcpy(d_rhs, rhs, sizeof(double) * nx * ny);

    // set directly on the device instead of memset on the host and copy
    setFieldToZero(q, nx, ny, d_res);
    setFieldToZero(q, nx, ny, d_p);
    setFieldToZero(q, nx, ny, d_ap);
    q.wait();

    // warm-up
    nItWarmUp = conjugateGradient(q, d_rhs, d_u, d_res, d_p, d_ap, d_sum, nx, ny, nItWarmUp);
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(q, d_rhs, d_u, d_res, d_p, d_ap, d_sum, nx, ny, nIt);
    q.wait();
    std::cout << "  CG steps:      " << nIt << std::endl;

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, 8 * sizeof(double), 15);

    q.memcpy(u, d_u, sizeof(double) * nx * ny);
    q.wait();

    // check solution
    checkSolutionConjugateGradient(u, rhs, nx, ny);

    sycl::free(d_u, q);
    sycl::free(d_rhs, q);

    sycl::free(d_res, q);
    sycl::free(d_p, q);
    sycl::free(d_ap, q);

    sycl::free(d_sum, q);

    sycl::free(u, q);
    sycl::free(rhs, q);

    return 0;
}
