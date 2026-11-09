#include "stream-util.h"

#include "../sycl-util.h"


inline void stream(sycl::queue &q, const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                dest[i0] = src[i0] + 1;
            }
        });
    });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    double *dest;
    dest = sycl::malloc_host<double>(nx, q);
    double *src;
    src = sycl::malloc_host<double>(nx, q);

    double *d_dest;
    d_dest = sycl::malloc_device<double>(nx, q);
    double *d_src;
    d_src = sycl::malloc_device<double>(nx, q);

    // init
    initStream(dest, src, nx);

    q.memcpy(d_dest, dest, sizeof(double) * nx);
    q.memcpy(d_src, src, sizeof(double) * nx);
    q.wait();

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(q, d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(q, d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    q.memcpy(dest, d_dest, sizeof(double) * nx);
    q.memcpy(src, d_src, sizeof(double) * nx);
    q.wait();

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    sycl::free(d_dest, q);
    sycl::free(d_src, q);

    sycl::free(dest, q);
    sycl::free(src, q);

    return 0;
}
