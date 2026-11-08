#include "increase-util.h"

#include "../sycl-util.h"


inline void init(double *data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}


inline void increase(sycl::queue &q, double *data, size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                data[i0] += 1;
            }
        });
    });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto startGlobal = std::chrono::steady_clock::now();

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    double *data;
    data = sycl::malloc_shared<double>(nx, q);

    // init
    init(data, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase(q, data, nx);
    }
    q.wait();

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(q, data, nx);
    }
    q.wait();

    auto stopWork = std::chrono::steady_clock::now();

    printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    sycl::free(data, q);

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
