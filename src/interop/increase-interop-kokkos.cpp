#include <Kokkos_Core.hpp>

#include "../increase/increase-util.h"

#include "increase-cuda.h"


inline void init(Kokkos::View<double *>::host_mirror_type &data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data(i0) = (double)i0;
    }
}


inline void increase(Kokkos::View<double *> &data, size_t nx) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, nx),
        KOKKOS_LAMBDA(const size_t i0) {
            data(i0) += 1;
        });
}


inline void increaseInterop(Kokkos::View<double *> &data, size_t nx) {
    // Kokkos kernels run asynchronously - make sure they are done before CUDA takes over
    Kokkos::fence();

    // with the CUDA backend, the raw pointer behind a View is a regular CUDA device pointer
    increaseCuda(data.data(), nx);
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto startGlobal = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point startWork, stopWork;

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double *> data("data", nx);

        auto h_data = Kokkos::create_mirror_view(data);

        // init
        init(h_data, nx);

        Kokkos::deep_copy(data, h_data);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            if (0 == i % 2)
                increase(data, nx);
            else
                increaseInterop(data, nx);
        }
        Kokkos::fence();

        // measurement
        startWork = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            if (0 == i % 2)
                increase(data, nx);
            else
                increaseInterop(data, nx);
        }
        Kokkos::fence();

        stopWork = std::chrono::steady_clock::now();

        printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

        Kokkos::deep_copy(h_data, data);

        // check solution
        checkSolutionIncrease(h_data.data(), nx, nIt + nItWarmUp);
    }
    Kokkos::finalize();

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
