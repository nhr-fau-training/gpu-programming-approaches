#include <Kokkos_Core.hpp>

#include "stream-util.h"


inline void stream(const Kokkos::View<double *> &src, Kokkos::View<double *> &dest, size_t nx) {
    Kokkos::parallel_for(                //
        Kokkos::RangePolicy<>(0, nx),    //
        KOKKOS_LAMBDA(const size_t i0) { //
            dest(i0) = src(i0) + 1;
        });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double *> dest("dest", nx);
        Kokkos::View<double *> src("src", nx);

        auto h_dest = Kokkos::create_mirror_view(dest);
        auto h_src = Kokkos::create_mirror_view(src);

        // init
        initStream(h_dest.data(), h_src.data(), nx);

        Kokkos::deep_copy(dest, h_dest);
        Kokkos::deep_copy(src, h_src);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            stream(src, dest, nx);
            std::swap(src, dest);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            stream(src, dest, nx);
            std::swap(src, dest);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

        Kokkos::deep_copy(h_dest, dest);
        Kokkos::deep_copy(h_src, src);

        // check solution
        checkSolutionStream(h_dest.data(), h_src.data(), nx, nIt + nItWarmUp);
    }
    Kokkos::finalize();

    return 0;
}
