#include <Kokkos_Core.hpp>

#include "stencil-2d-util.h"


inline void stencil2d(const Kokkos::View<double **> &u, Kokkos::View<double **> &uNew, size_t nx, size_t ny) {
    Kokkos::parallel_for(                                                                                   //
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::Schedule<Kokkos::Static>>({1, 1}, {nx - 1, ny - 1}), //
        KOKKOS_LAMBDA(const size_t i0, const size_t i1) {                                                   //
            uNew(i0, i1) = 0.25 * u(i0 + 1, i1) + 0.25 * u(i0 - 1, i1) + 0.25 * u(i0, i1 + 1) + 0.25 * u(i0, i1 - 1);
        });
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double **> u("u", nx, ny);
        Kokkos::View<double **> uNew("uNew", nx, ny);

        auto h_u = Kokkos::create_mirror_view(u);
        auto h_uNew = Kokkos::create_mirror_view(uNew);

        // the host mirrors share the layout of the views, which depends on the backend (LayoutLeft on CUDA, LayoutRight on host);
        constexpr bool layoutRight = std::is_same_v<decltype(h_u)::array_layout, Kokkos::LayoutRight>;

        // init
        initStencil2D(h_u.data(), h_uNew.data(), nx, ny, layoutRight);

        Kokkos::deep_copy(u, h_u);
        Kokkos::deep_copy(uNew, h_uNew);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            stencil2d(u, uNew, nx, ny);
            std::swap(u, uNew);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            stencil2d(u, uNew, nx, ny);
            std::swap(u, uNew);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double), 7);

        Kokkos::deep_copy(h_u, u);
        Kokkos::deep_copy(h_uNew, uNew);

        // check solution
        checkSolutionStencil2D(h_u.data(), h_uNew.data(), nx, ny, nIt + nItWarmUp, layoutRight);
    }
    Kokkos::finalize();

    return 0;
}
