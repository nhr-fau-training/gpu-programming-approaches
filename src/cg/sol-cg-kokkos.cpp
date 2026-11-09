#include <Kokkos_Core.hpp>

#include "cg-util.h"


inline size_t conjugateGradient(const Kokkos::View<double **> &rhs, Kokkos::View<double **> &u, Kokkos::View<double **> &res,
                                Kokkos::View<double **> &p, Kokkos::View<double **> &ap,
                                const size_t nx, const size_t ny, const size_t maxIt) {
    // all inner points
    const auto inner = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::Schedule<Kokkos::Static>>({1, 1}, {nx - 1, ny - 1});

    // initialization
    double initResSq = 0;

    // compute initial residual
    Kokkos::parallel_for(                                //
        inner,                                           //
        KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
            res(i, j) = rhs(i, j) - (4 * u(i, j) - (u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1)));
        });

    // compute residual norm
    Kokkos::parallel_reduce(                                         //
        inner,                                                       //
        KOKKOS_LAMBDA(const size_t i, const size_t j, double &acc) { //
            acc += res(i, j) * res(i, j);
        },
        initResSq);

    // init p
    Kokkos::parallel_for(                                //
        inner,                                           //
        KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
            p(i, j) = res(i, j);
        });

    double curResSq = initResSq;

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        // compute A * p
        Kokkos::parallel_for(                                //
            inner,                                           //
            KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
                ap(i, j) = 4 * p(i, j) - (p(i - 1, j) + p(i + 1, j) + p(i, j - 1) + p(i, j + 1));
            });
        double alphaNominator = curResSq;
        double alphaDenominator = 0;
        Kokkos::parallel_reduce(                                         //
            inner,                                                       //
            KOKKOS_LAMBDA(const size_t i, const size_t j, double &acc) { //
                acc += p(i, j) * ap(i, j);
            },
            alphaDenominator);
        double alpha = alphaNominator / alphaDenominator;

        // update solution
        Kokkos::parallel_for(                                //
            inner,                                           //
            KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
                u(i, j) += alpha * p(i, j);
            });

        // update residual
        Kokkos::parallel_for(                                //
            inner,                                           //
            KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
                res(i, j) -= alpha * ap(i, j);
            });

        // compute residual norm
        double nextResSq = 0;
        Kokkos::parallel_reduce(                                         //
            inner,                                                       //
            KOKKOS_LAMBDA(const size_t i, const size_t j, double &acc) { //
                acc += res(i, j) * res(i, j);
            },
            nextResSq);

        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        double beta = nextResSq / curResSq;
        curResSq = nextResSq;

        // update p
        Kokkos::parallel_for(                                //
            inner,                                           //
            KOKKOS_LAMBDA(const size_t i, const size_t j) {  //
                p(i, j) = res(i, j) + beta * p(i, j);
            });
    }

    return maxIt;
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double **> u("u", nx, ny);
        Kokkos::View<double **> rhs("rhs", nx, ny);

        // views are initialized to zero, which replaces the memset of the base version
        Kokkos::View<double **> res("res", nx, ny);
        Kokkos::View<double **> p("p", nx, ny);
        Kokkos::View<double **> ap("ap", nx, ny);

        auto h_u = Kokkos::create_mirror_view(u);
        auto h_rhs = Kokkos::create_mirror_view(rhs);

        // the host mirrors share the layout of the views, which depends on the backend (LayoutLeft on CUDA, LayoutRight on host);
        constexpr bool layoutRight = std::is_same_v<decltype(h_u)::array_layout, Kokkos::LayoutRight>;

        // init
        initConjugateGradient(h_u.data(), h_rhs.data(), nx, ny, layoutRight);

        Kokkos::deep_copy(u, h_u);
        Kokkos::deep_copy(rhs, h_rhs);

        // warm-up
        nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);
        Kokkos::fence();
        std::cout << "  CG steps:      " << nIt << std::endl;

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx * ny, 8 * sizeof(double), 15);

        Kokkos::deep_copy(h_u, u);

        // check solution
        checkSolutionConjugateGradient(h_u.data(), h_rhs.data(), nx, ny, layoutRight);
    }
    Kokkos::finalize();

    return 0;
}
