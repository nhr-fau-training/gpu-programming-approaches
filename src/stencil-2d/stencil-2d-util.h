#pragma once

#include "../util.h"


inline void initStencil2D(double *__restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny, bool layoutRight = false) {
    forEachCell2d(nx, ny, 0, nx, 0, ny, layoutRight, [&](size_t i0, size_t i1, size_t s0, size_t s1) {
        if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1) {
            u[i0 * s0 + i1 * s1] = 0;
            uNew[i0 * s0 + i1 * s1] = 0;
        } else {
            u[i0 * s0 + i1 * s1] = 1;
            uNew[i0 * s0 + i1 * s1] = 1;
        }
    });
}

inline void checkSolutionStencil2D(const double *const __restrict__ u, const double *const __restrict__ uNew, size_t nx, size_t ny, size_t nIt, bool layoutRight = false) {
    double res = 0;
    forEachCell2d(nx, ny, 1, nx - 1, 1, ny - 1, layoutRight, [&](size_t i0, size_t i1, size_t s0, size_t s1) {
        const double localRes = -u[(i0 + 1) * s0 + i1 * s1] - u[(i0 - 1) * s0 + i1 * s1] + 4 * u[i0 * s0 + i1 * s1] - u[i0 * s0 + (i1 + 1) * s1] - u[i0 * s0 + (i1 - 1) * s1];
        res += localRes * localRes;
    });

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}

inline void parseCLA_2d(int argc, char **argv, size_t &nx, size_t &ny, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 4096;
    ny = 4096;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        ny = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
