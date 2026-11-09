#pragma once

#include <cstring>

#include "../util.h"


void initConjugateGradient(double *__restrict__ u, double *__restrict__ rhs, const size_t nx, const size_t ny, bool layoutRight = false) {
    constexpr auto app = 1;

    switch (app) {
        case 0:
        // version 0 - zero boundary, unit inner values
        forEachCell2d(nx, ny, 0, nx, 0, ny, layoutRight, [&](size_t i0, size_t i1, size_t s0, size_t s1) {
            if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                u[i1 * s1 + i0 * s0] = 0;
            else
                u[i1 * s1 + i0 * s0] = 1;

            rhs[i1 * s1 + i0 * s0] = 0;
        });

        break;

        case 1:
        // version 1 - simple trigonometric test problem
        forEachCell2d(nx, ny, 0, nx, 0, ny, layoutRight, [&](size_t i0, size_t i1, size_t s0, size_t s1) {
            auto x = i0 - 1 + 0.5;
            auto y = i1 - 1 + 0.5;

            if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                u[i1 * s1 + i0 * s0] = sin(1e-3 * x) * cos(2e-3 * y);
            else
                u[i1 * s1 + i0 * s0] = 1;
            rhs[i1 * s1 + i0 * s0] = 5e-6 * sin(1e-3 * x) * cos(2e-3 * y);
        });
        break;
    }
}

void checkSolutionConjugateGradient(const double *const __restrict__ u, const double *const __restrict__ rhs, const size_t nx, const size_t ny, bool layoutRight = false) {
    double res = 0;
    forEachCell2d(nx, ny, 1, nx - 1, 1, ny - 1, layoutRight, [&](size_t i0, size_t i1, size_t s0, size_t s1) {
        const double localRes = rhs[i1 * s1 + i0 * s0] - (
                4 * u[i1 * s1 + i0 * s0] - (u[i1 * s1 + (i0 - 1) * s0] + u[i1 * s1 + (i0 + 1) * s0] + u[(i1 - 1) * s1 + i0 * s0] + u[(i1 + 1) * s1 + i0 * s0]));

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
