#pragma once

#include "../util.h"


inline void checkSolutionIncrease(double *data, size_t nx, size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((double)(i0 + nIt) != data[i0]) {
            std::cerr << "Increase check failed for element " << i0 << " (expected " << i0 + nIt << " but got " << data[i0] << ")" << std::endl;
            return;
        }
    }
}

inline void parseCLA_1d(int argc, char **argv, size_t &nx, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 67108864;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
