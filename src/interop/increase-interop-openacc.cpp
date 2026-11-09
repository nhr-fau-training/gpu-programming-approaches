#include "../increase/increase-util.h"

#include "increase-cuda.h"


inline void init(double *data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}


inline void increase(double *data, size_t nx) {
#pragma acc parallel loop present(data[0 : nx])
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] += 1;
    }
}


inline void increaseInterop(double *data, size_t nx) {
    // inside the region, data holds the device address of the present array
#pragma acc host_data use_device(data)
    {
        increaseCuda(data, nx);
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto startGlobal = std::chrono::steady_clock::now();

    double *data;
    data = new double[nx];

    // init
    init(data, nx);

#pragma acc enter data copyin(data[0 : nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        if (0 == i % 2)
            increase(data, nx);
        else
            increaseInterop(data, nx);
    }

    // measurement
    auto startWork = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        if (0 == i % 2)
            increase(data, nx);
        else
            increaseInterop(data, nx);
    }

    auto stopWork = std::chrono::steady_clock::now();

    printStats(stopWork - startWork, nIt, nx, sizeof(double) + sizeof(double), 1);

#pragma acc exit data copyout(data[0 : nx])

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    delete[] data;

    auto stopGlobal = std::chrono::steady_clock::now();

    printTotalStats(stopGlobal - startGlobal, stopWork - startWork);

    return 0;
}
