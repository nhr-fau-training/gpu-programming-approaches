#include "cg-util.h"


inline size_t conjugateGradient(const double *const __restrict__ rhs, double *__restrict__ u, double *__restrict__ res,
                                double *__restrict__ p, double *__restrict__ ap,
                                const size_t nx, const size_t ny, const size_t maxIt) {
    // initialization
    double initResSq = 0;

    // compute initial residual
#pragma acc parallel loop collapse(2) present(rhs[0 : nx * ny], u[0 : nx * ny], res[0 : nx * ny])
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            res[j * nx + i] = rhs[j * nx + i] - (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] + u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
        }
    }

    // compute residual norm
#pragma acc parallel loop collapse(2) present(res[0 : nx * ny]) reduction(+ : initResSq)
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            initResSq += res[j * nx + i] * res[j * nx + i];
        }
    }

    // init p
#pragma acc parallel loop collapse(2) present(res[0 : nx * ny], p[0 : nx * ny])
    for (size_t j = 1; j < ny - 1; ++j) {
        for (size_t i = 1; i < nx - 1; ++i) {
            p[j * nx + i] = res[j * nx + i];
        }
    }

    double curResSq = initResSq;

    // main loop
    for (size_t it = 0; it < maxIt; ++it) {
        // compute A * p
#pragma acc parallel loop collapse(2) present(p[0 : nx * ny], ap[0 : nx * ny])
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                ap[j * nx + i] = 4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] + p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
            }
        }
        double alphaNominator = curResSq;
        double alphaDenominator = 0;
#pragma acc parallel loop collapse(2) present(p[0 : nx * ny], ap[0 : nx * ny]) reduction(+ : alphaDenominator)
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                alphaDenominator += p[j * nx + i] * ap[j * nx + i];
            }
        }
        double alpha = alphaNominator / alphaDenominator;

        // update solution
#pragma acc parallel loop collapse(2) present(p[0 : nx * ny], u[0 : nx * ny])
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                u[j * nx + i] += alpha * p[j * nx + i];
            }
        }

        // update residual
#pragma acc parallel loop collapse(2) present(ap[0 : nx * ny], res[0 : nx * ny])
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                res[j * nx + i] -= alpha * ap[j * nx + i];
            }
        }

        // compute residual norm
        double nextResSq = 0;
#pragma acc parallel loop collapse(2) present(res[0 : nx * ny]) reduction(+ : nextResSq)
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                nextResSq += res[j * nx + i] * res[j * nx + i];
            }
        }

        // check exit criterion
        if (sqrt(nextResSq) <= 1e-12)
            return it;

        // if (0 == it % 100)
        //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

        // compute beta
        double beta = nextResSq / curResSq;
        curResSq = nextResSq;

        // update p
#pragma acc parallel loop collapse(2) present(res[0 : nx * ny], p[0 : nx * ny])
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
            }
        }
    }

    return maxIt;
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *u;
    u = new double[nx * ny];
    double *rhs;
    rhs = new double[nx * ny];

    double *res;
    res = new double[nx * ny];
    double *p;
    p = new double[nx * ny];
    double *ap;
    ap = new double[nx * ny];

    // init
    initConjugateGradient(u, rhs, nx, ny);

    memset(res, 0, nx * ny * sizeof(double));
    memset(p, 0, nx * ny * sizeof(double));
    memset(ap, 0, nx * ny * sizeof(double));

#pragma acc enter data copyin(u[0 : nx * ny], rhs[0 : nx * ny])
#pragma acc enter data copyin(res[0 : nx * ny], p[0 : nx * ny], ap[0 : nx * ny])

    // warm-up
    nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);
    std::cout << "  CG steps:      " << nIt << std::endl;

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, 8 * sizeof(double), 15);

#pragma acc exit data copyout(u[0 : nx * ny]) delete(rhs[0 : nx * ny])
#pragma acc exit data delete(res[0 : nx * ny], p[0 : nx * ny], ap[0 : nx * ny])

    // check solution
    checkSolutionConjugateGradient(u, rhs, nx, ny);

    delete[] u;
    delete[] rhs;

    delete[] res;
    delete[] p;
    delete[] ap;

    return 0;
}
