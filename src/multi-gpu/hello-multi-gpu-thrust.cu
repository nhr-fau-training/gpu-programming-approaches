#include <cstdio>
#include <cstdlib>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../cuda-util.h"


int main(int argc, char *argv[]) {
    size_t nx = argc > 1 ? atoi(argv[1]) : 4;

    int numDevices;
    checkCudaError(cudaGetDeviceCount(&numDevices));
    printf("Found %d GPU(s)\n", numDevices);

    for (int d = 0; d < numDevices; ++d) {
        // Thrust runs on the current CUDA device
        checkCudaError(cudaSetDevice(d));

        thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx),
                         [=] __host__ __device__ (size_t i0) {
                             printf("Hello from GPU %d, thread %ld\n", d, i0);
                         });
    }

    return 0;
}
