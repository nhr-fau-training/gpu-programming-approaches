#include <cstdio>
#include <cstdlib>

#include <omp.h>


int main(int argc, char *argv[]) {
    size_t nx = argc > 1 ? atoi(argv[1]) : 4;

    int numDevices = omp_get_num_devices();
    printf("Found %d GPU(s)\n", numDevices);

    for (int d = 0; d < numDevices; ++d) {
        // the device clause selects the GPU for this target region
#pragma omp target teams distribute parallel for device(d)
        for (size_t i0 = 0; i0 < nx; ++i0) {
            printf("Hello from GPU %d, thread %ld\n", d, i0);
        }
    }

    return 0;
}
