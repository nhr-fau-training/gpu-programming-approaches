#include <cstdio>
#include <cstdlib>

#include <openacc.h>


int main(int argc, char *argv[]) {
    size_t nx = argc > 1 ? atoi(argv[1]) : 4;

    int numDevices = acc_get_num_devices(acc_device_nvidia);
    printf("Found %d GPU(s)\n", numDevices);

    for (int d = 0; d < numDevices; ++d) {
        // all following compute and data constructs use this GPU
        acc_set_device_num(d, acc_device_nvidia);

#pragma acc parallel loop
        for (size_t i0 = 0; i0 < nx; ++i0) {
            printf("Hello from GPU %d, thread %ld\n", d, i0);
        }
    }

    return 0;
}
