#pragma once

#include <cstddef>


// Launches the CUDA increase kernel on device memory and waits for its completion.
// Implemented in increase-cuda.cu, which is compiled with nvcc and can be linked with all approaches.
// This header deliberately contains no CUDA types, so any compiler can include it.
void increaseCuda(double *d_data, size_t nx);
