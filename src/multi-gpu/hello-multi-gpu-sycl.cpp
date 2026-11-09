#include <cstdlib>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>


int main(int argc, char *argv[]) {
    size_t nx = argc > 1 ? atoi(argv[1]) : 4;

    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::cout << "Found " << gpus.size() << " GPU(s)" << std::endl;

    // one queue per GPU
    std::vector<sycl::queue> queues;
    for (auto &gpu : gpus)
        queues.emplace_back(gpu);

    // submit to all GPUs - submissions are asynchronous, so the GPUs work concurrently
    for (int d = 0; d < (int)queues.size(); ++d) {
        queues[d].submit([&](sycl::handler &h) {
            sycl::stream str(8192, 1024, h); // use sycl stream instead of printf
            h.parallel_for(nx, [=](auto i0) {
                str << "Hello from GPU " << d << ", thread " << (size_t)i0 << sycl::endl;
            });
        });
    }

    // wait for all GPUs
    for (auto &q : queues)
        q.wait();

    return 0;
}
