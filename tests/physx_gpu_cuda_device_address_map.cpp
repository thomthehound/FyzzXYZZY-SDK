// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_device_address_map.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using fyzz::physx_gpu::CudaDeviceAddressMap;

namespace {
void demand(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
template <typename E, typename Fn>
void rejected(Fn&& fn, const char* reason) {
    bool failed = false;
    try { fn(); }
    catch (const E&) { failed = true; }
    demand(failed, reason);
}
}

int main() {
    try {
        CudaDeviceAddressMap table;
        demand(table.live_allocations() == 0, "New CUDA mapping not empty");
        rejected<std::invalid_argument>(
            [&] { table.register_allocation(0, 64); }, "Zero buffer accepted");
        rejected<std::invalid_argument>(
            [&] { table.register_allocation(1, 0); }, "Zero allocation accepted");
        rejected<std::invalid_argument>(
            [&] { table.resolve(0, 4); }, "Null device pointer accepted");

        const auto a = table.register_allocation(101u, 257u);
        const auto b = table.register_allocation(202u, 1024u);
        demand(a >= (1ull << 32) && (a % 256ull) == 0, "CUDA virtual base");
        demand(b == a + 512u, "CUdeviceptr allocation stride must be aligned");
        demand(table.live_allocations() == 2, "CUDA allocation count");
        auto first = table.resolve(a + 255u, 2u);
        demand(first.buffer == 101u && first.offset == 255u &&
               first.remaining == 2u, "CUdeviceptr interior arithmetic");
        rejected<std::out_of_range>(
            [&] { table.resolve(a + 256u, 2u); },
            "Cross-end read accepted");
        rejected<std::out_of_range>(
            [&] { table.resolve(a + 257u, 1u); },
            "One-past-end read accepted");
        rejected<std::out_of_range>(
            [&] { table.resolve(a + 300u, 1u); },
            "Padding pointer accepted");
        rejected<std::out_of_range>(
            [&] { table.resolve(b - 1u, 1u); },
            "Virtual gap accepted");
        rejected<std::invalid_argument>(
            [&] { table.register_allocation(101u, 64u); },
            "Duplicate compute allocation accepted");
        rejected<std::invalid_argument>(
            [&] { table.resolve(a, 0); }, "Zero-length device access accepted");
        rejected<std::out_of_range>(
            [&] { table.resolve(b + 1000u, 25u); },
            "Oversize access accepted");
        demand(table.resolve(b + 1000u, 24u).buffer == 202u,
               "Valid final bytes rejected");

        const auto removed = table.unregister_allocation(a);
        demand(removed == 101u, "CUDA release lost native buffer handle");
        rejected<std::out_of_range>(
            [&] { table.resolve(a + 5u, 1u); },
            "Freed CUdeviceptr remained live");
        rejected<std::out_of_range>(
            [&] { table.unregister_allocation(a); }, "Double free accepted");
        rejected<std::out_of_range>(
            [&] { table.unregister_allocation(b + 4u); },
            "Freeing interior pointer accepted");
        const auto c = table.register_allocation(303u, 4u);
        demand(c > b, "Stale CUdeviceptr base was recycled");
        demand(table.resolve(c, 4u).buffer == 303u, "Reallocated mapping");

        // This is a virtual host address test, not a multi-gigabyte GPU
        // allocation. Prove PhysX's 64-bit CUdeviceptr arithmetic survives
        // addresses and offsets wider than 32 bits.
        const auto huge = table.register_allocation(
            404u, (std::size_t{1} << 32u) + 8192u);
        const auto big = table.resolve(huge + (1ull << 32u) + 4095u, 128u);
        demand(big.buffer == 404u && big.offset == (1ull << 32u) + 4095u,
               "Truncated 64-bit CUdeviceptr offset");

        demand(table.unregister_allocation(b) == 202u &&
               table.unregister_allocation(c) == 303u &&
               table.unregister_allocation(huge) == 404u &&
               table.live_allocations() == 0,
               "CUdeviceptr cleanup");
        std::cout << "PhysX CUDA host address mapping PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX CUDA host address mapping FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
