// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_device_address_map.h"

#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

CudaDeviceAddressMap::Address CudaDeviceAddressMap::register_allocation(
    compute::BufferId buffer, std::size_t allocation_bytes) {
    if (!buffer || !allocation_bytes)
        throw std::invalid_argument("PhysX CUDA allocation requires buffer and size");
    if (by_buffer_.find(buffer) != by_buffer_.end())
        throw std::invalid_argument("PhysX CUDA buffer registered twice");

    // All arithmetic here is checked against the virtual address width,
    // independently of host size_t width.
    constexpr auto max = std::numeric_limits<Address>::max();
    const Address bytes = static_cast<Address>(allocation_bytes);
    if (bytes > max - (kAlignment - 1))
        throw std::overflow_error("PhysX CUDA allocation size overflow");
    const Address span = (bytes + (kAlignment - 1)) & ~(kAlignment - 1);
    if (span > max - next_address_)
        throw std::overflow_error("PhysX CUDA virtual address exhaustion");

    const Address base = next_address_;
    allocations_.emplace(base, Allocation{buffer, allocation_bytes});
    try {
        by_buffer_.emplace(buffer, base);
    } catch (...) {
        allocations_.erase(base);
        throw;
    }
    next_address_ += span;
    return base;
}

CudaDeviceAddressMap::Resolved CudaDeviceAddressMap::resolve(
    Address address, std::size_t access_bytes) const {
    if (address == 0 || access_bytes == 0)
        throw std::invalid_argument("PhysX CUDA invalid device address/access");

    auto it = allocations_.upper_bound(address);
    if (it == allocations_.begin())
        throw std::out_of_range("PhysX CUDA device address unmapped");
    --it;

    const Address delta = address - it->first;
    if (delta >= static_cast<Address>(it->second.bytes) ||
        access_bytes > it->second.bytes - static_cast<std::size_t>(delta))
        throw std::out_of_range("PhysX CUDA device access crosses allocation");
    return {it->second.buffer, static_cast<std::size_t>(delta),
            it->second.bytes - static_cast<std::size_t>(delta)};
}

compute::BufferId CudaDeviceAddressMap::unregister_allocation(Address base) {
    auto it = allocations_.find(base);
    if (it == allocations_.end())
        throw std::out_of_range("PhysX CUDA free requires live allocation base");
    const compute::BufferId buffer = it->second.buffer;
    by_buffer_.erase(buffer);
    allocations_.erase(it);
    return buffer;
}

std::size_t CudaDeviceAddressMap::live_allocations() const noexcept {
    return allocations_.size();
}

} // namespace fyzz::physx_gpu
