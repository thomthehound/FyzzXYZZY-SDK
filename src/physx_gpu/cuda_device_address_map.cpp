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
    if (by_buffer_.find(buffer) != by_buffer_.end() ||
        slice_counts_.find(buffer) != slice_counts_.end())
        throw std::invalid_argument(
            "PhysX CUDA dedicated buffer already owned or referenced by slices");

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

CudaDeviceAddressMap::Address CudaDeviceAddressMap::register_slice(
    compute::BufferId buffer, std::size_t backing_offset,
    std::size_t allocation_bytes) {
    if (!buffer || !allocation_bytes || backing_offset % 4u ||
        backing_offset > std::numeric_limits<std::size_t>::max() - allocation_bytes ||
        by_buffer_.find(buffer) != by_buffer_.end())
        throw std::invalid_argument(
            "Invalid PhysX GPU arena slice or conflicting dedicated owner");
    constexpr Address max = std::numeric_limits<Address>::max();
    if (static_cast<Address>(allocation_bytes) > max - (kAlignment - 1u))
        throw std::overflow_error("PhysX GPU allocation slice size overflow");
    const auto bytes = static_cast<Address>(allocation_bytes);
    const auto span = (bytes + (kAlignment - 1u)) & ~(kAlignment - 1u);
    if (span > max - next_address_)
        throw std::overflow_error("PhysX GPU slice virtual address exhausted");

    const Address base = next_address_;
    allocations_.emplace(base, Allocation{buffer, allocation_bytes,
                                          backing_offset, false});
    try {
        auto [it, inserted] = slice_counts_.try_emplace(buffer, 0u);
        ++it->second;
    } catch (...) {
        allocations_.erase(base);
        throw;
    }
    next_address_ += span;
    return base;
}

void CudaDeviceAddressMap::relocate_slice(
    Address base, compute::BufferId buffer, std::size_t backing_offset) {
    auto it = allocations_.find(base);
    if (it == allocations_.end() || it->second.owned || !buffer ||
        backing_offset % 4u ||
        backing_offset > std::numeric_limits<std::size_t>::max() - it->second.bytes ||
        by_buffer_.find(buffer) != by_buffer_.end())
        throw std::invalid_argument("Invalid PhysX GPU arena slice relocation");
    const auto old_buffer = it->second.buffer;
    if (old_buffer != buffer) {
        // Acquire the new reference before releasing the old one so an
        // allocation failure cannot corrupt host lifetime bookkeeping.
        auto source = slice_counts_.find(old_buffer);
        if (source == slice_counts_.end() || !source->second)
            throw std::logic_error("PhysX CUDA slice lifetime bookkeeping corrupted");
        auto [dest, inserted] = slice_counts_.try_emplace(buffer, 0u);
        ++dest->second;
        if (--source->second == 0u) slice_counts_.erase(source);
        it->second.buffer = buffer;
    }
    it->second.backing_offset = backing_offset;
}

CudaDeviceAddressMap::Resolved CudaDeviceAddressMap::unregister_slice(Address base) {
    auto it = allocations_.find(base);
    if (it == allocations_.end() || it->second.owned)
        throw std::out_of_range("Free requires live PhysX arena slice base");
    const Resolved resolved{it->second.buffer, it->second.backing_offset,
                            it->second.bytes};
    auto references = slice_counts_.find(it->second.buffer);
    if (references == slice_counts_.end() || !references->second)
        throw std::logic_error("PhysX CUDA slice lifetime bookkeeping corrupted");
    if (--references->second == 0u) slice_counts_.erase(references);
    allocations_.erase(it);
    return resolved;
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
    return {it->second.buffer,
            it->second.backing_offset + static_cast<std::size_t>(delta),
            it->second.bytes - static_cast<std::size_t>(delta)};
}

compute::BufferId CudaDeviceAddressMap::unregister_allocation(Address base) {
    auto it = allocations_.find(base);
    if (it == allocations_.end() || !it->second.owned)
        throw std::out_of_range("PhysX CUDA free requires dedicated allocation base");
    const compute::BufferId buffer = it->second.buffer;
    by_buffer_.erase(buffer);
    allocations_.erase(it);
    return buffer;
}

std::size_t CudaDeviceAddressMap::live_allocations() const noexcept {
    return allocations_.size();
}

} // namespace fyzz::physx_gpu
