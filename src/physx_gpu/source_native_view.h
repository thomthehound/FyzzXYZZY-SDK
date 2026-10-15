// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Resolve one original CUDA virtual allocation to a bounded native buffer
// view. The native base is rounded down to the device descriptor alignment;
// the returned shader-local offset accounts for that rounding. Physical
// ownership, retirement and logical CUDA addresses remain with the caller.
inline compute::BufferBinding source_native_view(
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address address,
    std::size_t bytes, std::size_t alignment, std::uint32_t slot,
    compute::BufferAccess access, std::uint32_t& local_offset,
    std::uint64_t max_view_bytes) {
    if (!address || !bytes || address % 4u || !alignment ||
        alignment % 4u)
        throw std::invalid_argument("Invalid PhysX native source view parameters");
    const auto physical = addresses.resolve(address, bytes);
    const auto base = physical.offset - physical.offset % alignment;
    const auto delta = physical.offset - base;
    if (delta > std::numeric_limits<std::uint32_t>::max() ||
        bytes > std::numeric_limits<std::size_t>::max() - delta ||
        delta + bytes > max_view_bytes ||
        delta + bytes > std::numeric_limits<std::uint32_t>::max())
        throw std::out_of_range("PhysX allocation exceeds native storage view bounds");
    const auto view_bytes = delta + bytes;
    if (view_bytes % 4u)
        throw std::invalid_argument("PhysX native RAW storage view is not word-aligned");
    local_offset = static_cast<std::uint32_t>(delta);
    return {slot, physical.buffer, base, view_bytes, access};
}

inline std::size_t source_allocation_bytes(std::uint64_t count, std::uint64_t stride) {
    if (!count || !stride ||
        count > std::numeric_limits<std::size_t>::max() / stride)
        throw std::out_of_range("PhysX source allocation capacity overflow");
    return static_cast<std::size_t>(count * stride);
}

} // namespace fyzz::physx_gpu
