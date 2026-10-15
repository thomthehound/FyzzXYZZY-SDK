// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_aabb_launch_lowering.h"
#include "PxgKernelIndices.h"
#include "PxgBroadPhaseKernelIndices.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <limits>

namespace fyzz::physx_gpu {
namespace {
template <typename T>
T read_arg(const void* const* args, std::size_t index) {
    if (!args[index]) throw std::invalid_argument("Missing PhysX CUDA kernel argument");
    T result{};
    std::memcpy(&result, args[index], sizeof(result));
    return result;
}
}

CudaAabbLaunchLowering::CudaAabbLaunchLowering(
    const CudaDeviceAddressMap& addresses,
    const AabbEncodingStage& native_shader) noexcept
    : addresses_(addresses), native_shader_(native_shader) {}

void CudaAabbLaunchLowering::append(
    std::vector<compute::BatchCommand>& commands,
    std::uint32_t kernel_id, std::uint32_t cuda_grid_x,
    std::uint32_t cuda_block_x,
    const void* const* args, std::size_t count) const {
    // Source kernel identity, launch shape, argument order and scalar
    // value sizes are checked here. Do not silently interpret arbitrary
    // CUfunction handles as our one implemented kernel.
    if (kernel_id != physx::PxgKernelIds::BP_TRANSLATE_AABBS)
        throw std::invalid_argument("Unsupported PhysX CUDA kernel ID");
    if (!args || count != 11u ||
        cuda_block_x != physx::PxgBPKernelBlockDim::BP_TRANSLATE_AABBS)
        throw std::invalid_argument("Invalid PhysX translateAABBsLaunch ABI");

    const auto bounds = read_arg<std::uint64_t>(args, 0u);
    const auto encoded = read_arg<std::uint64_t>(args, 1u);
    const auto distance = read_arg<std::uint64_t>(args, 2u);
    const auto environment = read_arg<std::uint64_t>(args, 3u);
    const AabbEncodingParameters params{
        read_arg<std::uint32_t>(args, 4u),
        read_arg<std::uint32_t>(args, 5u),
        read_arg<std::uint32_t>(args, 6u),
        read_arg<std::uint32_t>(args, 7u),
        read_arg<std::uint32_t>(args, 8u),
        read_arg<std::uint32_t>(args, 9u),
        read_arg<std::uint32_t>(args, 10u)
    };
    if (!params.count)
        throw std::invalid_argument("Zero-sized PhysX AABB CUDA launch");
    const std::uint64_t original_blocks =
        (std::uint64_t(params.count) + 31u) / 32u; // 256 threads / 8 per box
    if (cuda_grid_x != original_blocks)
        throw std::invalid_argument("Source PhysX CUDA grid shape mismatched");

    const std::uint64_t bounds_bytes = std::uint64_t(params.count) * 24u;
    const std::uint64_t scalar_bytes = std::uint64_t(params.count) * 4u;
    if (bounds_bytes > std::numeric_limits<std::size_t>::max())
        throw std::out_of_range("CUDA PhysX AABB allocation size overflow");

    const auto input = addresses_.resolve(bounds, static_cast<std::size_t>(bounds_bytes));
    const auto output = addresses_.resolve(encoded, static_cast<std::size_t>(bounds_bytes));
    const auto distances = addresses_.resolve(distance, static_cast<std::size_t>(scalar_bytes));
    const auto env = environment != 0u ?
        addresses_.resolve(environment, static_cast<std::size_t>(scalar_bytes)) :
        distances;
    const std::array<compute::BufferBinding, 4> native_views{{
        {0u, input.buffer, input.offset, static_cast<std::size_t>(bounds_bytes),
         compute::BufferAccess::Read},
        {1u, distances.buffer, distances.offset, static_cast<std::size_t>(scalar_bytes),
         compute::BufferAccess::Read},
        {2u, env.buffer, env.offset, static_cast<std::size_t>(scalar_bytes),
         compute::BufferAccess::Read},
        {3u, output.buffer, output.offset, static_cast<std::size_t>(bounds_bytes),
         compute::BufferAccess::Write}
    }};

    // Appending to the original host's command batch ensures existing
    // Vulkan/D3D12 resource barriers and completion fences, not new waits.
    // Translated HLSL uses 128 threads and 16 boxes per block, while the
    // original CUDA launch used 256 and 32; work coverage stays identical.
    native_shader_.append_views(commands, native_views, params, environment != 0u);
}
} // namespace fyzz::physx_gpu
