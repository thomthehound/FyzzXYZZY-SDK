// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

std::uint32_t source_d32_fill_chunk_words(
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address address,
    std::uint32_t remaining_words,
    std::size_t alignment, std::uint64_t max_view_bytes) {
    if (!remaining_words || !address || address % 4u ||
        !alignment || alignment % 4u)
        throw std::invalid_argument("Invalid PhysX source D32 fill chunk");
    const auto physical = addresses.resolve(address, 4u);
    const auto view_alignment = std::lcm(alignment, std::size_t{16u});
    const auto delta = physical.offset % view_alignment;
    const auto view_limit = std::min<std::uint64_t>(
        max_view_bytes, static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) & ~3ull);
    if (view_limit <= delta || view_limit - delta < 4u)
        throw std::out_of_range("PhysX D32 source GPU view range too small");
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        remaining_words, (view_limit - delta) / 4u));
}

SourceFillD32NativeStage::SourceFillD32NativeStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& gpu = context_.device_info();
    if (gpu.max_workgroup_invocations < 256u ||
        gpu.max_workgroup_size[0] < 256u ||
        gpu.max_workgroup_count[0] == 0u)
        throw std::runtime_error("PhysX source D32 fill requires 256-thread groups");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1u;
    desc.push_constant_bytes = sizeof(std::uint32_t) * 4u;
    desc.storage_buffer_views = {compute::StorageBufferViewKind::RawReadWrite};
    pipeline_ = context_.create_compute_pipeline(desc);
}

SourceFillD32NativeStage::~SourceFillD32NativeStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SourceFillD32NativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address device_address,
    std::uint32_t value, std::uint32_t words) const {
    // CUDA zero-sized memory operations are no-ops, not GPU submissions.
    if (!words) return;
    const auto& gpu = context_.device_info();
    const auto alignment = std::max<std::size_t>(
        4u, static_cast<std::size_t>(gpu.storage_buffer_offset_alignment));
    if (alignment % 4u || !gpu.max_storage_buffer_range)
        throw std::runtime_error("Native PhysX source D32 storage limits unavailable");

    // One PhysX CUDA allocation can exceed the backend's maximum storage
    // view range. Slice that *same* native resource without allocating or
    // copying it. Stage every chunk before modifying the caller's batch:
    // an invalid trailing pointer must not leave a partial device fill.
    const auto total_bytes = source_allocation_bytes(words, 4u);
    addresses.resolve(device_address, total_bytes);
    const auto view_limit = std::min<std::uint64_t>(
        gpu.max_storage_buffer_range,
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) & ~3ull);
    std::vector<compute::BatchCommand> pending;
    std::uint64_t consumed_words = 0u;
    while (consumed_words < words) {
        const auto cursor = device_address + consumed_words * 4u;
        const auto chunk_words = source_d32_fill_chunk_words(
            addresses, cursor, static_cast<std::uint32_t>(
                words - consumed_words), alignment, view_limit);
        const auto desired_groups = (std::uint64_t(chunk_words - 1u) / 256u) + 1u;
        const auto groups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                {desired_groups, gpu.max_workgroup_count[0], 65535u}));
        if (!groups)
            throw std::runtime_error("Native PhysX source D32 GPU launch unavailable");

        std::array<std::uint32_t, 4> params{
            0u, chunk_words, value, groups * 256u};
        compute::DispatchCommand cmd;
        cmd.pipeline = pipeline_;
        cmd.groups = {groups, 1u, 1u};
        // Four-byte interior CUDA pointers remain exact shader offsets;
        // their physical RAW GPU descriptor bases remain 16-byte aligned.
        cmd.buffers.push_back(source_native_view(
            addresses, cursor, source_allocation_bytes(chunk_words, 4u),
            alignment, 0u, compute::BufferAccess::Write,
            params[0], view_limit));
        cmd.push_constants.resize(sizeof(params));
        std::memcpy(cmd.push_constants.data(), params.data(), sizeof(params));
        pending.emplace_back(std::move(cmd));
        consumed_words += chunk_words;
    }
    commands.reserve(commands.size() + pending.size());
    for (auto& command : pending)
        commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
