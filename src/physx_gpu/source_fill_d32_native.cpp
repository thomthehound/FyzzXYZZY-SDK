// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

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

    const auto desired_groups = (std::uint64_t(words - 1u) / 256u) + 1u;
    const auto groups = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(
            {desired_groups, gpu.max_workgroup_count[0], 65535u}));
    if (!groups)
        throw std::runtime_error("Native PhysX source D32 GPU launch unavailable");

    std::array<std::uint32_t, 4> params{0u, words, value, groups * 256u};
    compute::DispatchCommand cmd;
    cmd.pipeline = pipeline_;
    cmd.groups = {groups, 1u, 1u};
    // PhysX's CUDA scalar/device pointers can be four-byte aligned inside
    // a native heap root. The common resolver preserves their exact offset,
    // expanding the physical RAW UAV view base to 16-byte alignment.
    cmd.buffers.push_back(source_native_view(
        addresses, device_address, source_allocation_bytes(words, 4u),
        alignment, 0u, compute::BufferAccess::Write,
        params[0], gpu.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(), params.data(), sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
