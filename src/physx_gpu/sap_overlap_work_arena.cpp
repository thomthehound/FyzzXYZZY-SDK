// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_overlap_work_arena.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t boxes;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX overlap producer constants");
}
SapOverlapWorkArenaStage::SapOverlapWorkArenaStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < 256u ||
        device.max_workgroup_size[0] < 256u ||
        device.max_workgroup_count[0] < 32u)
        throw std::runtime_error("PhysX overlap producer requires 32x256");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1u;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}
SapOverlapWorkArenaStage::~SapOverlapWorkArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}
void SapOverlapWorkArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t boxes, std::uint32_t created_handles) const {
    if (!created_handles) return;
    // Preserve the original create-only dispatch gate and 32x256 work partition.
    // The arena ABI stores byte offsets in u32 and dense 64-region arrays.
    const auto dense_bytes = std::uint64_t{boxes} * 2u * 64u * 4u;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapOverlapWorkArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapOverlapWorkArenaDesc) > arena_bytes - descriptor_offset ||
        !boxes || created_handles > boxes ||
        boxes > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u ||
        dense_bytes > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid PhysX overlap producer dimensions");
    const Parameters params{descriptor_offset, boxes, arena_bytes, {}};
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {32u, 1u, 1u};
    command.buffers = {{0u, arena, 0u, arena_bytes,
                        compute::BufferAccess::ReadWrite}};
    command.push_constants.resize(sizeof(params));
    std::memcpy(command.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
