// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_newbounds_pairs_arena.h"
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX new-bounds pair constants");
}
SapNewBoundsPairsArenaStage::SapNewBoundsPairsArenaStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    // 64-bit GPU work indexing, native 32-bit wave compaction.
    // The feature test happens ONCE; the append/dispatch path never queries it.
    if (!device.supports_int64_shader || !device.supports_wave_arithmetic ||
        device.subgroup_size < 4u || device.subgroup_size > 256u ||
        256u % device.subgroup_size != 0u ||
        device.max_workgroup_invocations < 256u ||
        device.max_workgroup_size[0] < 256u ||
        device.max_workgroup_count[0] < 64u)
        throw std::runtime_error(
            "PhysX new-bounds pairs require uint64 shaders, subgroup arithmetic "
            "and original 64x256 grid");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1u;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}
SapNewBoundsPairsArenaStage::~SapNewBoundsPairsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}
void SapNewBoundsPairsArenaStage::append(
    std::vector<compute::BatchCommand>& commands,
    compute::BufferId arena, std::uint32_t arena_bytes,
    std::uint32_t descriptor_offset, std::uint32_t created_handles) const {
    if (!created_handles) return;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapNewBoundsPairsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapNewBoundsPairsArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX new-bounds pair arena");
    const Parameters params{descriptor_offset, arena_bytes, {}};
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {64u, 1u, 1u};
    command.buffers = {{0u, arena, 0u, arena_bytes,
                        compute::BufferAccess::ReadWrite}};
    command.push_constants.resize(sizeof(params));
    std::memcpy(command.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
