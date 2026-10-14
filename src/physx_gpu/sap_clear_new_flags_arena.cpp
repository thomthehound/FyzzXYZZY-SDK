// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_clear_new_flags_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups = 32;
constexpr std::uint32_t kThreads = 256;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX clear-new root constants changed");
}

SapClearNewFlagsArenaStage::SapClearNewFlagsArenaStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX BP_CLEAR_NEWFLAG requires 32x256 groups");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

SapClearNewFlagsArenaStage::~SapClearNewFlagsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SapClearNewFlagsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t created_count) const {
    // This conditional is in PxgCudaBroadPhaseSap::clearNewFlagKernel,
    // not an optimization invented by the portable host.
    if (!created_count) return;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapClearNewFlagsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapClearNewFlagsArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX BP_CLEAR_NEWFLAG descriptor range");

    const Parameters params{descriptor_offset, arena_bytes, {}};
    compute::DispatchCommand cmd;
    cmd.pipeline = pipeline_;
    cmd.groups = {kGroups, 1, 1};
    cmd.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
