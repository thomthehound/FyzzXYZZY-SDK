// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_copy_reports_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups = 64u;
constexpr std::uint32_t kThreads = 256u;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX copy-reports constants changed");
}

BroadphaseCopyReportsArenaStage::BroadphaseCopyReportsArenaStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX copy-reports requires 64x256 grid");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

BroadphaseCopyReportsArenaStage::~BroadphaseCopyReportsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void BroadphaseCopyReportsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset) const {
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(BroadphaseCopyReportsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(BroadphaseCopyReportsArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX copy-reports descriptor");
    const Parameters params{descriptor_offset, arena_bytes, {}};
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {kGroups, 1, 1};
    command.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    command.push_constants.resize(sizeof(params));
    std::memcpy(command.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
