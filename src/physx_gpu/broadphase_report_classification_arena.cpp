// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_report_classification_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups = 32u;
constexpr std::uint32_t kStreams = 4u;
constexpr std::uint32_t kThreads = 256u;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX report classification root constants changed");
}

BroadphaseReportClassificationArenaStage::BroadphaseReportClassificationArenaStage(
    compute::Context& context, compute::ShaderBinary first_shader,
    compute::ShaderBinary second_shader) : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups ||
        device.max_workgroup_count[1] < kStreams)
        throw std::runtime_error("PhysX report classification requires 32x4x256 grid");
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    desc.shader = std::move(first_shader);
    first_ = context_.create_compute_pipeline(desc);
    try {
        desc.shader = std::move(second_shader);
        second_ = context_.create_compute_pipeline(desc);
    } catch (...) {
        context_.destroy_compute_pipeline(first_);
        first_ = 0;
        throw;
    }
}
BroadphaseReportClassificationArenaStage::~BroadphaseReportClassificationArenaStage() {
    if (second_) context_.destroy_compute_pipeline(second_);
    if (first_) context_.destroy_compute_pipeline(first_);
}
void BroadphaseReportClassificationArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset) const {
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(BroadphaseReportClassificationArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(BroadphaseReportClassificationArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX classification descriptor range");

    const Parameters parameters{descriptor_offset, arena_bytes, {}};
    const auto make_dispatch = [&](compute::PipelineId pipeline) {
        compute::DispatchCommand cmd;
        cmd.pipeline = pipeline;
        cmd.groups = {kGroups, kStreams, 1};
        cmd.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
        cmd.push_constants.resize(sizeof(parameters));
        std::memcpy(cmd.push_constants.data(), &parameters, sizeof(parameters));
        return cmd;
    };
    // No partially submitted first stage on a later command construction failure.
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2u);
    staged.emplace_back(make_dispatch(first_));
    staged.emplace_back(make_dispatch(second_));
    commands.reserve(commands.size() + staged.size());
    commands.emplace_back(std::move(staged[0]));
    commands.emplace_back(std::move(staged[1]));
}
} // namespace fyzz::physx_gpu
