// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/removed_projections_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kWorkgroup = 256;
constexpr std::uint32_t kGrid = 32;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t thread_stride;
    std::uint32_t arena_bytes;
    std::uint32_t unused[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX SAP push/root ABI changed");
} // namespace

RemovedProjectionsArenaStage::RemovedProjectionsArenaStage(
    compute::Context& ctx, compute::ShaderBinary shader)
    : context_(ctx) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kWorkgroup ||
        device.max_workgroup_size[0] < kWorkgroup ||
        device.max_workgroup_count[0] < kGrid)
        throw std::runtime_error("PhysX SAP removal requires original 32x256 launch capability");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

RemovedProjectionsArenaStage::~RemovedProjectionsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void RemovedProjectionsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t removed_count) const {
    if (!removed_count) return;
    if (!arena || !arena_bytes || arena_bytes % 4 ||
        descriptor_offset % alignof(RemovedProjectionsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(RemovedProjectionsArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX projection removal descriptor range");

    const Parameters params{descriptor_offset, kGrid * kWorkgroup, arena_bytes, {}};
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {kGrid, 1, 1};
    dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(dispatch));
}

} // namespace fyzz::physx_gpu
