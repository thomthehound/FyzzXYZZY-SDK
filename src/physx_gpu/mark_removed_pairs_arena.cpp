// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/mark_removed_pairs_arena.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kWorkgroup = 128;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t thread_stride;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5];
};
static_assert(sizeof(Parameters) == 32, "Root/push parameter ABI changed");
} // namespace

MarkRemovedPairsArenaStage::MarkRemovedPairsArenaStage(compute::Context& ctx,
                                                       compute::ShaderBinary shader)
    : context_(ctx) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kWorkgroup ||
        device.max_workgroup_size[0] < kWorkgroup)
        throw std::runtime_error("PhysX removed-pair arena kernel requires 128 threads");

    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

MarkRemovedPairsArenaStage::~MarkRemovedPairsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void MarkRemovedPairsArenaStage::append(std::vector<compute::BatchCommand>& commands,
                                        compute::BufferId arena, std::uint32_t arena_bytes,
                                        std::uint32_t descriptor_offset,
                                        std::uint32_t count) const {
    if (!count) return; // Same host conditional as upstream removed-handle stage.
    if (!arena || !arena_bytes || arena_bytes % 4 ||
        descriptor_offset % alignof(MarkRemovedPairsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(MarkRemovedPairsArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX broadphase arena descriptor range");
    const auto& info = context_.device_info();
    const std::uint32_t groups = std::min(
        (static_cast<std::uint64_t>(count) + kWorkgroup - 1) / kWorkgroup,
        static_cast<std::uint64_t>(info.max_workgroup_count[0]));
    if (!groups) throw std::out_of_range("Compute dispatch group limit is zero");
    const Parameters params{descriptor_offset, groups * kWorkgroup, arena_bytes, {}};
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {groups, 1, 1};
    dispatch.buffers = {{0, arena, 0, arena_bytes}};
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
