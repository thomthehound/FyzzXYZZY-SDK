// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_incremental_comparisons_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kThreads = 256;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t previous_num_boxes;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX incremental comparison constants changed");
}

SapIncrementalComparisonsArenaStage::SapIncrementalComparisonsArenaStage(
    compute::Context& context, compute::ShaderBinary stage1_shader,
    compute::ShaderBinary stage2_shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kBlocks)
        throw std::runtime_error("PhysX incremental comparisons require 32x256 launch");

    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    desc.shader = std::move(stage1_shader);
    stage1_ = context_.create_compute_pipeline(desc);
    try {
        desc.shader = std::move(stage2_shader);
        stage2_ = context_.create_compute_pipeline(desc);
    } catch (...) {
        context_.destroy_compute_pipeline(stage1_);
        stage1_ = 0;
        throw;
    }
}

SapIncrementalComparisonsArenaStage::~SapIncrementalComparisonsArenaStage() {
    if (stage2_) context_.destroy_compute_pipeline(stage2_);
    if (stage1_) context_.destroy_compute_pipeline(stage1_);
}

void SapIncrementalComparisonsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t previous_num_boxes) const {
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapIncrementalComparisonArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapIncrementalComparisonArenaDesc) > arena_bytes - descriptor_offset ||
        previous_num_boxes > std::numeric_limits<std::uint32_t>::max() / 2u)
        throw std::invalid_argument("Invalid PhysX incremental comparison dimensions");

    const Parameters params{descriptor_offset, previous_num_boxes, arena_bytes, {}};
    const auto make_dispatch = [&](compute::PipelineId pipeline) {
        compute::DispatchCommand cmd;
        cmd.pipeline = pipeline;
        cmd.groups = {kBlocks, 1, 1};
        cmd.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
        cmd.push_constants.resize(sizeof(params));
        std::memcpy(cmd.push_constants.data(), &params, sizeof(params));
        return cmd;
    };
    // Both original launches are one host operation. Leave caller batches
    // unchanged on descriptor errors or command-construction failures.
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2);
    staged.emplace_back(make_dispatch(stage1_));
    staged.emplace_back(make_dispatch(stage2_));
    commands.reserve(commands.size() + staged.size());
    commands.emplace_back(std::move(staged[0]));
    commands.emplace_back(std::move(staged[1]));
}
} // namespace fyzz::physx_gpu
