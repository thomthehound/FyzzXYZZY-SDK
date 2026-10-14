// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_ranked_sort_arena.h"

#include "physx_gpu/radix_count_arena.h"
#include "physx_gpu/radix_scatter_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kGroups = 64;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t previous_num_boxes;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX ranked-sort constants changed");
}

SapRankedSortArenaStage::SapRankedSortArenaStage(
    compute::Context& context, compute::ShaderBinary initialize_shader,
    compute::ShaderBinary update_handles_shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX WR rank initialization requires 64x256 groups");
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    desc.shader = std::move(initialize_shader);
    initialize_pipeline_ = context_.create_compute_pipeline(desc);
    try {
        desc.shader = std::move(update_handles_shader);
        update_pipeline_ = context_.create_compute_pipeline(desc);
    } catch (...) {
        context_.destroy_compute_pipeline(initialize_pipeline_);
        initialize_pipeline_ = 0;
        throw;
    }
}

SapRankedSortArenaStage::~SapRankedSortArenaStage() {
    if (update_pipeline_) context_.destroy_compute_pipeline(update_pipeline_);
    if (initialize_pipeline_) context_.destroy_compute_pipeline(initialize_pipeline_);
}

void SapRankedSortArenaStage::append_dispatch(
    std::vector<compute::BatchCommand>& commands, compute::PipelineId pipeline,
    compute::BufferId arena, std::uint32_t arena_bytes,
    std::uint32_t descriptor_offset, std::uint32_t previous_num_boxes) const {
    const Parameters params{descriptor_offset, previous_num_boxes, arena_bytes, {}};
    compute::DispatchCommand cmd;
    cmd.pipeline = pipeline;
    cmd.groups = {kGroups, 1, 1};
    cmd.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(cmd));
}

void SapRankedSortArenaStage::append_source_sequence(
    std::vector<compute::BatchCommand>& commands,
    const RadixCountArenaStage& count_stage,
    const RadixScatterArenaStage& scatter_stage,
    compute::BufferId arena, std::uint32_t arena_bytes,
    std::uint32_t descriptor_offset,
    const std::uint32_t (&count_descriptor_offsets)[2],
    const std::uint32_t (&scatter_descriptor_offsets)[2],
    std::uint32_t previous_num_boxes) const {

    if (!previous_num_boxes) return; // Original WR host update gate.
    const std::uint64_t live_projections = std::uint64_t{previous_num_boxes} * 2u;
    const std::uint64_t padded = (live_projections + 3u) & ~std::uint64_t{3};
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapRankedSortArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapRankedSortArenaDesc) > arena_bytes - descriptor_offset ||
        padded > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid PhysX WR ranked-sort dimensions");

    append_dispatch(commands, initialize_pipeline_, arena, arena_bytes,
                    descriptor_offset, previous_num_boxes);
    for (std::uint32_t pass = 0; pass < 8u; ++pass) {
        const auto parity = pass & 1u;
        count_stage.append(commands, arena, arena_bytes,
                           count_descriptor_offsets[parity],
                           static_cast<std::uint32_t>(padded), pass * 4u);
        scatter_stage.append(commands, arena, arena_bytes,
                             scatter_descriptor_offsets[parity],
                             static_cast<std::uint32_t>(padded), pass * 4u);
    }
    append_dispatch(commands, update_pipeline_, arena, arena_bytes,
                    descriptor_offset, previous_num_boxes);
}
} // namespace fyzz::physx_gpu
