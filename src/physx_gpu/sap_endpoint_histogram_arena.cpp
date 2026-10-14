// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_endpoint_histogram_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kAxes = 3;
constexpr std::uint32_t kThreads = 256;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t previous_boxes;
    std::uint32_t created_boxes;
    std::uint32_t removed_boxes;
    std::uint32_t is_incremental;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[2] = {};
};
static_assert(sizeof(Parameters) == 32, "SAP endpoint root constants changed");
}

SapEndpointHistogramArenaStage::SapEndpointHistogramArenaStage(
    compute::Context& context, compute::ShaderBinary first_pass_shader,
    compute::ShaderBinary second_pass_shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kBlocks ||
        device.max_workgroup_count[1] < kAxes)
        throw std::runtime_error("PhysX endpoint histogram requires 32x3x256 dispatch");

    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    desc.shader = std::move(first_pass_shader);
    first_pipeline_ = context_.create_compute_pipeline(desc);
    try {
        desc.shader = std::move(second_pass_shader);
        second_pipeline_ = context_.create_compute_pipeline(desc);
    } catch (...) {
        context_.destroy_compute_pipeline(first_pipeline_);
        first_pipeline_ = 0;
        throw;
    }
}

SapEndpointHistogramArenaStage::~SapEndpointHistogramArenaStage() {
    if (second_pipeline_) context_.destroy_compute_pipeline(second_pipeline_);
    if (first_pipeline_) context_.destroy_compute_pipeline(first_pipeline_);
}

void SapEndpointHistogramArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t previous_boxes, std::uint32_t created_boxes,
    std::uint32_t removed_boxes, bool is_incremental) const {

    const std::uint64_t all_boxes =
        std::uint64_t{previous_boxes} + created_boxes;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapEndpointHistogramArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapEndpointHistogramArenaDesc) > arena_bytes - descriptor_offset ||
        removed_boxes > all_boxes ||
        2u * all_boxes > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid PhysX endpoint histogram dimensions");

    const Parameters params{descriptor_offset, previous_boxes, created_boxes,
                            removed_boxes, is_incremental ? 1u : 0u,
                            arena_bytes, {}};
    const auto make_dispatch = [&](compute::PipelineId pipeline) {
        compute::DispatchCommand dispatch;
        dispatch.pipeline = pipeline;
        dispatch.groups = {kBlocks, kAxes, 1};
        dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
        return dispatch;
    };

    // A multi-dispatch host append must not partially modify the batch.
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2);
    staged.emplace_back(make_dispatch(first_pipeline_));
    staged.emplace_back(make_dispatch(second_pipeline_));
    commands.reserve(commands.size() + staged.size());
    commands.emplace_back(std::move(staged[0]));
    commands.emplace_back(std::move(staged[1]));
}
} // namespace fyzz::physx_gpu
