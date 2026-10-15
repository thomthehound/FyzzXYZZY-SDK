// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_incremental_pairs_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kBlocks = 256;
constexpr std::uint32_t kAxes = 3;
constexpr std::uint32_t kThreads = 256;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t previous_num_boxes;
    std::uint32_t comparison_totals_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX incremental pair root layout changed");
}

SapIncrementalPairsArenaStage::SapIncrementalPairsArenaStage(
    compute::Context& context, compute::ShaderBinary shader,
    compute::ShaderBinary wave_shader, bool force_portable)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kBlocks ||
        device.max_workgroup_count[1] < kAxes)
        throw std::runtime_error("PhysX incremental SAP requires 256x3x256 launch");
    using_wave_ = !force_portable && device.supports_wave_arithmetic &&
                  device.subgroup_size >= 4u &&
                  256u % device.subgroup_size == 0u;
    compute::ComputePipelineDesc desc;
    desc.shader = using_wave_ ? std::move(wave_shader) : std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}
SapIncrementalPairsArenaStage::~SapIncrementalPairsArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SapIncrementalPairsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t previous_num_boxes,
    std::uint32_t comparison_totals_offset) const {
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapIncrementalPairsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapIncrementalPairsArenaDesc) > arena_bytes - descriptor_offset ||
        comparison_totals_offset % 4u ||
        comparison_totals_offset > arena_bytes ||
        12u > arena_bytes - comparison_totals_offset ||
        previous_num_boxes > std::numeric_limits<std::uint32_t>::max() / 2u)
        throw std::invalid_argument("Invalid PhysX incremental pair dimensions");

    const Parameters params{descriptor_offset, previous_num_boxes,
                            comparison_totals_offset, arena_bytes, {}};
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {kBlocks, kAxes, 1};
    dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
