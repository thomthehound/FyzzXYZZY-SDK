// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_active_region_lists_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t boxes;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX active-region constants changed");
}

SapActiveRegionListsArenaStage::SapActiveRegionListsArenaStage(
    compute::Context& context, compute::ShaderBinary writeout,
    compute::ShaderBinary compute_wave,
    compute::ShaderBinary compute_portable,
    compute::ShaderBinary output, bool force_portable)
    : context_(context) {
    using_wave_ = !force_portable &&
                  context.device_info().supports_wave_arithmetic;
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < 512u ||
        device.max_workgroup_size[0] < 512u ||
        device.max_workgroup_count[0] < 64u)
        throw std::runtime_error("PhysX active regions require 64x512 dispatch support");

    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    compute::ShaderBinary shaders[] = {
        std::move(writeout),
        using_wave_ ? std::move(compute_wave) : std::move(compute_portable),
        std::move(output)
    };
    try {
        for (std::uint32_t i = 0; i < 3u; ++i) {
            desc.shader = std::move(shaders[i]);
            pipelines_[i] = context_.create_compute_pipeline(desc);
        }
    } catch (...) {
        for (const auto pipeline : pipelines_)
            if (pipeline) context_.destroy_compute_pipeline(pipeline);
        throw;
    }
}
SapActiveRegionListsArenaStage::~SapActiveRegionListsArenaStage() {
    for (const auto pipeline : pipelines_)
        if (pipeline) context_.destroy_compute_pipeline(pipeline);
}
void SapActiveRegionListsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t boxes, std::uint32_t created_handles) const {
    if (!created_handles) return;
    // The lowered arena addresses 64 dense region arrays of 2*boxes
    // 32-bit words. Preserve the original region grid (4^3) and avoid
    // arithmetic wraparound in both descriptor offsets and shader loops.
    const std::uint64_t dense_bytes = std::uint64_t{boxes} * 2u * 64u * 4u;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapActiveRegionListsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapActiveRegionListsArenaDesc) > arena_bytes - descriptor_offset ||
        !boxes || created_handles > boxes ||
        boxes > (std::numeric_limits<std::uint32_t>::max() - 511u) / 2u ||
        dense_bytes > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid PhysX active-region dimensions");

    const Parameters params{descriptor_offset, boxes, arena_bytes, {}};
    const std::uint32_t groups[] = {32u, 64u, 64u};
    const std::uint32_t threads[] = {256u, 512u, 256u};
    (void)threads; // encoded by the original source shader binaries
    std::vector<compute::BatchCommand> staged;
    staged.reserve(3u);
    for (std::uint32_t i = 0; i < 3u; ++i) {
        compute::DispatchCommand cmd;
        cmd.pipeline = pipelines_[i];
        cmd.groups = {groups[i], 1u, 1u};
        cmd.buffers = {{0u, arena, 0u, arena_bytes, compute::BufferAccess::ReadWrite}};
        cmd.push_constants.resize(sizeof(params));
        std::memcpy(cmd.push_constants.data(), &params, sizeof(params));
        staged.emplace_back(std::move(cmd));
    }
    commands.reserve(commands.size() + staged.size());
    for (auto& cmd : staged)
        commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
