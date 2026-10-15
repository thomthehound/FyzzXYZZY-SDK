// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_region_histograms_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups = 32u;
constexpr std::uint32_t kThreads = 256u;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t boxes;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[5] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX region histogram constants changed");
}

SapRegionHistogramsArenaStage::SapRegionHistogramsArenaStage(
    compute::Context& context, compute::ShaderBinary start_first,
    compute::ShaderBinary start_second, compute::ShaderBinary region_first,
    compute::ShaderBinary region_second,
    compute::ShaderBinary start_first_wave,
    compute::ShaderBinary region_first_wave,
    bool force_portable) : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX region histogram requires 32x256 groups");

    // Source-faithful native-wave first pass on supported GPU devices.
    // All callers supply both codepaths so fast-capable adapters cannot
    // silently run the high-barrier fallback after integration.
    using_wave_ = !force_portable && device.supports_wave_arithmetic &&
                  device.subgroup_size >= 4u &&
                  256u % device.subgroup_size == 0u;
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    compute::ShaderBinary binaries[] = {
        using_wave_ ? std::move(start_first_wave) : std::move(start_first),
        std::move(start_second),
        using_wave_ ? std::move(region_first_wave) : std::move(region_first),
        std::move(region_second)
    };
    try {
        for (std::uint32_t i = 0; i < 4u; ++i) {
            desc.shader = std::move(binaries[i]);
            pipelines_[i] = context_.create_compute_pipeline(desc);
        }
    } catch (...) {
        for (const auto pipeline : pipelines_)
            if (pipeline) context_.destroy_compute_pipeline(pipeline);
        throw;
    }
}

SapRegionHistogramsArenaStage::~SapRegionHistogramsArenaStage() {
    for (const auto pipeline : pipelines_)
        if (pipeline) context_.destroy_compute_pipeline(pipeline);
}

void SapRegionHistogramsArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t boxes, std::uint32_t created_handles) const {
    if (created_handles == 0u) return;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapRegionHistogramsArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapRegionHistogramsArenaDesc) > arena_bytes - descriptor_offset ||
        boxes == 0u || created_handles > boxes ||
        // Upstream CUDA rounds 2*boxes up with +255 before /256.
        // Reject a count that would overflow that 32-bit addition.
        boxes > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u)
        throw std::invalid_argument("Invalid PhysX region histogram dimensions");

    const Parameters params{descriptor_offset, boxes, arena_bytes, {}};
    std::vector<compute::BatchCommand> staged;
    staged.reserve(4u);
    for (const auto pipeline : pipelines_) {
        compute::DispatchCommand dispatch;
        dispatch.pipeline = pipeline;
        dispatch.groups = {kGroups, 1u, 1u};
        dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
        staged.emplace_back(std::move(dispatch));
    }
    commands.reserve(commands.size() + staged.size());
    for (auto& command : staged)
        commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
