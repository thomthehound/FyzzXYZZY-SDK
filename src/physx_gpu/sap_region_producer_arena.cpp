// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_region_producer_arena.h"

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
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX region producer constants changed");
}
SapRegionProducerArenaStage::SapRegionProducerArenaStage(
    compute::Context& context, compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kGroups)
        throw std::runtime_error("PhysX createRegionsKernel requires 32x256");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}
SapRegionProducerArenaStage::~SapRegionProducerArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}
void SapRegionProducerArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t boxes, std::uint32_t created_handles) const {
    // Original computeRegionHistogramKernel returns immediately with no
    // created handles; the host still manages its pre-update elsewhere.
    if (!created_handles) return;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapRegionProducerArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapRegionProducerArenaDesc) > arena_bytes - descriptor_offset ||
        !boxes || created_handles > boxes ||
        // Upstream CUDA rounds 2*boxes up with +255 before /256.
        // Reject a count that would overflow that 32-bit addition.
        boxes > (std::numeric_limits<std::uint32_t>::max() - 255u) / 2u)
        throw std::invalid_argument("Invalid PhysX region-producer dimensions");
    const Parameters params{descriptor_offset, arena_bytes, {}};
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {kGroups, 1u, 1u};
    command.buffers = {{0u, arena, 0u, arena_bytes, compute::BufferAccess::ReadWrite}};
    command.push_constants.resize(sizeof(params));
    std::memcpy(command.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
