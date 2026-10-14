// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_box_indices_arena.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kThreads = 256;

struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t num_handles;
    std::uint32_t is_new;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(Parameters) == 32, "SAP index root constants changed");
}

SapBoxIndicesArenaStage::SapBoxIndicesArenaStage(compute::Context& context,
                                                 compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads)
        throw std::runtime_error("SAP initialization requires 256-thread groups");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

SapBoxIndicesArenaStage::~SapBoxIndicesArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void SapBoxIndicesArenaStage::append(std::vector<compute::BatchCommand>& commands,
                                    compute::BufferId arena, std::uint32_t arena_bytes,
                                    std::uint32_t descriptor_offset,
                                    std::uint32_t num_handles, bool is_new) const {
    if (!num_handles) return; // Original ceil(2*n/256) gate.
    const std::uint64_t projections = std::uint64_t{num_handles} * 2u;
    const std::uint64_t blocks = (projections + kThreads - 1u) / kThreads;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapBoxIndicesArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(SapBoxIndicesArenaDesc) > arena_bytes - descriptor_offset ||
        projections > std::numeric_limits<std::uint32_t>::max() ||
        blocks > context_.device_info().max_workgroup_count[0])
        throw std::invalid_argument("Invalid PhysX SAP index initialization dimensions");

    const Parameters params{descriptor_offset, num_handles, is_new ? 1u : 0u,
                            arena_bytes, {}};
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {static_cast<std::uint32_t>(blocks), 1, 1};
    dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
