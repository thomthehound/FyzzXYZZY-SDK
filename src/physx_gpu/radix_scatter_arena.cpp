// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/radix_scatter_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kThreads = 1024;
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t num_keys;
    std::uint32_t start_bit;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[4] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX radix-scatter root layout changed");
}

RadixScatterArenaStage::RadixScatterArenaStage(compute::Context& context,
                                               compute::ShaderBinary shader)
    : context_(context) {
    const auto& device = context_.device_info();
    if (device.max_workgroup_invocations < kThreads ||
        device.max_workgroup_size[0] < kThreads ||
        device.max_workgroup_count[0] < kBlocks ||
        device.max_workgroup_count[1] < 3)
        throw std::runtime_error("PhysX radix-scatter requires source 32x3x1024 launch");
    compute::ComputePipelineDesc desc;
    desc.shader = std::move(shader);
    desc.storage_buffer_binding_count = 1;
    desc.push_constant_bytes = sizeof(Parameters);
    pipeline_ = context_.create_compute_pipeline(desc);
}

RadixScatterArenaStage::~RadixScatterArenaStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void RadixScatterArenaStage::append(std::vector<compute::BatchCommand>& commands,
                                   compute::BufferId arena, std::uint32_t arena_bytes,
                                   std::uint32_t descriptor_offset,
                                   std::uint32_t num_keys, std::uint32_t start_bit) const {
    if (!num_keys) return; // Same host sort gate as PhysX.
    if (!arena || !arena_bytes || arena_bytes % 4 ||
        descriptor_offset % alignof(RadixScatterArenaDesc) ||
        descriptor_offset > arena_bytes ||
        sizeof(RadixScatterArenaDesc) > arena_bytes - descriptor_offset ||
        start_bit > 28 || start_bit % 4)
        throw std::invalid_argument("Invalid PhysX radix-scatter descriptor or bit pass");

    const Parameters params{descriptor_offset, num_keys, start_bit, arena_bytes, {}};
    compute::DispatchCommand dispatch;
    dispatch.pipeline = pipeline_;
    dispatch.groups = {kBlocks, 3, 1};
    dispatch.buffers = {{0, arena, 0, arena_bytes, compute::BufferAccess::ReadWrite}};
    dispatch.push_constants.resize(sizeof(params));
    std::memcpy(dispatch.push_constants.data(), &params, sizeof(params));
    commands.emplace_back(std::move(dispatch));
}
} // namespace fyzz::physx_gpu
