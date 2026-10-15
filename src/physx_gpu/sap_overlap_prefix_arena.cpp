// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_overlap_prefix_arena.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
struct Parameters {
    std::uint32_t descriptor_offset;
    std::uint32_t arena_bytes;
    std::uint32_t reserved[6] = {};
};
static_assert(sizeof(Parameters) == 32, "PhysX overlap prefix parameters");
}

SapOverlapPrefixArenaStage::SapOverlapPrefixArenaStage(
    compute::Context& context, compute::ShaderBinary compute_wave,
    compute::ShaderBinary output_wave) : context_(context) {
    const auto& device = context_.device_info();
    // Require native subgroup arithmetic, but never assume CUDA warp32.
    // Physical waves of 4..128 lanes are handled by the scan hierarchy.
    if (!device.supports_int64_wave || !device.supports_wave_arithmetic ||
        device.subgroup_size < 4u || device.subgroup_size > 128u ||
        256u % device.subgroup_size != 0u ||
        device.max_workgroup_invocations < 256u ||
        device.max_workgroup_size[0] < 256u ||
        device.max_workgroup_count[0] < 32u)
        throw std::runtime_error(
            "PhysX overlap prefix requires 64-bit subgroup arithmetic "
            "with subgroup waves dividing 256 and 32x256 dispatch support");
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count = 1u;
    desc.push_constant_bytes = sizeof(Parameters);
    try {
        desc.shader = std::move(compute_wave);
        pipelines_[0] = context_.create_compute_pipeline(desc);
        desc.shader = std::move(output_wave);
        pipelines_[1] = context_.create_compute_pipeline(desc);
    } catch (...) {
        for (auto p : pipelines_) if (p) context_.destroy_compute_pipeline(p);
        throw;
    }
}
SapOverlapPrefixArenaStage::~SapOverlapPrefixArenaStage() {
    for (auto p : pipelines_) if (p) context_.destroy_compute_pipeline(p);
}
void SapOverlapPrefixArenaStage::append(
    std::vector<compute::BatchCommand>& commands, compute::BufferId arena,
    std::uint32_t arena_bytes, std::uint32_t descriptor_offset,
    std::uint32_t created_handles) const {
    if (!created_handles) return;
    if (!arena || !arena_bytes || arena_bytes % 4u ||
        descriptor_offset % alignof(SapOverlapPrefixArenaDesc) != 0u ||
        descriptor_offset > arena_bytes ||
        sizeof(SapOverlapPrefixArenaDesc) > arena_bytes - descriptor_offset)
        throw std::invalid_argument("Invalid PhysX overlap prefix arena");
    const Parameters params{descriptor_offset, arena_bytes, {}};
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2u);
    for (auto pipeline : pipelines_) {
        compute::DispatchCommand command;
        command.pipeline = pipeline;
        command.groups = {32u, 1u, 1u};
        command.buffers = {{0u, arena, 0u, arena_bytes,
                            compute::BufferAccess::ReadWrite}};
        command.push_constants.resize(sizeof(params));
        std::memcpy(command.push_constants.data(), &params, sizeof(params));
        staged.emplace_back(std::move(command));
    }
    commands.reserve(commands.size() + staged.size());
    for (auto& command : staged) commands.emplace_back(std::move(command));
}
} // namespace fyzz::physx_gpu
