// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/aabb_encoding.h"

#include <cstring>
#include <limits>
#include <array>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {

constexpr std::uint32_t kThreadsPerWorkgroup = 128; // Vulkan core minimum.
constexpr std::uint32_t kAabbsPerWorkgroup = kThreadsPerWorkgroup / 8; // PhysX 8 lanes/AABB.
constexpr std::size_t kCoordinateCount = 6;

// Keep the established 8-DWORD native shader ABI (D3D12 root constants),
// packing original PhysX byte-sized shifts to make room for four binding
// element offsets. This is one shared Vulkan/D3D12 shader, not a fallback.
struct alignas(4) ShaderParameters {
    std::uint32_t count;
    std::uint32_t shifts_xyz;
    std::uint32_t env_shifts_xyz;
    std::uint32_t env_enabled;
    std::uint32_t input_bias;
    std::uint32_t distance_bias;
    std::uint32_t environment_bias;
    std::uint32_t output_bias;
};
std::uint32_t pack_xyz(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return x | (y << 8u) | (z << 16u);
}

static_assert(sizeof(ShaderParameters) == 32, "Shader push-constant layout mismatch");

void validate(const AabbEncodingParameters& parameters) {
    if (!parameters.count) {
        throw std::invalid_argument("PhysX AABB encoding requires a nonzero capacity");
    }
    if (parameters.shift_x > 31 || parameters.shift_y > 31 ||
        parameters.shift_z > 31 || parameters.env_bits_x > parameters.shift_x ||
        parameters.env_bits_y > parameters.shift_y ||
        parameters.env_bits_z > parameters.shift_z) {
        throw std::invalid_argument("Invalid PhysX GPU broadphase bit configuration");
    }
}

} // namespace

AlignedAabbViews lower_aabb_storage_views(
    const std::array<compute::BufferBinding, 4>& requested,
    std::uint64_t descriptor_alignment,
    std::uint64_t max_descriptor_bytes) {
    // minStorageBufferOffsetAlignment is fixed by the selected device,
    // never queried here or when appending the source PhysX CUDA launch.
    if (!descriptor_alignment || descriptor_alignment % 4u != 0u ||
        !max_descriptor_bytes)
        throw std::invalid_argument("Invalid native storage-buffer alignment");

    AlignedAabbViews result{};
    for (std::size_t i = 0; i < requested.size(); ++i) {
        const auto& view = requested[i];
        if (view.slot != i || !view.buffer || !view.bytes ||
            view.offset % 4u != 0u || view.bytes % 4u != 0u)
            throw std::invalid_argument("Invalid PhysX AABB CUDA buffer view");

        const auto remainder = static_cast<std::uint64_t>(view.offset) %
                               descriptor_alignment;
        if (remainder % 4u != 0u ||
            remainder / 4u > std::numeric_limits<std::uint32_t>::max() ||
            remainder > max_descriptor_bytes ||
            static_cast<std::uint64_t>(view.bytes) >
                max_descriptor_bytes - remainder)
            throw std::out_of_range("PhysX AABB aligned storage view exceeds limits");
        const auto aligned = static_cast<std::size_t>(
            static_cast<std::uint64_t>(view.offset) - remainder);
        const auto total = static_cast<std::uint64_t>(view.bytes) + remainder;
        if (remainder > max_descriptor_bytes ||
            total > max_descriptor_bytes ||
            total > std::numeric_limits<std::size_t>::max())
            throw std::out_of_range("PhysX AABB descriptor expansion exceeds limit");

        result.bindings[i] = view;
        result.bindings[i].offset = aligned;
        result.bindings[i].bytes = static_cast<std::size_t>(total);
        result.element_bias[i] = static_cast<std::uint32_t>(remainder / 4u);
    }
    return result;
}

AabbEncodingStage::AabbEncodingStage(compute::Context& context,
                                     compute::ShaderBinary shader)
    : context_(context) {
    compute::ComputePipelineDesc description;
    description.shader = std::move(shader);
    description.storage_buffer_binding_count = 4;
    description.push_constant_bytes = sizeof(ShaderParameters);
    const auto& device = context_.device_info();
    descriptor_alignment_ = device.storage_buffer_offset_alignment;
    max_descriptor_range_ = device.max_storage_buffer_range;
    max_workgroup_x_ = device.max_workgroup_count[0];
    if (device.max_workgroup_invocations < kThreadsPerWorkgroup ||
        device.max_workgroup_size[0] < kThreadsPerWorkgroup) {
        throw std::runtime_error("PhysX AABB encoding requires a 128-thread workgroup");
    }
    pipeline_ = context_.create_compute_pipeline(description);
}

AabbEncodingStage::~AabbEncodingStage() {
    if (pipeline_) context_.destroy_compute_pipeline(pipeline_);
}

void AabbEncodingStage::append(std::vector<compute::BatchCommand>& commands,
                               compute::BufferId bounds,
                               compute::BufferId contact_distances,
                               compute::BufferId environment_ids,
                               compute::BufferId encoded_bounds,
                               const AabbEncodingParameters& parameters) const {
    // Source-lowered CUDA pointer views use the same path as normal
    // zero-offset native stage callers: no second shader implementation.
    const std::uint64_t array_bytes =
        static_cast<std::uint64_t>(parameters.count) * kCoordinateCount * sizeof(float);
    const std::uint64_t scalar_bytes =
        static_cast<std::uint64_t>(parameters.count) * sizeof(float);
    if (array_bytes > std::numeric_limits<std::size_t>::max())
        throw std::out_of_range("PhysX AABB array exceeds host address space");
    std::array<compute::BufferBinding, 4> views{{
        {0u, bounds, 0u, static_cast<std::size_t>(array_bytes), compute::BufferAccess::Read},
        {1u, contact_distances, 0u, static_cast<std::size_t>(scalar_bytes), compute::BufferAccess::Read},
        {2u, environment_ids ? environment_ids : contact_distances, 0u,
         static_cast<std::size_t>(scalar_bytes), compute::BufferAccess::Read},
        {3u, encoded_bounds, 0u, static_cast<std::size_t>(array_bytes), compute::BufferAccess::Write}
    }};
    append_views(commands, views, parameters, environment_ids != 0u);
}

void AabbEncodingStage::append_views(
    std::vector<compute::BatchCommand>& commands,
    const std::array<compute::BufferBinding, 4>& views,
    const AabbEncodingParameters& parameters,
    bool environment_enabled) const {
    validate(parameters);
    const std::uint64_t groups =
        (static_cast<std::uint64_t>(parameters.count) + kAabbsPerWorkgroup - 1) /
        kAabbsPerWorkgroup;
    if (groups > max_workgroup_x_)
        throw std::out_of_range("PhysX AABB encoding exceeds GPU dispatch limit");

    const std::uint64_t array_bytes =
        static_cast<std::uint64_t>(parameters.count) * kCoordinateCount * sizeof(float);
    const std::uint64_t scalar_bytes =
        static_cast<std::uint64_t>(parameters.count) * sizeof(float);
    if (array_bytes > std::numeric_limits<std::size_t>::max())
        throw std::out_of_range("PhysX AABB array exceeds host address space");

    const auto expected = {array_bytes, scalar_bytes, scalar_bytes, array_bytes};
    std::size_t index = 0u;
    for (const auto required : expected) {
        const auto& binding = views[index];
        if (binding.slot != index || !binding.buffer ||
            binding.offset % alignof(std::uint32_t) != 0u ||
            binding.bytes != required)
            throw std::invalid_argument("Invalid PhysX CUDA-lowered AABB view");
        ++index;
    }
    const auto lowered = lower_aabb_storage_views(
        views, descriptor_alignment_, max_descriptor_range_);
    const ShaderParameters push{
        parameters.count,
        pack_xyz(parameters.shift_x, parameters.shift_y, parameters.shift_z),
        pack_xyz(parameters.env_bits_x, parameters.env_bits_y, parameters.env_bits_z),
        environment_enabled ? 1u : 0u,
        lowered.element_bias[0], lowered.element_bias[1],
        lowered.element_bias[2], lowered.element_bias[3]
    };
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {static_cast<std::uint32_t>(groups), 1u, 1u};
    command.buffers.assign(lowered.bindings.begin(), lowered.bindings.end());
    command.push_constants.resize(sizeof(push));
    std::memcpy(command.push_constants.data(), &push, sizeof(push));
    commands.emplace_back(std::move(command));
}

} // namespace fyzz::physx_gpu
