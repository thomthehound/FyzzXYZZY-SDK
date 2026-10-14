// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/aabb_encoding.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {

constexpr std::uint32_t kThreadsPerWorkgroup = 128; // Vulkan core minimum.
constexpr std::uint32_t kAabbsPerWorkgroup = kThreadsPerWorkgroup / 8; // PhysX 8 lanes/AABB.
constexpr std::size_t kCoordinateCount = 6;

struct alignas(4) ShaderParameters {
    std::uint32_t count;
    std::uint32_t shift_x;
    std::uint32_t shift_y;
    std::uint32_t shift_z;
    std::uint32_t env_bits_x;
    std::uint32_t env_bits_y;
    std::uint32_t env_bits_z;
    std::uint32_t env_enabled;
};

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

AabbEncodingStage::AabbEncodingStage(compute::Context& context,
                                     compute::ShaderBinary shader)
    : context_(context) {
    compute::ComputePipelineDesc description;
    description.shader = std::move(shader);
    description.storage_buffer_binding_count = 4;
    description.push_constant_bytes = sizeof(ShaderParameters);
    const auto& device = context_.device_info();
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
    validate(parameters);

    const std::uint64_t groups =
        (static_cast<std::uint64_t>(parameters.count) + kAabbsPerWorkgroup - 1) /
        kAabbsPerWorkgroup;
    if (groups > context_.device_info().max_workgroup_count[0]) {
        throw std::out_of_range("PhysX AABB encoding exceeds GPU dispatch limit");
    }

    const std::uint64_t array_bytes =
        static_cast<std::uint64_t>(parameters.count) * kCoordinateCount * sizeof(float);
    const std::uint64_t single_bytes =
        static_cast<std::uint64_t>(parameters.count) * sizeof(float);
    if (array_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range("PhysX AABB array exceeds host address space");
    }

    const ShaderParameters push{
        parameters.count, parameters.shift_x, parameters.shift_y, parameters.shift_z,
        parameters.env_bits_x, parameters.env_bits_y, parameters.env_bits_z,
        environment_ids != 0 ? 1u : 0u
    };
    compute::DispatchCommand command;
    command.pipeline = pipeline_;
    command.groups = {static_cast<std::uint32_t>(groups), 1, 1};
    command.buffers = {
        {0, bounds, 0, static_cast<std::size_t>(array_bytes), compute::BufferAccess::Read},
        {1, contact_distances, 0, static_cast<std::size_t>(single_bytes), compute::BufferAccess::Read},
        // A dummy descriptor is used only when the optional PhysX env IDs
        // array is absent. The shader never reads it in that configuration.
        {2, environment_ids ? environment_ids : contact_distances, 0,
         static_cast<std::size_t>(single_bytes), compute::BufferAccess::Read},
        {3, encoded_bounds, 0, static_cast<std::size_t>(array_bytes), compute::BufferAccess::Write}
    };
    command.push_constants.resize(sizeof(push));
    std::memcpy(command.push_constants.data(), &push, sizeof(push));
    commands.emplace_back(std::move(command));
}

} // namespace fyzz::physx_gpu
