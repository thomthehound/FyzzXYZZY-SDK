// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/floor_contact.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::compute {
namespace {

constexpr std::uint32_t kLocalSize = 64;

static_assert(sizeof(FloorBodyVector) == 16, "Floor GPU state must match vec4");
static_assert(alignof(FloorBodyVector) == 16, "Floor GPU state must be vec4-aligned");
static_assert(sizeof(FloorStepParameters) == 24, "Floor push constants must match shader");
static_assert(offsetof(FloorStepParameters, gravity_y) == 12, "Unexpected push-constant layout");

void validate(const Context& context, BufferId positions, BufferId velocities,
              const FloorStepParameters& values) {
    if (positions == 0 || velocities == 0 || positions == velocities) {
        throw std::invalid_argument("Floor state needs two distinct valid buffers");
    }
    if (values.count == 0 || !std::isfinite(values.dt) || values.dt <= 0.0f ||
        !std::isfinite(values.floor_y) || !std::isfinite(values.gravity_y) ||
        !std::isfinite(values.tangent_retention) ||
        values.tangent_retention < 0.0f || values.tangent_retention > 1.0f ||
        !std::isfinite(values.resting_speed) || values.resting_speed < 0.0f) {
        throw std::invalid_argument("Invalid floor-contact step parameters");
    }
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(values.count) * sizeof(FloorBodyVector);
    if (bytes > std::numeric_limits<std::size_t>::max() ||
        bytes > context.device_info().max_storage_buffer_range) {
        throw std::out_of_range("Floor-contact body data exceeds buffer limits");
    }
    const std::uint32_t groups =
        values.count / kLocalSize + (values.count % kLocalSize != 0u ? 1u : 0u);
    if (groups > context.device_info().max_workgroup_count[0]) {
        throw std::out_of_range("Floor-contact dispatch exceeds device workgroup limits");
    }
}

} // namespace

FloorContactKernel::FloorContactKernel(Context& context, ShaderBinary compiled_shader)
    : context_(context),
      pipeline_(context.create_compute_pipeline(ComputePipelineDesc{
          std::move(compiled_shader), 2, sizeof(FloorStepParameters)})) {}

FloorContactKernel::~FloorContactKernel() {
    // Context cleanup remains responsible for recovery from device loss.
    try {
        context_.destroy_compute_pipeline(pipeline_);
    } catch (...) {
    }
}

void FloorContactKernel::append_step(std::vector<BatchCommand>& batch,
                                     BufferId positions, BufferId velocities,
                                     const FloorStepParameters& parameters) const {
    validate(context_, positions, velocities, parameters);
    const auto bytes = static_cast<std::size_t>(
        static_cast<std::uint64_t>(parameters.count) * sizeof(FloorBodyVector));
    std::vector<std::uint8_t> constants(sizeof(parameters));
    std::memcpy(constants.data(), &parameters, sizeof(parameters));
    const std::uint32_t groups = parameters.count / kLocalSize +
        (parameters.count % kLocalSize != 0u ? 1u : 0u);
    batch.emplace_back(DispatchCommand{
        pipeline_, {groups, 1, 1},
        {{0, positions, 0, bytes}, {1, velocities, 0, bytes}},
        std::move(constants)
    });
}

SubmissionId FloorContactKernel::submit_step(
        BufferId positions, BufferId velocities,
        const FloorStepParameters& parameters) const {
    std::vector<BatchCommand> batch;
    append_step(batch, positions, velocities, parameters);
    return context_.submit_batch(batch);
}

void FloorContactKernel::step(BufferId positions, BufferId velocities,
                              const FloorStepParameters& parameters) const {
    std::vector<BatchCommand> batch;
    append_step(batch, positions, velocities, parameters);
    context_.execute_batch(batch);
}

} // namespace fyzz::compute
