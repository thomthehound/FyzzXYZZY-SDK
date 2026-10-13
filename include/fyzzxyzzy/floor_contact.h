// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"

#include <cstdint>
#include <vector>

namespace fyzz::compute {

// GPU storage-buffer element layout: std430 vec4. Position.w stores body
// half-height; velocity.w stores normal restitution in the range [0, 1].
struct alignas(16) FloorBodyVector {
    float x;
    float y;
    float z;
    float w;
};

// Matches the rigid_floor.comp push-constant layout. The caller owns scene
// state and body buffers; this operation does not create a second physics world.
struct FloorStepParameters {
    std::uint32_t count = 0;
    float dt = 0.0f;
    float floor_y = 0.0f;
    float gravity_y = -9.81f;
    float tangent_retention = 1.0f;
    float resting_speed = 0.0f;
};

// A reusable compute operation for independent, axis-aligned bodies
// contacting one infinite horizontal floor. No body-body contact or rotation.
// The Context must outlive the operation.
class FloorContactKernel final {
public:
    FloorContactKernel(Context& context, ShaderBinary compiled_shader);
    ~FloorContactKernel();

    FloorContactKernel(const FloorContactKernel&) = delete;
    FloorContactKernel& operator=(const FloorContactKernel&) = delete;
    FloorContactKernel(FloorContactKernel&&) = delete;
    FloorContactKernel& operator=(FloorContactKernel&&) = delete;

    // Appends a dispatch to an existing ordered batch. Validates parameters
    // without changing the batch on failure. Buffer allocation bounds are
    // checked by Context when the batch is submitted.
    void append_step(std::vector<BatchCommand>& batch,
                     BufferId positions, BufferId velocities,
                     const FloorStepParameters& parameters) const;

    // Single-step conveniences. The nonblocking variant returns a fence-backed
    // submission; callers may instead append several steps to one batch.
    SubmissionId submit_step(BufferId positions, BufferId velocities,
                             const FloorStepParameters& parameters) const;
    void step(BufferId positions, BufferId velocities,
              const FloorStepParameters& parameters) const;

private:
    Context& context_;
    PipelineId pipeline_;
};

} // namespace fyzz::compute
