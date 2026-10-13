// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/floor_contact.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using BodyVector = fyzz::compute::FloorBodyVector;
using StepParameters = fyzz::compute::FloorStepParameters;

static_assert(sizeof(BodyVector) == 16, "GPU state must match std430 vec4");
static_assert(alignof(BodyVector) == 16, "GPU state must have vec4 alignment");
static_assert(sizeof(StepParameters) == 24, "GPU push constants must match shader layout");

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open compiled floor-contact shader");
    }
    const std::vector<char> data(std::istreambuf_iterator<char>{stream},
                                 std::istreambuf_iterator<char>{});
    return {data.begin(), data.end()};
}

void cpu_step(std::vector<BodyVector>& positions,
              std::vector<BodyVector>& velocities,
              const StepParameters& parameters,
              std::uint32_t& contact_count) {
    for (std::size_t i = 0; i < positions.size(); ++i) {
        BodyVector& p = positions[i];
        BodyVector& v = velocities[i];
        v.y += parameters.gravity_y * parameters.dt;
        p.x += v.x * parameters.dt;
        p.y += v.y * parameters.dt;
        p.z += v.z * parameters.dt;
        const float contact_y = parameters.floor_y + p.w;
        if (p.y < contact_y) {
            p.y = contact_y;
            ++contact_count;
            if (v.y < 0.0f) {
                const float impact_speed = -v.y;
                const float restitution = std::max(0.0f, std::min(1.0f, v.w));
                v.y = impact_speed <= parameters.resting_speed
                    ? 0.0f : impact_speed * restitution;
                v.x *= parameters.tangent_retention;
                v.z *= parameters.tangent_retention;
            }
        }
    }
}

float difference(const BodyVector& a, const BodyVector& b) {
    return std::max({std::fabs(a.x - b.x), std::fabs(a.y - b.y),
                     std::fabs(a.z - b.z), std::fabs(a.w - b.w)});
}

void compare(const std::vector<BodyVector>& actual,
             const std::vector<BodyVector>& expected,
             const StepParameters& parameters, std::uint32_t frame,
             const char* category, float& maximum_error) {
    constexpr float tolerance = 0.012f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const BodyVector& a = actual[i];
        const float error = difference(a, expected[i]);
        maximum_error = std::max(maximum_error, error);
        if (!std::isfinite(error) || error > tolerance) {
            throw std::runtime_error(std::string(category) + " divergence at frame " +
                                     std::to_string(frame) + " body " +
                                     std::to_string(i) + " (error " +
                                     std::to_string(error) + ")");
        }
        if (category[0] == 'p' && a.y < parameters.floor_y + a.w - 0.0001f) {
            throw std::runtime_error("GPU body penetrated the static floor");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected SPIR-V shader path");
        }
        using namespace fyzz::compute;

        constexpr std::uint32_t kCount = 513; // Cross several workgroups.
        constexpr std::uint32_t kFrames = 160;
        constexpr std::uint32_t kBatchLength = 8;
        const StepParameters parameters{
            kCount, 1.0f / 120.0f, -1.25f, -9.81f, 0.93f, 0.12f
        };

        std::vector<BodyVector> position(kCount);
        std::vector<BodyVector> velocity(kCount);
        for (std::uint32_t i = 0; i < kCount; ++i) {
            const float half_height = 0.1f + static_cast<float>(i % 7u) * 0.05f;
            position[i] = {
                static_cast<float>(i % 29u) * 0.11f,
                parameters.floor_y + half_height +
                    (i == 0 ? 0.0f : 0.16f + static_cast<float>(i % 17u) * 0.14f),
                static_cast<float>(i / 29u) * 0.13f,
                half_height
            };
            velocity[i] = {
                (static_cast<float>(i % 9u) - 4.0f) * 0.08f,
                (static_cast<float>(i % 11u) - 5.0f) * 0.17f,
                (static_cast<float>(i % 13u) - 6.0f) * 0.06f,
                static_cast<float>(i % 5u) * 0.2f
            };
        }
        const std::vector<BodyVector> initial_position = position;
        const std::vector<BodyVector> initial_velocity = velocity;

        Context context(Backend::Vulkan, 0);
        const BufferId gpu_position = context.create_buffer(
            position.size() * sizeof(BodyVector), MemoryPlacement::DeviceLocal);
        const BufferId gpu_velocity = context.create_buffer(
            velocity.size() * sizeof(BodyVector), MemoryPlacement::DeviceLocal);
        context.upload(gpu_position, 0, position.data(), position.size() * sizeof(BodyVector));
        context.upload(gpu_velocity, 0, velocity.data(), velocity.size() * sizeof(BodyVector));

        ShaderBinary shader;
        shader.bytes = read_shader(argv[1]);
        FloorContactKernel physics(context, std::move(shader));

        float maximum_error = 0.0f;
        std::uint32_t contacts = 0;
        for (std::uint32_t frame = 0; frame < kFrames; frame += kBatchLength) {
            std::vector<BatchCommand> batch;
            batch.reserve(kBatchLength);
            for (std::uint32_t step = 0; step < kBatchLength; ++step) {
                physics.append_step(batch, gpu_position, gpu_velocity, parameters);
                cpu_step(position, velocity, parameters, contacts);
            }
            const SubmissionId submitted = context.submit_batch(batch);
            context.wait_submission(submitted);

            std::vector<BodyVector> measured_position(kCount);
            std::vector<BodyVector> measured_velocity(kCount);
            context.download(gpu_position, 0, measured_position.data(),
                             measured_position.size() * sizeof(BodyVector));
            context.download(gpu_velocity, 0, measured_velocity.data(),
                             measured_velocity.size() * sizeof(BodyVector));
            compare(measured_position, position, parameters, frame + kBatchLength,
                    "position", maximum_error);
            compare(measured_velocity, velocity, parameters, frame + kBatchLength,
                    "velocity", maximum_error);
        }
        if (contacts < kCount) {
            throw std::runtime_error("Acceptance scenario did not exercise floor contact");
        }
        if (position[0].y < parameters.floor_y + position[0].w ||
            velocity[0].w != initial_velocity[0].w ||
            position[0].w != initial_position[0].w) {
            throw std::runtime_error("Reference state contract was not preserved");
        }

        // Invalid parameters must be rejected without changing the batch.
        std::vector<BatchCommand> rejected_batch;
        auto invalid = parameters;
        invalid.dt = 0.0f;
        bool rejected = false;
        try {
            physics.append_step(rejected_batch, gpu_position, gpu_velocity, invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !rejected_batch.empty()) {
            throw std::runtime_error("Invalid physics step modified a command batch");
        }
        rejected = false;
        try {
            physics.append_step(rejected_batch, gpu_position, gpu_position, parameters);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !rejected_batch.empty()) {
            throw std::runtime_error("Aliased physics state buffers were accepted");
        }

        // Exercise both public one-step entry points in addition to batching.
        cpu_step(position, velocity, parameters, contacts);
        const SubmissionId additional = physics.submit_step(
            gpu_position, gpu_velocity, parameters);
        context.wait_submission(additional);
        cpu_step(position, velocity, parameters, contacts);
        physics.step(gpu_position, gpu_velocity, parameters);
        std::vector<BodyVector> final_position(kCount);
        std::vector<BodyVector> final_velocity(kCount);
        context.download(gpu_position, 0, final_position.data(),
                         final_position.size() * sizeof(BodyVector));
        context.download(gpu_velocity, 0, final_velocity.data(),
                         final_velocity.size() * sizeof(BodyVector));
        compare(final_position, position, parameters, kFrames + 2,
                "position", maximum_error);
        compare(final_velocity, velocity, parameters, kFrames + 2,
                "velocity", maximum_error);

        // The kernel owns its pipeline; buffers remain owned by the caller.
        context.destroy_buffer(gpu_velocity);
        context.destroy_buffer(gpu_position);
        std::cout << "Vulkan floor-contact physics acceptance PASS; "
                  << kCount << " bodies, " << kFrames + 2 << " steps, "
                  << contacts << " CPU reference contacts, "
                  << "max error " << maximum_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan floor-contact physics acceptance FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
