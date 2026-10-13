// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/floor_world.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot load floor-contact shader");
    const std::vector<char> bytes(std::istreambuf_iterator<char>{stream},
                                  std::istreambuf_iterator<char>{});
    return {bytes.begin(), bytes.end()};
}

void check_rest(const fyzz::World& world, fyzz::BodyId body, float half_height) {
    const auto state = world.body_state(body);
    if (!std::isfinite(state.position.y) || !std::isfinite(state.velocity.y) ||
        std::abs(state.position.y - half_height) > 0.08f ||
        std::abs(state.velocity.y) > 0.35f) {
        throw std::runtime_error("World did not settle on the floor");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected SPIR-V shader path");
        fyzz::compute::ShaderBinary shader;
        shader.bytes = read_shader(argv[1]);
        fyzz::World gpu = fyzz::compute::create_floor_world(std::move(shader));
        fyzz::World cpu;

        if (gpu.add_static_floor(0.0f) != 0 ||
            cpu.add_static_floor(0.0f) != 0) {
            throw std::runtime_error("World body ID mapping inconsistent");
        }

        bool rejected = false;
        try {
            gpu.add_static_box({1, 1, 1}, {0, 0, 0});
        } catch (const std::logic_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Unsupported GPU static box was accepted");

        rejected = false;
        try {
            gpu.add_static_floor(1.0f);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Second GPU floor was accepted");

        constexpr float dt = 1.0f / 120.0f;
        constexpr float half_height = 0.25f;
        std::vector<fyzz::BodyId> gpu_ids;
        std::vector<fyzz::BodyId> cpu_ids;
        // Exceed the initial allocation capacity before simulation.
        for (int i = 0; i < 10; ++i) {
            const fyzz::Vec3 position{static_cast<float>(i) * 3.0f,
                                      1.5f + static_cast<float>(i % 3) * 0.3f, 0.0f};
            gpu_ids.push_back(gpu.add_dynamic_box(
                {half_height, half_height, half_height}, position, 1.0f));
            cpu_ids.push_back(cpu.add_dynamic_box(
                {half_height, half_height, half_height}, position, 1.0f));
        }

        const float initial = gpu.body_state(gpu_ids[0]).position.y;
        for (int frame = 0; frame < 180; ++frame) {
            gpu.step(dt);
            cpu.step(dt);
        }
        if (gpu.body_state(gpu_ids[0]).position.y >= initial) {
            throw std::runtime_error("GPU world body did not fall");
        }

        // Grow the GPU buffers after simulation has already changed their
        // contents. Old body state must survive the device-local GPU copy.
        const float before_growth = gpu.body_state(gpu_ids[0]).position.y;
        for (int i = 10; i < 20; ++i) {
            const fyzz::Vec3 position{static_cast<float>(i) * 3.0f,
                                      1.7f + static_cast<float>(i % 4) * 0.2f, 0.0f};
            gpu_ids.push_back(gpu.add_dynamic_box(
                {half_height, half_height, half_height}, position, 1.0f));
            cpu_ids.push_back(cpu.add_dynamic_box(
                {half_height, half_height, half_height}, position, 1.0f));
        }
        if (std::abs(gpu.body_state(gpu_ids[0]).position.y - before_growth) > 0.0001f) {
            throw std::runtime_error("Growing GPU body storage discarded simulation state");
        }

        for (int frame = 0; frame < 300; ++frame) {
            gpu.step(dt);
            cpu.step(dt);
        }
        for (std::size_t i = 0; i < gpu_ids.size(); ++i) {
            if (gpu_ids[i] != cpu_ids[i]) {
                throw std::runtime_error("CPU/GPU World body ID mapping differs");
            }
            check_rest(gpu, gpu_ids[i], half_height);
            check_rest(cpu, cpu_ids[i], half_height);
        }
        if (gpu.body_state(0).position.y != 0.0f ||
            cpu.body_state(0).position.y != 0.0f) {
            throw std::runtime_error("Static floor state was not preserved");
        }
        rejected = false;
        try {
            gpu.body_state(999999);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Unknown GPU body ID was accepted");

        rejected = false;
        try {
            fyzz::World invalid(std::unique_ptr<fyzz::WorldBackend>{});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Null World backend was accepted");

        std::cout << "Vulkan World floor integration PASS; "
                  << gpu_ids.size() << " dynamic bodies, 480 steps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan World floor integration FAILED: " << error.what() << '\n';
        return 1;
    }
}
