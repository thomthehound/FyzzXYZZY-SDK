// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main() {
    try {
        {
            // The real production constructor must never report GPU
            // admission when the underlying PhysX scene is CPU-only.
            bool rejected_gpu = false;
            try {
                fyzz::World unsupported(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativePortableGpu});
            } catch (const std::logic_error& e) {
                rejected_gpu = std::string(e.what()).find("not yet implemented")
                    != std::string::npos;
            }
            if (!rejected_gpu)
                throw std::runtime_error("Native GPU request silently created reference PhysX scene");

            bool rejected_native_cpu = false;
            try {
                fyzz::World unsupported(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
            } catch (const std::logic_error& error) {
                rejected_native_cpu =
                    std::string(error.what()).find("CPU scene and solver") !=
                    std::string::npos;
            }
            if (!rejected_native_cpu)
                throw std::runtime_error("Native CPU request silently created reference PhysX scene");

            bool rejected_unknown = false;
            try {
                fyzz::World unsupported(fyzz::WorldOptions{
                    static_cast<fyzz::ExecutionMode>(0xff)});
            } catch (const std::invalid_argument&) {
                rejected_unknown = true;
            }
            if (!rejected_unknown)
                throw std::runtime_error("Invalid SDK execution selection was accepted");

            fyzz::World explicit_cpu(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            explicit_cpu.add_static_floor(0.0f);
            const auto cpu_test = explicit_cpu.add_dynamic_box(
                {0.2f, 0.2f, 0.2f}, {0.0f, 1.0f, 0.0f}, 1.0f);
            explicit_cpu.step(1.0f / 60.0f);
            if (explicit_cpu.body_state(cpu_test).position.y >= 1.0f)
                throw std::runtime_error("Explicit production CPU PhysX scene failed");

            fyzz::World world;
            world.add_static_box({5.0f, 0.5f, 5.0f}, {0.0f, -0.5f, 0.0f});
            const fyzz::BodyId box = world.add_dynamic_box(
                {0.25f, 0.25f, 0.25f}, {0.0f, 2.0f, 0.0f}, 1.0f);
            const float initial_y = world.body_state(box).position.y;

            constexpr float dt = 1.0f / 60.0f;
            for (int frame = 0; frame < 180; ++frame) {
                world.step(dt);
            }

            const fyzz::BodyState state = world.body_state(box);
            std::cout << "FyzzXYZZY headless CPU: initial_y=" << initial_y
                      << " final_y=" << state.position.y
                      << " vy=" << state.velocity.y << '\n';

            // Loose, behavioral checks: falling, contact with floor, then rest.
            if (!std::isfinite(state.position.y) ||
                state.position.y >= initial_y ||
                state.position.y < 0.1f || state.position.y > 0.6f ||
                !std::isfinite(state.velocity.y) || std::abs(state.velocity.y) > 1.0f) {
                std::cerr << "Rigid-body fall/contact smoke test failed\n";
                return 1;
            }
            // The provider-neutral floor API must also work with the default
            // PhysX backend, independently of the optional Vulkan target.
            fyzz::World floor_world;
            floor_world.add_static_floor(0.0f);
            const auto plane_box = floor_world.add_dynamic_box(
                {0.25f, 0.25f, 0.25f}, {0.0f, 1.5f, 0.0f}, 1.0f);
            for (int frame = 0; frame < 240; ++frame) {
                floor_world.step(1.0f / 120.0f);
            }
            const auto plane_state = floor_world.body_state(plane_box);
            if (!std::isfinite(plane_state.position.y) ||
                std::abs(plane_state.position.y - 0.25f) > 0.08f) {
                throw std::runtime_error("CPU infinite-floor contact failed");
            }
            // Both scenes coexist, but stepping one must not advance the other.
            const auto unchanged = world.body_state(box);
            if (std::abs(unchanged.position.y - state.position.y) > 0.0001f ||
                std::abs(unchanged.velocity.y - state.velocity.y) > 0.0001f) {
                throw std::runtime_error("Independent CPU World scenes changed each other");
            }
            // The runtime must survive destruction of one of several live scenes.
            {
                fyzz::World temporary;
                temporary.add_static_floor(-1.0f);
                const auto temporary_body = temporary.add_dynamic_box(
                    {0.2f, 0.2f, 0.2f}, {0.0f, 3.0f, 0.0f}, 1.0f);
                temporary.step(1.0f / 60.0f);
                if (temporary.body_state(temporary_body).position.y >= 3.0f) {
                    throw std::runtime_error("Temporary CPU world did not advance");
                }
            }
            floor_world.step(1.0f / 120.0f);
            if (!std::isfinite(floor_world.body_state(plane_box).position.y)) {
                throw std::runtime_error("Remaining CPU world failed after peer destruction");
            }
            bool rejected_null = false;
            try {
                fyzz::World invalid(std::unique_ptr<fyzz::WorldBackend>{});
            } catch (const std::invalid_argument&) {
                rejected_null = true;
            }
            if (!rejected_null) {
                throw std::runtime_error("Null World backend was accepted");
            }
        } // Release every CPU World and its shared PhysX runtime.

        // Creating a new world must also work after the last previous user
        // releases the process-level foundation/physics singleton.
        fyzz::World reinitialized;
        reinitialized.add_static_floor(0.0f);
        const auto renewed = reinitialized.add_dynamic_box(
            {0.25f, 0.25f, 0.25f}, {0.0f, 1.0f, 0.0f}, 1.0f);
        reinitialized.step(1.0f / 120.0f);
        if (reinitialized.body_state(renewed).position.y >= 1.0f) {
            throw std::runtime_error("CPU PhysX runtime recreation failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FyzzXYZZY headless smoke failed: " << error.what() << '\n';
        return 2;
    }
}
