// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
    try {
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
        bool rejected_null = false;
        try {
            fyzz::World invalid(std::unique_ptr<fyzz::WorldBackend>{});
        } catch (const std::invalid_argument&) {
            rejected_null = true;
        }
        if (!rejected_null) {
            throw std::runtime_error("Null World backend was accepted");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FyzzXYZZY headless smoke failed: " << error.what() << '\n';
        return 2;
    }
}
