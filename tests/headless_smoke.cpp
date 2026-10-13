// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <exception>
#include <iostream>

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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FyzzXYZZY headless smoke failed: " << error.what() << '\n';
        return 2;
    }
}
