// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
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

            fyzz::World native_cpu(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            const auto native_id = native_cpu.add_dynamic_box(
                {0.25f,0.25f,0.25f}, {0,2,0}, 1);
            if (native_cpu.body_state(native_id).position.y != 2)
                throw std::runtime_error("Native CPU scene did not own actor state");
            native_cpu.step(1.0f / 60.0f);
            if (native_cpu.body_state(native_id).position.y >= 2.0f)
                throw std::runtime_error("Native CPU did not integrate gravity");
            native_cpu.set_body_position(native_id, {0,3,0});
            if (native_cpu.body_state(native_id).position.y != 3)
                throw std::runtime_error("Native CPU scene relocation failed");
            native_cpu.remove_body(native_id);
            bool retired_native_actor = false;
            try { (void)native_cpu.body_state(native_id); }
            catch (const std::out_of_range&) { retired_native_actor = true; }
            if (!retired_native_actor)
                throw std::runtime_error("Native CPU scene failed to retire actor");

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
            const auto reference_retire = explicit_cpu.add_dynamic_box(
                {0.2f,0.2f,0.2f}, {0,4,0}, 1.0f);
            explicit_cpu.set_body_position(reference_retire, {0,5,0});
            if (explicit_cpu.body_state(reference_retire).position.y != 5)
                throw std::runtime_error("Reference actor relocation failed");
            explicit_cpu.remove_body(reference_retire);
            bool reference_retired = false;
            try { (void)explicit_cpu.body_state(reference_retire); }
            catch (const std::out_of_range&) { reference_retired = true; }
            if (!reference_retired)
                throw std::runtime_error("Reference actor removal failed");
            const auto launched_reference = explicit_cpu.add_dynamic_box(
                {0.2f,0.2f,0.2f}, {0,5,0}, 1.0f);
            explicit_cpu.set_body_velocity(launched_reference, {2,0,0});
            if (explicit_cpu.body_state(launched_reference).velocity.x != 2)
                throw std::runtime_error("Reference dynamic velocity edit failed");
            bool reference_static_rejected = false;
            try { explicit_cpu.set_body_velocity(0, {1,0,0}); }
            catch (const std::logic_error&) { reference_static_rejected = true; }
            if (!reference_static_rejected)
                throw std::runtime_error("Reference static velocity edit was admitted");


            // Independent reference comparison: unconstrained angular
            // velocity damping, source quaternion orientation and pose.
            // Full contact/angular solver parity is not claimed here.
            fyzz::World native_rotation(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World reference_rotation(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto nrot = native_rotation.add_dynamic_box(
                {0.25f,0.5f,0.75f},{0,10,0},1);
            const auto rrot = reference_rotation.add_dynamic_box(
                {0.25f,0.5f,0.75f},{0,10,0},1);
            native_rotation.set_body_angular_velocity(nrot,{0,2,0});
            reference_rotation.set_body_angular_velocity(rrot,{0,2,0});
            for(int frame=0;frame<120;++frame){
                native_rotation.step(1.0f/120.0f);
                reference_rotation.step(1.0f/120.0f);
            }
            const auto nr = native_rotation.body_state(nrot);
            const auto rr = reference_rotation.body_state(rrot);
            const float orientation_dot =
                nr.orientation.x*rr.orientation.x +
                nr.orientation.y*rr.orientation.y +
                nr.orientation.z*rr.orientation.z +
                nr.orientation.w*rr.orientation.w;
            if (!std::isfinite(orientation_dot) ||
                std::abs(orientation_dot)<0.995f ||
                !std::isfinite(nr.angular_velocity.y) ||
                !std::isfinite(rr.angular_velocity.y) ||
                std::abs(nr.angular_velocity.y - rr.angular_velocity.y)>0.025f)
                throw std::runtime_error(
                    "Native quaternion/angular damping diverged from PhysX source");

            // PxCreatePlane creates a rotated local +X plane. Relocating
            // its position must preserve this quaternion: replacing it
            // with an identity rotation changes a floor into a wall.
            fyzz::World raised_floor(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto raised_floor_id = raised_floor.add_static_floor(0.0f);
            raised_floor.set_body_position(raised_floor_id, {0,0.75f,0});
            const auto dropped_on_raised_floor = raised_floor.add_dynamic_box(
                {0.25f,0.25f,0.25f}, {0,2.5f,0}, 1.0f);
            for (int frame = 0; frame < 300; ++frame)
                raised_floor.step(1.0f / 120.0f);
            const auto settled_above_floor =
                raised_floor.body_state(dropped_on_raised_floor);
            if (!std::isfinite(settled_above_floor.position.y) ||
                std::abs(settled_above_floor.position.y - 1.0f) > 0.08f ||
                std::abs(settled_above_floor.velocity.y) > 1.0f) {
                throw std::runtime_error(
                    "Relocating PhysX reference plane changed its orientation");
            }


            // Compare an actually integrated native scene against its
            // independent PhysX CPU oracle. Tolerances cover different
            // contact solvers; they do not excuse falling through the floor.
            fyzz::World native_compare(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World reference_compare(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            native_compare.add_static_floor(0.0f);
            reference_compare.add_static_floor(0.0f);
            const auto native_box = native_compare.add_dynamic_box(
                {0.25f,0.25f,0.25f}, {0,2,0}, 1);
            const auto ref_box = reference_compare.add_dynamic_box(
                {0.25f,0.25f,0.25f}, {0,2,0}, 1);
            for (int frame = 0; frame < 360; ++frame) {
                native_compare.step(1.0f / 120.0f);
                reference_compare.step(1.0f / 120.0f);
            }
            const auto ns = native_compare.body_state(native_box);
            const auto rs = reference_compare.body_state(ref_box);
            if (!std::isfinite(ns.position.y) ||
                !std::isfinite(rs.position.y) ||
                !std::isfinite(ns.velocity.y) ||
                !std::isfinite(rs.velocity.y) ||
                std::abs(ns.position.y - rs.position.y) > 0.09f ||
                std::abs(ns.velocity.y) > 0.6f ||
                std::abs(rs.velocity.y) > 0.6f) {
                throw std::runtime_error(
                    "Native CPU gravity/contact diverged from reference PhysX");
            }

            // Source parity for the float speed-squared overflow branch:
            // a FLT_MAX linear velocity must become zero, not 1e16.
            fyzz::World native_speed(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World reference_speed(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto native_fast = native_speed.add_dynamic_box(
                {0.5f,0.5f,0.5f}, {0,5,0}, 1);
            const auto reference_fast = reference_speed.add_dynamic_box(
                {0.5f,0.5f,0.5f}, {0,5,0}, 1);
            const float extreme_velocity = std::numeric_limits<float>::max();
            native_speed.set_body_velocity(native_fast, {extreme_velocity,0,0});
            reference_speed.set_body_velocity(reference_fast, {extreme_velocity,0,0});
            native_speed.step(1.0f / 60.0f);
            reference_speed.step(1.0f / 60.0f);
            const auto nv = native_speed.body_state(native_fast);
            const auto rv = reference_speed.body_state(reference_fast);
            const auto finite = [](fyzz::Vec3 v) {
                return std::isfinite(v.x) && std::isfinite(v.y) &&
                       std::isfinite(v.z);
            };
            if (!finite(nv.position) || !finite(rv.position) ||
                !finite(nv.velocity) || !finite(rv.velocity) ||
                std::abs(nv.position.x - rv.position.x) > 0.001f ||
                std::abs(nv.position.y - rv.position.y) > 0.01f ||
                std::abs(nv.position.z - rv.position.z) > 0.001f ||
                std::abs(nv.velocity.x - rv.velocity.x) > 0.001f ||
                std::abs(nv.velocity.y - rv.velocity.y) > 0.01f ||
                std::abs(nv.velocity.z - rv.velocity.z) > 0.001f ||
                std::abs(nv.velocity.x) > 0.001f ||
                std::abs(rv.velocity.x) > 0.001f) {
                throw std::runtime_error(
                    "Native max-velocity clamp diverged from reference PhysX");
            }

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
