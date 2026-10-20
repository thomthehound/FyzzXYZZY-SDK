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

            // Do not let a single-axis quaternion test conceal incorrect
            // Hamilton-product order, sign or angular-component handling.
            // Both worlds stay in free flight: angular contact solver parity
            // is deliberately NOT claimed.
            fyzz::World native_multiaxis(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World reference_multiaxis(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto nm = native_multiaxis.add_dynamic_box(
                {0.4f,0.6f,0.8f},{0,20,0},1);
            const auto rm = reference_multiaxis.add_dynamic_box(
                {0.4f,0.6f,0.8f},{0,20,0},1);
            native_multiaxis.set_body_angular_velocity(nm,{1.4f,-2.3f,0.8f});
            reference_multiaxis.set_body_angular_velocity(rm,{1.4f,-2.3f,0.8f});
            for (int frame=0;frame<90;++frame) {
                native_multiaxis.step(1.0f/120.0f);
                reference_multiaxis.step(1.0f/120.0f);
            }
            const auto multi_n = native_multiaxis.body_state(nm);
            const auto multi_r = reference_multiaxis.body_state(rm);
            const auto quat_finite = [](fyzz::Quat q) {
                return std::isfinite(q.x) && std::isfinite(q.y) &&
                       std::isfinite(q.z) && std::isfinite(q.w);
            };
            const auto vec_finite = [](fyzz::Vec3 v) {
                return std::isfinite(v.x) && std::isfinite(v.y) &&
                       std::isfinite(v.z);
            };
            const float multi_dot =
                multi_n.orientation.x*multi_r.orientation.x +
                multi_n.orientation.y*multi_r.orientation.y +
                multi_n.orientation.z*multi_r.orientation.z +
                multi_n.orientation.w*multi_r.orientation.w;
            if (!quat_finite(multi_n.orientation) ||
                !quat_finite(multi_r.orientation) ||
                !vec_finite(multi_n.angular_velocity) ||
                !vec_finite(multi_r.angular_velocity) ||
                !std::isfinite(multi_dot) ||
                std::abs(multi_dot) < 0.999f ||
                std::abs(multi_n.angular_velocity.x -
                         multi_r.angular_velocity.x) > 0.03f ||
                std::abs(multi_n.angular_velocity.y -
                         multi_r.angular_velocity.y) > 0.03f ||
                std::abs(multi_n.angular_velocity.z -
                         multi_r.angular_velocity.z) > 0.03f)
                throw std::runtime_error(
                    "Native arbitrary-axis free rotation diverged from PhysX");

            // Source ScBodyCore.cpp defaults dynamic maxAngularVelocitySq
            // to 10000f; DyBodyCoreIntegrator.h clamps after angular
            // damping in PxReal, before quaternion integration.
            fyzz::World native_limited(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World reference_limited(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto nl = native_limited.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,10,0},1);
            const auto rl = reference_limited.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,10,0},1);
            native_limited.set_body_angular_velocity(nl,{250,0,0});
            reference_limited.set_body_angular_velocity(rl,{250,0,0});
            native_limited.step(1.0f/120.0f);
            reference_limited.step(1.0f/120.0f);
            const auto lim_n = native_limited.body_state(nl);
            const auto lim_r = reference_limited.body_state(rl);
            if (!vec_finite(lim_n.angular_velocity) ||
                !vec_finite(lim_r.angular_velocity) ||
                !quat_finite(lim_n.orientation) ||
                !quat_finite(lim_r.orientation) ||
                std::abs(lim_n.angular_velocity.x - 100.0f) > 0.05f ||
                std::abs(lim_n.angular_velocity.x -
                         lim_r.angular_velocity.x) > 0.05f ||
                std::abs(lim_n.orientation.x -
                         lim_r.orientation.x) > 0.02f ||
                std::abs(lim_n.orientation.w -
                         lim_r.orientation.w) > 0.02f)
                throw std::runtime_error(
                    "Native source angular speed clamp diverged from PhysX");

            // Source rigid-body mass/inertia + spatial impulse pipeline:
            // density-scaled nonuniform box, source PxRigidDynamic addTorque
            // eIMPULSE and source-driven pose/damping on the *actual*
            // native scene. This exercises the inertial data, not just
            // an isolated mass-properties calculation.
            fyzz::World native_inertial(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            fyzz::World physx_inertial(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            const auto n_i = native_inertial.add_dynamic_box(
                {0.5f,1.0f,2.0f},{0,15,0},2.0f);
            const auto r_i = physx_inertial.add_dynamic_box(
                {0.5f,1.0f,2.0f},{0,15,0},2.0f);
            const fyzz::Vec3 impulse{1.0f,2.0f,-0.75f};
            native_inertial.apply_body_torque_impulse(n_i,impulse);
            physx_inertial.apply_body_torque_impulse(r_i,impulse);
            for (unsigned frame=0;frame<72;++frame) {
                native_inertial.step(1.0f/120.0f);
                physx_inertial.step(1.0f/120.0f);
            }
            const auto ni=native_inertial.body_state(n_i);
            const auto ri=physx_inertial.body_state(r_i);
            const auto valid_vec=[](fyzz::Vec3 v) {
                return std::isfinite(v.x) && std::isfinite(v.y) &&
                       std::isfinite(v.z);
            };
            const auto valid_quat=[](fyzz::Quat q) {
                return std::isfinite(q.x) && std::isfinite(q.y) &&
                       std::isfinite(q.z) && std::isfinite(q.w);
            };
            const float inertia_orientation_dot=
                ni.orientation.x*ri.orientation.x +
                ni.orientation.y*ri.orientation.y +
                ni.orientation.z*ri.orientation.z +
                ni.orientation.w*ri.orientation.w;
            if (!valid_vec(ni.angular_velocity) || !valid_vec(ri.angular_velocity) ||
                !valid_quat(ni.orientation) || !valid_quat(ri.orientation) ||
                !std::isfinite(inertia_orientation_dot) ||
                std::abs(inertia_orientation_dot)<0.998f ||
                std::abs(ni.angular_velocity.x-ri.angular_velocity.x)>0.01f ||
                std::abs(ni.angular_velocity.y-ri.angular_velocity.y)>0.01f ||
                std::abs(ni.angular_velocity.z-ri.angular_velocity.z)>0.01f)
                throw std::runtime_error(
                    "Native source box inertia/torque impulse diverged from PhysX");
            // Exercise the *world-space* inertia transform after the
            // first torque has already rotated the nonuniform body.
            // This distinguishes R*diag(invI)*R^T from incorrectly using
            // its original body-local inverse inertia as a world tensor.
            const fyzz::Vec3 later_impulse{-0.5f,0.8f,1.2f};
            native_inertial.apply_body_torque_impulse(n_i,later_impulse);
            physx_inertial.apply_body_torque_impulse(r_i,later_impulse);
            for(unsigned frame=0;frame<60;++frame) {
                native_inertial.step(1.0f/120.0f);
                physx_inertial.step(1.0f/120.0f);
            }
            const auto after_n=native_inertial.body_state(n_i);
            const auto after_r=physx_inertial.body_state(r_i);
            const float after_dot=
                after_n.orientation.x*after_r.orientation.x +
                after_n.orientation.y*after_r.orientation.y +
                after_n.orientation.z*after_r.orientation.z +
                after_n.orientation.w*after_r.orientation.w;
            if (!valid_vec(after_n.angular_velocity) ||
                !valid_vec(after_r.angular_velocity) ||
                !valid_quat(after_n.orientation) ||
                !valid_quat(after_r.orientation) ||
                !std::isfinite(after_dot) ||
                std::abs(after_dot)<0.997f ||
                std::abs(after_n.angular_velocity.x-after_r.angular_velocity.x)>0.015f ||
                std::abs(after_n.angular_velocity.y-after_r.angular_velocity.y)>0.015f ||
                std::abs(after_n.angular_velocity.z-after_r.angular_velocity.z)>0.015f)
                throw std::runtime_error(
                    "Rotated world inverse-inertia torque response diverged from PhysX");

            // The SDK floor's public orientation is identity in BOTH
            // providers, even though PxCreatePlane rotates its local +X
            // plane internally. This internal correction must still remain
            // in the actual PhysX transform when the floor is relocated.
            fyzz::World raised_floor(fyzz::WorldOptions{
                fyzz::ExecutionMode::ReferencePhysXCpu});
            fyzz::World native_raised_floor(fyzz::WorldOptions{
                fyzz::ExecutionMode::NativeCpu});
            const auto raised_floor_id = raised_floor.add_static_floor(0.0f);
            const auto native_floor_id = native_raised_floor.add_static_floor(0.0f);
            const auto same_logical_pose = [](fyzz::BodyState a, fyzz::BodyState b) {
                const auto identity = [](fyzz::Quat q) {
                    return q.x == 0 && q.y == 0 && q.z == 0 && q.w == 1;
                };
                return identity(a.orientation) &&
                       identity(b.orientation) &&
                       a.position.x == b.position.x &&
                       a.position.y == b.position.y &&
                       a.position.z == b.position.z;
            };
            if (!same_logical_pose(raised_floor.body_state(raised_floor_id),
                                   native_raised_floor.body_state(native_floor_id)))
                throw std::runtime_error(
                    "Reference floor exposed its private PhysX +X alignment");
            raised_floor.set_body_position(raised_floor_id, {0,0.75f,0});
            native_raised_floor.set_body_position(native_floor_id, {0,0.75f,0});
            if (!same_logical_pose(raised_floor.body_state(raised_floor_id),
                                   native_raised_floor.body_state(native_floor_id)))
                throw std::runtime_error(
                    "Reference floor logical orientation changed on relocation");
            const auto static_reference = raised_floor.add_static_box(
                {1,1,1}, {10,2,0});
            const auto static_native = native_raised_floor.add_static_box(
                {1,1,1}, {10,2,0});
            if (!same_logical_pose(raised_floor.body_state(static_reference),
                                   native_raised_floor.body_state(static_native)))
                throw std::runtime_error(
                    "Non-plane static actor rotation was misclassified");
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

            // Independent source PGS oracle: the same authored material
            // pair must select the same restitution mode in both real
            // scenes, irrespective of actor creation order. This exercises
            // the live normal-contact solve and the saved position-phase
            // motion, not just standalone material-combiner arithmetic.
            for (const bool dynamic_first : {false,true}) {
                for (const auto combine_mode : {
                         fyzz::MaterialCombineMode::Minimum,
                         fyzz::MaterialCombineMode::Maximum}) {
                    fyzz::World native_bounce(fyzz::WorldOptions{
                        fyzz::ExecutionMode::NativeCpu});
                    fyzz::World reference_bounce(fyzz::WorldOptions{
                        fyzz::ExecutionMode::ReferencePhysXCpu});
                    fyzz::BodyId n_floor=0,n_box=0,r_floor=0,r_box=0;
                    if (dynamic_first) {
                        n_box=native_bounce.add_dynamic_box(
                            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                        r_box=reference_bounce.add_dynamic_box(
                            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                        n_floor=native_bounce.add_static_floor(0.0f);
                        r_floor=reference_bounce.add_static_floor(0.0f);
                    } else {
                        n_floor=native_bounce.add_static_floor(0.0f);
                        r_floor=reference_bounce.add_static_floor(0.0f);
                        n_box=native_bounce.add_dynamic_box(
                            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                        r_box=reference_bounce.add_dynamic_box(
                            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                    }
                    fyzz::Material floor_mat{};
                    floor_mat.restitution=0.0f;
                    floor_mat.restitution_combine=
                        fyzz::MaterialCombineMode::Average;
                    floor_mat.disable_friction=true;
                    fyzz::Material box_mat{};
                    box_mat.restitution=1.0f;
                    box_mat.restitution_combine=combine_mode;
                    box_mat.disable_friction=true;
                    native_bounce.set_body_material(n_floor,floor_mat);
                    native_bounce.set_body_material(n_box,box_mat);
                    reference_bounce.set_body_material(r_floor,floor_mat);
                    reference_bounce.set_body_material(r_box,box_mat);
                    native_bounce.set_body_velocity(n_box,{0,-4,0});
                    reference_bounce.set_body_velocity(r_box,{0,-4,0});
                    native_bounce.step(1.0f/120.0f);
                    reference_bounce.step(1.0f/120.0f);
                    const auto n=native_bounce.body_state(n_box);
                    const auto p=reference_bounce.body_state(r_box);
                    if (!std::isfinite(n.velocity.y) ||
                        !std::isfinite(p.velocity.y) ||
                        !std::isfinite(n.position.y) ||
                        !std::isfinite(p.position.y) ||
                        std::abs(n.velocity.y-p.velocity.y)>0.12f ||
                        std::abs(n.position.y-p.position.y)>0.015f)
                        throw std::runtime_error(
                            "Native PGS material restitution diverged from PhysX CPU");
                }
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
