// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

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


            // Compare original PhysX axis locks with native PGS
            // integration under gravity, torque and changing timesteps.
            {
                fyzz::World native_locks(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World source_locks(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                const fyzz::Vec3 half{0.35f,0.45f,0.6f};
                const auto nl=native_locks.add_dynamic_box(half,{0,8,0},1.0f);
                const auto pl=source_locks.add_dynamic_box(half,{0,8,0},1.0f);
                fyzz::RigidBodyDynamics d{};
                d.lock_flags=fyzz::LockLinearY | fyzz::LockAngularZ;
                native_locks.set_body_dynamics(nl,d);
                source_locks.set_body_dynamics(pl,d);
                native_locks.set_body_velocity(nl,{0.7f,2.0f,-0.4f});
                source_locks.set_body_velocity(pl,{0.7f,2.0f,-0.4f});
                native_locks.set_body_angular_velocity(nl,{0.8f,-1.1f,3.0f});
                source_locks.set_body_angular_velocity(pl,{0.8f,-1.1f,3.0f});
                const float start_y=native_locks.body_state(nl).position.y;
                for(int frame=0;frame<96;++frame) {
                    if(frame==32) {
                        native_locks.apply_body_torque_impulse(nl,{0.1f,0.2f,0.15f});
                        source_locks.apply_body_torque_impulse(pl,{0.1f,0.2f,0.15f});
                    }
                    if(frame==64) {
                        d.lock_flags=0;
                        native_locks.set_body_dynamics(nl,d);
                        source_locks.set_body_dynamics(pl,d);
                    }
                    const float dt=frame<48?1.0f/120.0f:1.0f/60.0f;
                    native_locks.step(dt);
                    source_locks.step(dt);
                    const auto n=native_locks.body_state(nl);
                    const auto p=source_locks.body_state(pl);
                    const auto error=[](fyzz::Vec3 a,fyzz::Vec3 b) {
                        return std::max({std::abs(a.x-b.x),
                                         std::abs(a.y-b.y),
                                         std::abs(a.z-b.z)});
                    };
                    const float quat_dot=n.orientation.x*p.orientation.x+
                        n.orientation.y*p.orientation.y+
                        n.orientation.z*p.orientation.z+
                        n.orientation.w*p.orientation.w;
                    if(!std::isfinite(quat_dot) ||
                       error(n.position,p.position)>0.015f ||
                       error(n.velocity,p.velocity)>0.035f ||
                       error(n.angular_velocity,p.angular_velocity)>0.035f ||
                       std::abs(quat_dot)<0.998f ||
                       (frame<64 &&
                        (std::abs(n.position.y-start_y)>1.0e-6f ||
                         n.velocity.y!=0.0f ||
                         n.angular_velocity.z!=0.0f)))
                        throw std::runtime_error(
                            "Native locked-body motion diverged from PhysX CPU");
                }
                if(native_locks.body_state(nl).position.y>=start_y-0.01f)
                    throw std::runtime_error(
                        "Releasing source linear lock did not restore motion");
            }

            // Gyroscopic correction is part of
            // DyRigidBodyToSolverBody::copyToSolverBodyData, including
            // bodies with zero contacts. Compare the actual PhysX CPU
            // eENABLE_GYROSCOPIC_FORCES path over multiple timesteps.
            {
                fyzz::World native_gyro(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World source_gyro(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                fyzz::World source_disabled(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                constexpr fyzz::Vec3 half{0.3f,0.5f,0.9f};
                const auto n=native_gyro.add_dynamic_box(half,{0,10,0},1.0f);
                const auto p=source_gyro.add_dynamic_box(half,{0,10,0},1.0f);
                const auto off=source_disabled.add_dynamic_box(half,{0,10,0},1.0f);
                fyzz::RigidBodyDynamics d{};
                d.gravity_enabled=false;
                d.gyroscopic_forces_enabled=true;
                native_gyro.set_body_dynamics(n,d);
                source_gyro.set_body_dynamics(p,d);
                d.gyroscopic_forces_enabled=false;
                source_disabled.set_body_dynamics(off,d);
                for(auto* world:{&native_gyro,&source_gyro,&source_disabled})
                    world->set_body_angular_velocity(
                        world==&native_gyro?n:
                        world==&source_gyro?p:off,
                        {1.2f,1.6f,-1.9f});
                float source_effect=0.0f;
                for(int frame=0;frame<128;++frame) {
                    const float dt=frame<64?1.0f/120.0f:1.0f/240.0f;
                    native_gyro.step(dt);
                    source_gyro.step(dt);
                    source_disabled.step(dt);
                    const auto ns=native_gyro.body_state(n);
                    const auto ps=source_gyro.body_state(p);
                    const auto os=source_disabled.body_state(off);
                    const auto max_error=[](fyzz::Vec3 a,fyzz::Vec3 b) {
                        return std::max({std::abs(a.x-b.x),
                                         std::abs(a.y-b.y),
                                         std::abs(a.z-b.z)});
                    };
                    source_effect=std::max(source_effect,
                        max_error(ps.angular_velocity,os.angular_velocity));
                    const float quat_dot=
                        ns.orientation.x*ps.orientation.x+
                        ns.orientation.y*ps.orientation.y+
                        ns.orientation.z*ps.orientation.z+
                        ns.orientation.w*ps.orientation.w;
                    if(!std::isfinite(quat_dot) ||
                       max_error(ns.angular_velocity,ps.angular_velocity)>0.04f ||
                       std::abs(quat_dot)<0.998f)
                        throw std::runtime_error(
                            "Native gyroscopic response diverged from PhysX CPU");
                }
                if(source_effect<=0.05f)
                    throw std::runtime_error(
                        "PhysX gyroscopic fixture produced no distinct torque");
            }

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
            // The public orientation validator accepts sane, near-unit
            // quaternions. A scaled logical identity must relocate a floor
            // in all execution modes without exposing or changing the
            // reference plane's private +X-to-+Y alignment.
            for (auto mode : {fyzz::ExecutionMode::NativeCpu,
                              fyzz::ExecutionMode::ReferencePhysXCpu,
                              fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu}) {
                fyzz::World floor_pose(fyzz::WorldOptions{mode});
                const auto floor_id = floor_pose.add_static_floor(0.0f);
                for (float scalar : {0.999f, 1.001f}) {
                    const float height = scalar < 1.0f ? 0.5f : 0.75f;
                    floor_pose.set_body_pose(floor_id, {0,height,0},
                                             {0,0,0,scalar});
                    const auto state = floor_pose.body_state(floor_id);
                    if (state.position.y != height ||
                        state.orientation.x != 0.0f ||
                        state.orientation.y != 0.0f ||
                        state.orientation.z != 0.0f ||
                        state.orientation.w != 1.0f)
                        throw std::runtime_error(
                            "Sane scaled floor identity changed logical pose");
                }
                bool rejected_rotation = false;
                try {
                    floor_pose.set_body_pose(floor_id, {0,2.0f,0},
                                             {0,0.1f,0,std::sqrt(0.99f)});
                } catch (const std::logic_error&) {
                    rejected_rotation = true;
                }
                const auto after_rejection = floor_pose.body_state(floor_id);
                if (!rejected_rotation ||
                    after_rejection.position.y != 0.75f ||
                    after_rejection.orientation.x != 0.0f ||
                    after_rejection.orientation.y != 0.0f ||
                    after_rejection.orientation.z != 0.0f ||
                    after_rejection.orientation.w != 1.0f)
                    throw std::runtime_error(
                        "Rotated floor pose admitted or mutated a fixed plane");
                // BodyState intentionally conceals the PhysX plane's
                // private alignment. Prove that the physical normal still
                // supports a falling box after the normalized pose edits.
                const auto falling = floor_pose.add_dynamic_box(
                    {0.25f,0.25f,0.25f}, {0,2.5f,0}, 1.0f);
                for (int frame=0; frame<300; ++frame)
                    floor_pose.step(1.0f/120.0f);
                const auto supported=floor_pose.body_state(falling);
                if (!std::isfinite(supported.position.y) ||
                    !std::isfinite(supported.velocity.y) ||
                    std::abs(supported.position.y-1.0f)>0.1f ||
                    std::abs(supported.velocity.y)>1.0f)
                    throw std::runtime_error(
                        "Normalized floor pose changed physical plane alignment");
            }
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

            // Independent PxScene dynamic island reentry. Reversing
            // creation order reverses the contact normal's BodyId order
            // without changing the physical left/right roles. A closing
            // relative normal speed exercises actual velocity-iteration
            // impulses rather than just penetration-bias pose correction.
            for(const bool reverse_creation : {false,true}) {
                fyzz::World native_islands(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World source_islands(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                fyzz::BodyId n_left=0,n_right=0,r_left=0,r_right=0;
                const fyzz::Vec3 half{0.5f,0.5f,0.5f};
                if(reverse_creation) {
                    n_right=native_islands.add_dynamic_box(
                        half,{4,5,0},1.0f);
                    r_right=source_islands.add_dynamic_box(
                        half,{4,5,0},1.0f);
                    n_left=native_islands.add_dynamic_box(
                        half,{0,5,0},1.0f);
                    r_left=source_islands.add_dynamic_box(
                        half,{0,5,0},1.0f);
                } else {
                    n_left=native_islands.add_dynamic_box(
                        half,{0,5,0},1.0f);
                    r_left=source_islands.add_dynamic_box(
                        half,{0,5,0},1.0f);
                    n_right=native_islands.add_dynamic_box(
                        half,{4,5,0},1.0f);
                    r_right=source_islands.add_dynamic_box(
                        half,{4,5,0},1.0f);
                }
                fyzz::Material no_friction{};
                no_friction.disable_friction=true;
                no_friction.restitution=0.0f;
                for (const auto id : {n_left,n_right})
                    native_islands.set_body_material(id,no_friction);
                for (const auto id : {r_left,r_right})
                    source_islands.set_body_material(id,no_friction);
                constexpr float h=1.0f/120.0f;
                native_islands.step(h);
                source_islands.step(h);
                native_islands.set_body_position(n_right,{0.98f,
                    native_islands.body_state(n_right).position.y,0});
                source_islands.set_body_position(r_right,{0.98f,
                    source_islands.body_state(r_right).position.y,0});
                const auto nl_before=native_islands.body_state(n_left);
                const auto nr_before=native_islands.body_state(n_right);
                const auto rl_before=source_islands.body_state(r_left);
                const auto rr_before=source_islands.body_state(r_right);
                native_islands.set_body_velocity(n_left,
                    {0.8f,nl_before.velocity.y,0});
                native_islands.set_body_velocity(n_right,
                    {-0.8f,nr_before.velocity.y,0});
                source_islands.set_body_velocity(r_left,
                    {0.8f,rl_before.velocity.y,0});
                source_islands.set_body_velocity(r_right,
                    {-0.8f,rr_before.velocity.y,0});
                for(int frame=0;frame<4;++frame) {
                    native_islands.step(h);
                    source_islands.step(h);
                    if(frame==0 &&
                       std::abs(source_islands.body_state(r_left).
                           velocity.x-0.8f)<0.35f)
                        throw std::runtime_error(
                            "PhysX CPU island oracle did not exercise collision impulse");
                    for(const auto [n,r] :
                        {std::pair{n_left,r_left},
                         std::pair{n_right,r_right}}) {
                        const auto a=native_islands.body_state(n);
                        const auto b=source_islands.body_state(r);
                        if (!vec_finite(a.position) ||
                            !vec_finite(a.velocity) ||
                            !vec_finite(b.position) ||
                            !vec_finite(b.velocity) ||
                            std::abs(a.position.x-b.position.x)>0.025f ||
                            std::abs(a.position.y-b.position.y)>0.025f ||
                            std::abs(a.velocity.x-b.velocity.x)>0.12f ||
                            std::abs(a.velocity.y-b.velocity.y)>0.12f)
                            throw std::runtime_error(
                                "Native dynamic island collision diverged from PhysX CPU");
                    }
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

            // Independent PhysX CPU contact conformance: the same
            // one-body floor contact runs through two different worlds.
            // Sample every timestep rather than comparing only the
            // eventual resting pose: a solver can reach an apparently
            // plausible rest state while producing wrong impact impulses.
            // This is a bounded rigid floor-contact gate, not yet a claim
            // of stacked, rotational, multi-material or sleep/wake parity.
            {
                fyzz::World native_contact(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World reference_contact(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                native_contact.add_static_floor(0.0f);
                reference_contact.add_static_floor(0.0f);
                const auto nb=native_contact.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.0f,0.52f,0.0f},1.0f);
                const auto rb=reference_contact.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.0f,0.52f,0.0f},1.0f);
                float max_position_error=0.0f;
                float max_velocity_error=0.0f;
                int max_position_frame=0;
                int max_velocity_frame=0;
                for(int frame=0;frame<120;++frame) {
                    constexpr float contact_dt=1.0f/120.0f;
                    native_contact.step(contact_dt);
                    reference_contact.step(contact_dt);
                    const auto n=native_contact.body_state(nb);
                    const auto p=reference_contact.body_state(rb);
                    const float pos_error=std::abs(n.position.y-p.position.y);
                    const float vel_error=std::abs(n.velocity.y-p.velocity.y);
                    if (!std::isfinite(pos_error) ||
                        !std::isfinite(vel_error))
                        throw std::runtime_error(
                            "Native/PhysX contact trajectory became nonfinite");
                    if(pos_error>max_position_error) {
                        max_position_error=pos_error;
                        max_position_frame=frame+1;
                    }
                    if(vel_error>max_velocity_error) {
                        max_velocity_error=vel_error;
                        max_velocity_frame=frame+1;
                    }
                    // Exact SIMD rounding and comprehensive contact
                    // correlation are future gates, but these bounds
                    // require agreement in both trajectory and response.
                    if(pos_error>0.02f || vel_error>0.12f) {
                        std::cerr << "Native/PhysX floor contact divergence"
                                  << " frame=" << frame+1
                                  << " native_y=" << n.position.y
                                  << " physx_y=" << p.position.y
                                  << " native_vy=" << n.velocity.y
                                  << " physx_vy=" << p.velocity.y
                                  << " pos_error=" << pos_error
                                  << " vel_error=" << vel_error << '\n';
                        throw std::runtime_error(
                            "Native rigid floor-contact PGS disagrees with independent PhysX CPU");
                    }
                }
                std::cout << "Native/PhysX floor-contact PGS:"
                          << " max_y_error=" << max_position_error
                          << " at_frame=" << max_position_frame
                          << " max_vy_error=" << max_velocity_error
                          << " at_frame=" << max_velocity_frame << '\n';
            }

            // Reference PhysX CPU 6DOF PGS cross-check: unequal-mass
            // dynamic collision, both creation orders, and a timestep
            // change while both bodies share the same accurate island.
            // No floor, friction or restitution obscures the normal
            // impulse. Every frame compares the *actual* scene motion,
            // not a separately invented two-body collision model.
            for (bool reversed : {false,true}) {
                for (bool heavy_left : {false,true}) {
                    fyzz::World native_pair(fyzz::WorldOptions{
                        fyzz::ExecutionMode::NativeCpu});
                    fyzz::World physx_pair(fyzz::WorldOptions{
                        fyzz::ExecutionMode::ReferencePhysXCpu});
                    fyzz::BodyId nl=0,nr=0,pl=0,pr=0;
                    const fyzz::Vec3 half{0.5f,0.5f,0.5f};
                    const float left_density=heavy_left?4.0f:1.0f;
                    const float right_density=heavy_left?1.0f:4.0f;
                    if(reversed) {
                        nr=native_pair.add_dynamic_box(
                            half,{0.515f,12,0},right_density);
                        pr=physx_pair.add_dynamic_box(
                            half,{0.515f,12,0},right_density);
                        nl=native_pair.add_dynamic_box(
                            half,{-0.515f,12,0},left_density);
                        pl=physx_pair.add_dynamic_box(
                            half,{-0.515f,12,0},left_density);
                    } else {
                        nl=native_pair.add_dynamic_box(
                            half,{-0.515f,12,0},left_density);
                        pl=physx_pair.add_dynamic_box(
                            half,{-0.515f,12,0},left_density);
                        nr=native_pair.add_dynamic_box(
                            half,{0.515f,12,0},right_density);
                        pr=physx_pair.add_dynamic_box(
                            half,{0.515f,12,0},right_density);
                    }
                    fyzz::Material material{};
                    material.disable_friction=true;
                    material.restitution=0.0f;
                    for(const auto id:{nl,nr})
                        native_pair.set_body_material(id,material);
                    for(const auto id:{pl,pr})
                        physx_pair.set_body_material(id,material);
                    native_pair.set_body_velocity(nl,{1.2f,0,0});
                    physx_pair.set_body_velocity(pl,{1.2f,0,0});
                    native_pair.set_body_velocity(nr,{-0.4f,0,0});
                    physx_pair.set_body_velocity(pr,{-0.4f,0,0});
                    bool source_impact=false;
                    for(int frame=0;frame<48;++frame) {
                        // Both worlds must consume identical dt changes.
                        const float dt=(frame>=12 && frame<28)
                            ? 1.0f/60.0f : 1.0f/120.0f;
                        native_pair.step(dt);
                        physx_pair.step(dt);
                        const auto left_reference=physx_pair.body_state(pl);
                        source_impact |= std::abs(left_reference.velocity.x-1.2f)>0.2f;
                        for(const auto [n_id,p_id,side]:
                            {std::tuple{nl,pl,"left"},
                             std::tuple{nr,pr,"right"}}) {
                            const auto n=native_pair.body_state(n_id);
                            const auto p=physx_pair.body_state(p_id);
                            const auto pos_x=std::abs(n.position.x-p.position.x);
                            const auto pos_y=std::abs(n.position.y-p.position.y);
                            const auto vel_x=std::abs(n.velocity.x-p.velocity.x);
                            const auto vel_y=std::abs(n.velocity.y-p.velocity.y);
                            if(!std::isfinite(pos_x) ||
                               !std::isfinite(pos_y) ||
                               !std::isfinite(vel_x) ||
                               !std::isfinite(vel_y) ||
                               pos_x>0.03f || pos_y>0.025f ||
                               vel_x>0.13f || vel_y>0.13f) {
                                std::cerr
                                    << "Native/PhysX unequal-mass contact"
                                    << " frame=" << frame+1
                                    << " side=" << side
                                    << " reversed=" << reversed
                                    << " heavy_left=" << heavy_left
                                    << " dt=" << dt
                                    << " native_x=" << n.position.x
                                    << " physx_x=" << p.position.x
                                    << " native_vx=" << n.velocity.x
                                    << " physx_vx=" << p.velocity.x
                                    << " native_y=" << n.position.y
                                    << " physx_y=" << p.position.y
                                    << " native_vy=" << n.velocity.y
                                    << " physx_vy=" << p.velocity.y << '\n';
                                throw std::runtime_error(
                                    "Native unequal-mass dynamic PGS diverged from independent PhysX");
                            }
                        }
                    }
                    if(!source_impact)
                        throw std::runtime_error(
                            "Independent PhysX unequal-mass fixture did not produce an impact");
                }
            }

            // Noncentral dynamic rigid contact: the original non-PCM Gu
            // manifold and Dy normal Jacobians must transfer linear and
            // angular momentum together. Compare the actual native
            // production scene with the independent PhysX legacy-Gu PGS
            // backend across creation orders and changing timesteps.
            for(const bool reversed:{false,true}) {
                fyzz::World native_offcenter(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World source_offcenter(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu});
                constexpr fyzz::Vec3 half{0.5f,0.5f,0.5f};
                fyzz::BodyId n_left=0,n_right=0,p_left=0,p_right=0;
                const auto add_left=[&] {
                    n_left=native_offcenter.add_dynamic_box(
                        half,{-0.52f,6.0f,-0.17f},1.0f);
                    p_left=source_offcenter.add_dynamic_box(
                        half,{-0.52f,6.0f,-0.17f},1.0f);
                };
                const auto add_right=[&] {
                    n_right=native_offcenter.add_dynamic_box(
                        half,{0.52f,6.0f,0.17f},3.0f);
                    p_right=source_offcenter.add_dynamic_box(
                        half,{0.52f,6.0f,0.17f},3.0f);
                };
                if(reversed) {add_right();add_left();}
                else {add_left();add_right();}
                fyzz::Material frictionless{};
                frictionless.disable_friction=true;
                frictionless.restitution=0.0f;
                for(const auto id:{n_left,n_right})
                    native_offcenter.set_body_material(id,frictionless);
                for(const auto id:{p_left,p_right})
                    source_offcenter.set_body_material(id,frictionless);
                native_offcenter.set_body_velocity(n_left,{1.3f,0,0});
                source_offcenter.set_body_velocity(p_left,{1.3f,0,0});
                native_offcenter.set_body_velocity(n_right,{-0.35f,0,0});
                source_offcenter.set_body_velocity(p_right,{-0.35f,0,0});
                float source_angular_response=0.0f;
                for(int frame=0;frame<72;++frame) {
                    const float h=frame<24?1.0f/240.0f:1.0f/120.0f;
                    native_offcenter.step(h);
                    source_offcenter.step(h);
                    for(const auto [nid,pid,side]:
                        {std::tuple{n_left,p_left,"left"},
                         std::tuple{n_right,p_right,"right"}}) {
                        const auto n=native_offcenter.body_state(nid);
                        const auto p=source_offcenter.body_state(pid);
                        const auto abs_error=[](fyzz::Vec3 a,fyzz::Vec3 b) {
                            return std::max({
                                std::abs(a.x-b.x),
                                std::abs(a.y-b.y),
                                std::abs(a.z-b.z)});
                        };
                        const float position_error=abs_error(
                            n.position,p.position);
                        const float linear_error=abs_error(
                            n.velocity,p.velocity);
                        const float angular_error=abs_error(
                            n.angular_velocity,p.angular_velocity);
                        const float orientation_dot=
                            n.orientation.x*p.orientation.x+
                            n.orientation.y*p.orientation.y+
                            n.orientation.z*p.orientation.z+
                            n.orientation.w*p.orientation.w;
                        source_angular_response=std::max(
                            source_angular_response,
                            std::abs(p.angular_velocity.y));
                        if(!vec_finite(n.position) ||
                           !vec_finite(p.position) ||
                           !vec_finite(n.velocity) ||
                           !vec_finite(p.velocity) ||
                           !vec_finite(n.angular_velocity) ||
                           !vec_finite(p.angular_velocity) ||
                           !std::isfinite(orientation_dot) ||
                           !std::isfinite(position_error) ||
                           !std::isfinite(linear_error) ||
                           !std::isfinite(angular_error) ||
                           position_error>0.065f ||
                           linear_error>0.18f ||
                           angular_error>0.3f ||
                           std::abs(orientation_dot)<0.995f) {
                            std::cerr<<"Native/PhysX off-center PGS"
                                     <<" frame="<<frame+1
                                     <<" reversed="<<reversed
                                     <<" side="<<side
                                     <<" dt="<<h
                                     <<" position_error="<<position_error
                                     <<" linear_error="<<linear_error
                                     <<" angular_error="<<angular_error
                                     <<" orientation_dot="<<orientation_dot
                                     <<'\n';
                            throw std::runtime_error(
                                "Off-center angular contact diverged from PhysX legacy Gu/PGS");
                        }
                    }
                }
                if(source_angular_response<=0.01f)
                    throw std::runtime_error(
                        "PhysX off-center contact fixture produced no angular response");
            }

            // Coupled PGS conformance: unlike a free two-body impact,
            // the upper dynamic contact and lower static-world contact
            // must both be solved in the same source accurate island.
            // Reverse the mass ratio and change dt after initial impact.
            // Comparison is frame-by-frame, before bodies are guaranteed
            // to satisfy the original sleep-deactivation predicates.
            for(bool bottom_heavy:{false,true}) {
                fyzz::World native_stack(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World physx_stack(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                native_stack.add_static_floor(0.0f);
                physx_stack.add_static_floor(0.0f);
                const fyzz::Vec3 half{0.5f,0.5f,0.5f};
                const float lower_density=bottom_heavy?4.0f:1.0f;
                const float upper_density=bottom_heavy?1.0f:4.0f;
                const auto n_lower=native_stack.add_dynamic_box(
                    half,{0,0.53f,0},lower_density);
                const auto p_lower=physx_stack.add_dynamic_box(
                    half,{0,0.53f,0},lower_density);
                const auto n_upper=native_stack.add_dynamic_box(
                    half,{0,1.56f,0},upper_density);
                const auto p_upper=physx_stack.add_dynamic_box(
                    half,{0,1.56f,0},upper_density);
                fyzz::Material mat{};
                mat.disable_friction=true;
                mat.restitution=0.0f;
                for(auto id:{n_lower,n_upper})
                    native_stack.set_body_material(id,mat);
                for(auto id:{p_lower,p_upper})
                    physx_stack.set_body_material(id,mat);
                // Remove material mismatch at the shared static floor.
                native_stack.set_body_material(0,mat);
                physx_stack.set_body_material(0,mat);
                for(int frame=0;frame<48;++frame) {
                    const float dt=frame<24?1.0f/120.0f:1.0f/60.0f;
                    native_stack.step(dt);
                    physx_stack.step(dt);
                    for(const auto [n_id,p_id,level]:
                        {std::tuple{n_lower,p_lower,"lower"},
                         std::tuple{n_upper,p_upper,"upper"}}) {
                        const auto n=native_stack.body_state(n_id);
                        const auto p=physx_stack.body_state(p_id);
                        const float y_error=std::abs(n.position.y-p.position.y);
                        const float vy_error=std::abs(n.velocity.y-p.velocity.y);
                        if(!std::isfinite(y_error) ||
                           !std::isfinite(vy_error) ||
                           y_error>0.045f || vy_error>0.22f) {
                            std::cerr << "Native/PhysX stacked contact"
                                      << " frame=" << frame+1
                                      << " level=" << level
                                      << " bottom_heavy=" << bottom_heavy
                                      << " dt=" << dt
                                      << " native_y=" << n.position.y
                                      << " physx_y=" << p.position.y
                                      << " native_vy=" << n.velocity.y
                                      << " physx_vy=" << p.velocity.y << '\n';
                            throw std::runtime_error(
                                "Native coupled stack PGS diverged from independent PhysX");
                        }
                    }
                }
            }

            // PhysX default PxTolerancesScale.speed is 10 m/s, so the
            // default sleep energy threshold is 5e-3, not 5e-5.
            // A very slow, frictionless translation on a static floor
            // distinguishes those values: the wrong lower threshold
            // continually renews the wake counter and never sleeps.
            for(bool reference:{false,true}) {
                fyzz::World threshold_world(fyzz::WorldOptions{
                    reference?fyzz::ExecutionMode::ReferencePhysXCpu:
                              fyzz::ExecutionMode::NativeCpu});
                const auto floor=threshold_world.add_static_floor(0);
                const auto body=threshold_world.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
                fyzz::Material frictionless{};
                frictionless.disable_friction=true;
                threshold_world.set_body_material(floor,frictionless);
                threshold_world.set_body_material(body,frictionless);
                threshold_world.set_body_velocity(body,{0.002f,0,0});
                for(int frame=0;frame<360;++frame)
                    threshold_world.step(1.0f/120.0f);
                if(!threshold_world.body_state(body).sleeping) {
                    std::cerr<<"Default PhysX tolerance-scale sleep"
                             <<" reference="<<reference
                             <<" y="<<threshold_world.body_state(body).position.y
                             <<" vx="<<threshold_world.body_state(body).velocity.x
                             <<" vy="<<threshold_world.body_state(body).velocity.y
                             <<'\n';
                    throw std::runtime_error(
                        "Very slow frictionless floor contact failed source sleep threshold");
                }
            }

            // Actor retirement removes the supporting Gu contact
            // manager. Compare whether a dormant body wakes and falls
            // through its original source world, not a separate formula.
            {
                fyzz::World native_support(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World reference_support(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                const auto nf=native_support.add_static_floor(0);
                const auto pf=reference_support.add_static_floor(0);
                const auto nb=native_support.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1);
                const auto pb=reference_support.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1);
                for(int frame=0;frame<360;++frame) {
                    native_support.step(1.0f/120.0f);
                    reference_support.step(1.0f/120.0f);
                }
                if(!native_support.body_state(nb).sleeping ||
                   !reference_support.body_state(pb).sleeping)
                    throw std::runtime_error(
                        "PhysX support-retirement fixture never became dormant");
                const float original_y=native_support.body_state(nb).position.y;
                native_support.remove_body(nf);
                reference_support.remove_body(pf);
                for(int frame=0;frame<12;++frame) {
                    native_support.step(1.0f/120.0f);
                    reference_support.step(1.0f/120.0f);
                }
                const auto n=native_support.body_state(nb);
                const auto p=reference_support.body_state(pb);
                if(n.sleeping || p.sleeping ||
                   n.position.y>=original_y-0.005f ||
                   std::abs(n.position.y-p.position.y)>0.02f ||
                   std::abs(n.velocity.y-p.velocity.y)>0.12f) {
                    std::cerr<<"Native/PhysX retired-support behavior"
                             <<" native_sleeping="<<n.sleeping
                             <<" physx_sleeping="<<p.sleeping
                             <<" native_y="<<n.position.y
                             <<" physx_y="<<p.position.y
                             <<" native_vy="<<n.velocity.y
                             <<" physx_vy="<<p.velocity.y<<'\n';
                    throw std::runtime_error(
                        "Sleeping body did not resume source gravity on support retirement");
                }
            }

            // ScPipeline first-pass pair publication and IslandSim
            // activation precede Dy unconstrained velocity preparation.
            // A dynamic body introduced against a sleeping contact island
            // must wake its peer on that same frame in both implementations.
            {
                fyzz::World native_join(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World reference_join(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu});
                // The reference scene uses the same original PhysX
                // non-PCM Gu generator as the native admitted subset.
                // Retain default friction/restitution so this same-frame
                // activation fixture exercises coupled contact response.
                native_join.add_static_floor(0);
                reference_join.add_static_floor(0);
                const auto n_sleeper=native_join.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1);
                const auto p_sleeper=reference_join.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1);
                for(int frame=0;frame<360;++frame) {
                    native_join.step(1.0f/120.0f);
                    reference_join.step(1.0f/120.0f);
                }
                if(!native_join.body_state(n_sleeper).sleeping ||
                   !reference_join.body_state(p_sleeper).sleeping)
                    throw std::runtime_error(
                        "Dynamic island wake fixture failed to become dormant");
                native_join.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1);
                reference_join.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1);
                native_join.step(1.0f/120.0f);
                reference_join.step(1.0f/120.0f);
                const auto n=native_join.body_state(n_sleeper);
                const auto p=reference_join.body_state(p_sleeper);
                if(n.sleeping || p.sleeping ||
                   std::abs(n.position.x-p.position.x)>0.03f ||
                   std::abs(n.velocity.x-p.velocity.x)>0.15f) {
                    std::cerr<<"Source dynamic island first-pass wake"
                             <<" native_sleeping="<<n.sleeping
                             <<" physx_sleeping="<<p.sleeping
                             <<" native_x="<<n.position.x
                             <<" physx_x="<<p.position.x
                             <<" native_vx="<<n.velocity.x
                             <<" physx_vx="<<p.velocity.x<<'\n';
                    throw std::runtime_error(
                        "New dynamic touch failed source same-frame island activation");
                }
            }

            // PxsBodyCore defaults penBiasClamp to -1e32f. A typical
            // shallow overlap does not discriminate that source default
            // from an invented finite -8, -10, or even -100 m/s clamp.
            // Compare a deep, symmetric original-Gu box contact at a
            // short timestep with independent PhysX CPU using the SAME
            // non-PCM generator and the real Dy PGS solver. The source
            // must itself produce a nontrivial bias correction.
            for(const bool reversed : {false,true}) {
                fyzz::World native_bias(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World source_bias(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu});
                fyzz::BodyId nl=0,nr=0,pl=0,pr=0;
                constexpr fyzz::Vec3 h{0.5f,0.5f,0.5f};
                if(reversed) {
                    nr=native_bias.add_dynamic_box(h,{0.65f,5,0},1.0f);
                    pr=source_bias.add_dynamic_box(h,{0.65f,5,0},1.0f);
                    nl=native_bias.add_dynamic_box(h,{0,5,0},1.0f);
                    pl=source_bias.add_dynamic_box(h,{0,5,0},1.0f);
                } else {
                    nl=native_bias.add_dynamic_box(h,{0,5,0},1.0f);
                    pl=source_bias.add_dynamic_box(h,{0,5,0},1.0f);
                    nr=native_bias.add_dynamic_box(h,{0.65f,5,0},1.0f);
                    pr=source_bias.add_dynamic_box(h,{0.65f,5,0},1.0f);
                }
                fyzz::Material noncompliant{};
                noncompliant.disable_friction=true;
                noncompliant.restitution=0.0f;
                for(const auto id:{nl,nr})
                    native_bias.set_body_material(id,noncompliant);
                for(const auto id:{pl,pr})
                    source_bias.set_body_material(id,noncompliant);
                constexpr float short_dt=1.0f/480.0f;
                native_bias.step(short_dt);
                source_bias.step(short_dt);
                const auto left=native_bias.body_state(nl);
                const auto reference=source_bias.body_state(pl);
                const auto right=native_bias.body_state(nr);
                const auto reference_right=source_bias.body_state(pr);
                if(!vec_finite(left.position) ||
                   !vec_finite(left.velocity) ||
                   !vec_finite(reference.position) ||
                   !vec_finite(reference.velocity) ||
                   !vec_finite(right.position) ||
                   !vec_finite(reference_right.position) ||
                   reference.position.x>-0.08f ||
                   std::abs(left.position.x-reference.position.x)>0.02f ||
                   std::abs(right.position.x-reference_right.position.x)>0.02f) {
                    std::cerr<<"PhysX default penetration bias"
                             <<" reversed="<<reversed
                             <<" native_left_x="<<left.position.x
                             <<" source_left_x="<<reference.position.x
                             <<" native_right_x="<<right.position.x
                             <<" source_right_x="<<reference_right.position.x
                             <<" dt="<<short_dt<<'\n';
                    throw std::runtime_error(
                        "Native PGS default penetration bias diverged from PhysX");
                }
            }

            // Source Dy::sleepCheck + PxsIslandSim::deactivateIsland:
            // a settled body must *actually* stop participating in
            // integration, agree with independent PxRigidDynamic::isSleeping,
            // and wake on user-authored motion without losing contact links.
            {
                fyzz::World native_sleep(fyzz::WorldOptions{
                    fyzz::ExecutionMode::NativeCpu});
                fyzz::World physx_sleep(fyzz::WorldOptions{
                    fyzz::ExecutionMode::ReferencePhysXCpu});
                native_sleep.add_static_floor(0.0f);
                physx_sleep.add_static_floor(0.0f);
                const auto n=native_sleep.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
                const auto p=physx_sleep.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
                for(unsigned frame=0;frame<360;++frame) {
                    native_sleep.step(1.0f/120.0f);
                    physx_sleep.step(1.0f/120.0f);
                }
                const auto asleep_native=native_sleep.body_state(n);
                const auto asleep_physx=physx_sleep.body_state(p);
                if(!asleep_native.sleeping || !asleep_physx.sleeping) {
                    std::cerr<<"Native/PhysX sleep divergence"
                             <<" native_sleeping="<<asleep_native.sleeping
                             <<" physx_sleeping="<<asleep_physx.sleeping
                             <<" native_vy="<<asleep_native.velocity.y
                             <<" physx_vy="<<asleep_physx.velocity.y<<'\n';
                    throw std::runtime_error(
                        "Source island deactivation failed settled floor sleep");
                }
                native_sleep.set_body_velocity(n,{1,0,0});
                physx_sleep.set_body_velocity(p,{1,0,0});
                if(native_sleep.body_state(n).sleeping ||
                   physx_sleep.body_state(p).sleeping)
                    throw std::runtime_error(
                        "Explicit motion failed to wake PhysX island body");
                native_sleep.step(1.0f/120.0f);
                physx_sleep.step(1.0f/120.0f);
                const auto active_native=native_sleep.body_state(n);
                const auto active_physx=physx_sleep.body_state(p);
                if(active_native.sleeping || active_physx.sleeping ||
                   std::abs(active_native.velocity.x-
                            active_physx.velocity.x)>0.12f ||
                   std::abs(active_native.position.x-
                            active_physx.position.x)>0.02f)
                    throw std::runtime_error(
                        "Source island did not resume PhysX-equivalent motion after wake");
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
