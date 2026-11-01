// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"
#include "gpu_integration_metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace {
void require(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
void verify(fyzz::compute::Backend backend,const char* sap,
            const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    require(producer->has_gpu_pcm_box(),
            "GPU scene has no installed native Gu PCM producer");
    gpu.set_gpu_broadphase(std::move(owner));
    const auto cpu_a=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
    const auto cpu_b=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto gpu_a=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
    const auto gpu_b=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto check=[&](bool contact) {
        cpu.advance_collision();
        gpu.advance_collision();
        require(cpu.pairs().size()==gpu.pairs().size() &&
                cpu.contacts().size()==gpu.contacts().size(),
                "Sc GPU PCM pair or contact manager publication differs");
        require((gpu.contacts().size()==1u)==contact,
                "Source GPU PCM scene has unexpected contact manager count");
        if(!contact)return;
        const auto& c=cpu.contacts().front();
        const auto& g=gpu.contacts().front();
        require(c.count==g.count && c.count>0u &&
                c.edge_handle==g.edge_handle,
                "Source GPU PCM did not preserve Sc manager identity/count");
        for(std::uint8_t i=0;i<c.count;++i) {
            const auto& x=c.points[i];
            const auto& y=g.points[i];
            require(std::fabs(x.separation-y.separation)<0.002f &&
                    std::fabs(x.position.x-y.position.x)<0.002f &&
                    std::fabs(x.position.y-y.position.y)<0.002f &&
                    std::fabs(x.position.z-y.position.z)<0.002f &&
                    std::fabs(x.normal.x-y.normal.x)<0.002f &&
                    std::fabs(x.normal.y-y.normal.y)<0.002f &&
                    std::fabs(x.normal.z-y.normal.z)<0.002f,
                    "Source GPU Sc contact witness differs from CPU Gu PCM");
        }
    };
    check(true);
    const auto edge=gpu.contact_edge_index({gpu_a,gpu_b});
    // The original GPU producer now downloads the 96-byte shader stream
    // directly into Sc's result layout. Inspect the live producer result
    // without an intermediate host-side contact conversion.
    const std::array<native::GpuPcmBoxPair,1> request{{
        {gpu_a,gpu_b,edge,false}
    }};
    std::array<native::GpuPcmBoxResult,1> direct{};
    producer->generate_gpu_pcm_box_contacts(request,direct);
    require(direct[0].edge==edge && direct[0].status==0u &&
            direct[0].cached_count<=4u &&
            direct[0].count<=direct[0].cached_count &&
            (direct[0].selected_and_regenerated&0x7fffffffu)<=4u &&
            direct[0].count>0u &&
            std::isfinite(direct[0].normal.x) &&
            std::isfinite(direct[0].points[0].separation),
            "Direct Sc GPU PCM readback disagrees with the shader record");
    // Force the original AABB volume domain to grow while the first
    // contact manager retains GPU-local cached shape geometry.
    for(std::uint32_t i=0u;i<36u;++i) {
        const Vec3 position{20.f+3.f*float(i),1.f,0.f};
        cpu.add_static_box({0.5f,0.5f,0.5f},position);
        gpu.add_static_box({0.5f,0.5f,0.5f},position);
    }
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})==edge,
            "AABB capacity growth discarded an unchanged GPU PCM manifold");
    cpu.set_position(cpu_a,{0.02f,1.0f,0.0f});
    gpu.set_position(gpu_a,{0.02f,1.0f,0.0f});
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})==edge,
            "Quiet Sc GPU PCM invalidated a persistent contact edge");
    // The device stream reconstructs the original Gu B-local witnesses
    // in world space, including rotated source-order second shapes.
    constexpr float angle=0.08f;
    const Quat orientation{0.f,0.f,std::sin(angle*0.5f),
                           std::cos(angle*0.5f)};
    cpu.set_pose(cpu_b,{0.8f,1.f,0.f},orientation);
    gpu.set_pose(gpu_b,{0.8f,1.f,0.f},orientation);
    check(true);
    cpu.set_pose(cpu_b,{0.8f,1.f,0.f},{});
    gpu.set_pose(gpu_b,{0.8f,1.f,0.f},{});
    check(true);
    cpu.set_position(cpu_a,{5.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{5.0f,1.0f,0.0f});
    check(false);
    cpu.set_position(cpu_a,{0.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{0.0f,1.0f,0.0f});
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})!=
                std::numeric_limits<std::uint32_t>::max(),
            "Source Sc GPU PCM failed contact retirement and re-admission");
    // The source-owned Gu contacts must enter the existing Dy contact
    // preparation/solve path, not merely appear in collision queries.
    constexpr float dt=1.f/120.f;
    cpu.step(dt);
    gpu.step(dt);
    const auto cpu_body=cpu.body_state(cpu_a);
    const auto gpu_body=gpu.body_state(gpu_a);
    require(gpu.solved_result_frame()==gpu.frame_index() &&
            cpu.solved_result_frame()==cpu.frame_index() &&
            std::fabs(cpu_body.position.x-gpu_body.position.x)<0.002f &&
            std::fabs(cpu_body.position.y-gpu_body.position.y)<0.002f &&
            std::fabs(cpu_body.position.z-gpu_body.position.z)<0.002f,
            "Source GPU PCM did not feed the existing Sc/Dy rigid solve");
}

// Sparse live Sc managers keep their original edge handles. GPU readback
// must scale with active managers, not the largest surviving edge index.
void verify_sparse_contact_stream(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    gpu.set_gpu_broadphase(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm)));
    std::array<BodyId,12> moving_cpu{},moving_gpu{};
    std::array<BodyId,12> fixed_cpu{},fixed_gpu{};
    for(std::uint32_t i=0u;i<12u;++i) {
        const float x=5.f*float(i);
        moving_cpu[i]=cpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x,1.f,0.f},1.f);
        fixed_cpu[i]=cpu.add_static_box(
            {0.5f,0.5f,0.5f},{x+0.8f,1.f,0.f});
        moving_gpu[i]=gpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x,1.f,0.f},1.f);
        fixed_gpu[i]=gpu.add_static_box(
            {0.5f,0.5f,0.5f},{x+0.8f,1.f,0.f});
    }
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==12u && gpu.contacts().size()==12u,
            "Sparse GPU PCM fixture did not admit original Sc managers");
    const auto retained_edge=gpu.contact_edge_index(
        {moving_gpu.back(),fixed_gpu.back()});
    require(retained_edge!=std::numeric_limits<std::uint32_t>::max(),
            "Sparse GPU PCM fixture has no retained Sc edge");
    for(std::uint32_t i=0u;i<11u;++i) {
        cpu.remove(moving_cpu[i]);
        gpu.remove(moving_gpu[i]);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==1u && gpu.contacts().size()==1u &&
            gpu.contacts()[0].edge_handle==retained_edge &&
            cpu.contacts()[0].count==gpu.contacts()[0].count,
            "Sparse Sc contact stream misidentified surviving edge handle");
    const auto c=cpu.contacts()[0].points[0];
    const auto g=gpu.contacts()[0].points[0];
    require(std::fabs(c.separation-g.separation)<0.002f &&
            std::fabs(c.position.x-g.position.x)<0.002f &&
            std::fabs(c.normal.x-g.normal.x)<0.002f,
            "Sparse Sc GPU contact stream changed the retained Gu witness");

    // Re-grow the GPU contact readback after the 12-pair -> 1-pair
    // retirement. The source PCM commands are now still outstanding
    // when Vulkan enlarges its cached staging buffer for this fetch.
    constexpr std::uint32_t added=16u;
    for(std::uint32_t i=0u;i<added;++i) {
        const Vec3 position{200.f+4.f*float(i),1.f,0.f};
        cpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
        gpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
        const Vec3 other{position.x+0.8f,position.y,position.z};
        cpu.add_static_box({0.5f,0.5f,0.5f},other);
        gpu.add_static_box({0.5f,0.5f,0.5f},other);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==added+1u &&
            gpu.contacts().size()==cpu.contacts().size() &&
            gpu.contact_edge_index(
                {moving_gpu.back(),fixed_gpu.back()})==retained_edge,
            "Growing source GPU PCM readback lost an existing Sc contact manager");
    for(std::size_t i=0;i<cpu.contacts().size();++i) {
        const auto& prior=cpu.contacts()[i];
        const auto& current=gpu.contacts()[i];
        require(prior.pair==current.pair &&
                prior.count==current.count &&
                current.count>0u,
                "Expanded GPU PCM readback changed source Gu manager order/count");
        for(std::uint8_t j=0;j<prior.count;++j)
            require(std::fabs(prior.points[j].separation-
                              current.points[j].separation)<0.002f &&
                    std::fabs(prior.points[j].normal.x-
                              current.points[j].normal.x)<0.002f,
                    "Expanded source GPU PCM stream changed Gu contact geometry");
    }
}

// Coupled box contacts exercise source Sc manager identity and Dy's
// real contact partition/PGS path across multiple dependent bodies.
void verify_coupled_rigid_stack(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    gpu.set_gpu_broadphase(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm)));
    cpu.add_static_box({0.5f,0.5f,0.5f},{0.f,-0.5f,0.f});
    const BodyId fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,-0.5f,0.f});
    std::array<BodyId,3> dynamic_cpu{},dynamic_gpu{};
    for(std::size_t i=0;i<dynamic_cpu.size();++i) {
        const Vec3 position{0.f,0.4f+0.9f*float(i),0.f};
        dynamic_cpu[i]=cpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},position,1.f);
        dynamic_gpu[i]=gpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},position,1.f);
    }
    constexpr float dt=1.f/120.f;
    bool coupled=false;
    for(std::uint32_t frame=0u;frame<12u;++frame) {
        cpu.step(dt);
        gpu.step(dt);
        require(cpu.pairs().size()==gpu.pairs().size() &&
                cpu.contacts().size()==gpu.contacts().size(),
                "GPU coupled rigid stack lost Sc contact-manager parity");
        for(const auto& contact:gpu.contacts())
            if(contact.pair.a!=fixed_gpu &&
               contact.pair.b!=fixed_gpu && contact.count>0u)
                coupled=true;
        for(std::size_t i=0;i<dynamic_cpu.size();++i) {
            const auto a=cpu.body_state(dynamic_cpu[i]);
            const auto b=gpu.body_state(dynamic_gpu[i]);
            require(std::fabs(a.position.x-b.position.x)<0.025f &&
                    std::fabs(a.position.y-b.position.y)<0.025f &&
                    std::fabs(a.position.z-b.position.z)<0.025f &&
                    std::fabs(a.velocity.x-b.velocity.x)<0.025f &&
                    std::fabs(a.velocity.y-b.velocity.y)<0.025f &&
                    std::fabs(a.velocity.z-b.velocity.z)<0.025f,
                    "GPU coupled rigid stack diverged from source Sc/Dy motion");
        }
        require(gpu.solved_result_frame()==gpu.frame_index(),
                "GPU coupled rigid stack did not complete Dy solve");
    }
    require(coupled,
            "GPU coupled rigid stack never admitted dynamic-dynamic contacts");
}
// Source Gu PCM rows, friction patches, and 1D joints share the same
// production Dy task partitions and device execution on both backends.
void verify_gpu_coupled_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader,
    std::uint32_t spring_mode) {
    using namespace fyzz;
    std::ifstream stream(dy_shader,std::ios::binary);
    require(bool(stream),"Source Dy shader bundle missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    require(producer->has_gpu_pcm_box() && producer->has_gpu_dy_distance(),
            "Source coupled GPU PCM/Dy stages were not installed");
    gpu.set_gpu_broadphase(std::move(owner));
    cpu.add_static_box({0.5f,0.5f,0.5f},{0.0f,0.0f,0.0f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{0.0f,0.0f,0.0f});
    std::array<BodyId,2> a{},b{};
    for(std::size_t i=0;i<a.size();++i) {
        const Vec3 p{0.0f,0.83f+0.83f*float(i),0.0f};
        a[i]=cpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
        b[i]=gpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
    }
    cpu.create_center_distance_constraint(a[0],a[1],0.82f);
    gpu.create_center_distance_constraint(b[0],b[1],0.82f);
    if(spring_mode) {
        Material spring{};
        spring.restitution=-100.0f;
        spring.damping=2.0f;
        spring.compliant_acceleration_spring=spring_mode==2u;
        for(std::size_t i=0;i<a.size();++i) {
            cpu.set_material(a[i],spring);
            gpu.set_material(b[i],spring);
        }
    }
    // Zero-impulse Gu managers retain contact geometry but deliberately
    // prepare no Dy solver rows. Coexist with an independently live
    // contact and switch the same manager between zero/live responses.
    cpu.add_static_box({0.5f,0.5f,0.5f},{10.f,0.f,0.f});
    const auto capped_fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{10.f,0.f,0.f});
    const auto capped_cpu=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.f,0.84f,0.f},1.f);
    const auto capped_gpu=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.f,0.84f,0.f},1.f);
    RigidBodyDynamics capped_dynamics{};
    capped_dynamics.gravity_enabled=false;
    capped_dynamics.max_contact_impulse=0.f;
    cpu.set_dynamics(capped_cpu,capped_dynamics);
    gpu.set_dynamics(capped_gpu,capped_dynamics);
    cpu.set_velocity(a[0],{0.25f,-0.1f,0.0f});
    gpu.set_velocity(b[0],{0.25f,-0.1f,0.0f});
    constexpr float dt=1.f/120.f;
    bool saw_contact=false,saw_force=false,saw_friction=false;
    bool saw_correlated_multi_point=false;
    double max_normal_error=0.0,max_friction_error=0.0;
    float max_position_error=0.f,max_velocity_error=0.f;
    std::size_t uploaded_bytes=0,downloaded_bytes=0;
    for(std::uint32_t frame=0;frame<12u;++frame) {
        if(frame==4u || frame==8u) {
            capped_dynamics.max_contact_impulse=frame==4u
                ? std::numeric_limits<float>::max():0.f;
            cpu.set_dynamics(capped_cpu,capped_dynamics);
            gpu.set_dynamics(capped_gpu,capped_dynamics);
        }
        cpu.step(dt);
        gpu.step(dt);
        const auto [uploaded,downloaded]=producer->dy_distance_transfer_bytes();
        require(downloaded>0u && downloaded<uploaded/2u,
            "Source coupled GPU Dy writeback did not use compact output");
        uploaded_bytes+=uploaded;
        downloaded_bytes+=downloaded;
        require(producer->dy_distance_dispatch_count()==frame+1u,
            "Source coupled Sc scene never dispatched the GPU Dy solver");
        require(cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size() &&
                cpu.solved_friction_patches().size()==
                    gpu.solved_friction_patches().size(),
            "GPU Dy contact/friction writeback has wrong source ownership");
        saw_contact |= !gpu.contacts().empty();
        bool found_capped=false;
        for(const auto& contact:gpu.contacts()) {
            const native::Pair capped_pair{capped_fixed_gpu,capped_gpu};
            if(!(contact.pair==capped_pair)) {
                saw_correlated_multi_point |= contact.count>1u;
                continue;
            }
            found_capped=true;
            if(frame<4u || frame>=8u)
                for(std::size_t k=0;k<contact.count;++k)
                    require(gpu.solved_normal_impulses()[
                                contact.contact_offset+k]==0.0,
                        "Zero-cap GPU Gu manager generated a Dy contact impulse");
        }
        require(found_capped,
            "Zero-cap source Gu manager lost its Sc contact identity");
        for(std::size_t i=0;i<cpu.solved_normal_impulses().size();++i) {
            const auto force=cpu.solved_normal_impulses()[i];
            const auto actual=gpu.solved_normal_impulses()[i];
            saw_force |= force>0.0;
            max_normal_error=std::max(max_normal_error,std::fabs(force-actual));
            require(std::fabs(force-actual)<0.06,
                "Source GPU Dy normal impulse differs from CPU PGS");
        }
        for(std::size_t i=0;i<cpu.solved_friction_patches().size();++i) {
            const auto& ref=cpu.solved_friction_patches()[i];
            const auto& actual=gpu.solved_friction_patches()[i];
            require(ref.anchor_count==actual.anchor_count &&
                    ref.broken==actual.broken,
                "Source Dy GPU friction patch lost anchor or break state");
            saw_friction |= ref.anchor_count>0;
            for(std::size_t k=0;k<ref.anchor_count;++k) {
                const double dx=std::fabs(ref.anchor_impulses[k].x-
                                           actual.anchor_impulses[k].x);
                const double dy=std::fabs(ref.anchor_impulses[k].y-
                                           actual.anchor_impulses[k].y);
                max_friction_error=std::max({max_friction_error,dx,dy});
                require(dx<0.06 && dy<0.06,
                    "Source Dy GPU friction impulses differ from CPU PGS");
            }
        }
        for(std::size_t i=0;i<a.size();++i) {
            const auto expected=cpu.body_state(a[i]);
            const auto actual=gpu.body_state(b[i]);
            const float position_error=std::max(
                std::fabs(expected.position.x-actual.position.x),
                std::fabs(expected.position.y-actual.position.y));
            const float velocity_error=std::max(
                std::fabs(expected.velocity.x-actual.velocity.x),
                std::fabs(expected.velocity.y-actual.velocity.y));
            max_position_error=std::max(max_position_error,position_error);
            max_velocity_error=std::max(max_velocity_error,velocity_error);
            require(position_error<0.03f && velocity_error<0.03f,
                "Coupled GPU Gu PCM/1D/contact/friction diverged from CPU Dy");
        }
    }
    require(saw_contact && saw_force && saw_friction &&
            saw_correlated_multi_point,
        "Source coupled GPU test never exercised multi-point contact/friction");
    std::cout<<"FYZZXYZZY_METRIC suite=dy_pgs spring_mode="<<spring_mode
             <<" frames=12 gpu_dispatches="
             <<producer->dy_distance_dispatch_count()
             <<" max_normal_abs_error="<<max_normal_error
             <<" max_friction_abs_error="<<max_friction_error
             <<" max_position_abs_error="<<max_position_error
             <<" max_velocity_abs_error="<<max_velocity_error
             <<" host_upload_bytes="<<uploaded_bytes
             <<" host_download_bytes="<<downloaded_bytes<<'\n';
}

// The explicitly installed device Dy 1D stage must replace the
// production PGS solve, not merely compute a disconnected fixture result.
void verify_gpu_distance_solver(
    fyzz::compute::Backend backend,const char* sap,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled source Dy shader unavailable");
    std::vector<char> binary(
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());

    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,std::nullopt,std::move(shader));
    auto* producer=owner.get();
    require(producer->has_gpu_dy_distance(),
            "Explicit source Dy distance stage was not installed");
    gpu.set_gpu_broadphase(std::move(owner));
    constexpr float dt=1.f/120.f;
    std::array<BodyId,4> ids_cpu{},ids_gpu{};
    for(std::size_t i=0;i<ids_cpu.size();++i) {
        const Vec3 position{float(i)*4.0f,3.0f,0.0f};
        ids_cpu[i]=cpu.add_dynamic_box({0.3f,0.3f,0.3f},position,1.f);
        ids_gpu[i]=gpu.add_dynamic_box({0.3f,0.3f,0.3f},position,1.f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.position_iterations=5;
        d.velocity_iterations=2;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        cpu.set_dynamics(ids_cpu[i],d);
        gpu.set_dynamics(ids_gpu[i],d);
    }
    // Two disconnected accurate islands must advance through independent
    // source task workgroups. Off-center anchors exercise angular Jacobians.
    cpu.create_anchored_distance_constraint(ids_cpu[0],ids_cpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    gpu.create_anchored_distance_constraint(ids_gpu[0],ids_gpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    cpu.create_center_distance_constraint(ids_cpu[2],ids_cpu[3],3.5f);
    gpu.create_center_distance_constraint(ids_gpu[2],ids_gpu[3],3.5f);
    // A source 1D joint against an immovable body must use the
    // original static-world sentinel, never a fabricated solver body.
    const auto fixed_cpu=cpu.add_static_box(
        {0.3f,0.3f,0.3f},{40.f,3.f,0.f});
    const auto fixed_gpu=gpu.add_static_box(
        {0.3f,0.3f,0.3f},{40.f,3.f,0.f});
    const auto anchored_cpu=cpu.add_dynamic_box(
        {0.3f,0.3f,0.3f},{44.f,3.f,0.f},1.f);
    const auto anchored_gpu=gpu.add_dynamic_box(
        {0.3f,0.3f,0.3f},{44.f,3.f,0.f},1.f);
    RigidBodyDynamics anchored_dynamics{};
    anchored_dynamics.gravity_enabled=false;
    anchored_dynamics.linear_damping=0.f;
    anchored_dynamics.angular_damping=0.f;
    cpu.set_dynamics(anchored_cpu,anchored_dynamics);
    gpu.set_dynamics(anchored_gpu,anchored_dynamics);
    cpu.create_center_distance_constraint(
        fixed_cpu,anchored_cpu,3.8f);
    gpu.create_center_distance_constraint(
        fixed_gpu,anchored_gpu,3.8f);
    cpu.set_velocity(ids_cpu[0],{0.4f,0.2f,0.0f});
    gpu.set_velocity(ids_gpu[0],{0.4f,0.2f,0.0f});
    cpu.set_velocity(ids_cpu[2],{-0.6f,0.0f,0.0f});
    gpu.set_velocity(ids_gpu[2],{-0.6f,0.0f,0.0f});
    for(int frame=0;frame<12;++frame) {
        cpu.step(dt);
        gpu.step(dt);
        require(cpu.contacts().empty() && gpu.contacts().empty(),
            "Source GPU Dy 1D fixture unexpectedly generated Gu contacts");
        require(producer->dy_distance_dispatch_count()==
                    static_cast<std::uint64_t>(frame+1),
            "Source GPU Dy 1D task never dispatched on the device");
        const auto [uploaded,downloaded]=
            producer->dy_distance_transfer_bytes();
        require(downloaded>0u && downloaded<uploaded/2u,
            "Source Dy GPU writeback still transfers full solver rows");
        const auto fixed_expected=cpu.body_state(anchored_cpu);
        const auto fixed_actual=gpu.body_state(anchored_gpu);
        require(std::fabs(fixed_expected.position.x-
                          fixed_actual.position.x)<0.001f &&
                std::fabs(fixed_expected.velocity.x-
                          fixed_actual.velocity.x)<0.001f,
                "Static-world Dy GPU 1D constraint changed source dynamics");
        for(std::size_t i=0;i<ids_cpu.size();++i) {
            const auto expected=cpu.body_state(ids_cpu[i]);
            const auto actual=gpu.body_state(ids_gpu[i]);
            require(std::fabs(expected.position.x-actual.position.x)<0.001f &&
                    std::fabs(expected.position.y-actual.position.y)<0.001f &&
                    std::fabs(expected.velocity.x-actual.velocity.x)<0.001f &&
                    std::fabs(expected.angular_velocity.z-
                              actual.angular_velocity.z)<0.001f,
                "Source GPU Dy 1D solve diverged from production CPU PGS");
        }
    }
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==4 || argc==5,
                "Expected SAP/PCM manifests, source Dy shader and optional scenario");
        const std::string_view scenario=argc==5?argv[4]:"all";
        require(scenario=="all" || scenario=="scene" ||
                scenario=="sparse_contacts" || scenario=="coupled_stack" ||
                scenario=="coupled_solver" || scenario=="distance_solver",
                "Unknown GPU scene regression scenario");
#ifdef FYZZXYZZY_TEST_D3D12
        constexpr auto backend=fyzz::compute::Backend::D3D12;
#else
        constexpr auto backend=fyzz::compute::Backend::Vulkan;
#endif
        if(scenario=="all" || scenario=="scene")
            fyzz::tests::measure_gpu_case("sc_pcm","scene",
                [&]{verify(backend,argv[1],argv[2]);});
        if(scenario=="all" || scenario=="sparse_contacts")
            fyzz::tests::measure_gpu_case("sc_pcm","sparse_contacts",
                [&]{verify_sparse_contact_stream(backend,argv[1],argv[2]);});
        if(scenario=="all" || scenario=="coupled_stack")
            fyzz::tests::measure_gpu_case("sc_pcm","coupled_stack",
                [&]{verify_coupled_rigid_stack(backend,argv[1],argv[2]);});
        if(scenario=="all" || scenario=="coupled_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","coupled_solver",
                [&]{
                    for(std::uint32_t spring=0u;spring<3u;++spring)
                        verify_gpu_coupled_solver(
                            backend,argv[1],argv[2],argv[3],spring);
                });
        if(scenario=="all" || scenario=="distance_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","distance_solver",
                [&]{verify_gpu_distance_solver(backend,argv[1],argv[3]);});
        std::cout<<"Native Sc GPU PCM scene "<<scenario<<" PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Native Sc GPU PCM scene regression FAILED: "
                  <<error.what()<<'\n';
        return 1;
    }
}
