// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/convex.h"

inline void verify_convex_plane_gpu_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    const auto check=[](bool good,const char* message) {
        if(!good)throw std::runtime_error(message);
    };
    const std::array<Vec3,4> vertices{{
        {0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f}}};
    const std::array<ConvexTriangle,4> faces{{
        {0,2,1},{0,1,3},{0,3,2},{1,2,3}}};
    const auto hull=cook_convex_mesh(vertices,faces);
    std::ifstream stream(dy_shader,std::ios::binary);
    check(bool(stream),"Convex GPU Dy shader bundle missing");
    std::vector<char> bytes(std::istreambuf_iterator<char>{stream},
                            std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto assemble=[&](native::Scene& scene) {
        scene.add_static_floor(0.f);
        const auto body=scene.create_body({{0.f,-0.025f,0.f},{}},true);
        scene.attach_convex(body,hull,{});
        scene.update_mass_properties(body,6.f);
        RigidBodyDynamics settings{};
        settings.gravity_enabled=false;
        settings.position_iterations=16u;
        scene.set_dynamics(body,settings);
        scene.set_velocity(body,{0.f,-1.f,0.f});
        return body;
    };
    const auto a=assemble(cpu);
    const auto b=assemble(gpu);
    for(std::uint32_t frame=0u;frame<5u;++frame) {
        if(frame==2u) {
            constexpr Quat turn{0.f,0.19866933f,0.f,0.98006658f};
            const auto pa=cpu.body_state(a).position;
            const auto pb=gpu.body_state(b).position;
            cpu.set_pose(a,pa,turn);
            gpu.set_pose(b,pb,turn);
        }
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        check(cpu.contacts().size()==gpu.contacts().size(),
            "Cooked convex GPU changed plane Sc manager count");
        if(frame==0u) {
            check(cpu.contacts().size()==1u &&
                  cpu.contacts().front().count==3u &&
                  gpu.contacts().front().count==3u,
                "Convex plane must produce three actual hull-vertex witnesses");
            check(producer->dy_cpu_gu_geometry_bytes()==0u &&
                  producer->dy_pcm_device_contact_rows()==3u &&
                  producer->pcm_contact_readback_bytes()==
                      sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
                "Convex plane bypassed device PCM and Dy contact geometry");
        }
        for(std::size_t j=0u;j<cpu.contacts().size();++j) {
            const auto& c=cpu.contacts()[j];
            const auto& d=gpu.contacts()[j];
            check(c.count==d.count && c.pair==d.pair,
                "Convex GPU changed source contact identity");
            for(std::size_t k=0u;k<c.count;++k) {
                const auto& p=c.points[k],&q=d.points[k];
                check(std::fabs(p.position.x-q.position.x)<0.003f &&
                      std::fabs(p.position.z-q.position.z)<0.003f &&
                      std::fabs(p.separation-q.separation)<0.003f &&
                      std::fabs(p.normal.y-1.f)<0.001f &&
                      std::fabs(q.normal.y-1.f)<0.001f,
                    "Convex GPU contact differs from cooked CPU witness");
            }
        }
        test::compound_compare_state(cpu.body_state(a),gpu.body_state(b),
                                     0.008f,"Convex GPU body response");
        check(producer->dy_gpu_integrated_body_count()>0u,
            "Convex GPU body bypassed Dy integration");
    }
}
