// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/convex.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>

inline void verify_convex_finite_gpu_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader,const char* convex_dy_shader) {
    using namespace fyzz;
    constexpr std::array<Vec3,4> tetra{{
        {0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f}}};
    constexpr std::array<ConvexTriangle,4> faces{{
        {0,2,1},{0,1,3},{0,3,2},{1,2,3}}};
    const auto hull=cook_convex_mesh(tetra,faces);
    auto offsetTetra=tetra;
    for(auto& vertex:offsetTetra) {
        vertex.x+=100.f;
        vertex.y+=100.f;
        vertex.z+=100.f;
    }
    const auto offsetHull=cook_convex_mesh(offsetTetra,faces);
    constexpr std::array<Vec3,8> cubeVertices{{
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}};
    constexpr std::array<ConvexTriangle,12> cubeFaces{{
        {0,2,1},{0,3,2},{4,5,6},{4,6,7},
        {0,1,5},{0,5,4},{3,7,6},{3,6,2},
        {0,4,7},{0,7,3},{1,2,6},{1,6,5}}};
    auto translatedCube=cubeVertices;
    for(auto& vertex:translatedCube) {
        vertex.x+=100.f;
        vertex.y+=100.f;
        vertex.z+=100.f;
    }
    const auto cube=cook_convex_mesh(translatedCube,cubeFaces);
    auto smallCubeVertices=cubeVertices;
    for(auto& vertex:smallCubeVertices) {
        vertex.x*=0.001f;
        vertex.y*=0.001f;
        vertex.z*=0.001f;
    }
    const auto smallCube=cook_convex_mesh(smallCubeVertices,cubeFaces);
    const auto load=[backend](const char* path) {
        std::ifstream input(path,std::ios::binary);
        if(!input)throw std::runtime_error("Finite convex Dy shader missing");
        compute::ShaderBinary shader{};
        shader.format=backend==compute::Backend::Vulkan
            ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
        const std::vector<char> binary{
            std::istreambuf_iterator<char>{input},
            std::istreambuf_iterator<char>{}};
        shader.bytes.assign(binary.begin(),binary.end());
        return shader;
    };
    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        load(dy_shader),load(convex_dy_shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto assemble=[&](native::Scene& scene) {
        std::array<BodyId,9> dynamics{};
        scene.add_static_box({1.f,0.5f,1.f},{0.f,-0.5f,0.f});
        dynamics[0]=scene.create_body({{0.f,-0.12f,0.f},{}},true);
        scene.attach_convex(dynamics[0],hull,{});
        scene.update_mass_properties(dynamics[0],1.f);

        scene.add_static_sphere(0.38f,{4.45f,0.25f,0.25f});
        dynamics[1]=scene.create_body({{4.f,0.f,0.f},{}},true);
        scene.attach_convex(dynamics[1],hull,{});
        scene.update_mass_properties(dynamics[1],1.f);

        const auto fixed_hull=scene.create_body({{8.f,0.f,0.f},{}},false);
        scene.attach_convex(fixed_hull,hull,{});
        dynamics[2]=scene.create_body({{8.15f,0.12f,0.08f},{}},true);
        scene.attach_convex(dynamics[2],hull,{});
        scene.update_mass_properties(dynamics[2],1.f);

        const auto fixed_sphere_hull=scene.create_body({{12.f,0.f,0.f},{}},false);
        scene.attach_convex(fixed_sphere_hull,hull,{});
        dynamics[3]=scene.add_dynamic_sphere(
            0.35f,{12.35f,0.20f,0.25f},1.f);

        const auto fixed_box_hull=scene.create_body({{16.f,0.f,0.f},{}},false);
        scene.attach_convex(fixed_box_hull,hull,{});
        dynamics[4]=scene.add_dynamic_box(
            {0.4f,0.4f,0.4f},{16.35f,0.15f,0.25f},1.f);

        const auto face_hull=scene.create_body({{-76.f,-100.f,-100.f},{}},false);
        scene.attach_convex(face_hull,cube,{});
        dynamics[5]=scene.add_dynamic_sphere(
            0.2f,{24.273157f,-0.794949f,-1.0039f},1.f);

        const auto fixed_offset=scene.create_body({{-72.f,-100.f,-100.f},{}},false);
        scene.attach_convex(fixed_offset,offsetHull,{});
        dynamics[6]=scene.create_body({
            {-71.85f,-99.88f,-99.92f},{}},true);
        scene.attach_convex(dynamics[6],offsetHull,{});
        scene.update_mass_properties(dynamics[6],1.f);

        const auto face_base=scene.create_body({
            {-40.f,-100.f,-100.f},{}},false);
        scene.attach_convex(face_base,cube,{});
        dynamics[7]=scene.create_body({
            {-39.8f,-98.15f,-99.9f},{}},true);
        scene.attach_convex(dynamics[7],cube,{});
        scene.update_mass_properties(dynamics[7],1.f);

        const auto small_base=scene.create_body({{40.f,0.f,0.f},{}},false);
        scene.attach_convex(small_base,smallCube,{});
        dynamics[8]=scene.create_body({{40.0015f,0.00001f,0.00002f},{}},true);
        scene.attach_convex(dynamics[8],smallCube,{});
        scene.update_mass_properties(dynamics[8],1.e9f);

        RigidBodyDynamics settings{};
        settings.gravity_enabled=false;
        settings.position_iterations=24u;
        for(const BodyId body:dynamics)
            scene.set_dynamics(body,settings);
        return dynamics;
    };
    const auto expected=assemble(cpu);
    const auto actual=assemble(gpu);
    constexpr float dt=1.f/120.f;
    for(unsigned frame=0u;frame<3u;++frame) {
        if(frame==1u) {
            constexpr Quat turn{0.f,0.09983342f,0.f,0.99500417f};
            const Vec3 p1=cpu.body_state(expected[2]).position;
            const Vec3 p2=gpu.body_state(actual[2]).position;
            cpu.set_pose(expected[2],p1,turn);
            gpu.set_pose(actual[2],p2,turn);
        }
        cpu.step(dt);
        std::clog<<"convex_finite: gpu frame "<<frame<<" begin\n"<<std::flush;
        gpu.step(dt);
        std::clog<<"convex_finite: gpu frame "<<frame<<" complete\n"<<std::flush;
        if(cpu.contacts().size()!=9u ||
           cpu.contacts().size()!=gpu.contacts().size())
            throw std::runtime_error(
                "Finite cooked hull PCM lost box, sphere or hull contact managers at frame "+
                std::to_string(frame)+": cpu="+
                std::to_string(cpu.contacts().size())+" gpu="+
                std::to_string(gpu.contacts().size())+
                " cpu_pairs="+std::to_string(cpu.pairs().size())+
                " gpu_pairs="+std::to_string(gpu.pairs().size())+
                " gpu_found="+std::to_string(gpu.found_pairs().size())+
                " gpu_readback="+
                std::to_string(producer->pcm_contact_readback_bytes()));
        constexpr auto pcm_stride=
            sizeof(physx_gpu::SourcePcmGpuContactStreamNative);
        constexpr auto header_stride=5u*sizeof(std::uint32_t);
        if(producer->pcm_header_join_bytes()!=9u*header_stride ||
           producer->pcm_deferred_point_bytes()!=9u*pcm_stride-header_stride ||
           producer->pcm_contact_readback_bytes()!=
               9u*(pcm_stride+header_stride)-header_stride ||
           producer->dy_cpu_gu_geometry_bytes()!=0u ||
           producer->dy_pcm_device_contact_rows()<9u)
            throw std::runtime_error(
                "Finite convex contact geometry bypassed GPU PCM/Dy");
        for(std::size_t i=0;i<cpu.contacts().size();++i) {
            const auto& ca=cpu.contacts()[i];
            const auto& cb=gpu.contacts()[i];
            if(ca.pair!=cb.pair || ca.count!=cb.count || ca.count==0u ||
               ca.count>4u)
                throw std::runtime_error(
                    "Finite convex Gu/GPU contact topology diverged at frame "+
                    std::to_string(frame)+" manager "+std::to_string(i)+
                    " cpu_pair="+std::to_string(ca.pair.a)+","+
                    std::to_string(ca.pair.b)+
                    " gpu_pair="+std::to_string(cb.pair.a)+","+
                    std::to_string(cb.pair.b)+
                    " counts="+std::to_string(ca.count)+"/"+
                    std::to_string(cb.count)+
                    " cpu0_normal="+
                    (ca.count?std::to_string(ca.points[0].normal.x)+","+
                        std::to_string(ca.points[0].normal.y)+","+
                        std::to_string(ca.points[0].normal.z):"empty")+
                    " gpu0_normal="+
                    (cb.count?std::to_string(cb.points[0].normal.x)+","+
                        std::to_string(cb.points[0].normal.y)+","+
                        std::to_string(cb.points[0].normal.z):"empty")+
                    " cpu0_depth="+
                    (ca.count?std::to_string(ca.points[0].separation):"empty")+
                    " gpu0_depth="+
                    (cb.count?std::to_string(cb.points[0].separation):"empty"));
            const bool small_pair=ca.points[0].position.x>40.f &&
                                  ca.points[0].position.x<40.01f;
            const float tolerance=small_pair?0.00015f:0.02f;
            const auto close=[tolerance](float x,float y) {
                return std::fabs(x-y)<tolerance;
            };
            // Contact-manifold order is not a physical property. Verify
            // a bijection of all witnesses, not corresponding array slots.
            std::array<unsigned,4> indices{0u,1u,2u,3u};
            bool equivalent=false;
            do {
                bool all_match=true;
                for(std::size_t j=0u;j<ca.count;++j) {
                    const auto& a=ca.points[j];
                    const auto& b=cb.points[indices[j]];
                    if(!close(a.separation,b.separation) ||
                       !close(a.position.x,b.position.x) ||
                       !close(a.position.y,b.position.y) ||
                       !close(a.position.z,b.position.z) ||
                       !close(a.normal.x,b.normal.x) ||
                       !close(a.normal.y,b.normal.y) ||
                       !close(a.normal.z,b.normal.z)) {
                        all_match=false;
                        break;
                    }
                }
                if(all_match) {
                    equivalent=true;
                    break;
                }
            }while(std::next_permutation(indices.begin(),
                                        indices.begin()+ca.count));
            if(!equivalent) {
                std::string message=
                    "Finite convex CPU/GPU contact manifolds diverged at frame "+
                    std::to_string(frame)+" pair "+std::to_string(i);
                const auto append=[&](const auto& manager,const char* label) {
                    message+=" "+std::string(label)+"=[";
                    for(std::size_t j=0u;j<manager.count;++j) {
                        const auto& point=manager.points[j];
                        if(j>0u)message+=";";
                        message+=std::to_string(point.position.x)+","+
                            std::to_string(point.position.y)+","+
                            std::to_string(point.position.z)+" n="+
                            std::to_string(point.normal.x)+","+
                            std::to_string(point.normal.y)+","+
                            std::to_string(point.normal.z)+" d="+
                            std::to_string(point.separation);
                    }
                    message+="]";
                };
                append(ca,"cpu");
                append(cb,"gpu");
                throw std::runtime_error(message);
            }
        }
        if(frame==0u) {
            bool near_face=false,off_origin_pair=false,clipped_face=false;
            bool small_hull=false;
            for(const auto& manager:cpu.contacts())
                if(manager.count>0u) {
                    const auto& point=manager.points[0];
                    if(point.position.x>58.f && point.position.x<62.f &&
                       manager.count==4u)
                        clipped_face=true;
                    if(point.position.x>23.f && point.position.x<26.f &&
                       std::abs(point.separation+0.1961f)<0.015f &&
                       std::abs(point.normal.z)>0.95f)
                        near_face=true;
                    if(point.position.x>27.f && point.position.x<30.f &&
                       point.separation<0.f)
                        off_origin_pair=true;
                    if(point.position.x>40.f && point.position.x<40.01f &&
                       std::abs(point.separation+0.0005f)<0.0001f &&
                       std::abs(point.normal.x)>0.95f)
                        small_hull=true;
                }
            if(!near_face || !off_origin_pair || !clipped_face || !small_hull)
                throw std::runtime_error(
                    "Cooked convex PCM lost translated, small-scale, sphere or face contacts");
        }
        for(std::size_t i=0;i<expected.size();++i)
            test::compound_compare_state(
                cpu.body_state(expected[i]),gpu.body_state(actual[i]),
                0.03f,"Finite convex GPU rigid response");
    }
}

inline void verify_convex_face_dispatch_window(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    constexpr std::array<Vec3,4> vertices{{
        {0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f}}};
    constexpr std::array<ConvexTriangle,4> faces{{
        {0,2,1},{0,1,3},{0,3,2},{1,2,3}}};
    const auto hull=cook_convex_mesh(vertices,faces);
    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto assemble=[&](native::Scene& scene) {
        scene.add_static_box({0.5f,0.5f,0.5f},{0.f,0.f,0.f});
        scene.add_dynamic_box({0.5f,0.5f,0.5f},{0.5f,0.f,0.f},1.f);
        const auto fixed_hull=scene.create_body({{4.f,0.f,0.f},{}},false);
        scene.attach_convex(fixed_hull,hull,{});
        scene.add_dynamic_box({0.4f,0.4f,0.4f},
                              {4.35f,0.15f,0.25f},1.f);
        scene.add_static_box({0.5f,0.5f,0.5f},{8.f,0.f,0.f});
        scene.add_dynamic_box({0.5f,0.5f,0.5f},{8.5f,0.f,0.f},1.f);
        return fixed_hull;
    };
    const auto cpu_hull=assemble(cpu);
    const auto gpu_hull=assemble(gpu);
    const auto check=[&](std::size_t managers,std::size_t face_span) {
        cpu.advance_collision(true);
        gpu.advance_collision(true);
        if(cpu.contacts().size()!=managers ||
           gpu.contacts().size()!=managers ||
           producer->pcm_face_pair_scan_count()!=face_span ||
           producer->pcm_contact_readback_bytes()!=
               managers*sizeof(physx_gpu::SourcePcmGpuContactStreamNative))
            throw std::runtime_error(
                "Convex face GPU dispatch did not respect Sc pair range");
        for(std::size_t i=0u;i<managers;++i)
            if(cpu.contacts()[i].pair!=gpu.contacts()[i].pair ||
               cpu.contacts()[i].count!=gpu.contacts()[i].count)
                throw std::runtime_error(
                    "Bounded convex face dispatch changed source Gu contacts");
    };
    check(3u,1u);
    // Cooked vertices remain device-resident while only the two
    // box contacts are active; no convex face shader should execute.
    cpu.set_position(cpu_hull,{80.f,0.f,0.f});
    gpu.set_position(gpu_hull,{80.f,0.f,0.f});
    check(2u,0u);
    cpu.set_position(cpu_hull,{4.f,0.f,0.f});
    gpu.set_position(gpu_hull,{4.f,0.f,0.f});
    check(3u,1u);
}

inline void verify_convex_plane_gpu_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader,const char* convex_dy_shader) {
    using namespace fyzz;
    const auto check=[](bool good,const char* message) {
        if(!good)throw std::runtime_error(message);
    };
    const std::array<Vec3,4> vertices{{
        {0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f}}};
    const std::array<ConvexTriangle,4> faces{{
        {0,2,1},{0,1,3},{0,3,2},{1,2,3}}};
    const auto hull=cook_convex_mesh(vertices,faces);
    const std::array<Vec3,6> prismVertices{{
        {0,0,0},{1,0,0},{0,1,0},
        {0,0,2},{1,0,2},{0,1,2}}};
    const std::array<ConvexTriangle,8> prismFaces{{
        {0,2,1},{3,4,5},
        {0,1,4},{0,4,3},
        {0,3,5},{0,5,2},{1,2,5},{1,5,4}}};
    const auto prism=cook_convex_mesh(prismVertices,prismFaces);
    const std::array<Vec3,6> pyramidVertices{{
        {-1,0,-1},{-1,0,1},{0,0,1.5f},
        {1,0,1},{1,0,-1},{0,1,0}}};
    const std::array<ConvexTriangle,8> pyramidFaces{{
        {0,2,1},{0,3,2},{0,4,3},
        {0,1,5},{1,2,5},{2,3,5},{3,4,5},{4,0,5}}};
    const auto pyramid=cook_convex_mesh(pyramidVertices,pyramidFaces);
    std::ifstream stream(dy_shader,std::ios::binary);
    check(bool(stream),"Convex GPU Dy shader bundle missing");
    std::vector<char> bytes(std::istreambuf_iterator<char>{stream},
                            std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    std::ifstream convex_stream(convex_dy_shader,std::ios::binary);
    check(bool(convex_stream),"Convex-integrating GPU Dy shader bundle missing");
    std::vector<char> convex_bytes(
        std::istreambuf_iterator<char>{convex_stream},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary convex_shader{};
    convex_shader.format=shader.format;
    convex_shader.bytes.assign(convex_bytes.begin(),convex_bytes.end());
    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader),std::move(convex_shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto assemble=[&](native::Scene& scene) {
        scene.add_static_floor(0.f);
        const auto tetra=scene.create_body({{0.f,-0.025f,0.f},{}},true);
        scene.attach_convex(tetra,hull,{});
        scene.update_mass_properties(tetra,6.f);
        const auto wedge=scene.create_body({{3.f,-0.025f,0.f},{}},true);
        scene.attach_convex(wedge,prism,{});
        scene.update_mass_properties(wedge,2.f);
        const auto pentagonal=scene.create_body({{8.f,-0.025f,0.f},{}},true);
        scene.attach_convex(pentagonal,pyramid,{});
        scene.update_mass_properties(pentagonal,1.f);
        // Two attached hulls borrow the same cooked mesh and its one
        // device vertex span; Sc must preserve both contact managers.
        const auto compound=scene.create_body({{14.f,-0.025f,0.f},{}},true);
        scene.attach_convex(compound,hull,{{-0.8f,0.f,0.f},{}});
        scene.attach_convex(compound,hull,{{0.8f,0.f,0.f},{}});
        scene.update_mass_properties(compound,3.f);
        RigidBodyDynamics settings{};
        settings.gravity_enabled=false;
        settings.position_iterations=16u;
        scene.set_dynamics(tetra,settings);
        scene.set_dynamics(wedge,settings);
        scene.set_dynamics(pentagonal,settings);
        scene.set_dynamics(compound,settings);
        scene.set_velocity(tetra,{0.f,-1.f,0.f});
        scene.set_velocity(wedge,{0.f,-1.f,0.f});
        scene.set_velocity(pentagonal,{0.f,-1.f,0.f});
        scene.set_velocity(compound,{0.f,-1.f,0.f});
        return std::array<BodyId,4>{tetra,wedge,pentagonal,compound};
    };
    const auto a=assemble(cpu);
    const auto b=assemble(gpu);
    for(std::uint32_t frame=0u;frame<5u;++frame) {
        if(frame==2u) {
            constexpr Quat turn{0.f,0.19866933f,0.f,0.98006658f};
            for(std::size_t i=0u;i<a.size();++i) {
                const auto pa=cpu.body_state(a[i]).position;
                const auto pb=gpu.body_state(b[i]).position;
                cpu.set_pose(a[i],pa,turn);
                gpu.set_pose(b[i],pb,turn);
            }
        }
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        check(producer->pcm_face_pair_scan_count()==0u,
            "Plane-only cooked hulls dispatched finite convex face expansion");
        check(cpu.contacts().size()==gpu.contacts().size(),
            "Cooked convex GPU changed plane Sc manager count");
        if(frame==0u) {
            check(cpu.contacts().size()==5u &&
                  cpu.contacts()[0].count==3u &&
                  cpu.contacts()[1].count==4u &&
                  cpu.contacts()[2].count==4u &&
                  cpu.contacts()[3].count==3u &&
                  cpu.contacts()[4].count==3u &&
                  gpu.contacts()[0].count==3u &&
                  gpu.contacts()[1].count==4u &&
                  gpu.contacts()[2].count==4u &&
                  gpu.contacts()[3].count==3u &&
                  gpu.contacts()[4].count==3u,
                "Cooked hull contact managers lost tetrahedron, wedge, polygon or compound witnesses");
            check(producer->dy_cpu_gu_geometry_bytes()==0u &&
                  producer->dy_pcm_device_contact_rows()==17u &&
                  producer->pcm_header_join_bytes()==
                      5u*5u*sizeof(std::uint32_t) &&
                  producer->pcm_deferred_point_bytes()==
                      5u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative)-
                          5u*sizeof(std::uint32_t) &&
                  producer->pcm_contact_readback_bytes()==
                      5u*(sizeof(physx_gpu::SourcePcmGpuContactStreamNative)+
                          5u*sizeof(std::uint32_t))-
                          5u*sizeof(std::uint32_t),
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
        for(std::size_t i=0u;i<a.size();++i)
            test::compound_compare_state(
                cpu.body_state(a[i]),gpu.body_state(b[i]),
                0.008f,"Convex GPU body response");
        check(producer->dy_gpu_integrated_body_count()>0u,
            "Convex GPU body bypassed Dy integration");
    }
}
