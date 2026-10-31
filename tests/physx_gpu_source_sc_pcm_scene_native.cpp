// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

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
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected native SAP and PCM shader manifests");
        verify(
#ifdef FYZZXYZZY_TEST_D3D12
            fyzz::compute::Backend::D3D12,
#else
            fyzz::compute::Backend::Vulkan,
#endif
            argv[1],argv[2]);
        verify_sparse_contact_stream(
#ifdef FYZZXYZZY_TEST_D3D12
            fyzz::compute::Backend::D3D12,
#else
            fyzz::compute::Backend::Vulkan,
#endif
            argv[1],argv[2]);
        std::cout<<"Native Sc GPU PCM scene publication PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Native Sc GPU PCM scene publication FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
