// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

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
    cpu.set_position(cpu_a,{5.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{5.0f,1.0f,0.0f});
    check(false);
    cpu.set_position(cpu_a,{0.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{0.0f,1.0f,0.0f});
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})!=
                std::numeric_limits<std::uint32_t>::max(),
            "Source Sc GPU PCM failed contact retirement and re-admission");
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
        std::cout<<"Native Sc GPU PCM scene publication PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Native Sc GPU PCM scene publication FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
