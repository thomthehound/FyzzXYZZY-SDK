// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_box_transform_bounds_native.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
ShaderBinary shader(const char* name) {
    std::ifstream file(name,std::ios::binary);
    require(bool(file),"Missing source PCM GPU shader");
    ShaderBinary result;
#ifdef FYZZXYZZY_TEST_D3D12
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    require(!result.bytes.empty(),"Empty source PCM GPU shader");
    return result;
}
void verify(Context& context,const char* transform_shader,
            const char* candidate_shader) {
    constexpr float z=0.3826834323650898f;
    constexpr float w=0.9238795325112867f;
    const std::array<SourceScBoxTransformInputNative,6> boxes{{
        {1u,{0.f,0.f,0.f},{0.f,0.f,0.f,1.f},{1.f,2.f,3.f},0.2f},
        {6u,{1.25f,0.1f,0.f},{0.f,0.f,z,w},{1.f,1.5f,2.5f},0.3f},
        {31u,{30.f,0.f,0.f},{0.f,0.f,0.f,1.f},{1.f,2.f,2.f},0.1f},
        {12u,{0.3f,0.4f,2.f},{z,0.f,0.f,w},{1.f,1.f,1.f},0.125f},
        {9u,{0.f,1.1f,-1.f},{0.f,z,0.f,w},{0.75f,1.7f,0.5f},0.15f},
        {20u,{-0.1f,-0.2f,0.75f},{0.f,0.f,0.707106781f,0.707106781f},
         {2.f,0.45f,0.8f},0.35f}
    }};
    constexpr std::size_t base_pairs=6u*5u;
    std::array<SourcePcmBoxPairNative,base_pairs+3u> pairs{};
    constexpr std::array<std::uint32_t,6> ids{1u,6u,31u,12u,9u,20u};
    std::size_t index=0u;
    for(const auto a:ids)
        for(const auto b:ids)
            if(a!=b) {
                const auto edge=17u+static_cast<std::uint32_t>(index)*13u;
                pairs[index++]={a,b,edge};
            }
    pairs[index++]={31u,31u,505u};
    pairs[index++]={1u,6u,0xffffffffu};
    pairs[index++]={1u,6u,710u};
    require(index==pairs.size(),"Source PCM pair fixture size mismatch");
    constexpr std::uint32_t per_pair=64u;
    for(std::uint32_t i=0u;i<pairs.size();++i) {
        pairs[i].candidate_offset=i*per_pair;
        pairs[i].candidate_capacity=i==pairs.size()-1u?1u:per_pair;
    }
    constexpr std::uint32_t capacity=32u;
    std::array<SourcePcmBoxPairResultNative,pairs.size()> results{};
    std::vector<SourcePcmBoxCandidateNative> candidates(
        pairs.size()*per_pair);
    const std::size_t input_offset=4u;
    const std::size_t bounds_offset=input_offset+sizeof(boxes)+12u;
    const std::size_t transform_offset=bounds_offset+capacity*24u+4u;
    const std::size_t geometry_offset=transform_offset+capacity*32u+4u;
    const std::size_t pair_offset=geometry_offset+capacity*16u+4u;
    const std::size_t result_offset=pair_offset+sizeof(pairs)+4u;
    const std::size_t candidate_offset=result_offset+sizeof(results)+4u;
    const auto root=context.create_buffer(
        candidate_offset+candidates.size()*sizeof(candidates[0])+16u);
    context.upload(root,input_offset,boxes.data(),sizeof(boxes));
    context.upload(root,pair_offset,pairs.data(),sizeof(pairs));
    CudaDeviceAddressMap addresses;
    const auto inputs=addresses.register_slice(root,input_offset,sizeof(boxes));
    const auto bounds=addresses.register_slice(root,bounds_offset,capacity*24u);
    const auto transforms=addresses.register_slice(root,transform_offset,capacity*32u);
    const auto geometry=addresses.register_slice(root,geometry_offset,capacity*16u);
    const auto pair_data=addresses.register_slice(root,pair_offset,sizeof(pairs));
    const auto result_data=addresses.register_slice(root,result_offset,sizeof(results));
    const auto candidate_data=addresses.register_slice(
        root,candidate_offset,candidates.size()*sizeof(candidates[0]));
    {
        SourceScBoxTransformBoundsNativeStage producer(
            context,shader(transform_shader));
        SourcePcmBoxCandidatesNativeStage pcm(context,shader(candidate_shader));
        SourceSapDmaNative stream(context,addresses);
        producer.append(stream,addresses,inputs,bounds,transforms,geometry,
                        static_cast<std::uint32_t>(boxes.size()),capacity);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()));
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Source PCM candidate stage split the original device stream batch");
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    context.download(root,candidate_offset,candidates.data(),
                     candidates.size()*sizeof(candidates[0]));
    std::array<SourceScBoxTransformInputNative,capacity> by_id{};
    for(const auto& box:boxes)by_id[box.element_index]=box;
    const auto near=[](float a,float b) {return std::abs(a-b)<0.001f;};
    const auto vec=[](const float x[3]) {
        return fyzz::Vec3{x[0],x[1],x[2]};
    };
    const auto quat=[](const float x[4]) {
        return fyzz::Quat{x[0],x[1],x[2],x[3]};
    };
    std::uint32_t generated=0u;
    for(std::size_t i=0;i<pairs.size();++i) {
        require(results[i].edge_handle==pairs[i].edge_handle &&
                results[i].candidate_offset==pairs[i].candidate_offset,
                "Source PCM GPU candidate lost original Sc edge identity");
        if(pairs[i].first_element==pairs[i].second_element ||
           pairs[i].edge_handle==0xffffffffu) {
            require(results[i].status==1u,
                    "Source PCM GPU admitted a retired/invalid pair");
            continue;
        }
        const auto& a=by_id[pairs[i].first_element];
        const auto& b=by_id[pairs[i].second_element];
        std::array<fyzz::native::PcmContactCandidate,64> expected{};
        std::size_t count=0u;
        const bool overlap=fyzz::native::source_pcm_box_box_candidates(
            vec(a.half_extents),vec(a.position),quat(a.orientation),
            vec(b.half_extents),vec(b.position),quat(b.orientation),
            a.contact_distance+b.contact_distance,expected,count);
        require(results[i].overlapping==static_cast<std::uint32_t>(overlap),
                "Source PCM GPU SAT diverged from source separation");
        if(i==pairs.size()-1u) {
            require(count>1u && results[i].status==2u,
                    "Source PCM GPU did not report a bounded candidate overflow");
            continue;
        }
        require(results[i].status==0u && results[i].candidate_count==count,
                "Source PCM GPU candidate count differs from source");
        for(std::size_t k=0u;k<count;++k) {
            const auto& gpu=candidates[pairs[i].candidate_offset+k];
            const auto& cpu=expected[k];
            require(near(gpu.local_a[0],cpu.local_a.x) &&
                    near(gpu.local_a[1],cpu.local_a.y) &&
                    near(gpu.local_a[2],cpu.local_a.z) &&
                    near(gpu.local_b[0],cpu.local_b.x) &&
                    near(gpu.local_b[1],cpu.local_b.y) &&
                    near(gpu.local_b[2],cpu.local_b.z) &&
                    near(gpu.normal_b[0],cpu.normal_b.x) &&
                    near(gpu.normal_b[1],cpu.normal_b.y) &&
                    near(gpu.normal_b[2],cpu.normal_b.z) &&
                    near(gpu.separation,cpu.separation),
                    "Source PCM GPU candidate order or geometry differs from source");
        }
        generated+=static_cast<std::uint32_t>(count);
    }
    require(generated>=4u,"Source PCM GPU fixture produced no useful witnesses");
    addresses.unregister_slice(candidate_data);
    addresses.unregister_slice(result_data);
    addresses.unregister_slice(pair_data);
    addresses.unregister_slice(geometry);
    addresses.unregister_slice(transforms);
    addresses.unregister_slice(bounds);
    addresses.unregister_slice(inputs);
    require(addresses.live_allocations()==0u,
            "Source PCM GPU candidate fixture leaked address mappings");
    context.destroy_buffer(root);
}
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected box transform and source PCM shaders");
#ifdef FYZZXYZZY_TEST_D3D12
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1],argv[2]);
        std::cout<<"Source PCM GPU box candidate parity PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Source PCM GPU box candidate parity FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
