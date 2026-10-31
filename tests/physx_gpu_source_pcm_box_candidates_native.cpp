// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_box_transform_bounds_native.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <algorithm>
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
    std::array<SourceScBoxTransformInputNative,7> boxes{{
        {1u,{0.f,0.f,0.f},{0.f,0.f,0.f,1.f},{1.f,2.f,3.f},0.2f},
        {6u,{1.25f,0.1f,0.f},{0.f,0.f,z,w},{1.f,1.5f,2.5f},0.3f},
        {31u,{30.f,0.f,0.f},{0.f,0.f,0.f,1.f},{1.f,2.f,2.f},0.1f},
        {12u,{0.3f,0.4f,2.f},{z,0.f,0.f,w},{1.f,1.f,1.f},0.125f},
        {9u,{0.f,1.1f,-1.f},{0.f,z,0.f,w},{0.75f,1.7f,0.5f},0.15f},
        {20u,{-0.1f,-0.2f,0.75f},{0.f,0.f,0.707106781f,0.707106781f},
         {2.f,0.45f,0.8f},0.35f},
        {}
    }};
    constexpr std::size_t base_pairs=6u*5u;
    std::array<SourcePcmBoxPairNative,base_pairs+4u> pairs{};
    constexpr std::array<std::uint32_t,6> ids{1u,6u,31u,12u,9u,20u};
    std::size_t index=0u;
    for(const auto a:ids)
        for(const auto b:ids)
            if(a!=b) {
                const auto edge=17u+static_cast<std::uint32_t>(index)*13u;
                pairs[index++]={a,b,edge};
            }
    // Find a genuine source PCM batch of 5+ ordered witnesses; unlike
    // fixed 2-4 candidate fixtures, this exercises swap-last reduction.
    bool found_large_batch=false;
    for(unsigned a=1u;a<=8u && !found_large_batch;++a)
        for(unsigned b=1u;b<=8u && !found_large_batch;++b) {
            const float az=0.15f*static_cast<float>(a);
            const float ax=0.11f*static_cast<float>(b);
            const float sx=std::sin(ax*0.5f),cx=std::cos(ax*0.5f);
            const float sz=std::sin(az*0.5f),cz=std::cos(az*0.5f);
            const std::array<float,4> q{cz*sx,sz*sx,sz*cx,cz*cx};
            const auto trial=SourceScBoxTransformInputNative{
                2u,{0.3f,0.25f,0.35f},{q[0],q[1],q[2],q[3]},
                {1.2f,1.4f,1.1f},0.15f};
            std::array<fyzz::native::PcmContactCandidate,64> witness{};
            std::size_t n=0;
            const auto source_vec=[](const float v[3]) {
                return fyzz::Vec3{v[0],v[1],v[2]};
            };
            const auto source_quat=[](const float v[4]) {
                return fyzz::Quat{v[0],v[1],v[2],v[3]};
            };
            const bool overlap=fyzz::native::source_pcm_box_box_candidates(
                source_vec(boxes[0].half_extents),
                source_vec(boxes[0].position),
                source_quat(boxes[0].orientation),
                source_vec(trial.half_extents),source_vec(trial.position),
                source_quat(trial.orientation),
                boxes[0].contact_distance+trial.contact_distance,witness,n);
            if(overlap && n>4u) {
                boxes[6]=trial;
                found_large_batch=true;
            }
        }
    require(found_large_batch,"Source PCM test needs a multi-contact reduction fixture");
    pairs[index++]={1u,2u,504u};
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
    constexpr std::uint32_t edge_capacity=768u;
    std::vector<SourcePcmScEdgeCacheNative> edge_caches(edge_capacity);
    const std::size_t input_offset=4u;
    const std::size_t bounds_offset=input_offset+sizeof(boxes)+12u;
    const std::size_t transform_offset=bounds_offset+capacity*24u+4u;
    const std::size_t geometry_offset=transform_offset+capacity*32u+4u;
    const std::size_t pair_offset=geometry_offset+capacity*16u+4u;
    const std::size_t result_offset=pair_offset+sizeof(pairs)+4u;
    const std::size_t candidate_offset=result_offset+sizeof(results)+4u;
    const std::size_t edge_offset=
        candidate_offset+candidates.size()*sizeof(candidates[0])+4u;
    const auto root=context.create_buffer(
        edge_offset+edge_caches.size()*sizeof(edge_caches[0])+16u);
    context.upload(root,input_offset,boxes.data(),sizeof(boxes));
    context.upload(root,pair_offset,pairs.data(),sizeof(pairs));
    context.upload(root,edge_offset,edge_caches.data(),
                   edge_caches.size()*sizeof(edge_caches[0]));
    CudaDeviceAddressMap addresses;
    const auto inputs=addresses.register_slice(root,input_offset,sizeof(boxes));
    const auto bounds=addresses.register_slice(root,bounds_offset,capacity*24u);
    const auto transforms=addresses.register_slice(root,transform_offset,capacity*32u);
    const auto geometry=addresses.register_slice(root,geometry_offset,capacity*16u);
    const auto pair_data=addresses.register_slice(root,pair_offset,sizeof(pairs));
    const auto result_data=addresses.register_slice(root,result_offset,sizeof(results));
    const auto candidate_data=addresses.register_slice(
        root,candidate_offset,candidates.size()*sizeof(candidates[0]));
    const auto edge_data=addresses.register_slice(
        root,edge_offset,edge_caches.size()*sizeof(edge_caches[0]));
    SourceScBoxTransformBoundsNativeStage producer(
        context,shader(transform_shader));
    SourcePcmBoxCandidatesNativeStage pcm(context,shader(candidate_shader));
    {
        SourceSapDmaNative stream(context,addresses);
        producer.append(stream,addresses,inputs,bounds,transforms,geometry,
                        static_cast<std::uint32_t>(boxes.size()),capacity);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),
                   edge_capacity);
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Source PCM candidate stage split the original device stream batch");
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    context.download(root,candidate_offset,candidates.data(),
                     candidates.size()*sizeof(candidates[0]));
    context.download(root,edge_offset,edge_caches.data(),
                     edge_caches.size()*sizeof(edge_caches[0]));
    std::array<SourceScBoxTransformInputNative,capacity> by_id{};
    for(const auto& box:boxes)by_id[box.element_index]=box;
    const auto near=[](float a,float b) {return std::abs(a-b)<0.001f;};
    const auto vec=[](const float x[3]) {
        return fyzz::Vec3{x[0],x[1],x[2]};
    };
    const auto quat=[](const float x[4]) {
        return fyzz::Quat{x[0],x[1],x[2],x[3]};
    };
    std::uint32_t generated=0u,reduced_batches=0u;
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
        if(results[i].status!=0u || results[i].candidate_count!=count) {
            std::cerr<<"PCM count mismatch pair "<<i<<" elements ("
                     <<pairs[i].first_element<<","<<pairs[i].second_element
                     <<") source="<<count<<" gpu="<<results[i].candidate_count
                     <<" status="<<results[i].status
                     <<" overlap="<<results[i].overlapping<<std::endl;
            for(std::size_t k=0;k<results[i].candidate_count && k<pairs[i].candidate_capacity;++k) {
                const auto& witness=candidates[pairs[i].candidate_offset+k];
                std::cerr<<"  GPU "<<k<<" A="<<witness.local_a[0]<<","
                         <<witness.local_a[1]<<","<<witness.local_a[2]
                         <<" B="<<witness.local_b[0]<<","<<witness.local_b[1]
                         <<","<<witness.local_b[2]
                         <<" normal="<<witness.normal_b[0]<<","
                         <<witness.normal_b[1]<<","<<witness.normal_b[2]
                         <<" separation="<<witness.separation<<std::endl;
            }
            for(std::size_t k=0;k<count;++k) {
                const auto& witness=expected[k];
                std::cerr<<"  CPU "<<k<<" A="<<witness.local_a.x<<","
                         <<witness.local_a.y<<","<<witness.local_a.z
                         <<" B="<<witness.local_b.x<<","<<witness.local_b.y
                         <<","<<witness.local_b.z
                         <<" normal="<<witness.normal_b.x<<","
                         <<witness.normal_b.y<<","<<witness.normal_b.z
                         <<" separation="<<witness.separation<<std::endl;
            }
        }
        require(results[i].regenerated==1u,
                "Source PCM GPU initial frame bypassed regeneration");
        require(results[i].status==(overlap && count==0u?3u:0u) &&
                results[i].candidate_count==count,
                "Source PCM GPU candidate count differs from source");
        require(edge_caches[pairs[i].edge_handle].initialized==1u &&
                edge_caches[pairs[i].edge_handle].count==
                    std::min<std::size_t>(count,4u),
                "Source PCM GPU did not retain Sc edge-owned contacts");
        require(results[i].selected_count==std::min<std::size_t>(count,4u),
                "Source PCM GPU reduced witness count differs from source");
        std::array<std::uint8_t,4> chosen{};
        if(count>4u)
            chosen=fyzz::native::source_pcm_box_batch_selection(
                std::span<const fyzz::native::PcmContactCandidate>{
                    expected.data(),count},1.0f);
        else
            for(std::size_t k=0;k<count;++k)
                chosen[k]=static_cast<std::uint8_t>(k);
        for(std::size_t k=0;k<results[i].selected_count;++k) {
            const auto gpu=(results[i].selected_indices>>(8u*k))&255u;
            require(gpu==chosen[k],
                    "Source PCM GPU reduced witness ordering differs from source");
        }
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
        if(count>4u)++reduced_batches;
    }
    require(generated>=4u && reduced_batches>0u,
            "Source PCM GPU fixture did not execute batch reduction");
    // An overflowing regeneration must not turn into a cached quiet
    // frame. Retry the same pose, then enlarge only the candidate window.
    const auto overflow_edge=pairs.back().edge_handle;
    require(edge_caches[overflow_edge].initialized==0u &&
            edge_caches[overflow_edge].count==0u,
            "Overflow committed an incomplete Sc PCM edge cache");
    {
        SourceSapDmaNative stream(context,addresses);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),edge_capacity);
        stream.wait_at_fetch();
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    require(results.back().status==2u && results.back().regenerated==1u,
            "Same-pose overflow incorrectly used the empty Sc PCM cache");
    pairs.back().candidate_capacity=per_pair;
    context.upload(root,pair_offset,pairs.data(),sizeof(pairs));
    {
        SourceSapDmaNative stream(context,addresses);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),edge_capacity);
        stream.wait_at_fetch();
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    context.download(root,edge_offset,edge_caches.data(),
                     edge_caches.size()*sizeof(edge_caches[0]));
    require(results.back().status==0u && results.back().regenerated==1u &&
            results.back().candidate_count>1u &&
            edge_caches[overflow_edge].initialized==1u &&
            edge_caches[overflow_edge].count==results.back().selected_count,
            "Sc PCM retry did not regenerate complete witnesses after overflow");
    // A second GPU frame reuses the same Sc edge records without a host
    // candidate upload, second pipeline, or per-pair CPU collision call.
    {
        SourceSapDmaNative stream(context,addresses);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),edge_capacity);
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Source PCM refresh split the native device stream");
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    for(std::size_t i=0;i<pairs.size();++i) {
        if(pairs[i].first_element==pairs[i].second_element ||
           pairs[i].edge_handle==0xffffffffu)continue;
        require(results[i].regenerated==0u && results[i].status!=2u,
                "Unchanged Sc PCM edge unnecessarily regenerated");
        require(results[i].candidate_count<=4u,
                "Quiet Sc PCM frame did not reuse reduced manifold");
    }
    // Refresh moves the cached witness in the second box's local frame.
    // A small pose change must update separation without invoking SAT.
    const auto prior=edge_caches[pairs[0].edge_handle];
    require(prior.count>0u,"PCM refresh fixture requires retained witnesses");
    const auto bQuat=boxes[1].orientation;
    const float deltaX=0.002f;
    boxes[0].position[0]+=deltaX;
    context.upload(root,input_offset,boxes.data(),sizeof(boxes));
    {
        SourceSapDmaNative stream(context,addresses);
        producer.append(stream,addresses,inputs,bounds,transforms,geometry,
                        static_cast<std::uint32_t>(boxes.size()),capacity);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),edge_capacity);
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Sc PCM pose refresh split the native device stream");
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    context.download(root,edge_offset,edge_caches.data(),
                     edge_caches.size()*sizeof(edge_caches[0]));
    require(results[0].regenerated==0u,
            "Small Sc box motion needlessly regenerated the PCM manifold");
    const auto& old=prior.contacts[0];
    const auto& now=edge_caches[pairs[0].edge_handle].contacts[0];
    const float rz=1.0f-2.0f*bQuat[2]*bQuat[2];
    const float ry=-2.0f*bQuat[2]*bQuat[3];
    const float expectedDepth=old.separation+
        deltaX*(rz*old.normal_b[0]+ry*old.normal_b[1]);
    require(std::abs(now.separation-expectedDepth)<0.001f,
            "Sc PCM GPU refresh did not update B-local witness depth");
    // A larger motion must invalidate this edge and regenerate from the
    // device-owned shape cache; the old manifold cannot be republished.
    boxes[0].position[0]+=0.25f;
    context.upload(root,input_offset,boxes.data(),sizeof(boxes));
    {
        SourceSapDmaNative stream(context,addresses);
        producer.append(stream,addresses,inputs,bounds,transforms,geometry,
                        static_cast<std::uint32_t>(boxes.size()),capacity);
        pcm.append(stream,addresses,transforms,geometry,pair_data,
                   result_data,candidate_data,edge_data,
                   static_cast<std::uint32_t>(pairs.size()),capacity,
                   static_cast<std::uint32_t>(candidates.size()),edge_capacity);
        stream.wait_at_fetch();
    }
    context.download(root,result_offset,results.data(),sizeof(results));
    require(results[0].regenerated==1u,
            "Sc PCM GPU failed to invalidate and regenerate after large motion");
    addresses.unregister_slice(edge_data);
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
