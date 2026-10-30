// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_box_transform_bounds_native.h"
#include "physx_gpu/source_gu_box_box_axis_native.h"
#include "physx_gpu/source_gu_box_box_contacts_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}
ShaderBinary shader(const char* filename) {
    std::ifstream file(filename,std::ios::binary);
    require(bool(file),"Missing source Gu GPU shader");
    ShaderBinary binary;
#ifdef FYZZXYZZY_TEST_D3D12
    binary.format=ShaderFormat::Dxil;
#else
    binary.format=ShaderFormat::Spirv;
#endif
    binary.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    require(!binary.bytes.empty(),"Empty source Gu GPU shader");
    return binary;
}
void verify(Context& context,const char* transform_shader,
            const char* axis_shader,const char* contacts_shader) {
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
    constexpr std::size_t fixed_pairs=6u;
    std::array<SourceGuBoxPairAxisNative,
               fixed_pairs+6u*5u*7u+2u> pairs{{
        {1u,6u,0u},{1u,6u,1u},{6u,1u,0u},
        {1u,31u,0u},{1u,12u,0u},{6u,12u,0u}
    }};
    constexpr std::array<std::uint32_t,6> ids{1u,6u,31u,12u,9u,20u};
    std::size_t next=fixed_pairs;
    for(const auto a:ids)
        for(const auto b:ids)
            if(a!=b)
                for(std::uint32_t axis=0u;axis<=6u;++axis)
                    pairs[next++]={a,b,axis};
    pairs[next++]={31u,31u,0u};
    pairs[next++]={1u,6u,0u};
    require(next==pairs.size(),"Gu pair coverage fixture size mismatch");
    constexpr std::uint32_t contacts_per_pair=80u;
    for(std::uint32_t i=0u;i<pairs.size();++i) {
        pairs[i].contact_offset=i*contacts_per_pair;
        pairs[i].contact_capacity=i==pairs.size()-1u?1u:contacts_per_pair;
    }
    constexpr std::uint32_t capacity=32u;
    std::array<std::uint32_t,pairs.size()> results{};
    std::array<SourceGuBoxPairResultNative,pairs.size()> manifolds{};
    std::vector<SourceGuBoxContactNative> gpu_contacts(
        pairs.size()*contacts_per_pair);
    const std::size_t input_offset=4u;
    const std::size_t bounds_offset=input_offset+sizeof(boxes)+12u;
    const std::size_t transform_offset=bounds_offset+capacity*24u+4u;
    const std::size_t geometry_offset=transform_offset+capacity*32u+4u;
    const std::size_t pair_offset=geometry_offset+capacity*16u+4u;
    const std::size_t axis_offset=pair_offset+sizeof(pairs)+4u;
    const std::size_t result_offset=axis_offset+sizeof(results)+4u;
    const std::size_t contact_offset=result_offset+sizeof(manifolds)+4u;
    const auto root=context.create_buffer(
        contact_offset+gpu_contacts.size()*sizeof(gpu_contacts[0])+16u);
    context.upload(root,input_offset,boxes.data(),sizeof(boxes));
    context.upload(root,pair_offset,pairs.data(),sizeof(pairs));
    CudaDeviceAddressMap addresses;
    const auto input=addresses.register_slice(root,input_offset,sizeof(boxes));
    const auto bounds=addresses.register_slice(root,bounds_offset,capacity*24u);
    const auto transforms=addresses.register_slice(root,transform_offset,capacity*32u);
    const auto geometry=addresses.register_slice(root,geometry_offset,capacity*16u);
    const auto pair_data=addresses.register_slice(root,pair_offset,sizeof(pairs));
    const auto axes=addresses.register_slice(root,axis_offset,sizeof(results));
    const auto result_data=addresses.register_slice(root,result_offset,sizeof(manifolds));
    const auto contact_data=addresses.register_slice(
        root,contact_offset,gpu_contacts.size()*sizeof(gpu_contacts[0]));
    {
        SourceScBoxTransformBoundsNativeStage producer(
            context,shader(transform_shader));
        SourceGuBoxBoxAxisNativeStage gu(context,shader(axis_shader));
        SourceGuBoxBoxContactsNativeStage contact_stage(
            context,shader(contacts_shader));
        SourceSapDmaNative stream(context,addresses);
        producer.append(stream,addresses,input,bounds,transforms,geometry,
                        static_cast<std::uint32_t>(boxes.size()),capacity);
        gu.append(stream,addresses,transforms,geometry,pair_data,axes,
                  static_cast<std::uint32_t>(pairs.size()),capacity);
        contact_stage.append(stream,addresses,transforms,geometry,pair_data,
                             axes,result_data,contact_data,
                             static_cast<std::uint32_t>(pairs.size()),capacity,
                             static_cast<std::uint32_t>(gpu_contacts.size()));
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Source Gu witness production split the device command batch");
    }
    context.download(root,axis_offset,results.data(),sizeof(results));
    context.download(root,result_offset,manifolds.data(),sizeof(manifolds));
    context.download(root,contact_offset,gpu_contacts.data(),
                     gpu_contacts.size()*sizeof(gpu_contacts[0]));
    std::array<SourceScBoxTransformInputNative,capacity> by_id{};
    for(const auto& box:boxes)by_id[box.element_index]=box;
    unsigned matched=0;
    for(std::size_t i=0;i<pairs.size();++i) {
        if(pairs[i].first_element==pairs[i].second_element) {
            require(results[i]==0xffffffffu && manifolds[i].status==1u,
                    "Source Gu device contacts accepted the same shape twice");
            continue;
        }
        const auto& a=by_id[pairs[i].first_element];
        const auto& b=by_id[pairs[i].second_element];
        const auto vec=[](const float v[3]) {
            return fyzz::Vec3{v[0],v[1],v[2]};
        };
        const auto quat=[](const float q[4]) {
            return fyzz::Quat{q[0],q[1],q[2],q[3]};
        };
        std::array<fyzz::native::ContactPoint,255> contacts{};
        auto previous=static_cast<std::uint8_t>(pairs[i].previous_axis);
        const auto count=fyzz::native::source_box_box_contacts(
            vec(a.half_extents),vec(a.position),quat(a.orientation),
            vec(b.half_extents),vec(b.position),quat(b.orientation),
            a.contact_distance+b.contact_distance,previous,contacts);
        if(i==pairs.size()-1u) {
            require(count>1u && manifolds[i].status==2u,
                    "GPU Gu contact window overflow was not reported");
            continue;
        }
        require(manifolds[i].status==0u && manifolds[i].count==count &&
                manifolds[i].collision_axis==previous,
                "Device Gu contact count or pair-axis state differs from source");
        const auto near=[](float a,float b) {
            return std::abs(a-b)<0.001f;
        };
        for(std::uint32_t contact=0u;contact<count;++contact) {
            const auto& gpu=gpu_contacts[pairs[i].contact_offset+contact];
            const auto& cpu=contacts[contact];
            require(near(gpu.position[0],cpu.position.x) &&
                    near(gpu.position[1],cpu.position.y) &&
                    near(gpu.position[2],cpu.position.z) &&
                    near(gpu.normal[0],cpu.normal.x) &&
                    near(gpu.normal[1],cpu.normal.y) &&
                    near(gpu.normal[2],cpu.normal.z) &&
                    near(gpu.separation,cpu.separation),
                    "Device Gu witness differs from source ordered contact");
        }
        if(count) {
            require(results[i]==previous,
                    "GPU Gu SAT selected a different source contact axis");
            ++matched;
        }else if(pairs[i].second_element==31u) {
            require(results[i]==0u,
                    "GPU Gu cross/face separation admitted a distant box");
            ++matched;
        }
    }
    require(matched>=4u,"Source Gu contact fixtures did not exercise enough axes");
    addresses.unregister_slice(contact_data);
    addresses.unregister_slice(result_data);
    addresses.unregister_slice(axes);
    addresses.unregister_slice(pair_data);
    addresses.unregister_slice(geometry);
    addresses.unregister_slice(transforms);
    addresses.unregister_slice(bounds);
    addresses.unregister_slice(input);
    require(addresses.live_allocations()==0u,
            "Source Gu GPU axis fixture leaked address mappings");
    context.destroy_buffer(root);
}
}
int main(int argc,char** argv) {
    try {
        require(argc==4,"Expected transform, Gu axis, and contact shader binaries");
#ifdef FYZZXYZZY_TEST_D3D12
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1],argv[2],argv[3]);
        std::cout<<"Source Gu GPU box contact parity PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Source Gu GPU box axis parity FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
