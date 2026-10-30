// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_sc_box_transform_bounds_native.h"

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
using fyzz::physx_gpu::SourceScBoxTransformInputNative;

struct Bounds {
    float minimum[3];
    float maximum[3];
};
struct CachedTransform {
    float orientation[4];
    float position[3];
    std::uint32_t flags;
};
static_assert(sizeof(Bounds)==24u);
static_assert(sizeof(CachedTransform)==32u);

void require(bool condition,const char* error) {
    if(!condition)throw std::runtime_error(error);
}
ShaderBinary load(const char* path) {
    std::ifstream stream(path,std::ios::binary);
    require(bool(stream),"Source Sc GPU box transform shader missing");
    ShaderBinary binary;
#ifdef FYZZXYZZY_TEST_D3D12
    binary.format=ShaderFormat::Dxil;
#else
    binary.format=ShaderFormat::Spirv;
#endif
    binary.bytes.assign(std::istreambuf_iterator<char>{stream},
                        std::istreambuf_iterator<char>{});
    require(!binary.bytes.empty(),"Source Sc box transform shader empty");
    return binary;
}

void verify(Context& context,const char* shader) {
    using namespace fyzz::physx_gpu;
    constexpr float c=0.7071067811865476f;
    std::array<SourceScBoxTransformInputNative,4> shapes{{
        {1u,{2.f,3.f,4.f},{0.f,0.f,0.f,1.f},
         {1.f,2.f,3.f},0u},
        {6u,{4.f,-2.f,0.f},{0.f,0.f,c,c},
         {2.f,1.f,3.f},0u},
        {31u,{0.f,10.f,0.f},{c,0.f,0.f,c},
         {1.f,2.f,3.f},0u},
        {12u,{1.5f,-2.2f,0.75f},
         {0.09539871f,0.19079742f,0.28619614f,0.93412376f},
         {0.5f,1.2f,2.3f},0u}
    }};
    constexpr std::uint32_t capacity=32u;
    constexpr Bounds unchanged{{-88.f,-77.f,-66.f},{88.f,77.f,66.f}};
    std::array<Bounds,capacity> bounds;
    bounds.fill(unchanged);
    CachedTransform invalid{};
    invalid.flags=0xdeadbeefu;
    std::array<CachedTransform,capacity> transforms;
    transforms.fill(invalid);
    const std::size_t input_offset=4u;
    const std::size_t bounds_offset=input_offset+sizeof(shapes)+12u;
    const std::size_t cache_offset=bounds_offset+sizeof(bounds)+4u;
    const auto root=context.create_buffer(cache_offset+sizeof(transforms)+16u);
    context.upload(root,input_offset,shapes.data(),sizeof(shapes));
    context.upload(root,bounds_offset,bounds.data(),sizeof(bounds));
    context.upload(root,cache_offset,transforms.data(),sizeof(transforms));

    CudaDeviceAddressMap addresses;
    const auto input=addresses.register_slice(
        root,input_offset,sizeof(shapes));
    const auto output=addresses.register_slice(
        root,bounds_offset,sizeof(bounds));
    const auto cache=addresses.register_slice(
        root,cache_offset,sizeof(transforms));
    SourceSapDmaNative stream(context,addresses);
    {
        SourceScBoxTransformBoundsNativeStage stage(context,load(shader));
        stage.append(stream,addresses,input,output,cache,
                     static_cast<std::uint32_t>(shapes.size()),capacity);
        require(stream.commands().size()==1u,
                "Sc box transform did not enter source GPU stream");
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Sc box transform introduced an extra GPU submission");
        context.download(root,bounds_offset,bounds.data(),sizeof(bounds));
        context.download(root,cache_offset,transforms.data(),sizeof(transforms));

        const auto near=[](float a,float b) {
            return std::abs(a-b)<0.0001f;
        };
        const auto check_bounds=[&](std::uint32_t i,
                                    std::array<float,3> low,
                                    std::array<float,3> high) {
            for(unsigned axis=0;axis<3u;++axis)
                require(near(bounds[i].minimum[axis],low[axis]) &&
                        near(bounds[i].maximum[axis],high[axis]),
                        "Sc source GPU box bounds differ from PxBounds3::poseExtent");
        };
        check_bounds(1u,{1.f,1.f,1.f},{3.f,5.f,7.f});
        check_bounds(6u,{3.f,-4.f,-3.f},{5.f,0.f,3.f});
        check_bounds(31u,{-1.f,7.f,-2.f},{1.f,13.f,2.f});
        check_bounds(12u,{-0.42506928f,-3.62586260f,-1.83653189f},
                         {3.42506928f,-0.77413740f,3.33653189f});
        for(std::size_t shape=0;shape<shapes.size();++shape) {
            const auto index=shapes[shape].element_index;
            for(unsigned i=0;i<4u;++i)
                require(near(transforms[index].orientation[i],
                             shapes[shape].orientation[i]),
                        "GPU source transform cache lost Sc quaternion");
            for(unsigned i=0;i<3u;++i)
                require(near(transforms[index].position[i],
                             shapes[shape].position[i]),
                        "GPU source transform cache lost Sc translation");
            require(transforms[index].flags==0u,
                    "GPU source transform cache flags not initialized");
        }
        require(bounds[8u].minimum[0]==unchanged.minimum[0] &&
                transforms[8u].flags==invalid.flags,
                "Unchanged source element cache or AABB was overwritten");

        shapes[1].position[0]=7.f;
        context.upload(root,input_offset+sizeof(shapes[0]),
                       &shapes[1],sizeof(shapes[1]));
        stage.append(stream,addresses,input+sizeof(shapes[0]),
                     output,cache,1u,capacity);
        stream.wait_at_fetch();
        context.download(root,bounds_offset,bounds.data(),sizeof(bounds));
        context.download(root,cache_offset,transforms.data(),sizeof(transforms));
        check_bounds(6u,{6.f,-4.f,-3.f},{8.f,0.f,3.f});
        require(near(transforms[6u].position[0],7.f) &&
                bounds[1u].minimum[0]==1.f &&
                bounds[31u].minimum[1]==7.f,
                "Sparse device transform update corrupted quiet source shapes");
        bool rejected=false;
        try {
            stage.append(stream,addresses,input,output,cache,33u,capacity);
        }catch(const std::invalid_argument&) {rejected=true;}
        require(rejected && stream.commands().empty(),
                "Invalid source transform capacity partially enqueued GPU work");
    }
    addresses.unregister_slice(cache);
    addresses.unregister_slice(output);
    addresses.unregister_slice(input);
    require(addresses.live_allocations()==0u,
            "Source box transform test leaked GPU address mappings");
    context.destroy_buffer(root);
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected compiled Sc box transform shader");
#ifdef FYZZXYZZY_TEST_D3D12
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1]);
        std::cout<<"Source GPU box transform/cache/bounds PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Source GPU box transform/cache/bounds FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
