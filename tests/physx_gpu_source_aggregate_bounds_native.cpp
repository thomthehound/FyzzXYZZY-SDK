// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_gpu_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;

struct Bounds {
    float minimum[3];
    float maximum[3];
};
static_assert(sizeof(Bounds)==24u);

using Aggregate=fyzz::physx_gpu::SourceAggregateGpuNativeRecord;

void require(bool condition,const char* reason) {
    if(!condition)throw std::runtime_error(reason);
}

ShaderBinary shader(const char* path) {
    std::ifstream stream(path,std::ios::binary);
    if(!stream)throw std::runtime_error("Source aggregate GPU shader missing");
    ShaderBinary binary;
#if defined(FYZZXYZZY_TEST_D3D12)
    binary.format=ShaderFormat::Dxil;
#else
    binary.format=ShaderFormat::Spirv;
#endif
    binary.bytes.assign(std::istreambuf_iterator<char>{stream},
                        std::istreambuf_iterator<char>{});
    require(!binary.bytes.empty(),"Empty source aggregate GPU shader");
    return binary;
}

void verify(Context& context,const char* shader_path,
            const char* changed_shader_path) {
    // Original PxgAggregate::updateBoundIndices list: one aggregate has
    // more than a warp of shapes, one is inactive and two are partial.
    constexpr std::array<Aggregate,6> aggregates{{
        {80u,0u,3u,0u},
        {81u,3u,1u,0u},
        {82u,4u,0u,0u},           // Unadmitted empty aggregate.
        {83u,4u,2u,0u},
        {84u,6u,35u,0u},
        {0xffffffffu,6u,1u,0u}   // Retired aggregate slot.
    }};
    std::array<std::uint32_t,41> members{};
    for(std::size_t i=0;i<members.size();++i)
        members[i]=static_cast<std::uint32_t>(i);
    std::array<Bounds,96> bounds{};
    std::array<float,96> distances{};
    for(std::size_t i=0;i<bounds.size();++i) {
        const float x=static_cast<float>(i)*2.0f;
        bounds[i]={{x,-2.0f,-3.0f},{x+1.0f,2.0f,3.0f}};
        distances[i]=i%3u==0u?0.5f:0.25f;
    }
    std::array<std::uint32_t,3> changed{};
    for(const auto member:members)
        changed[member>>5u] |= 1u<<(member&31u);
    auto expected_changed=changed;
    const auto original=bounds;
    auto expected=bounds;
    for(const auto& aggregate:aggregates) {
        if(!aggregate.size || aggregate.root==0xffffffffu)continue;
        constexpr float limit=std::numeric_limits<float>::max()*0.25f;
        Bounds combined={{limit,limit,limit},{-limit,-limit,-limit}};
        for(std::uint32_t i=0;i<aggregate.size;++i) {
            const auto id=members[aggregate.first_member+i];
            for(unsigned axis=0;axis<3u;++axis) {
                combined.minimum[axis]=std::min(
                    combined.minimum[axis],
                    original[id].minimum[axis]-distances[id]);
                combined.maximum[axis]=std::max(
                    combined.maximum[axis],
                    original[id].maximum[axis]+distances[id]);
            }
        }
        expected[aggregate.root]=combined;
        expected_changed[aggregate.root>>5u] |=
            1u<<(aggregate.root&31u);
        for(std::uint32_t i=0;i<aggregate.size;++i) {
            const auto member=members[aggregate.first_member+i];
            expected_changed[member>>5u] &=
                ~(1u<<(member&31u));
        }
    }

    // PhysX's original device allocator can place multiple independent
    // PxgCudaBuffers in one buddy-heap root. These five 4-byte-aligned
    // slices intentionally start at different offsets within that root.
    constexpr std::size_t bounds_off=4u;
    constexpr std::size_t distances_off=bounds_off+sizeof(bounds)+4u;
    constexpr std::size_t aggregates_off=distances_off+sizeof(distances)+4u;
    constexpr std::size_t members_off=aggregates_off+sizeof(aggregates)+4u;
    constexpr std::size_t changed_off=members_off+sizeof(members)+4u;
    const auto root=context.create_buffer(changed_off+sizeof(changed)+4u);
    context.upload(root,bounds_off,bounds.data(),sizeof(bounds));
    context.upload(root,distances_off,distances.data(),sizeof(distances));
    context.upload(root,aggregates_off,aggregates.data(),sizeof(aggregates));
    context.upload(root,members_off,members.data(),sizeof(members));
    context.upload(root,changed_off,changed.data(),sizeof(changed));

    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    const auto bound_address=addresses.register_slice(
        root,bounds_off,sizeof(bounds));
    const auto distance_address=addresses.register_slice(
        root,distances_off,sizeof(distances));
    const auto aggregate_address=addresses.register_slice(
        root,aggregates_off,sizeof(aggregates));
    const auto member_address=addresses.register_slice(
        root,members_off,sizeof(members));
    const auto changed_address=addresses.register_slice(
        root,changed_off,sizeof(changed));
    SourceSapDmaNative stream(context,addresses);
    {
        SourceAggregateGpuNativeStage stage(
            context,shader(shader_path),shader(changed_shader_path));
        const SourceAggregateGpuNativeBuffers source{
            bound_address,distance_address,aggregate_address,
            member_address,changed_address,
            static_cast<std::uint32_t>(bounds.size()),
            static_cast<std::uint32_t>(aggregates.size()),
            static_cast<std::uint32_t>(members.size()),
            static_cast<std::uint32_t>(changed.size()),
            static_cast<std::uint32_t>(aggregates.size())
        };
        stage.append(stream,addresses,source);
        require(stream.commands().size()==2u,
                "Original aggregate owner did not append both GPU stages");
        // No intermediate host readback: original aggregate stages
        // share the source queue and complete at the native fetch fence.
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Source aggregate owner submitted separate GPU queues");
    }
    context.download(root,bounds_off,bounds.data(),sizeof(bounds));
    context.download(root,changed_off,changed.data(),sizeof(changed));
    require(changed==expected_changed,
            "Original GPU aggregate root/member changed bitmap does not match source");

    for(std::size_t id=0;id<bounds.size();++id)
        for(unsigned axis=0;axis<3u;++axis) {
            require(std::abs(bounds[id].minimum[axis]-
                             expected[id].minimum[axis])<=0.0001f &&
                    std::abs(bounds[id].maximum[axis]-
                             expected[id].maximum[axis])<=0.0001f,
                    "Original GPU aggregate union bound disagreed with source PxBounds3 envelope");
        }
    require(bounds[82].minimum[0]==original[82].minimum[0] &&
            bounds[82].maximum[0]==original[82].maximum[0],
            "Unadmitted aggregate modified a source BP volume");
    addresses.unregister_slice(changed_address);
    addresses.unregister_slice(member_address);
    addresses.unregister_slice(aggregate_address);
    addresses.unregister_slice(distance_address);
    addresses.unregister_slice(bound_address);
    require(addresses.live_allocations()==0u,
            "Original source aggregate GPU address mappings leaked");
    context.destroy_buffer(root);
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected compiled original aggregate bounds and changed-map shaders");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1],argv[2]);
        std::cout<<"Original source GPU aggregate union bounds PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Original source GPU aggregate union bounds FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
