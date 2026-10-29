// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

struct Aggregate {
    std::uint32_t root;
    std::uint32_t first;
    std::uint32_t size;
    std::uint32_t valid;
};
static_assert(sizeof(Aggregate)==16u);

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
    binary.bytes.assign(std::istreambuf_iterator<char>(stream),{});
    require(!binary.bytes.empty(),"Empty source aggregate GPU shader");
    return binary;
}

void verify(Context& context,const char* shader_path) {
    // Original PxgAggregate::updateBoundIndices list: one aggregate has
    // more than a warp of shapes, one is inactive and two are partial.
    constexpr std::array<Aggregate,5> aggregates{{
        {80u,0u,3u,1u},
        {81u,3u,1u,1u},
        {82u,4u,0u,0u},
        {83u,4u,2u,1u},
        {84u,6u,35u,1u}
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
    const auto original=bounds;
    auto expected=bounds;
    for(const auto& aggregate:aggregates) {
        if(!aggregate.valid)continue;
        constexpr float limit=std::numeric_limits<float>::max()*0.25f;
        Bounds combined={{limit,limit,limit},{-limit,-limit,-limit}};
        for(std::uint32_t i=0;i<aggregate.size;++i) {
            const auto id=members[aggregate.first+i];
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
    }

    const auto a=context.create_buffer(sizeof(bounds));
    const auto b=context.create_buffer(sizeof(distances));
    const auto c=context.create_buffer(sizeof(aggregates));
    const auto d=context.create_buffer(sizeof(members));
    context.upload(a,0u,bounds.data(),sizeof(bounds));
    context.upload(b,0u,distances.data(),sizeof(distances));
    context.upload(c,0u,aggregates.data(),sizeof(aggregates));
    context.upload(d,0u,members.data(),sizeof(members));

    ComputePipelineDesc pipeline;
    pipeline.shader=shader(shader_path);
    pipeline.storage_buffer_binding_count=4u;
    pipeline.push_constant_bytes=16u;
    pipeline.storage_buffer_views.assign(
        4u,StorageBufferViewKind::RawReadWrite);
    const auto program=context.create_compute_pipeline(pipeline);
    const std::array<BufferBinding,4> bindings{{
        {0u,a,0u,sizeof(bounds),BufferAccess::ReadWrite},
        {1u,b,0u,sizeof(distances),BufferAccess::Read},
        {2u,c,0u,sizeof(aggregates),BufferAccess::Read},
        {3u,d,0u,sizeof(members),BufferAccess::Read}
    }};
    constexpr std::array<std::uint32_t,4> params{{
        static_cast<std::uint32_t>(aggregates.size()),0u,0u,0u
    }};
    const auto ticket=context.submit_dispatch(
        program,{2u,1u,1u},
        std::vector<BufferBinding>(bindings.begin(),bindings.end()),
        params.data(),sizeof(params));
    context.wait_submission(ticket);
    context.download(a,0u,bounds.data(),sizeof(bounds));

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
    context.destroy_compute_pipeline(program);
    context.destroy_buffer(d);
    context.destroy_buffer(c);
    context.destroy_buffer(b);
    context.destroy_buffer(a);
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected compiled original aggregate bounds shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1]);
        std::cout<<"Original source GPU aggregate union bounds PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Original source GPU aggregate union bounds FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
