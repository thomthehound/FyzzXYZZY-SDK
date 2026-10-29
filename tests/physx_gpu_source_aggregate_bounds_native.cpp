// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"

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

struct Aggregate {
    std::uint32_t root;
    std::uint32_t first;
    std::uint32_t size;
    std::uint32_t reserved;
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
        expected_changed[aggregate.root>>5u] |=
            1u<<(aggregate.root&31u);
        for(std::uint32_t i=0;i<aggregate.size;++i) {
            const auto member=members[aggregate.first+i];
            expected_changed[member>>5u] &=
                ~(1u<<(member&31u));
        }
    }

    const auto a=context.create_buffer(sizeof(bounds));
    const auto b=context.create_buffer(sizeof(distances));
    const auto c=context.create_buffer(sizeof(aggregates));
    const auto d=context.create_buffer(sizeof(members));
    const auto e=context.create_buffer(sizeof(changed));
    context.upload(a,0u,bounds.data(),sizeof(bounds));
    context.upload(b,0u,distances.data(),sizeof(distances));
    context.upload(c,0u,aggregates.data(),sizeof(aggregates));
    context.upload(d,0u,members.data(),sizeof(members));
    context.upload(e,0u,changed.data(),sizeof(changed));

    ComputePipelineDesc pipeline;
    pipeline.shader=shader(shader_path);
    pipeline.storage_buffer_binding_count=4u;
    pipeline.push_constant_bytes=16u;
    pipeline.storage_buffer_views.assign(
        4u,StorageBufferViewKind::RawReadWrite);
    const auto program=context.create_compute_pipeline(pipeline);
    ComputePipelineDesc changed_pipeline;
    changed_pipeline.shader=shader(changed_shader_path);
    changed_pipeline.storage_buffer_binding_count=3u;
    changed_pipeline.push_constant_bytes=16u;
    changed_pipeline.storage_buffer_views.assign(
        3u,StorageBufferViewKind::RawReadWrite);
    const auto mark_program=context.create_compute_pipeline(changed_pipeline);
    const std::array<BufferBinding,4> bindings{{
        {0u,a,0u,sizeof(bounds),BufferAccess::ReadWrite},
        {1u,b,0u,sizeof(distances),BufferAccess::Read},
        {2u,c,0u,sizeof(aggregates),BufferAccess::Read},
        {3u,d,0u,sizeof(members),BufferAccess::Read}
    }};
    constexpr std::array<std::uint32_t,4> params{{
        static_cast<std::uint32_t>(aggregates.size()),0u,0u,0u
    }};
    // Source order: computeAggregateBounds then GPU changed-handle
    // bitmap translation. Both stages run in one ordered native batch,
    // with no CPU readback between the aggregate and SAP handoff.
    DispatchCommand union_command;
    union_command.pipeline=program;
    union_command.groups={2u,1u,1u};
    union_command.buffers.assign(bindings.begin(),bindings.end());
    union_command.push_constants.resize(sizeof(params));
    std::memcpy(union_command.push_constants.data(),
                params.data(),sizeof(params));
    DispatchCommand mark_command;
    mark_command.pipeline=mark_program;
    mark_command.groups={2u,1u,1u};
    mark_command.buffers={
        {0u,c,0u,sizeof(aggregates),BufferAccess::Read},
        {1u,d,0u,sizeof(members),BufferAccess::Read},
        {2u,e,0u,sizeof(changed),BufferAccess::ReadWrite}
    };
    mark_command.push_constants.resize(sizeof(params));
    std::memcpy(mark_command.push_constants.data(),
                params.data(),sizeof(params));
    const std::vector<BatchCommand> commands{
        std::move(union_command),std::move(mark_command)
    };
    const auto ticket=context.submit_batch(commands);
    context.wait_submission(ticket);
    context.download(a,0u,bounds.data(),sizeof(bounds));
    context.download(e,0u,changed.data(),sizeof(changed));
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
    context.destroy_compute_pipeline(mark_program);
    context.destroy_compute_pipeline(program);
    context.destroy_buffer(e);
    context.destroy_buffer(d);
    context.destroy_buffer(c);
    context.destroy_buffer(b);
    context.destroy_buffer(a);
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
