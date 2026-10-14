// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/created_pairs_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::CreatedPairsArenaDesc;
using fyzz::physx_gpu::CreatedPairsArenaStage;
constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kInvalidProjection = 0xffffffffu;
constexpr std::uint32_t kInvalidHandle = 0x3fffffffu;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX created-pairs shader not found");
    const std::vector<char> values{std::istreambuf_iterator<char>{in},
                                   std::istreambuf_iterator<char>{}};
    return {values.begin(), values.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    CreatedPairsArenaDesc descriptor{};
    std::vector<std::uint32_t> created;
    std::uint32_t descriptor_offset = 0;
    std::uint32_t pair_capacity = 0;

    std::uint32_t reserve(std::size_t words, std::size_t alignment = 128) {
        while (arena.size()*4 % alignment) arena.push_back(kUnused);
        const auto index = arena.size();
        if (index > std::numeric_limits<std::uint32_t>::max()/4 ||
            words > std::numeric_limits<std::uint32_t>::max()/4 - index)
            throw std::out_of_range("PhysX arena offsets exceed uint32");
        arena.resize(index + words, kUnused);
        return static_cast<std::uint32_t>(index*4);
    }
    std::uint32_t& at(std::uint32_t address) {
        if (address % 4 || address/4 >= arena.size())
            throw std::out_of_range("PhysX test buffer address out of range");
        return arena[address/4];
    }
};

Packed pack(std::uint32_t capacity, std::uint32_t previous,
            std::uint32_t count, std::uint32_t descriptor_offset) {
    if (!capacity || count > capacity || previous > capacity-count ||
        descriptor_offset % 128)
        throw std::invalid_argument("Invalid PhysX created-pairs test dimensions");
    Packed p;
    p.arena.resize(descriptor_offset/4, kUnused);
    p.descriptor_offset = p.reserve(sizeof(CreatedPairsArenaDesc)/4, 16);
    if (p.descriptor_offset != descriptor_offset)
        throw std::runtime_error("PhysX descriptor alignment changed");
    auto& d = p.descriptor;
    d.num_created_handles = count;
    d.num_previous_handles = previous;
    d.prior.common.box_capacity = capacity;
    d.prior.common.endpoint_capacity = capacity*2;
    d.created_handles_offset = p.reserve(std::max(1u,count));
    for (std::uint32_t i=0; i<count; ++i) {
        const auto handle = capacity-i-1; // Distinct, non-contiguous when count is small.
        p.created.push_back(handle);
        p.at(d.created_handles_offset+i*4) = handle;
    }

    const std::uint32_t n = previous+count;
    p.pair_capacity = n+(n&1u);
    for (std::uint32_t axis=0; axis<3; ++axis) {
        d.prior.projection_offsets[axis] = p.reserve(std::max(1u,p.pair_capacity*2u));
        d.prior.common.box_handle_offsets[0][axis] = p.reserve(std::max(1u,p.pair_capacity*2u));
        for (std::uint32_t i=0; i<p.pair_capacity*2; ++i) {
            p.at(d.prior.projection_offsets[axis]+i*4) = 0x11110000u + axis*0x10000u+i;
            p.at(d.prior.common.box_handle_offsets[0][axis]+i*4) = 0x22220000u + axis*0x10000u+i;
        }
    }
    d.new_integer_bounds_offset = p.reserve(capacity*6u);
    d.prior.common.old_integer_bounds_offset = p.reserve(capacity*6u);
    for(std::uint32_t h=0;h<capacity;++h)
        for(std::uint32_t c=0;c<6;++c) {
            p.at(d.new_integer_bounds_offset+(h*6u+c)*4u) =
                (c < 3u ? 0x30000000u : 0x60000000u) + h*48u+c*2u;
            p.at(d.prior.common.old_integer_bounds_offset+(h*6u+c)*4u) =
                0x90000000u + h*12u+c;
        }
    std::memcpy(p.arena.data()+descriptor_offset/4,&d,sizeof(d));
    return p;
}

void physx_created_oracle(Packed& p) {
    // Source: broadphase.cu::markCreatedPairsLaunch, including uint2
    // projections/handles, end/start bit tags, empty old bounds, padding.
    const auto& d=p.descriptor;
    for(std::uint32_t i=0;i<d.num_created_handles;++i) {
        const auto h=p.created[i];
        const auto index=d.num_previous_handles+i;
        for(std::uint32_t axis=0;axis<3;++axis) {
            const auto projection=d.prior.projection_offsets[axis]+index*8u;
            const auto handles=d.prior.common.box_handle_offsets[0][axis]+index*8u;
            p.at(projection)=p.at(d.new_integer_bounds_offset+(h*6u+axis)*4u);
            p.at(projection+4)=p.at(d.new_integer_bounds_offset+(h*6u+axis+3u)*4u);
            p.at(handles)=(h<<3u)|3u;
            p.at(handles+4)=(h<<3u)|2u;
        }
        for(std::uint32_t c=0;c<6;++c)
            p.at(d.prior.common.old_integer_bounds_offset+(h*6u+c)*4u)=
                c<3u?0xff7fffffu:0x00800000u;
    }
    const auto n=d.num_previous_handles+d.num_created_handles;
    if(n&1u) for(std::uint32_t axis=0;axis<3u;++axis) {
        const auto projection=d.prior.projection_offsets[axis]+n*8u;
        const auto handles=d.prior.common.box_handle_offsets[0][axis]+n*8u;
        p.at(projection)=p.at(projection+4)=kInvalidProjection;
        p.at(handles)=p.at(handles+4)=kInvalidHandle;
    }
}

void run_case(Context& ctx, CreatedPairsArenaStage& stage,
              std::uint32_t capacity, std::uint32_t previous,
              std::uint32_t count, std::uint32_t offset) {
    auto input=pack(capacity,previous,count,offset);
    auto expected=input;
    if(count) physx_created_oracle(expected); // no dispatch for empty created list
    const auto bytes=static_cast<std::uint32_t>(input.arena.size()*4);
    const auto arena=ctx.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto mirror=ctx.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    ctx.upload(arena,0,input.arena.data(),bytes);
    std::vector<BatchCommand> ops;
    stage.append(ops,arena,bytes,offset,count);
    if(ops.size() != (count ? 1u:0u))
        throw std::runtime_error("Original markCreatedPairs host dispatch conditional changed");
    if(count) {
        const auto& dispatch=std::get<DispatchCommand>(ops.front());
        if(dispatch.groups.x!=32u || dispatch.groups.y!=1u || dispatch.groups.z!=1u)
            throw std::runtime_error("Source fixed 32x256 launch grid changed");
        std::uint32_t params[3]{};
        std::memcpy(params,dispatch.push_constants.data(),sizeof(params));
        if(params[0]!=offset || params[1]!=8192u || params[2]!=bytes)
            throw std::runtime_error("PhysX created-pair dispatch push constants changed");
    }
    ops.emplace_back(CopyCommand{arena,0,mirror,0,bytes});
    ctx.wait_submission(ctx.submit_batch(ops));
    std::vector<std::uint32_t> actual(input.arena.size());
    ctx.download(mirror,0,actual.data(),bytes);
    for(std::size_t i=0;i<actual.size();++i) {
        if(actual[i]!=expected.arena[i])
            throw std::runtime_error("PhysX markCreatedPairs mismatch at word "+
              std::to_string(i)+"; boxes "+std::to_string(capacity)+
              ", previous "+std::to_string(previous)+
              ", created "+std::to_string(count));
    }
    ctx.destroy_buffer(mirror);
    ctx.destroy_buffer(arena);
}

void negative_cases(CreatedPairsArenaStage& stage) {
    std::vector<BatchCommand> ops;
    for (const auto offset : {1u, 128u, 256u}) {
        bool rejected=false;
        try {
            stage.append(ops,1,192u,offset,1);
        }catch(const std::invalid_argument&){rejected=true;}
        if(!rejected || !ops.empty())
            throw std::runtime_error("Invalid markCreatedPairs arena descriptor accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Expected compiled created-pairs shader");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        shader.format=ShaderFormat::Dxil;
#else
        shader.format=ShaderFormat::Spirv;
#endif
        shader.bytes=load_shader(argv[1]);
        CreatedPairsArenaStage stage(context,std::move(shader));
        negative_cases(stage);
        run_case(context,stage,1,0,1,0);
        run_case(context,stage,17,8,0,128);
        run_case(context,stage,17,7,5,128);
        run_case(context,stage,33,11,4,256);
        run_case(context,stage,513,128,73,128);
        run_case(context,stage,8193,0,8193,256);
        std::cout<<"PhysX created-pairs and radix padding source oracle PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"PhysX created-pairs GPU oracle FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
