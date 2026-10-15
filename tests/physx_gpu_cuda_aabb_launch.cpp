// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_aabb_launch_lowering.h"
#include "PxgKernelIndices.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
struct Box { float p[6]; };
struct Encoded { std::uint32_t v[6]; };
static_assert(sizeof(Box) == sizeof(Encoded) && sizeof(Box) == 24, "PhysX AABB ABI");
constexpr std::uint32_t sentinel = 0xcdcdcdcd;

std::vector<std::uint8_t> read_shader(const char* file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("Missing AABB shader");
    std::vector<char> bytes{std::istreambuf_iterator<char>{f},
                            std::istreambuf_iterator<char>{}};
    return {bytes.begin(), bytes.end()};
}
std::uint32_t encode(float value, std::uint32_t shift,
                     std::uint32_t envbits, std::uint32_t env,
                     bool maxbound, bool enable) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    auto ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    auto out = ((ordered >> shift) & ~1u) | (maxbound ? 1u : 0u);
    if (enable && envbits) {
        if (env != 0xffffffffu) out |= env << (32u - envbits);
        else out = maxbound ? 0xfe7fffffu : 0x01800000u;
    }
    return out;
}
void negative(CudaAabbLaunchLowering& bridge,
              const std::array<const void*,11>& args,
              std::uint32_t grid) {
    std::vector<BatchCommand> calls;
    auto invoke = [&](std::uint32_t id, std::uint32_t gx, std::uint32_t tx,
                      const void* const* values, std::size_t n) {
        bridge.append(calls,id,gx,tx,values,n);
    };
    auto rejected = [&](auto&& fn) {
        bool failed=false;
        try { fn(); }
        catch (const std::exception&) { failed=true; }
        if (!failed || !calls.empty())
            throw std::runtime_error("Invalid upstream CUDA launch was accepted");
    };
    rejected([&] { invoke(0xffffffffu,grid,256u,args.data(),args.size()); });
    rejected([&] { invoke(physx::PxgKernelIds::BP_TRANSLATE_AABBS,grid+1u,256u,args.data(),args.size()); });
    rejected([&] { invoke(physx::PxgKernelIds::BP_TRANSLATE_AABBS,grid,128u,args.data(),args.size()); });
    rejected([&] { invoke(physx::PxgKernelIds::BP_TRANSLATE_AABBS,grid,256u,args.data(),10u); });
    rejected([&] { invoke(physx::PxgKernelIds::BP_TRANSLATE_AABBS,grid,256u,nullptr,11u); });
}
void run(Context& ctx, AabbEncodingStage& shader) {
    constexpr std::uint32_t n = 37u, pad = 19u;
    constexpr std::size_t box_offset = 64u, dist_offset = 32u;
    constexpr std::size_t env_offset = 16u, output_offset = 128u;
    const auto box_bytes = (n + pad) * sizeof(Box) + box_offset;
    const auto dist_bytes = (n + pad) * sizeof(float) + dist_offset;
    const auto env_bytes = (n + pad) * sizeof(std::uint32_t) + env_offset;
    const auto output_bytes = (n + pad) * sizeof(Encoded) + output_offset;
    auto box = ctx.create_buffer(box_bytes, MemoryPlacement::DeviceLocal);
    auto dist = ctx.create_buffer(dist_bytes, MemoryPlacement::DeviceLocal);
    auto env = ctx.create_buffer(env_bytes, MemoryPlacement::DeviceLocal);
    auto out = ctx.create_buffer(output_bytes, MemoryPlacement::DeviceLocal);
    auto snapshot = ctx.create_buffer(output_bytes, MemoryPlacement::DeviceLocal);
    CudaDeviceAddressMap map;
    const auto bp = map.register_allocation(box,box_bytes);
    const auto dp = map.register_allocation(dist,dist_bytes);
    const auto ep = map.register_allocation(env,env_bytes);
    const auto op = map.register_allocation(out,output_bytes);
    std::vector<Box> boxes(n+pad);
    std::vector<float> distances(n+pad);
    std::vector<std::uint32_t> envs(n+pad);
    for(std::uint32_t i=0;i<n+pad;++i) {
        const float x = static_cast<float>(int(i)-21) * 0.25f;
        boxes[i] = {{x,x*0.25f,1.0f,x+1.0f,x*0.25f+2.0f,4.0f}};
        distances[i] = static_cast<float>(i%7u)*0.125f;
        envs[i] = i%13u==0u ? 0xffffffffu : i%7u;
    }
    std::vector<std::uint8_t> zero_box(box_bytes, 0);
    std::vector<std::uint8_t> zero_dist(dist_bytes, 0);
    std::vector<std::uint8_t> zero_env(env_bytes, 0);
    std::vector<std::uint32_t> untouched(output_bytes / 4u, sentinel);
    std::memcpy(zero_box.data()+box_offset,boxes.data(),sizeof(Box)*boxes.size());
    std::memcpy(zero_dist.data()+dist_offset,distances.data(),sizeof(float)*distances.size());
    std::memcpy(zero_env.data()+env_offset,envs.data(),sizeof(std::uint32_t)*envs.size());
    ctx.upload(box,0,zero_box.data(),zero_box.size());
    ctx.upload(dist,0,zero_dist.data(),zero_dist.size());
    ctx.upload(env,0,zero_env.data(),zero_env.size());
    ctx.upload(out,0,untouched.data(),output_bytes);
    ctx.upload(snapshot,0,untouched.data(),output_bytes);

    std::uint64_t argBounds = bp + box_offset;
    std::uint64_t argOutput = op + output_offset;
    std::uint64_t argDistance = dp + dist_offset;
    std::uint64_t argEnv = ep + env_offset;
    std::uint32_t shifts[3] = {4u, 4u, 4u};
    std::uint32_t envbits[3] = {2u, 3u, 4u};
    const std::array<const void*,11> args{{
        &argBounds,&argOutput,&argDistance,&argEnv,&n,
        &shifts[0],&shifts[1],&shifts[2],
        &envbits[0],&envbits[1],&envbits[2]
    }};
    const std::uint32_t cuda_grid = (n+31u)/32u;
    CudaAabbLaunchLowering bridge(map, shader);
    negative(bridge,args,cuda_grid);
    std::vector<BatchCommand> commands;
    bridge.append(commands,physx::PxgKernelIds::BP_TRANSLATE_AABBS,
                  cuda_grid,256u,args.data(),args.size());
    commands.emplace_back(CopyCommand{out,0u,snapshot,0u,output_bytes});
    argEnv = 0ull; // source omitted optional environment IDs for second launch
    bridge.append(commands,physx::PxgKernelIds::BP_TRANSLATE_AABBS,
                  cuda_grid,256u,args.data(),args.size());
    if (commands.size()!=3u)
        throw std::runtime_error("Missing source CUDA translation dispatch");
    for(std::size_t i : {0u,2u}) {
        const auto& d = std::get<DispatchCommand>(commands[i]);
        if (d.groups.x!=(n+15u)/16u || d.buffers.size()!=4u ||
            d.buffers[0].offset!=box_offset ||
            d.buffers[1].offset!=dist_offset ||
            d.buffers[3].offset!=output_offset ||
            d.buffers[0].buffer!=box ||
            d.buffers[3].buffer!=out ||
            d.buffers[2].buffer!=(i?dist:env) ||
            d.buffers[2].offset!=(i?dist_offset:env_offset))
            throw std::runtime_error("CUDA launch lowering lost source pointer binding");
    }
    ctx.wait_submission(ctx.submit_batch(commands));
    std::vector<std::uint32_t> first(output_bytes/4u), second(output_bytes/4u);
    ctx.download(snapshot,0,first.data(),output_bytes);
    ctx.download(out,0,second.data(),output_bytes);
    for(std::uint32_t idx=0;idx<output_bytes/4u;++idx) {
        auto check = [&](const std::vector<std::uint32_t>& output,bool with_env) {
            if(idx < output_offset/4u ||
               idx >= (output_offset+size_t(n)*sizeof(Encoded))/4u) {
                if(output[idx]!=sentinel)
                    throw std::runtime_error("CUDA AABB launch modified view canary");
                return;
            }
            const std::uint32_t local = idx-output_offset/4u;
            const std::uint32_t b = local/6u, c = local%6u, axis = c%3u;
            const float source = boxes[b].p[c];
            const float projected = source + (c>=3u?distances[b]:-distances[b]);
            const auto want = encode(projected,shifts[axis],envbits[axis],
                                     envs[b],c>=3u,with_env);
            if(output[idx]!=want)
                throw std::runtime_error(
                    "CUDA AABB native kernel output mismatch index "+
                    std::to_string(idx));
        };
        check(first,true);check(second,false);
    }
    if(map.unregister_allocation(bp)!=box ||
       map.unregister_allocation(dp)!=dist ||
       map.unregister_allocation(ep)!=env ||
       map.unregister_allocation(op)!=out)
        throw std::runtime_error("CUDA native GPU host mapping lost allocation");
    ctx.destroy_buffer(snapshot);ctx.destroy_buffer(out);
    ctx.destroy_buffer(env);ctx.destroy_buffer(dist);ctx.destroy_buffer(box);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected AABB shader path");
        Context ctx(
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
        shader.bytes=read_shader(argv[1]);
        AabbEncodingStage stage(ctx,std::move(shader));
        run(ctx,stage);
        std::cout<<"PhysX source CUDA AABB launch to native GPU PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"PhysX source CUDA AABB lowering FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
