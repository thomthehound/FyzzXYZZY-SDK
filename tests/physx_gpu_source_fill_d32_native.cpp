// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
constexpr std::size_t kBytes = 32768u;
constexpr std::uint32_t kCanary = 0xa5a5a5a5u;

void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
template<class F> void reject(F&& fn, const char* why) {
    bool did_throw = false;
    try { fn(); } catch (const std::exception&) { did_throw = true; }
    check(did_throw, why);
}
ShaderBinary load_shader(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Missing source D32 GPU fill shader");
    const std::vector<char> data{std::istreambuf_iterator<char>{file},
                                 std::istreambuf_iterator<char>{}};
    ShaderBinary b;
#if defined(FYZZXYZZY_TEST_D3D12)
    b.format = ShaderFormat::Dxil;
#else
    b.format = ShaderFormat::Spirv;
#endif
    b.bytes.assign(data.begin(), data.end());
    return b;
}

void run(Context& ctx, SourceFillD32NativeStage& stage) {
    std::array<BufferId,2> buffers{};
    std::array<std::vector<std::uint32_t>,2> expected{
        std::vector<std::uint32_t>(kBytes/4u,kCanary),
        std::vector<std::uint32_t>(kBytes/4u,kCanary)};
    std::array<std::vector<std::uint32_t>,2> initial = expected;
    for (auto& b : buffers)
        b = ctx.create_buffer(kBytes,MemoryPlacement::DeviceLocal);

    CudaDeviceAddressMap addresses;
    // 4-byte-aligned PhysX interior pointers; the native RAW base must
    // nonetheless be 16-byte aligned on D3D12.
    const auto a=addresses.register_slice(buffers[0],260u,4096u);
    const auto b=addresses.register_slice(buffers[1],1036u,4096u);

    // Force tiny RAW storage view limits to exercise the SAME window
    // calculation used for real multi-gigabyte PhysX allocations. Check
    // that each dispatch view maps to the exact original CUDA words,
    // without skipped or repeated words as alignment residues change.
    // This is CPU-only planning; GPU execution below remains unmodified.
    {
        constexpr std::uint32_t kFillWords = 300u;
        std::uint32_t visited = 0u;
        std::uint32_t window_count = 0u;
        while (visited < kFillWords) {
            const auto pointer = a + std::uint64_t(visited) * 4u;
            const auto count = source_d32_fill_chunk_words(
                addresses, pointer, kFillWords - visited, 256u, 256u);
            if (window_count == 0u) check(count == 63u,
                "Source D32 first chunk forgot the unaligned physical view base");
            if (window_count == 1u) check(count == 64u,
                "Source D32 subsequent chunk retained stale offset bias");
            check(count > 0u && count <= kFillWords - visited,
                  "Source D32 chunk planner failed to make bounded progress");
            std::uint32_t bias = 0u;
            const auto view = source_native_view(
                addresses, pointer, source_allocation_bytes(count, 4u),
                256u, 0u, BufferAccess::Write, bias, 256u);
            const auto resolved = addresses.resolve(pointer, count * 4u);
            check(view.buffer == buffers[0] &&
                  view.offset + bias == resolved.offset &&
                  view.offset % 256u == 0u && view.bytes <= 256u,
                  "Source D32 chunk did not preserve original address/view extent");
            visited += count;
            ++window_count;
        }
        check(visited == kFillWords && window_count == 5u,
              "Source D32 chunk boundaries did not cover requested allocation");
        reject([&] {
            source_d32_fill_chunk_words(addresses, a, 16u, 256u, 4u);
        }, "Source D32 chunk planner accepted view too small for offset bias");
        reject([&] {
            source_d32_fill_chunk_words(addresses, a + 2u, 16u, 256u, 256u);
        }, "Source D32 chunk planner accepted non-word-aligned virtual address");
    }

    std::vector<BatchCommand> commands;
    stage.append(commands, addresses, 0u, 0x12345678u, 0u);
    check(commands.empty(),"Zero-word CUDA memset created a GPU operation");
    reject([&] {stage.append(commands, addresses, a+2u, 0u, 16u);},
           "Unaligned original CUDA device pointer accepted");
    reject([&] {stage.append(commands, addresses, a+4096u-4u, 0u, 2u);},
           "Out-of-bounds original CUDA D32 memset accepted");
    check(commands.empty(),"Invalid CUDA memset partially appended a command");

    stage.append(commands,addresses,a,0u,257u);
    stage.append(commands,addresses,a+20u*4u,0x13579bdfu,71u);
    stage.append(commands,addresses,b,0xdeadbeefu,128u);
    check(commands.size()==3u,"Source D32 memset command count changed");
    for(const auto& command:commands) {
        const auto& op=std::get<DispatchCommand>(command);
        check(op.buffers.size()==1u&&op.buffers[0].offset%16u==0u&&
              op.buffers[0].access==BufferAccess::Write,
              "CUDA source memset native RAW view ABI changed");
    }

    for(std::uint32_t j=0u;j<257u;++j)
        expected[0][260u/4u+j]=0u;
    for(std::uint32_t j=20u;j<91u;++j)
        expected[0][260u/4u+j]=0x13579bdfu;
    for(std::uint32_t j=0u;j<128u;++j)
        expected[1][1036u/4u+j]=0xdeadbeefu;

    const std::vector<UploadSpan> upload{
        {buffers[0],0u,initial[0].data(),kBytes},
        {buffers[1],0u,initial[1].data(),kBytes}
    };
    ctx.submit_upload_batch(upload);
    // Source memset operations and earlier upload are GPU-ordered with
    // ordinary following kernels; no intermediary readback or CPU copy.
    const auto submission=ctx.submit_batch(commands);
    ctx.wait_submission(submission);
    std::vector<std::uint32_t> actual(kBytes/4u);
    for(std::size_t i=0u;i<buffers.size();++i) {
        ctx.download(buffers[i],0u,actual.data(),kBytes);
        check(actual==expected[i],
              "Source D32 memset contents or adjacent native allocation canaries corrupted");
    }
    for(auto id:buffers)ctx.destroy_buffer(id);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected source D32 GPU shader file");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        if(ctx.device_info().max_workgroup_invocations<256u ||
           ctx.device_info().max_workgroup_size[0]<256u ||
           !ctx.device_info().max_workgroup_count[0]) {
            std::cout<<"SKIP: 256-thread source D32 fill unavailable\n";
            return 77;
        }
        SourceFillD32NativeStage stage(ctx,load_shader(argv[1]));
        run(ctx,stage);
        std::cout<<"Original PhysX CUDA memsetD32Async native GPU fill PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Source native D32 fill FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
