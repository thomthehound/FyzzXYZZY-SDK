// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"

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

void exercise_source_dma_order(Context& ctx, SourceFillD32NativeStage& stage) {
    // The original SAP host enqueues uploads before gpuDMAUp's 14-source-
    // operation segment. Buffer growth may insert raw DtoD copies between
    // upload groups; later descriptor uploads cannot bypass those copies.
    constexpr std::size_t words=1024u, oldAt=260u, grownAt=784u, descriptorAt=2048u;
    const auto root=ctx.create_buffer(words*4u,MemoryPlacement::DeviceLocal);
    std::vector<std::uint32_t> initial(words,kCanary), expected=initial;
    ctx.upload(root,0u,initial.data(),initial.size()*4u);

    CudaDeviceAddressMap addresses;
    const auto rawOld=addresses.register_source_slice(root,oldAt,256u,4u);
    const auto rawGrown=addresses.register_source_slice(root,grownAt,512u,16u);
    const auto descriptor=addresses.register_slice(root,descriptorAt,256u);
    std::array<std::uint32_t,64> before{},after{};
    std::array<std::uint32_t,16> header{};
    for(std::uint32_t i=0u;i<64u;++i){
        before[i]=i*31u+9u;
        after[i]=i*19u+7u;
    }
    header.fill(0x13572468u);
    SourceSapDmaNative stream(ctx,addresses);

    // A malformed later upload invalidates the whole adjacent group,
    // before any native transfer is submitted.
    reject([&]{
        stream.queue_uploads({{rawOld,before.data(),sizeof(before)},
                              {rawOld+253u,after.data(),sizeof(after)}});
    },"PhysX source DMA accepted a partially invalid upload group");
    stream.queue_uploads({{rawOld,before.data(),sizeof(before)},
                          {descriptor,header.data(),sizeof(header)}});
    // Source gpuDMAUp growth preserves the raw allocation's old bytes.
    stream.queue_device_copy(rawGrown,rawOld,sizeof(before));
    reject([&]{stream.queue_device_copy(rawGrown+4u,rawGrown,128u);},
           "PhysX source DMA accepted overlapping device copy");

    // A subsequent HtoD upload must remain AFTER the earlier DtoD, even
    // though that device copy was only queued in a native command batch.
    stream.queue_uploads({{rawOld,after.data(),sizeof(after)}});
    stage.append(stream.commands(),addresses,descriptor,0xcafebabeu,16u);
    stream.queue_device_copy(rawGrown+256u,descriptor,sizeof(header));
    check(stream.submit()!=0u,"Source GPU DMA stream lost its final submission");
    stream.wait_at_fetch();

    std::copy(before.begin(),before.end(),expected.begin()+grownAt/4u);
    std::copy(after.begin(),after.end(),expected.begin()+oldAt/4u);
    std::fill_n(expected.begin()+descriptorAt/4u,header.size(),0xcafebabeu);
    std::fill_n(expected.begin()+(grownAt+256u)/4u,header.size(),0xcafebabeu);
    std::vector<std::uint32_t> observed(words);
    ctx.download(root,0u,observed.data(),observed.size()*4u);
    check(observed==expected,
          "Original PhysX SAP HtoD/DtoD/dispatch source order or canaries changed");
    ctx.destroy_buffer(root);
}

void exercise_source_aabb_bitmap_retirement(
    Context& context,SourceFillD32NativeStage& stage) {
    const auto root=context.create_buffer(4096u,MemoryPlacement::DeviceLocal);
    CudaDeviceAddressMap addresses;
    const auto a=addresses.register_slice(root,256u,512u);
    const auto b=addresses.register_slice(root,1280u,512u);
    std::array<std::uint32_t,1024u> starting{};
    starting.fill(kCanary);
    context.upload(root,0u,starting.data(),sizeof(starting));
    std::array<std::uint32_t,4u> addedHost{1u,2u,3u,4u};
    std::array<std::uint32_t,4u> removedHost{5u,6u,7u,8u};
    SourceSapDmaNative stream(context,addresses);
    const SourceSapAabbReportBitmapNative added{a,512u,addedHost};
    const SourceSapAabbReportBitmapNative removed{b,512u,removedHost};

    // An invalid LATE GPU bitmap must not zero either host map or append
    // the earlier valid clear to the original PhysX source command list.
    reject([&] {
        retire_source_sap_aabb_report_bitmaps_native(
            stream,stage,addresses,added,
            {b+508u,512u,removedHost});
    },"Invalid second original AABB bitmap was accepted");
    check(stream.commands().empty() && addedHost[0]==1u &&
          removedHost[0]==5u,
          "Rejected AABB bitmap retirement consumed the source frame");

    // A fetched pair can fail Sc admission after both native bitmap
    // addresses validated. Discarded retirement preparations must leave
    // both host bitmaps and the pending GPU stream entirely unchanged.
    {
        const auto uncommitted=prepare_source_sap_aabb_report_retirement_native(
            stream,stage,addresses,added,removed);
        check(uncommitted.commands.size()==2u && stream.commands().empty() &&
              addedHost[0]==1u && removedHost[0]==5u,
              "Rejected Sc fetch partially retired original source bitmaps");
    }

    retire_source_sap_aabb_report_bitmaps_native(
        stream,stage,addresses,added,removed);
    check(stream.commands().size()==2u &&
          std::all_of(addedHost.begin(),addedHost.end(),
                      [](auto word){return word==0u;}) &&
          std::all_of(removedHost.begin(),removedHost.end(),
                      [](auto word){return word==0u;}),
          "Source AABB report publication did not retire both bitmaps");

    // Source's next frame uploads after the bitmap clears, in the SAME
    // native stream, without an extra CPU wait between frames.
    constexpr std::uint32_t nextFrame=0x197bdeafu;
    stream.queue_uploads({{a+8u,&nextFrame,sizeof(nextFrame)}});
    stream.wait_at_fetch();
    auto expected=starting;
    std::fill_n(expected.begin()+256u/4u,128u,0u);
    std::fill_n(expected.begin()+1280u/4u,128u,0u);
    expected[(256u+8u)/4u]=nextFrame;
    std::array<std::uint32_t,1024u> observed{};
    context.download(root,0u,observed.data(),sizeof(observed));
    check(observed==expected,
          "PhysX GPU bitmap retirement reordered fills and next-frame uploads");
    context.destroy_buffer(root);
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
        exercise_source_dma_order(ctx,stage);
        exercise_source_aabb_bitmap_retirement(ctx,stage);
        std::cout<<"Original PhysX CUDA memsetD32Async native GPU fill PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Source native D32 fill FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
