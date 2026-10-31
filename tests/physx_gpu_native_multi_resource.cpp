// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace fyzz::compute;

namespace {
constexpr std::size_t kElements = 64u;
constexpr std::size_t kViewBytes = kElements * sizeof(std::uint32_t);
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::uint32_t kOutputCanary = 0xbabababa;

void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

ShaderBinary read_shader(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Native multi-resource shader not found");
    std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>()};
    ShaderBinary shader;
#ifdef FYZZXYZZY_TEST_D3D12
    shader.format = ShaderFormat::Dxil;
#else
    shader.format = ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(), bytes.end());
    return shader;
}

void run_case(Context& context, PipelineId pipeline, bool shared_backing) {
    const std::size_t alignment = std::max<std::size_t>(
        kViewBytes, static_cast<std::size_t>(
            context.device_info().storage_buffer_offset_alignment));
    check(alignment % sizeof(std::uint32_t) == 0u,
          "Native descriptor alignment not word-aligned");
    const std::size_t second_offset = shared_backing ? alignment : 0u;
    const std::size_t first_bytes = shared_backing
        ? second_offset + 2u * kViewBytes : 2u * kViewBytes;

    std::vector<std::uint32_t> first(first_bytes / 4u, kCanary);
    std::vector<std::uint32_t> second(2u * kElements, kCanary);
    std::vector<std::uint32_t> output(2u * kElements, kOutputCanary);
    std::vector<std::uint32_t> mirror(2u * kElements, kOutputCanary);

    for (std::size_t i = 0; i < kElements; ++i) {
        first[i] = static_cast<std::uint32_t>(10u + i);
        if (shared_backing)
            first[second_offset / 4u + i] =
                static_cast<std::uint32_t>(200u + 2u * i);
        else
            second[i] = static_cast<std::uint32_t>(200u + 2u * i);
    }

    const auto source_a = context.create_buffer(first_bytes, MemoryPlacement::DeviceLocal);
    const auto source_b = shared_backing
        ? source_a : context.create_buffer(second.size() * 4u, MemoryPlacement::DeviceLocal);
    const auto destination = context.create_buffer(output.size() * 4u, MemoryPlacement::DeviceLocal);
    const auto snapshot = context.create_buffer(mirror.size() * 4u, MemoryPlacement::DeviceLocal);
    context.upload(source_a, 0u, first.data(), first_bytes);
    if (!shared_backing)
        context.upload(source_b, 0u, second.data(), second.size() * 4u);
    context.upload(destination, 0u, output.data(), output.size() * 4u);
    context.upload(snapshot, 0u, mirror.data(), mirror.size() * 4u);

    DispatchCommand dispatch;
    dispatch.pipeline = pipeline;
    dispatch.groups = {1u, 1u, 1u};
    dispatch.buffers = {
        {0u, source_a, 0u, kViewBytes, BufferAccess::ReadWrite},
        {1u, source_b, second_offset, kViewBytes, BufferAccess::ReadWrite},
        {2u, destination, 0u, kViewBytes, BufferAccess::Write}
    };
    const std::array<std::uint32_t, 8> parameters{{kElements, 0u, 0u, 0u, 0u, 0u, 0u, 0u}};
    dispatch.push_constants.resize(sizeof(parameters));
    std::memcpy(dispatch.push_constants.data(), parameters.data(), sizeof(parameters));

    // Validate the entire batch before any dispatch can execute. A
    // malformed later kernel must not mutate the buffers from the
    // preceding otherwise-valid kernel.
    auto invalid=dispatch;
    invalid.buffers[2].slot=1u;
    bool rejected=false;
    try {(void)context.submit_batch({dispatch,invalid});}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Malformed later dispatch entered native GPU batch");

    // Two dependent dispatches and a GPU copy must execute in source order
    // with no host synchronization between them. Both source arrays are
    // modified by the shader; the second dispatch reads their new contents.
    auto second_dispatch=dispatch;
#ifndef FYZZXYZZY_TEST_D3D12
    // Vulkan slot order is explicitly specified by BufferBinding::slot.
    // Its flattened prevalidated descriptor array must preserve the slots
    // even if a later dispatch supplies bindings in a different order.
    std::rotate(second_dispatch.buffers.begin(),
                second_dispatch.buffers.begin()+1,
                second_dispatch.buffers.end());
#endif
    const std::vector<BatchCommand> commands{
        dispatch, second_dispatch,
        CopyCommand{destination, 0u, snapshot, 0u, kViewBytes}
    };
    const auto submission = context.submit_batch(commands);
    context.wait_submission(submission);

    std::vector<std::uint32_t> actual_first(first.size());
    std::vector<std::uint32_t> actual_second(second.size());
    std::vector<std::uint32_t> actual_output(output.size());
    std::vector<std::uint32_t> actual_mirror(mirror.size());
    std::vector<DownloadSpan> readback{{
        source_a,0u,actual_first.data(),first_bytes}};
    if(!shared_backing)
        readback.push_back({
            source_b,0u,actual_second.data(),second.size()*4u});
    readback.push_back({
        destination,0u,actual_output.data(),output.size()*4u});
    readback.push_back({
        snapshot,0u,actual_mirror.data(),mirror.size()*4u});
    // Fetch GPU-written outputs and a dependent device copy together.
    // Multiple ranges may share one backing heap without separate fences.
    context.download_batch(readback);

    for (std::size_t i = 0; i < kElements; ++i) {
        const auto a = static_cast<std::uint32_t>(10u + i);
        const auto b = static_cast<std::uint32_t>(200u + 2u * i);
        first[i] = a + 2u;
        if (shared_backing)
            first[second_offset / 4u + i] = b + 4u;
        else
            second[i] = b + 4u;
        output[i] = 3u * (a + 1u) + (b + 2u);
        mirror[i] = output[i];
    }
    check(actual_first == first,
          "Native multi-resource source A changed outside its view or has incorrect results");
    if (!shared_backing)
        check(actual_second == second,
              "Native multi-resource source B changed outside its view or has incorrect results");
    check(actual_output == output,
          "Native multi-resource output or canary mismatch");
    check(actual_mirror == mirror,
          "Native multi-resource ordered GPU copy or canary mismatch");

    // Recycle a completed native batch with a different view offset but
    // the same buffer identity. A subsequent identical binding must remain
    // valid when its source buffer contents change between submissions.
    if(!shared_backing) {
        auto redirected=dispatch;
        for(auto& binding:redirected.buffers)
            binding.offset=kViewBytes;
        auto verify_redirected=[&](std::uint32_t seed,bool remap_one_slot=false) {
            auto current=redirected;
            if(remap_one_slot)
                current.buffers[1].offset=0u;
            std::vector<std::uint32_t> next_a(kElements),next_b(kElements);
            for(std::uint32_t i=0;i<kElements;++i) {
                next_a[i]=seed+i;
                next_b[i]=seed*2u+3u*i;
            }
            context.upload(source_a,kViewBytes,next_a.data(),kViewBytes);
            context.upload(source_b,current.buffers[1].offset,
                           next_b.data(),kViewBytes);
            const auto id=context.submit_batch({current});
            context.wait_submission(id);
            std::vector<std::uint32_t> next_result(kElements);
            context.download(destination,kViewBytes,
                             next_result.data(),kViewBytes);
            for(std::uint32_t i=0;i<kElements;++i)
                check(next_result[i]==3u*next_a[i]+next_b[i],
                      "Reused GPU descriptor set retained an outdated view");
        };
        verify_redirected(300u);
        verify_redirected(500u);
        // Only the second descriptor slot changes its original view.
        // All remaining completed native bindings are still reusable.
        verify_redirected(700u,true);
        verify_redirected(900u,true);
    }

    context.destroy_buffer(snapshot);
    context.destroy_buffer(destination);
    if (!shared_backing) context.destroy_buffer(source_b);
    context.destroy_buffer(source_a);
}

// A real native command batch may carry successive PhysX device-to-device
// relocations within one GPU allocation. Every copy reads the preceding
// copy's results; a shared staging resource must not corrupt later reads.
void verify_ordered_same_resource_copies(Context& context) {
    constexpr std::size_t words=128u;
    const auto buffer=context.create_buffer(words*sizeof(std::uint32_t),
                                            MemoryPlacement::DeviceLocal);
    const std::vector<BatchCommand> copies{
        CopyCommand{buffer,0u,buffer,128u,64u},
        CopyCommand{buffer,128u,buffer,256u,128u},
        CopyCommand{buffer,256u,buffer,448u,64u}
    };
    for(std::uint32_t seed:{13u,91u}) {
        std::vector<std::uint32_t> original(words),expected(words),actual(words);
        for(std::uint32_t i=0;i<words;++i)
            original[i]=seed*31u+i*17u;
        expected=original;
        // Nonoverlapping locations make sequential std::copy equivalent
        // to the original source CUDA device-copy ordering contract.
        for(const auto& op:copies) {
            const auto& copy=std::get<CopyCommand>(op);
            std::copy(expected.begin()+copy.source_offset/4u,
                      expected.begin()+(copy.source_offset+copy.bytes)/4u,
                      expected.begin()+copy.destination_offset/4u);
        }
        context.upload(buffer,0u,original.data(),words*4u);
        // A later invalid operation must reject the entire batch before
        // recording or submitting the earlier, otherwise-valid copy.
        auto invalid=copies;
        invalid.push_back(CopyCommand{buffer,0u,buffer,32u,64u});
        bool rejected=false;
        try {(void)context.submit_batch(invalid);}
        catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"Overlapping late DtoD copy passed native batch preflight");
        context.wait_submission(context.submit_batch(copies));
        context.download(buffer,0u,actual.data(),words*4u);
        check(actual==expected,
              "Ordered native same-buffer copies violated source dependencies or canaries");
    }
    // Two distinct source heap roots may relocate concurrently on the
    // ordered native queue. Their staging resources must remain exclusive
    // until both completion fences retire, even when the later fence is
    // reaped first at the original PhysX fetch boundary.
    const auto second_root=context.create_buffer(
        words*sizeof(std::uint32_t),MemoryPlacement::DeviceLocal);
    std::vector<std::uint32_t> initial_a(words),initial_b(words);
    for(std::uint32_t i=0;i<words;++i) {
        initial_a[i]=i*43u+37u;
        initial_b[i]=i*71u+59u;
    }
    auto expected_a=initial_a,expected_b=initial_b;
    auto copy_expected=[&](std::vector<std::uint32_t>& values) {
        for(const auto& op:copies) {
            const auto& copy=std::get<CopyCommand>(op);
            std::copy(values.begin()+copy.source_offset/4u,
                      values.begin()+(copy.source_offset+copy.bytes)/4u,
                      values.begin()+copy.destination_offset/4u);
        }
    };
    copy_expected(expected_a);
    copy_expected(expected_b);
    context.upload(buffer,0u,initial_a.data(),words*4u);
    context.upload(second_root,0u,initial_b.data(),words*4u);
    auto other_copies=copies;
    for(auto& op:other_copies) {
        auto& copy=std::get<CopyCommand>(op);
        copy.source=second_root;
        copy.destination=second_root;
    }
    const auto earlier=context.submit_batch(copies);
    const auto later=context.submit_batch(other_copies);
    context.wait_submission(later);
    context.wait_submission(earlier);
    std::vector<std::uint32_t> actual_a(words),actual_b(words);
    context.download(buffer,0u,actual_a.data(),words*4u);
    context.download(second_root,0u,actual_b.data(),words*4u);
    check(actual_a==expected_a && actual_b==expected_b,
          "Concurrent source relocations shared in-flight scratch or changed copy order");
    context.destroy_buffer(second_root);
    context.destroy_buffer(buffer);
}

// Opt-in CPU queue measurement for the original native CopyCommand data
// path. Reset and GPU completion are deliberately outside submit timing;
// source dataflow conformance above runs first on the same backend.
void profile_ordered_same_resource_copies(Context& context) {
    using Clock=std::chrono::steady_clock;
    constexpr std::size_t words=128u;
    constexpr std::size_t warmup=3u,trials=11u;
    const auto root=context.create_buffer(words*sizeof(std::uint32_t),
                                          MemoryPlacement::DeviceLocal);
    const std::vector<BatchCommand> copies{
        CopyCommand{root,0u,root,128u,64u},
        CopyCommand{root,128u,root,256u,128u},
        CopyCommand{root,256u,root,448u,64u}
    };
    std::vector<std::uint32_t> initial(words);
    for(std::uint32_t i=0;i<words;++i)
        initial[i]=173u+i*31u;
    std::vector<double> submitted,completed;
    submitted.reserve(trials);
    completed.reserve(trials);
    for(std::size_t iteration=0;iteration<warmup+trials;++iteration) {
        context.upload(root,0u,initial.data(),words*sizeof(std::uint32_t));
        const auto begin=Clock::now();
        const auto token=context.submit_batch(copies);
        const auto queued=Clock::now();
        context.wait_submission(token);
        const auto done=Clock::now();
        if(iteration>=warmup) {
            submitted.push_back(std::chrono::duration<double,std::micro>(
                queued-begin).count());
            completed.push_back(std::chrono::duration<double,std::micro>(
                done-begin).count());
        }
    }
    const auto median=[](std::vector<double> samples) {
        std::sort(samples.begin(),samples.end());
        return samples[samples.size()/2u];
    };
    std::printf(
        "native_copy_profile scope=same_buffer_chain backend=%s"
        " device=\"%s\" copies=3 bytes=64,128,64 trials=%zu"
        " median_host_submit_us=%.3f median_submit_wait_us=%.3f"
        " timed_upload_reset=no\n",
#ifdef FYZZXYZZY_TEST_D3D12
        "d3d12",
#else
        "vulkan",
#endif
        context.device_info().name.c_str(),trials,
        median(submitted),median(completed));
    context.destroy_buffer(root);
}

void verify_ordered_source_fence_retirement(Context& context) {
    constexpr std::size_t count=16u;
    std::array<std::uint32_t,count> first{},second{},actual{};
    for(std::size_t i=0;i<count;++i) {
        first[i]=static_cast<std::uint32_t>(i+17u);
        second[i]=static_cast<std::uint32_t>(i+103u);
    }
    const auto source=context.create_buffer(sizeof(first),MemoryPlacement::DeviceLocal);
    const auto target=context.create_buffer(sizeof(first),MemoryPlacement::DeviceLocal);
    const auto first_upload=context.submit_upload_batch(
        {{source,0u,first.data(),sizeof(first)}});
    const auto second_upload=context.submit_upload_batch(
        {{target,0u,second.data(),sizeof(second)}});
    const std::vector<BatchCommand> copy{CopyCommand{
        source,0u,target,0u,sizeof(first)}};
    const auto last=context.submit_batch(copy);
    const std::array<SubmissionId,3> ordered{
        first_upload,second_upload,last};
    const std::array<SubmissionId,3> invalid{
        first_upload,last,second_upload};
    bool rejected=false;
    try {context.wait_submissions(invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"Out-of-order source submissions were partially retired");
    context.wait_submissions(ordered);
    context.download(target,0u,actual.data(),sizeof(actual));
    check(actual==first,
          "Single-fence source retirement did not preserve queue copy order");
    rejected=false;
    try {context.wait_submission(first_upload);}
    catch(const std::exception&) {rejected=true;}
    check(rejected,"Single-fence source retirement retained consumed upload token");
    context.destroy_buffer(target);
    context.destroy_buffer(source);
}

void verify_batched_gpu_readback(Context& context) {
    const std::array<std::uint32_t,16> a{{
        3u,5u,7u,11u,13u,17u,19u,23u,
        29u,31u,37u,41u,43u,47u,53u,59u}};
    const std::array<std::uint32_t,16> b{{
        61u,67u,71u,73u,79u,83u,89u,97u,
        101u,103u,107u,109u,113u,127u,131u,137u}};
    const auto gpu_a=context.create_buffer(sizeof(a),MemoryPlacement::DeviceLocal);
    const auto gpu_b=context.create_buffer(sizeof(b),MemoryPlacement::DeviceLocal);
    context.upload(gpu_a,0u,a.data(),sizeof(a));
    context.upload(gpu_b,0u,b.data(),sizeof(b));
    std::array<std::uint32_t,8> out_a{},out_b{},out_repeated{};
    out_a.fill(kOutputCanary);
    out_b.fill(kOutputCanary);
    out_repeated.fill(kOutputCanary);
    const std::vector<DownloadSpan> transfers{{
        gpu_a,2u*4u,out_a.data()+1u,4u*4u},{
        gpu_b,5u*4u,out_b.data()+2u,3u*4u},{
        gpu_a,9u*4u,out_repeated.data()+3u,2u*4u}};
    auto invalid=transfers;
    invalid.back().source_offset=sizeof(a);
    bool rejected=false;
    try{context.download_batch(invalid);}
    catch(const std::out_of_range&){rejected=true;}
    check(rejected && out_a[1]==kOutputCanary &&
          out_b[2]==kOutputCanary &&
          out_repeated[3]==kOutputCanary,
          "Invalid late GPU readback span published partial CPU data");
    context.download_batch(transfers);
    for(std::size_t i=0;i<4u;++i)
        check(out_a[i+1u]==a[i+2u],
              "Packed GPU readback from first resource differs");
    for(std::size_t i=0;i<3u;++i)
        check(out_b[i+2u]==b[i+5u],
              "Packed GPU readback from second resource differs");
    for(std::size_t i=0;i<2u;++i)
        check(out_repeated[i+3u]==a[i+9u],
              "Packed GPU readback repeated source aliases differ");
    check(out_a.front()==kOutputCanary && out_a.back()==kOutputCanary &&
          out_b.front()==kOutputCanary && out_b.back()==kOutputCanary &&
          out_repeated.front()==kOutputCanary &&
          out_repeated.back()==kOutputCanary,
          "Packed GPU readback overwrote destination canaries");
    context.destroy_buffer(gpu_b);
    context.destroy_buffer(gpu_a);
}

// A GPU producer writes dispatch dimensions consumed by a dependent kernel
// within the same native batch, without a CPU readback.
void verify_gpu_written_indirect_dispatch(Context& context,PipelineId pipeline) {
    constexpr std::size_t bytes=kElements*sizeof(std::uint32_t);
    std::array<std::uint32_t,kElements> zeros{},ones{},output{};
    ones.fill(1u);
    output.fill(kOutputCanary);
    const auto a=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto b=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto args=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto result=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    context.upload(a,0u,zeros.data(),bytes);
    context.upload(b,0u,ones.data(),bytes);
    context.upload(result,0u,output.data(),bytes);
    const auto params=[](std::uint32_t count) {
        std::array<std::uint32_t,8> raw{count,0u,0u,0u,0u,0u,0u,0u};
        std::vector<std::uint8_t> data(sizeof(raw));
        std::memcpy(data.data(),raw.data(),sizeof(raw));
        return data;
    };
    DispatchCommand producer{};
    producer.pipeline=pipeline;
    producer.groups={1u,1u,1u};
    producer.buffers={
        {0u,a,0u,bytes,BufferAccess::ReadWrite},
        {1u,b,0u,bytes,BufferAccess::ReadWrite},
        {2u,args,0u,bytes,BufferAccess::Write}};
    producer.push_constants=params(3u);
    DispatchCommand consumer{};
    consumer.pipeline=pipeline;
    consumer.buffers={
        {0u,a,0u,bytes,BufferAccess::ReadWrite},
        {1u,b,0u,bytes,BufferAccess::ReadWrite},
        {2u,result,0u,bytes,BufferAccess::Write}};
    consumer.push_constants=params(kElements);
    consumer.indirect=DispatchIndirectArgs{args,0u};
    // Preflight must reject the whole batch before the producer writes.
    auto invalid=consumer;
    invalid.indirect->offset=bytes-8u;
    bool rejected=false;
    try {(void)context.submit_batch({producer,invalid});}
    catch(const std::out_of_range&){rejected=true;}
    check(rejected,"Invalid indirect argument window entered GPU batch");
    rejected=false;
    invalid=consumer;
    invalid.indirect->offset=2u;
    try {(void)context.submit_batch({producer,invalid});}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Misaligned indirect dispatch arguments were accepted");
    const auto done=context.submit_batch({producer,consumer});
    context.wait_submission(done);
    context.download(result,0u,output.data(),bytes);
    for(std::size_t i=0;i<kElements;++i)
        check(output[i]==(i<3u?6u:1u),
              "GPU-generated indirect dispatch dimensions or ordering incorrect");
    std::array<std::uint32_t,3> dimensions{};
    context.download(args,0u,dimensions.data(),sizeof(dimensions));
    check(dimensions==std::array<std::uint32_t,3>{1u,1u,1u},
          "GPU producer did not write the expected indirect dimensions");
    // With no sparse work, a GPU-written (0,0,0) must skip the
    // consumer entirely, leaving the destination canaries unchanged.
    context.upload(a,0u,zeros.data(),bytes);
    context.upload(b,0u,zeros.data(),bytes);
    output.fill(kOutputCanary);
    context.upload(result,0u,output.data(),bytes);
    context.wait_submission(context.submit_batch({producer,consumer}));
    context.download(result,0u,output.data(),bytes);
    check(std::all_of(output.begin(),output.end(),
                      [](auto v){return v==kOutputCanary;}),
          "Zero-work GPU indirect dispatch modified the destination");
    context.download(args,0u,dimensions.data(),sizeof(dimensions));
    check(dimensions==std::array<std::uint32_t,3>{0u,0u,0u},
          "GPU producer did not write zero indirect dimensions");
    context.destroy_buffer(result);
    context.destroy_buffer(args);
    context.destroy_buffer(b);
    context.destroy_buffer(a);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool benchmark=argc==3 &&
            std::string_view(argv[2])=="--benchmark-self-copy";
        if(argc!=2 && !benchmark)
            throw std::invalid_argument(
                "Expected compiled native multi-resource shader and optional benchmark");
        Context context(
#ifdef FYZZXYZZY_TEST_D3D12
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& device = context.device_info();
        if (device.max_workgroup_invocations < 64u ||
            device.max_workgroup_size[0] < 64u ||
            device.max_workgroup_count[0] < 1u) {
            std::cout << "SKIP: native 64-thread compute unavailable\n";
            return 77;
        }
        ComputePipelineDesc desc;
        desc.shader = read_shader(argv[1]);
        desc.storage_buffer_binding_count = 3u;
        desc.storage_buffer_views = {
            StorageBufferViewKind::RawReadWrite,
            StorageBufferViewKind::RawReadWrite,
            StorageBufferViewKind::RawReadWrite
        };
        desc.push_constant_bytes = 32u;
        const auto pipeline = context.create_compute_pipeline(desc);
        run_case(context, pipeline, false);
        run_case(context, pipeline, true);
        verify_gpu_written_indirect_dispatch(context,pipeline);
        verify_ordered_same_resource_copies(context);
        verify_batched_gpu_readback(context);
        verify_ordered_source_fence_retirement(context);
        if(benchmark)
            profile_ordered_same_resource_copies(context);
        context.destroy_compute_pipeline(pipeline);
        std::cout << "Native multi-resource binding and ordering PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Native multi-resource binding FAILED: " << ex.what() << '\n';
        return 1;
    }
}
