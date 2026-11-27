// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"
#include "compute_readback_cache.h"
#include "compute_range_hazards.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void verify_range_hazards() {
    using namespace fyzz::compute;
    constexpr RangeAccessScope upload{1ull<<32u,1u},shader_read{2u,1ull<<33u},
        shader_write{2u,1ull<<34u},download{1ull<<32u,8u},indirect{4u,16u};
    BufferRangeHazards hazards;
    struct Dependency {
        std::size_t begin,end;
        RangeAccessScope source,target;
        bool operator==(const Dependency&) const=default;
    };
    std::vector<Dependency> dependencies;
    const auto use=[&](std::size_t begin,std::size_t end,
                       RangeAccessScope scope,bool writes) {
        dependencies.clear();
        hazards.use(begin,end,scope,writes,
            [&](std::size_t a,std::size_t b,RangeAccessScope src,RangeAccessScope dst) {
                dependencies.push_back({a,b,src,dst});
            });
    };
    const auto expect=[&](std::initializer_list<Dependency> expected) {
        if(dependencies!=std::vector<Dependency>(expected))
            throw std::runtime_error("Incorrect byte-range dependency");
    };
    use(0u,64u,upload,true); expect({});
    use(128u,192u,upload,true); expect({});
    use(32u,96u,shader_read,false);
    expect({{32u,64u,upload,shader_read}});
    use(32u,96u,shader_read,false); expect({});
    use(48u,80u,download,false);
    expect({{48u,64u,upload,download}});
    use(48u,80u,indirect,false);
    expect({{48u,64u,upload,indirect}});
    use(48u,64u,shader_write,true);
    expect({{48u,64u,{(1ull<<32u)|6u,1u},shader_write}});
    use(32u,64u,download,false);
    expect({{32u,48u,upload,download},{48u,64u,shader_write,download}});
    use(64u,80u,upload,true);
    expect({{64u,80u,{(1ull<<32u)|6u,0u},upload}});
    use(0u,32u,download,false);
    expect({{0u,32u,upload,download}});
    use(128u,192u,shader_read,false);
    expect({{128u,192u,upload,shader_read}});
    // Stable subranges retain the producer and do not broaden dependencies.
    use(144u,160u,shader_read,false); expect({});
    use(144u,160u,download,false);
    expect({{144u,160u,upload,download}});
    use(0u,32u,shader_write,true);
    expect({{0u,32u,upload,shader_write}});
    for(unsigned i=0u;i<64u;++i) {
        use(8u,24u,shader_write,true);
        expect({{8u,24u,shader_write,shader_write}});
    }
    use(0u,32u,download,false);
    expect({{0u,32u,shader_write,download}});
    BufferRangeHazards contiguous;
    for(std::size_t i=0u;i<256u;++i)
        contiguous.use(i*4u,(i+1u)*4u,upload,true,
            [](auto,auto,auto,auto) {
                throw std::runtime_error("Disjoint writes synchronized");
            });
    dependencies.clear();
    contiguous.use(0u,1024u,download,false,
        [&](std::size_t a,std::size_t b,RangeAccessScope src,RangeAccessScope dst) {
            dependencies.push_back({a,b,src,dst});
        });
    expect({{0u,1024u,upload,download}});

    BufferRangeHazards host;
    const auto ignore=[](auto,auto,auto,auto) {};
    host.use(0u,64u,upload,true,ignore,true);
    host.use(16u,32u,shader_write,true,ignore,true);
    host.use(32u,48u,shader_read,false,ignore);
    host.use(128u,192u,upload,true,ignore);
    dependencies.clear();
    const auto publish=[&] {
        host.publish_host_writes([&](auto begin,auto end,auto writer) {
            dependencies.push_back({begin,end,writer,{}});
        });
    };
    publish();
    expect({{0u,16u,upload,{}},{16u,32u,shader_write,{}},
            {32u,48u,upload,{}},{48u,64u,upload,{}}});
    dependencies.clear();
    publish();
    expect({});
    host.use(48u,80u,shader_write,true,ignore,true);
    host.use(56u,64u,shader_write,true,ignore,true);
    publish();
    expect({{48u,80u,shader_write,{}}});
    dependencies.clear();
    host.use(56u,64u,shader_write,true,ignore,true);
    publish();
    expect({{56u,64u,shader_write,{}}});
}

void verify_operation_ranges() {
    using namespace fyzz::compute;
    constexpr RangeAccessScope read{2u,2u},write{2u,4u},both{2u,6u},
        upload{1u,1u},download{1u,8u};
    std::vector<OperationRangeUse> uses{
        {1u,32u,64u,write,true},
        {2u,0u,16u,write,true},
        {1u,0u,96u,read,false},
        {1u,48u,80u,write,true},
        {1u,100u,112u,read,false},
        {1u,112u,128u,read,false},
        {1u,40u,44u,read,false}};
    std::vector<OperationRangeUse> normalized;
    normalize_operation_ranges(uses,normalized);
    const std::vector<OperationRangeUse> expected{
        {1u,0u,32u,read,false},
        {1u,32u,80u,both,true},
        {1u,80u,96u,read,false},
        {1u,100u,128u,read,false},
        {2u,0u,16u,write,true}};
    if(normalized!=expected)
        throw std::runtime_error("Overlapping accesses lost exact byte-range intent");

    BufferRangeHazards hazards;
    hazards.use(0u,128u,upload,true,[](auto,auto,auto,auto) {});
    for(const auto& use:normalized)
        if(use.buffer==1u)
            hazards.use(use.begin,use.end,use.scope,use.writes,
                [](auto,auto,auto,auto) {});
    std::vector<OperationRangeUse> producers;
    hazards.use(0u,96u,download,false,
        [&](auto begin,auto end,auto source,auto) {
            producers.push_back({1u,begin,end,source,true});
        });
    if(producers!=std::vector<OperationRangeUse>{
        {1u,0u,32u,upload,true},
        {1u,32u,80u,both,true},
        {1u,80u,96u,upload,true}})
        throw std::runtime_error("Read-only aliases replaced the original producer");
}

void verify_dispatch_range_plans() {
    using namespace fyzz::compute;
    DispatchRangePlans plans;
    DispatchCommand first{};
    first.pipeline=1u;
    first.buffers={{0u,7u,0u,64u,BufferAccess::ReadWrite,{}}};
    first.accesses={{0u,{0u,16u},BufferAccess::Write}};
    auto repeat=first;
    repeat.groups.x=4u;
    repeat.push_constants={1u,2u,3u,4u};
    auto other=first;
    other.accesses[0].extent.offset=16u;
    auto reader=other;
    reader.accesses[0].access=BufferAccess::Read;
    auto indirect=reader;
    indirect.indirect=DispatchIndirectArgs{8u,0u};
    auto invalid=reader;
    invalid.accesses[0].extent.bytes=65u;
    std::size_t builds=0u;
    const auto prepare=[&](std::size_t table,const DispatchCommand& cmd) {
        return plans.prepare(table,cmd,[&](auto& ranges) {
            ++builds;
            cmd.for_each_access([&](const auto& binding,auto extent,auto access) {
                const auto begin=binding.offset+extent.offset;
                ranges.push_back({binding.buffer,begin,begin+extent.bytes,
                                  {2u,4u},access!=BufferAccess::Read});
            });
            if(cmd.indirect)
                ranges.push_back({cmd.indirect->buffer,cmd.indirect->offset,
                                  cmd.indirect->offset+12u,{4u,8u},false});
        });
    };
    const auto a=prepare(0u,first);
    const auto b=prepare(1u,other);
    if(prepare(0u,repeat)!=a || builds!=2u)
        throw std::runtime_error("Repeated dispatch rebuilt invariant access ranges");
    const auto c=prepare(1u,reader);
    const auto e=prepare(1u,indirect);
    if(b==c || c==e || plans.ranges(b)[0].begin!=16u ||
       !plans.ranges(b)[0].writes || plans.ranges(c)[0].writes ||
       plans.ranges(e).size()!=2u || plans.ranges(a)[0].begin!=0u)
        throw std::runtime_error("Distinct dispatch declarations lost their access ranges");
    bool rejected=false;
    try { prepare(1u,invalid); }
    catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected || prepare(1u,indirect)!=e)
        throw std::runtime_error("Malformed access reused or invalidated a valid range plan");
}

void verify_buffer_ranges() {
    using fyzz::compute::PendingBufferRanges;
    PendingBufferRanges pending;
    const auto expect=[&](std::size_t begin,std::size_t end,
                          bool writes,bool conflict) {
        if(pending.conflicts(begin,end,writes)!=conflict)
            throw std::runtime_error("Incorrect D3D12 buffer ordering dependency");
    };
    pending.record(0u,32u,false);
    expect(16u,48u,false,false);
    expect(16u,48u,true,true);
    expect(32u,64u,true,false);
    pending.record(64u,96u,true);
    expect(80u,112u,false,true);
    expect(80u,112u,true,true);
    expect(32u,64u,true,true);
    // One resource barrier orders all preceding accesses.
    pending.clear();
    expect(0u,128u,true,false);
    // Aliased views of a single dispatch contribute all their ranges.
    pending.record(64u,128u,false);
    pending.record(0u,64u,true);
    pending.record(32u,96u,true);
    expect(92u,96u,false,true);
    expect(96u,128u,false,false);
    expect(96u,128u,true,true);
    expect(128u,160u,true,true);
    expect(128u,160u,false,false);
    pending.clear();
    for(std::size_t i=128u;i>0u;--i) {
        const auto begin=(i-1u)*4u;
        expect(begin,begin+4u,true,i!=128u);
        expect(begin,begin+4u,false,false);
        pending.record(begin,begin+4u,true);
    }
    expect(0u,512u,false,true);
    expect(512u,516u,false,false);
    // Reader intervals are needed only until the first writer in an epoch.
    pending.clear();
    for(std::size_t i=0u;i<64u;++i)
        pending.record(i*8u,i*8u+4u,false);
    expect(16u,20u,true,true);
    expect(20u,24u,true,false);
    pending.record(20u,24u,true);
    for(std::size_t i=0u;i<64u;++i)
        pending.record(i*8u,i*8u+4u,false);
    expect(16u,20u,false,false);
    expect(20u,24u,false,true);
    expect(24u,28u,true,true);
    pending.clear();
    expect(16u,20u,true,false);
    pending.record(16u,20u,false);
    expect(16u,20u,true,true);
}

void verify_cache_working_set() {
    fyzz::compute::CompletedReadbackCache<unsigned> cache;
    unsigned released=0u;
    const auto release=[&](unsigned id) noexcept { released|=1u<<id; };
    cache.recycle(1u,4096u,release);
    if(cache.acquire(64u,release).capacity)
        throw std::runtime_error("Peak staging reused for a much smaller request");
    cache.recycle(2u,64u,release);
    cache.recycle(3u,96u,release);
    for(std::size_t i=0u;i<cache.reuse_window;++i) {
        auto entry=cache.acquire(48u,release);
        if(entry.resource!=2u || entry.capacity!=64u)
            throw std::runtime_error("Readback cache lost a near-sized working allocation");
        cache.recycle(entry.resource,entry.capacity,release);
    }
    if(released!=((1u<<1u)|(1u<<3u)) ||
       cache.acquire(4096u,release).capacity)
        throw std::runtime_error("Idle peak readback storage was not reclaimed");
    cache.recycle(4u,4096u,release);
    cache.recycle(5u,128u,release);
    cache.recycle(6u,256u,release);
    // A recent 64-byte entry must outlive older, larger allocations.
    auto entry=cache.acquire(64u,release);
    if(entry.resource!=2u)
        throw std::runtime_error("Readback cache lost best-fit storage");
    cache.recycle(entry.resource,entry.capacity,release);
    cache.recycle(7u,512u,release);
    if((released&(1u<<4u))==0u || (released&(1u<<2u))!=0u)
        throw std::runtime_error("Readback cache eviction did not follow demand");
}

} // namespace

int main() {
    try {
        verify_range_hazards();
        verify_operation_ranges();
        verify_dispatch_range_plans();
        verify_buffer_ranges();
        verify_cache_working_set();
        using namespace fyzz::compute;
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        std::array<std::uint32_t,64u> input{};
        for(std::uint32_t i=0u;i<input.size();++i)
            input[i]=i*29u+13u;
        const auto gpu=context.create_buffer(
            sizeof(input),MemoryPlacement::DeviceLocal);
        context.upload(gpu,0u,input.data(),sizeof(input));

        constexpr std::uint32_t sentinel=0x5a17cd03u;
        std::array<std::uint32_t,3u> first{};
        std::array<std::uint32_t,11u> second{};
        first.fill(sentinel);
        second.fill(sentinel);
        const std::array<DownloadSpan,1u> small{{
            {gpu,0u,first.data(),sizeof(first)}
        }};
        const std::array<DownloadSpan,1u> large{{
            {gpu,20u*sizeof(std::uint32_t),second.data(),sizeof(second)}
        }};
        const auto first_id=context.submit_download_batch(small);
        const auto second_id=context.submit_download_batch(large);
        if(first[0]!=sentinel || second[0]!=sentinel)
            throw std::runtime_error(
                "Queued GPU readbacks published without a host completion");
        context.finish_download_batch(second_id,true);
        for(std::size_t i=0u;i<second.size();++i)
            if(second[i]!=input[20u+i])
                throw std::runtime_error(
                    "Later readback did not preserve its own staging");
        if(first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error(
                "Later readback published the earlier Sc-owned output");
        context.finish_download_batch(first_id,true);
        for(std::size_t i=0u;i<first.size();++i)
            if(first[i]!=input[i])
                throw std::runtime_error(
                    "Earlier readback was corrupted by later staging");

        first.fill(sentinel);
        second.fill(sentinel);
        const auto live_id=context.submit_download_batch(large);
        const auto discarded_id=context.submit_download_batch(small);
        context.finish_download_batch(discarded_id,false);
        if(first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error(
                "Rejected readback mutated the caller-owned output");
        context.finish_download_batch(live_id,true);
        for(std::size_t i=0u;i<second.size();++i)
            if(second[i]!=input[20u+i])
                throw std::runtime_error(
                    "Abandoning one readback lost an independent result");
        first.fill(sentinel);
        second.fill(sentinel);
        const auto cancelled=context.submit_download_batch(small);
        if(!context.cancel_download_batch(cancelled))
            throw std::runtime_error("Readback cancellation retired its native token");
        const auto successor=context.submit_download_batch(large);
        const std::array<SubmissionId,2> cancelled_join{{cancelled,successor}};
        context.wait_submissions(cancelled_join);
        context.finish_download_batch(successor,true);
        for(std::size_t i=0u;i<second.size();++i)
            if(second[i]!=input[20u+i])
                throw std::runtime_error("Cancelled readback contaminated successor");
        if(first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error("Cancelled native readback published results");
        const auto reaped=context.submit_download_batch(small);
        context.wait_submission(reaped);
        if(context.cancel_download_batch(reaped) ||
           first[0]!=sentinel || first[2]!=sentinel)
            throw std::runtime_error("Cancelling reaped readback lost publication ownership");

        std::array<std::uint32_t,64u> peak{};
        const std::array<DownloadSpan,1u> peak_download{{
            {gpu,0u,peak.data(),sizeof(peak)}
        }};
        context.finish_download_batch(context.submit_download_batch(peak_download),true);
        if(peak!=input)
            throw std::runtime_error("Peak readback lost source data");
        for(std::uint32_t round=0u;round<6u;++round) {
            first.fill(sentinel);
            second.fill(sentinel);
            const auto earlier=context.submit_download_batch(small);
            const auto later=context.submit_download_batch(large);
            context.finish_download_batch(later,true);
            context.finish_download_batch(earlier,true);
            for(std::size_t i=0u;i<first.size();++i)
                if(first[i]!=input[i])
                    throw std::runtime_error(
                        "Reused small readback staging lost source data");
            for(std::size_t i=0u;i<second.size();++i)
                if(second[i]!=input[20u+i])
                    throw std::runtime_error(
                        "Reused large readback staging lost source data");
        }
        const auto staging=context.create_buffer(
            sizeof(input),MemoryPlacement::Upload);
        bool rejected=false;
        try {
            context.map_buffer(staging,HostMapAccess::Read);
        }catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected)
            throw std::runtime_error("Upload storage allowed CPU read mapping");
        auto* mapped=static_cast<std::uint32_t*>(
            context.map_buffer(staging,HostMapAccess::Write));
        for(std::size_t i=0u;i<input.size();++i)mapped[i]=input[i]+1000u;
        context.unmap_buffer(staging);
        rejected=false;
        try {
            context.submit_copy_buffer(gpu,0u,staging,0u,sizeof(input));
        }catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected)
            throw std::runtime_error("Upload storage allowed GPU destination");
        std::vector<BatchCommand> batch{
            CopyCommand{staging,0u,gpu,0u,sizeof(first)},
            CopyCommand{staging,sizeof(first),gpu,sizeof(first),4u},
            CopyCommand{staging,20u*sizeof(std::uint32_t),gpu,
                        20u*sizeof(std::uint32_t),sizeof(second)},
            // Overlapping destinations must retain last-writer order.
            CopyCommand{staging,40u*sizeof(std::uint32_t),gpu,4u,4u}
        };
        // Rejected producers neither execute earlier operations nor
        // publish a readback; the same output can be submitted again.
        const auto prior_upload=context.submit_upload_buffer(
            gpu,17u*4u,&input[17u],4u);
        auto invalid_batch=batch;
        invalid_batch.emplace_back(UploadCommand{gpu,16u*4u,{0u,0u,0u,0u}});
        invalid_batch.emplace_back(CopyCommand{gpu,0u,gpu,0u,4u});
        const auto invalid_size=invalid_batch.size();
        peak.fill(sentinel);
        rejected=false;
        try {
            context.submit_batch_with_readback(invalid_batch,peak_download);
        }catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || peak[0]!=sentinel || peak.back()!=sentinel ||
           invalid_batch.size()!=invalid_size)
            throw std::runtime_error("Rejected producer published or modified its command batch");
        context.wait_submission(prior_upload);
        context.finish_download_batch(
            context.submit_batch_with_readback(batch,peak_download),true);
        for(std::size_t i=0u;i<input.size();++i) {
            const bool written=i<=first.size() ||
                (i>=20u && i<20u+second.size());
            const auto expected=i==1u?input[40u]+1000u:
                input[i]+(written?1000u:0u);
            if(peak[i]!=expected)
                throw std::runtime_error("Mapped upload lost ordered copy or untouched range");
        }
        // Standalone and batched transfers share one ordered queue.
        // Read the later result first while the earlier ticket is live.
        const auto replacement=input[9u]+2000u;
        const auto single=context.submit_upload_buffer(
            gpu,9u*sizeof(std::uint32_t),&replacement,sizeof(replacement));
        const auto copied=context.submit_copy_buffer(
            gpu,9u*sizeof(std::uint32_t),gpu,10u*sizeof(std::uint32_t),4u);
        context.finish_download_batch(
            context.submit_download_batch(peak_download),true);
        if(peak[9u]!=replacement || peak[10u]!=replacement)
            throw std::runtime_error("Standalone transfer lost ordered dependency");
        const std::array<SubmissionId,2u> ordered{{single,copied}};
        context.wait_submissions(ordered);
        // Synchronous transfer observes the same destination and preserves
        // neighboring data; no independently recorded copy path is needed.
        std::uint32_t observed=0u;
        context.download(gpu,10u*sizeof(std::uint32_t),&observed,sizeof(observed));
        if(observed!=replacement)
            throw std::runtime_error("Synchronous transfer lost ordered result");
        // Retired handles must preserve pending GPU transfer resources.
        const auto retired=context.create_buffer(sizeof(input),MemoryPlacement::DeviceLocal);
        const auto retired_upload=context.submit_upload_buffer(
            retired,0u,input.data(),sizeof(input));
        const auto retired_copy=context.submit_copy_buffer(
            retired,0u,retired,32u*4u,32u*4u);
        const std::array<DownloadSpan,1u> retired_download{{
            {retired,0u,peak.data(),sizeof(peak)}
        }};
        const auto retired_readback=context.submit_download_batch(retired_download);
        context.retire_buffer(retired);
        context.finish_download_batch(retired_readback,true);
        for(std::size_t i=0u;i<peak.size();++i)
            if(peak[i]!=input[i%32u])
                throw std::runtime_error("Retired buffer lost a pending transfer");
        const std::array<SubmissionId,2u> retired_tokens{{retired_upload,retired_copy}};
        context.wait_submissions(retired_tokens);
        const auto small_copy=context.submit_copy_buffer(gpu,0u,gpu,4u,4u);
        context.wait_submission(small_copy);
        context.download(gpu,0u,first.data(),sizeof(first));
        if(first[0]!=first[1])
            throw std::runtime_error("Small relocation after peak lost source data");
        // Dependent relocations consume earlier writes within and across submissions.
        std::vector<BatchCommand> relocations{
            CopyCommand{gpu,9u*4u,gpu,11u*4u,4u},
            CopyCommand{gpu,11u*4u,gpu,12u*4u,4u}
        };
        context.finish_download_batch(
            context.submit_batch_with_readback(relocations,peak_download),true);
        if(peak[11u]!=replacement || peak[12u]!=replacement)
            throw std::runtime_error("Dependent relocations lost intra-batch ordering");
        const auto reused_copy=context.submit_copy_buffer(gpu,12u*4u,gpu,13u*4u,4u);
        context.finish_download_batch(
            context.submit_download_batch(peak_download),true);
        if(peak[13u]!=replacement)
            throw std::runtime_error("Direct relocation lost submission ordering");
        context.wait_submission(reused_copy);
        context.destroy_buffer(staging);
        context.destroy_buffer(gpu);
        std::cout<<"Compute independent readback ownership PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Compute independent readback ownership FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
