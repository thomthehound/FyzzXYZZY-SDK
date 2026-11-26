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
    constexpr RangeAccessScope upload{1u,1u},shader_read{2u,2u},
        shader_write{2u,4u},download{1u,8u},indirect{4u,16u};
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
    expect({{48u,64u,{7u,1u},shader_write}});
    use(32u,64u,download,false);
    expect({{32u,48u,upload,download},{48u,64u,shader_write,download}});
    use(64u,80u,upload,true);
    expect({{64u,80u,{7u,0u},upload}});
    use(0u,32u,download,false);
    expect({{0u,32u,upload,download}});
    use(128u,192u,shader_read,false);
    expect({{128u,192u,upload,shader_read}});
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
}

void verify_uav_ranges() {
    using fyzz::compute::PendingUavRanges;
    PendingUavRanges pending;
    const auto expect=[&](std::size_t begin,std::size_t end,
                          bool writes,bool conflict) {
        if(pending.conflicts(begin,end,writes)!=conflict)
            throw std::runtime_error("Incorrect UAV ordering dependency");
    };
    pending.record(0u,32u,false);
    expect(16u,48u,false,false);
    expect(16u,48u,true,true);
    expect(32u,64u,true,false);
    pending.record(64u,96u,true);
    expect(80u,112u,false,true);
    expect(80u,112u,true,true);
    expect(32u,64u,true,false);
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
    expect(128u,160u,true,false);
    pending.clear();
    for(std::size_t i=128u;i>0u;--i) {
        const auto begin=(i-1u)*4u;
        expect(begin,begin+4u,true,false);
        pending.record(begin,begin+4u,true);
    }
    expect(0u,512u,false,true);
    expect(512u,516u,false,false);
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
        verify_uav_ranges();
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
