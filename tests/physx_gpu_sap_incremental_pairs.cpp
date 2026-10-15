// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_endpoint_histogram_arena.h"
#include "physx_gpu/sap_incremental_comparisons_arena.h"
#include "physx_gpu/sap_incremental_pairs_arena.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::SapEndpointHistogramArenaDesc;
using fyzz::physx_gpu::SapEndpointHistogramArenaStage;
using fyzz::physx_gpu::SapIncrementalComparisonArenaDesc;
using fyzz::physx_gpu::SapIncrementalComparisonsArenaStage;
using fyzz::physx_gpu::SapIncrementalPairsArenaDesc;
using fyzz::physx_gpu::SapIncrementalPairsArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kBlocks = 32u;
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kNoEnv = 0xffffffffu;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX incremental SAP shader missing");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{in},
                                  std::istreambuf_iterator<char>{}};
    return {bytes.begin(), bytes.end()};
}

struct Endpoint {
    std::uint32_t oldKey, newKey, tagged, rank;
};

struct Packed {
    std::vector<std::uint32_t> words;
    SapEndpointHistogramArenaDesc hist{};
    SapIncrementalComparisonArenaDesc comparison{};
    SapIncrementalPairsArenaDesc pairs{};
    std::uint32_t histDesc = 0u, comparisonDesc = 0u, pairDesc = 0u;
    std::uint32_t oldBounds = 0u, newBounds = 0u, groups = 0u, envs = 0u;
    std::uint32_t foundReport = 0u, lostReport = 0u, counters = 0u;
    std::uint32_t boxes = 0u, capacity = 0u;
    bool envEnabled = false;

    std::uint32_t reserve(std::uint32_t wordsToAdd) {
        while (words.size() % 32u) words.push_back(kUnused);
        const auto start = words.size();
        if (start + wordsToAdd > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("PhysX SAP fixture arena exceeds offsets");
        words.resize(start + wordsToAdd, kUnused);
        return static_cast<std::uint32_t>(start * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t element = 0u) {
        return words[offset / 4u + element];
    }
    std::uint32_t at(std::uint32_t offset, std::uint32_t element = 0u) const {
        return words[offset / 4u + element];
    }
    template<class T>
    void put(std::uint32_t offset, const T& value) {
        std::memcpy(words.data() + offset / 4u, &value, sizeof(value));
    }
};

std::uint32_t lo(const Packed& p, bool newer, std::uint32_t h, std::uint32_t a) {
    return p.at(newer ? p.newBounds : p.oldBounds, h * 6u + a);
}
std::uint32_t hi(const Packed& p, bool newer, std::uint32_t h, std::uint32_t a) {
    return p.at(newer ? p.newBounds : p.oldBounds, h * 6u + 3u + a);
}
bool overlaps(const Packed& p, bool newer, std::uint32_t a, std::uint32_t b) {
    for (std::uint32_t axis = 0; axis < 3u; ++axis)
        if (lo(p,newer,b,axis) > hi(p,newer,a,axis) ||
            lo(p,newer,a,axis) > hi(p,newer,b,axis)) return false;
    return true;
}
bool allowed(const Packed& p, std::uint32_t a, std::uint32_t b) {
    if (p.at(p.groups,a) == p.at(p.groups,b)) return false;
    if (!p.envEnabled) return true;
    const auto x = p.at(p.envs,a), y = p.at(p.envs,b);
    return x == y || x == kNoEnv || y == kNoEnv;
}

Packed prepare(std::uint32_t boxes, std::uint32_t capacity, bool envEnabled) {
    Packed p; p.boxes=boxes; p.capacity=capacity; p.envEnabled=envEnabled;
    p.histDesc=p.reserve(sizeof(p.hist)/4u);
    p.comparisonDesc=p.reserve(sizeof(p.comparison)/4u);
    p.pairDesc=p.reserve(sizeof(p.pairs)/4u);
    const auto projections = boxes*2u, size = std::max(4u,projections);
    p.oldBounds=p.reserve(std::max(1u,boxes)*6u);
    p.newBounds=p.reserve(std::max(1u,boxes)*6u);
    p.groups=p.reserve(std::max(1u,boxes));
    p.envs=p.reserve(std::max(1u,boxes));

    for (std::uint32_t h=0u; h<boxes; ++h) {
        p.at(p.groups,h) = h%7u;
        p.at(p.envs,h) = (h%5u==0u) ? kNoEnv : h%3u;
        for (std::uint32_t axis=0u; axis<3u; ++axis) {
            // PhysX keys have even minima and odd maxima; movement includes
            // unchanged, expansion of overlap, and loss of prior overlap.
            const std::uint32_t base=64u+((h*37u+axis*19u+h/11u)%19u)*8u;
            const std::int32_t movement=
                (static_cast<std::int32_t>((h*11u+axis*7u)%7u)-3)*8;
            const std::uint32_t fresh=static_cast<std::uint32_t>(
                static_cast<std::int32_t>(base)+movement);
            p.at(p.oldBounds,h*6u+axis)=base;
            p.at(p.oldBounds,h*6u+axis+3u)=base+23u;
            p.at(p.newBounds,h*6u+axis)=fresh;
            p.at(p.newBounds,h*6u+axis+3u)=fresh+23u;
        }
    }

    for (std::uint32_t axis=0u; axis<3u; ++axis) {
        std::vector<Endpoint> oldList;
        oldList.reserve(projections);
        for (std::uint32_t h=0u;h<boxes;++h) {
            oldList.push_back({lo(p,false,h,axis),lo(p,true,h,axis),
                               (h<<3u)|1u,0u});
            oldList.push_back({hi(p,false,h,axis),hi(p,true,h,axis),
                               (h<<3u),0u});
        }
        std::stable_sort(oldList.begin(),oldList.end(),
            [](const Endpoint& a,const Endpoint& b){return a.oldKey<b.oldKey;});
        for(std::uint32_t i=0u;i<projections;++i) oldList[i].rank=i;
        auto current=oldList;
        std::stable_sort(current.begin(),current.end(),
            [](const Endpoint& a,const Endpoint& b){return a.newKey<b.newKey;});

        auto& haxis=p.hist.axes[axis];
        for(std::uint32_t frame=0u;frame<2u;++frame) {
            auto& d=haxis.frame[frame];
            d.handles_offset=p.reserve(size);
            d.end_histogram_offset=p.reserve(size);
            d.start_histogram_offset=p.reserve(size);
            d.ordered_end_handles_offset=p.reserve(size);
            d.ordered_start_handles_offset=p.reserve(size);
            d.block_end_histogram_offset=p.reserve(kBlocks);
            d.block_start_histogram_offset=p.reserve(kBlocks);
            const auto& source=frame==0u ? current:oldList;
            for(std::uint32_t i=0u;i<projections;++i)
                p.at(d.handles_offset,i)=source[i].tagged;
        }
        haxis.total_end_histogram_offset=p.reserve(size);
        haxis.block_total_end_histogram_offset=p.reserve(kBlocks);

        auto& c=p.comparison.axes[axis];
        c.sorted_handles_offset=haxis.frame[0].handles_offset;
        c.projection_ranks_offset=p.reserve(size);
        c.current_start_histogram_offset=haxis.frame[0].start_histogram_offset;
        c.current_end_histogram_offset=haxis.frame[0].end_histogram_offset;
        c.previous_start_histogram_offset=haxis.frame[1].start_histogram_offset;
        c.previous_end_histogram_offset=haxis.frame[1].end_histogram_offset;
        c.comparison_histogram_offset=p.reserve(size);
        c.block_comparison_histogram_offset=p.reserve(kBlocks);
        for(std::uint32_t i=0u;i<projections;++i)
            p.at(c.projection_ranks_offset,i)=current[i].rank;

        auto& out=p.pairs.axes[axis];
        out.sorted_handles_offset=c.sorted_handles_offset;
        out.projection_ranks_offset=c.projection_ranks_offset;
        out.comparison_histogram_offset=c.comparison_histogram_offset;
        out.current_start_histogram_offset=c.current_start_histogram_offset;
        out.current_end_histogram_offset=c.current_end_histogram_offset;
        out.previous_start_histogram_offset=c.previous_start_histogram_offset;
        out.previous_end_histogram_offset=c.previous_end_histogram_offset;
        out.current_start_handles_offset=haxis.frame[0].ordered_start_handles_offset;
        out.current_end_handles_offset=haxis.frame[0].ordered_end_handles_offset;
        out.previous_start_handles_offset=haxis.frame[1].ordered_start_handles_offset;
        out.previous_end_handles_offset=haxis.frame[1].ordered_end_handles_offset;
    }
    p.comparison.total_comparisons_offset=p.reserve(3u);
    p.foundReport=p.reserve(std::max(1u,capacity)*2u);
    p.lostReport=p.reserve(std::max(1u,capacity)*2u);
    p.counters=p.reserve(2u);
    p.at(p.counters)=0u; p.at(p.counters,1u)=0u;

    p.pairs.old_integer_bounds_offset=p.oldBounds;
    p.pairs.new_integer_bounds_offset=p.newBounds;
    p.pairs.group_ids_offset=p.groups;
    p.pairs.environment_ids_offset=p.envs;
    p.pairs.found_report_offset=p.foundReport;
    p.pairs.lost_report_offset=p.lostReport;
    p.pairs.found_lost_counts_offset=p.counters;
    p.pairs.box_capacity=boxes;
    p.pairs.report_capacity=capacity;
    p.pairs.has_environment_ids=envEnabled?1u:0u;
    p.put(p.histDesc,p.hist);
    p.put(p.comparisonDesc,p.comparison);
    p.put(p.pairDesc,p.pairs);
    return p;
}

void oracle_histogram(Packed& p, std::uint32_t axis,std::uint32_t frame) {
    const auto& d=p.hist.axes[axis].frame[frame];
    const auto& a=p.hist.axes[axis];
    const std::uint32_t n=p.boxes*2u;
    const auto blocksRequired=(n+255u)/256u;
    const auto iterations=(blocksRequired+31u)/32u;
    const auto perBlock=iterations*256u;
    std::uint32_t start=0u,end=0u,allEnd=0u;
    for(std::uint32_t block=0u;block<32u;++block) {
        const auto beforeStart=start,beforeEnd=end,beforeAll=allEnd;
        const auto first=block*perBlock,last=std::min(n,first+perBlock);
        for(auto i=first;i<last;++i) {
            const auto tagged=p.at(d.handles_offset,i);
            const bool isStart=(tagged&1u)!=0u;
            p.at(d.start_histogram_offset,i)=start;
            p.at(d.end_histogram_offset,i)=end;
            if(frame==0u)p.at(a.total_end_histogram_offset,i)=allEnd;
            if(isStart) p.at(d.ordered_start_handles_offset,start++)=tagged>>3u;
            else p.at(d.ordered_end_handles_offset,end++)=tagged>>3u;
            if(!isStart) ++allEnd;
        }
        p.at(d.block_start_histogram_offset,block)=start-beforeStart;
        p.at(d.block_end_histogram_offset,block)=end-beforeEnd;
        if(frame==0u)
            p.at(a.block_total_end_histogram_offset,block)=allEnd-beforeAll;
    }
}
void oracle_comparisons(Packed& p) {
    const auto n=p.boxes*2u,required=(n+255u)/256u;
    const auto iterations=(required+31u)/32u,perBlock=iterations*256u;
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto& d=p.comparison.axes[axis];
        std::uint32_t global=0u;
        for(std::uint32_t block=0u;block<32u;++block) {
            const auto begin=block*perBlock,finish=std::min(n,begin+perBlock);
            const auto prior=global;
            for(auto i=begin;i<finish;++i) {
                const auto rank=p.at(d.projection_ranks_offset,i);
                const auto start=(p.at(d.sorted_handles_offset,i)&1u)!=0u;
                const auto down=i<=rank;
                const auto hist=down ?
                    (start?d.previous_end_histogram_offset:d.previous_start_histogram_offset):
                    (start?d.current_end_histogram_offset:d.current_start_histogram_offset);
                const auto loIndex=down?i:rank+1u;
                const auto hiIndex=down?std::min(rank,n-1u):i;
                p.at(d.comparison_histogram_offset,i)=global;
                global+=p.at(hist,hiIndex)-p.at(hist,loIndex);
            }
            p.at(d.block_comparison_histogram_offset,block)=global-prior;
        }
        p.at(p.comparison.total_comparisons_offset,axis)=global;
    }
}

using Pair=std::pair<std::uint32_t,std::uint32_t>;
std::array<std::vector<Pair>,2> expected_changes(const Packed& p) {
    std::array<std::vector<Pair>,2> result;
    for(std::uint32_t a=0u;a<p.boxes;++a)
        for(std::uint32_t b=a+1u;b<p.boxes;++b) {
            if(!allowed(p,a,b))continue;
            const bool oldOverlap=overlaps(p,false,a,b);
            const bool newOverlap=overlaps(p,true,a,b);
            if(newOverlap&&!oldOverlap)result[0].emplace_back(a,b);
            else if(oldOverlap&&!newOverlap)result[1].emplace_back(a,b);
        }
    return result;
}

void run_case(Context& context,SapEndpointHistogramArenaStage& hist,
              SapIncrementalComparisonsArenaStage& cmp,
              SapIncrementalPairsArenaStage& pairs,
              std::uint32_t boxes,std::uint32_t cap,bool envEnabled) {
    auto p=prepare(boxes,cap,envEnabled);
    // The previous-frame lists already exist on GPU at this point in the
    // real PhysX host graph. Supply their persistent histograms explicitly.
    for(std::uint32_t axis=0u;axis<3u;++axis)oracle_histogram(p,axis,1u);
    auto expected=p;
    for(std::uint32_t axis=0u;axis<3u;++axis)oracle_histogram(expected,axis,0u);
    oracle_comparisons(expected);
    const auto truth=expected_changes(p);

    const auto bytes=static_cast<std::uint32_t>(p.words.size()*4u);
    const auto gpu=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto mirror=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    context.upload(gpu,0u,p.words.data(),bytes);

    std::vector<BatchCommand> commands;
    hist.append(commands,gpu,bytes,p.histDesc,boxes,0u,0u,true);
    cmp.append(commands,gpu,bytes,p.comparisonDesc,boxes);
    pairs.append(commands,gpu,bytes,p.pairDesc,boxes,
                 p.comparison.total_comparisons_offset);
    if(commands.size()!=5u)
        throw std::runtime_error("PhysX histogram/comparison/report host sequence changed");
    for(std::uint32_t i=0u;i<5u;++i){
        const auto& dispatch=std::get<DispatchCommand>(commands[i]);
        const auto expectedY = (i < 2u || i == 4u) ? 3u : 1u;
        if(dispatch.groups.x!=(i==4u?256u:32u) ||
           dispatch.groups.y!=expectedY || dispatch.groups.z!=1u)
            throw std::runtime_error("PhysX SAP native dispatch geometry changed");
    }
    commands.emplace_back(CopyCommand{gpu,0u,mirror,0u,bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(p.words.size());
    context.download(mirror,0u,actual.data(),bytes);

    for(std::uint32_t kind=0u;kind<2u;++kind) {
        const auto counter=actual[p.counters/4u+kind];
        if(counter!=truth[kind].size())
            throw std::runtime_error("PhysX found/lost report counter disagrees with overlap diff");
        const auto offset=kind==0u?p.foundReport:p.lostReport;
        const auto used=std::min(counter,cap);
        std::vector<Pair> observed;
        for(std::uint32_t i=0u;i<used;++i) {
            const Pair pair{actual[offset/4u+i*2u],
                            actual[offset/4u+i*2u+1u]};
            if(pair.first>=pair.second)
                throw std::runtime_error("PhysX pair report not canonical");
            observed.push_back(pair);
        }
        auto known=truth[kind];
        std::sort(observed.begin(),observed.end());
        std::sort(known.begin(),known.end());
        if(cap>=counter) {
            if(observed!=known)
                throw std::runtime_error("PhysX incremental pair set differs from 3D overlap changes");
        } else {
            for(auto pair:observed) {
                auto at=std::lower_bound(known.begin(),known.end(),pair);
                if(at==known.end()||*at!=pair)
                    throw std::runtime_error("Capacity-clipped PhysX report contains a spurious pair");
                known.erase(at);
            }
        }
        // Preserve all uncommitted capacity slots (and allocate-no-report
        // sentinel when cap=0) when comparing against the full CPU oracle.
        for(std::uint32_t i=0u;i<used*2u;++i)
            expected.at(offset,i)=actual[offset/4u+i];
    }
    expected.at(p.counters)=actual[p.counters/4u];
    expected.at(p.counters,1u)=actual[p.counters/4u+1u];

    if(actual!=expected.words) {
        const auto mismatch=std::mismatch(actual.begin(),actual.end(),expected.words.begin());
        throw std::runtime_error(
            "PhysX incremental SAP changed unexpected arena word " +
            std::to_string(std::distance(actual.begin(),mismatch.first)) +
            " (boxes="+std::to_string(boxes)+", cap="+std::to_string(cap)+")");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(SapIncrementalPairsArenaStage& pairs) {
    for(const auto offset:{1u,128u,192u}) {
        std::vector<BatchCommand> ops;
        ops.emplace_back(CopyCommand{1u,0u,2u,0u,4u});
        bool rejected=false;
        try {pairs.append(ops,1u,256u,offset,3u,208u);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected||ops.size()!=1u)
            throw std::runtime_error("Invalid incremental pair descriptor queued work");
    }
    std::vector<BatchCommand> ops;
    bool rejected=false;
    try {pairs.append(ops,1u,256u,0u,0xffffffffu,208u);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected||!ops.empty())
        throw std::runtime_error("Incremental pair projection overflow accepted");
}

} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::invalid_argument(
            "Expected endpoint first/second, comparison first/second, pair shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto shader=[](const char* file) {
            ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
            result.format=ShaderFormat::Dxil;
#else
            result.format=ShaderFormat::Spirv;
#endif
            result.bytes=load_shader(file);
            return result;
        };
        SapEndpointHistogramArenaStage hist(context,shader(argv[1]),shader(argv[2]),shader(argv[6]));
        const auto& caps = context.device_info();
        const bool waveCapable = caps.supports_wave_arithmetic &&
            caps.subgroup_size >= 4u && 256u % caps.subgroup_size == 0u;
        if (hist.using_packed_wave_scan() != waveCapable)
            throw std::runtime_error("Incremental SAP endpoint stage bypassed native packed wave path");
        SapIncrementalComparisonsArenaStage cmp(context,shader(argv[3]),shader(argv[4]));
        SapIncrementalPairsArenaStage pairs(context,shader(argv[5]));
        negative_cases(pairs);
        run_case(context,hist,cmp,pairs,0u,8u,false);
        run_case(context,hist,cmp,pairs,1u,8u,false);
        run_case(context,hist,cmp,pairs,16u,128u,false);
        run_case(context,hist,cmp,pairs,129u,4096u,true);
        run_case(context,hist,cmp,pairs,257u,16384u,true);
        run_case(context,hist,cmp,pairs,257u,4u,true);
        run_case(context,hist,cmp,pairs,129u,0u,false);
        std::cout<<"PhysX incremental SAP found/lost pair oracle PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"PhysX incremental SAP pair oracle FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
