// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/sap_newbounds_pairs_arena.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using fyzz::physx_gpu::SapNewBoundsPairsArenaDesc;
using fyzz::physx_gpu::SapNewBoundsPairsArenaStage;
namespace {
constexpr std::uint32_t kCanary=0xcdcdcdcd;
constexpr std::uint32_t kInvalidEnv=0xffffffffu;

std::vector<std::uint8_t> load(const char* filename) {
    std::ifstream f(filename,std::ios::binary);
    if (!f) throw std::runtime_error("New-bounds shader not found");
    std::vector<char> bytes{std::istreambuf_iterator<char>{f},
                            std::istreambuf_iterator<char>{}};
    return {bytes.begin(),bytes.end()};
}
struct Case {
    std::uint32_t boxes=3u;
    std::uint32_t region=0u;
    bool canonical=true;
    bool newFlag=true;
    bool useEnv=false;
    bool filters=false;
    bool noCandidates=false;
    bool created=true;
    std::uint32_t capacity=0u;
    std::uint32_t priorCount=0u;
};
struct Fixture {
    std::vector<std::uint32_t> words;
    SapNewBoundsPairsArenaDesc desc{};
    std::vector<std::pair<std::uint32_t,std::uint32_t>> expectedPairs;
    std::uint32_t expectedCount=0u;
    std::uint32_t initialCount=0u;
    std::uint32_t reportSlots=0u;
    std::uint32_t reserve(std::uint32_t wordCount) {
        while (words.size()%16u) words.push_back(kCanary);
        const auto start=words.size();
        words.resize(start+wordCount,kCanary);
        return static_cast<std::uint32_t>(start*4u);
    }
    void put(std::uint32_t byteAddress,std::uint32_t value) {
        words.at(byteAddress/4u)=value;
    }
    std::uint32_t get(std::uint32_t byteAddress) const {
        return words.at(byteAddress/4u);
    }
    void put64(std::uint32_t byteAddress,std::uint64_t value) {
        put(byteAddress,static_cast<std::uint32_t>(value));
        put(byteAddress+4u,static_cast<std::uint32_t>(value>>32u));
    }
};
Fixture build(Case c) {
    if(c.boxes<2u)throw std::runtime_error("Fixture needs at least 2 boxes");
    Fixture f;
    const auto descOffset=f.reserve(sizeof(f.desc)/4u);
    if(descOffset!=0u)throw std::runtime_error("Descriptor moved");
    auto& d=f.desc;
    const auto n=c.boxes;
    const auto p=2u*n;
    d.current_boxes=n;d.box_capacity=n;
    d.has_environment_ids=c.useEnv?1u:0u;
    d.new_integer_bounds_offset=f.reserve(n*6u);
    d.group_ids_offset=f.reserve(n);
    d.environment_ids_offset=f.reserve(n);
    d.sorted_x_handles_offset=f.reserve(p);
    d.x_sap_positions_offset=f.reserve(n*2u);
    d.region_ranges_offset=f.reserve(n*8u);
    d.overlap_prefix_offset=f.reserve(2u);
    d.region_work_total_offset=f.reserve(1u);
    d.overlap_total_offset=f.reserve(2u);
    d.handle_regions_offset=f.reserve(2u);
    d.active_histograms_offset=f.reserve(64u*p);
    d.start_histograms_offset=f.reserve(64u*p);
    d.ordered_active_handles_offset=f.reserve(64u*p);
    d.ordered_start_handles_offset=f.reserve(64u*p);
    f.reportSlots=c.capacity?c.capacity:(n+8u+c.priorCount);
    d.max_found_lost_pairs=c.capacity?c.capacity:f.reportSlots;
    d.found_reports_offset=f.reserve(f.reportSlots*2u);
    d.found_counter_offset=f.reserve(1u);
    f.reserve(16u); // guard beyond reports/counter
    f.initialCount=c.priorCount;
    f.put(d.found_counter_offset,c.priorCount);
    f.put(d.region_work_total_offset,c.noCandidates?0u:1u);
    f.put64(d.overlap_total_offset,c.noCandidates?0ull:std::uint64_t(n-1u));
    f.put64(d.overlap_prefix_offset,0ull);
    const auto regionBase=c.region*p;
    f.put(d.handle_regions_offset,0u);
    f.put(d.handle_regions_offset+4u,regionBase);
    // Sorted starts: owner (index 0), then partners 1..N-1.
    // Endpoints: owner (index N), followed by partners.
    for (std::uint32_t h=0;h<n;++h) {
        const auto lo=h;
        const auto hi=h==0u?n:n+h;
        f.put(d.sorted_x_handles_offset+lo*4u,(h<<3u)|1u|
              ((h==0u && c.newFlag)?2u:0u));
        f.put(d.sorted_x_handles_offset+hi*4u,h<<3u);
        f.put(d.x_sap_positions_offset+h*8u,lo);
        f.put(d.x_sap_positions_offset+h*8u+4u,hi);
        f.put(d.region_ranges_offset+h*32u,
              c.canonical&&c.region==1u&&h==0u?1u:0u);
        f.put(d.region_ranges_offset+h*32u+4u,0u);
        f.put(d.region_ranges_offset+h*32u+8u,0u);
        f.put(d.group_ids_offset+h*4u,
              h!=0u&&c.filters&&h%7u==0u?10u:10u+h);
        const auto environment=h==0u?3u:(h%5u==0u?4u:3u);
        f.put(d.environment_ids_offset+h*4u,
              c.useEnv&&h%13u==0u&&h!=0u?kInvalidEnv:environment);
        const auto bb=d.new_integer_bounds_offset+h*24u;
        f.put(bb,h==0u?0u:h);             // min X
        f.put(bb+4u,h!=0u&&c.filters&&h%11u==0u?100001u:0u);
        f.put(bb+8u,0u);
        f.put(bb+12u,h==0u?100000u:h+5u); // max X
        f.put(bb+16u,h!=0u&&c.filters&&h%11u==0u?100010u:10u);
        f.put(bb+20u,10u);
    }
    const auto hist=c.newFlag?d.start_histograms_offset:d.active_histograms_offset;
    const auto order=c.newFlag?d.ordered_start_handles_offset:d.ordered_active_handles_offset;
    // Source histogram: exclusive start-prefix for region selected by owner.
    for (std::uint32_t x=0u;x<p;++x) {
        const auto prior=x==0u?0u:(x<=n?x-1u:n-1u);
        f.put(hist+(regionBase+x)*4u,prior);
    }
    for(std::uint32_t i=0u;i+1u<n;++i)
        f.put(order+(regionBase+i)*4u,i+1u);

    // CPU oracle directly tests all owner/partner pairs. This doesn't
    // replicate the shader's candidate search/compaction machinery.
    for(std::uint32_t h=1u;h<n&&!c.noCandidates;++h) {
        if(c.region!=0u&&!(c.canonical&&c.region==1u))continue;
        if(c.filters&&h%7u==0u)continue;
        if(c.filters&&h%11u==0u)continue;
        if(c.useEnv&&h%5u==0u&&h%13u!=0u)continue;
        f.expectedPairs.emplace_back(0u,h);
    }
    f.expectedCount=c.priorCount+
        static_cast<std::uint32_t>(f.expectedPairs.size());
    std::memcpy(f.words.data(),&d,sizeof(d));
    return f;
}
void run(Context& context,SapNewBoundsPairsArenaStage& stage,Case c) {
    auto f=build(c);
    const auto bytes=static_cast<std::uint32_t>(f.words.size()*4u);
    const auto gpu=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    const auto mirror=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    context.upload(gpu,0u,f.words.data(),bytes);
    std::vector<BatchCommand> ops;
    stage.append(ops,gpu,bytes,0u,c.created?1u:0u);
    if(ops.size()!=(c.created?1u:0u))
        throw std::runtime_error("New-bounds created-handles gate changed");
    if(c.created) {
        const auto& dispatch=std::get<DispatchCommand>(ops.front());
        if(dispatch.groups.x!=32u||dispatch.groups.y!=1u||
           dispatch.groups.z!=1u||dispatch.push_constants.size()!=32u)
            throw std::runtime_error("New-bounds dispatch geometry changed");
    }
    ops.emplace_back(CopyCommand{gpu,0u,mirror,0u,bytes});
    context.wait_submission(context.submit_batch(ops));
    std::vector<std::uint32_t> actual(f.words.size());
    context.download(mirror,0u,actual.data(),bytes);
    context.destroy_buffer(mirror);context.destroy_buffer(gpu);
    if(!c.created) {
        if(actual!=f.words)
            throw std::runtime_error("No-created frame mutated GPU arena");
        return;
    }
    auto expectedCount=f.expectedCount;
    if(actual.at(f.desc.found_counter_offset/4u)!=expectedCount)
        throw std::runtime_error("New-bounds found counter mismatch: expected "+
            std::to_string(expectedCount)+" observed "+
            std::to_string(actual[f.desc.found_counter_offset/4u])+
            " boxes "+std::to_string(c.boxes));
    // Reports across workgroups have intentionally unspecified order.
    // Validate exact set if uncapped, or unique subset if capacity binds.
    const auto written=std::min<std::uint32_t>(
        static_cast<std::uint32_t>(f.expectedPairs.size()),
        f.reportSlots>c.priorCount?f.reportSlots-c.priorCount:0u);
    std::set<std::pair<std::uint32_t,std::uint32_t>> expectedSet(
        f.expectedPairs.begin(),f.expectedPairs.end());
    std::set<std::pair<std::uint32_t,std::uint32_t>> observedSet;
    for(std::uint32_t j=0u;j<written;++j) {
        const auto offset=f.desc.found_reports_offset+(c.priorCount+j)*8u;
        const auto pair=std::make_pair(actual[offset/4u],actual[offset/4u+1u]);
        if(!expectedSet.count(pair)||!observedSet.insert(pair).second)
            throw std::runtime_error("Found pairs contain invalid or duplicate report");
    }
    if(observedSet.size()!=written ||
       (written==f.expectedPairs.size()&&observedSet!=expectedSet))
        throw std::runtime_error("New-bounds pair report set incorrect");

    // Everything except the counter and actual report entries is immutable.
    auto normalized=actual;
    normalized[f.desc.found_counter_offset/4u]=f.words[f.desc.found_counter_offset/4u];
    for(std::uint32_t j=0u;j<written;++j) {
        const auto word=f.desc.found_reports_offset/4u+(c.priorCount+j)*2u;
        normalized[word]=kCanary; normalized[word+1u]=kCanary;
    }
    if(normalized!=f.words) {
        const auto it=std::mismatch(normalized.begin(),normalized.end(),f.words.begin());
        throw std::runtime_error("Unexpected arena modification at word "+
            std::to_string(std::distance(normalized.begin(),it.first)));
    }
}
void negative(SapNewBoundsPairsArenaStage& stage) {
    std::vector<BatchCommand> ops;
    stage.append(ops,0u,0u,3u,0u);
    if(!ops.empty())throw std::runtime_error("Unexpected quiet-frame dispatch");
    for(auto bad : {1u,64u,96u}) {
        bool rejected=false;
        try {stage.append(ops,1u,128u,bad,1u);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected||!ops.empty())
            throw std::runtime_error("Invalid found-pair descriptor accepted");
    }
    stage.append(ops,1u,128u,32u,1u);
    if(ops.size()!=1u)throw std::runtime_error("Valid exact-fit descriptor rejected");
}
}
int main(int argc,char** argv){
    try {
        if(argc!=2)throw std::invalid_argument("Expected new-bounds wave shader");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& info=context.device_info();
        if(!info.supports_int64_shader||!info.supports_wave_arithmetic||
           info.subgroup_size<4u||info.subgroup_size>256u||
           256u%info.subgroup_size!=0u) {
            std::cout<<"SKIP: native uint64 shader or subgroup ops unavailable\n";
            return 77;
        }
        ShaderBinary binary;
#if defined(FYZZXYZZY_TEST_D3D12)
        binary.format=ShaderFormat::Dxil;
#else
        binary.format=ShaderFormat::Spirv;
#endif
        binary.bytes=load(argv[1]);
        SapNewBoundsPairsArenaStage stage(context,std::move(binary));
        negative(stage);
        for(auto n:{2u,3u,19u,301u,8203u}) {
            Case c;c.boxes=n;
            run(context,stage,c);
            c.filters=true; run(context,stage,c);
            c.useEnv=true; run(context,stage,c);
        }
        Case special;
        special.boxes=19u;special.newFlag=false;run(context,stage,special);
        special.newFlag=true;special.region=1u;special.canonical=true;
        run(context,stage,special);
        special.canonical=false;run(context,stage,special);
        special.region=0u;special.noCandidates=true;run(context,stage,special);
        special.noCandidates=false;special.created=false;run(context,stage,special);
        special.created=true;special.capacity=3u;special.priorCount=1u;
        run(context,stage,special);
        std::cout<<"PhysX native-wave new-bounds found pairs PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"PhysX new-bounds found pairs FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
