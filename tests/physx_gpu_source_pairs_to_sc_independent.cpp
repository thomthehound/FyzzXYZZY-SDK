// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_sequence_native.h"
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_sap_fetch_core_native.h"
#include "physx_gpu/source_sap_scene_reports.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
constexpr std::uint32_t kBoxes=2u,kSlots=6u,kRegions=kBoxes*64u;
constexpr std::size_t kRootBytes=65536u;
constexpr std::uint32_t kCanary=0xcdcdcdcd;
using Address=CudaDeviceAddressMap::Address;
using Pair=physx::PxgBroadPhasePair;

void check(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
ShaderBinary shader(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Missing source SAP pipeline shader");
    std::vector<char> data{std::istreambuf_iterator<char>{f},
                           std::istreambuf_iterator<char>{}};
    ShaderBinary bin;
#if defined(FYZZXYZZY_TEST_D3D12)
    bin.format=ShaderFormat::Dxil;
#else
    bin.format=ShaderFormat::Spirv;
#endif
    bin.bytes.assign(data.begin(),data.end());
    return bin;
}
struct Root {
    BufferId buffer=0;
    std::vector<std::uint32_t> image=
        std::vector<std::uint32_t>(kRootBytes/4u,kCanary);
    std::uint32_t next=256u;
    std::uint32_t reserve(std::uint32_t bytes) {
        const auto offset=(next+127u)&~127u;
        if(!bytes||bytes>kRootBytes-offset)
            throw std::runtime_error("Original GPU source root exhausted");
        next=offset+bytes;
        return offset;
    }
    void put(std::uint32_t offset,std::uint32_t index,std::uint32_t value) {
        image.at(offset/4u+index)=value;
    }
};
void run(Context& ctx,const SourceNewBoundsSequenceNativeStage& producer,
         const SourceSapReportNativeStage& reporter,
         bool shared,bool matching_group) {
    CudaDeviceAddressMap addresses;
    std::array<Root,4> roots;
    for(auto& root:roots)
        root.buffer=ctx.create_buffer(kRootBytes,MemoryPlacement::DeviceLocal);
    const unsigned r0=0u,r1=shared?0u:1u,r2=shared?0u:2u,
                   r3=shared?0u:3u;
    const auto allocate=[&](unsigned root,std::uint32_t bytes) {
        auto& r=roots.at(root);
        const auto offset=r.reserve(bytes);
        return std::pair<Address,std::uint32_t>{
            addresses.register_slice(r.buffer,offset,bytes),offset};
    };
    const auto bind=[&](auto& field,unsigned root,std::uint32_t bytes) {
        const auto [pointer,offset]=allocate(root,bytes);
        using Type=std::remove_reference_t<decltype(field)>;
        field=reinterpret_cast<Type>(static_cast<std::uintptr_t>(pointer));
        return offset;
    };

    physx::PxgBroadPhaseDesc desc{};
    desc.numPreviousHandles=kBoxes;
    desc.numCreatedHandles=1u;
    desc.numRemovedHandles=1u;
    desc.max_found_lost_pairs=4u;
    desc.max_found_lost_agg_pairs=4u;
    desc.regionAccumTotal=1u;

    const auto [descriptor,descriptorOffset]=allocate(r0,sizeof(desc));
    const auto sorted=bind(desc.boxHandles[0][0],r1,kSlots*4u);
    const auto sap=bind(desc.boxSapBox1D[0],r1,kBoxes*8u);
    const auto regions=bind(desc.regionRange,r1,kBoxes*32u);
    const auto prefix=bind(desc.regionAccum,r3,kSlots*4u);
    const auto activeHist=bind(desc.activeRegionsHistogram,r2,64u*kSlots*4u);
    const auto startHist=bind(desc.startRegionsHistogram,r2,64u*kSlots*4u);
    const auto activeOrder=bind(desc.orderedActiveRegionHandles,r2,64u*kSlots*4u);
    const auto startOrder=bind(desc.orderedStartRegionHandles,r2,64u*kSlots*4u);
    const auto overlap=bind(desc.overlapChecksRegion,r3,kRegions*8u);
    const auto handles=bind(desc.overlapChecksHandleRegiones,r3,kRegions*8u);
    const auto blocks=bind(desc.blockOverlapChecksRegion,r3,32u*8u);
    const auto bounds=bind(desc.newIntegerBounds,r1,kBoxes*24u);
    const auto groups=bind(desc.updateData_groups,r1,kBoxes*4u);
    const auto rawFound=bind(desc.foundPairReport,r0,4u*sizeof(Pair));
    const auto rawLost=bind(desc.lostPairReport,r0,4u*sizeof(Pair));
    const auto actorFound=bind(desc.foundActorPairReport,r0,4u*sizeof(Pair));
    const auto actorLost=bind(desc.lostActorPairReport,r0,4u*sizeof(Pair));
    const auto aggFound=bind(desc.foundAggPairReport,r0,4u*sizeof(Pair));
    const auto aggLost=bind(desc.lostAggPairReport,r0,4u*sizeof(Pair));
    const auto mappedFound=bind(desc.foundPairReportMap,r0,4u*sizeof(Pair));
    const auto mappedLost=bind(desc.lostPairReportMap,r0,4u*sizeof(Pair));
    for(unsigned i=0u;i<2u;++i) {
        bind(desc.aggReportBlock[i],r0,32u*4u);
        bind(desc.actorReportBlock[i],r0,32u*4u);
    }

    std::memcpy(reinterpret_cast<std::uint8_t*>(roots[r0].image.data())+
                descriptorOffset,&desc,sizeof(desc));
    for(std::uint32_t i=0u;i<kSlots;++i)
        roots[r3].put(prefix,i,i?1u:0u);
    roots[r1].put(sorted,0u,3u);
    roots[r1].put(sorted,1u,9u);
    roots[r1].put(sorted,2u,0u);
    roots[r1].put(sorted,3u,8u);
    roots[r1].put(sap,0u,0u);
    roots[r1].put(sap,1u,2u);
    roots[r1].put(sap,2u,1u);
    roots[r1].put(sap,3u,3u);
    for(unsigned i=0;i<16u;++i)roots[r1].put(regions,i,0u);
    roots[r2].put(startHist,1u,0u);
    roots[r2].put(startHist,2u,1u);
    roots[r2].put(startOrder,0u,1u);
    roots[r1].put(groups,0u,10u);
    roots[r1].put(groups,1u,matching_group?10u:11u);
    for(unsigned handle=0u;handle<kBoxes;++handle) {
        const auto x=handle?1u:0u;
        roots[r1].put(bounds,handle*6u,x);
        roots[r1].put(bounds,handle*6u+1u,0u);
        roots[r1].put(bounds,handle*6u+2u,0u);
        roots[r1].put(bounds,handle*6u+3u,10u);
        roots[r1].put(bounds,handle*6u+4u,10u);
        roots[r1].put(bounds,handle*6u+5u,10u);
    }
    const SourceNewBoundsSequenceNativeLayout produced{
        descriptor,kBoxes,kBoxes,kSlots,kRegions};
    const SourceSapReportNativeLayout reported{descriptor,0u,0u,0u};
    std::vector<BatchCommand> commands;
    auto invalid=desc;
    invalid.lostPairReportMap=reinterpret_cast<Pair*>(0x100u);
    producer.append(commands,desc,addresses,produced);
    check(commands.size()==4u,"Source pair producer lost its four GPU stages");
    bool rejected=false;
    try{reporter.append(commands,invalid,addresses,reported);}
    catch(const std::exception&){rejected=true;}
    check(rejected&&commands.size()==4u,
          "Invalid source report partially committed the next GPU stages");
    reporter.append(commands,desc,addresses,reported);
    check(commands.size()==7u,"Source GPU producer/report sequence diverged");
    const std::array<std::uint32_t,7> geometry{32u,32u,32u,64u,32u,32u,64u};
    const std::array<std::uint32_t,7> bindings{9u,4u,4u,16u,8u,13u,7u};
    for(std::uint32_t i=0u;i<7u;++i) {
        const auto& call=std::get<DispatchCommand>(commands[i]);
        check(call.groups.x==geometry[i]&&call.buffers.size()==bindings[i],
              "Original GPU producer/report geometry changed");
    }

    for(const auto& root:roots)
        ctx.upload(root.buffer,0u,root.image.data(),kRootBytes);
    ctx.wait_submission(ctx.submit_batch(commands));

    physx::PxgBroadPhaseDesc fetched{};
    std::array<Pair,4u> found{},lost{};
    std::array<fyzz::BodyId,kBoxes> actors{7u,3u};
    std::vector<fyzz::native::Pair> scFound,scLost;
    bool published=false,rotated=false;
    const auto counts=fetch_source_sap_results_native(
        ctx,addresses,descriptor,fetched,{found.data(),found.size()},
        {lost.data(),lost.size()},false,
        [&]() noexcept {rotated=true;},nullptr,
        [&](const SourceSapFetchedCounts& results){
            lower_source_sap_actor_reports_to_sc(
                {found.data(),results.found_actors},
                {lost.data(),results.lost_actors},
                actors,scFound,scLost);
            published=true;
        });
    const auto expected=matching_group?0u:1u;
    check(published&&rotated&&counts.found_actors==expected&&
          counts.lost_actors==0u&&counts.pairs_needed==expected&&
          fetched.sharedFoundPairIndex==expected&&
          fetched.sharedFoundAggPairIndex==0u&&!counts.overflow,
          "GPU producer counters diverged at source fetch boundary");
    if(matching_group)
        check(scFound.empty()&&scLost.empty(),
              "Filtered source pair escaped Sc report admission");
    else
        check(scFound==std::vector<fyzz::native::Pair>{{3u,7u}}&&
              scLost.empty()&&found[0].mVolA==0u&&found[0].mVolB==1u,
              "Original GPU-generated overlap was not published to Sc");
    for(auto& root:roots)ctx.destroy_buffer(root.buffer);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=10)throw std::invalid_argument("Require four producer and five report shaders");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& gpu=ctx.device_info();
        if(!gpu.supports_int64_shader||!gpu.supports_int64_wave||
           !gpu.supports_wave_arithmetic||gpu.subgroup_size<4u||
           gpu.subgroup_size>128u||256u%gpu.subgroup_size!=0u||
           gpu.max_workgroup_invocations<256u||
           gpu.max_workgroup_size[0]<256u||gpu.max_workgroup_count[0]<64u) {
            std::cout<<"SKIP: original GPU SAP new-bounds wave requirements unavailable\n";
            return 77;
        }
        SourceOverlapWorkNativeStage work(ctx,shader(argv[1]));
        SourceOverlapPrefixNativeStage prefix(ctx,shader(argv[2]),shader(argv[3]));
        SourceNewBoundsPairsNativeStage pairs(ctx,shader(argv[4]));
        SourceNewBoundsSequenceNativeStage producer(work,prefix,pairs);
        SourceSapReportNativeStage reporter(
            ctx,shader(argv[5]),shader(argv[6]),shader(argv[7]),
            shader(argv[8]),shader(argv[9]),true);
        for(const bool shared:{false,true})
            for(const bool filtered:{false,true})
                run(ctx,producer,reporter,shared,filtered);
        std::cout<<"Original GPU pair producer -> reports -> Sc fetch PASS\n";
    }catch(const std::exception& e) {
        std::cerr<<"Original GPU SAP pair-to-Sc pipeline FAILED: "<<e.what()<<'\n';
        return 1;
    }
    return 0;
}
