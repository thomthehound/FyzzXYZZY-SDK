// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_projection_native.h"
#include "physx_gpu/source_aggregate_self_collision_native.h"

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
void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
ShaderBinary load(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Original GPU aggregate shader not found");
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(std::istreambuf_iterator<char>(in),
                        std::istreambuf_iterator<char>());
    require(!result.bytes.empty(),"Original aggregate kernel empty");
    return result;
}
using Pair=std::pair<std::uint32_t,std::uint32_t>;
static_assert(sizeof(Pair)==8u);
constexpr std::uint32_t kBounds=8,kMembers=5,kAggregates=3,kReports=8;
constexpr std::array<SourceAggregateGpuNativeRecord,kAggregates> records{{
    {6u,0u,3u,1u},
    {7u,3u,2u,0u},
    {0xffffffffu,5u,0u,0u}
}};
constexpr std::array<std::uint32_t,kMembers> members{{0u,1u,2u,3u,4u}};
constexpr std::array<std::uint32_t,kBounds> groups{{0u,1u,1u,2u,3u,0u,0u,0u}};
constexpr std::array<std::array<std::uint32_t,6>,5> frame0{{
    {{100u,100u,100u,251u,151u,151u}},
    {{151u,100u,100u,221u,151u,151u}},
    {{171u,100u,100u,231u,151u,151u}},
    {{110u,100u,100u,215u,151u,151u}},
    {{141u,100u,100u,245u,151u,151u}}
}};
struct Buffer {
    BufferId id=0;
    CudaDeviceAddressMap::Address address=0;
    std::size_t size=0;
};
void verify(Context& context,ShaderBinary projection,ShaderBinary collision) {
    CudaDeviceAddressMap addresses;
    std::vector<Buffer> allocated;
    const auto make=[&](std::size_t bytes) {
        auto id=context.create_buffer(bytes);
        auto address=addresses.register_slice(id,0,bytes);
        allocated.push_back({id,address,bytes});
        return address;
    };
    const auto transfer=[&](CudaDeviceAddressMap::Address address,
                            const void* source,std::size_t bytes) {
        const auto view=addresses.resolve(address,bytes);
        context.upload(view.buffer,view.offset,source,bytes);
    };
    const auto download=[&](CudaDeviceAddressMap::Address address,
                            void* dest,std::size_t bytes) {
        const auto view=addresses.resolve(address,bytes);
        context.download(view.buffer,view.offset,dest,bytes);
    };
    constexpr auto boundsBytes=kBounds*24u;
    constexpr auto aggregateBytes=kAggregates*16u;
    constexpr auto membersBytes=kMembers*4u;
    constexpr auto endpointsBytes=kMembers*8u;
    constexpr auto masksBytes=kMembers*4u;
    constexpr auto reportsBytes=kReports*8u;
    const auto newBounds=make(boundsBytes),oldBounds=make(boundsBytes);
    const auto aggregateRecords=make(aggregateBytes);
    const auto gpuMembers=make(membersBytes);
    const auto groupIds=make(kBounds*4u);
    struct History {
        CudaDeviceAddressMap::Address keys,handles,sap,masks,members;
    };
    const std::array<History,2> history{{
        {make(endpointsBytes),make(endpointsBytes),make(endpointsBytes),
         make(masksBytes),make(membersBytes)},
        {make(endpointsBytes),make(endpointsBytes),make(endpointsBytes),
         make(masksBytes),make(membersBytes)}
    }};
    const auto scratchKeys=make(endpointsBytes),scratchHandles=make(endpointsBytes);
    const auto previousSizes=make(kAggregates*4u);
    const auto found=make(reportsBytes),lost=make(reportsBytes);
    const auto counters=make(8u);
    transfer(aggregateRecords,records.data(),sizeof(records));
    transfer(gpuMembers,members.data(),sizeof(members));
    transfer(groupIds,groups.data(),sizeof(groups));
    const std::array<std::uint32_t,kAggregates> initialSizes{};
    transfer(previousSizes,initialSizes.data(),sizeof(initialSizes));

    SourceSapDmaNative stream(context,addresses);
    SourceAggregateProjectionNativeStage projectionStage(
        context,std::move(projection));
    SourceAggregateSelfCollisionNativeStage collisionStage(
        context,std::move(collision));

    SourceAggregateSelfCollisionNativeBuffers source{
        newBounds,oldBounds,aggregateRecords,
        history[0].handles,history[0].sap,history[0].members,
        history[0].masks,history[1].handles,history[1].sap,
        history[1].members,history[1].masks,
        groupIds,previousSizes,found,lost,counters,
        kAggregates,kAggregates,kMembers,kBounds,kReports};

    // The independent test checks source preparation rejection before
    // submitting either projection or collision GPU work.
    auto invalid=source;
    invalid.previous_handles=invalid.current_handles;
    bool rejected=false;
    try {
        collisionStage.append(stream,addresses,invalid);
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected && stream.commands().empty(),
            "Aliased original self-collision histories were admitted");
    invalid=source;
    invalid.report_capacity=0u;
    rejected=false;
    try {
        collisionStage.append(stream,addresses,invalid);
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected && stream.commands().empty(),
            "Zero-capacity original self-collision reports were admitted");

    std::array<std::uint32_t,kBounds*6u> previous{};
    std::array<std::uint32_t,kBounds*6u> current{};
    for(std::uint32_t i=0;i<kBounds;++i)
        for(std::uint32_t axis=0;axis<3u;++axis) {
            previous[6u*i+axis]=0xff7fffffu;
            previous[6u*i+axis+3u]=0x00800000u;
            current[6u*i+axis]=0xff7fffffu;
            current[6u*i+axis+3u]=0x00800000u;
        }
    const std::array<std::vector<Pair>,7> expectedFound{{
        {{0u,1u},{0u,2u}}, {}, {}, {{0u,1u},{0u,2u}},
        {}, {}, {{0u,1u},{0u,2u}}
    }};
    const std::array<std::vector<Pair>,7> expectedLost{{
        {},{{0u,1u}},{{0u,2u}}, {}, {},
        {{0u,1u},{0u,2u}}, {}
    }};
    std::uint32_t actualSlot=0;
    for(std::uint32_t frame=0;frame<7u;++frame) {
        previous=current;
        for(std::uint32_t i=0;i<frame0.size();++i)
            for(std::uint32_t axis=0;axis<6u;++axis)
                current[6u*i+axis]=frame0[i][axis];
        if(frame==1u) {
            current[6u*0u]=100u;
            current[6u*0u+3u]=181u;
            current[6u*1u]=201u;
            current[6u*1u+3u]=231u;
        }else if(frame==2u) {
            current[6u*0u]=301u;
            current[6u*0u+3u]=351u;
            current[6u*1u]=201u;
            current[6u*1u+3u]=231u;
        }else if(frame==5u) {
            current[6u*0u]=301u;
            current[6u*0u+3u]=351u;
        }
        transfer(newBounds,current.data(),sizeof(current));
        transfer(oldBounds,previous.data(),sizeof(previous));
        const std::array<std::uint32_t,2> zeros{};
        auto& uploads=stream.upload_scratch();
        uploads.push_back({counters,zeros.data(),sizeof(zeros)});
        stream.queue_uploads(uploads);

        const History& now=history[actualSlot];
        const History& old=history[actualSlot^1u];
        SourceAggregateProjectionNativeBuffers projectionSource{
            newBounds,aggregateRecords,gpuMembers,now.keys,now.handles,
            scratchKeys,scratchHandles,now.sap,now.masks,now.members,
            kBounds,kAggregates,kMembers,kAggregates,actualSlot};
        projectionStage.append(stream,addresses,projectionSource);
        source.current_handles=now.handles;
        source.current_sap=now.sap;
        source.current_member_indices=now.members;
        source.current_start_masks=now.masks;
        source.previous_handles=old.handles;
        source.previous_sap=old.sap;
        source.previous_member_indices=old.members;
        source.previous_start_masks=old.masks;
        collisionStage.append(stream,addresses,source);
        require(stream.commands().size()==2u,
                "Original aggregate projection/self-collision split source dispatch");
        stream.wait_at_fetch();

        std::array<std::uint32_t,2> counts{};
        download(counters,counts.data(),sizeof(counts));
        std::vector<Pair> actualFound(counts[0]),actualLost(counts[1]);
        require(counts[0]<=kReports && counts[1]<=kReports,
                "Unanticipated original aggregate self-collision report overflow");
        if(counts[0])download(found,actualFound.data(),counts[0]*sizeof(Pair));
        if(counts[1])download(lost,actualLost.data(),counts[1]*sizeof(Pair));
        std::sort(actualFound.begin(),actualFound.end());
        std::sort(actualLost.begin(),actualLost.end());
        require(actualFound==expectedFound[frame] &&
                actualLost==expectedLost[frame],
                "Original aggregate GPU self-collision found/lost history differs");
        std::array<std::uint32_t,kAggregates> sizes{};
        download(previousSizes,sizes.data(),sizeof(sizes));
        require(sizes==std::array<std::uint32_t,kAggregates>{{3u,2u,0u}},
                "Original aggregate self-collision previous size history differs");
        actualSlot^=1u;
    }
    // A source report overflow never silently drops the event count:
    // both lost pairs are counted even though only one fits in the
    // configured mapped report capacity.
    previous=current;
    current[0u]=301u;
    current[3u]=351u;
    transfer(newBounds,current.data(),sizeof(current));
    transfer(oldBounds,previous.data(),sizeof(previous));
    const std::array<std::uint32_t,2> zeroCounts{};
    auto& finalUploads=stream.upload_scratch();
    finalUploads.push_back({counters,zeroCounts.data(),sizeof(zeroCounts)});
    stream.queue_uploads(finalUploads);
    const History& now=history[actualSlot];
    const History& old=history[actualSlot^1u];
    SourceAggregateProjectionNativeBuffers projectionSource{
        newBounds,aggregateRecords,gpuMembers,now.keys,now.handles,
        scratchKeys,scratchHandles,now.sap,now.masks,now.members,
        kBounds,kAggregates,kMembers,kAggregates,actualSlot};
    projectionStage.append(stream,addresses,projectionSource);
    source.current_handles=now.handles;
    source.current_sap=now.sap;
    source.current_member_indices=now.members;
    source.current_start_masks=now.masks;
    source.previous_handles=old.handles;
    source.previous_sap=old.sap;
    source.previous_member_indices=old.members;
    source.previous_start_masks=old.masks;
    source.report_capacity=1u;
    collisionStage.append(stream,addresses,source);
    stream.wait_at_fetch();
    std::array<std::uint32_t,2> overflowCounts{};
    download(counters,overflowCounts.data(),sizeof(overflowCounts));
    require(overflowCounts[0]==0u && overflowCounts[1]==2u,
            "Original aggregate GPU self-collision lost pair overflow was not reported");
    require(stream.submission_stats().command_batches==8u,
            "Original aggregate pair stage introduced extra GPU command batches");
    stream.wait_at_fetch();
    for(auto iter=allocated.rbegin();iter!=allocated.rend();++iter) {
        addresses.unregister_slice(iter->address);
        context.destroy_buffer(iter->id);
    }
    require(addresses.live_allocations()==0u,
            "Original aggregate self-collision source GPU views leaked");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument(
            "Expected original aggregate projection and self-collision kernels");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load(argv[1]),load(argv[2]));
        std::cout<<"Original PhysX aggregate GPU self collision PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Original PhysX aggregate GPU self collision FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
