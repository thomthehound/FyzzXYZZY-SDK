// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_projection_native.h"
#include "physx_gpu/source_aggregate_persistent_collisions_native.h"
#include "physx_gpu/source_aggregate_self_collision_native.h"
#include "physx_gpu/source_aggregate_pair_retirement_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
using Pair=std::pair<std::uint32_t,std::uint32_t>;
static_assert(sizeof(Pair)==8u);
constexpr std::uint32_t kBounds=7u,kAggregates=2u,kMembers=4u,kPairs=4u,kReports=8u;
void require(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
ShaderBinary load(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Source aggregate GPU collision shader missing");
    ShaderBinary s;
#if defined(FYZZXYZZY_TEST_D3D12)
    s.format=ShaderFormat::Dxil;
#else
    s.format=ShaderFormat::Spirv;
#endif
    s.bytes.assign(std::istreambuf_iterator<char>{in},
                   std::istreambuf_iterator<char>{});
    require(!s.bytes.empty(),"Source aggregate GPU collision shader empty");
    return s;
}
void verify(Context& context,ShaderBinary projection,ShaderBinary collision,
            ShaderBinary self,ShaderBinary histogram,
            ShaderBinary mark,ShaderBinary publish) {
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourceAggregateProjectionNativeStage projection_stage(
        context,std::move(projection));
    SourceAggregatePersistentCollisionNativeStage pair_stage(
        context,std::move(collision));
    SourceAggregateSelfCollisionNativeStage self_stage(
        context,std::move(self));
    SourceAggregatePairRetirementNativeStage retirement_stage(
        context,std::move(histogram),std::move(mark),std::move(publish));
    struct Owned {BufferId buffer;CudaDeviceAddressMap::Address address;};
    std::vector<Owned> allocated;
    const auto alloc=[&](std::size_t size) {
        const auto gpu=context.create_buffer(size);
        const auto addr=addresses.register_slice(gpu,0u,size);
        allocated.push_back({gpu,addr});
        return addr;
    };
    const auto put=[&](CudaDeviceAddressMap::Address addr,const void* data,std::size_t bytes) {
        const auto view=addresses.resolve(addr,bytes);
        context.upload(view.buffer,view.offset,data,bytes);
    };
    const auto get=[&](CudaDeviceAddressMap::Address addr,void* data,std::size_t bytes) {
        const auto view=addresses.resolve(addr,bytes);
        context.download(view.buffer,view.offset,data,bytes);
    };
    const auto current_bounds=alloc(kBounds*24u);
    const auto old_bounds=alloc(kBounds*24u);
    const auto volume_data=alloc(kBounds*sizeof(physx::Bp::VolumeData));
    const auto aggregates=alloc(kAggregates*16u);
    const auto members=alloc(kMembers*4u);
    const auto groups=alloc(kBounds*4u);
    const auto previous_sizes=alloc(kAggregates*4u);
    const auto scratch_keys=alloc(kMembers*8u);
    const auto scratch_handles=alloc(kMembers*8u);
    struct History {
        CudaDeviceAddressMap::Address keys,handles,sap,masks,members;
    };
    std::array<History,2> history{};
    for(auto& h:history)
        h={alloc(kMembers*8u),alloc(kMembers*8u),alloc(kMembers*8u),
           alloc(kMembers*4u),alloc(kMembers*4u)};
    const auto pair_slots=alloc(kPairs*12u);
    const auto pair_count=alloc(4u);
    const auto removed_roots=alloc(4u);
    const auto remove_pair_bitmap=alloc(kPairs*4u);
    const auto removed_histogram=alloc(4u);
    const auto free_indices=alloc(kPairs*4u);
    const auto free_list=alloc(8u);
    const auto removed_count=alloc(4u);
    const auto found=alloc(kReports*8u);
    const auto lost=alloc(kReports*8u);
    const auto reports=alloc(8u);

    const std::array<SourceAggregateGpuNativeRecord,kAggregates> slots{{
        {0u,0u,2u,0u},{1u,2u,2u,0u}
    }};
    const std::array<std::uint32_t,kMembers> source_members{{2u,3u,4u,5u}};
    const std::array<std::uint32_t,kBounds> filter_groups{{
        10u,11u,20u,21u,30u,31u,40u
    }};
    std::array<physx::Bp::VolumeData,kBounds> volumes{};
    for(auto& volume:volumes)volume.reset();
    volumes[0].setAggregate(0u);
    volumes[1].setAggregate(1u);
    volumes[2].setAggregated(0u);
    volumes[3].setAggregated(0u);
    volumes[4].setAggregated(1u);
    volumes[5].setAggregated(1u);
    const std::array<physx::PxgAggregatePair,2> root_pairs{{
        {0u,6u,true,false,0u},
        {0u,1u,true,false,0u}
    }};
    const std::uint32_t initial_pair_count=2u;
    const std::array<std::uint32_t,2> initial_list{{0u,2u}};
    const std::array<std::uint32_t,kAggregates> initial_sizes{{0u,0u}};
    const std::uint32_t zero=0u;
    put(aggregates,slots.data(),sizeof(slots));
    put(members,source_members.data(),sizeof(source_members));
    put(groups,filter_groups.data(),sizeof(filter_groups));
    put(volume_data,volumes.data(),sizeof(volumes));
    put(previous_sizes,initial_sizes.data(),sizeof(initial_sizes));
    put(pair_slots,root_pairs.data(),sizeof(root_pairs));
    put(pair_count,&initial_pair_count,4u);
    put(free_list,initial_list.data(),sizeof(initial_list));
    put(removed_roots,&zero,4u);

    // Original encoded integer AABBs: x changes across the second
    // source frame; Y/Z retain their full overlap.
    std::array<std::uint32_t,kBounds*6u> current{},previous{};
    const auto assign=[&](std::uint32_t id,std::uint32_t xmin,std::uint32_t xmax) {
        for(std::uint32_t axis=0;axis<3u;++axis) {
            current[id*6u+axis]=axis==0u?xmin:100u;
            current[id*6u+axis+3u]=axis==0u?xmax:150u;
        }
    };
    for(std::uint32_t id=0;id<kBounds;++id)
        for(std::uint32_t axis=0;axis<3u;++axis) {
            previous[id*6u+axis]=0xff7fffffu;
            previous[id*6u+axis+3u]=0x00800000u;
        }
    assign(0u,100u,290u);
    assign(1u,150u,300u);
    assign(2u,100u,180u);
    assign(3u,230u,280u);
    assign(4u,150u,205u);
    assign(5u,240u,280u);
    assign(6u,110u,190u);

    auto invalid=SourceAggregatePersistentCollisionNativeBuffers{};
    invalid.pair_capacity=kPairs;
    bool rejected=false;
    try {pair_stage.append(stream,addresses,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected && stream.commands().empty(),
            "Source persistent GPU pair collision admitted invalid memory ownership");

    for(std::uint32_t frame=0;frame<2u;++frame) {
        if(frame!=0u) {
            previous=current;
            assign(1u,310u,410u);
            assign(4u,310u,350u);
            assign(5u,360u,400u);
            assign(6u,330u,360u);
        }
        put(current_bounds,current.data(),sizeof(current));
        put(old_bounds,previous.data(),sizeof(previous));
        const std::array<std::uint32_t,2> zero_reports{};
        put(reports,zero_reports.data(),sizeof(zero_reports));
        const History& now=history[frame];
        const History& old=history[frame^1u];
        SourceAggregateProjectionNativeBuffers p{
            current_bounds,aggregates,members,now.keys,now.handles,
            scratch_keys,scratch_handles,now.sap,now.masks,now.members,
            kBounds,kAggregates,kMembers,kAggregates,frame};
        projection_stage.append(stream,addresses,p);

        SourceAggregatePersistentCollisionNativeBuffers collision_buffers{
            current_bounds,old_bounds,volume_data,aggregates,
            now.keys,now.handles,now.members,now.masks,
            old.keys,old.handles,old.members,old.masks,
            previous_sizes,groups,pair_slots,pair_count,
            removed_roots,remove_pair_bitmap,found,lost,reports,
            kPairs,kAggregates,kMembers,kBounds,kReports};
        pair_stage.append(stream,addresses,collision_buffers);
        SourceAggregateSelfCollisionNativeBuffers self_buffers{
            current_bounds,old_bounds,aggregates,
            now.handles,now.sap,now.members,now.masks,
            old.handles,old.sap,old.members,old.masks,
            groups,previous_sizes,found,lost,reports,
            kAggregates,kAggregates,kMembers,kBounds,kReports};
        self_stage.append(stream,addresses,self_buffers);
        if(frame!=0u) {
            SourceAggregatePairRetirementNativeBuffers retire{
                pair_slots,remove_pair_bitmap,removed_histogram,
                free_indices,free_list,pair_count,removed_count,kPairs};
            retirement_stage.append(stream,addresses,retire);
            require(stream.commands().size()==6u,
                    "Original persistent aggregate collision/retirement split GPU order");
        }else
            require(stream.commands().size()==3u,
                    "Original persistent pair collision/self-history GPU order changed");
        stream.wait_at_fetch();

        std::array<std::uint32_t,2> counters{};
        get(reports,counters.data(),sizeof(counters));
        require(counters==(frame==0u
                              ?std::array<std::uint32_t,2>{{3u,0u}}
                              :std::array<std::uint32_t,2>{{0u,3u}}),
                "Original persistent aggregate GPU found/lost member history differs");
        std::vector<Pair> result(counters[frame==0u?0u:1u]);
        if(!result.empty())
            get(frame==0u?found:lost,result.data(),result.size()*sizeof(Pair));
        std::sort(result.begin(),result.end());
        const std::vector<Pair> expected{{2u,4u},{2u,6u},{3u,5u}};
        require(result==expected,
                "Original aggregate–actor/aggregate–aggregate GPU member pairs differ");

        std::array<physx::PxgAggregatePair,2u> slot_state{};
        get(pair_slots,slot_state.data(),sizeof(slot_state));
        require(!slot_state[0].isNew && !slot_state[1].isNew &&
                (slot_state[0].isDead==bool(frame)) &&
                (slot_state[1].isDead==bool(frame)),
                "Original persistent root pair new/dead state differs");
        std::array<std::uint32_t,2u> old_sizes{};
        get(previous_sizes,old_sizes.data(),sizeof(old_sizes));
        require(old_sizes==std::array<std::uint32_t,2u>{{2u,2u}},
                "Original GPU source aggregate previous size not advanced at fetch");
    }

    std::array<std::uint32_t,2> final_list{};
    std::array<std::uint32_t,2> retired{};
    get(free_list,final_list.data(),sizeof(final_list));
    get(free_indices,retired.data(),sizeof(retired));
    require(final_list==std::array<std::uint32_t,2>{{2u,2u}} &&
            retired==std::array<std::uint32_t,2>{{0u,1u}},
            "Source aggregate root retirement failed to return both persistent GPU slots");
    require(stream.submission_stats().command_batches==2u,
            "Original persistent aggregate collisions added a second GPU submission");
    for(auto it=allocated.rbegin();it!=allocated.rend();++it) {
        addresses.unregister_slice(it->address);
        context.destroy_buffer(it->buffer);
    }
    require(addresses.live_allocations()==0u,
            "Source aggregate persistent GPU pair views leaked");
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::invalid_argument(
            "Expected projection, persistent collision, self and three retirement shaders");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load(argv[1]),load(argv[2]),load(argv[3]),
            load(argv[4]),load(argv[5]),load(argv[6]));
        std::cout<<"Original GPU aggregate persistent member collisions PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original GPU aggregate persistent member collisions FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
