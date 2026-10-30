// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_projection_native.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
ShaderBinary load_shader(const char* path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Original aggregate projection shader missing");
    ShaderBinary binary;
#if defined(FYZZXYZZY_TEST_D3D12)
    binary.format=ShaderFormat::Dxil;
#else
    binary.format=ShaderFormat::Spirv;
#endif
    binary.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    require(!binary.bytes.empty(),"Empty aggregate projection shader");
    return binary;
}
using Aggregate=SourceAggregateGpuNativeRecord;
constexpr std::uint32_t kBounds=320u,kMembers=208u;
constexpr std::array<Aggregate,8> kAggregates{{
    {120u,0u,3u,0u},
    {121u,3u,37u,0u},
    {122u,40u,0u,0u},
    {123u,40u,1u,0u},
    {0xffffffffu,41u,2u,0u},
    {125u,44u,35u,0u},
    {126u,79u,1u,0u},
    {127u,80u,128u,0u}
}};
constexpr std::size_t boundsBytes=kBounds*6u*4u;
constexpr std::size_t aggregateBytes=sizeof(kAggregates);
constexpr std::size_t memberBytes=kMembers*4u;
constexpr std::size_t doubleProjectionBytes=kMembers*16u;
constexpr std::size_t scratchBytes=kMembers*8u;
constexpr std::size_t doubleMaskBytes=kMembers*8u;
constexpr std::size_t doubleMembersBytes=kMembers*8u;
constexpr std::size_t boundsOffset=4u;
constexpr std::size_t aggregatesOffset=boundsOffset+boundsBytes+4u;
constexpr std::size_t membersOffset=aggregatesOffset+aggregateBytes+4u;
constexpr std::size_t projectionsOffset=membersOffset+memberBytes+4u;
constexpr std::size_t handlesOffset=projectionsOffset+doubleProjectionBytes+4u;
constexpr std::size_t scratchProjectionsOffset=handlesOffset+doubleProjectionBytes+4u;
constexpr std::size_t scratchHandlesOffset=scratchProjectionsOffset+scratchBytes+4u;
constexpr std::size_t sapOffset=scratchHandlesOffset+scratchBytes+4u;
constexpr std::size_t masksOffset=sapOffset+doubleProjectionBytes+4u;
constexpr std::size_t historyOffset=masksOffset+doubleMaskBytes+4u;
constexpr std::size_t totalBytes=historyOffset+doubleMembersBytes+4u;
constexpr std::uint32_t kCanary=0xc7c7c7c7u;

template<std::size_t N>
void read(Context& context,BufferId root,std::size_t offset,
          std::array<std::uint32_t,N>& output) {
    context.download(root,offset,output.data(),sizeof(output));
}
void validate_slot(
    const std::array<std::uint32_t,kBounds*6u>& encoded_bounds,
    const std::array<std::uint32_t,kMembers>& members,
    std::uint32_t slot,
    const std::array<std::uint32_t,kMembers*4u>& projections,
    const std::array<std::uint32_t,kMembers*4u>& handles,
    const std::array<std::uint32_t,kMembers*4u>& sap,
    const std::array<std::uint32_t,kMembers*2u>& masks,
    const std::array<std::uint32_t,kMembers*2u>& history) {
    for(const auto& aggregate:kAggregates) {
        if(aggregate.root==0xffffffffu || !aggregate.size)continue;
        std::vector<std::pair<std::uint32_t,std::uint32_t>> oracle;
        oracle.reserve(aggregate.size*2u);
        for(std::uint32_t i=0;i<aggregate.size;++i) {
            const auto member=members[aggregate.first_member+i];
            oracle.emplace_back(encoded_bounds[member*6u],(i<<3u)|1u);
            oracle.emplace_back(encoded_bounds[member*6u+3u],i<<3u);
        }
        std::stable_sort(oracle.begin(),oracle.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        const auto projectionBase=slot*(2u*kMembers)+2u*aggregate.first_member;
        const auto historyBase=slot*kMembers+aggregate.first_member;
        for(std::uint32_t index=0;index<oracle.size();++index) {
            require(projections[projectionBase+index]==oracle[index].first &&
                    handles[projectionBase+index]==oracle[index].second,
                    "GPU aggregate projection radix output differs from stable source order");
            const auto local=oracle[index].second>>3u;
            const auto isStart=(oracle[index].second&1u)!=0u;
            require(sap[projectionBase+local*2u+(isStart?0u:1u)]==index,
                    "Source aggregate GPU SAP local start/end indices differ");
        }
        for(std::uint32_t i=0;i<aggregate.size;++i)
            require(history[historyBase+i]==members[aggregate.first_member+i],
                    "Source aggregate sorted member identity history differs");
        for(std::uint32_t word=0;word<(oracle.size()+31u)/32u;++word) {
            std::uint32_t expected=0u;
            for(std::uint32_t bit=0;bit<32u;++bit) {
                const auto index=word*32u+bit;
                if(index<oracle.size() && (oracle[index].second&1u))
                    expected|=1u<<bit;
            }
            require(masks[historyBase+word]==expected,
                    "Source aggregate projection start-mask ballot differs");
        }
    }
}

void verify(Context& context,ShaderBinary shader) {
    std::array<std::uint32_t,kBounds*6u> encoded_bounds{};
    for(std::uint32_t handle=0;handle<kBounds;++handle) {
        const std::uint32_t low=200u+((handle*13u)%17u)*6u;
        const std::uint32_t high=low+3u+(handle%3u)*2u;
        for(std::uint32_t axis=0;axis<3u;++axis) {
            encoded_bounds[handle*6u+axis]=low+axis*10u;
            encoded_bounds[handle*6u+axis+3u]=high+axis*10u;
        }
    }
    std::array<std::uint32_t,kMembers> members{};
    for(std::uint32_t i=0;i<members.size();++i)members[i]=i+1u;

    std::vector<std::uint8_t> initial(totalBytes,0xc7u);
    const auto root=context.create_buffer(totalBytes);
    context.upload(root,0u,initial.data(),initial.size());
    context.upload(root,boundsOffset,encoded_bounds.data(),boundsBytes);
    context.upload(root,aggregatesOffset,kAggregates.data(),aggregateBytes);
    context.upload(root,membersOffset,members.data(),memberBytes);

    CudaDeviceAddressMap addresses;
    const auto boundsAddress=addresses.register_slice(root,boundsOffset,boundsBytes);
    const auto aggregatesAddress=addresses.register_slice(
        root,aggregatesOffset,aggregateBytes);
    const auto membersAddress=addresses.register_slice(root,membersOffset,memberBytes);
    const auto projectionsAddress=addresses.register_slice(
        root,projectionsOffset,doubleProjectionBytes);
    const auto handlesAddress=addresses.register_slice(
        root,handlesOffset,doubleProjectionBytes);
    const auto scratchProjectionsAddress=addresses.register_slice(
        root,scratchProjectionsOffset,scratchBytes);
    const auto scratchHandlesAddress=addresses.register_slice(
        root,scratchHandlesOffset,scratchBytes);
    const auto sapAddress=addresses.register_slice(root,sapOffset,doubleProjectionBytes);
    const auto masksAddress=addresses.register_slice(root,masksOffset,doubleMaskBytes);
    const auto historyAddress=addresses.register_slice(
        root,historyOffset,doubleMembersBytes);

    std::array<std::uint32_t,kMembers*4u> projections{},handles{},sap{};
    std::array<std::uint32_t,kMembers*2u> masks{},history{};
    const auto download=[&] {
        read(context,root,projectionsOffset,projections);
        read(context,root,handlesOffset,handles);
        read(context,root,sapOffset,sap);
        read(context,root,masksOffset,masks);
        read(context,root,historyOffset,history);
    };
    {
        SourceSapDmaNative stream(context,addresses);
        SourceAggregateProjectionNativeStage stage(context,std::move(shader));
        SourceAggregateProjectionNativeBuffers source{
            boundsAddress,aggregatesAddress,membersAddress,
            projectionsAddress,handlesAddress,
            scratchProjectionsAddress,scratchHandlesAddress,
            sapAddress,masksAddress,historyAddress,
            kBounds,static_cast<std::uint32_t>(kAggregates.size()),
            kMembers,static_cast<std::uint32_t>(kAggregates.size()),0u};
        source.aggregate_count=0u;
        stage.append(stream,addresses,source);
        require(stream.commands().empty(),
                "Quiet source aggregate projection sort dispatched");
        source.aggregate_count=static_cast<std::uint32_t>(kAggregates.size());
        source.current_slot=2u;
        bool rejected=false;
        try {
            stage.append(stream,addresses,source);
        }catch(const std::invalid_argument&) {
            rejected=true;
        }
        require(rejected && stream.commands().empty(),
                "Invalid source aggregate projection slot admitted");
        source.current_slot=0u;
        stage.append(stream,addresses,source);
        require(stream.commands().size()==1u,
                "Original GPU aggregate projection sort split into multiple stages");
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Original aggregate projection sort introduced a new GPU queue");
        download();
        validate_slot(encoded_bounds,members,0u,
                      projections,handles,sap,masks,history);

        const auto previous_projections=projections;
        const auto previous_handles=handles;
        const auto previous_sap=sap;
        const auto previous_masks=masks;
        const auto previous_history=history;
        // Motion changes current projections; the original previous-slot
        // buffer must remain untouched until its consumer has finished.
        encoded_bounds[members[4u]*6u]=100u;
        encoded_bounds[members[4u]*6u+3u]=111u;
        encoded_bounds[members[70u]*6u]=94u;
        encoded_bounds[members[70u]*6u+3u]=109u;
        context.upload(root,boundsOffset,encoded_bounds.data(),boundsBytes);
        // Source PxgAggregate flips pointers between two independently
        // allocated slot views; it never shifts a previous-slot offset.
        source.current_slot=1u;
        source.sorted_projections+=kMembers*8u;
        source.sorted_handles+=kMembers*8u;
        source.sap_boxes+=kMembers*8u;
        source.start_masks+=kMembers*4u;
        source.bound_index_history+=kMembers*4u;
        stage.append(stream,addresses,source);
        stream.wait_at_fetch();
        download();
        validate_slot(encoded_bounds,members,1u,
                      projections,handles,sap,masks,history);
        for(std::uint32_t i=0;i<2u*kMembers;++i) {
            require(projections[i]==previous_projections[i] &&
                    handles[i]==previous_handles[i] &&
                    sap[i]==previous_sap[i],
                    "Original aggregate previous projection history was overwritten");
        }
        for(std::uint32_t i=0;i<kMembers;++i)
            require(masks[i]==previous_masks[i] &&
                    history[i]==previous_history[i],
                    "Original aggregate previous member/mask history was overwritten");
        require(stream.submission_stats().command_batches==2u,
                "Original aggregate projection histories split their source fetch");
        // A third GPU frame reuses the first history slot after the second
        // frame has consumed it. The large aggregate crosses eight 32-lane
        // chunks, and equal-key stable scatter must not overwrite slot one.
        const auto second_projections=projections;
        const auto second_handles=handles;
        const auto second_sap=sap;
        const auto second_masks=masks;
        const auto second_history=history;
        encoded_bounds[members[130u]*6u]=42u;
        encoded_bounds[members[130u]*6u+3u]=54u;
        context.upload(root,boundsOffset,encoded_bounds.data(),boundsBytes);
        source.current_slot=0u;
        source.sorted_projections-=kMembers*8u;
        source.sorted_handles-=kMembers*8u;
        source.sap_boxes-=kMembers*8u;
        source.start_masks-=kMembers*4u;
        source.bound_index_history-=kMembers*4u;
        stage.append(stream,addresses,source);
        stream.wait_at_fetch();
        download();
        validate_slot(encoded_bounds,members,0u,
                      projections,handles,sap,masks,history);
        for(std::uint32_t i=2u*kMembers;i<4u*kMembers;++i)
            require(projections[i]==second_projections[i] &&
                    handles[i]==second_handles[i] &&
                    sap[i]==second_sap[i],
                    "Aggregate radix third frame overwrote previous projection slot");
        for(std::uint32_t i=kMembers;i<2u*kMembers;++i)
            require(masks[i]==second_masks[i] &&
                    history[i]==second_history[i],
                    "Aggregate radix third frame overwrote previous mask/member history");
        require(stream.submission_stats().command_batches==3u,
                "Aggregate third-frame radix history added an unnecessary submission");
        // Invalid and unadmitted source aggregate records cannot write a
        // projection/history slot through the shared root allocation.
        require(projections[2u*41u]==kCanary,
                "Retired aggregate wrote into native projection history");

        // Opt-in device measurement uses the exact production dispatch
        // and source buffers. Validation completes before benchmarking;
        // no timing or extra command work enters ordinary conformance.
        if(const auto* profile=std::getenv("FYZZXYZZY_PROFILE_AGGREGATE_SORT");
           profile && std::string_view(profile)=="1") {
            std::vector<BatchCommand> commands;
            stage.append(commands,addresses,source);
            require(commands.size()==1u,
                    "Aggregate projection profiling changed source dispatch count");
            constexpr std::size_t warmup=4u,trials=19u;
            for(std::size_t i=0;i<warmup;++i)
                (void)context.profile_batch_gpu_nanoseconds(commands);
            std::vector<double> gpu_samples;
            gpu_samples.reserve(trials);
            for(std::size_t i=0;i<trials;++i) {
                const auto duration=context.profile_batch_gpu_nanoseconds(commands);
                if(!duration) {
                    std::cout<<"aggregate_projection_gpu_timestamp_unavailable device="
                             <<context.device_info().name<<'\n';
                    break;
                }
                gpu_samples.push_back(*duration);
            }
            if(!gpu_samples.empty()) {
                std::sort(gpu_samples.begin(),gpu_samples.end());
                std::cout<<"aggregate_projection_gpu_ns_median="
                         <<gpu_samples[gpu_samples.size()/2u]
                         <<" gpu_ns_min="<<gpu_samples.front()
                         <<" gpu_ns_max="<<gpu_samples.back()
                         <<" samples="<<gpu_samples.size()
                         <<" aggregates="<<kAggregates.size()
                         <<" members="<<kMembers
                         <<" endpoint_radix_passes=8"
                         <<" device="<<context.device_info().name<<'\n';
            }
        }
    }
    std::vector<std::uint8_t> completed(totalBytes);
    context.download(root,0u,completed.data(),completed.size());
    for(const auto offset:{
        std::size_t{0},aggregatesOffset-1,membersOffset-1,
        projectionsOffset-1,handlesOffset-1,
        scratchProjectionsOffset-1,scratchHandlesOffset-1,
        sapOffset-1,masksOffset-1,historyOffset-1,totalBytes-1})
        require(completed[offset]==0xc7u,
                "Original aggregate projection kernel wrote outside a heap slice");

    for(const auto address:{historyAddress,masksAddress,sapAddress,
        scratchHandlesAddress,scratchProjectionsAddress,handlesAddress,
        projectionsAddress,membersAddress,aggregatesAddress,boundsAddress})
        addresses.unregister_slice(address);
    require(addresses.live_allocations()==0u,
            "Original aggregate projection device views leaked");
    context.destroy_buffer(root);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument(
            "Expected compiled original GPU aggregate projection shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load_shader(argv[1]));
        std::cout<<"Original PhysX aggregate projection GPU histories PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original PhysX aggregate projection GPU histories FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
