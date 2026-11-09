// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Transient view of stable Sc contact-manager and constraint identities.
// Only responding solver nodes have a dense ordinal. Static and
// kinematic endpoints use invalid_node and cannot bridge two islands.
struct SourceScDeviceIslandEdge {
    std::uint32_t edge_index=0;
    std::uint32_t node_a=0xffffffffu,node_b=0xffffffffu;
    std::uint32_t source_index=0;
    // Bit 0 selects joint (1) or physical contact (0); bit 1 marks
    // an edge with no solver response. Bit 2 selects a touching CPU Gu
    // manager whose physical contact has already been published by Sc;
    // bit 3 marks its new physical touch. Bit 4 selects device-authored
    // joint rows rather than full source 1D descriptors. Device PCM
    // managers derive touch from persistent EdgeIndex history.
    std::uint32_t kind=0;
};
static_assert(sizeof(SourceScDeviceIslandEdge)==20u);
inline constexpr std::uint32_t source_sc_device_island_joint_bit=1u;
inline constexpr std::uint32_t source_sc_device_island_no_response_bit=2u;
inline constexpr std::uint32_t source_sc_device_island_host_touch_bit=4u;
inline constexpr std::uint32_t source_sc_device_island_host_found_bit=8u;
// Direct joint tokens require device-authored 1D rows; other joints
// use a source-indexed full Dy descriptor.
inline constexpr std::uint32_t source_sc_device_island_device_joint_bit=16u;

// Entries preserve the source input order. They are GPU-consumed Dy
// admission records, not another persistent edge or island registry.
// An invalid root means the edge has no connected responding endpoint.
// The kind bits determine whether a connected edge admits a solver row.
// The root's count record contains the compact GPU task ordinal.
struct SourceScDeviceIslandAdmission {
    std::uint32_t source_index=0,edge_index=0;
    std::uint32_t node_a=0xffffffffu,node_b=0xffffffffu;
    std::uint32_t island_root=0xffffffffu,kind=0;
};
static_assert(sizeof(SourceScDeviceIslandAdmission)==24u);

// Sc supplies authored iteration words and active-node seeds. A newly
// touching GPU PCM contact also contributes an activation reason from
// the committed prior touch snapshot. GPU connectivity propagates both
// through the accurate component without a CPU witness join.
inline constexpr std::uint32_t source_sc_device_island_active_bit=1u<<16u;
inline constexpr std::uint32_t source_sc_device_island_iteration_mask=0xffffu;
struct SourceScDeviceIslandCount {
    std::uint32_t responding_nodes=0,admitted_edges=0;
    std::uint32_t active_seeds=0u;
    std::uint32_t position_iterations=0u,velocity_iterations=0u;
    // Set only for runnable roots; dormant islands have no Dy task.
    std::uint32_t task_index=0xffffffffu;
};
static_assert(sizeof(SourceScDeviceIslandCount)==24u);

// Task admission is selected after every component has published its
// active seeds and iteration maxima. The root is a transient device
// ordinal, not a public IslandSim handle or a persistent scheduling ID.
struct SourceScDeviceIslandTask {
    std::uint32_t root=0,responding_nodes=0,admitted_edges=0;
    std::uint32_t position_iterations=0,velocity_iterations=0;
    // Exclusive prefix of the runnable components' response edges.
    // A stable device descriptor scatter can write this task's compact
    // source-ordered constraint span starting at response_begin.
    std::uint32_t response_begin=0u,body_begin=0u,reserved=0u;
};
static_assert(sizeof(SourceScDeviceIslandTask)==32u);
struct SourceScDeviceIslandHeader {
    std::uint32_t errors=0,runnable_components=0;
    std::uint32_t responding_nodes=0,admitted_edges=0;
    // This triple is directly consumable as DispatchIndirectArgs at +16.
    // The launch is capped by the API's workgroup limit; a Dy consumer
    // strides over runnable_components when the grid is saturated.
    std::uint32_t dispatch_x=0,dispatch_y=1,dispatch_z=1;
    // Number of source admissions, including dormant and nonresponding edges.
    std::uint32_t source_edges=0u;
};
static_assert(sizeof(SourceScDeviceIslandHeader)==32u &&
              offsetof(SourceScDeviceIslandHeader,dispatch_x)==16u);

// One source-owned transient allocation holds union parents (finally
// canonical island roots), per-root admission counts and source-ordered
// edge records, stable body/response index streams and typed source
// tokens. No CPU readback is required between these phases.
struct SourceScDeviceIslandLayout {
    std::size_t roots=sizeof(SourceScDeviceIslandHeader);
    std::size_t counts=0u,admissions=0u,tasks=0u;
    std::size_t block_offsets=0u,response_primary=0u,response_scratch=0u;
    std::size_t response_order=0u,body_order=0u,response_tokens=0u;
    std::size_t radix_histogram=0u,bytes=0u;
    static std::uint32_t radix_passes(std::uint32_t nodes) noexcept {
        // Body, joint and contact identities share root-ordered
        // keys, with 1D constraints preceding contact managers.
        const auto key=std::uint64_t(nodes)*3u+2u;
        return key<=0xffu?1u:key<=0xffffu?2u:
               key<=0xffffffu?3u:4u;
    }
    static SourceScDeviceIslandLayout for_counts(
        std::uint32_t node_count,std::uint32_t edge_count) {
        const auto align=[](std::uint64_t value) {
            return (value+15u)&~std::uint64_t{15u};
        };
        SourceScDeviceIslandLayout result{};
        result.counts=static_cast<std::size_t>(
            align(std::uint64_t(result.roots)+
                  std::uint64_t(node_count)*4u));
        result.admissions=static_cast<std::size_t>(
            align(std::uint64_t(result.counts)+
                  std::uint64_t(node_count)*
                      sizeof(SourceScDeviceIslandCount)));
        result.tasks=static_cast<std::size_t>(
            align(std::uint64_t(result.admissions)+
                  std::uint64_t(edge_count)*
                      sizeof(SourceScDeviceIslandAdmission)));
        const auto combined=std::uint64_t(node_count)+edge_count;
        if(combined>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Device Sc combined admission overflow");
        result.block_offsets=static_cast<std::size_t>(
            align(std::uint64_t(result.tasks)+std::uint64_t(node_count)*
                  sizeof(SourceScDeviceIslandTask)));
        const auto block_count=(std::uint64_t(node_count)+127u)/128u;
        result.response_primary=static_cast<std::size_t>(
            align(std::uint64_t(result.block_offsets)+
                  block_count*sizeof(std::uint32_t)*4u));
        auto bytes=align(std::uint64_t(result.response_primary)+combined*4u);
        if(combined>256u) {
            result.response_scratch=static_cast<std::size_t>(bytes);
            result.radix_histogram=static_cast<std::size_t>(
                align(bytes+combined*4u));
            bytes=align(std::uint64_t(result.radix_histogram)+
                        ((combined+127u)/128u)*256u*4u);
        }
        result.response_order=static_cast<std::size_t>(align(bytes));
        result.body_order=static_cast<std::size_t>(
            align(std::uint64_t(result.response_order)+
                  std::uint64_t(edge_count)*4u));
        result.response_tokens=static_cast<std::size_t>(
            align(std::uint64_t(result.body_order)+
                  std::uint64_t(node_count)*4u));
        bytes=align(std::uint64_t(result.response_tokens)+
                    std::uint64_t(edge_count)*4u);
        if(bytes>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Device Sc island work exceeds 32-bit view");
        result.bytes=static_cast<std::size_t>(bytes);
        return result;
    }
};

class SourceScDeviceIslandsNativeStage final {
public:
    SourceScDeviceIslandsNativeStage(
            compute::Context& context,compute::ShaderBinary shader)
        :context_(context) {
        const auto& info=context_.device_info();
        if(info.max_workgroup_invocations<128u ||
           info.max_workgroup_size[0]<128u ||
           !info.max_workgroup_count[0] ||
           !info.max_storage_buffer_range)
            throw std::runtime_error("Device Sc accurate islands unavailable");
        alignment_=std::max<std::uint64_t>(
            4u,info.storage_buffer_offset_alignment);
        max_view_=info.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.shader=std::move(shader);
        pipeline.storage_buffer_binding_count=4u;
        pipeline.storage_buffer_views.assign(
            4u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=112u;
        pipeline_=context_.create_compute_pipeline(pipeline);
    }
    SourceScDeviceIslandsNativeStage(
        const SourceScDeviceIslandsNativeStage&)=delete;
    SourceScDeviceIslandsNativeStage& operator=(
        const SourceScDeviceIslandsNativeStage&)=delete;
    ~SourceScDeviceIslandsNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    // Appends one ordered device-only frame-topology transaction to the
    // source queue. Sc supplies the live EdgeIndices and typed responding
    // endpoints; the actual touch comes exclusively from device history.
    // The resulting source-indexed admission stream has no atomically
    // reordered edges, including when multiple islands are discovered.
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address edges,
                CudaDeviceAddressMap::Address touch_history,
                CudaDeviceAddressMap::Address node_flags,
                CudaDeviceAddressMap::Address output,
                std::uint32_t node_count,std::uint32_t edge_count,
                std::uint32_t edge_capacity) const {
        // Even a frame without responding nodes must clear the device
        // runnable-task header. Reused output storage may contain a
        // preceding frame's nonzero task admission.
        const auto layout=SourceScDeviceIslandLayout::for_counts(
            node_count,edge_count);
        if(!output || (node_count && !node_flags) ||
           (edge_count && !edges) ||
           (edge_capacity && !touch_history))
            throw std::invalid_argument("Missing device Sc island resource");
        if(layout.bytes>max_view_)
            throw std::out_of_range("Device Sc island output view unavailable");
        const auto edge_bytes=edge_count
            ?source_allocation_bytes(edge_count,
                sizeof(SourceScDeviceIslandEdge))
            :layout.bytes;
        const auto touch_bytes=edge_capacity
            ?source_allocation_bytes(edge_capacity,240u):edge_bytes;
        const auto edge_address=edge_count?edges:output;
        const auto touch_address=edge_capacity?touch_history:edge_address;
        const auto node_bytes=node_count
            ?source_allocation_bytes(node_count,4u):layout.bytes;
        const auto node_address=node_count?node_flags:output;
        std::array<std::uint32_t,28> args{};
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.buffers={
            source_native_view(addresses,edge_address,edge_bytes,
                alignment_,0u,compute::BufferAccess::Read,args[0],max_view_),
            source_native_view(addresses,touch_address,touch_bytes,
                alignment_,1u,compute::BufferAccess::Read,args[1],max_view_),
            source_native_view(addresses,node_address,node_bytes,
                alignment_,2u,compute::BufferAccess::Read,
                args[2],max_view_),
            source_native_view(addresses,output,layout.bytes,
                alignment_,3u,compute::BufferAccess::ReadWrite,
                args[3],max_view_)};
        const auto offset=[](std::size_t value,std::uint32_t bias) {
            const auto sum=std::uint64_t(value)+bias;
            if(sum>=std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Device Sc island offset overflow");
            return static_cast<std::uint32_t>(sum);
        };
        args[4]=offset(layout.roots,args[3]);
        args[5]=offset(layout.counts,args[3]);
        args[6]=offset(layout.admissions,args[3]);
        args[7]=offset(layout.tasks,args[3]);
        args[8]=offset(layout.block_offsets,args[3]);
        args[9]=node_count;
        args[10]=edge_count;
        args[11]=edge_capacity;
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        args[14]=max_groups;
        args[15]=static_cast<std::uint32_t>(
            (std::uint64_t(node_count)+127u)/128u);
        args[16]=offset(layout.response_primary,args[3]);
        args[17]=layout.response_scratch
            ?offset(layout.response_scratch,args[3]):args[16];
        args[18]=layout.radix_histogram
            ?offset(layout.radix_histogram,args[3]):args[16];
        args[20]=offset(layout.response_tokens,args[3]);
        args[21]=SourceScDeviceIslandLayout::radix_passes(node_count);
        args[22]=offset(layout.body_order,args[3]);
        args[23]=static_cast<std::uint32_t>(
            std::uint64_t(node_count)+edge_count);
        args[24]=offset(layout.response_order,args[3]);
        dispatch.push_constants.resize(sizeof(args));
        std::vector<compute::BatchCommand> commands;
        const auto launch=[&](std::uint32_t phase,std::uint32_t count) {
            args[12]=phase;
            for(std::uint32_t first=0u;first<count;) {
                const auto groups=std::min<std::uint64_t>(
                    max_groups,(std::uint64_t(count-first)+127u)/128u);
                args[13]=first;
                dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
                std::memcpy(dispatch.push_constants.data(),
                            args.data(),sizeof(args));
                commands.emplace_back(dispatch);
                first+=static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    count-first,groups*128u));
            }
        };
        launch(0u,std::max(1u,node_count));
        launch(1u,edge_count);
        launch(2u,std::max(node_count,edge_count));
        // Stable task ordinals are the exclusive prefix of runnable
        // roots. The compact block scan avoids atomic task ordering,
        // and the final scatter preserves the canonical root order.
        launch(3u,std::max(1u,node_count));
        launch(4u,1u);
        launch(5u,node_count);
        // A single root/type sort generates both body and response spans.
        const auto combined=static_cast<std::uint32_t>(
            std::uint64_t(node_count)+edge_count);
        launch(6u,combined);
        if(combined>256u)
            for(std::uint32_t pass=0u;
                pass<SourceScDeviceIslandLayout::radix_passes(node_count);
                ++pass) {
                launch(7u+pass*3u,combined);
                launch(8u+pass*3u,1u);
                launch(9u+pass*3u,combined);
            }
        auto& ordered=stream.commands();
        ordered.insert(ordered.end(),commands.begin(),commands.end());
    }

private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0u;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
};

} // namespace fyzz::physx_gpu
