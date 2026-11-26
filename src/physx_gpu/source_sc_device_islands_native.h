// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Transient view of stable Sc contact-manager and constraint identities.
// Only responding solver nodes have a dense ordinal. Static and
// kinematic endpoints use invalid_node and cannot bridge two islands.
using SourceScDeviceIslandEdge=native::GpuDyIslandEdge;
using SourceScDeviceIslandSources=native::GpuDyIslandSources;
static_assert(sizeof(SourceScDeviceIslandEdge)==20u);
// A portable two-dimensional indirect launch never exceeds the 32-bit
// task ordinal range, even if a backend advertises larger grid axes.
inline constexpr std::uint32_t source_sc_indirect_axis_limit=65535u;
inline std::array<std::uint32_t,2> source_sc_indirect_group_limits(
        const compute::DeviceInfo& device) noexcept {
    return {
        std::min(device.max_workgroup_count[0],source_sc_indirect_axis_limit),
        std::min(device.max_workgroup_count[1],source_sc_indirect_axis_limit)};
}
inline std::uint32_t source_sc_indirect_task_limit(
        const compute::DeviceInfo& device) noexcept {
    const auto groups=source_sc_indirect_group_limits(device);
    return static_cast<std::uint32_t>(std::uint64_t(groups[0])*groups[1]);
}
// Typed numerical table ordinals are distinct from graph-edge identities.
// Every source index is validated against its own resident Dy table.
inline constexpr std::uint32_t source_sc_device_island_joint_bit=1u;
inline constexpr std::uint32_t source_sc_device_island_no_response_bit=2u;
inline constexpr std::uint32_t source_sc_device_island_reverse_contact_bit=4u;
// Direct joint tokens require device-authored 1D rows; other joints
// use a source-indexed full Dy descriptor.
inline constexpr std::uint32_t source_sc_device_island_device_joint_bit=16u;
// A connected active kinematic vetoes deactivation without merging nodes.
inline constexpr std::uint32_t source_sc_device_island_kinematic_awake_bit=32u;

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
// touching Gu contact also contributes an activation reason from the
// journaled prior touch snapshot. GPU connectivity propagates both
// through the accurate component without a CPU witness join.
inline constexpr std::uint32_t source_sc_device_island_active_bit=1u<<16u;
inline constexpr std::uint32_t source_sc_device_island_nonresponding_bit=1u<<17u;
inline constexpr std::uint32_t source_sc_device_island_contact_task_bit=1u<<18u;
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
// The device prefix writes the numerical solver's task record directly.
// The root-to-task index remains in SourceScDeviceIslandCount.
using SourceScDeviceIslandTask=native::GpuDyDistanceTask;
static_assert(sizeof(SourceScDeviceIslandTask)==36u);
struct SourceScDeviceIslandHeader {
    std::uint32_t errors=0,runnable_components=0;
    std::uint32_t responding_nodes=0,admitted_edges=0;
    // This triple is directly consumable as DispatchIndirectArgs at +16.
    // Device admission uses both dispatch axes for large compact task
    // sets, retaining device-authored tasks without host re-admission.
    std::uint32_t dispatch_x=0,dispatch_y=1,dispatch_z=1;
    // Number of source admissions, including dormant and nonresponding edges.
    std::uint32_t source_edges=0u;
    // Sc owns the decision to pack all one-body/one-constraint tasks.
    std::uint32_t has_non_singleton_task=0u;
    std::uint32_t connectivity_reused=0u;
    std::uint32_t connectivity_dirty=0u,ordering_reused=0u;
};
static_assert(sizeof(SourceScDeviceIslandHeader)==48u &&
              offsetof(SourceScDeviceIslandHeader,dispatch_x)==16u &&
              offsetof(SourceScDeviceIslandHeader,has_non_singleton_task)==32u);

// Canonical roots and source admission records survive committed frames.
// Counts, ordering, and typed numerical tokens are rebuilt for each frame.
// Body-remap storage serves as union-parent scratch before Dy remapping.
struct SourceScDeviceIslandLayout {
    std::size_t roots=sizeof(SourceScDeviceIslandHeader);
    std::size_t counts=0u,admissions=0u,tasks=0u;
    std::size_t block_offsets=0u,block_tiles=0u;
    std::size_t response_primary=0u,response_scratch=0u;
    std::size_t response_order=0u,body_order=0u;
    std::size_t contact_endpoints=0u;
    std::size_t response_tokens=0u,response_extents=0u;
    std::size_t response_prefixes=0u,contact_admissions=0u;
    std::size_t body_remap=0u,joint_remap=0u;
    bool device_contact_prefix=false;
    std::size_t radix_histogram=0u,bytes=0u;
    static std::uint32_t radix_passes(std::uint32_t nodes) noexcept {
        // Body, joint and contact identities share root-ordered
        // keys, with 1D constraints preceding contact managers.
        const auto key=std::uint64_t(nodes)*3u+2u;
        return key<=0xffu?1u:key<=0xffffu?2u:
               key<=0xffffffu?3u:4u;
    }
    static SourceScDeviceIslandLayout for_counts(
        std::uint32_t node_count,std::uint32_t edge_count,
        std::uint32_t contact_manager_count) {
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
        const auto block_end=align(
            std::uint64_t(result.block_offsets)+block_count*16u);
        if(block_count>128u) {
            result.block_tiles=static_cast<std::size_t>(block_end);
            result.response_primary=static_cast<std::size_t>(
                align(block_end+((block_count+127u)/128u)*16u));
        } else {
            result.response_primary=static_cast<std::size_t>(block_end);
        }
        const bool parallel=node_count>256u || edge_count>256u;
        const auto passes=radix_passes(node_count);
        // The primary range also holds speculative wake bits. Only
        // intermediate radix passes need complete index streams.
        auto bytes=align(std::uint64_t(result.response_primary)+
                         (parallel && passes>2u?combined:node_count)*4u);
        if(parallel) {
            if(passes>1u) {
                result.response_scratch=static_cast<std::size_t>(bytes);
                bytes=align(bytes+combined*4u);
            }
            result.radix_histogram=static_cast<std::size_t>(bytes);
            bytes=align(bytes+((combined+127u)/128u)*256u*4u);
        }
        result.response_order=static_cast<std::size_t>(align(bytes));
        result.body_order=static_cast<std::size_t>(
            align(std::uint64_t(result.response_order)+
                  std::uint64_t(edge_count)*4u));
        result.contact_endpoints=static_cast<std::size_t>(
            align(std::uint64_t(result.body_order)+
                  std::uint64_t(node_count)*4u));
        result.response_tokens=static_cast<std::size_t>(
            align(std::uint64_t(result.contact_endpoints)+
                  std::uint64_t(contact_manager_count)*8u));
        result.response_extents=static_cast<std::size_t>(
            align(std::uint64_t(result.response_tokens)+
                  std::uint64_t(edge_count)*4u));
        result.response_prefixes=static_cast<std::size_t>(
            align(std::uint64_t(result.response_extents)+
                  std::uint64_t(edge_count)*4u));
        result.contact_admissions=static_cast<std::size_t>(
            align(std::uint64_t(result.response_prefixes)+
                  std::uint64_t(edge_count)*4u));
        result.device_contact_prefix=true;
        bytes=align(std::uint64_t(result.contact_admissions)+
                    ((std::uint64_t(contact_manager_count)+31u)/32u)*4u);
        result.body_remap=static_cast<std::size_t>(bytes);
        result.joint_remap=static_cast<std::size_t>(
            align(bytes+std::uint64_t(node_count)*4u));
        bytes=align(std::uint64_t(result.joint_remap)+
                    std::uint64_t(edge_count)*4u);
        if(bytes>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Device Sc island work exceeds 32-bit view");
        result.bytes=static_cast<std::size_t>(bytes);
        return result;
    }
};

// The device source stream is an ordered producer for Dy's transient
// task, body-ordinal, and typed descriptor arrays. The GPU header is
// validated on the numerical queue before any solver reads these arrays.
struct SourceScDeviceIslandDyView {
    CudaDeviceAddressMap::Address work=0u;
    SourceScDeviceIslandLayout layout{};
    compute::DispatchIndirectArgs task_dispatch{};
    std::uint32_t response_capacity=0u;
    std::uint32_t contact_manager_capacity=0u;
    bool deferred_body_remap=false;
    bool deferred_joint_remap=false;
    std::uint32_t joint_source_count=0u;
    std::span<const std::uint32_t> source_to_solver_row{};
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
           !info.max_workgroup_count[1] ||
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
        pipeline.push_constant_bytes=128u;
        pipeline_=context_.create_compute_pipeline(pipeline);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,"sc.device_islands");
#endif
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
                std::uint32_t edge_capacity,
                SourceScDeviceIslandSources sources,
                bool retain_roots=false) const {
        // Even a frame without responding nodes must clear the device
        // runnable-task header. Reused output storage may contain a
        // preceding frame's nonzero task admission.
        const auto layout=SourceScDeviceIslandLayout::for_counts(
            node_count,edge_count,sources.contact_managers);
        if(!output || (node_count && !node_flags) ||
           (edge_count && !edges) ||
           (edge_capacity && !touch_history))
            throw std::invalid_argument("Missing device Sc island resource");
        if(layout.bytes>max_view_)
            throw std::out_of_range("Device Sc island output view unavailable");
        // Unused source bindings reuse a real read-only input whenever
        // an edge, node, or contact-history allocation exists.
        const auto edge_bytes=edge_count
            ?source_allocation_bytes(edge_count,
                sizeof(SourceScDeviceIslandEdge))
            :node_count?source_allocation_bytes(node_count,4u)
            :edge_capacity?source_allocation_bytes(edge_capacity,240u)
            :layout.bytes;
        const auto edge_address=edge_count?edges:
            node_count?node_flags:edge_capacity?touch_history:output;
        const auto node_bytes=node_count
            ?source_allocation_bytes(node_count,
                sources.speculative_wake_groups?8u:4u):edge_bytes;
        const auto node_address=node_count?node_flags:edge_address;
        const auto touch_bytes=edge_capacity
            ?source_allocation_bytes(edge_capacity,240u):edge_bytes;
        const auto touch_address=edge_capacity?touch_history:edge_address;
        std::array<std::uint32_t,32> args{};
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
        const auto limits=source_sc_indirect_group_limits(context_.device_info());
        args[14]=limits[0];
        args[29]=limits[1];
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
        args[19]=sources.contact_managers;
        args[25]=sources.joint_rows;
        args[26]=sources.full_descriptors;
        args[27]=layout.block_tiles
            ?offset(layout.block_tiles,args[3]):args[8];
        args[28]=offset(layout.response_extents,args[3]);
        args[30]=offset(layout.contact_admissions,args[3]);
        args[31]=offset(layout.response_prefixes,args[3]);
        dispatch.push_constants.resize(sizeof(args));
        // A bounded single-workgroup graph avoids the multi-dispatch
        // union, prefix and response-ordering schedule.
        if(node_count<=256u && edge_count<=256u) {
            args[12]=sources.speculative_wake_groups?39u:32u;
            args[13]=retain_roots?1u:0u;
            dispatch.groups={1u,1u,1u};
            std::memcpy(dispatch.push_constants.data(),
                        args.data(),sizeof(args));
            stream.commands().emplace_back(std::move(dispatch));
            return;
        }
        auto& ordered=stream.commands();
        const auto original_command_count=ordered.size();
        try {
        const auto launch=[&](std::uint32_t phase,std::uint32_t count) {
            if(!count)return;
            args[12]=phase;
            declare_phase_accesses(dispatch,layout,node_count,edge_count,
                                   sources.contact_managers,edge_capacity,phase,
                                   0u,retain_roots);
            for(std::uint32_t first=0u;first<count;) {
                const auto groups=std::min<std::uint64_t>(
                    max_groups,(std::uint64_t(count-first)+127u)/128u);
                args[13]=first;
                dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
                std::memcpy(dispatch.push_constants.data(),
                            args.data(),sizeof(args));
                ordered.emplace_back(dispatch);
                first+=static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    count-first,groups*128u));
            }
        };
        const auto initialize=retain_roots
            ?(sources.speculative_wake_groups?44u:43u)
            :(sources.speculative_wake_groups?42u:0u);
        launch(initialize,std::max(
            std::max(std::max(1u,node_count),retain_roots?edge_count:0u),
            static_cast<std::uint32_t>(
                (std::uint64_t(sources.contact_managers)+31u)/32u)));
        launch(1u,edge_count);
        launch(sources.speculative_wake_groups?40u:2u,
               std::max(node_count,edge_count));
        if(sources.speculative_wake_groups)launch(41u,node_count);
        // Stable task ordinals are the exclusive prefix of runnable
        // roots. The compact block scan avoids atomic task ordering,
        // and the final scatter preserves the canonical root order.
        launch(3u,std::max(std::max(1u,node_count),
                          retain_roots?edge_count:0u));
        const auto blocks=static_cast<std::uint32_t>(
            (std::uint64_t(node_count)+127u)/128u);
        launch(4u,blocks>128u?blocks:1u);
        if(blocks>128u)launch(5u,1u);
        launch(6u,node_count);
        // A single root/type sort generates both body and response spans.
        const auto combined=static_cast<std::uint32_t>(
            std::uint64_t(node_count)+edge_count);
        for(std::uint32_t pass=0u;
            pass<SourceScDeviceIslandLayout::radix_passes(node_count);
            ++pass) {
            launch(8u+pass*3u,combined);
            launch(9u+pass*3u,1u);
            launch(10u+pass*3u,combined);
        }
        // Response extents already belong to the device. Scan their
        // exact counts in bounded workgroups and retain the global
        // source-ordered row prefix for Dy's numerical submission.
        if(edge_count && sources.contact_managers!=0u) {
            if(edge_count<=128u) {
                launch(34u,edge_count);
            } else if(edge_count<=512u) {
                // One workgroup scans bounded exact numerical extents
                // without a block-prefix submission or scratch histogram.
                launch(38u,1u);
            } else {
                const auto contact_blocks=
                    (std::uint64_t(edge_count)+127u)/128u;
                const auto contact_groups=(contact_blocks+127u)/128u;
                if(!layout.radix_histogram ||
                   layout.response_order<layout.radix_histogram ||
                   (contact_blocks+contact_groups)*sizeof(std::uint32_t)>
                       layout.response_order-layout.radix_histogram)
                    throw std::out_of_range(
                        "Device Sc contact prefix has no scan workspace");
                launch(34u,edge_count);
                if(contact_blocks>1u) {
                    launch(35u,static_cast<std::uint32_t>(contact_blocks));
                    if(contact_blocks>128u)launch(36u,1u);
                    launch(37u,edge_count);
                }
            }
        }
        } catch (...) {
            ordered.erase(ordered.begin()+original_command_count,ordered.end());
            throw;
        }
    }

    // Compact Sc node ordinals become Dy solver-row indices after
    // accurate IslandSim publication, without rebuilding connectivity.
    void append_body_remap(SourceSapDmaNative& stream,
                           const CudaDeviceAddressMap& addresses,
                           CudaDeviceAddressMap::Address mapping,
                           CudaDeviceAddressMap::Address output,
                           std::uint32_t node_count,
                           std::uint32_t edge_count,
                           std::uint32_t solver_row_capacity,
                           std::uint32_t joint_row_capacity=0u,
                           std::uint32_t joint_source_count=0u,
                           std::uint32_t contact_manager_count=0u) const {
        const auto layout=SourceScDeviceIslandLayout::for_counts(
            node_count,edge_count,contact_manager_count);
        if(!mapping || !output || !node_count || !solver_row_capacity ||
           (joint_source_count && !joint_row_capacity) ||
           layout.bytes>max_view_)
            throw std::invalid_argument("Missing Sc numerical remap resource");
        std::array<std::uint32_t,32> args{};
        const auto bytes=source_allocation_bytes(
            std::uint64_t(node_count)+joint_source_count,4u);
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.buffers={
            source_native_view(addresses,mapping,bytes,
                alignment_,0u,compute::BufferAccess::Read,args[0],max_view_),
            source_native_view(addresses,mapping,bytes,
                alignment_,1u,compute::BufferAccess::Read,args[1],max_view_),
            source_native_view(addresses,mapping,bytes,
                alignment_,2u,compute::BufferAccess::Read,args[2],max_view_),
            source_native_view(addresses,output,layout.bytes,
                alignment_,3u,compute::BufferAccess::ReadWrite,
                args[3],max_view_)};
        const auto offset=[](std::size_t pos,std::uint32_t bias) {
            const auto value=std::uint64_t(pos)+bias;
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Sc body remap offset overflow");
            return static_cast<std::uint32_t>(value);
        };
        args[9]=node_count;
        args[12]=33u;
        args[19]=solver_row_capacity;
        args[20]=offset(layout.response_tokens,args[3]);
        args[22]=offset(layout.body_order,args[3]);
        args[25]=joint_row_capacity;
        args[26]=joint_source_count;
        args[29]=source_sc_indirect_group_limits(context_.device_info())[1];
        args[30]=offset(layout.contact_admissions,args[3]);
        args[31]=offset(layout.response_prefixes,args[3]);
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        if(!max_groups)
            throw std::logic_error("Device Sc remap has no compute groups");
        dispatch.push_constants.resize(sizeof(args));
        declare_phase_accesses(dispatch,layout,node_count,edge_count,
                               contact_manager_count,0u,33u,joint_source_count);
        // Contact and full-descriptor tokens are already final. Only
        // device-joint tokens can require a numerical source remap.
        const auto total=joint_source_count
            ?std::max(node_count,edge_count):node_count;
        for(std::uint64_t first=0u;first<total;) {
            const auto groups=std::min<std::uint64_t>(
                max_groups,(std::uint64_t(total)-first+127u)/128u);
            args[13]=static_cast<std::uint32_t>(first);
            dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
            std::memcpy(dispatch.push_constants.data(),
                        args.data(),sizeof(args));
            stream.commands().emplace_back(dispatch);
            first+=std::min<std::uint64_t>(
                std::uint64_t(total)-first,groups*128u);
        }
    }

private:
    static void declare_phase_accesses(
            compute::DispatchCommand& dispatch,const SourceScDeviceIslandLayout& layout,
            std::size_t nodes,std::size_t edges,std::size_t contacts,
            std::size_t edge_capacity,std::uint32_t phase,
            std::size_t joint_sources=0u,bool retain_roots=false) {
        using A=compute::BufferAccess;
        auto& accesses=dispatch.accesses;
        accesses.clear();
        const auto input=[&](std::uint32_t slot) {
            accesses.push_back({slot,dispatch.buffers[slot].accessed_extent(),A::Read});
        };
        const auto work=[&](std::size_t offset,std::size_t bytes,A access) {
            if(bytes)
                accesses.push_back({3u,{dispatch.buffers[3].accessed_extent().offset+
                                        offset,bytes},access});
        };
        const auto header=[&](A access) {
            work(0u,sizeof(SourceScDeviceIslandHeader),access);
        };
        const auto roots=[&](A access) {work(layout.roots,nodes*4u,access);};
        const auto counts=[&](A access) {
            work(layout.counts,nodes*sizeof(SourceScDeviceIslandCount),access);
        };
        const auto admissions=[&](A access) {
            work(layout.admissions,edges*sizeof(SourceScDeviceIslandAdmission),access);
        };
        const auto tasks=[&](A access) {
            work(layout.tasks,nodes*sizeof(SourceScDeviceIslandTask),access);
        };
        const auto blocks=(nodes+127u)/128u;
        const auto tiles=(blocks+127u)/128u;
        const auto block_ranges=[&](A access) {
            work(layout.block_offsets,blocks*16u,access);
        };
        const auto block_tiles=[&](A access) {
            if(layout.block_tiles)work(layout.block_tiles,tiles*16u,access);
        };
        const auto wake_bits=[&](A access) {
            work(layout.response_primary,nodes*4u,access);
        };
        const auto contact_bits=[&](A access) {
            work(layout.contact_admissions,((contacts+31u)/32u)*4u,access);
        };
        const auto keys=[&]() {
            header(A::Read);roots(A::Read);counts(A::Read);admissions(A::Read);
            if(nodes)input(2u);
        };
        const auto scatter=[&]() {
            header(A::ReadWrite);tasks(A::Read);
            if(edge_capacity)input(1u);
            work(layout.body_order,nodes*4u,A::Write);
            work(layout.response_order,edges*4u,
                 retain_roots?A::ReadWrite:A::Write);
            work(layout.response_tokens,edges*4u,A::Write);
            work(layout.response_extents,edges*4u,A::Write);
            work(layout.contact_endpoints,contacts*8u,A::Write);
            contact_bits(A::ReadWrite);
        };
        if(phase==0u || phase==42u || phase==43u || phase==44u) {
            const bool retained=phase==43u || phase==44u;
            header(retained?A::ReadWrite:A::Write);
            counts(A::Write);contact_bits(A::Write);
            work(layout.body_remap,nodes*4u,A::Write);
            if(retained) {
                roots(A::Read);admissions(A::Read);
                if(edges)input(0u);
                if(edge_capacity)input(1u);
                if(nodes)input(2u);
            }
            if(phase==42u || phase==44u)wake_bits(A::Write);
        }else if(phase==1u || phase==2u || phase==40u) {
            header(A::ReadWrite);
            work(layout.body_remap,nodes*4u,A::ReadWrite);
            if(phase!=1u)roots(A::ReadWrite);
            if(edges)input(0u);
            if(edge_capacity)input(1u);
            if(nodes)input(2u);
            if(phase!=1u) {
                counts(A::ReadWrite);admissions(A::Write);
                if(phase==40u)wake_bits(A::ReadWrite);
            }
        }else if(phase==41u) {
            header(A::ReadWrite);roots(A::Read);counts(A::ReadWrite);
            wake_bits(A::Read);input(2u);
        }else if(phase==3u || phase==6u) {
            header(phase==3u?A::ReadWrite:A::Read);
            roots(A::Read);counts(phase==3u?A::Read:A::ReadWrite);
            if(nodes)input(2u);
            block_ranges(phase==3u?A::Write:A::Read);
            if(phase==3u && retain_roots) {
                admissions(A::Read);
                work(layout.body_order,nodes*4u,A::Read);
                work(layout.response_order,edges*4u,A::Read);
            }
            if(phase==6u) {block_tiles(A::Read);tasks(A::Write);}
        }else if(phase==4u || phase==5u) {
            if(phase==4u) {
                block_ranges(A::ReadWrite);
                block_tiles(A::Write);
            }else block_tiles(A::ReadWrite);
            if(phase==5u || blocks<=128u)header(A::ReadWrite);
        }else if(phase==33u) {
            header(A::ReadWrite);input(2u);
            work(layout.body_order,nodes*4u,A::ReadWrite);
            if(joint_sources)work(layout.response_tokens,edges*4u,A::ReadWrite);
        }else if(phase>=34u && phase<=38u) {
            const auto contact_blocks=(edges+127u)/128u;
            const auto contact_groups=(contact_blocks+127u)/128u;
            header(phase==38u?A::ReadWrite:A::Read);
            if(phase==34u || phase==38u) {
                work(layout.response_extents,edges*4u,A::Read);
                work(layout.response_prefixes,edges*4u,A::Write);
                if(phase==34u)work(layout.radix_histogram,contact_blocks*4u,A::Write);
            }else if(phase==35u) {
                work(layout.radix_histogram,contact_blocks*4u,A::ReadWrite);
                work(layout.radix_histogram+contact_blocks*4u,contact_groups*4u,A::Write);
            }else if(phase==36u) {
                work(layout.radix_histogram+contact_blocks*4u,contact_groups*4u,A::ReadWrite);
            }else {
                work(layout.radix_histogram,(contact_blocks+contact_groups)*4u,A::Read);
                work(layout.response_prefixes,edges*4u,A::ReadWrite);
            }
        }else if(phase>=8u && phase<20u) {
            const auto pass=(phase-8u)/3u;
            const auto operation=(phase-8u)%3u;
            const auto source=pass&1u?layout.response_scratch:layout.response_primary;
            const auto target=pass&1u?layout.response_primary:layout.response_scratch;
            const auto items=nodes+edges;
            const auto histogram=((items+127u)/128u)*1024u;
            if(operation==1u) {
                header(A::ReadWrite);
                work(layout.radix_histogram,histogram,A::ReadWrite);
            }else {
                keys();
                if(pass!=0u)work(source,items*4u,A::Read);
                work(layout.radix_histogram,histogram,
                     operation==0u?A::Write:A::Read);
                if(operation==2u) {
                    if(pass+1u==SourceScDeviceIslandLayout::radix_passes(
                            static_cast<std::uint32_t>(nodes)))scatter();
                    else work(target,items*4u,A::Write);
                }
            }
        }else throw std::logic_error("Unknown Sc admission phase access");
    }

    compute::Context& context_;
    compute::PipelineId pipeline_=0u;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
};

} // namespace fyzz::physx_gpu
