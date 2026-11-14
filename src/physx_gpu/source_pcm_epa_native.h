// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// A GuEPA continuation into existing Sc EdgeIndex and GuPCM candidate
// storage. The preceding GPU worklist supplies the indirect group count.
class SourcePcmEpaNativeStage final {
public:
    SourcePcmEpaNativeStage(compute::Context& context,
                            compute::ShaderBinary shader,
                            bool face_geometry=false,
                            bool direct_gjk=false,
                            bool staged_push_constants=false,
                            bool compact_staged_gjk=false)
        : context_(context),face_geometry_(face_geometry),
          direct_gjk_(direct_gjk),
          compact_staged_gjk_(compact_staged_gjk),
          step_parameters_(staged_push_constants || compact_staged_gjk_) {
        if(compact_staged_gjk_ && (!direct_gjk_ || !step_parameters_))
            throw std::invalid_argument(
                "Compacted staged GJK requires the explicit source pass");
        const auto& device=context.device_info();
        if(!device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source GPU EPA dispatch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=face_geometry_?10u:9u;
        desc.storage_buffer_views.assign(
            desc.storage_buffer_binding_count,
            compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=face_geometry_?76u:
            (step_parameters_?64u:60u);
        pipeline_=context_.create_compute_pipeline(desc);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,face_geometry_?"pcm.face_contacts":
            (direct_gjk_?"pcm.gjk":"pcm.epa"));
#endif
    }
    SourcePcmEpaNativeStage(const SourcePcmEpaNativeStage&)=delete;
    SourcePcmEpaNativeStage& operator=(const SourcePcmEpaNativeStage&)=delete;
    ~SourcePcmEpaNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }
#if defined(FYZZXYZZY_PROFILE_GPU)
    void set_profile_label(const char* label) const noexcept {
        context_.set_pipeline_profile_label(pipeline_,label);
    }
#endif

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                CudaDeviceAddressMap::Address output,
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t convex_vertex_count,
                std::uint32_t pair_count,std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity,
                CudaDeviceAddressMap::Address convex_surface=0u,
                std::uint32_t convex_surface_bytes=0u,
                std::uint32_t convex_faces_base=0u,
                std::uint32_t convex_indices_base=0u,
                std::uint32_t gjk_pass=0u,
                std::uint32_t compact_header_pair_count=0u,
                std::uint32_t first_pair=0u) const {
        if(!pair_count)return;
        if(compact_header_pair_count &&
           (first_pair>compact_header_pair_count ||
            pair_count>compact_header_pair_count-first_pair))
            throw std::invalid_argument(
                "Source PCM header window exceeds global pair stream");
        const auto group_count=face_geometry_
            ?(std::uint64_t(pair_count)+63u)/64u
            :direct_gjk_?(std::uint64_t(pair_count)+31u)/32u
                        :std::uint64_t(pair_count);
        if(!shape_capacity || !candidate_capacity || !edge_capacity ||
           group_count>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Source GPU PCM dispatch exceeds device capacity");
        std::array<std::uint32_t,19> params{
            pair_count,0u,0u,0u,0u,0u,0u,0u,
            shape_capacity,candidate_capacity,edge_capacity,0u,
            0u,convex_vertex_count,0u,convex_faces_base,
            convex_indices_base,convex_surface_bytes,0xffffffffu};
        if(step_parameters_)params[14]=gjk_pass;
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,
                slot==7u?params[11]:
                    (slot==8u?params[12]:
                     (slot==9u?params[14]:params[slot+1u])),
                max_view_);
        };
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={1u,1u,1u};
        dispatch.buffers={
            view(transforms,source_allocation_bytes(shape_capacity,32u),
                 0u,compute::BufferAccess::Read),
            view(geometry,source_allocation_bytes(shape_capacity,64u),
                 1u,compute::BufferAccess::Read),
            view(pairs,source_allocation_bytes(pair_count,20u),
                 2u,compute::BufferAccess::Read),
            view(results,source_allocation_bytes(pair_count,32u),
                 3u,compute::BufferAccess::ReadWrite),
            view(candidates,source_allocation_bytes(candidate_capacity,40u),
                 4u,compute::BufferAccess::Write),
            view(edge_cache,source_allocation_bytes(edge_capacity,240u),
                 5u,compute::BufferAccess::ReadWrite),
            view(work,source_allocation_bytes(
                     std::size_t(pair_count)+compact_staged_gjk_,4u),
                 6u,compute::BufferAccess::Read),
            view(output,source_allocation_bytes(1u,
                     compact_header_pair_count
                         ? (std::size_t(compact_header_pair_count)-first_pair)*84u+
                           std::size_t(compact_header_pair_count)*20u
                         : std::size_t(pair_count)*84u),
                 7u,compute::BufferAccess::Write),
            view(convex_vertex_count?convex_vertices:transforms,
                 convex_vertex_count
                     ?source_allocation_bytes(convex_vertex_count,12u)
                     :source_allocation_bytes(shape_capacity,32u),
                 8u,compute::BufferAccess::Read)};
        if(face_geometry_)
            dispatch.buffers.push_back(
                view(convex_surface_bytes?convex_surface:transforms,
                     convex_surface_bytes
                        ?convex_surface_bytes
                        :source_allocation_bytes(shape_capacity,32u),
                     9u,compute::BufferAccess::Read));
        if(face_geometry_ || (direct_gjk_ && !compact_staged_gjk_)) {
            // Non-staged GuGJK and convex face contacts use the dense
            // source pair extent; the other stages consume GPU worklists.
            dispatch.groups={
                static_cast<std::uint32_t>(group_count),1u,1u};
        }else {
            const auto arguments=addresses.resolve(dispatch_args,12u);
            if(arguments.offset%4u)
                throw std::invalid_argument("Source PCM indirect arguments misaligned");
            for(const auto& binding:dispatch.buffers)
                if(binding.buffer==arguments.buffer)
                    throw std::invalid_argument(
                        "Source PCM indirect arguments alias Sc data");
            dispatch.indirect=compute::DispatchIndirectArgs{
                arguments.buffer,arguments.offset};
        }
        const auto header_index=face_geometry_?18u:
            (step_parameters_?15u:14u);
        if(compact_header_pair_count) {
            const auto offset=std::uint64_t(params[11])+
                (std::uint64_t(compact_header_pair_count)-first_pair)*84u+
                std::uint64_t(first_pair)*20u;
            if(offset>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error(
                    "Source PCM compact header offset overflow");
            params[header_index]=static_cast<std::uint32_t>(offset);
        }else params[header_index]=0xffffffffu;
        const auto params_bytes=face_geometry_?19u*sizeof(std::uint32_t):
            (step_parameters_?16u:15u)*sizeof(std::uint32_t);
        dispatch.push_constants.resize(params_bytes);
        std::memcpy(dispatch.push_constants.data(),params.data(),params_bytes);
        commands.emplace_back(std::move(dispatch));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                CudaDeviceAddressMap::Address output,
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t convex_vertex_count,
                std::uint32_t pair_count,std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity,
                CudaDeviceAddressMap::Address convex_surface=0u,
                std::uint32_t convex_surface_bytes=0u,
                std::uint32_t convex_faces_base=0u,
                std::uint32_t convex_indices_base=0u,
                std::uint32_t gjk_pass=0u,
                std::uint32_t compact_header_pair_count=0u,
                std::uint32_t first_pair=0u) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,candidates,edge_cache,work,dispatch_args,output,
               convex_vertices,convex_vertex_count,
               pair_count,shape_capacity,candidate_capacity,edge_capacity,
               convex_surface,convex_surface_bytes,
               convex_faces_base,convex_indices_base,gjk_pass,
               compact_header_pair_count,first_pair);
    }
private:
    compute::Context& context_;
    bool face_geometry_=false;
    bool direct_gjk_=false;
    bool compact_staged_gjk_=false;
    bool step_parameters_=false;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
