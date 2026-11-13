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

// Source Sc contact-manager identity and GuPCM candidate destination.
struct SourcePcmBoxPairNative {
    std::uint32_t first_element;
    std::uint32_t second_element;
    std::uint32_t edge_handle;
    std::uint32_t candidate_offset;
    // The high bit requests an in-kernel reset of this Sc edge's
    // persistent Gu PCM witness cache before any refresh reads.
    std::uint32_t candidate_capacity;
};
static_assert(sizeof(SourcePcmBoxPairNative)==20u);
constexpr std::uint32_t source_pcm_reset_cache_bit=0x80000000u;
// Plane-box descriptors reuse the candidate offset and first-element
// words for the height and contact distance, since they emit no candidates.
constexpr std::uint32_t source_pcm_plane_bit=0x40000000u;
// Seed the edge's prior physical-touch state when its Gu cache resets.
constexpr std::uint32_t source_pcm_prior_touch_bit=0x20000000u;
// CPU Gu has already published this Sc ordinal into the resident PCM
// stream. Only GPU-owned pairs participate in device witness updates.
constexpr std::uint32_t source_pcm_host_gu_bit=0x10000000u;
constexpr std::uint32_t source_pcm_candidate_capacity_mask=0x0fffffffu;

// GuPCMContactBoxBox::pcmContactBoxBox requires local A/B witnesses and a
// B-local normal before the source persistent-manifold reduction.
struct SourcePcmBoxCandidateNative {
    float local_a[3];
    float local_b[3];
    float normal_b[3];
    float separation;
};
static_assert(sizeof(SourcePcmBoxCandidateNative)==40u);

struct SourcePcmBoxPairResultNative {
    std::uint32_t candidate_count;
    std::uint32_t overlapping;
    // 0 valid, 1 invalid manager/geometry/window, 2 contact-window overflow,
    // 3 overlapping box pair awaiting GJK/EPA continuation.
    std::uint32_t status;
    std::uint32_t edge_handle;
    std::uint32_t candidate_offset;
    // Source GuPersistentContactManifold output: zero for invalid or
    // empty, otherwise 1-4 indices into the original candidate window.
    std::uint32_t selected_count;
    std::uint32_t selected_indices;
    std::uint32_t regenerated;
};
static_assert(sizeof(SourcePcmBoxPairResultNative)==32u);

// Borrowed persistent Sc contact-manager slots. The scene owns their
// initialization, retirement and device storage across frames.
struct SourcePcmScEdgeCacheNative {
    float relative_position[3];
    std::uint32_t count;
    float relative_rotation[4];
    float orientation_a[4];
    float orientation_b[4];
    std::uint32_t initialized;
    // Overlap without clipped witnesses requires GJK/EPA continuation.
    std::uint32_t needs_gjk;
    // Device Sc physical-touch history, indexed by this same EdgeIndex.
    std::uint32_t last_published_touch;
    std::uint32_t prior_batch_touch;
    SourcePcmBoxCandidateNative contacts[4];
};
static_assert(sizeof(SourcePcmScEdgeCacheNative)==240u);

class SourcePcmBoxCandidatesNativeStage final {
public:
    SourcePcmBoxCandidatesNativeStage(compute::Context& context,
                                      compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source PCM box GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(std::max<std::uint64_t>(
            4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=9u;
        desc.storage_buffer_views.assign(
            9u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=60u;
        pipeline_=context_.create_compute_pipeline(desc);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,"pcm.candidates");
#endif
    }
    SourcePcmBoxCandidatesNativeStage(const SourcePcmBoxCandidatesNativeStage&)=delete;
    SourcePcmBoxCandidatesNativeStage& operator=(
        const SourcePcmBoxCandidatesNativeStage&)=delete;
    ~SourcePcmBoxCandidatesNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address dispatch_args,
                CudaDeviceAddressMap::Address output,
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t vertex_count,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity,
                bool compact_header=false) const {
        if(!pair_count)return;
        if(!shape_capacity || !candidate_capacity || !edge_capacity)
            throw std::invalid_argument("Source PCM GPU requires shapes and candidate capacity");
        const auto groups=(std::uint64_t(pair_count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source PCM GPU dispatch exceeds device limit");
        std::array<std::uint32_t,15> params{
            pair_count,0u,0u,0u,0u,0u,0u,
            shape_capacity,candidate_capacity,edge_capacity,0u,0u,0u,
            vertex_count,0xffffffffu};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,
                slot==6u?params[10]:(slot==7u?params[11]:
                                      (slot==8u?params[12]:params[slot+1u])),
                max_view_);
        };
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
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
                 4u,compute::BufferAccess::ReadWrite),
            view(edge_cache,source_allocation_bytes(edge_capacity,240u),
                 5u,compute::BufferAccess::ReadWrite),
            view(dispatch_args,12u,6u,compute::BufferAccess::ReadWrite),
            view(output,source_allocation_bytes(
                     pair_count,compact_header?104u:84u),
                 7u,compute::BufferAccess::Write),
            view(vertex_count?convex_vertices:transforms,
                 vertex_count?source_allocation_bytes(vertex_count,12u):
                              source_allocation_bytes(shape_capacity,32u),
                 8u,compute::BufferAccess::Read)};
        for(std::size_t i=0u;i<9u;++i)
            if(i!=6u && dispatch.buffers[i].buffer==dispatch.buffers[6].buffer)
                throw std::invalid_argument(
                    "Source PCM indirect arguments alias Sc data");
        if(compact_header) {
            const auto offset=std::uint64_t(params[11])+pair_count*84ull;
            if(offset>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source PCM compact header offset overflow");
            params[14]=static_cast<std::uint32_t>(offset);
        }
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
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
                CudaDeviceAddressMap::Address dispatch_args,
                CudaDeviceAddressMap::Address output,
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t vertex_count,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity,
                bool compact_header=false) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,candidates,edge_cache,dispatch_args,output,
               convex_vertices,vertex_count,pair_count,shape_capacity,
               candidate_capacity,edge_capacity,compact_header);
    }

private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
