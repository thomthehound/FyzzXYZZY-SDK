// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_sc_gpu_broadphase_producer_native.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"
#include "physx_gpu/source_pcm_sparse_epa_native.h"
#include "physx_gpu/source_pcm_epa_native.h"
#include "physx_gpu/source_pcm_sc_contact_stream_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// The Sc result span is the exact output of the source GPU contact shader.
// Both representations are defined layouts, not an implicit C++ struct cast.
static_assert(sizeof(native::GpuPcmBoxResult)==
              sizeof(SourcePcmScContactStreamNative));
static_assert(offsetof(native::GpuPcmBoxResult,edge)==
              offsetof(SourcePcmScContactStreamNative,edge_handle));
static_assert(offsetof(native::GpuPcmBoxResult,cached_count)==
              offsetof(SourcePcmScContactStreamNative,cached_count));
static_assert(offsetof(native::GpuPcmBoxResult,status)==
              offsetof(SourcePcmScContactStreamNative,status));
static_assert(offsetof(native::GpuPcmBoxResult,selected_and_regenerated)==
              offsetof(SourcePcmScContactStreamNative,selected_and_regenerated));
static_assert(offsetof(native::GpuPcmBoxResult,normal)==
              offsetof(SourcePcmScContactStreamNative,normal));
static_assert(offsetof(native::GpuPcmBoxResult,count)==
              offsetof(SourcePcmScContactStreamNative,emitted_count));
static_assert(offsetof(native::GpuPcmBoxResult,points)==
              offsetof(SourcePcmScContactStreamNative,contacts));

// Sc owns every EdgeIndex and its lifetime. Device buffers below are
// physical mirrors for the source PCM kernels, never a pair registry.
class SourceScGpuPcmNative final {
public:
    SourceScGpuPcmNative(
        compute::Context& context,CudaDeviceAddressMap& addresses,
        SourceSapDmaNative& stream,physx::PxgHeapMemoryAllocator& allocator,
        const SourceScGpuBroadphaseProducerNative& sc,
        physx::PxgCudaBuffer& transforms,physx::PxgCudaBuffer& geometry,
        physx::PxgCudaBuffer& convex_vertices,
        const std::uint32_t& convex_vertex_count,
        physx::PxgCudaBuffer& convex_surface,
        const std::uint32_t& convex_surface_bytes,
        const std::uint32_t& convex_faces_base,
        const std::uint32_t& convex_indices_base,
        compute::ShaderBinary candidate,compute::ShaderBinary compact,
        compute::ShaderBinary gjk,compute::ShaderBinary epa,
        compute::ShaderBinary face_contacts)
        : context_(context),addresses_(addresses),stream_(stream),sc_(sc),
          transforms_(transforms),geometry_(geometry),
          convex_vertices_(convex_vertices),
          convex_vertex_count_(convex_vertex_count),
          convex_surface_(convex_surface),
          convex_surface_bytes_(convex_surface_bytes),
          convex_faces_base_(convex_faces_base),
          convex_indices_base_(convex_indices_base),
          candidates_stage_(context,std::move(candidate)),
          sparse_stage_(context,std::move(compact)),
          gjk_stage_(context,std::move(gjk),false,true),
          epa_stage_(context,std::move(epa)),
          face_stage_(context,std::move(face_contacts),true),
          pairs_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          results_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          candidates_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          edges_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          work_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          dense_contacts_(allocator,physx::PxsHeapStats::eNARROWPHASE) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument("Source Sc GPU PCM has another device stream");
        // D3D12 indirect arguments require their own physical resource.
        // Separate source buddy-heap allocations can share one root UAV.
        indirect_buffer_=context_.create_buffer(
            16u,compute::MemoryPlacement::DeviceLocal);
        try {
            indirect_address_=addresses_.register_slice(indirect_buffer_,0u,12u);
        } catch (...) {
            context_.destroy_buffer(indirect_buffer_);
            indirect_buffer_=0u;
            throw;
        }
    }
    ~SourceScGpuPcmNative() {
        if(indirect_address_)addresses_.unregister_slice(indirect_address_);
        if(indirect_buffer_)context_.destroy_buffer(indirect_buffer_);
    }

    void invalidate_after_rollback() noexcept { reset_manifolds_=true; }

    void generate(std::span<const native::GpuPcmBoxPair> requests,
                  std::span<native::GpuPcmBoxResult> output,
                  std::uint32_t shape_capacity) {
        if(requests.size()!=output.size())
            throw std::invalid_argument("Source Sc GPU PCM result count mismatch");
        // The previous successful source stream is authoritative for
        // whether this pair buffer has usable resident input records.
        const auto previous_count=published_pair_count_;
        published_pair_count_=0u;
        last_pair_upload_bytes_=0u;
        last_contact_download_bytes_=0u;
        last_edge_history_copy_bytes_=0u;
        if(requests.empty())return;
        if(!shape_capacity ||
           requests.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Source GPU PCM pair capacity overflow");
        const auto pair_count=static_cast<std::uint32_t>(requests.size());
        std::uint32_t candidate_count=0u;
        std::uint32_t largest_edge=0u;
        bool full_upload=reset_manifolds_ ||
            previous_count!=pair_count || pair_host_.size()!=pair_count;
        std::uint32_t first_dirty=pair_count,last_dirty=0u;
        pair_host_.resize(pair_count);
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& request=requests[i];
            if(request.first==request.second)
                throw std::invalid_argument("Sc PCM pair references one actor twice");
            if(request.plane_pair &&
               (!std::isfinite(request.plane_height) ||
                !std::isfinite(request.contact_distance) ||
                request.contact_distance<0.0f))
                throw std::invalid_argument("Sc PCM plane descriptor is invalid");
            if(request.edge==std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Source Sc edge watermark overflow");
            largest_edge=std::max(largest_edge,request.edge);
            // A plane ElementSim has no finite AABB storage; only its
            // participating box needs a device geometry and transform.
            const auto first=request.plane_pair?0u:
                sc_.source_element_handle(request.first);
            const auto second=sc_.source_element_handle(request.second);
            if((!request.plane_pair && first>=shape_capacity) ||
               second>=shape_capacity)
                throw std::out_of_range("Source Sc PCM element outside original AABB capacity");
            if(request.candidate_capacity!=1u &&
               request.candidate_capacity!=4u &&
               request.candidate_capacity!=16u)
                throw std::invalid_argument(
                    "GPU PCM candidate count must match its geometry");
            const auto reservation=request.plane_pair?0u:
                request.candidate_capacity;
            if(reservation>
               std::numeric_limits<std::uint32_t>::max()-candidate_count)
                throw std::out_of_range(
                    "Source GPU PCM candidate capacity overflow");
            const auto reset=(request.reset_cache || reset_manifolds_)
                ? source_pcm_reset_cache_bit : 0u;
            const SourcePcmBoxPairNative next=request.plane_pair
                ? SourcePcmBoxPairNative{
                    std::bit_cast<std::uint32_t>(request.plane_height),
                    second,request.edge,
                    std::bit_cast<std::uint32_t>(request.contact_distance),
                    source_pcm_plane_bit | reset}
                : SourcePcmBoxPairNative{
                    first,second,request.edge,candidate_count,
                    request.candidate_capacity | reset};
            candidate_count+=reservation;
            // Sc's previous pair records are the transfer baseline.
            // Preserve a single ordered upload by widening one dirty
            // interval over changed descriptors, never creating per-pair
            // submissions or another pair registry.
            if(!full_upload &&
               std::memcmp(&pair_host_[i],&next,sizeof(next))!=0) {
                first_dirty=std::min(first_dirty,i);
                last_dirty=i+1u;
            }
            pair_host_[i]=next;
        }
        const auto edge_capacity=largest_edge+1u;
        // Reserve the download destination before queueing any source
        // allocation or GPU work. Its address remains fixed through fetch.
        readback_.resize(pair_count);
        const auto requested_pair_bytes=source_allocation_bytes(pair_count,20u);
        const auto previous_pair_capacity=pairs_.getSize();
        const auto previous_pair_address=pairs_.getDevicePtr();
        pairs_.allocate(requested_pair_bytes,PX_FL);
        full_upload |= requested_pair_bytes>previous_pair_capacity ||
                       pairs_.getDevicePtr()!=previous_pair_address;
        results_.allocate(source_allocation_bytes(pair_count,32u),PX_FL);
        // Plane-only PCM frames need no candidate storage; retain the
        // source stages' nonzero view contract without 16 unused slots
        // per plane pair.
        const auto candidate_capacity=std::max(candidate_count,1u);
        candidates_.allocate(source_allocation_bytes(candidate_capacity,40u),PX_FL);
        // Only previously published EdgeIndex records belong to the
        // persistent Gu manifold. Capacity slack and the alignment pad
        // are not historical source contacts. Preserve the high-watermark
        // across dormant and temporarily excluded managers: their EdgeIndex
        // slots may be needed when their rigid actors wake again.
        const auto requested_edge_bytes=source_allocation_bytes(
            edge_capacity,240u);
        if(requested_edge_bytes>edges_.getSize())
            last_edge_history_copy_bytes_=published_edge_span_bytes_;
        edges_.allocateCopyPrefixAsync(requested_edge_bytes,
            published_edge_span_bytes_,stream_,PX_FL);
        work_.allocate(source_allocation_bytes(pair_count,4u),PX_FL);
        dense_contacts_.allocate(source_allocation_bytes(
            pair_count,sizeof(SourcePcmGpuContactStreamNative)),PX_FL);
        // Sc's reset intent travels with the source pair. The GPU
        // candidate owner initializes its own EdgeIndex record before
        // refreshing it, without a separate upload per new manager.
        if(full_upload) {
            first_dirty=0u;
            last_dirty=pair_count;
        }
        if(first_dirty<last_dirty) {
            const auto stride=sizeof(pair_host_[0]);
            const auto bytes=std::size_t(last_dirty-first_dirty)*stride;
            auto& uploads=stream_.upload_scratch();
            uploads.push_back({
                pairs_.getDevicePtr()+std::size_t(first_dirty)*stride,
                pair_host_.data()+first_dirty,bytes});
            stream_.queue_uploads(uploads);
            last_pair_upload_bytes_=bytes;
        }
        reset_manifolds_=false;

        candidates_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            indirect_address_,dense_contacts_.getDevicePtr(),
            convex_vertices_.getDevicePtr(),convex_vertex_count_,
            pair_count,shape_capacity,candidate_capacity,edge_capacity);
        sparse_stage_.append(stream_,addresses_,
            results_.getDevicePtr(),work_.getDevicePtr(),
            indirect_address_,pair_count);
        gjk_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            work_.getDevicePtr(),indirect_address_,
            dense_contacts_.getDevicePtr(),
            convex_vertices_.getDevicePtr(),convex_vertex_count_,
            pair_count,shape_capacity,candidate_capacity,edge_capacity);
        epa_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            work_.getDevicePtr(),indirect_address_,
            dense_contacts_.getDevicePtr(),
            convex_vertices_.getDevicePtr(),convex_vertex_count_,
            pair_count,shape_capacity,candidate_capacity,edge_capacity,
            convex_surface_.getDevicePtr(),convex_surface_bytes_,
            convex_faces_base_,convex_indices_base_);
        if(convex_vertex_count_)
            face_stage_.append(stream_,addresses_,
                transforms_.getDevicePtr(),geometry_.getDevicePtr(),
                pairs_.getDevicePtr(),results_.getDevicePtr(),
                candidates_.getDevicePtr(),edges_.getDevicePtr(),
                work_.getDevicePtr(),indirect_address_,
                dense_contacts_.getDevicePtr(),
                convex_vertices_.getDevicePtr(),convex_vertex_count_,
                pair_count,shape_capacity,candidate_capacity,edge_capacity,
                convex_surface_.getDevicePtr(),convex_surface_bytes_,
                convex_faces_base_,convex_indices_base_);

        // Fetch the losslessly packed device stream, then reconstruct
        // Sc's original host record. The Dy shader reads the same device
        // allocation; no GPU repacking command or second fence is needed.
        const auto readback_bytes=readback_.size()*sizeof(readback_[0]);
        const auto mapped=addresses_.resolve(
            dense_contacts_.getDevicePtr(),readback_bytes);
        const std::array<compute::DownloadSpan,1> downloads{
            {mapped.buffer,mapped.offset,readback_.data(),readback_bytes}
        };
        stream_.submit();
        try {
            context_.download_batch(downloads);
        } catch (...) {
            // Readback failure may not strand source GPU submission
            // tokens or their heap allocations.
            stream_.wait_at_fetch();
            throw;
        }
        // The readback fence has already completed. Retire the source
        // command/upload tokens without an intervening CPU/GPU stall.
        stream_.wait_at_fetch();
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& packed=readback_[i];
            const auto header=unpack_source_pcm_gpu_header_native(packed);
            auto& stored=output[i];
            stored.edge=header.edge_handle;
            stored.cached_count=header.cached_count;
            stored.status=header.status;
            stored.selected_and_regenerated=header.selected_and_regenerated;
            stored.count=header.emitted_count;
            const auto& request=requests[i];
            const auto selected_count=
                stored.selected_and_regenerated&0x7fffffffu;
            const bool regenerated=
                (stored.selected_and_regenerated&0x80000000u)!=0u;
            // A quiet refresh retains persistent witnesses beyond the
            // contact-distance window. Sc filters them at publication.
            if(stored.edge!=request.edge || stored.status!=0u ||
               selected_count>4u || stored.cached_count>4u ||
               selected_count>stored.cached_count ||
               (regenerated && selected_count!=stored.cached_count) ||
               stored.count>stored.cached_count)
                throw std::logic_error(
                    "Source GPU PCM publication disagrees with original Sc manager");
            // Only live Gu contacts cross into Sc's host manifold arena;
            // do not copy unused lanes or materialize a second 96-byte
            // temporary record for every pair.
            std::memcpy(&stored.normal,packed.normal,sizeof(packed.normal));
            std::memcpy(stored.points.data(),packed.contacts,
                        std::size_t(stored.count)*sizeof(packed.contacts[0]));
        }
        published_pair_count_=pair_count;
        published_edge_span_bytes_=std::max(
            published_edge_span_bytes_,std::size_t(edge_capacity)*240u);
        last_contact_download_bytes_=readback_bytes;
    }
    CudaDeviceAddressMap::Address contact_stream_address() const noexcept {
        return published_pair_count_?dense_contacts_.getDevicePtr():0u;
    }
    std::uint32_t contact_stream_pair_count() const noexcept {
        return published_pair_count_;
    }
    std::size_t last_pair_upload_bytes() const noexcept {
        return last_pair_upload_bytes_;
    }
    std::size_t last_contact_download_bytes() const noexcept {
        return last_contact_download_bytes_;
    }
    std::size_t last_edge_history_copy_bytes() const noexcept {
        return last_edge_history_copy_bytes_;
    }
private:
    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& stream_;
    const SourceScGpuBroadphaseProducerNative& sc_;
    physx::PxgCudaBuffer& transforms_;
    physx::PxgCudaBuffer& geometry_;
    physx::PxgCudaBuffer& convex_vertices_;
    const std::uint32_t& convex_vertex_count_;
    physx::PxgCudaBuffer& convex_surface_;
    const std::uint32_t& convex_surface_bytes_;
    const std::uint32_t& convex_faces_base_;
    const std::uint32_t& convex_indices_base_;
    SourcePcmBoxCandidatesNativeStage candidates_stage_;
    SourcePcmSparseEpaNativeStage sparse_stage_;
    SourcePcmEpaNativeStage gjk_stage_,epa_stage_,face_stage_;
    physx::PxgCudaBuffer pairs_,results_,candidates_,edges_,work_,dense_contacts_;
    compute::BufferId indirect_buffer_=0u;
    CudaDeviceAddressMap::Address indirect_address_=0u;
    std::vector<SourcePcmBoxPairNative> pair_host_;
    std::vector<SourcePcmGpuContactStreamNative> readback_;
    bool reset_manifolds_=false;
    std::uint32_t published_pair_count_=0u;
    std::size_t last_pair_upload_bytes_=0u;
    std::size_t last_contact_download_bytes_=0u;
    std::size_t last_edge_history_copy_bytes_=0u;
    std::size_t published_edge_span_bytes_=0u;
};
} // namespace fyzz::physx_gpu
