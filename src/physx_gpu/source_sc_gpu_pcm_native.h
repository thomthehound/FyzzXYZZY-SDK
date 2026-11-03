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
#include <cstdint>
#include <cstring>
#include <functional>
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
        compute::ShaderBinary candidate,compute::ShaderBinary compact,
        compute::ShaderBinary epa)
        : context_(context),addresses_(addresses),stream_(stream),sc_(sc),
          transforms_(transforms),geometry_(geometry),
          candidates_stage_(context,std::move(candidate)),
          sparse_stage_(context,std::move(compact)),
          epa_stage_(context,std::move(epa)),
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
                  std::uint32_t shape_capacity,
                  const std::function<void()>& on_submitted = {}) {
        if(requests.size()!=output.size())
            throw std::invalid_argument("Source Sc GPU PCM result count mismatch");
        // The previous successful source stream is authoritative for
        // whether this pair buffer has usable resident input records.
        const auto previous_count=published_pair_count_;
        published_pair_count_=0u;
        last_pair_upload_bytes_=0u;
        if(requests.empty())return;
        if(!shape_capacity || requests.size()>
              std::numeric_limits<std::uint32_t>::max()/16u)
            throw std::out_of_range("Source GPU PCM pair or candidate capacity overflow");
        const auto pair_count=static_cast<std::uint32_t>(requests.size());
        const auto candidate_count=pair_count*16u; // 4 inside, 4 corners, 8 edge intersections
        std::uint32_t largest_edge=0u;
        bool full_upload=reset_manifolds_ ||
            previous_count!=pair_count || pair_host_.size()!=pair_count;
        std::uint32_t first_dirty=pair_count,last_dirty=0u;
        pair_host_.resize(pair_count);
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& request=requests[i];
            if(request.first==request.second)
                throw std::invalid_argument("Sc PCM pair references one actor twice");
            if(request.edge==std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Source Sc edge watermark overflow");
            largest_edge=std::max(largest_edge,request.edge);
            const auto first=sc_.source_element_handle(request.first);
            const auto second=sc_.source_element_handle(request.second);
            if(first>=shape_capacity || second>=shape_capacity)
                throw std::out_of_range("Source Sc PCM element outside original AABB capacity");
            const SourcePcmBoxPairNative next{
                first,second,request.edge,i*16u,
                16u | ((request.reset_cache || reset_manifolds_)
                    ? source_pcm_reset_cache_bit : 0u)};
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
        const auto requested_pair_bytes=source_allocation_bytes(pair_count,20u);
        const auto previous_pair_capacity=pairs_.getSize();
        const auto previous_pair_address=pairs_.getDevicePtr();
        pairs_.allocate(requested_pair_bytes,PX_FL);
        full_upload |= requested_pair_bytes>previous_pair_capacity ||
                       pairs_.getDevicePtr()!=previous_pair_address;
        results_.allocate(source_allocation_bytes(pair_count,32u),PX_FL);
        candidates_.allocate(source_allocation_bytes(candidate_count,40u),PX_FL);
        edges_.allocateCopyOldDataAsync(
            source_allocation_bytes(edge_capacity,240u),stream_,PX_FL);
        work_.allocate(source_allocation_bytes(pair_count,4u),PX_FL);
        dense_contacts_.allocate(source_allocation_bytes(pair_count,96u),PX_FL);
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
            pair_count,shape_capacity,candidate_count,edge_capacity);
        sparse_stage_.append(stream_,addresses_,
            results_.getDevicePtr(),work_.getDevicePtr(),
            indirect_address_,pair_count);
        epa_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            work_.getDevicePtr(),indirect_address_,
            dense_contacts_.getDevicePtr(),pair_count,shape_capacity,
            candidate_count,edge_capacity);


        // The shader emits the exact Sc-owned 96-byte record. Resolve
        // the download destination before submitting the source stream.
        const auto mapped=addresses_.resolve(
            dense_contacts_.getDevicePtr(),output.size_bytes());
        // Reuse the backend's cached readback staging. The download is
        // queued after all preceding PCM commands on this same Context;
        // waiting for its fence also completes every earlier source token.
        const std::array<compute::DownloadSpan,1> downloads{
            {mapped.buffer,mapped.offset,output.data(),output.size_bytes()}
        };
        stream_.submit();
        try {
            // Sc plane contact managers have independent PCM caches.
            // Prepare their CPU Gu witnesses while box PCM executes on
            // the original GPU stream, before Sc requires GPU readback.
            if(on_submitted)on_submitted();
            context_.download_batch(downloads);
        } catch (...) {
            // Neither CPU preparation nor readback failure may strand
            // source GPU submission tokens or their heap allocations.
            stream_.wait_at_fetch();
            throw;
        }
        // The readback fence has already completed. Retire the source
        // command/upload tokens without an intervening CPU/GPU stall.
        stream_.wait_at_fetch();
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& request=requests[i];
            const auto& stored=output[i];
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
        }
        published_pair_count_=pair_count;
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
private:
    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& stream_;
    const SourceScGpuBroadphaseProducerNative& sc_;
    physx::PxgCudaBuffer& transforms_;
    physx::PxgCudaBuffer& geometry_;
    SourcePcmBoxCandidatesNativeStage candidates_stage_;
    SourcePcmSparseEpaNativeStage sparse_stage_;
    SourcePcmEpaNativeStage epa_stage_;
    physx::PxgCudaBuffer pairs_,results_,candidates_,edges_,work_,dense_contacts_;
    compute::BufferId indirect_buffer_=0u;
    CudaDeviceAddressMap::Address indirect_address_=0u;
    std::vector<SourcePcmBoxPairNative> pair_host_;
    bool reset_manifolds_=false;
    std::uint32_t published_pair_count_=0u;
    std::size_t last_pair_upload_bytes_=0u;
};
} // namespace fyzz::physx_gpu
