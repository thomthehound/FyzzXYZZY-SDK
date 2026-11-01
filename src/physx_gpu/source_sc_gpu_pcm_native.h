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
        compute::ShaderBinary candidate,compute::ShaderBinary reset,
        compute::ShaderBinary compact,compute::ShaderBinary epa,
        compute::ShaderBinary contact_stream)
        : context_(context),addresses_(addresses),stream_(stream),sc_(sc),
          transforms_(transforms),geometry_(geometry),
          candidates_stage_(context,std::move(candidate)),
          sparse_stage_(context,std::move(reset),std::move(compact)),
          epa_stage_(context,std::move(epa)),
          contact_stream_stage_(context,std::move(contact_stream)),
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
        // No contact managers means no indirect consumer this frame.
        // The next nonempty frame resets its count before compaction.
        if(requests.empty())return;
        if(!shape_capacity || requests.size()>
              std::numeric_limits<std::uint32_t>::max()/16u)
            throw std::out_of_range("Source GPU PCM pair or candidate capacity overflow");
        const auto pair_count=static_cast<std::uint32_t>(requests.size());
        const auto candidate_count=pair_count*16u; // 4 inside, 4 corners, 8 edge intersections
        std::uint32_t largest_edge=0u;
        for(const auto& request:requests) {
            if(request.first==request.second)
                throw std::invalid_argument("Sc PCM pair references one actor twice");
            largest_edge=std::max(largest_edge,request.edge);
        }
        if(largest_edge==std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Source Sc edge watermark overflow");
        const auto edge_capacity=largest_edge+1u;
        pair_host_.resize(pair_count);
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& request=requests[i];
            const auto first=sc_.source_element_handle(request.first);
            const auto second=sc_.source_element_handle(request.second);
            if(first>=shape_capacity || second>=shape_capacity)
                throw std::out_of_range("Source Sc PCM element outside original AABB capacity");
            pair_host_[i]={first,second,request.edge,i*16u,16u};
        }
        pairs_.allocate(source_allocation_bytes(pair_count,20u),PX_FL);
        results_.allocate(source_allocation_bytes(pair_count,32u),PX_FL);
        candidates_.allocate(source_allocation_bytes(candidate_count,40u),PX_FL);
        edges_.allocateCopyOldDataAsync(
            source_allocation_bytes(edge_capacity,240u),stream_,PX_FL);
        work_.allocate(source_allocation_bytes(pair_count,4u),PX_FL);
        dense_contacts_.allocate(source_allocation_bytes(pair_count,96u),PX_FL);
        auto& uploads=stream_.upload_scratch();
        uploads.reserve(pair_count+1u);
        uploads.push_back({pairs_.getDevicePtr(),pair_host_.data(),
                           pair_host_.size()*sizeof(pair_host_[0])});
        for(const auto& request:requests)
            if(request.reset_cache || reset_manifolds_)
                uploads.push_back({
                    edges_.getDevicePtr()+std::uint64_t(request.edge)*240u,
                    empty_cache_.data(),sizeof(empty_cache_)});
        stream_.queue_uploads(uploads);
        reset_manifolds_=false;

        candidates_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            pair_count,shape_capacity,candidate_count,edge_capacity);
        sparse_stage_.append(stream_,addresses_,
            results_.getDevicePtr(),work_.getDevicePtr(),
            indirect_address_,pair_count);
        epa_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            work_.getDevicePtr(),indirect_address_,
            pair_count,shape_capacity,candidate_count,edge_capacity);
        contact_stream_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            edges_.getDevicePtr(),dense_contacts_.getDevicePtr(),
            pair_count,shape_capacity,edge_capacity);

        // The shader emits the exact Sc-owned 96-byte record. Resolve
        // the download destination before submitting the source stream.
        const auto mapped=addresses_.resolve(
            dense_contacts_.getDevicePtr(),output.size_bytes());
        // Reuse the backend's cached readback staging. The download is
        // queued after all preceding PCM commands on this same Context;
        // waiting for its fence also completes every earlier source token.
        const std::vector<compute::DownloadSpan> downloads{
            {mapped.buffer,mapped.offset,output.data(),output.size_bytes()}
        };
        stream_.submit();
        try {
            context_.download_batch(downloads);
        } catch (...) {
            // Reap outstanding source allocations even when the readback
            // itself cannot publish. Sc rolls back the rejected frame.
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
    SourcePcmScContactStreamNativeStage contact_stream_stage_;
    physx::PxgCudaBuffer pairs_,results_,candidates_,edges_,work_,dense_contacts_;
    compute::BufferId indirect_buffer_=0u;
    CudaDeviceAddressMap::Address indirect_address_=0u;
    std::array<std::uint8_t,240u> empty_cache_{};
    std::vector<SourcePcmBoxPairNative> pair_host_;
    bool reset_manifolds_=false;
};
} // namespace fyzz::physx_gpu
