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
        dense_contacts_.allocate(source_allocation_bytes(pair_count,176u),PX_FL);
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
            pairs_.getDevicePtr(),edges_.getDevicePtr(),
            dense_contacts_.getDevicePtr(),pair_count,edge_capacity);
        stream_.wait_at_fetch();

        result_host_.resize(pair_count);
        packed_host_.resize(pair_count);
        const auto mapped_results=addresses_.resolve(
            results_.getDevicePtr(),result_host_.size()*sizeof(result_host_[0]));
        const auto mapped_packed=addresses_.resolve(
            dense_contacts_.getDevicePtr(),
            packed_host_.size()*sizeof(packed_host_[0]));
        const std::vector<compute::DownloadSpan> downloads{
            {mapped_results.buffer,mapped_results.offset,result_host_.data(),
                result_host_.size()*sizeof(result_host_[0])},
            {mapped_packed.buffer,mapped_packed.offset,packed_host_.data(),
                packed_host_.size()*sizeof(packed_host_[0])}
        };
        context_.download_batch(downloads);
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& result=result_host_[i];
            const auto& request=requests[i];
            const auto& stored=packed_host_[i];
            // A quiet refresh retains persistent witnesses beyond the
            // contact-distance window. Sc filters them at publication.
            if(result.edge_handle!=request.edge ||
               stored.edge_handle!=request.edge || result.status!=0u ||
               result.selected_count>4u || stored.count>4u ||
               result.selected_count>stored.count ||
               (result.regenerated!=0u &&
                result.selected_count!=stored.count))
                throw std::logic_error(
                    "Source GPU PCM publication disagrees with original Sc manager");
            auto& item=output[i];
            item={};
            item.edge=request.edge;
            item.count=static_cast<std::uint8_t>(stored.count);
            item.regenerated=result.regenerated!=0u;
            for(std::uint32_t j=0u;j<stored.count;++j) {
                const auto& c=stored.contacts[j];
                item.points[j]={
                    {c.local_a[0],c.local_a[1],c.local_a[2]},
                    {c.local_b[0],c.local_b[1],c.local_b[2]},
                    {c.normal_b[0],c.normal_b[1],c.normal_b[2]},
                    c.separation};
            }
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
    std::vector<SourcePcmBoxPairResultNative> result_host_;
    std::vector<SourcePcmScContactStreamNative> packed_host_;
    bool reset_manifolds_=false;
};
} // namespace fyzz::physx_gpu
