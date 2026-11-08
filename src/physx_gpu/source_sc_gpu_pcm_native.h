// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_sc_gpu_broadphase_producer_native.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"
#include "physx_gpu/source_pcm_sparse_epa_native.h"
#include "physx_gpu/source_pcm_epa_native.h"
#include "physx_gpu/source_pcm_sc_contact_stream_native.h"
#include "physx_gpu/source_sc_contact_history_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
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
        physx::PxgCudaBuffer& convex_vertices,
        const std::uint32_t& convex_vertex_count,
        physx::PxgCudaBuffer& convex_surface,
        const std::uint32_t& convex_surface_bytes,
        const std::uint32_t& convex_faces_base,
        const std::uint32_t& convex_indices_base,
        compute::ShaderBinary candidate,compute::ShaderBinary compact,
        compute::ShaderBinary gjk,compute::ShaderBinary gjk_primitive,
        compute::ShaderBinary gjk_reduce,
        compute::ShaderBinary gjk_publish,
        compute::ShaderBinary epa_args_reset,compute::ShaderBinary epa,
        compute::ShaderBinary face_contacts,
        compute::ShaderBinary contact_journal)
        : context_(context),addresses_(addresses),stream_(stream),sc_(sc),
          transforms_(transforms),geometry_(geometry),
          convex_vertices_(convex_vertices),
          convex_vertex_count_(convex_vertex_count),
          convex_surface_(convex_surface),
          convex_surface_bytes_(convex_surface_bytes),
          convex_faces_base_(convex_faces_base),
          convex_indices_base_(convex_indices_base),
          candidates_stage_(context,std::move(candidate)),
          sparse_stage_(context,compact,3u,epa_args_reset),
          epa_pending_stage_(context,std::move(compact),4u,
                             std::move(epa_args_reset)),
          staged_gjk_(gjk.format==compute::ShaderFormat::Dxil),
          gjk_stage_(context,std::move(gjk),false,true,false,staged_gjk_),
          gjk_reduce_stage_(context,std::move(gjk_reduce),false,true,false,staged_gjk_),
          gjk_publish_stage_(context,std::move(gjk_publish),false,true,false,staged_gjk_),
          epa_stage_(context,std::move(epa)),
          face_stage_(context,std::move(face_contacts),true),
          journal_stage_(context,std::move(contact_journal)),
          pairs_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          results_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          candidates_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          edges_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          work_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          dense_contacts_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          journal_(allocator,physx::PxsHeapStats::eNARROWPHASE) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument("Source Sc GPU PCM has another device stream");
        if(staged_gjk_)
            primitive_gjk_stage_.emplace(
                context_,std::move(gjk_primitive),false,true,false,true);
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

    void invalidate_after_rollback() noexcept {
        published_pair_count_=0u;
        rebuild_required_=true;
        if(pcm_transaction_pending_) {
            // A submitted PCM epoch may have changed Gu witnesses even
            // when Sc/Dy later rejected the complete simulation frame.
            restore_pending_=journal_pair_count_!=0u;
            pcm_transaction_pending_=false;
            published_pair_count_=0u;
            // Refreshed PCM epochs restore their journal before the next submission.
            reset_manifolds_=false;
        } else reset_manifolds_=true;
    }

    bool needs_contact_rebuild() const noexcept {
        return restore_pending_ || rebuild_required_;
    }

    void commit_step() {
        if(!pcm_transaction_pending_)return;
        published_edge_span_bytes_=std::max(
            published_edge_span_bytes_,pending_edge_span_bytes_);
        pcm_transaction_pending_=false;
        journal_pair_count_=0u;
    }

    // Primitive CPU Gu contacts are already authoritative for Sc.
    // Upload their Dy layout once rather than executing GPU PCM and
    // downloading a duplicate numerical manifold for verification.
    void upload_host_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<const native::GpuPcmBoxResult> results) {
        if(requests.empty() || requests.size()!=results.size() ||
           requests.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid Sc primitive PCM upload");
        if(pcm_transaction_pending_)
            throw std::logic_error("Source PCM host upload before GPU epoch commit");
        restore_rejected_epoch();
        for(std::size_t i=0;i<requests.size();++i)
            validate_host_contact(requests[i],results[i]);
        const auto bytes=source_allocation_bytes(
            requests.size(),sizeof(SourcePcmGpuContactStreamNative));
        dense_contacts_.allocate(bytes,PX_FL);
        const auto transfer=results.size_bytes();
        auto& uploads=stream_.upload_scratch();
        uploads.push_back({
            dense_contacts_.getDevicePtr(),results.data(),transfer});
        stream_.queue_uploads(uploads);
        published_pair_count_=static_cast<std::uint32_t>(requests.size());
        last_pair_upload_bytes_=0u;
        last_contact_download_bytes_=0u;
        last_host_contact_upload_bytes_=transfer;
        last_edge_history_copy_bytes_=0u;
        last_continuation_commands_=0u;
        last_face_pair_scan_count_=0u;
        rebuild_required_=false;
    }

    void generate(std::span<const native::GpuPcmBoxPair> requests,
                  std::span<native::GpuPcmBoxResult> output,
                  std::uint32_t shape_capacity) {
        if(requests.size()!=output.size())
            throw std::invalid_argument("Source Sc GPU PCM result count mismatch");
        if(pcm_transaction_pending_)
            throw std::logic_error("Sc PCM frame advanced before its prior epoch committed");
        restore_rejected_epoch();
        // The previous successful source stream is authoritative for
        // whether this pair buffer has usable resident input records.
        const auto previous_count=published_pair_count_;
        published_pair_count_=0u;
        last_pair_upload_bytes_=0u;
        last_contact_download_bytes_=0u;
        last_host_contact_upload_bytes_=0u;
        last_edge_history_copy_bytes_=0u;
        last_continuation_commands_=0u;
        last_face_pair_scan_count_=0u;
        if(requests.empty()) {
            rebuild_required_=false;
            return;
        }
        if((!shape_capacity && !requests.empty()) ||
           requests.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Source GPU PCM pair capacity overflow");
        const auto pair_count=static_cast<std::uint32_t>(requests.size());
        std::uint32_t candidate_count=0u;
        std::uint32_t largest_edge=0u;
        bool needs_continuation=false;
        std::uint32_t first_face_pair=pair_count,face_pair_end=0u;
        bool full_upload=reset_manifolds_ ||
            previous_count!=pair_count || pair_host_.size()!=pair_count;
        std::uint32_t first_dirty=pair_count,last_dirty=0u;
        pair_host_.resize(pair_count);
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& request=requests[i];
            needs_continuation|=!request.candidate_complete;
            if(request.host_generated) {
                validate_host_contact(request,output[i]);
                if(!request.candidate_complete)
                    throw std::logic_error(
                        "CPU source Gu cannot own GPU continuation pair");
            }
            if(!request.plane_pair && request.candidate_capacity==4u) {
                first_face_pair=std::min(first_face_pair,i);
                face_pair_end=i+1u;
            }
            if(request.candidate_complete && !request.plane_pair &&
               request.candidate_capacity!=1u)
                throw std::invalid_argument(
                    "Candidate-complete finite PCM must use sphere geometry");
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
            const auto reservation=
                request.host_generated || request.plane_pair
                    ?0u:request.candidate_capacity;
            if(reservation>
               std::numeric_limits<std::uint32_t>::max()-candidate_count)
                throw std::out_of_range(
                    "Source GPU PCM candidate capacity overflow");
            const auto resetting=request.reset_cache || reset_manifolds_;
            const auto reset=resetting?source_pcm_reset_cache_bit:0u;
            const auto prior=resetting && request.previously_touching
                ? source_pcm_prior_touch_bit:0u;
            const SourcePcmBoxPairNative next=request.plane_pair
                ? SourcePcmBoxPairNative{
                    std::bit_cast<std::uint32_t>(request.plane_height),
                    second,request.edge,
                    std::bit_cast<std::uint32_t>(request.contact_distance),
                    source_pcm_plane_bit | reset | prior |
                        (request.host_generated?source_pcm_host_gu_bit:0u)}
                : SourcePcmBoxPairNative{
                    first,second,request.edge,candidate_count,
                    request.candidate_capacity | reset | prior |
                        (request.host_generated?source_pcm_host_gu_bit:0u)};
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
        if(pair_count) {
            const auto requested_pair_bytes=source_allocation_bytes(pair_count,20u);
            const auto previous_pair_capacity=pairs_.getSize();
            const auto previous_pair_address=pairs_.getDevicePtr();
            pairs_.allocate(requested_pair_bytes,PX_FL);
            full_upload |= requested_pair_bytes>previous_pair_capacity ||
                           pairs_.getDevicePtr()!=previous_pair_address;
            results_.allocate(source_allocation_bytes(pair_count,32u),PX_FL);
        }
        // Plane-only PCM frames need no candidate storage; retain the
        // source stages' nonzero view contract without 16 unused slots
        // per plane pair.
        const auto candidate_capacity=std::max(candidate_count,1u);
        if(pair_count)
            candidates_.allocate(source_allocation_bytes(candidate_capacity,40u),PX_FL);
        // Only previously published EdgeIndex records belong to the
        // persistent Gu manifold. Capacity slack and the alignment pad
        // are not historical source contacts. Preserve the high-watermark
        // across dormant and temporarily excluded managers: their EdgeIndex
        // slots may be needed when their rigid actors wake again.
        const auto requested_edge_bytes=source_allocation_bytes(
            edge_capacity,240u);
        const bool edge_grew=requested_edge_bytes>edges_.getSize();
        if(edge_grew)
            last_edge_history_copy_bytes_=published_edge_span_bytes_;
        edges_.allocateCopyPrefixAsync(requested_edge_bytes,
            published_edge_span_bytes_,stream_,PX_FL);
        // A grown EdgeIndex pool must retain its last committed prefix
        // even if later frame preparation fails before PCM submission.
        // Submit the source device copy without a host fence. Failed
        // steps can then discard only the unsubmitted PCM commands.
        if(edge_grew && published_edge_span_bytes_)stream_.submit();
        if(pair_count) {
            journal_.allocate(source_allocation_bytes(pair_count,240u),PX_FL);
        }
        if(pair_count) {
            work_.allocate(source_allocation_bytes(
                std::size_t(pair_count)+1u,4u),PX_FL);
            dense_contacts_.allocate(source_allocation_bytes(
                pair_count,sizeof(SourcePcmGpuContactStreamNative)),PX_FL);
        }
        // Sc's reset intent travels with the source pair. The GPU
        // candidate owner initializes its own EdgeIndex record before
        // refreshing it, without a separate upload per new manager.
        if(full_upload) {
            first_dirty=0u;
            last_dirty=pair_count;
        }
        auto& uploads=stream_.upload_scratch();
        if(first_dirty<last_dirty) {
            const auto stride=sizeof(pair_host_[0]);
            const auto bytes=std::size_t(last_dirty-first_dirty)*stride;
            uploads.push_back({
                pairs_.getDevicePtr()+std::size_t(first_dirty)*stride,
                pair_host_.data()+first_dirty,bytes});
            last_pair_upload_bytes_=bytes;
        }
        // Sc primitive slots share the GPU PCM world-contact ABI.
        // Transfer only contiguous host-owned spans; general GPU
        // publishers fill the remaining ordinals in this same buffer.
        for(std::size_t i=0u;i<pair_count;) {
            if(!requests[i].host_generated) {
                ++i;
                continue;
            }
            const auto first=i;
            while(i<pair_count && requests[i].host_generated)++i;
            const auto stride=sizeof(native::GpuPcmBoxResult);
            const auto bytes=(i-first)*stride;
            uploads.push_back({
                dense_contacts_.getDevicePtr()+first*stride,
                output.data()+first,bytes});
            last_host_contact_upload_bytes_+=bytes;
        }
        if(!uploads.empty())stream_.queue_uploads(uploads);
        reset_manifolds_=false;
        // The source EdgeIndex pool is allowed to grow beyond previously
        // committed history. Initialize only newly admitted records;
        // recycled slots retain their original PCM reset semantics.
        const auto initialized_edges=static_cast<std::uint32_t>(
            std::min<std::size_t>(edge_capacity,
                                 published_edge_span_bytes_/240u));
        if(initialized_edges<edge_capacity)
            journal_stage_.append(stream_,addresses_,
                SourceScContactHistoryNativeStage::Phase::Initialize,
                pairs_.getDevicePtr(),edges_.getDevicePtr(),
                journal_.getDevicePtr(),0u,edge_capacity,initialized_edges);
        if(pair_count)
            journal_stage_.append(stream_,addresses_,
                SourceScContactHistoryNativeStage::Phase::Snapshot,
                pairs_.getDevicePtr(),edges_.getDevicePtr(),
                journal_.getDevicePtr(),pair_count,edge_capacity);

        candidates_stage_.append(stream_,addresses_,
            transforms_.getDevicePtr(),geometry_.getDevicePtr(),
            pairs_.getDevicePtr(),results_.getDevicePtr(),
            candidates_.getDevicePtr(),edges_.getDevicePtr(),
            indirect_address_,dense_contacts_.getDevicePtr(),
            convex_vertices_.getDevicePtr(),convex_vertex_count_,
            pair_count,shape_capacity,candidate_capacity,edge_capacity);
        const auto before_continuation=stream_.commands().size();
        if(needs_continuation) {
            const auto append_gjk=[&](const SourcePcmEpaNativeStage& stage,
                                       std::uint32_t pass) {
                stage.append(stream_,addresses_,
                    transforms_.getDevicePtr(),geometry_.getDevicePtr(),
                    pairs_.getDevicePtr(),results_.getDevicePtr(),
                    candidates_.getDevicePtr(),edges_.getDevicePtr(),
                    work_.getDevicePtr(),indirect_address_,
                    dense_contacts_.getDevicePtr(),
                    convex_vertices_.getDevicePtr(),convex_vertex_count_,
                    pair_count,shape_capacity,candidate_capacity,edge_capacity,
                    0u,0u,0u,0u,pass);
            };
            if(staged_gjk_) {
                // Direct Vulkan GJK scans source pair status in one pass;
                // only D3D12's staged continuation consumes this list.
                sparse_stage_.append(stream_,addresses_,
                    results_.getDevicePtr(),work_.getDevicePtr(),
                    indirect_address_,pair_count);
                // The GPU compacted status-3 list drives every staged GJK
                // pass. The list remains immutable until EPA work is rebuilt.
                const auto& advance=!convex_vertex_count_
                    ?*primitive_gjk_stage_:gjk_stage_;
                for(std::uint32_t pass=0u;pass<32u;++pass) {
                    append_gjk(advance,pass);
                    append_gjk(gjk_reduce_stage_,pass);
                }
                append_gjk(advance,32u);
                append_gjk(gjk_publish_stage_,32u);
            }else append_gjk(gjk_stage_,0u);
            epa_pending_stage_.append(stream_,addresses_,
                results_.getDevicePtr(),work_.getDevicePtr(),
                indirect_address_,pair_count);
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
            // Sc has already classified which finite pairs contain
            // convex geometry. Shift source pair/result/contact views
            // to that interval without a separate index upload, GPU
            // compaction or report readback.
            if(first_face_pair<face_pair_end &&
               convex_vertex_count_ && convex_surface_bytes_) {
                const auto span=face_pair_end-first_face_pair;
                last_face_pair_scan_count_=span;
                face_stage_.append(stream_,addresses_,
                    transforms_.getDevicePtr(),geometry_.getDevicePtr(),
                    pairs_.getDevicePtr()+std::size_t(first_face_pair)*20u,
                    results_.getDevicePtr()+std::size_t(first_face_pair)*32u,
                    candidates_.getDevicePtr(),edges_.getDevicePtr(),
                    work_.getDevicePtr(),indirect_address_,
                    dense_contacts_.getDevicePtr()+
                        std::size_t(first_face_pair)*
                            sizeof(SourcePcmGpuContactStreamNative),
                    convex_vertices_.getDevicePtr(),convex_vertex_count_,
                    span,shape_capacity,candidate_capacity,edge_capacity,
                    convex_surface_.getDevicePtr(),convex_surface_bytes_,
                    convex_faces_base_,convex_indices_base_);
            }
        }
        last_continuation_commands_=
            stream_.commands().size()-before_continuation;

        // Only general GPU Gu slots cross into Sc. The primitive
        // slots are already present in output and resident in the
        // unchanged 84-byte PCM stream used by Dy. All readback spans
        // share one GPU completion boundary.
        gpu_downloads_.clear();
        for(std::size_t i=0u;i<pair_count;) {
            if(requests[i].host_generated) {
                ++i;
                continue;
            }
            const auto first=i;
            while(i<pair_count && !requests[i].host_generated)++i;
            const auto stride=sizeof(native::GpuPcmBoxResult);
            const auto bytes=(i-first)*stride;
            const auto mapped=addresses_.resolve(
                dense_contacts_.getDevicePtr()+first*stride,bytes);
            gpu_downloads_.push_back({
                mapped.buffer,mapped.offset,output.data()+first,bytes});
            last_contact_download_bytes_+=bytes;
        }
        pcm_transaction_pending_=true;
        journal_pair_count_=pair_count;
        journal_edge_capacity_=edge_capacity;
        pending_edge_span_bytes_=std::size_t(edge_capacity)*240u;
        if(!gpu_downloads_.empty())stream_.fetch(gpu_downloads_);
        else stream_.submit();
        for(std::uint32_t i=0u;i<pair_count;++i) {
            const auto& stored=output[i];
            const auto& request=requests[i];
            const auto selected_count=
                stored.selected_and_regenerated()&0x7fffffffu;
            const bool regenerated=
                (stored.selected_and_regenerated()&0x80000000u)!=0u;
            // A quiet refresh retains persistent witnesses beyond the
            // contact-distance window. Sc filters them at publication.
            if(stored.edge!=request.edge || stored.status()!=0u ||
               selected_count>4u || stored.cached_count()>4u ||
               selected_count>stored.cached_count() ||
               (regenerated && selected_count!=stored.cached_count()) ||
               stored.count()>stored.cached_count() ||
               ((stored.touch_transition()==native::GpuPcmTouchTransition::Found ||
                 stored.touch_transition()==native::GpuPcmTouchTransition::Persists)
                  != (stored.count()>0u)))
                throw std::logic_error(
                    "Source GPU PCM publication disagrees with original Sc manager");
        }
        published_pair_count_=pair_count;
        rebuild_required_=false;
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
    std::size_t last_host_contact_upload_bytes() const noexcept {
        return last_host_contact_upload_bytes_;
    }
    std::size_t last_edge_history_copy_bytes() const noexcept {
        return last_edge_history_copy_bytes_;
    }
    std::size_t last_continuation_commands() const noexcept {
        return last_continuation_commands_;
    }
    std::size_t last_face_pair_scan_count() const noexcept {
        return last_face_pair_scan_count_;
    }
private:
    static void validate_host_contact(
            const native::GpuPcmBoxPair& request,
            const native::GpuPcmBoxResult& result) {
        if(!request.candidate_complete ||
           !(request.candidate_capacity==1u ||
             (request.plane_pair && request.candidate_capacity==16u)) ||
           request.edge!=result.edge || result.count()>4u ||
           result.status()!=0u ||
           result.cached_count()!=result.count() ||
           (result.selected_and_regenerated()&0x7fffffffu)!=result.count() ||
           ((result.count()!=0u)!=
            (result.touch_transition()==native::GpuPcmTouchTransition::Found ||
             result.touch_transition()==native::GpuPcmTouchTransition::Persists)))
            throw std::logic_error("Invalid source Gu primitive contact stream");
        if(!result.count())return;
        const auto n=result.normal;
        if(!std::isfinite(n.x) || !std::isfinite(n.y) ||
           !std::isfinite(n.z))
            throw std::overflow_error("Nonfinite Sc primitive PCM normal");
        for(std::uint32_t j=0u;j<result.count();++j) {
            const auto& point=result.points[j];
            if(!std::isfinite(point.position.x) ||
               !std::isfinite(point.position.y) ||
               !std::isfinite(point.position.z) ||
               !std::isfinite(point.separation))
                throw std::overflow_error("Nonfinite Sc primitive PCM witness");
        }
    }
    void restore_rejected_epoch() {
        if(!restore_pending_)return;
        // A rejected GPU Gu journal must be restored before another
        // producer can touch the same persistent EdgeIndex history.
        journal_stage_.append(stream_,addresses_,
            SourceScContactHistoryNativeStage::Phase::Restore,
            pairs_.getDevicePtr(),edges_.getDevicePtr(),
            journal_.getDevicePtr(),journal_pair_count_,
            journal_edge_capacity_);
        stream_.submit();
        stream_.wait_at_fetch();
        restore_pending_=false;
        journal_pair_count_=0u;
    }
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
    SourcePcmSparseEpaNativeStage sparse_stage_,epa_pending_stage_;
    bool staged_gjk_=false;
    SourcePcmEpaNativeStage gjk_stage_,gjk_reduce_stage_,gjk_publish_stage_,epa_stage_,face_stage_;
    std::optional<SourcePcmEpaNativeStage> primitive_gjk_stage_;
    SourceScContactHistoryNativeStage journal_stage_;
    physx::PxgCudaBuffer pairs_,results_,candidates_,edges_,work_,dense_contacts_,journal_;
    compute::BufferId indirect_buffer_=0u;
    CudaDeviceAddressMap::Address indirect_address_=0u;
    std::vector<SourcePcmBoxPairNative> pair_host_;
    std::vector<compute::DownloadSpan> gpu_downloads_;
    bool reset_manifolds_=false;
    bool rebuild_required_=false;
    bool pcm_transaction_pending_=false,restore_pending_=false;
    std::uint32_t journal_pair_count_=0u,journal_edge_capacity_=0u;
    std::size_t pending_edge_span_bytes_=0u;
    std::uint32_t published_pair_count_=0u;
    std::size_t last_pair_upload_bytes_=0u;
    std::size_t last_contact_download_bytes_=0u;
    std::size_t last_host_contact_upload_bytes_=0u;
    std::size_t last_edge_history_copy_bytes_=0u;
    std::size_t last_continuation_commands_=0u;
    std::size_t last_face_pair_scan_count_=0u;
    std::size_t published_edge_span_bytes_=0u;
};
} // namespace fyzz::physx_gpu
