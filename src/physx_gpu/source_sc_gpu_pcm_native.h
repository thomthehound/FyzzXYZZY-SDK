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
#include "physx_gpu/source_sc_device_islands_native.h"
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
        compute::ShaderBinary contact_journal,
        compute::ShaderBinary device_islands)
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
          island_stage_(context,std::move(device_islands)),
          pairs_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          results_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          candidates_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          edges_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          work_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          dense_contacts_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          journal_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          gu_touch_records_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          gu_touch_journal_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          island_edges_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          island_nodes_(allocator,physx::PxsHeapStats::eNARROWPHASE),
          island_work_(allocator,physx::PxsHeapStats::eNARROWPHASE) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument("Source Sc GPU PCM has another device stream");
        if(staged_gjk_)
            primitive_gjk_stage_.emplace(
                context_,std::move(gjk_primitive),false,true,false,true);
#if defined(FYZZXYZZY_PROFILE_GPU)
        gjk_stage_.set_profile_label("pcm.gjk_initial");
        gjk_reduce_stage_.set_profile_label("pcm.gjk_reduce");
        gjk_publish_stage_.set_profile_label("pcm.gjk_publish");
        if(primitive_gjk_stage_)
            primitive_gjk_stage_->set_profile_label("pcm.gjk_primitive");
#endif
        // PCM GJK/EPA and Sc island admission share one physical
        // indirect-argument resource.
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
        // A rejected Sc transaction must not publish its outstanding
        // asynchronous witness DMA into the borrowed Gu result arena.
        stream_.abandon_fetch();
        host_upload_pending_=false;
        result_pending_=false;
        island_work_ready_=false;
        pending_host_inputs_=false;
        pending_requests_={};
        pending_output_={};
        published_pair_count_=0u;
        rebuild_required_=true;
        if(pcm_transaction_pending_) {
            // A submitted PCM epoch may have changed Gu witnesses even
            // when Sc/Dy later rejected the complete simulation frame.
            restore_pending_=journal_pair_count_!=0u || gu_touch_count_!=0u;
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
        if(result_pending_)
            throw std::logic_error("Source PCM epoch committed before Sc publication");
        if(!pcm_transaction_pending_) {
            island_work_ready_=false;
            return;
        }
        // A collision-only publication has no Dy device consumer to
        // retire the primitive upload. Complete it at the existing Sc
        // transaction commit, after independent island publication.
        if(host_upload_pending_) {
            stream_.wait_at_fetch();
            host_upload_pending_=false;
        }
        published_edge_span_bytes_=std::max(
            published_edge_span_bytes_,pending_edge_span_bytes_);
        pcm_transaction_pending_=false;
        journal_pair_count_=0u;
        gu_touch_count_=0u;
        island_work_ready_=false;
        journal_host_only_=false;
    }

    // Primitive CPU Gu contacts are already authoritative for Sc.
    // Upload their Dy layout once rather than executing GPU PCM and
    // downloading a duplicate numerical manifold for verification.
    void upload_host_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<const native::GpuPcmBoxResult> results,
            std::uint32_t minimum_edge_capacity=0u) {
        if(requests.empty() || requests.size()!=results.size() ||
           requests.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid Sc primitive PCM upload");
        if(pcm_transaction_pending_)
            throw std::logic_error("Source PCM host upload before GPU epoch commit");
        restore_rejected_epoch();
        std::uint32_t largest_edge=0u;
        for(std::size_t i=0;i<requests.size();++i) {
            validate_host_contact(requests[i],results[i]);
            if(requests[i].edge==std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Source Sc edge watermark overflow");
            largest_edge=std::max(largest_edge,requests[i].edge);
        }
        const auto pair_count=static_cast<std::uint32_t>(requests.size());
        const auto edge_capacity=std::max(
            largest_edge+1u,minimum_edge_capacity);
        const auto bytes=source_allocation_bytes(
            pair_count,sizeof(SourcePcmGpuContactStreamNative));
        dense_contacts_.allocate(bytes,PX_FL);
        const auto requested_edge_bytes=source_allocation_bytes(
            edge_capacity,240u);
        const bool edge_grew=requested_edge_bytes>edges_.getSize();
        last_edge_history_copy_bytes_=edge_grew
            ?published_edge_span_bytes_:0u;
        edges_.allocateCopyPrefixAsync(requested_edge_bytes,
            published_edge_span_bytes_,stream_,PX_FL);
        if(edge_grew && published_edge_span_bytes_)stream_.submit();
        journal_.allocate(source_allocation_bytes(pair_count,8u),PX_FL);

        // The host Gu stream already contains every EdgeIndex and touch
        // transition. Use it for the same device history transaction
        // without uploading a redundant pair-index stream.
        auto& uploads=stream_.upload_scratch();
        uploads.push_back({
            dense_contacts_.getDevicePtr(),results.data(),results.size_bytes()});
        stream_.queue_uploads(uploads);
        const auto initialized_edges=static_cast<std::uint32_t>(
            std::min<std::size_t>(edge_capacity,
                                  published_edge_span_bytes_/240u));
        if(initialized_edges<edge_capacity)
            journal_stage_.append(stream_,addresses_,
                SourceScContactHistoryNativeStage::Phase::Initialize,
                dense_contacts_.getDevicePtr(),edges_.getDevicePtr(),
                journal_.getDevicePtr(),0u,edge_capacity,initialized_edges);
        journal_stage_.append(stream_,addresses_,
            SourceScContactHistoryNativeStage::Phase::PublishHostWithSnapshot,
            dense_contacts_.getDevicePtr(),edges_.getDevicePtr(),
            journal_.getDevicePtr(),pair_count,edge_capacity);
        // Submit the snapshot before Sc can reject this epoch.
        pcm_transaction_pending_=true;
        journal_host_only_=true;
        journal_pair_count_=pair_count;
        journal_edge_capacity_=edge_capacity;
        pending_edge_span_bytes_=std::size_t(edge_capacity)*240u;
        published_pair_count_=pair_count;
        pending_host_inputs_=false;
        host_upload_pending_=true;
        stream_.submit();
        last_pair_upload_bytes_=0u;
        last_contact_download_bytes_=0u;
        last_host_contact_upload_bytes_=results.size_bytes();
        last_continuation_commands_=0u;
        last_face_pair_scan_count_=0u;
        rebuild_required_=false;
    }

    void enqueue(std::span<const native::GpuPcmBoxPair> requests,
                  std::span<native::GpuPcmBoxResult> output,
                  std::uint32_t shape_capacity,
                  std::uint32_t minimum_edge_capacity=0u) {
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
        host_upload_pending_=false;
        island_work_ready_=false;
        pending_host_inputs_=false;
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
            if(request.host_generated && !request.candidate_complete)
                throw std::logic_error(
                    "CPU source Gu cannot own GPU continuation pair");
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
        const auto edge_capacity=std::max(
            largest_edge+1u,minimum_edge_capacity);
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
        }
        pcm_transaction_pending_=true;
        result_pending_=true;
        pending_requests_=requests;
        pending_output_=output;
        journal_pair_count_=pair_count;
        journal_edge_capacity_=edge_capacity;
        journal_host_only_=false;
        pending_host_inputs_=std::any_of(requests.begin(),requests.end(),
            [](const auto& request){return request.host_generated;});
        pending_edge_span_bytes_=std::size_t(edge_capacity)*240u;
        // The DMA follows the last PCM shader immediately, while Sc
        // independently admits speculative topology and CPU Gu.
        // No host wait occurs until the first GPU witness consumer.
        if(gpu_downloads_.empty())stream_.submit();
        else stream_.begin_fetch(gpu_downloads_);
    }
    // Source Sc produces bounded primitive witnesses independently of
    // general GPU PCM. Snapshot their uploads before the Sc witness join;
    // the Context owns both staging and the borrowed result buffers.
    // GPU Gu ordinals are not inspected until their readback completes.
    void enqueue_host_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<const native::GpuPcmBoxResult> output) {
        if(!result_pending_ || !pcm_transaction_pending_ ||
           host_upload_pending_ ||
           requests.data()!=pending_requests_.data() ||
           requests.size()!=pending_requests_.size() ||
           output.data()!=pending_output_.data() ||
           output.size()!=pending_output_.size())
            throw std::logic_error("Mixed source Gu admission is not the pending PCM epoch");
        auto& uploads=stream_.upload_scratch();
        std::size_t bytes=0u;
        for(std::size_t i=0u;i<requests.size();) {
            if(!requests[i].host_generated) {
                ++i;
                continue;
            }
            const auto first=i;
            while(i<requests.size() && requests[i].host_generated) {
                validate_host_contact(requests[i],output[i]);
                ++i;
            }
            const auto stride=sizeof(native::GpuPcmBoxResult);
            const auto size=(i-first)*stride;
            uploads.push_back({
                dense_contacts_.getDevicePtr()+first*stride,
                output.data()+first,size});
            bytes+=size;
        }
        if(uploads.empty())
            throw std::logic_error("Mixed PCM epoch has no bounded CPU Gu records");
        stream_.queue_uploads(uploads);
        // This updates both GPU PCM and CPU Gu physical-touch words in
        // source order. It executes after the independent GPU contact
        // producer and the CPU Gu upload, without blocking Sc's readback.
        journal_stage_.append(stream_,addresses_,
            SourceScContactHistoryNativeStage::Phase::PublishHostTouch,
            dense_contacts_.getDevicePtr(),edges_.getDevicePtr(),
            journal_.getDevicePtr(),journal_pair_count_,
            journal_edge_capacity_);
        last_host_contact_upload_bytes_=bytes;
        host_upload_pending_=true;
    }

    // Legacy CPU Gu publishes only its existing EdgeIndex and physical
    // transition, never a second contact-manager or PCM witness. The
    // resident touch journal remains authoritative for both producers.
    void enqueue_cpu_gu_touches(
            std::span<const native::GpuGuTouchRecord> touches) {
        if(touches.empty())return;
        if(!pcm_transaction_pending_ || !device_contact_ready() ||
           gu_touch_count_ || touches.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::logic_error("CPU Gu touch journal has no active PCM epoch");
        const auto capacity=device_edge_history_capacity();
        for(const auto& record:touches)
            if(record.edge_index>=capacity || (record.touch_bits&~3u)!=0u)
                throw std::out_of_range("CPU Gu touch exceeds resident EdgeIndex history");
        const auto count=static_cast<std::uint32_t>(touches.size());
        const auto bytes=touches.size_bytes();
        gu_touch_records_.allocate(bytes,PX_FL);
        gu_touch_journal_.allocate(bytes,PX_FL);
        auto& uploads=stream_.upload_scratch();
        uploads.push_back({
            gu_touch_records_.getDevicePtr(),touches.data(),bytes});
        stream_.queue_uploads(uploads);
        journal_stage_.append(stream_,addresses_,
            SourceScContactHistoryNativeStage::Phase::PublishGuTouch,
            gu_touch_records_.getDevicePtr(),edges_.getDevicePtr(),
            gu_touch_journal_.getDevicePtr(),count,capacity);
        gu_touch_count_=count;
    }

    // Sc supplies only stable, typed endpoints and authored iteration
    // metadata. GPU PCM publishes actual touch on the same queue; the
    // accurate island stage consumes its resident history without a
    // host witness join or another contact-manager registry.
    void enqueue_device_islands(
            std::span<const SourceScDeviceIslandEdge> edges,
            std::span<const std::uint32_t> node_flags,
            SourceScDeviceIslandSources sources,
            bool defer_to_dy=false) {
        if(island_work_ready_ || !device_contact_ready())
            throw std::logic_error(
                "Device island admission requires one complete queued Gu epoch");
        if(edges.size()>std::numeric_limits<std::uint32_t>::max() ||
           node_flags.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Device Sc island input exceeds 32-bit view");
        const auto node_count=static_cast<std::uint32_t>(node_flags.size());
        const auto edge_count=static_cast<std::uint32_t>(edges.size());
        const auto edge_capacity=device_edge_history_capacity();
        constexpr std::uint32_t invalid=0xffffffffu;
        for(const auto& edge:edges) {
            if(edge.edge_index==invalid ||
               (edge.kind&~(source_sc_device_island_joint_bit|
                             source_sc_device_island_no_response_bit|
                             source_sc_device_island_device_joint_bit))!=0u ||
               ((edge.kind&source_sc_device_island_device_joint_bit)!=0u &&
                (edge.kind&source_sc_device_island_joint_bit)==0u) ||
               ((edge.kind&source_sc_device_island_joint_bit)==0u &&
                edge.edge_index>=edge_capacity) ||
               (edge.node_a!=invalid && edge.node_a>=node_count) ||
               (edge.node_b!=invalid && edge.node_b>=node_count))
                throw std::invalid_argument("Device Sc island edge is not admitted");
        }
        for(const auto word:node_flags)
            if((word&0xfffc0000u)!=0u ||
               ((word&source_sc_device_island_nonresponding_bit)==0u &&
                (word&0xffu)==0u))
                throw std::invalid_argument("Device Sc node iterations invalid");
        island_layout_=SourceScDeviceIslandLayout::for_counts(
            node_count,edge_count);
        island_work_.allocate(island_layout_.bytes,PX_FL);
        if(edge_count)
            island_edges_.allocate(source_allocation_bytes(
                edge_count,sizeof(SourceScDeviceIslandEdge)),PX_FL);
        if(node_count)
            island_nodes_.allocate(source_allocation_bytes(
                node_count,sizeof(std::uint32_t)),PX_FL);
        auto& uploads=stream_.upload_scratch();
        if(edge_count)
            uploads.push_back({
                island_edges_.getDevicePtr(),edges.data(),edges.size_bytes()});
        if(node_count)
            uploads.push_back({
                island_nodes_.getDevicePtr(),node_flags.data(),
                node_flags.size_bytes()});
        if(!uploads.empty())stream_.queue_uploads(uploads);
        island_stage_.append(stream_,addresses_,
            edge_count?island_edges_.getDevicePtr():0u,
            device_edge_history_address(),
            node_count?island_nodes_.getDevicePtr():0u,
            island_work_.getDevicePtr(),
            node_count,edge_count,edge_capacity,sources);
        // Reuse the PCM indirect resource once its earlier launches
        // have completed on this ordered queue.
        stream_.queue_device_copy(
            indirect_address_,
            island_work_.getDevicePtr()+
                offsetof(SourceScDeviceIslandHeader,dispatch_x),
            sizeof(compute::DispatchSize));
        // Production Dy appends its numerical inputs and solve directly
        // behind this transaction, so both stages share one submission.
        if(!defer_to_dy)stream_.submit();
        island_work_ready_=true;
    }
    CudaDeviceAddressMap::Address device_island_work_address() const noexcept {
        return island_work_ready_?island_work_.getDevicePtr():0u;
    }
    SourceScDeviceIslandLayout device_island_layout() const noexcept {
        return island_work_ready_?island_layout_:SourceScDeviceIslandLayout{};
    }
    std::optional<compute::DispatchIndirectArgs>
    device_island_indirect_args() const noexcept {
        if(!island_work_ready_)return std::nullopt;
        return compute::DispatchIndirectArgs{indirect_buffer_,0u};
    }

    void finish() {
        if(!result_pending_)
            throw std::logic_error("Source PCM join has no queued epoch");
        const auto requests=pending_requests_;
        const auto output=pending_output_;
        // General GPU PCM and host Gu occupy disjoint source ordinals.
        // Sc needs the GPU witnesses now, not the CPU Gu device upload.
        // The independent CPU Gu upload was already queued after the
        // witness DMA; only the earlier GPU result is a host dependency.
        if(!gpu_downloads_.empty())stream_.finish_fetch();
        else stream_.wait_at_fetch();
        for(const auto& span:gpu_downloads_)
            last_contact_download_bytes_+=span.bytes;
        for(std::uint32_t i=0u;i<journal_pair_count_;++i) {
            const auto& stored=output[i];
            const auto& request=requests[i];
            const auto selected_count=
                stored.selected_and_regenerated()&0x7fffffffu;
            const bool regenerated=
                (stored.selected_and_regenerated()&0x80000000u)!=0u;
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
        // CPU Gu was submitted before this host witness join. No later
        // device continuation can extend the readback dependency.
        if(pending_host_inputs_ && !host_upload_pending_)
            throw std::logic_error("Mixed PCM host Gu was not admitted before fetch");
        published_pair_count_=journal_pair_count_;
        result_pending_=false;
        pending_host_inputs_=false;
        pending_requests_={};
        pending_output_={};
        rebuild_required_=false;
    }
    // GPU task admission consumes the queued numerical stream without
    // joining its host witness DMA. Mixed epochs become complete only
    // after their bounded CPU Gu upload enters the same ordered queue.
    // A rejected transaction invalidates the view immediately.
    bool device_contact_ready() const noexcept {
        return !restore_pending_ &&
            (!rebuild_required_ || pcm_transaction_pending_) &&
            (!result_pending_ || !pending_host_inputs_ || host_upload_pending_);
    }
    std::uint32_t device_contact_pair_count() const noexcept {
        if(!device_contact_ready())return 0u;
        return result_pending_?journal_pair_count_:published_pair_count_;
    }
    CudaDeviceAddressMap::Address device_contact_stream_address() const noexcept {
        return device_contact_pair_count()?dense_contacts_.getDevicePtr():0u;
    }
    // Dormant managers can retain their last committed physical touch
    // even when no new PCM pair is admitted this frame. The journal
    // high-watermark and the last published EdgeIndex prefix coexist.
    std::uint32_t device_edge_history_capacity() const noexcept {
        if(!device_contact_ready())return 0u;
        const auto published=static_cast<std::uint32_t>(
            published_edge_span_bytes_/240u);
        return pcm_transaction_pending_
            ?std::max(published,journal_edge_capacity_):published;
    }
    CudaDeviceAddressMap::Address device_edge_history_address() const noexcept {
        return device_edge_history_capacity()?edges_.getDevicePtr():0u;
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
        // Undo CPU-only Gu transitions first, then restore the preceding
        // PCM snapshot. Both affect one authoritative device history.
        if(gu_touch_count_)
            journal_stage_.append(stream_,addresses_,
                SourceScContactHistoryNativeStage::Phase::RestoreGuTouch,
                gu_touch_records_.getDevicePtr(),edges_.getDevicePtr(),
                gu_touch_journal_.getDevicePtr(),gu_touch_count_,
                journal_edge_capacity_);
        // A rejected GPU Gu journal must be restored before another
        // producer can touch the same persistent EdgeIndex history.
        journal_stage_.append(stream_,addresses_,
            journal_host_only_
                ?SourceScContactHistoryNativeStage::Phase::RestoreHost
                :SourceScContactHistoryNativeStage::Phase::Restore,
            journal_host_only_?dense_contacts_.getDevicePtr():
                               pairs_.getDevicePtr(),
            edges_.getDevicePtr(),journal_.getDevicePtr(),
            journal_pair_count_,journal_edge_capacity_);
        stream_.submit();
        stream_.wait_at_fetch();
        restore_pending_=false;
        journal_pair_count_=0u;
        gu_touch_count_=0u;
        journal_host_only_=false;
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
    SourceScDeviceIslandsNativeStage island_stage_;
    physx::PxgCudaBuffer pairs_,results_,candidates_,edges_,work_,dense_contacts_,journal_;
    physx::PxgCudaBuffer gu_touch_records_,gu_touch_journal_;
    physx::PxgCudaBuffer island_edges_,island_nodes_,island_work_;
    SourceScDeviceIslandLayout island_layout_{};
    compute::BufferId indirect_buffer_=0u;
    CudaDeviceAddressMap::Address indirect_address_=0u;
    std::vector<SourcePcmBoxPairNative> pair_host_;
    std::vector<compute::DownloadSpan> gpu_downloads_;
    bool reset_manifolds_=false;
    bool rebuild_required_=false;
    bool pcm_transaction_pending_=false,restore_pending_=false;
    bool result_pending_=false,host_upload_pending_=false;
    bool pending_host_inputs_=false;
    bool island_work_ready_=false;
    bool journal_host_only_=false;
    // Borrowed Sc views survive until finish() or rollback; GPU output
    // is written directly into the enqueued Sc result arena at the join.
    std::span<const native::GpuPcmBoxPair> pending_requests_{};
    std::span<native::GpuPcmBoxResult> pending_output_{};
    std::uint32_t journal_pair_count_=0u,journal_edge_capacity_=0u;
    std::uint32_t gu_touch_count_=0u;
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
