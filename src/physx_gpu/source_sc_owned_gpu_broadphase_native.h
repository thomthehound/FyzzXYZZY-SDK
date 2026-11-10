// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sc_gpu_broadphase_producer_native.h"
#include "physx_gpu/source_sap_aggregate_owned_stages_native.h"
#include "hardware_profile.h"
#include "fyzzxyzzy/native_gpu_world.h"
#include "physx_foundation_lifetime.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_translate_aabbs_native.h"
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sapbox_indices_native.h"
#include "physx_gpu/source_sc_gpu_pcm_native.h"
#include "physx_gpu/source_dy_distance_native.h"
#include "physx_gpu/source_sap_endpoint_histograms_native.h"
#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_region_create_native.h"
#include "physx_gpu/source_region_histograms_native.h"
#include "physx_gpu/source_region_newbounds_chain_native.h"
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_sap_updated_native.h"
#include "physx_gpu/source_sap_incremental_counts_native.h"
#include "physx_gpu/source_sap_incremental_pairs_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// The source GPU stage order is the same for SPIR-V and DXIL. A production
// shader provider supplies compiled stages; the scene never invokes DXC
// while stepping or silently selects a CPU broadphase.
using SourceSapShaderBinaries = NativeGpuSapShaders;

class SourceScOwnedGpuBroadphaseNative final : public native::GpuBroadphaseProducer {
    static compute::ShaderBinary take_shader(
        SourceSapShaderBinaries& shaders,NativeGpuSapStage stage) {
        return std::move(shaders[static_cast<std::size_t>(stage)]);
    }
    static compute::ShaderBinary take_pcm_shader(
        NativeGpuPcmShaders& shaders,NativeGpuPcmStage stage) {
        return std::move(shaders[static_cast<std::size_t>(stage)]);
    }
    static void abort_source(void* state) noexcept {
        *static_cast<bool*>(state)=true;
    }

public:
    explicit SourceScOwnedGpuBroadphaseNative(
        compute::Backend backend,SourceSapShaderBinaries shaders,
        std::uint32_t max_pairs=32768u,
        std::uint32_t max_aggregate_pairs=32768u,
        std::optional<NativeGpuAggregateShaders> aggregate_shaders=std::nullopt,
        std::optional<NativeGpuPcmShaders> pcm_shaders=std::nullopt,
        std::optional<compute::ShaderBinary> dy_shader=std::nullopt,
        std::optional<compute::ShaderBinary> convex_dy_shader=std::nullopt)
        : foundation_(physx_runtime::acquire_foundation()),
          context_(backend),
          stream_(context_,addresses_),
          device_roots_(context_,addresses_),
          device_callback_(device_roots_,&stream_),
          pinned_roots_(context_,addresses_,stream_),
          mapped_roots_(context_,addresses_,stream_),
          manager_(std::in_place,1u<<20u,device_callback_,
                   pinned_roots_,mapped_roots_,abort_source,&aborted_),
          allocators_(physx::PxgCreateAllocatorDesc(*manager_)),
          sap_(std::make_unique<PxgSapSourceMembersNative>(
              allocators_,max_pairs,max_aggregate_pairs)),
          aabb_(std::make_unique<PxgAabbSourceMembersNative>(
              allocators_,0u,aggregate_shaders?max_aggregate_pairs:0u,
              aggregate_shaders?max_aggregate_pairs:0u)),
          encoding_(context_,take_shader(shaders,NativeGpuSapStage::TranslateAabbs)),
          box_transforms_(context_,take_shader(shaders,NativeGpuSapStage::BoxTransformBounds)),
          box_input_(allocators_.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          box_cache_(allocators_.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          box_geometry_(allocators_.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          convex_vertices_(allocators_.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          convex_surface_(allocators_.deviceAlloc,physx::PxsHeapStats::eNARROWPHASE),
          box_owners_{box_transforms_,box_input_,box_cache_,box_geometry_,
                      convex_vertices_},
          lowering_(addresses_,encoding_),
          created_(context_,take_shader(shaders,NativeGpuSapStage::CreatedPairs)),
          radix_(context_,take_shader(shaders,NativeGpuSapStage::RadixInitialize),take_shader(shaders,NativeGpuSapStage::RadixCount),
                 take_shader(shaders,NativeGpuSapStage::RadixScatter),take_shader(shaders,NativeGpuSapStage::RadixGather)),
          indices_(context_,take_shader(shaders,NativeGpuSapStage::SapBoxIndices)),
          histograms_(context_,take_shader(shaders,NativeGpuSapStage::EndpointHistogramFirst),take_shader(shaders,NativeGpuSapStage::EndpointHistogramSecond)),
          fill_(context_,take_shader(shaders,NativeGpuSapStage::FillD32)),
          region_(context_,take_shader(shaders,NativeGpuSapStage::RegionCreate),fill_),
          region_histograms_(
              context_,take_shader(shaders,NativeGpuSapStage::RegionHistogramFirst),take_shader(shaders,NativeGpuSapStage::RegionHistogramWave),
              take_shader(shaders,NativeGpuSapStage::RegionHistogramSecond)),
          active_(context_,take_shader(shaders,NativeGpuSapStage::ActiveWriteout),take_shader(shaders,NativeGpuSapStage::ActiveWave),
                  take_shader(shaders,NativeGpuSapStage::ActivePortable),take_shader(shaders,NativeGpuSapStage::ActiveOutput)),
          overlap_(context_,take_shader(shaders,NativeGpuSapStage::OverlapWork)),
          prefix_(context_,take_shader(shaders,NativeGpuSapStage::OverlapScanFirst),take_shader(shaders,NativeGpuSapStage::OverlapScanSecond)),
          newbounds_pairs_(context_,take_shader(shaders,NativeGpuSapStage::NewBoundsPairs)),
          newbounds_(overlap_,prefix_,newbounds_pairs_),
          regions_(region_,region_histograms_,active_,newbounds_),
          reports_(context_,take_shader(shaders,NativeGpuSapStage::ReportFirst),take_shader(shaders,NativeGpuSapStage::ReportSecond),
                   take_shader(shaders,NativeGpuSapStage::ReportCopy),take_shader(shaders,NativeGpuSapStage::ReportWaveFirst),
                   take_shader(shaders,NativeGpuSapStage::ReportWaveSecond)),
          removed_(context_,take_shader(shaders,NativeGpuSapStage::RemovedMark),take_shader(shaders,NativeGpuSapStage::RemovedProjections)),
          updated_(context_,take_shader(shaders,NativeGpuSapStage::UpdatedBounds)),
          comparisons_(context_,take_shader(shaders,NativeGpuSapStage::IncrementalCountFirst),
                       take_shader(shaders,NativeGpuSapStage::IncrementalCountSecond)),
          incremental_(context_,take_shader(shaders,NativeGpuSapStage::IncrementalPairs)),
          clear_(context_,take_shader(shaders,NativeGpuSapStage::ClearNewFlags)),
          stages_{lowering_,removed_,updated_,radix_,histograms_,
                  comparisons_,incremental_,indices_,created_,regions_,
                  clear_,reports_},
          filter_(false,false),
          aggregate_stages_(aggregate_shaders
              ? std::make_unique<SourceSapAggregateOwnedStagesNative>(
                  context_,std::move(*aggregate_shaders))
              : nullptr),
          producer_(std::make_unique<SourceScGpuBroadphaseProducerNative>(
              *aabb_,*sap_,stream_,stages_,addresses_,context_,filter_,
              fill_,mapped_roots_,
              aggregate_stages_?&aggregate_stages_->borrowed():nullptr,
              &box_owners_)) {
        if(pcm_shaders)
            pcm_=std::make_unique<SourceScGpuPcmNative>(
                context_,addresses_,stream_,allocators_.deviceAlloc,
                *producer_,box_cache_,box_geometry_,
                convex_vertices_,box_owners_.convex_vertex_count,
                convex_surface_,convex_surface_bytes_,
                convex_surface_faces_base_,convex_surface_indices_base_,
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::Candidates),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaCompact),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::GjkContact),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::GjkPrimitive),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::GjkReduce),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::GjkPublish),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaArgsReset),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaContact),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::FaceContacts),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::ContactJournal),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::DeviceIslands));
        if(dy_shader)
            dy_distance_=std::make_unique<SourceDyDistanceNativeStage>(
                context_,addresses_,stream_,allocators_.deviceAlloc,
                std::move(*dy_shader),std::move(convex_dy_shader));
        else if(convex_dy_shader)
            throw std::invalid_argument(
                "Convex Dy shader requires the primitive Dy solver");
    }

    SourceScOwnedGpuBroadphaseNative(const SourceScOwnedGpuBroadphaseNative&)=delete;
    SourceScOwnedGpuBroadphaseNative& operator=(
        const SourceScOwnedGpuBroadphaseNative&)=delete;

    ~SourceScOwnedGpuBroadphaseNative() override {
        // Drain every source launch before retiring the PCM indirect
        // argument resource, source SAP pipelines or the original heap.
        // Failed submissions stay owned by Context through teardown.
        (void)stream_.finish_shutdown();
        dy_distance_.reset();
        pcm_.reset();
        producer_.reset();
        box_input_.deallocate();
        box_cache_.deallocate();
        box_geometry_.deallocate();
        convex_vertices_.deallocate();
        convex_surface_.deallocate();
        aabb_.reset();
        sap_.reset();
        manager_->flushDeferredDeallocs();
        manager_.reset();
        try {
            device_roots_.close(&stream_);
        } catch (...) {
            // SourcePhysxHeapRoots owns all remaining roots and performs
            // non-throwing retirement during member destruction.
        }
    }

    void upload_convex_vertices(std::span<const Vec3> vertices) override {
        static_assert(sizeof(Vec3)==12u);
        if(vertices.size()>std::numeric_limits<std::uint32_t>::max()/12u)
            throw std::overflow_error("Convex GPU vertex arena exceeds 32-bit size");
        const auto bytes=vertices.size()*12u;
        convex_vertices_.allocate(bytes,PX_FL);
        if(bytes)stream_.queue_uploads({{
            convex_vertices_.getDevicePtr(),vertices.data(),bytes}});
        box_owners_.convex_vertex_count=
            static_cast<std::uint32_t>(vertices.size());
    }
    void upload_convex_geometry(
        std::span<const Vec3> vertices,
        std::span<const native::GpuConvexFaceSpan> spans,
        std::span<const ConvexFace> faces,
        std::span<const std::uint32_t> indices) override {
        if(spans.size()!=vertices.size() ||
           vertices.size()>std::numeric_limits<std::uint32_t>::max()/8u ||
           faces.size()>std::numeric_limits<std::uint32_t>::max()/24u ||
           indices.size()>std::numeric_limits<std::uint32_t>::max()/4u)
            throw std::overflow_error("Convex face resource exceeds device limits");
        const std::uint64_t bytes=spans.size()*8ull+
            faces.size()*24ull+indices.size()*4ull;
        if(bytes>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Convex face resource exceeds 32-bit addressing");
        // One immutable resident byte stream: per-vertex face span,
        // polygon plane records, then ordered local vertex indices.
        const auto faces_base=static_cast<std::uint32_t>(spans.size()*8u);
        const auto indices_base=static_cast<std::uint32_t>(
            spans.size()*8u+faces.size()*24u);
        std::vector<std::uint32_t> packed(bytes/4u);
        for(std::size_t i=0u;i<spans.size();++i) {
            packed[i*2u]=spans[i].first_face;
            packed[i*2u+1u]=spans[i].face_count;
        }
        for(std::size_t i=0u;i<faces.size();++i) {
            const auto& f=faces[i];
            const auto offset=faces_base/4u+i*6u;
            packed[offset]=std::bit_cast<std::uint32_t>(f.normal.x);
            packed[offset+1u]=std::bit_cast<std::uint32_t>(f.normal.y);
            packed[offset+2u]=std::bit_cast<std::uint32_t>(f.normal.z);
            packed[offset+3u]=std::bit_cast<std::uint32_t>(f.distance);
            packed[offset+4u]=f.first_index;
            packed[offset+5u]=f.index_count;
        }
        std::copy(indices.begin(),indices.end(),
                  packed.begin()+indices_base/4u);
        upload_convex_vertices(vertices);
        convex_surface_words_=std::move(packed);
        convex_surface_.allocate(bytes,PX_FL);
        if(bytes)stream_.queue_uploads({{
            convex_surface_.getDevicePtr(),convex_surface_words_.data(),
            static_cast<std::size_t>(bytes)}});
        convex_surface_bytes_=static_cast<std::uint32_t>(bytes);
        convex_surface_faces_base_=faces_base;
        convex_surface_indices_base_=indices_base;
    }
    native::GpuBroadphaseReports advance(
        std::span<const native::GpuBroadphaseBox> live,
        bool force_update,bool geometry_dirty) override {
        if(aborted_)
            throw std::runtime_error("Original GPU SAP heap aborted");
        return producer_->advance(live,force_update,geometry_dirty);
    }
    bool has_gpu_pcm_box() const noexcept override { return bool(pcm_); }
    bool has_gpu_dy_distance() const noexcept override {
        return bool(dy_distance_);
    }
    std::uint32_t gpu_dy_direct_task_limit() const noexcept override {
        return dy_distance_ ? context_.device_info().max_workgroup_count[0] : 0u;
    }
    void enqueue_gpu_dy_distance(
        std::span<native::GpuDyDistanceBody> bodies,
        std::span<native::GpuDyJointRow> rows,
        std::span<const native::GpuDyDistanceDescriptor> descriptors,
        std::span<const std::uint32_t> partitions,
        std::span<const std::uint32_t> ordinals,
        std::span<native::GpuDyDistanceTask> tasks,
        std::span<float> contact_forces,
        std::span<const native::GpuDyGuContactInput> cpu_gu,
        std::span<native::GpuDyFrictionWriteback> friction,
        std::span<const native::GpuDyFrictionAnchorInput> anchors,
        std::span<native::GpuDyContactPatch> patches,
        std::span<const native::GpuDyContactManifold> manifolds,
        std::span<const std::uint32_t> contact_order,
        std::span<const native::GpuDyContactPrepPair> contact_prep,
        std::span<native::GpuDyBodyIntegrationInput> integration,
        std::span<native::GpuDyIntegratedPose> integrated_poses,
        bool device_partitions=false,
        std::span<native::GpuDyFrictionAnchorImpulse> anchor_impulses={},
        bool device_initial_bodies=false,
        bool device_world_motion=false,
        bool device_unconstrained_preparation=false,
        std::span<native::GpuDySleepState> sleep_states={},
        std::span<const native::GpuDyJointInput> joint_inputs={}) override {
        FYZZ_PROFILE_SCOPE("dy.device_enqueue");
        if(!dy_distance_)
            throw std::logic_error("Source GPU Dy shader not installed");
        if(dy_distance_->has_queued_solve())
            throw std::logic_error("Source GPU Dy numerical epoch already queued");
        if(pcm_ && !pcm_->device_contact_ready())
            throw std::logic_error(
                "Source Dy cannot consume an incomplete PCM contact epoch");
        // Resolve the first attached shape into the ElementSim cache.
        // Body integration also covers shapes without a device target.
        for(auto& input:integration) {
            if(input.active&16u)continue;
            const auto dy_flags=input.active&32u;
            if(!input.active || input.source_element==0xffffffffu) {
                input.active=(input.active&1u)|dy_flags;
                continue;
            }
            // Authorization is consumed by this device submission. Read
            // it before obtaining the writable ElementSim handle.
            const bool resident=(input.active&4u)!=0u &&
                producer_->has_authorized_device_cached_pose(input.source_element);
            input.active=1u|(resident?2u:0u)|dy_flags;
            input.source_element=producer_->device_shape_cache_handle(input.source_element);
            if(input.source_element==0xffffffffu) {
                input.source_shape_count=0u;
                input.active=1u|dy_flags;
            }
        }
        dy_distance_->enqueue(bodies,rows,descriptors,partitions,ordinals,tasks,
                            contact_forces,cpu_gu,friction,anchors,patches,manifolds,contact_order,
                            contact_prep,integration,integrated_poses,
                            pcm_?pcm_->device_contact_stream_address():0u,
                            pcm_?pcm_->device_contact_pair_count():0u,
                            box_cache_.getDevicePtr(),
                            sap_->pre_broadphase_buffers().bounds.getDevicePtr(),
                            box_geometry_.getDevicePtr(),aabb_->capacity(),
                            convex_vertices_.getDevicePtr(),
                            box_owners_.convex_vertex_count,device_partitions,
                            anchor_impulses,device_initial_bodies,
                            device_world_motion,
                            device_unconstrained_preparation,sleep_states,
                            joint_inputs,
                            pcm_ && pcm_->device_island_work_address()
                                ?std::optional<SourceScDeviceIslandDyView>{
                                    SourceScDeviceIslandDyView{
                                        pcm_->device_island_work_address(),
                                        pcm_->device_island_layout(),
                                        *pcm_->device_island_indirect_args()}}
                                :std::nullopt);
        ++dy_distance_dispatch_count_;
#if defined(FYZZXYZZY_PROFILE_GPU)
        const auto& schedule=dy_distance_->last_schedule();
        FYZZ_PROFILE_COUNTER("dy.dispatches",schedule.dispatches);
        FYZZ_PROFILE_COUNTER("dy.indirect_dispatches",schedule.device_island_indirect_dispatches);
        FYZZ_PROFILE_COUNTER("dy.wide_tasks",schedule.wide_tasks);
        FYZZ_PROFILE_COUNTER("dy.partition_groups",schedule.max_partition_groups);
        FYZZ_PROFILE_COUNTER("dy.tile_bytes",schedule.tile_bytes);
        FYZZ_PROFILE_COUNTER("dy.device_colored_descriptors",schedule.device_colored_descriptors);
        FYZZ_PROFILE_COUNTER("dy.host_degree_scanned",schedule.host_degree_scanned_descriptors);
        FYZZ_PROFILE_COUNTER("dy.device_admitted_tasks",schedule.device_admitted_tasks);
        FYZZ_PROFILE_COUNTER("dy.device_admitted_descriptors",schedule.device_admitted_descriptors);
        FYZZ_PROFILE_COUNTER("dy.device_generated_bytes",schedule.device_generated_bytes);
        FYZZ_PROFILE_COUNTER("dy.source_descriptor_upload_bytes",schedule.source_descriptor_upload_bytes);
        FYZZ_PROFILE_COUNTER("dy.body_upload_bytes",schedule.device_body_upload_bytes);
        FYZZ_PROFILE_COUNTER("dy.actor_local_upload_bytes",schedule.actor_local_upload_bytes);
        FYZZ_PROFILE_COUNTER("dy.resident_poses_reused",schedule.reused_actor_local_poses);
        FYZZ_PROFILE_COUNTER("dy.resident_velocities_reused",schedule.resident_velocity_reuses);
        FYZZ_PROFILE_COUNTER("dy.sleep_readback_bytes",schedule.sleep_readback_bytes);
#endif
    }
    void finish_gpu_dy_distance() override {
        FYZZ_PROFILE_SCOPE("dy.device_result_join");
        if(!dy_distance_ || !dy_distance_->has_queued_solve())
            throw std::logic_error("Source GPU Dy has no queued numerical epoch");
        dy_distance_->finish();
    }
    bool has_queued_gpu_dy_distance() const noexcept {
        return dy_distance_ && dy_distance_->has_queued_solve();
    }
    std::size_t dy_resident_pose_count() const noexcept {
        return dy_distance_ ? dy_distance_->last_resident_pose_count() : 0u;
    }
    std::size_t dy_host_pose_upload_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_host_pose_upload_bytes() : 0u;
    }
    std::uint64_t dy_distance_dispatch_count() const noexcept {
        return dy_distance_dispatch_count_;
    }
    std::uint64_t source_handle_projection_count() const noexcept {
        return producer_->source_handle_projection_count();
    }
    std::uint64_t source_sap_fetch_count() const noexcept {
        return producer_->source_sap_fetch_count();
    }
    std::size_t dy_gpu_integrated_body_count() const noexcept {
        return dy_distance_?dy_distance_->last_integrated_body_count():0u;
    }
    SourceDyLaunchStats dy_launch_stats() const noexcept {
        return dy_distance_?dy_distance_->last_schedule():SourceDyLaunchStats{};
    }
    const SourceSapDmaNative::SubmissionStats&
    stream_submission_stats() const noexcept {
        return stream_.submission_stats();
    }
    std::size_t pcm_pair_upload_bytes() const noexcept {
        return pcm_ ? pcm_->last_pair_upload_bytes() : 0u;
    }
    std::size_t pcm_contact_readback_bytes() const noexcept {
        return pcm_ ? pcm_->last_contact_download_bytes() : 0u;
    }
    // The device dependency is ready as soon as the source contact
    // producer and all host Gu spans are queued, not when Sc joins DMA.
    CudaDeviceAddressMap::Address device_pcm_contact_stream() const noexcept {
        return pcm_ ? pcm_->device_contact_stream_address() : 0u;
    }
    std::uint32_t device_pcm_contact_count() const noexcept {
        return pcm_ ? pcm_->device_contact_pair_count() : 0u;
    }
    CudaDeviceAddressMap::Address device_pcm_touch_history() const noexcept {
        return pcm_ ? pcm_->device_edge_history_address() : 0u;
    }
    std::uint32_t device_pcm_touch_capacity() const noexcept {
        return pcm_ ? pcm_->device_edge_history_capacity() : 0u;
    }
    void enqueue_gpu_dy_island_admission(
            std::span<const native::GpuDyIslandEdge> edges,
            std::span<const std::uint32_t> node_flags,
            native::GpuDyIslandSources sources) override {
        if(!pcm_)
            throw std::logic_error("Source Sc GPU island admission unavailable");
        pcm_->enqueue_device_islands(edges,node_flags,sources);
    }
    CudaDeviceAddressMap::Address device_island_work_address() const noexcept {
        return pcm_?pcm_->device_island_work_address():0u;
    }
    SourceScDeviceIslandLayout device_island_layout() const noexcept {
        return pcm_?pcm_->device_island_layout():SourceScDeviceIslandLayout{};
    }
    std::optional<compute::DispatchIndirectArgs>
    device_island_indirect_args() const noexcept {
        return pcm_?pcm_->device_island_indirect_args():std::nullopt;
    }
    std::size_t pcm_host_contact_upload_bytes() const noexcept {
        return pcm_ ? pcm_->last_host_contact_upload_bytes() : 0u;
    }
    std::size_t pcm_edge_history_copy_bytes() const noexcept {
        return pcm_ ? pcm_->last_edge_history_copy_bytes() : 0u;
    }
    std::size_t pcm_continuation_commands() const noexcept {
        return pcm_ ? pcm_->last_continuation_commands() : 0u;
    }
    std::size_t pcm_face_pair_scan_count() const noexcept {
        return pcm_ ? pcm_->last_face_pair_scan_count() : 0u;
    }
    std::size_t dy_pcm_device_contact_rows() const noexcept {
        return dy_distance_ ? dy_distance_->last_reused_pcm_contact_rows() : 0u;
    }
    std::size_t dy_pcm_contact_upload_bytes_saved() const noexcept {
        return dy_distance_ ? dy_distance_->last_omitted_pcm_upload_bytes() : 0u;
    }
    std::size_t dy_cpu_gu_geometry_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_cpu_gu_geometry_bytes() : 0u;
    }
    std::size_t dy_upload_span_count() const noexcept {
        return dy_distance_ ? dy_distance_->last_upload_spans() : 0u;
    }
    std::size_t dy_friction_anchor_upload_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_friction_anchor_upload_bytes() : 0u;
    }
    std::size_t dy_friction_writeback_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_friction_writeback_bytes() : 0u;
    }
    std::size_t dy_friction_prepared_rows() const noexcept {
        return dy_distance_ ? dy_distance_->last_friction_prepared_rows() : 0u;
    }
    std::int64_t dy_contact_prep_upload_saving() const noexcept {
        return dy_distance_ ? dy_distance_->last_contact_prep_upload_saving() : 0;
    }
    std::pair<std::size_t,std::size_t>
    dy_distance_transfer_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_transfer_bytes()
                            : std::pair<std::size_t,std::size_t>{};
    }
    void enqueue_gpu_pcm_box_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<native::GpuPcmBoxResult> results,
            std::uint32_t edge_capacity=0u) override {
        if(!pcm_)
            throw std::logic_error("Source Sc GPU PCM shaders not installed");
        pcm_->enqueue(requests,results,aabb_->capacity(),edge_capacity);
    }
    void finish_gpu_pcm_box_contacts() override {
        if(!pcm_)
            throw std::logic_error("Source Sc GPU PCM shaders not installed");
        pcm_->finish();
    }
    void enqueue_cpu_gu_touch_history(
            std::span<const native::GpuGuTouchRecord> touches) override {
        if(!pcm_)
            throw std::logic_error("Source Gu history requires PCM ownership");
        pcm_->enqueue_cpu_gu_touches(touches);
    }
    void enqueue_host_pcm_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<const native::GpuPcmBoxResult> results) override {
        if(!pcm_)
            throw std::logic_error("Source Sc GPU PCM is unavailable");
        pcm_->enqueue_host_contacts(requests,results);
    }
    bool supports_host_pcm_contacts() const noexcept override {
        return pcm_ && dy_distance_;
    }
    void upload_host_pcm_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<const native::GpuPcmBoxResult> results,
            std::uint32_t edge_capacity=0u) override {
        if(!pcm_ || !dy_distance_)
            throw std::logic_error("Source Sc/Dy primitive PCM is unavailable");
        pcm_->upload_host_contacts(requests,results,edge_capacity);
    }
    void authorize_device_integrated_pose(BodyId id) override {
        producer_->authorize_device_integrated_pose(id);
    }
    void mark_shape_bounds_changed(BodyId id,float offset) override {
        producer_->mark_shape_bounds_changed(id,offset);
    }
    void set_device_shape_pose_cache_enabled(BodyId id,bool enabled) override {
        producer_->set_device_shape_pose_cache_enabled(id,enabled);
    }
    void retain_rigid_actor(BodyId body) override {producer_->retain_rigid_actor(body);}
    void release_rigid_actor(BodyId body) override {
        producer_->release_rigid_actor(body);
        if(dy_distance_)dy_distance_->retire_rigid_actor(body);
    }
    void admit_plane_element(ShapeId shape,BodyId body) override {producer_->admit_plane_element(shape,body);}
    void admit_aggregate_shape(AggregateId aggregate,ShapeId shape,BodyId body,
                              native::GpuBroadphaseBox::Motion motion,float offset) override {
        producer_->admit_aggregate_shape(aggregate,shape,body,motion,offset);
    }
    void admit_shape(BodyId id,native::GpuBroadphaseBox::Motion motion,
                     float offset) override {
        producer_->admit_shape(id,motion,offset);
    }
    void admit_shape_element(
        ShapeId shape,BodyId actor,
        native::GpuBroadphaseBox::Motion motion,float offset) override {
        producer_->admit_shape_element(shape,actor,motion,offset);
    }
    void admit_plane(BodyId id) override {
        producer_->admit_plane(id);
    }
    AggregateId create_aggregate(
        std::uint32_t max_shapes,bool self_collision) override {
        return producer_->create_aggregate(max_shapes,self_collision);
    }
    void destroy_aggregate(AggregateId id) override {
        producer_->destroy_aggregate(id);
    }
    void remove_actor_from_aggregate(
        AggregateId aggregate,BodyId actor) override {
        producer_->remove_actor_from_aggregate(aggregate,actor);
    }
    void admit_aggregate_member(
        AggregateId aggregate,BodyId actor,
        native::GpuBroadphaseBox::Motion motion,float contact_offset) override {
        producer_->admit_aggregate_member(
            aggregate,actor,motion,contact_offset);
    }
    void retire_shape(BodyId id) override { producer_->retire_shape(id); }
    void set_shape_simulation_enabled(BodyId id,bool enabled) override {
        producer_->set_shape_simulation_enabled(id,enabled);
    }
    void set_actor_motion(BodyId id,native::GpuBroadphaseBox::Motion motion) override {
        producer_->set_actor_motion(id,motion);
    }
    void post_reports_cleanup() override { producer_->post_reports_cleanup(); }
    void commit_gpu_step() override {
        if(dy_distance_ && dy_distance_->has_queued_solve())
            throw std::logic_error(
                "Source Sc frame cannot commit before Dy numerical writeback");
        if(pcm_)pcm_->commit_step();
    }
    bool needs_gpu_contact_rebuild() const noexcept override {
        return pcm_ && pcm_->needs_contact_rebuild();
    }
    void invalidate_after_rollback() noexcept override {
        stream_.discard_unsubmitted_commands();
        producer_->invalidate_after_rollback();
        if(pcm_)pcm_->invalidate_after_rollback();
        if(dy_distance_) {
            // The GPU may still own a Dy output transfer even if its
            // result join threw after consuming the pending task token.
            // Retire that transfer and its ordered solver commands before
            // the next scene transaction can reuse their allocations.
            stream_.abandon_fetch();
            try { stream_.wait_at_fetch(); } catch (...) {}
            dy_distance_->invalidate_after_rollback();
        }
    }
    const SourceSapSceneFrameNativeResult& last_frame() const {
        return producer_->last_frame();
    }

private:
    friend struct SourceScDeviceIslandFixture;
    // Shared with the reference PhysX runtime; released after GPU owners.
    std::shared_ptr<physx_runtime::Foundation> foundation_;
    compute::Context context_;
    CudaDeviceAddressMap addresses_;
    SourceSapDmaNative stream_;
    SourcePhysxHeapRoots device_roots_;
    SourcePhysxDeviceRootCallback device_callback_;
    SourcePhysxMappedReportRoots pinned_roots_,mapped_roots_;
    bool aborted_=false;
    std::optional<physx::PxgHeapMemoryAllocatorManager> manager_;
    physx::PxgAllocatorDesc allocators_;
    std::unique_ptr<PxgSapSourceMembersNative> sap_;
    std::unique_ptr<PxgAabbSourceMembersNative> aabb_;

    AabbEncodingStage encoding_;
    SourceScBoxTransformBoundsNativeStage box_transforms_;
    physx::PxgCudaBuffer box_input_,box_cache_,box_geometry_,convex_vertices_;
    physx::PxgCudaBuffer convex_surface_;
    std::vector<std::uint32_t> convex_surface_words_;
    std::uint32_t convex_surface_bytes_=0u;
    std::uint32_t convex_surface_faces_base_=0u;
    std::uint32_t convex_surface_indices_base_=0u;
    SourceSapGpuBoxTransformOwnersNative box_owners_;
    CudaAabbLaunchLowering lowering_;
    SourceSapCreatedNativeStage created_;
    SourceSapRadixNativeStage radix_;
    SourceSapBoxIndicesNativeStage indices_;
    SourceSapEndpointHistogramsNativeStage histograms_;
    SourceFillD32NativeStage fill_;
    SourceRegionCreateNativeStage region_;
    SourceRegionHistogramsNativeStage region_histograms_;
    SourceActiveRegionListsNativeStage active_;
    SourceOverlapWorkNativeStage overlap_;
    SourceOverlapPrefixNativeStage prefix_;
    SourceNewBoundsPairsNativeStage newbounds_pairs_;
    SourceNewBoundsSequenceNativeStage newbounds_;
    SourceRegionNewBoundsNativeStage regions_;
    SourceSapReportNativeStage reports_;
    SourceSapRemovedNativeStage removed_;
    SourceSapUpdatedNativeStage updated_;
    SourceSapIncrementalCountsNativeStage comparisons_;
    SourceSapIncrementalPairsNativeStage incremental_;
    SourceClearNewFlagsNativeStage clear_;
    SourceSapUpdateNativeStages stages_;
    physx::Bp::BpFilter filter_;
    std::unique_ptr<SourceSapAggregateOwnedStagesNative> aggregate_stages_;
    std::unique_ptr<SourceScGpuBroadphaseProducerNative> producer_;
    std::unique_ptr<SourceScGpuPcmNative> pcm_;
    std::unique_ptr<SourceDyDistanceNativeStage> dy_distance_;
    std::uint64_t dy_distance_dispatch_count_=0;
};

} // namespace fyzz::physx_gpu
