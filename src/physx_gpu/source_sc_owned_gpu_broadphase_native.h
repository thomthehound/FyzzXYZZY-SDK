// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sc_gpu_broadphase_producer_native.h"
#include "physx_gpu/source_sap_aggregate_owned_stages_native.h"
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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

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
        std::optional<compute::ShaderBinary> dy_shader=std::nullopt)
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
          box_owners_{box_transforms_,box_input_,box_cache_,box_geometry_},
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
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::Candidates),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaReset),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaCompact),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::EpaContact),
                take_pcm_shader(*pcm_shaders,NativeGpuPcmStage::ContactStream));
        if(dy_shader)
            dy_distance_=std::make_unique<SourceDyDistanceNativeStage>(
                context_,addresses_,stream_,allocators_.deviceAlloc,
                std::move(*dy_shader));
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
    void solve_gpu_dy_distance(
        std::span<native::GpuDyDistanceBody> bodies,
        std::span<native::GpuDyDistanceRow> rows,
        std::span<const native::GpuDyDistanceDescriptor> descriptors,
        std::span<const std::uint32_t> partitions,
        std::span<const std::uint32_t> ordinals,
        std::span<native::GpuDyDistanceTask> tasks) override {
        if(!dy_distance_)
            throw std::logic_error("Source GPU Dy distance shader not installed");
        dy_distance_->solve(bodies,rows,descriptors,partitions,ordinals,tasks);
        ++dy_distance_dispatch_count_;
    }
    std::uint64_t dy_distance_dispatch_count() const noexcept {
        return dy_distance_dispatch_count_;
    }
    std::pair<std::size_t,std::size_t>
    dy_distance_transfer_bytes() const noexcept {
        return dy_distance_ ? dy_distance_->last_transfer_bytes()
                            : std::pair<std::size_t,std::size_t>{};
    }
    void generate_gpu_pcm_box_contacts(
            std::span<const native::GpuPcmBoxPair> requests,
            std::span<native::GpuPcmBoxResult> results) override {
        if(!pcm_)
            throw std::logic_error("Source Sc GPU PCM shaders not installed");
        pcm_->generate(requests,results,aabb_->capacity());
    }
    void mark_shape_bounds_changed(BodyId id,float offset) override {
        producer_->mark_shape_bounds_changed(id,offset);
    }
    void admit_shape(BodyId id,native::GpuBroadphaseBox::Motion motion,
                     float offset) override {
        producer_->admit_shape(id,motion,offset);
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
    void set_shape_motion(BodyId id,native::GpuBroadphaseBox::Motion motion) override {
        producer_->set_shape_motion(id,motion);
    }
    void post_reports_cleanup() override { producer_->post_reports_cleanup(); }
    void invalidate_after_rollback() noexcept override {
        producer_->invalidate_after_rollback();
        if(pcm_)pcm_->invalidate_after_rollback();
    }
    const SourceSapSceneFrameNativeResult& last_frame() const {
        return producer_->last_frame();
    }

private:
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
    physx::PxgCudaBuffer box_input_,box_cache_,box_geometry_;
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
