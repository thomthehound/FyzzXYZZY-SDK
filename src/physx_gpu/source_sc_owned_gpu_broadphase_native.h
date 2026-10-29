// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sc_gpu_broadphase_producer_native.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_translate_aabbs_native.h"
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sapbox_indices_native.h"
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
#include <span>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// The source GPU stage order is the same for SPIR-V and DXIL. A production
// shader provider supplies compiled stages; the scene never invokes DXC
// while stepping or silently selects a CPU broadphase.
using SourceSapShaderBinaries = std::array<compute::ShaderBinary, 34>;

class SourceScOwnedGpuBroadphaseNative final : public native::GpuBroadphaseProducer {
    struct FoundationOwner {
        struct Errors final : physx::PxErrorCallback {
            void reportError(physx::PxErrorCode::Enum,
                             const char*,const char*,int) override {}
        };
        physx::PxDefaultAllocator allocator;
        Errors errors;
        physx::PxFoundation* instance=nullptr;
        FoundationOwner() {
            instance=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
            if(!instance)
                throw std::runtime_error("Native PhysX Foundation creation failed");
        }
        ~FoundationOwner() { instance->release(); }
    };

    static FoundationOwner& foundation() {
        static FoundationOwner owner;
        return owner;
    }
    static void abort_source(void* state) noexcept {
        *static_cast<bool*>(state)=true;
    }

public:
    explicit SourceScOwnedGpuBroadphaseNative(
        compute::Backend backend,SourceSapShaderBinaries shaders,
        std::uint32_t max_pairs=32768u,
        std::uint32_t max_aggregate_pairs=32768u)
        : foundation_(foundation()),
          context_(backend),
          stream_(context_,addresses_),
          device_roots_(context_,addresses_),
          device_callback_(device_roots_,&stream_),
          pinned_roots_(context_,addresses_,stream_),
          mapped_roots_(context_,addresses_,stream_),
          manager_(std::make_unique<physx::PxgHeapMemoryAllocatorManager>(
              1u<<20u,device_callback_,pinned_roots_,mapped_roots_,
              abort_source,&aborted_)),
          allocators_(physx::PxgCreateAllocatorDesc(*manager_)),
          sap_(std::make_unique<PxgSapSourceMembersNative>(
              allocators_,max_pairs,max_aggregate_pairs)),
          aabb_(std::make_unique<PxgAabbSourceMembersNative>(allocators_)),
          encoding_(context_,std::move(shaders[0])),
          lowering_(addresses_,encoding_),
          created_(context_,std::move(shaders[1])),
          radix_(context_,std::move(shaders[2]),std::move(shaders[3]),
                 std::move(shaders[4]),std::move(shaders[5])),
          indices_(context_,std::move(shaders[6])),
          histograms_(context_,std::move(shaders[7]),std::move(shaders[8])),
          fill_(context_,std::move(shaders[9])),
          region_(context_,std::move(shaders[10]),fill_),
          region_histograms_(
              context_,std::move(shaders[11]),std::move(shaders[12]),
              std::move(shaders[13])),
          active_(context_,std::move(shaders[14]),std::move(shaders[15]),
                  std::move(shaders[16]),std::move(shaders[17])),
          overlap_(context_,std::move(shaders[18])),
          prefix_(context_,std::move(shaders[19]),std::move(shaders[20])),
          newbounds_pairs_(context_,std::move(shaders[21])),
          newbounds_(overlap_,prefix_,newbounds_pairs_),
          regions_(region_,region_histograms_,active_,newbounds_),
          reports_(context_,std::move(shaders[22]),std::move(shaders[23]),
                   std::move(shaders[24]),std::move(shaders[25]),
                   std::move(shaders[26])),
          removed_(context_,std::move(shaders[27]),std::move(shaders[28])),
          updated_(context_,std::move(shaders[29])),
          comparisons_(context_,std::move(shaders[30]),
                       std::move(shaders[31])),
          incremental_(context_,std::move(shaders[32])),
          clear_(context_,std::move(shaders[33])),
          stages_{lowering_,removed_,updated_,radix_,histograms_,
                  comparisons_,incremental_,indices_,created_,regions_,
                  clear_,reports_},
          filter_(false,false),
          producer_(std::make_unique<SourceScGpuBroadphaseProducerNative>(
              *aabb_,*sap_,stream_,stages_,addresses_,context_,filter_,
              fill_,mapped_roots_)) {}

    SourceScOwnedGpuBroadphaseNative(const SourceScOwnedGpuBroadphaseNative&)=delete;
    SourceScOwnedGpuBroadphaseNative& operator=(
        const SourceScOwnedGpuBroadphaseNative&)=delete;

    ~SourceScOwnedGpuBroadphaseNative() override {
        producer_.reset();
        // Original report-retirement commands remain on the source stream
        // after Sc publication. Complete them before releasing SAP/AABB
        // allocations, pipeline objects or the owning native Context.
        stream_.wait_at_fetch();
        aabb_.reset();
        sap_.reset();
        manager_->flushDeferredDeallocs();
        manager_.reset();
        device_roots_.close(&stream_);
    }

    native::GpuBroadphaseReports advance(
        std::span<const native::GpuBroadphaseBox> live,
        bool force_update,bool geometry_dirty) override {
        if(aborted_)
            throw std::runtime_error("Original GPU SAP heap aborted");
        return producer_->advance(live,force_update,geometry_dirty);
    }
    void mark_shape_bounds_changed(BodyId id,float offset) override {
        producer_->mark_shape_bounds_changed(id,offset);
    }
    void admit_shape(BodyId id,native::GpuBroadphaseBox::Motion motion,
                     float offset) override {
        producer_->admit_shape(id,motion,offset);
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
    }
    const SourceSapSceneFrameNativeResult& last_frame() const {
        return producer_->last_frame();
    }

private:
    // The Foundation is shared across independent native GPU scenes.
    // References and root allocators outlive every source buffer and stage.
    FoundationOwner& foundation_;
    compute::Context context_;
    CudaDeviceAddressMap addresses_;
    SourceSapDmaNative stream_;
    SourcePhysxHeapRoots device_roots_;
    SourcePhysxDeviceRootCallback device_callback_;
    SourcePhysxMappedReportRoots pinned_roots_,mapped_roots_;
    bool aborted_=false;
    std::unique_ptr<physx::PxgHeapMemoryAllocatorManager> manager_;
    physx::PxgAllocatorDesc allocators_;
    std::unique_ptr<PxgSapSourceMembersNative> sap_;
    std::unique_ptr<PxgAabbSourceMembersNative> aabb_;

    AabbEncodingStage encoding_;
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
    std::unique_ptr<SourceScGpuBroadphaseProducerNative> producer_;
};

} // namespace fyzz::physx_gpu
