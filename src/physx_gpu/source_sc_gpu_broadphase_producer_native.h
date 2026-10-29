// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_gpu/source_sc_scene_elements_native.h"

#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Native Scene's GPU broadphase continuation over the original source
// PxgAABBManager/PxgCudaBroadPhaseSap owners. Sc retains the actor->element
// handle association and authors all AABB admission/removal/changed bits.
// This producer owns only transient ordered handles and fetched pair streams;
// it allocates no GPU histories, shape IDs, alternative reports or scheduler.
class SourceScGpuBroadphaseProducerNative final
    : public native::GpuBroadphaseProducer {
public:
    SourceScGpuBroadphaseProducerNative(
        PxgAabbSourceMembersNative& aabb,
        PxgSapSourceMembersNative& sap,
        SourceSapDmaNative& stream,
        const SourceSapUpdateNativeStages& stages,
        const CudaDeviceAddressMap& addresses,
        compute::Context& context,
        const physx::Bp::BpFilter& filter,
        const SourceFillD32NativeStage& bitmap_fill,
        SourceSapMappedReportView& mapped_reports,
        const std::vector<physx::Bp::ShapeHandle>& sc_actor_element_ids)
        : aabb_(aabb), sap_(sap), stream_(stream), stages_(stages),
          addresses_(addresses), context_(context), filter_(filter),
          bitmap_fill_(bitmap_fill), mapped_reports_(mapped_reports),
          sc_actor_element_ids_(&sc_actor_element_ids) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument(
                "Source Sc GPU broadphase stream has a different device owner");
    }

    // Native Scene mode: retain original Sc ElementSim identity and
    // ObjectIDTracker through the complete source GPU report lifecycle.
    SourceScGpuBroadphaseProducerNative(
        PxgAabbSourceMembersNative& aabb,
        PxgSapSourceMembersNative& sap,
        SourceSapDmaNative& stream,
        const SourceSapUpdateNativeStages& stages,
        const CudaDeviceAddressMap& addresses,
        compute::Context& context,
        const physx::Bp::BpFilter& filter,
        const SourceFillD32NativeStage& bitmap_fill,
        SourceSapMappedReportView& mapped_reports)
        : aabb_(aabb),sap_(sap),stream_(stream),stages_(stages),
          addresses_(addresses),context_(context),filter_(filter),
          bitmap_fill_(bitmap_fill),mapped_reports_(mapped_reports),
          source_sc_elements_(std::make_unique<SourceScSceneElementsNative>(aabb)) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument(
                "Source Sc GPU broadphase stream has a different device owner");
        sc_actor_element_ids_=&source_sc_elements_->handles();
    }

    native::GpuBroadphaseReports advance(
        std::span<const native::GpuBroadphaseBox> live_boxes,
        bool force_update, bool /* geometry_dirty */) override {
        if(invalidated_)
            throw std::logic_error(
                "Source GPU SAP histories require reconstruction after Sc rollback");
        // Resolve Sc-owned element handles without allocating a second
        // identity table. The AABB manager's live volume mapping is the
        // authority, including after post-report ID recycling.
        const auto source_actors=aabb_.volume_to_actor();
        handles_.clear();
        handles_.reserve(live_boxes.size());
        for(const auto& box:live_boxes) {
            if(box.id>=sc_actor_element_ids_->size())
                throw std::out_of_range("Source Sc actor has no element handle");
            const auto handle=(*sc_actor_element_ids_)[box.id];
            if(handle>=source_actors.size() ||
               source_actors[handle]!=box.id)
                throw std::invalid_argument(
                    "Source Sc element handle does not match AABB volume owner");
            handles_.push_back(handle);
        }
        // The owner already authored its changed bitmap. A generic scene
        // dirtiness flag must not turn sparse source updates into a full
        // bounds-upload pass. Only the original mForceUpdate flag is raised.
        if(force_update)sap_.force_update()=true;
        const SourceSapUpdateNativeFrame source_frame{};
        try {
            last_frame_=advance_source_sap_owned_scene_frame_native(
                aabb_,sap_,
                {live_boxes,handles_,filter_,source_frame,found_,lost_,
                 bitmap_fill_,mapped_reports_,false},
                stream_,stages_,addresses_,context_);
        } catch(...) {
            // The source GPU submission and fetch may already have advanced.
            // Do not accept another frame against possibly divergent history.
            invalidated_=true;
            throw;
        }
        return {found_,lost_,false};
    }

    void mark_shape_bounds_changed(
        BodyId actor,float contact_offset) override {
        if(invalidated_)
            throw std::logic_error(
                "Source GPU SAP histories require reconstruction after Sc rollback");
        // The Sc element handle, not the actor ID, addresses the native
        // AABB manager's changed bitmap and contact-distance array.
        if(actor>=sc_actor_element_ids_->size())
            throw std::out_of_range("Source Sc shape has no element handle");
        // The owned Sc ElementSim survives removal from broadphase.
        // It retains shape metadata even after AABB actor identity was
        // retired by a previous completed fetch.
        if(source_sc_elements_) {
            source_sc_elements_->mark_changed(actor,contact_offset);
            return;
        }
        const auto handle=(*sc_actor_element_ids_)[actor];
        const auto owners=aabb_.volume_to_actor();
        if(handle>=owners.size() || owners[handle]!=actor)
            throw std::invalid_argument(
                "Source Sc edited shape has no matching AABB volume");
        aabb_.mark_changed(handle,contact_offset);
    }

    void set_shape_motion(BodyId actor,
                          native::GpuBroadphaseBox::Motion motion) override {
        require_scene_elements().set_motion(actor,motion);
    }

    void admit_shape(BodyId actor,
        native::GpuBroadphaseBox::Motion motion,float contact_distance) override {
        require_scene_elements().admit(actor,motion,contact_distance);
    }
    void retire_shape(BodyId actor) override {
        require_scene_elements().retire(actor);
    }
    void set_shape_simulation_enabled(BodyId actor,bool enabled) override {
        require_scene_elements().set_simulation_enabled(actor,enabled);
    }
    void post_reports_cleanup() override {
        if(source_sc_elements_)source_sc_elements_->post_reports_cleanup();
    }

    physx::Bp::ShapeHandle source_element_handle(BodyId actor) const {
        if(!source_sc_elements_)
            throw std::logic_error("Source Sc element ownership is borrowed");
        return source_sc_elements_->handle(actor);
    }
    bool source_element_deleted(physx::Bp::ShapeHandle handle) const {
        if(!source_sc_elements_)
            throw std::logic_error("Source Sc element ownership is borrowed");
        return source_sc_elements_->element_deleted(handle);
    }

    void invalidate_after_rollback() noexcept override {
        invalidated_=true;
    }

    const SourceSapSceneFrameNativeResult& last_frame() const {
        if(!last_frame_)
            throw std::logic_error("Source Sc GPU broadphase has no fetched frame");
        return *last_frame_;
    }

private:
    SourceScSceneElementsNative& require_scene_elements() {
        if(invalidated_)
            throw std::logic_error(
                "Source GPU SAP histories require reconstruction after Sc rollback");
        if(!source_sc_elements_)
            throw std::logic_error("Source Sc element admission is borrowed");
        return *source_sc_elements_;
    }

    PxgAabbSourceMembersNative& aabb_;
    PxgSapSourceMembersNative& sap_;
    SourceSapDmaNative& stream_;
    const SourceSapUpdateNativeStages& stages_;
    const CudaDeviceAddressMap& addresses_;
    compute::Context& context_;
    const physx::Bp::BpFilter& filter_;
    const SourceFillD32NativeStage& bitmap_fill_;
    SourceSapMappedReportView& mapped_reports_;
    std::unique_ptr<SourceScSceneElementsNative> source_sc_elements_;
    const std::vector<physx::Bp::ShapeHandle>* sc_actor_element_ids_=nullptr;
    std::vector<physx::Bp::ShapeHandle> handles_;
    std::vector<native::Pair> found_,lost_;
    std::optional<SourceSapSceneFrameNativeResult> last_frame_;
    bool invalidated_=false;
};

} // namespace fyzz::physx_gpu
