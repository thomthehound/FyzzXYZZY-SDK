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
        SourceSapMappedReportView& mapped_reports,
        const SourceSapAggregateStagesNative* aggregate_stages=nullptr,
        const SourceSapGpuBoxTransformOwnersNative* gpu_box_owners=nullptr)
        : aabb_(aabb),sap_(sap),stream_(stream),stages_(stages),
          addresses_(addresses),context_(context),filter_(filter),
          bitmap_fill_(bitmap_fill),mapped_reports_(mapped_reports),
          aggregate_stages_(aggregate_stages),
          gpu_box_owners_(gpu_box_owners),
          source_sc_elements_(std::make_unique<SourceScSceneElementsNative>(aabb)) {
        if(!stream_.bound_to(context_,addresses_))
            throw std::invalid_argument(
                "Source Sc GPU broadphase stream has a different device owner");
        sc_actor_element_ids_=&source_sc_elements_->handles();
    }

    native::GpuBroadphaseReports advance(
        std::span<const native::GpuBroadphaseBox> live_boxes,
        bool force_update, bool /* geometry_dirty */) override {
        // A rejected Sc/Dy frame may have advanced the source GPU
        // endpoint histories before Sc's transaction was rolled back.
        // Drain that same original stream, discard only source SAP
        // history counters, and republish all still-admitted AABB
        // handles on the ordinary create-only GPU path.
        bool rebuilt=false;
        if(invalidated_) {
            stream_.wait_at_fetch();
            aabb_.reset_for_source_rebuild();
            sap_.reset_for_source_rebuild();
            rebuilt=true;
        }
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
                 bitmap_fill_,mapped_reports_,false,aggregate_stages_,
                 gpu_box_owners_},
                stream_,stages_,addresses_,context_);
        } catch(...) {
            // The source GPU submission and fetch may already have advanced.
            // Do not accept another frame against possibly divergent history.
            invalidated_=true;
            throw;
        }
        invalidated_=false;
        return {found_,lost_,false,rebuilt};
    }

    void mark_shape_bounds_changed(
        BodyId actor,float contact_offset) override {
        // Authored Sc shape edits remain valid after a rejected frame.
        // The NEXT GPU fetch reconstructs the original SAP histories
        // from these authoritative live actor bounds and contact data.
        // Reject stale GPU execution, not source Sc shape authoring.
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
    AggregateId create_aggregate(
        std::uint32_t max_shapes,bool self_collision) override {
        if(!aggregate_stages_)
            throw std::logic_error(
                "Native Sc aggregate admission requires complete source GPU kernels");
        return require_scene_elements().create_aggregate(
            max_shapes,physx::Bp::getFilterGroup_Statics(),
            self_collision?1u:0u);
    }
    void destroy_aggregate(AggregateId id) override {
        require_scene_elements().destroy_aggregate(id);
    }
    void remove_actor_from_aggregate(
        AggregateId aggregate,BodyId actor) override {
        require_scene_elements().remove_actor_from_aggregate(aggregate,actor);
    }
    void admit_aggregate_member(
        AggregateId aggregate,BodyId actor,
        native::GpuBroadphaseBox::Motion motion,
        float contact_distance) override {
        if(!aggregate_stages_)
            throw std::logic_error(
                "Native Sc aggregate member requires complete source GPU kernels");
        require_scene_elements().admit_aggregate_member(
            aggregate,actor,motion,contact_distance);
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
        // Sc::ActorSim/ElementSim lifecycle remains authoritative even
        // while old GPU SAP endpoint histories await reconstruction.
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
    const SourceSapAggregateStagesNative* aggregate_stages_=nullptr;
    const SourceSapGpuBoxTransformOwnersNative* gpu_box_owners_=nullptr;
    std::unique_ptr<SourceScSceneElementsNative> source_sc_elements_;
    const std::vector<physx::Bp::ShapeHandle>* sc_actor_element_ids_=nullptr;
    std::vector<physx::Bp::ShapeHandle> handles_;
    std::vector<native::Pair> found_,lost_;
    std::optional<SourceSapSceneFrameNativeResult> last_frame_;
    bool invalidated_=false;
};

} // namespace fyzz::physx_gpu
