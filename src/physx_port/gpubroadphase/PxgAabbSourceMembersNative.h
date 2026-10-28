// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_single_volume_scene_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_port/common/src/CmPinnableArray.h"
#include "physx_port/common/src/CmPinnableBitMap.h"
#include "BpAABBManagerBase.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"

#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Persistent original PxgAABBManager/AABBManagerBase member storage for
// single-actor GPU broadphase volumes. The original BoundsArray, pinned
// group/distance/VolumeData arrays, PinnableBitMaps, PinnableArray handle
// lists and PxgCudaBuffers have ONE owner for the lifetime of a scene.
// Source shape handles originate at Sc/Bp admission, not from a second
// SAP allocator. Aggregate admission is intentionally unsupported.
class PxgAabbSourceMembersNative final {
public:
    explicit PxgAabbSourceMembersNative(
        physx::PxgAllocatorDesc& allocators,physx::PxU32 initial_shapes=0u)
        : bounds_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          distances_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          groups_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          volumes_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          added_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          removed_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          changed_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregated_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          created_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          retired_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          updated_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          added_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          removed_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          changed_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregated_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          volumes_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE) {
        if(initial_shapes)reserve_shapes(initial_shapes);
    }
    PxgAabbSourceMembersNative(const PxgAabbSourceMembersNative&)=delete;
    PxgAabbSourceMembersNative& operator=(const PxgAabbSourceMembersNative&)=delete;

    void reserve_shapes(physx::PxU32 count) {
        if(count<=bounds_.size())return;
        if(count>=BP_INVALID_BP_HANDLE-1u)
            throw std::out_of_range("Source AABB volume capacity exceeds PhysX handle range");
        // The source BoundsArray grows to a power of two and retains a
        // SIMD-readable extra slot. Its size is the Bp update capacity.
        if(!bounds_.initEntry(count-1u))
            throw std::bad_alloc();
        const auto capacity=bounds_.size();
        const auto prior=volumes_.size();
        groups_.resize(capacity);
        volumes_.resize(capacity);
        distances_.resize(capacity);
        volume_to_actor_.resize(capacity,
            std::numeric_limits<BodyId>::max());
        if(!added_.resize(capacity) || !removed_.resize(capacity) ||
           !changed_.resize(capacity) || !aggregated_.resize(capacity))
            throw std::bad_alloc();
        for(auto i=prior;i<capacity;++i) {
            volumes_[i].reset();
            groups_[i]=physx::Bp::FilterGroup::eINVALID;
            distances_[i]=0.0f;
        }
        persistent_state_changed_=true;
    }

    SourceSapSingleVolumeMembersNative single_volumes() noexcept {
        return {added_,removed_,changed_,aggregated_,
            {volumes_.begin(),volumes_.size()},
            {groups_.begin(),groups_.size()},
            {distances_.begin(),distances_.size()},
            persistent_state_changed_,gpu_state_changed_};
    }
    SourceSapVolumeActorLifetimeNative actor_lifetime() noexcept {
        return {volume_to_actor_,{volumes_.begin(),volumes_.size()}};
    }
    std::span<const BodyId> volume_to_actor() const noexcept {
        return volume_to_actor_;
    }
    SourceSapSceneBounds scene_bounds() noexcept {
        return {{bounds_.begin(),bounds_.size()},
            {distances_.begin(),distances_.size()},
            {groups_.begin(),groups_.size()}};
    }

    void admit(physx::Bp::ShapeHandle handle,BodyId actor,
               physx::PxReal contact_distance,
               physx::Bp::FilterGroup::Enum group,void* user_data) {
        if(!user_data)
            throw std::invalid_argument("Source AABB Sc volume has no ElementSim identity");
        if(handle>=BP_INVALID_BP_HANDLE-1u)
            throw std::out_of_range("Source AABB shape handle is invalid");
        reserve_shapes(handle+1u);
        add_source_sap_single_volume_actor_native(
            single_volumes(),actor_lifetime(),handle,
            contact_distance,group,user_data,actor);
    }
    bool remove(physx::Bp::ShapeHandle handle) {
        return remove_source_sap_single_volume_actor_native(
            single_volumes(),actor_lifetime(),handle);
    }
    void mark_changed(physx::Bp::ShapeHandle handle,
                      physx::PxReal contact_distance) {
        mark_source_sap_single_volume_changed_native(
            single_volumes(),handle,contact_distance);
    }

    // Called after Sc has marked source dirty volumes and before the
    // original preBroadPhase task. The returned update borrows this
    // object's arrays until SAP's GPU fetch continuation finishes.
    physx::Bp::BroadPhaseUpdateData prepare(
        std::span<const native::GpuBroadphaseBox> live,
        std::span<const physx::Bp::ShapeHandle> handles,
        const physx::Bp::BpFilter& filter) {
        if(live.size()!=handles.size())
            throw std::invalid_argument(
                "Source SAP live actor and original volume handle counts disagree");
        // Sc owns actor identity, while PxgAABBManager owns the volume
        // handle. Their persistent admission binding must agree before
        // changing any bounds, groups, handle lists, or GPU descriptors.
        for(std::size_t i=0;i<live.size();++i) {
            if(handles[i]>=volume_to_actor_.size() ||
               volume_to_actor_[handles[i]]!=live[i].id)
                throw std::invalid_argument(
                    "Source GPU volume handle no longer belongs to the supplied Sc actor");
        }
        auto update=prepare_source_sap_single_volume_scene_native(
            single_volumes(),{created_,retired_,updated_},live,handles,
            scene_bounds(),filter,bounds_.hasChanged(),bounds_scratch_);
        validate_source_sap_volume_actor_lifetime_native(
            actor_lifetime(),update);
        return update;
    }

    SourceSapAabbManagerPreUpdateNative pre_update(
        SourceSapPreBroadPhaseBuffers sap) noexcept {
        return {sap,source_sap_aabb_bitmap_native(added_device_,added_),
            source_sap_aabb_bitmap_native(removed_device_,removed_),
            source_sap_aabb_bitmap_native(aggregated_device_,aggregated_),
            source_sap_aabb_bitmap_native(changed_device_,changed_),
            volumes_device_,{volumes_.begin(),volumes_.size()},
            persistent_state_changed_,gpu_state_changed_,false};
    }
    SourceSapAabbReportBitmapNative added_report_bitmap() noexcept {
        return {added_device_.getDevicePtr(),added_device_.getSize(),
            {added_.getWords(),added_.getWordCount()}};
    }
    SourceSapAabbReportBitmapNative removed_report_bitmap() noexcept {
        return {removed_device_.getDevicePtr(),removed_device_.getSize(),
            {removed_.getWords(),removed_.getWordCount()}};
    }

    // Called AFTER successful source GPU fetch and the original bitmap
    // D32 retirement commands have been enqueued on the same stream.
    // The GPU frame path owns history rotation and pair publication;
    // there is no second clock, stream wait or pair state in this class.
    void complete_fetch(const physx::Bp::BroadPhaseUpdateData& update) noexcept {
        clear_source_sap_single_volume_report_bitmaps_native(single_volumes());
        retire_source_sap_volume_actor_lifetime_native(actor_lifetime(),update);
        changed_.clear();
        persistent_state_changed_=false;
        gpu_state_changed_=false;
        bounds_.resetChangedState();
    }

    physx::PxU32 capacity() const noexcept {return bounds_.size();}
    const physx::Bp::BoundsArray& source_bounds() const noexcept {return bounds_;}
    physx::Cm::PinnableBitMap& source_added() noexcept {return added_;}
    physx::Cm::PinnableBitMap& source_removed() noexcept {return removed_;}
    physx::Cm::PinnableBitMap& source_changed() noexcept {return changed_;}

private:
    physx::Bp::BoundsArray bounds_;
    physx::Cm::PinnableArray<physx::PxReal> distances_;
    physx::Cm::PinnableArray<physx::Bp::FilterGroup::Enum> groups_;
    physx::Cm::PinnableArray<physx::Bp::VolumeData> volumes_;
    physx::Cm::PinnableBitMap added_,removed_,changed_,aggregated_;
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle> created_,retired_,updated_;
    physx::PxgCudaBuffer added_device_,removed_device_,changed_device_;
    physx::PxgCudaBuffer aggregated_device_,volumes_device_;
    std::vector<BodyId> volume_to_actor_;
    SourceSapSceneBoundsScratch bounds_scratch_;
    bool persistent_state_changed_=true,gpu_state_changed_=false;
};

} // namespace fyzz::physx_gpu
