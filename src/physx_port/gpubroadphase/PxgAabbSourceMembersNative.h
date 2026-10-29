// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_single_volume_scene_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_member_history_native.h"
#include "physx_port/common/src/CmPinnableArray.h"
#include "physx_port/common/src/CmPinnableBitMap.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
#include "physx_port/lowlevelaabb/include/BpVolumeData.h"
#include "physx_port/lowlevelaabb/include/BpSourceBoundsArrayNative.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "foundation/PxMemory.h"
#include "physx_port/gpubroadphase/include/PxgAggregate.h"
#include "physx_port/common/src/CmIDPool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Persistent original PxgAABBManager/AABBManagerBase member storage for
// GPU broadphase volumes. The original BoundsArray, pinned
// group/distance/VolumeData arrays, PinnableBitMaps, PinnableArray handle
// lists and PxgCudaBuffers have ONE owner for the lifetime of a scene.
// Source shape handles originate at Sc/Bp admission, not from a second
// SAP allocator. Aggregate members use original PxgAggregate records and
// deferred aggregate IDs; GPU aggregate collision/report publication remains
// gated until the source aggregate device buffers and tasks are connected.
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
          added_aggregate_members_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          removed_aggregate_members_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          added_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          removed_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          changed_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregated_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          volumes_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_slots_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_members_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          added_aggregate_members_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          removed_aggregate_members_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          dirty_aggregates_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          dirty_aggregate_indices_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE) {
        if(initial_shapes)reserve_shapes(initial_shapes);
    }
    ~PxgAabbSourceMembersNative() {
        for(auto& aggregate:aggregates_)aggregate.reset();
    }
    PxgAabbSourceMembersNative(const PxgAabbSourceMembersNative&)=delete;
    PxgAabbSourceMembersNative& operator=(const PxgAabbSourceMembersNative&)=delete;

    void reserve_shapes(physx::PxU32 count) {
        if(count<=bounds_.size())return;
        if(count>=BP_INVALID_BP_HANDLE-1u)
            throw std::out_of_range("Source AABB volume capacity exceeds PhysX handle range");
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
    // PxgAABBManager::createAggregate. An empty aggregate has no BP
    // endpoint; its root enters SAP only when its first member arrives.
    physx::Bp::AggregateHandle create_aggregate(
        physx::Bp::ShapeHandle root,
        physx::Bp::FilterGroup::Enum group,
        void* user_data,physx::PxU32 max_shapes,
        physx::PxU32 env_id=PX_INVALID_U32,
        physx::PxU32 filter_hint=0u) {
        if(!max_shapes || group==physx::Bp::FilterGroup::eINVALID ||
           !user_data || (reinterpret_cast<std::uintptr_t>(user_data)&3u) ||
           root>=BP_INVALID_BP_HANDLE-1u)
            throw std::invalid_argument("Original source aggregate root admission invalid");
        reserve_shapes(root+1u);
        if(groups_[root]!=physx::Bp::FilterGroup::eINVALID)
            throw std::logic_error("Source aggregate root already admitted");
        const auto id=aggregate_ids_.getNewID();
        if(id>=std::numeric_limits<physx::PxU32>::max()/2u)
            throw std::out_of_range("PhysX aggregate ID exceeds VolumeData encoding");
        if(aggregates_.size()<=id)aggregates_.resize(2u*id+1u);
        if(aggregate_max_shapes_.size()<=id)
            aggregate_max_shapes_.resize(2u*id+1u);
        auto& aggregate=aggregates_[id];
        aggregate.reset();
        aggregate.updateBoundIndices=PX_ALLOCATE(
            physx::PxU32,max_shapes,"updateBoundIndices");
        if(!aggregate.updateBoundIndices)throw std::bad_alloc();
        aggregate.mIndex=root;
        aggregate.filterHint=filter_hint;
        aggregate.mEnvID=env_id;
        aggregate_max_shapes_[id]=max_shapes;
        // Original per-aggregate updateBoundIndices device allocation,
        // lowered to a stable non-overlapping slice of one source-owned
        // PxgCudaBuffer so native shaders can bind a single member view.
        if(aggregate_gpu_member_starts_.size()<=id) {
            aggregate_gpu_member_starts_.resize(2u*id+1u,0u);
            aggregate_gpu_member_reservations_.resize(2u*id+1u,0u);
        }
        if(aggregate_gpu_member_reservations_[id]<max_shapes) {
            if(max_shapes>std::numeric_limits<physx::PxU32>::max()-
                              aggregate_gpu_member_capacity_)
                throw std::out_of_range("Original aggregate GPU member address range exceeded");
            aggregate_gpu_member_starts_[id]=aggregate_gpu_member_capacity_;
            aggregate_gpu_member_reservations_[id]=max_shapes;
            aggregate_gpu_member_capacity_+=max_shapes;
        }
        aggregate_gpu_slot_count_=std::max(aggregate_gpu_slot_count_,id+1u);
        volumes_[root].setUserData(user_data);
        volumes_[root].setVolumeType(physx::Bp::ElementType::eSHAPE);
        volumes_[root].setAggregate(id);
        groups_[root]=group;
        distances_[root]=0.0f;
        bounds_.setBounds(physx::PxBounds3::empty(),root);
        added_.growAndReset(root);
        aggregated_.growAndReset(root);
        persistent_state_changed_=true;
        mark_aggregate_dirty(id);
        ++num_active_aggregates_;
        return id;
    }

    // PxgAABBManager::addBounds aggregate branch. Members are never
    // independent SAP boxes: the first member registers the root only.
    void admit_aggregate_member(
        physx::Bp::AggregateHandle id,
        physx::Bp::ShapeHandle handle,BodyId actor,
        physx::PxReal contact_distance,
        physx::Bp::FilterGroup::Enum group,void* user_data) {
        auto& aggregate=checked_aggregate(id);
        if(!user_data || (reinterpret_cast<std::uintptr_t>(user_data)&3u) ||
           group==physx::Bp::FilterGroup::eINVALID ||
           !std::isfinite(contact_distance) || contact_distance<0.0f ||
           handle==aggregate.mIndex ||
           handle>=BP_INVALID_BP_HANDLE-1u ||
           aggregate.size>=aggregate_max_shapes_[id])
            throw std::invalid_argument("Invalid original aggregate member admission");
        reserve_shapes(handle+1u);
        if(groups_[handle]!=physx::Bp::FilterGroup::eINVALID ||
           volume_to_actor_[handle]!=std::numeric_limits<BodyId>::max())
            throw std::logic_error("Source aggregate member handle already live");
        if(!aggregate.size) {
            if(removed_.boundedTest(aggregate.mIndex))
                removed_.boundedReset(aggregate.mIndex);
            else if(!added_.growAndSet(aggregate.mIndex))
                throw std::bad_alloc();
        } else if(!added_.boundedTest(aggregate.mIndex)) {
            if(!changed_.growAndSet(aggregate.mIndex))
                throw std::bad_alloc();
        }
        if(!aggregated_.growAndSet(handle))throw std::bad_alloc();
        auto& volume=volumes_[handle];
        volume.setUserData(user_data);
        volume.setVolumeType(physx::Bp::ElementType::eSHAPE);
        volume.setAggregated(id);
        groups_[handle]=group;
        distances_[handle]=contact_distance;
        volume_to_actor_[handle]=actor;
        aggregate.updateBoundIndices[aggregate.size++]=handle;
        // The source member ID can be readmitted before GPU fetch. Cancel
        // its pending deletion but retain new admission so the prior
        // integer AABB is invalidated without clearing its current bounds.
        removed_aggregate_members_.findAndReplaceWithLast(handle);
        added_aggregate_members_.pushBack(handle);
        mark_aggregate_dirty(id);
        persistent_state_changed_=true;
    }

    // AABBManagerBase::setContactDistance / source changed bitmap.
    // Aggregate member motion or contact-offset changes remain keyed by
    // the member's original shape handle. The existing GPU aggregate
    // dirty-bit stage will promote those bits to the root SAP endpoint.
    void mark_aggregate_member_changed(
        physx::Bp::ShapeHandle handle,physx::PxReal contact_distance) {
        if(handle>=volumes_.size() || !volumes_[handle].isAggregated() ||
           groups_[handle]==physx::Bp::FilterGroup::eINVALID ||
           !std::isfinite(contact_distance) || contact_distance<0.0f)
            throw std::invalid_argument("Original aggregate member changed while inactive");
        const auto id=volumes_[handle].getAggregateOwner();
        (void)checked_aggregate(id);
        if(!changed_.growAndSet(handle))throw std::bad_alloc();
        distances_[handle]=contact_distance;
        persistent_state_changed_=true;
    }

    // PxgAABBManager::removeBounds: swap-with-last source member removal.
    // Removing the final member cancels an unpublished root or retires
    // the previously published root at the normal BP fetch boundary.
    bool remove_aggregate_member(physx::Bp::ShapeHandle handle) {
        if(handle>=volumes_.size() || !volumes_[handle].isAggregated() ||
           groups_[handle]==physx::Bp::FilterGroup::eINVALID)
            throw std::invalid_argument("Original aggregate member is not live");
        const auto id=volumes_[handle].getAggregateOwner();
        auto& aggregate=checked_aggregate(id);
        physx::PxU32 i=0u;
        while(i<aggregate.size && aggregate.updateBoundIndices[i]!=handle)++i;
        if(i==aggregate.size)throw std::logic_error(
            "Source aggregate VolumeData differs from member array");
        aggregate.updateBoundIndices[i]=
            aggregate.updateBoundIndices[--aggregate.size];
        if(!aggregate.size) {
            if(added_.boundedTest(aggregate.mIndex))
                added_.boundedReset(aggregate.mIndex);
            else if(!removed_.growAndSet(aggregate.mIndex))
                throw std::bad_alloc();
            changed_.boundedReset(aggregate.mIndex);
        }else if(!removed_.boundedTest(aggregate.mIndex)) {
            if(!changed_.growAndSet(aggregate.mIndex))
                throw std::bad_alloc();
        }
        // Source PxgAABBManager cancels admission on same-frame removal;
        // otherwise the old and new GPU integer AABBs must be retired.
        if(!added_aggregate_members_.findAndReplaceWithLast(handle))
            removed_aggregate_members_.pushBack(handle);
        changed_.boundedReset(handle);
        aggregated_.boundedReset(handle);
        groups_[handle]=physx::Bp::FilterGroup::eINVALID;
        distances_[handle]=0.0f;
        volumes_[handle].reset();
        bounds_.setBounds(physx::PxBounds3::empty(),handle);
        volume_to_actor_[handle]=std::numeric_limits<BodyId>::max();
        mark_aggregate_dirty(id);
        persistent_state_changed_=true;
        return aggregate.size==0u;
    }

    // PxgAABBManager::destroyAggregate requires all member bounds
    // retired; ID reuse remains deferred until source post-fetch cleanup.
    void destroy_aggregate(physx::Bp::AggregateHandle id) {
        auto& aggregate=checked_aggregate(id);
        if(aggregate.size)
            throw std::logic_error("Cannot destroy original aggregate with live members");
        const auto root=aggregate.mIndex;
        added_.boundedReset(root);
        changed_.boundedReset(root);
        aggregated_.boundedReset(root);
        groups_[root]=physx::Bp::FilterGroup::eINVALID;
        distances_[root]=0.0f;
        volumes_[root].reset();
        bounds_.setBounds(physx::PxBounds3::empty(),root);
        aggregate.reset();
        aggregate_max_shapes_[id]=0u;
        aggregate_ids_.deferredFreeID(id);
        --num_active_aggregates_;
        persistent_state_changed_=true;
        mark_aggregate_dirty(id);
    }

    // PxgAABBManager::gpuDmaDataUp source dirty-index lowering.
    // Only dirty PxgAggregate records and their authoritative
    // updateBoundIndices are uploaded; reused aggregate handles keep
    // their previous device-member slice until a larger capacity is
    // requested. There is no second host membership registry.
    void queue_aggregate_gpu_dma_up(SourceSapDmaNative& stream) {
        if(dirty_aggregate_indices_.empty())return;
        const std::uint64_t record_bytes=
            std::uint64_t(aggregate_gpu_slot_count_)*
            sizeof(SourceAggregateGpuNativeRecord);
        const std::uint64_t member_bytes=
            std::uint64_t(aggregate_gpu_member_capacity_)*sizeof(physx::PxU32);
        if(record_bytes>std::numeric_limits<physx::PxU32>::max() ||
           member_bytes>std::numeric_limits<physx::PxU32>::max())
            throw std::out_of_range("Original aggregate GPU DMA size exceeded");
        aggregate_slots_device_.allocateCopyOldDataAsync(
            record_bytes,stream,PX_FL);
        aggregate_members_device_.allocateCopyOldDataAsync(
            member_bytes,stream,PX_FL);
        gpu_aggregate_record_staging_.resize(dirty_aggregate_indices_.size());
        for(physx::PxU32 i=0;i<dirty_aggregate_indices_.size();++i) {
            const auto id=dirty_aggregate_indices_[i];
            const auto& aggregate=aggregates_[id];
            gpu_aggregate_record_staging_[i]={
                aggregate.mIndex,aggregate_gpu_member_starts_[id],
                aggregate.size,0u};
            if(aggregate.size>aggregate_gpu_member_reservations_[id])
                throw std::logic_error(
                    "Source aggregate GPU member count exceeds reserved capacity");
        }
        auto& uploads=stream.upload_scratch();
        uploads.reserve(2u*dirty_aggregate_indices_.size());
        for(physx::PxU32 i=0;i<dirty_aggregate_indices_.size();++i) {
            const auto id=dirty_aggregate_indices_[i];
            const auto& aggregate=aggregates_[id];
            uploads.push_back({
                aggregate_slots_device_.getDevicePtr()+
                    std::uint64_t(id)*sizeof(SourceAggregateGpuNativeRecord),
                &gpu_aggregate_record_staging_[i],
                sizeof(SourceAggregateGpuNativeRecord)});
            if(aggregate.size)
                uploads.push_back({
                    aggregate_members_device_.getDevicePtr()+
                        std::uint64_t(aggregate_gpu_member_starts_[id])*
                        sizeof(physx::PxU32),
                    aggregate.updateBoundIndices,
                    std::size_t(aggregate.size)*sizeof(physx::PxU32)});
        }
        stream.queue_uploads(uploads);
    }

    // Source gpuDmaDataUp uploads added and deleted aggregate members from
    // the same host lists used by markAddedAndDeletedAggregatedBounds.
    // The native history invalidation dispatch follows the SAP update,
    // before aggregate collision-history kernels on the shared stream.
    void queue_aggregate_member_history_gpu_dma_up(SourceSapDmaNative& stream) {
        const auto added=added_aggregate_members_.size();
        const auto removed=removed_aggregate_members_.size();
        if(!added && !removed)return;
        const auto checked_bytes=[](physx::PxU32 count) {
            if(std::uint64_t(count)*sizeof(physx::PxU32)>
               std::numeric_limits<physx::PxU32>::max())
                throw std::out_of_range(
                    "Original aggregate member history DMA capacity exceeded");
            return std::size_t(count)*sizeof(physx::PxU32);
        };
        const auto added_bytes=checked_bytes(added);
        const auto removed_bytes=checked_bytes(removed);
        if(added_bytes)added_aggregate_members_device_.allocate(added_bytes,PX_FL);
        if(removed_bytes)removed_aggregate_members_device_.allocate(removed_bytes,PX_FL);
        auto& uploads=stream.upload_scratch();
        uploads.reserve(2u);
        if(added_bytes)uploads.push_back({
            added_aggregate_members_device_.getDevicePtr(),
            added_aggregate_members_.begin(),added_bytes});
        if(removed_bytes)uploads.push_back({
            removed_aggregate_members_device_.getDevicePtr(),
            removed_aggregate_members_.begin(),removed_bytes});
        stream.queue_uploads(uploads);
    }

    SourceAggregateMemberHistoryNativeBuffers aggregate_member_history_buffers(
        CudaDeviceAddressMap::Address new_bounds,
        CudaDeviceAddressMap::Address old_bounds) const noexcept {
        return {new_bounds,old_bounds,
            added_aggregate_members_device_.getDevicePtr(),
            removed_aggregate_members_device_.getDevicePtr(),
            bounds_.size(),added_aggregate_members_.size(),
            removed_aggregate_members_.size()};
    }

    // Borrow the original SAP bounds/contact arrays and this AABB owner's
    // changed bitmap after its normal preBroadPhase upload. The translated
    // aggregate views point to this owner's actual GPU heap allocations.
    SourceAggregateGpuNativeBuffers aggregate_gpu_buffers(
        CudaDeviceAddressMap::Address gpu_bounds,
        CudaDeviceAddressMap::Address gpu_contact_distances) const noexcept {
        return {gpu_bounds,gpu_contact_distances,
            aggregate_slots_device_.getDevicePtr(),
            aggregate_members_device_.getDevicePtr(),
            changed_device_.getDevicePtr(),
            bounds_.size(),aggregate_gpu_slot_count_,
            aggregate_gpu_member_capacity_,changed_.getWordCount(),
            aggregate_gpu_slot_count_};
    }

    const physx::PxgAggregate& source_aggregate(
        physx::Bp::AggregateHandle id) const {
        if(id>=aggregates_.size())throw std::out_of_range(
            "Original PxgAggregate slot unavailable");
        return aggregates_[id];
    }
    std::span<const physx::PxU32> dirty_aggregate_indices() const noexcept {
        return {dirty_aggregate_indices_.begin(),dirty_aggregate_indices_.size()};
    }
    void finish_aggregate_report_fetch() {
        for(const auto id:dirty_aggregate_indices())
            aggregates_[id].isNew=false;
        dirty_aggregate_indices_.resetOrClear();
        dirty_aggregates_.clear();
        added_aggregate_members_.resetOrClear();
        removed_aggregate_members_.resetOrClear();
        aggregate_ids_.processDeferredIds();
    }
    physx::Cm::PinnableBitMap& source_aggregated() noexcept {
        return aggregated_;
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
        // Aggregate root pairs require source aggregate expansion before
        // Sc publication. Even an empty aggregate must not acquire a
        // single-volume endpoint; keep this O(1) on ordinary scenes.
        if(num_active_aggregates_)
            throw std::logic_error(
                "Source aggregate GPU pair expansion not yet installed");
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
        const bool aggregate_slots=aggregate_gpu_slot_count_!=0u;
        return {sap,source_sap_aabb_bitmap_native(added_device_,added_),
            source_sap_aabb_bitmap_native(removed_device_,removed_),
            source_sap_aabb_bitmap_native(aggregated_device_,aggregated_),
            source_sap_aabb_bitmap_native(changed_device_,changed_),
            volumes_device_,{volumes_.begin(),volumes_.size()},
            persistent_state_changed_,gpu_state_changed_ || aggregate_slots,
            aggregate_slots};
    }

    // PxgAABBManager::updateBPFirstPass: validate the exact original
    // AABB scene arrays, enqueue gpuDmaDataUp for dirty aggregate members,
    // and return the borrowed source preBpUpdate_GPU descriptor. The
    // pre-update's original preBroadPhase uploads, followed by the union
    // and dirty-root kernels, all run on that same source stream.
    SourceSapAabbManagerPreUpdateNative aggregate_pre_update(
        const physx::Bp::BroadPhaseUpdateData& update,
        SourceSapPreBroadPhaseBuffers sap,
        const SourceAggregateGpuNativeStage& stage,
        const CudaDeviceAddressMap& addresses,
        SourceSapDmaNative& stream) {
        if(!aggregate_gpu_slot_count_)
            throw std::logic_error("No original aggregate slots for GPU pre-update");
        validate_pre_broadphase_native(update);
        if(update.getCapacity()!=bounds_.size() ||
           update.getAABBs()!=bounds_.begin() ||
           update.getContactDistance()!=distances_.begin() ||
           update.getGroups()!=groups_.begin())
            throw std::invalid_argument(
                "Aggregate GPU first pass does not borrow original AABB manager arrays");
        queue_aggregate_gpu_dma_up(stream);
        queue_aggregate_member_history_gpu_dma_up(stream);
        auto manager=pre_update(sap);
        manager.aggregate_stage=&stage;
        manager.aggregate_addresses=&addresses;
        manager.aggregate_slots_device=&aggregate_slots_device_;
        manager.aggregate_members_device=&aggregate_members_device_;
        manager.aggregate_slots_count=aggregate_gpu_slot_count_;
        manager.aggregate_member_capacity=aggregate_gpu_member_capacity_;
        return manager;
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
    void complete_fetch(const physx::Bp::BroadPhaseUpdateData& update) {
        clear_source_sap_single_volume_report_bitmaps_native(single_volumes());
        retire_source_sap_volume_actor_lifetime_native(actor_lifetime(),update);
        changed_.clear();
        persistent_state_changed_=false;
        gpu_state_changed_=false;
        bounds_.resetChangedState();
        finish_aggregate_report_fetch();
    }

    // Source GPU SAP history was invalidated after a rejected Sc/Dy
    // frame. Rebuild from the ORIGINAL AABB manager's admitted volumes:
    // no additional actor/handle pool and no CPU geometry overlap pass.
    // Every still-live source volume becomes a fresh created handle.
    // The next Sc geometry publication refreshes all authored bounds.
    void reset_for_source_rebuild() {
        if(num_active_aggregates_)
            throw std::logic_error(
                "Source aggregate history reconstruction requires aggregate GPU reports");
        added_.clear();
        removed_.clear();
        changed_.clear();
        aggregated_.clear();
        for(physx::PxU32 handle=0;handle<groups_.size();++handle) {
            if(groups_[handle]!=physx::Bp::FilterGroup::eINVALID) {
                if(!added_.growAndSet(handle))throw std::bad_alloc();
            } else {
                // Retired handles have no SAP history after reset.
                volume_to_actor_[handle]=
                    std::numeric_limits<BodyId>::max();
            }
        }
        persistent_state_changed_=true;
        gpu_state_changed_=false;
        bounds_.resetChangedState();
    }

    physx::PxU32 capacity() const noexcept {return bounds_.size();}
    const physx::Bp::SourceBoundsArrayNative& source_bounds() const noexcept {return bounds_;}
    physx::Cm::PinnableBitMap& source_added() noexcept {return added_;}
    physx::Cm::PinnableBitMap& source_removed() noexcept {return removed_;}
    physx::Cm::PinnableBitMap& source_changed() noexcept {return changed_;}

private:
    physx::PxgAggregate& checked_aggregate(physx::Bp::AggregateHandle id) {
        if(id>=aggregates_.size() ||
           aggregates_[id].mIndex==PX_INVALID_U32)
            throw std::out_of_range("Original PhysX aggregate handle is inactive");
        return aggregates_[id];
    }
    void mark_aggregate_dirty(physx::Bp::AggregateHandle id) {
        if(!dirty_aggregates_.boundedTest(id)) {
            if(!dirty_aggregates_.growAndSet(id))throw std::bad_alloc();
            dirty_aggregate_indices_.pushBack(id);
        }
    }

    physx::Bp::SourceBoundsArrayNative bounds_;
    physx::Cm::PinnableArray<physx::PxReal> distances_;
    physx::Cm::PinnableArray<physx::Bp::FilterGroup::Enum> groups_;
    physx::Cm::PinnableArray<physx::Bp::VolumeData> volumes_;
    physx::Cm::PinnableBitMap added_,removed_,changed_,aggregated_;
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle> created_,retired_,updated_;
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle>
        added_aggregate_members_,removed_aggregate_members_;
    physx::PxgCudaBuffer added_device_,removed_device_,changed_device_;
    physx::PxgCudaBuffer aggregated_device_,volumes_device_;
    physx::PxgCudaBuffer aggregate_slots_device_,aggregate_members_device_;
    physx::PxgCudaBuffer
        added_aggregate_members_device_,removed_aggregate_members_device_;
    std::vector<BodyId> volume_to_actor_;
    SourceSapSceneBoundsScratch bounds_scratch_;
    // Original PxgAABBManager::mAggregates and mAggregatesIdPool. Their
    // dirty indices are drained only after source GPU fetch.
    physx::PxArray<physx::PxgAggregate> aggregates_;
    std::vector<physx::PxU32> aggregate_max_shapes_;
    // CUDA per-aggregate member device pointers lowered to reserved
    // offsets in the same source-managed GPU allocation. CPU membership
    // stays exclusively in PxgAggregate::updateBoundIndices.
    std::vector<physx::PxU32> aggregate_gpu_member_starts_;
    std::vector<physx::PxU32> aggregate_gpu_member_reservations_;
    std::vector<SourceAggregateGpuNativeRecord> gpu_aggregate_record_staging_;
    physx::PxU32 aggregate_gpu_slot_count_=0u;
    physx::PxU32 aggregate_gpu_member_capacity_=0u;
    physx::Cm::DeferredIDPool aggregate_ids_;
    physx::Cm::PinnableBitMap dirty_aggregates_;
    physx::Cm::PinnableArray<physx::Bp::AggregateHandle> dirty_aggregate_indices_;
    physx::PxU32 num_active_aggregates_=0u;
    bool persistent_state_changed_=true,gpu_state_changed_=false;
};

} // namespace fyzz::physx_gpu
