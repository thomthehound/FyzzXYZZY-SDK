// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_single_volume_scene_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_member_history_native.h"
#include "physx_gpu/source_aggregate_projection_native.h"
#include "physx_gpu/source_aggregate_self_collision_native.h"
#include "physx_gpu/source_aggregate_report_copy_native.h"
#include "physx_gpu/source_aggregate_pair_admission_native.h"
#include "physx_gpu/source_aggregate_pair_retirement_native.h"
#include "physx_gpu/source_aggregate_persistent_collisions_native.h"
#include "physx_gpu/source_sap_mapped_report_view.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhasePairReport.h"
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
#include <array>
#include <cmath>
#include <cstddef>
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
        physx::PxgAllocatorDesc& allocators,physx::PxU32 initial_shapes=0u,
        physx::PxU32 aggregate_report_capacity=0u,
        physx::PxU32 aggregate_pair_capacity=0u)
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
          self_previous_sizes_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          self_found_pairs_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          self_lost_pairs_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          self_report_counts_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_fetch_snapshot_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_slots_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_free_indices_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_free_list_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_count_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_overflow_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_remove_bitmap_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_remove_histogram_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_pair_removed_count_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_found_mapped_(allocators.hostMappedAlloc,
              physx::PxsHeapStats::eBROADPHASE,
              physx::Cm::PinnableAllocatorFallback::eDISABLED),
          aggregate_lost_mapped_(allocators.hostMappedAlloc,
              physx::PxsHeapStats::eBROADPHASE,
              physx::Cm::PinnableAllocatorFallback::eDISABLED),
          projection_keys_0_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_keys_1_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_handles_0_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_handles_1_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_scratch_keys_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_scratch_handles_device_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_sap_boxes_0_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_sap_boxes_1_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_start_masks_0_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_start_masks_1_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_bound_history_0_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          projection_bound_history_1_(allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE),
          aggregate_report_capacity_(aggregate_report_capacity),
          aggregate_pair_capacity_(aggregate_pair_capacity),
          dirty_aggregates_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE),
          dirty_aggregate_indices_(allocators.hostAlloc,physx::PxsHeapStats::eBROADPHASE) {
        if(aggregate_report_capacity_ &&
           (!aggregate_found_mapped_.reserve(aggregate_report_capacity_) ||
            !aggregate_lost_mapped_.reserve(aggregate_report_capacity_)))
            throw std::bad_alloc();
        if(aggregate_report_capacity_)
            aggregate_fetch_snapshot_device_.allocate(
                sizeof(AggregateFetchSnapshot),PX_FL);
        if(initial_shapes)reserve_shapes(initial_shapes);
    }
    ~PxgAabbSourceMembersNative() {
        for(auto& aggregate:aggregates_)aggregate.reset();
    }
    PxgAabbSourceMembersNative(const PxgAabbSourceMembersNative&)=delete;
    PxgAabbSourceMembersNative& operator=(const PxgAabbSourceMembersNative&)=delete;

    void reserve_shapes(physx::PxU32 count) {
        if(count>=BP_INVALID_BP_HANDLE-1u)
            throw std::out_of_range(
                "Source AABB volume capacity exceeds PhysX handle range");
        const auto complete=[&] {
            const auto capacity=bounds_.size();
            const auto has_bits=[&](const auto& bitmap) {
                return std::uint64_t(bitmap.getWordCount())*32u>=capacity;
            };
            return count<=capacity &&
                groups_.size()==capacity &&
                volumes_.size()==capacity &&
                distances_.size()==capacity &&
                volume_to_actor_.size()==capacity &&
                has_bits(added_) && has_bits(removed_) &&
                has_bits(changed_) && has_bits(aggregated_);
        };
        if(complete())return;
        if(count>bounds_.size() && !bounds_.initEntry(count-1u))
            throw std::bad_alloc();
        const auto capacity=bounds_.size();

        // PxArray::resize returns false on allocation failure. Initialize
        // each newly acquired range immediately, before another source
        // allocation can fail. A retry completes partially grown arrays
        // without replacing live source volume identities or BP groups.
        const auto prior_groups=groups_.size();
        if(!groups_.resize(capacity))throw std::bad_alloc();
        for(auto i=prior_groups;i<capacity;++i)
            groups_[i]=physx::Bp::FilterGroup::eINVALID;

        const auto prior_volumes=volumes_.size();
        if(!volumes_.resize(capacity))throw std::bad_alloc();
        for(auto i=prior_volumes;i<capacity;++i)
            volumes_[i].reset();

        const auto prior_distances=distances_.size();
        if(!distances_.resize(capacity))throw std::bad_alloc();
        for(auto i=prior_distances;i<capacity;++i)
            distances_[i]=0.0f;

        volume_to_actor_.resize(capacity,
            std::numeric_limits<BodyId>::max());
        if(!added_.resize(capacity) || !removed_.resize(capacity) ||
           !changed_.resize(capacity) || !aggregated_.resize(capacity))
            throw std::bad_alloc();
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
        // GPU member slices use 32-bit byte offsets. Reject impossible
        // reservations before allocating host members or consuming an ID.
        if(max_shapes>std::numeric_limits<physx::PxU32>::max()/8u)
            throw std::out_of_range(
                "Original aggregate GPU member address range exceeded");
#if BP_USE_AGGREGATE_GROUP_TAIL
        if(free_aggregate_groups_.empty() &&
           aggregate_group_tide_<=physx::Bp::FilterGroup::eDYNAMICS_BASE)
            throw std::out_of_range(
                "Original aggregate BP filter group tail exhausted");
#endif
        const auto id=aggregate_ids_.getNewID();
        bool acquired_slice=false;
        physx::PxU32 slice_first=0u;
        try {
            if(id>=std::numeric_limits<physx::PxU32>::max()/2u)
                throw std::out_of_range(
                    "PhysX aggregate ID exceeds VolumeData encoding");
            if(aggregates_.size()<=id &&
               !aggregates_.resize(2u*id+1u))
                throw std::bad_alloc();
            if(aggregate_max_shapes_.size()<=id)
                aggregate_max_shapes_.resize(2u*id+1u);
            if(aggregate_gpu_member_starts_.size()<=id)
                aggregate_gpu_member_starts_.resize(2u*id+1u,0u);
            if(aggregate_gpu_member_reservations_.size()<=id)
                aggregate_gpu_member_reservations_.resize(2u*id+1u,0u);
            if(self_state_uploaded_.size()<=id)
                self_state_uploaded_.resize(2u*id+1u,false);
            auto& aggregate=aggregates_[id];
            aggregate.reset();
            aggregate.updateBoundIndices=PX_ALLOCATE(
                physx::PxU32,max_shapes,"updateBoundIndices");
            if(!aggregate.updateBoundIndices)throw std::bad_alloc();

            // All fallible source bitmap/index allocations precede the
            // GPU slice and root publication, so the committed identity
            // cannot become detached from its Sc user-data pointer.
            if(!added_.growAndReset(root) ||
               !aggregated_.growAndReset(root) ||
               !dirty_aggregates_.growAndReset(id) ||
               !dirty_aggregate_indices_.reserve(
                   dirty_aggregate_indices_.size()+1u))
                throw std::bad_alloc();
            free_aggregate_member_slices_.reserve(
                free_aggregate_member_slices_.size()+1u);
            slice_first=allocate_aggregate_member_slice(max_shapes);
            acquired_slice=true;
            aggregate_gpu_member_starts_[id]=slice_first;
            aggregate_gpu_member_reservations_[id]=max_shapes;
            self_state_uploaded_[id]=false;
            aggregate.mIndex=root;
            aggregate.filterHint=filter_hint;
            aggregate.mEnvID=env_id;
            aggregate_max_shapes_[id]=max_shapes;
            aggregate_gpu_slot_count_=std::max(aggregate_gpu_slot_count_,id+1u);
            volumes_[root].setUserData(user_data);
            volumes_[root].setVolumeType(physx::Bp::ElementType::eSHAPE);
            volumes_[root].setAggregate(id);
#if BP_USE_AGGREGATE_GROUP_TAIL
            // The tail has been preflighted; the original group allocator
            // now only pops a completed free group or decrements its tide.
            groups_[root]=original_aggregate_group();
            PX_UNUSED(group);
#else
            groups_[root]=group;
#endif
            distances_[root]=0.0f;
            bounds_.setBounds(physx::PxBounds3::empty(),root);
            persistent_state_changed_=true;
            mark_aggregate_dirty(id);
            return id;
        }catch(...) {
            // An unadmitted root never entered the GPU stream. Return
            // its source ID and member slice immediately, not after fetch.
            if(acquired_slice) {
                free_aggregate_member_slices_.push_back(
                    {slice_first,max_shapes,false});
                aggregate_gpu_member_starts_[id]=0u;
                aggregate_gpu_member_reservations_[id]=0u;
            }
            if(id<aggregates_.size())aggregates_[id].reset();
            if(id<aggregate_max_shapes_.size())
                aggregate_max_shapes_[id]=0u;
            if(id<self_state_uploaded_.size())self_state_uploaded_[id]=false;
            groups_[root]=physx::Bp::FilterGroup::eINVALID;
            distances_[root]=0.0f;
            volumes_[root].reset();
            added_.boundedReset(root);
            aggregated_.boundedReset(root);
            dirty_aggregates_.boundedReset(id);
            aggregate_ids_.freeID(id);
            throw;
        }
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
        // An original ScShapeSim can disable and re-enable its SAME
        // member before fetch. Its prior source actor identity and
        // aggregate owner survive inside the original VolumeData slot.
        // A different actor or aggregate cannot claim the retired ID.
        const bool canceled_retirement=
            std::find(removed_aggregate_members_.begin(),
                      removed_aggregate_members_.end(),handle)!=
            removed_aggregate_members_.end();
        reserve_shapes(handle+1u);
        if(groups_[handle]!=physx::Bp::FilterGroup::eINVALID ||
           (canceled_retirement
               ? (volume_to_actor_[handle]!=actor ||
                  !volumes_[handle].isAggregated() ||
                  volumes_[handle].getAggregateOwner()!=id)
               : volume_to_actor_[handle]!=
                    std::numeric_limits<BodyId>::max()))
            throw std::logic_error(
                "Original aggregate member handle has a different live or retired owner");
        if(!dirty_aggregate_indices_.reserve(
               dirty_aggregate_indices_.size()+1u) ||
           (!canceled_retirement &&
            !added_aggregate_members_.reserve(
                added_aggregate_members_.size()+1u)))
            throw std::bad_alloc();
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
        const bool first_member=aggregate.size==0u;
        aggregate.updateBoundIndices[aggregate.size++]=handle;
        if(canceled_retirement) {
            // No member-history removal/addition is emitted: the GPU
            // still owns its previous bound and comparison history.
            // The re-enabled member's current geometry must be restaged.
            if(!removed_aggregate_members_.findAndReplaceWithLast(handle))
                throw std::logic_error(
                    "Original aggregate pending retirement disappeared");
            if(!changed_.growAndSet(handle))throw std::bad_alloc();
        }else if(!added_aggregate_members_.pushBack(handle))
            throw std::bad_alloc();
        mark_aggregate_dirty(id);
        persistent_state_changed_=true;
        if(first_member)++num_nonempty_aggregates_;
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
        const bool unpublished_candidate=
            std::find(added_aggregate_members_.begin(),
                      added_aggregate_members_.end(),handle)!=
                added_aggregate_members_.end();
        if(!dirty_aggregate_indices_.reserve(
               dirty_aggregate_indices_.size()+1u) ||
           (!unpublished_candidate &&
            !removed_aggregate_members_.reserve(
                removed_aggregate_members_.size()+1u)))
            throw std::bad_alloc();
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
        const bool unpublished=
            added_aggregate_members_.findAndReplaceWithLast(handle);
        if(!unpublished && !removed_aggregate_members_.pushBack(handle))
            throw std::bad_alloc();
        changed_.boundedReset(handle);
        aggregated_.boundedReset(handle);
        groups_[handle]=physx::Bp::FilterGroup::eINVALID;
        distances_[handle]=0.0f;
        volumes_[handle].reset();
        // Until the original fetch retires this handle, preserve its
        // aggregate tag for same-Sc-shape removal cancellation. The
        // inactive group and null user data keep it out of live BP.
        if(!unpublished)volumes_[handle].setAggregated(id);
        bounds_.setBounds(physx::PxBounds3::empty(),handle);
        // Original lost GPU member reports may still reference the
        // retired source volume. Keep its Sc actor identity through the
        // successful fetch; same-frame canceled admission has no history.
        if(unpublished)
            volume_to_actor_[handle]=std::numeric_limits<BodyId>::max();
        mark_aggregate_dirty(id);
        persistent_state_changed_=true;
        if(!aggregate.size)--num_nonempty_aggregates_;
        return aggregate.size==0u;
    }

    // PxgAABBManager::destroyAggregate requires all member bounds
    // retired; ID reuse remains deferred until source post-fetch cleanup.
    void destroy_aggregate(physx::Bp::AggregateHandle id) {
        auto& aggregate=checked_aggregate(id);
        if(aggregate.size)
            throw std::logic_error("Cannot destroy original aggregate with live members");
        // Reserve the retirement record before mutating the root. GPU
        // history can reuse this range only after source fetch completes.
        free_aggregate_member_slices_.reserve(
            free_aggregate_member_slices_.size()+1u);
        if(!dirty_aggregate_indices_.reserve(
            dirty_aggregate_indices_.size()+1u))
            throw std::bad_alloc();
        const auto root=aggregate.mIndex;
        added_.boundedReset(root);
        changed_.boundedReset(root);
        aggregated_.boundedReset(root);
#if BP_USE_AGGREGATE_GROUP_TAIL
        // AABBManagerBase::releaseAggregateGroup at the same source
        // aggregate teardown boundary, not at GPU report fetch.
        free_aggregate_groups_.push_back(groups_[root]);
#endif
        groups_[root]=physx::Bp::FilterGroup::eINVALID;
        distances_[root]=0.0f;
        volumes_[root].reset();
        bounds_.setBounds(physx::PxBounds3::empty(),root);
        aggregate.reset();
        aggregate_max_shapes_[id]=0u;
        free_aggregate_member_slices_.push_back({
            aggregate_gpu_member_starts_[id],
            aggregate_gpu_member_reservations_[id],true});
        aggregate_gpu_member_starts_[id]=0u;
        aggregate_gpu_member_reservations_[id]=0u;
        aggregate_ids_.deferredFreeID(id);
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
        self_previous_sizes_device_.allocateCopyOldDataAsync(
            std::uint64_t(aggregate_gpu_slot_count_)*sizeof(physx::PxU32),
            stream,PX_FL);
        gpu_aggregate_record_staging_.resize(dirty_aggregate_indices_.size());
        for(physx::PxU32 i=0;i<dirty_aggregate_indices_.size();++i) {
            const auto id=dirty_aggregate_indices_[i];
            const auto& aggregate=aggregates_[id];
            gpu_aggregate_record_staging_[i]={
                aggregate.mIndex,aggregate_gpu_member_starts_[id],
                aggregate.size,aggregate.filterHint};
            if(aggregate.size>aggregate_gpu_member_reservations_[id])
                throw std::logic_error(
                    "Source aggregate GPU member count exceeds reserved capacity");
        }
        auto& uploads=stream.upload_scratch();
        uploads.reserve(3u*dirty_aggregate_indices_.size());
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
            // Source newly allocated aggregate IDs have no prior
            // self-comparison history, including after deferred reuse.
            if(!self_state_uploaded_[id])
                uploads.push_back({
                    self_previous_sizes_device_.getDevicePtr()+
                        std::uint64_t(id)*sizeof(physx::PxU32),
                    &self_previous_size_zero_,sizeof(physx::PxU32)});
        }
        stream.queue_uploads(uploads);
        // The original GPU previous-size history is initialized once per
        // aggregate-ID lifetime, not on every same-frame dirty DMA. Host
        // isNew remains owned by PxgAggregate until successful fetch.
        for(const auto id:dirty_aggregate_indices_)
            self_state_uploaded_[id]=true;
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

    // The two original PxgAggregate history slots are separate PhysX GPU
    // allocations. Each grows with the source reserved-member domain and
    // retains its own prefix; growing one must never relocate the other.
    void queue_aggregate_projection_gpu_storage(SourceSapDmaNative& stream) {
        const auto members=std::uint64_t(aggregate_gpu_member_capacity_);
        if(!members)return;
        if(members>std::numeric_limits<physx::PxU32>::max()/8u)
            throw std::out_of_range(
                "Original aggregate projection GPU buffers exceed byte range");
        for(auto* buffer:{
            &projection_keys_0_,&projection_keys_1_,
            &projection_handles_0_,&projection_handles_1_,
            &projection_scratch_keys_device_,
            &projection_scratch_handles_device_,
            &projection_sap_boxes_0_,&projection_sap_boxes_1_})
            buffer->allocateCopyOldDataAsync(members*8u,stream,PX_FL);
        // A mask slot reserves one word per maximum member to preserve
        // nonoverlapping per-aggregate ranges across every capacity.
        for(auto* buffer:{
            &projection_start_masks_0_,&projection_start_masks_1_,
            &projection_bound_history_0_,&projection_bound_history_1_})
            buffer->allocateCopyOldDataAsync(members*4u,stream,PX_FL);
    }

    SourceAggregateProjectionNativeBuffers aggregate_projection_gpu_buffers(
        CudaDeviceAddressMap::Address current_integer_bounds,
        bool previous=false) const noexcept {
        const auto slot=aggregate_projection_current_slot_^physx::PxU32(previous);
        const auto& keys=slot?projection_keys_1_:projection_keys_0_;
        const auto& handles=slot?projection_handles_1_:projection_handles_0_;
        const auto& sap=slot?projection_sap_boxes_1_:projection_sap_boxes_0_;
        const auto& masks=slot?projection_start_masks_1_:projection_start_masks_0_;
        const auto& history=slot?projection_bound_history_1_:projection_bound_history_0_;
        return {current_integer_bounds,
            aggregate_slots_device_.getDevicePtr(),
            aggregate_members_device_.getDevicePtr(),
            keys.getDevicePtr(),
            handles.getDevicePtr(),
            projection_scratch_keys_device_.getDevicePtr(),
            projection_scratch_handles_device_.getDevicePtr(),
            sap.getDevicePtr(),
            masks.getDevicePtr(),
            history.getDevicePtr(),
            bounds_.size(),aggregate_gpu_slot_count_,
            aggregate_gpu_member_capacity_,aggregate_gpu_slot_count_,slot};
    }

    // Source PxgAggregateDesc found/lost capacity is configuration-owned.
    // Counters are reset by original GPU DMA before post-SAP collisions,
    // and a report overflow must be rejected by the later source fetch.
    void queue_aggregate_self_report_gpu_dma_up(SourceSapDmaNative& stream) {
        if(!aggregate_report_capacity_)return;
        if(aggregate_report_capacity_>
           std::numeric_limits<physx::PxU32>::max()/8u)
            throw std::out_of_range("Original aggregate GPU report capacity overflow");
        const auto bytes=std::uint64_t(aggregate_report_capacity_)*8u;
        self_found_pairs_device_.allocateCopyOldDataAsync(bytes,stream,PX_FL);
        self_lost_pairs_device_.allocateCopyOldDataAsync(bytes,stream,PX_FL);
        self_report_counts_device_.allocate(8u,PX_FL);
        auto& uploads=stream.upload_scratch();
        uploads.push_back({
            self_report_counts_device_.getDevicePtr(),
            self_report_counters_zero_.data(),sizeof(self_report_counters_zero_)});
        stream.queue_uploads(uploads);
    }

    SourceAggregateSelfCollisionNativeBuffers aggregate_self_collision_gpu_buffers(
        CudaDeviceAddressMap::Address new_integer_bounds,
        CudaDeviceAddressMap::Address old_integer_bounds,
        CudaDeviceAddressMap::Address gpu_group_ids) const noexcept {
        const auto current=aggregate_projection_gpu_buffers(new_integer_bounds);
        const auto previous=aggregate_projection_gpu_buffers(old_integer_bounds,true);
        return {new_integer_bounds,old_integer_bounds,
            aggregate_slots_device_.getDevicePtr(),
            current.sorted_handles,current.sap_boxes,current.bound_index_history,
            current.start_masks,previous.sorted_handles,previous.sap_boxes,
            previous.bound_index_history,previous.start_masks,
            gpu_group_ids,self_previous_sizes_device_.getDevicePtr(),
            self_found_pairs_device_.getDevicePtr(),
            self_lost_pairs_device_.getDevicePtr(),
            self_report_counts_device_.getDevicePtr(),
            aggregate_gpu_slot_count_,aggregate_gpu_slot_count_,
            aggregate_gpu_member_capacity_,bounds_.size(),
            aggregate_report_capacity_};
    }

    SourceAggregateReportCopyNativeBuffers aggregate_report_copy_gpu_buffers(
        const SourceSapMappedReportView& mapped) const {
        if(!aggregate_report_capacity_)
            throw std::logic_error(
                "Original aggregate mapped report capacity not configured");
        const auto bytes=std::size_t(aggregate_report_capacity_)*
                          sizeof(physx::PxgBroadPhasePair);
        return {self_found_pairs_device_.getDevicePtr(),
            self_lost_pairs_device_.getDevicePtr(),
            mapped.device_address(aggregate_found_mapped_.begin(),bytes),
            mapped.device_address(aggregate_lost_mapped_.begin(),bytes),
            self_report_counts_device_.getDevicePtr(),
            aggregate_report_capacity_};
    }

    struct AggregateFetchSnapshot {
        physx::PxgBroadPhaseDesc broadphase{};
        std::array<physx::PxU32,2> counts{};
        std::array<physx::PxU32,3> pair_overflow{};
    };
    // Capture the source descriptor and aggregate counters after the
    // original GPU kernels, on the SAME stream before the fetch fence.
    void queue_aggregate_fetch_snapshot(
        SourceSapDmaNative& stream,
        CudaDeviceAddressMap::Address broadphase_descriptor) {
        if(!aggregate_report_capacity_ || !broadphase_descriptor)
            throw std::logic_error("Source aggregate fetch snapshot owner unavailable");
        const auto start=aggregate_fetch_snapshot_device_.getDevicePtr();
        stream.queue_device_copy(start,
            broadphase_descriptor,sizeof(physx::PxgBroadPhaseDesc));
        stream.queue_device_copy(
            start+offsetof(AggregateFetchSnapshot,counts),
            self_report_counts_device_.getDevicePtr(),
            sizeof(AggregateFetchSnapshot::counts));
        if(aggregate_pair_capacity_)
            stream.queue_device_copy(
                start+offsetof(AggregateFetchSnapshot,pair_overflow),
                aggregate_pair_overflow_device_.getDevicePtr(),
                sizeof(AggregateFetchSnapshot::pair_overflow));
    }
    AggregateFetchSnapshot download_aggregate_fetch_snapshot(
        compute::Context& context,
        const CudaDeviceAddressMap& addresses) const {
        AggregateFetchSnapshot snapshot{};
        const auto source=addresses.resolve(
            aggregate_fetch_snapshot_device_.getDevicePtr(),
            sizeof(snapshot));
        context.download(source.buffer,source.offset,&snapshot,sizeof(snapshot));
        return snapshot;
    }
    struct AggregateReports {
        std::span<const physx::PxgBroadPhasePair> found,lost;
    };
    // Validate source GPU report capacity before any actor or aggregate
    // payload becomes visible to Sc. Aggregate report arrays remain owned
    // by the original mapped PhysX heap.
    std::array<SourceSapMappedReportSpan,2>
    validated_aggregate_report_spans(
        const std::array<physx::PxU32,2>& counts,
        const std::array<physx::PxU32,3>& overflow) {
        if(!aggregate_report_capacity_)
            throw std::logic_error(
                "Original aggregate report fetch has no configured capacity");
        if(counts[0]>aggregate_report_capacity_ ||
           counts[1]>aggregate_report_capacity_)
            throw std::overflow_error(
                "Original aggregate GPU found/lost member report capacity overflow");
        if(aggregate_pair_capacity_ && (overflow[1u] || overflow[2u]))
            throw std::overflow_error(
                "Original GPU persistent aggregate-root pair capacity overflow");
        return {{
            {aggregate_found_mapped_.begin(),
             std::size_t(counts[0])*sizeof(physx::PxgBroadPhasePair)},
            {aggregate_lost_mapped_.begin(),
             std::size_t(counts[1])*sizeof(physx::PxgBroadPhasePair)}}};
    }
    AggregateReports complete_aggregate_mapped_reports(
        const std::array<physx::PxU32,2>& counts) {
        aggregate_found_mapped_.forceSize_Unsafe(counts[0]);
        aggregate_lost_mapped_.forceSize_Unsafe(counts[1]);
        return {
            {aggregate_found_mapped_.begin(),aggregate_found_mapped_.size()},
            {aggregate_lost_mapped_.begin(),aggregate_lost_mapped_.size()}};
    }
    AggregateReports publish_aggregate_mapped_reports(
        SourceSapMappedReportView& mapped,
        const std::array<physx::PxU32,2>& counts,
        const std::array<physx::PxU32,3>& overflow) {
        const auto spans=validated_aggregate_report_spans(counts,overflow);
        if(spans[0].bytes || spans[1].bytes)
            mapped.publish_batch_at_fetch(spans);
        return complete_aggregate_mapped_reports(counts);
    }
    // Independent aggregate kernel tests fetch the original source device
    // buffers directly; scene execution consumes the stream-owned snapshot.
    AggregateReports fetch_aggregate_mapped_reports(
        compute::Context& context,
        const CudaDeviceAddressMap& addresses,
        SourceSapMappedReportView& mapped) {
        std::array<physx::PxU32,2> counts{};
        const auto view=addresses.resolve(
            self_report_counts_device_.getDevicePtr(),sizeof(counts));
        context.download(view.buffer,view.offset,counts.data(),sizeof(counts));
        std::array<physx::PxU32,3> overflow{};
        if(aggregate_pair_capacity_) {
            const auto source=addresses.resolve(
                aggregate_pair_overflow_device_.getDevicePtr(),sizeof(overflow));
            context.download(source.buffer,source.offset,
                             overflow.data(),sizeof(overflow));
        }
        return publish_aggregate_mapped_reports(mapped,counts,overflow);
    }

    // Original PxgAggregateDesc pair slots and free-list are one source
    // GPU heap owner. Initialized once, with all subsequent free-index
    // reuse and max-index publication performed by native GPU stages.
    void queue_aggregate_pair_gpu_storage(SourceSapDmaNative& stream) {
        if(!aggregate_pair_capacity_)return;
        if(aggregate_pair_capacity_>
           std::numeric_limits<physx::PxU32>::max()/12u)
            throw std::out_of_range(
                "Original aggregate root pair slot capacity exceeded");
        const auto count=std::uint64_t(aggregate_pair_capacity_);
        aggregate_pair_slots_device_.allocateCopyOldDataAsync(
            count*12u,stream,PX_FL);
        aggregate_pair_free_indices_device_.allocateCopyOldDataAsync(
            count*4u,stream,PX_FL);
        aggregate_pair_free_list_device_.allocate(8u,PX_FL);
        aggregate_pair_count_device_.allocate(4u,PX_FL);
        aggregate_pair_overflow_device_.allocate(12u,PX_FL);
        aggregate_pair_remove_bitmap_device_.allocateCopyOldDataAsync(
            count*4u,stream,PX_FL);
        aggregate_pair_remove_histogram_device_.allocateCopyOldDataAsync(
            ((count+255u)/256u)*4u,stream,PX_FL);
        aggregate_pair_removed_count_device_.allocate(4u,PX_FL);
        if(!aggregate_pair_storage_initialized_) {
            auto& uploads=stream.upload_scratch();
            uploads.push_back({aggregate_pair_free_list_device_.getDevicePtr(),
                               aggregate_pair_empty_free_list_.data(),8u});
            uploads.push_back({aggregate_pair_count_device_.getDevicePtr(),
                               &aggregate_pair_empty_count_,4u});
            uploads.push_back({aggregate_pair_overflow_device_.getDevicePtr(),
                               aggregate_pair_empty_overflow_.data(),12u});
            stream.queue_uploads(uploads);
            aggregate_pair_storage_initialized_=true;
        }
    }
    SourceAggregatePairAdmissionNativeBuffers aggregate_pair_admission_gpu_buffers(
        CudaDeviceAddressMap::Address descriptor,
        CudaDeviceAddressMap::Address found_root_pairs,
        physx::PxU32 found_root_report_capacity) const noexcept {
        return {descriptor,found_root_pairs,
            aggregate_pair_slots_device_.getDevicePtr(),
            aggregate_pair_free_indices_device_.getDevicePtr(),
            aggregate_pair_free_list_device_.getDevicePtr(),
            aggregate_pair_count_device_.getDevicePtr(),
            aggregate_pair_overflow_device_.getDevicePtr(),
            found_root_report_capacity,aggregate_pair_capacity_};
    }
    SourceAggregatePersistentCollisionNativeBuffers aggregate_persistent_collision_gpu_buffers(
        CudaDeviceAddressMap::Address current_bounds,
        CudaDeviceAddressMap::Address previous_bounds,
        CudaDeviceAddressMap::Address volume_data,
        CudaDeviceAddressMap::Address group_ids,
        CudaDeviceAddressMap::Address removed_root_bitmap) const noexcept {
        const auto current=aggregate_projection_gpu_buffers(current_bounds);
        const auto previous=aggregate_projection_gpu_buffers(previous_bounds,true);
        return {current_bounds,previous_bounds,volume_data,
            aggregate_slots_device_.getDevicePtr(),
            current.sorted_projections,current.sorted_handles,
            current.bound_index_history,current.start_masks,
            previous.sorted_projections,previous.sorted_handles,
            previous.bound_index_history,previous.start_masks,
            self_previous_sizes_device_.getDevicePtr(),group_ids,
            aggregate_pair_slots_device_.getDevicePtr(),
            aggregate_pair_count_device_.getDevicePtr(),
            removed_root_bitmap,aggregate_pair_remove_bitmap_device_.getDevicePtr(),
            self_found_pairs_device_.getDevicePtr(),
            self_lost_pairs_device_.getDevicePtr(),
            self_report_counts_device_.getDevicePtr(),
            aggregate_pair_capacity_,aggregate_gpu_slot_count_,
            aggregate_gpu_member_capacity_,bounds_.size(),
            aggregate_report_capacity_};
    }
    SourceAggregatePairRetirementNativeBuffers aggregate_pair_retirement_gpu_buffers() const noexcept {
        return {aggregate_pair_slots_device_.getDevicePtr(),
            aggregate_pair_remove_bitmap_device_.getDevicePtr(),
            aggregate_pair_remove_histogram_device_.getDevicePtr(),
            aggregate_pair_free_indices_device_.getDevicePtr(),
            aggregate_pair_free_list_device_.getDevicePtr(),
            aggregate_pair_count_device_.getDevicePtr(),
            aggregate_pair_removed_count_device_.getDevicePtr(),
            aggregate_pair_capacity_};
    }
    physx::PxU32 aggregate_pair_capacity() const noexcept {
        return aggregate_pair_capacity_;
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
        const bool advanced_aggregate_history=aggregate_gpu_work_pending();
        for(const auto id:dirty_aggregate_indices()) {
            // Original clearDirtyAggregates retires the source per-frame
            // size history only after GPU aggregate reports are fetched.
            aggregates_[id].prevSize=aggregates_[id].size;
            aggregates_[id].isNew=false;
        }
        dirty_aggregate_indices_.resetOrClear();
        dirty_aggregates_.clear();
        added_aggregate_members_.resetOrClear();
        // Retire the original source-to-Sc actor identities only after
        // all GPU aggregate lost reports have been fetched and admitted.
        for(const auto handle:removed_aggregate_members_) {
            volume_to_actor_[handle]=std::numeric_limits<BodyId>::max();
            volumes_[handle].reset();
        }
        removed_aggregate_members_.resetOrClear();
        if(advanced_aggregate_history)aggregate_projection_current_slot_^=1u;
        // Only completed source GPU histories release member storage.
        for(auto& slice:free_aggregate_member_slices_)slice.pending=false;
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

    // Retired GPU slots remain allocated for ID reuse, but do not imply
    // active aggregate work after the last report has been fetched.
    bool aggregate_gpu_work_pending() const noexcept {
        return num_nonempty_aggregates_!=0u ||
            !dirty_aggregate_indices_.empty() ||
            !added_aggregate_members_.empty() ||
            !removed_aggregate_members_.empty();
    }

    // Called after Sc has marked source dirty volumes and before the
    // original preBroadPhase task. The returned update borrows this
    // object's arrays until SAP's GPU fetch continuation finishes.
    physx::Bp::BroadPhaseUpdateData prepare(
        std::span<const native::GpuBroadphaseBox> live,
        std::span<const physx::Bp::ShapeHandle> handles,
        const physx::Bp::BpFilter& filter,
        bool gpu_bounds = false) {
        box_transform_records_.clear();
        auto* records=gpu_bounds ? &box_transform_records_ : nullptr;
        if(live.size()!=handles.size())
            throw std::invalid_argument(
                "Source SAP live actor and original volume handle counts disagree");
        // The original PxgAABBManager first pass owns the root's BP
        // created/removed handles, while Sc supplies the actual member
        // geometry. A root is NOT a shape in live actor geometry.
        if(aggregate_gpu_work_pending()) {
            for(std::size_t i=0;i<live.size();++i) {
                const auto handle=handles[i];
                if(handle>=volumes_.size() ||
                   handle>=volume_to_actor_.size() ||
                   volume_to_actor_[handle]!=live[i].id ||
                   groups_[handle]==physx::Bp::FilterGroup::eINVALID ||
                   !volumes_[handle].getUserData() ||
                   volumes_[handle].isAggregate())
                    throw std::invalid_argument(
                        "Source aggregate Sc geometry does not match original member/actor ownership");
                const auto group=groups_[handle];
                const bool static_group=
                    group==physx::Bp::getFilterGroup_Statics();
                const auto type=static_cast<physx::PxU32>(group)&
                    BP_FILTERING_TYPE_MASK;
                const auto motion=live[i].motion;
                const bool compatible=
                    motion==native::GpuBroadphaseBox::Motion::Static
                        ? static_group
                        : !static_group && type==(
                            motion==native::GpuBroadphaseBox::Motion::Kinematic
                                ? physx::Bp::FilterType::KINEMATIC
                                : physx::Bp::FilterType::DYNAMIC);
                if(!compatible)
                    throw std::invalid_argument(
                        "Source aggregate member motion differs from original Sc BP group");
            }
            collect_source_sap_single_volume_first_pass_native(
                single_volumes(),{created_,retired_,updated_});
            // Original first-pass aggregate member admissions are not
            // BP created handles. They still need their initial bounds
            // staged into the owner's host arrays before gpuDmaDataUp.
            aggregate_geometry_created_.clear();
            aggregate_geometry_created_.reserve(
                created_.size()+added_aggregate_members_.size());
            for(const auto handle:created_)
                if(volumes_[handle].isSingleActor())
                    aggregate_geometry_created_.push_back(handle);
            for(const auto handle:added_aggregate_members_)
                aggregate_geometry_created_.push_back(handle);
            std::sort(aggregate_geometry_created_.begin(),
                      aggregate_geometry_created_.end());
            if(std::adjacent_find(aggregate_geometry_created_.begin(),
                                  aggregate_geometry_created_.end())!=
               aggregate_geometry_created_.end())
                throw std::logic_error(
                    "Original AABB aggregate geometry admitted a source handle twice");
            const SourceSapSceneFrameChanges geometry_changes{
                {aggregate_geometry_created_.data(),
                 aggregate_geometry_created_.size()},
                {},{},
                {changed_.getWords(),changed_.getWordCount()}};
            write_source_sap_scene_bounds(
                live,handles,scene_bounds(),bounds_scratch_,
                &geometry_changes,true,records);
            const auto update=physx::Bp::BroadPhaseUpdateData(
                created_.begin(),created_.size(),
                nullptr,0u,retired_.begin(),retired_.size(),
                bounds_.begin(),groups_.begin(),distances_.begin(),
                bounds_.size(),filter,
                persistent_state_changed_||bounds_.hasChanged(),
                gpu_state_changed_||aggregate_gpu_work_pending());
            validate_source_sap_volume_actor_lifetime_native(
                actor_lifetime(),update,true);
            return update;
        }
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
            scene_bounds(),filter,bounds_.hasChanged(),bounds_scratch_,records);
        validate_source_sap_volume_actor_lifetime_native(
            actor_lifetime(),update);
        return update;
    }

    std::span<const SourceScBoxTransformInputNative> box_transform_records() const noexcept {
        return box_transform_records_;
    }

    SourceSapAabbManagerPreUpdateNative pre_update(
        SourceSapPreBroadPhaseBuffers sap,
        const SourceSapGpuBoxTransformsNative* gpu_boxes=nullptr) noexcept {
        const bool aggregate_slots=aggregate_gpu_work_pending();
        return {sap,source_sap_aabb_bitmap_native(added_device_,added_),
            source_sap_aabb_bitmap_native(removed_device_,removed_),
            source_sap_aabb_bitmap_native(aggregated_device_,aggregated_),
            source_sap_aabb_bitmap_native(changed_device_,changed_),
            volumes_device_,{volumes_.begin(),volumes_.size()},
            persistent_state_changed_,gpu_state_changed_ || aggregate_slots,
            aggregate_slots, nullptr,nullptr,nullptr,nullptr,0u,0u,
            gpu_boxes};
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
        SourceSapDmaNative& stream,
        const SourceSapGpuBoxTransformsNative* gpu_boxes=nullptr) {
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
        queue_aggregate_projection_gpu_storage(stream);
        queue_aggregate_self_report_gpu_dma_up(stream);
        queue_aggregate_pair_gpu_storage(stream);
        auto manager=pre_update(sap,gpu_boxes);
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
        // A rejected Sc/Dy frame invalidates all GPU comparison history,
        // not the authoritative Sc VolumeData or PxgAggregate membership.
        // Re-enter the original create path for live single volumes and
        // nonempty aggregate roots; members are GPU geometry, not SAP boxes.
        added_.clear();
        removed_.clear();
        changed_.clear();
        aggregated_.clear();
        added_aggregate_members_.resetOrClear();
        removed_aggregate_members_.resetOrClear();
        dirty_aggregates_.clear();
        dirty_aggregate_indices_.resetOrClear();
        aggregate_projection_current_slot_=0u;
        aggregate_pair_storage_initialized_=false;
        // The rollback caller already waited the abandoned source stream.
        for(auto& slice:free_aggregate_member_slices_)slice.pending=false;

        for(physx::PxU32 handle=0;handle<groups_.size();++handle) {
            if(groups_[handle]==physx::Bp::FilterGroup::eINVALID) {
                // Lost reports from the abandoned GPU history will not
                // be fetched. Retired handles release their actor mapping.
                volume_to_actor_[handle]=
                    std::numeric_limits<BodyId>::max();
                volumes_[handle].reset();
            } else if(volumes_[handle].isAggregated()) {
                if(!aggregated_.growAndSet(handle))throw std::bad_alloc();
                if(!added_aggregate_members_.pushBack(handle))
                    throw std::bad_alloc();
            } else if(!volumes_[handle].isAggregate()) {
                if(!added_.growAndSet(handle))throw std::bad_alloc();
            }
        }
        for(physx::PxU32 id=0;id<aggregate_gpu_slot_count_;++id) {
            auto& aggregate=aggregates_[id];
            // Even retired aggregate slots may still exist on the GPU
            // when a rejected frame discards their deletion upload.
            // Publish their empty source records before self collision
            // dispatch examines the reserved aggregate ID domain.
            mark_aggregate_dirty(id);
            if(aggregate.mIndex==PX_INVALID_U32)continue;
            // Fresh GPU projections and self history begin at size zero;
            // do not read stale member bounds or pair slots from the
            // rejected frame's previously published projection buffers.
            aggregate.prevSize=0u;
            aggregate.isNew=true;
            self_state_uploaded_[id]=false;
            if(aggregate.size &&
               !added_.growAndSet(aggregate.mIndex))
                throw std::bad_alloc();
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
#if BP_USE_AGGREGATE_GROUP_TAIL
    physx::Bp::FilterGroup::Enum original_aggregate_group() {
        if(!free_aggregate_groups_.empty()) {
            const auto group=free_aggregate_groups_.back();
            free_aggregate_groups_.pop_back();
            return group;
        }
        // The original tail starts at eAGGREGATE_BASE and uses the
        // low three bits for FilterType::AGGREGATE.
        if(aggregate_group_tide_<=physx::Bp::FilterGroup::eDYNAMICS_BASE)
            throw std::out_of_range("Original aggregate BP filter group tail exhausted");
        const auto id=aggregate_group_tide_--;
        return physx::Bp::FilterGroup::Enum(
            (id<<BP_FILTERING_TYPE_SHIFT_BIT) |
            physx::Bp::FilterType::AGGREGATE);
    }
#endif
    struct AggregateMemberSlice {
        physx::PxU32 first=0u,count=0u;
        bool pending=false;
    };
    // Only aggregate creation visits this free list; no per-frame
    // compaction, GPU history copies or relocation of live source slots.
    physx::PxU32 allocate_aggregate_member_slice(physx::PxU32 count) {
        std::sort(free_aggregate_member_slices_.begin(),
                  free_aggregate_member_slices_.end(),
                  [](const auto& a,const auto& b) {
                      return a.first<b.first;
                  });
        std::size_t write=0;
        for(const auto slice:free_aggregate_member_slices_) {
            if(write && !slice.pending &&
               !free_aggregate_member_slices_[write-1u].pending &&
               free_aggregate_member_slices_[write-1u].first+
                   free_aggregate_member_slices_[write-1u].count==
                   slice.first) {
                free_aggregate_member_slices_[write-1u].count+=slice.count;
            } else free_aggregate_member_slices_[write++]=slice;
        }
        free_aggregate_member_slices_.resize(write);
        // Trimming only the completed tail permits capacity growth in
        // place on aggregate-ID reuse without discarding GPU allocation.
        while(!free_aggregate_member_slices_.empty()) {
            const auto& tail=free_aggregate_member_slices_.back();
            if(tail.pending ||
               tail.first+tail.count!=aggregate_gpu_member_capacity_)break;
            aggregate_gpu_member_capacity_=tail.first;
            free_aggregate_member_slices_.pop_back();
        }
        for(std::size_t i=0;i<free_aggregate_member_slices_.size();++i) {
            auto& slice=free_aggregate_member_slices_[i];
            if(slice.pending || slice.count<count)continue;
            const auto first=slice.first;
            slice.first+=count;
            slice.count-=count;
            if(!slice.count)free_aggregate_member_slices_.erase(
                free_aggregate_member_slices_.begin()+i);
            return first;
        }
        // Every source projection key/handle/SAP history allocates
        // eight bytes per reserved member, not merely four for its ID.
        constexpr auto max_members=
            std::numeric_limits<physx::PxU32>::max()/8u;
        if(aggregate_gpu_member_capacity_>max_members ||
           count>max_members-aggregate_gpu_member_capacity_)
            throw std::out_of_range(
                "Original aggregate GPU projection address range exceeded");
        const auto first=aggregate_gpu_member_capacity_;
        aggregate_gpu_member_capacity_+=count;
        return first;
    }
    physx::PxgAggregate& checked_aggregate(physx::Bp::AggregateHandle id) {
        if(id>=aggregates_.size() ||
           aggregates_[id].mIndex==PX_INVALID_U32)
            throw std::out_of_range("Original PhysX aggregate handle is inactive");
        return aggregates_[id];
    }
    void mark_aggregate_dirty(physx::Bp::AggregateHandle id) {
        if(!dirty_aggregates_.boundedTest(id)) {
            if(!dirty_aggregates_.growAndReset(id))
                throw std::bad_alloc();
            if(!dirty_aggregate_indices_.pushBack(id))
                throw std::bad_alloc();
            dirty_aggregates_.set(id);
        }
    }

    physx::Bp::SourceBoundsArrayNative bounds_;
    physx::Cm::PinnableArray<physx::PxReal> distances_;
    physx::Cm::PinnableArray<physx::Bp::FilterGroup::Enum> groups_;
#if BP_USE_AGGREGATE_GROUP_TAIL
    physx::PxU32 aggregate_group_tide_=
        physx::Bp::FilterGroup::eAGGREGATE_BASE;
    std::vector<physx::Bp::FilterGroup::Enum> free_aggregate_groups_;
#endif
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
    physx::PxgCudaBuffer self_previous_sizes_device_;
    physx::PxgCudaBuffer self_found_pairs_device_,self_lost_pairs_device_;
    physx::PxgCudaBuffer self_report_counts_device_;
    physx::PxgCudaBuffer aggregate_fetch_snapshot_device_;
    physx::PxgCudaBuffer aggregate_pair_slots_device_;
    physx::PxgCudaBuffer aggregate_pair_free_indices_device_;
    physx::PxgCudaBuffer aggregate_pair_free_list_device_;
    physx::PxgCudaBuffer aggregate_pair_count_device_;
    physx::PxgCudaBuffer aggregate_pair_overflow_device_;
    physx::PxgCudaBuffer aggregate_pair_remove_bitmap_device_;
    physx::PxgCudaBuffer aggregate_pair_remove_histogram_device_;
    physx::PxgCudaBuffer aggregate_pair_removed_count_device_;
    physx::Cm::PinnableArray<physx::PxgBroadPhasePair>
        aggregate_found_mapped_,aggregate_lost_mapped_;
    physx::PxgCudaBuffer projection_keys_0_,projection_keys_1_;
    physx::PxgCudaBuffer projection_handles_0_,projection_handles_1_;
    physx::PxgCudaBuffer projection_scratch_keys_device_,projection_scratch_handles_device_;
    physx::PxgCudaBuffer projection_sap_boxes_0_,projection_sap_boxes_1_;
    physx::PxgCudaBuffer projection_start_masks_0_,projection_start_masks_1_;
    physx::PxgCudaBuffer projection_bound_history_0_,projection_bound_history_1_;
    std::vector<BodyId> volume_to_actor_;
    SourceSapSceneBoundsScratch bounds_scratch_;
    std::vector<SourceScBoxTransformInputNative> box_transform_records_;
    // Transient original first-pass member geometry handles; Bp root
    // deltas and all persistent identity/history remain AABB-owned.
    std::vector<physx::Bp::ShapeHandle> aggregate_geometry_created_;
    // Original PxgAABBManager::mAggregates and mAggregatesIdPool. Their
    // dirty indices are drained only after source GPU fetch.
    physx::PxArray<physx::PxgAggregate> aggregates_;
    std::vector<physx::PxU32> aggregate_max_shapes_;
    // CUDA per-aggregate member device pointers lowered to reserved
    // offsets in the same source-managed GPU allocation. CPU membership
    // stays exclusively in PxgAggregate::updateBoundIndices.
    std::vector<physx::PxU32> aggregate_gpu_member_starts_;
    std::vector<physx::PxU32> aggregate_gpu_member_reservations_;
    std::vector<AggregateMemberSlice> free_aggregate_member_slices_;
    std::vector<bool> self_state_uploaded_;
    std::vector<SourceAggregateGpuNativeRecord> gpu_aggregate_record_staging_;
    physx::PxU32 aggregate_gpu_slot_count_=0u;
    physx::PxU32 aggregate_gpu_member_capacity_=0u;
    physx::PxU32 aggregate_projection_current_slot_=0u;
    physx::PxU32 aggregate_report_capacity_=0u;
    physx::PxU32 aggregate_pair_capacity_=0u;
    bool aggregate_pair_storage_initialized_=false;
    std::array<physx::PxU32,2> aggregate_pair_empty_free_list_{{0u,0u}};
    physx::PxU32 aggregate_pair_empty_count_=0u;
    std::array<physx::PxU32,3> aggregate_pair_empty_overflow_{{0u,0u,0u}};
    physx::PxU32 self_previous_size_zero_=0u;
    std::array<physx::PxU32,2> self_report_counters_zero_{{0u,0u}};
    physx::Cm::DeferredIDPool aggregate_ids_;
    physx::Cm::PinnableBitMap dirty_aggregates_;
    physx::Cm::PinnableArray<physx::Bp::AggregateHandle> dirty_aggregate_indices_;
    physx::PxU32 num_nonempty_aggregates_=0u;
    bool persistent_state_changed_=true,gpu_state_changed_=false;
};

} // namespace fyzz::physx_gpu
