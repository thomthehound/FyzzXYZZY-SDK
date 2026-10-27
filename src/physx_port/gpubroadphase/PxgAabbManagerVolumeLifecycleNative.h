// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgAabbManagerPreUpdateNative.h"
#include "physx_gpu/source_sap_volume_actor_lifetime_native.h"
#include "BpBroadPhaseUpdate.h"
#include "BpVolumeData.h"
#include "foundation/PxBounds3.h"
#include "CmPinnableArray.h"
#include "CmPinnableBitMap.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Bind original PinnableBitMap words directly. No copied bitmap/vector
// shadows may sit between AABB-manager admissions and GPU descriptor DMA.
inline SourceSapAabbManagerBitmapNative source_sap_aabb_bitmap_native(
    physx::PxgCudaBuffer& device,
    const physx::Cm::PinnableBitMap& bitmap) noexcept {
    return {device,{bitmap.getWords(),bitmap.getWordCount()}};
}

// Borrowed PxgAABBManager/AABBManagerBase members for single-actor volumes.
// The source AABB manager retains ALL bitmaps, storage and handle ownership;
// these operations do not introduce a parallel volume or pair registry.
struct SourceSapSingleVolumeMembersNative {
    physx::Cm::PinnableBitMap& added;
    physx::Cm::PinnableBitMap& removed;
    physx::Cm::PinnableBitMap& changed;
    physx::Cm::PinnableBitMap& aggregated;
    std::span<physx::Bp::VolumeData> volume_data;
    std::span<physx::Bp::FilterGroup::Enum> groups;
    std::span<physx::PxReal> contact_distances;
    bool& persistent_state_changed;
    bool& gpu_state_changed;
};

inline void validate_source_sap_single_volume_native(
    const SourceSapSingleVolumeMembersNative& m,
    physx::PxU32 index) {
    if(index>=m.volume_data.size() || index>=m.groups.size() ||
       index>=m.contact_distances.size() ||
       index>=std::numeric_limits<physx::PxU32>::max()-1u)
        throw std::out_of_range("Original AABB-manager volume handle is outside capacity");
}

// Bp::AABBManagerBase::addBPEntry and PxgAABBManager::addBounds,
// single-actor branch. An add after a pending remove cancels the remove;
// it must NOT generate another created handle.
inline void add_source_sap_single_volume_native(
    SourceSapSingleVolumeMembersNative m,
    physx::PxU32 index,
    physx::PxReal contact_distance,
    physx::Bp::FilterGroup::Enum group,
    void* user_data,
    physx::Bp::ElementType::Enum volume_type=physx::Bp::ElementType::eSHAPE) {
    validate_source_sap_single_volume_native(m,index);
    if(m.groups[index]!=physx::Bp::FilterGroup::eINVALID ||
       group==physx::Bp::FilterGroup::eINVALID ||
       !std::isfinite(contact_distance) || contact_distance<0.0f ||
       volume_type>=physx::Bp::ElementType::eCOUNT ||
       (reinterpret_cast<std::uintptr_t>(user_data)&3u))
        throw std::invalid_argument("Invalid source AABB-manager volume admission");

    // Allocate bitmap capacity before mutating any admission state. This
    // preserves the original bitmap representation and same-frame changes.
    const auto size=index+1u;
    if(!m.added.resize(size) || !m.removed.resize(size) ||
       !m.aggregated.resize(size))
        throw std::bad_alloc();

    auto& volume=m.volume_data[index];
    volume.setUserData(user_data);
    volume.setVolumeType(volume_type);
    volume.setSingleActor();
    m.groups[index]=group;
    m.contact_distances[index]=contact_distance;

    if(m.removed.boundedTest(index))m.removed.boundedReset(index);
    else m.added.set(index);
    m.aggregated.boundedReset(index);
    m.persistent_state_changed=true;
}

// Sc single-volume admission binds its actor ID to the SAME source
// manager handle on which addBounds operates. The existing map is
// retained through GPU fetch; reject reuse by a different actor while
// any old lost-pair report may still reference that handle. Complete
// original bitmap/VolumeData admission first, so its validation failure
// cannot partially overwrite the actor identity mapping.
inline void add_source_sap_single_volume_actor_native(
    SourceSapSingleVolumeMembersNative manager,
    SourceSapVolumeActorLifetimeNative identity,
    physx::PxU32 index,
    physx::PxReal contact_distance,
    physx::Bp::FilterGroup::Enum group,
    void* user_data,BodyId actor,
    physx::Bp::ElementType::Enum volume_type=physx::Bp::ElementType::eSHAPE) {
    if(identity.volume_data.data()!=manager.volume_data.data() ||
       identity.volume_data.size()!=manager.volume_data.size())
        throw std::invalid_argument(
            "Source actor identity does not alias original AABB volume owner");
    if(index>=identity.volume_to_actor.size() ||
       actor==std::numeric_limits<BodyId>::max())
        throw std::out_of_range("Source AABB actor identity outside volume mapping");
    const auto previous=identity.volume_to_actor[index];
    if(previous!=std::numeric_limits<BodyId>::max() && previous!=actor)
        throw std::logic_error(
            "Source AABB handle reused before GPU lost-pair publication");
    add_source_sap_single_volume_native(
        manager,index,contact_distance,group,user_data,volume_type);
    identity.volume_to_actor[index]=actor;
}

// Bp::AABBManagerBase::removeBPEntry and PxgAABBManager::removeBounds,
// single-actor branch. Source VolumeData is reset immediately, while the
// scene's fetch-time volume-to-actor lifetime is owned by its caller.
inline bool remove_source_sap_single_volume_native(
    SourceSapSingleVolumeMembersNative m,
    physx::PxU32 index) {
    validate_source_sap_single_volume_native(m,index);
    if(m.groups[index]==physx::Bp::FilterGroup::eINVALID ||
       !m.volume_data[index].isSingleActor())
        throw std::invalid_argument("Source AABB manager cannot retire inactive volume");
    const bool cancelled=m.added.boundedTest(index)!=0;
    if(cancelled)m.added.boundedReset(index);
    else if(!m.removed.growAndSet(index))throw std::bad_alloc();
    m.aggregated.boundedReset(index);
    m.groups[index]=physx::Bp::FilterGroup::eINVALID;
    m.contact_distances[index]=0.0f;
    m.volume_data[index].reset();
    m.persistent_state_changed=true;
    return cancelled;
}

// A canceled same-frame add has NEVER entered GPU SAP history, so
// there is no eventual lost report and the actor identity is released
// immediately. A previously published source volume's removal keeps
// its actor ID until post-fetch retirement.
inline bool remove_source_sap_single_volume_actor_native(
    SourceSapSingleVolumeMembersNative manager,
    SourceSapVolumeActorLifetimeNative identity,
    physx::PxU32 index) {
    if(identity.volume_data.data()!=manager.volume_data.data() ||
       identity.volume_data.size()!=manager.volume_data.size() ||
       index>=identity.volume_to_actor.size() ||
       identity.volume_to_actor[index]==
           std::numeric_limits<BodyId>::max())
        throw std::invalid_argument(
            "Source AABB removal has no matching original Sc actor identity");
    const bool cancelled=remove_source_sap_single_volume_native(
        manager,index);
    if(cancelled)
        identity.volume_to_actor[index]=
            std::numeric_limits<BodyId>::max();
    return cancelled;
}

// AABBManagerBase::setContactDistance and the GPU changed-map update.
// Source first-pass determines GPUStateChanged from this bitmap; a changed
// bit must not be replaced by per-frame geometry comparisons.
inline void mark_source_sap_single_volume_changed_native(
    SourceSapSingleVolumeMembersNative m,
    physx::PxU32 index,
    physx::PxReal contact_distance) {
    validate_source_sap_single_volume_native(m,index);
    if(m.groups[index]==physx::Bp::FilterGroup::eINVALID ||
       !m.volume_data[index].isSingleActor() ||
       !std::isfinite(contact_distance) || contact_distance<0.0f)
        throw std::invalid_argument("Source AABB manager changed inactive volume");
    if(!m.changed.growAndSet(index))throw std::bad_alloc();
    m.contact_distances[index]=contact_distance;
    m.persistent_state_changed=true;
}

struct SourceSapSingleVolumeFirstPassNative {
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle>& created;
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle>& removed;
    physx::Cm::PinnableArray<physx::Bp::ShapeHandle>& updated;
};

// PxgAABBManager::updateBPFirstPass, single-actor bitmap enumeration.
// The original maps are scanned in ascending bit order, directly into
// PhysX's existing PinnableArray handle storage. Updated-list size is 0:
// GPU AABB-manager mode DMA-transfers the changed bitmap instead.
inline void collect_source_sap_single_volume_first_pass_native(
    SourceSapSingleVolumeMembersNative m,
    SourceSapSingleVolumeFirstPassNative out) {
    const auto enumerate=[](
        const physx::Cm::PinnableBitMap& bitmap,
        physx::Cm::PinnableArray<physx::Bp::ShapeHandle>& dst) {
        dst.resetOrClear();
        const auto* words=bitmap.getWords();
        if(!words)return;
        for(physx::PxU32 word=0;word<bitmap.getWordCount();++word) {
            for(auto bits=words[word];bits;bits&=bits-1u) {
                const auto handle=(word<<5u)+std::countr_zero(bits);
                dst.pushBack(handle);
            }
        }
    };
    enumerate(m.added,out.created);
    out.updated.forceSize_Unsafe(0);
    if(m.changed.hasAnyBitSet())m.gpu_state_changed=true;
    enumerate(m.removed,out.removed);
}

// PxgAABBManager::updateBPSecondPass constructs the exact Bp update
// from its first-pass lists, source AABB arrays and dirty flags. The owner
// retains all referenced storage through GPU update and fetch.
inline physx::Bp::BroadPhaseUpdateData
source_sap_single_volume_update_native(
    SourceSapSingleVolumeMembersNative m,
    SourceSapSingleVolumeFirstPassNative handles,
    std::span<const physx::PxBounds3> bounds,
    const physx::Bp::BpFilter& filter,
    bool bounds_changed) {
    if(bounds.size()!=m.groups.size() ||
       bounds.size()!=m.contact_distances.size() ||
       bounds.size()>std::numeric_limits<physx::PxU32>::max())
        throw std::invalid_argument(
            "Source AABB-manager frame arrays disagree on capacity");
    collect_source_sap_single_volume_first_pass_native(m,handles);
    return physx::Bp::BroadPhaseUpdateData(
        handles.created.begin(),handles.created.size(),
        nullptr,0u,handles.removed.begin(),handles.removed.size(),
        bounds.data(),m.groups.data(),m.contact_distances.data(),
        nullptr,static_cast<physx::PxU32>(bounds.size()),filter,
        m.persistent_state_changed || bounds_changed,m.gpu_state_changed);
}

// PxgAABBManager::processFoundPairs/processLostPairs clear the host maps
// only after publication. GPU-side memsetD32Async is performed by the
// existing SourceFillD32NativeStage on the source stream.
inline void clear_source_sap_single_volume_report_bitmaps_native(
    SourceSapSingleVolumeMembersNative m) {
    m.added.clear();
    m.removed.clear();
}

} // namespace fyzz::physx_gpu
