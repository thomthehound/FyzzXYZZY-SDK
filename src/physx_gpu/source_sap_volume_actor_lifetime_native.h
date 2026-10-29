// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
#include "physx_port/lowlevelaabb/include/BpVolumeData.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhasePairReport.h"

#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Source PxgAABBManager::removeBounds resets VolumeData before SAP's
// found/lost GPU reports return. This BORROWS the scene owner's existing
// volume-handle -> Sc actor admission map, not a second registry.
// A removed handle retains its prior actor ID until successful fetch,
// and becomes reusable only after the lost-pair result is published.
struct SourceSapVolumeActorLifetimeNative {
    std::span<BodyId> volume_to_actor;
    std::span<const physx::Bp::VolumeData> volume_data;
};

// An existing removed handle keeps its PREVIOUS actor identity until
// the lost-report fetch. Rebinding it to a different actor beforehand
// would misattribute the GPU pair history. Re-admission of the SAME
// actor is permitted, matching addBPEntry's removed-bit cancellation.
inline void admit_source_sap_volume_actor_identity_native(
    SourceSapVolumeActorLifetimeNative state,
    physx::Bp::ShapeHandle handle,BodyId actor) {
    if(handle>=state.volume_to_actor.size() ||
       handle>=state.volume_data.size() ||
       actor==std::numeric_limits<BodyId>::max())
        throw std::out_of_range(
            "Original AABB source admission references invalid Sc identity");
    const auto previous=state.volume_to_actor[handle];
    if(previous!=std::numeric_limits<BodyId>::max() &&
       previous!=actor)
        throw std::logic_error(
            "Source volume handle reused before GPU lost-pair fetch");
    state.volume_to_actor[handle]=actor;
}

// Validate the original source created/removed handle lists before
// preBroadPhase queues GPU work. No full-map walk, staging allocation,
// additional handle allocator, or host-side broadphase is involved.
inline void validate_source_sap_volume_actor_lifetime_native(
    SourceSapVolumeActorLifetimeNative state,
    const physx::Bp::BroadPhaseUpdateData& update,
    bool has_aggregate_roots=false) {
    const auto capacity=update.getCapacity();
    if(state.volume_to_actor.size()<capacity ||
       state.volume_data.size()<capacity)
        throw std::invalid_argument(
            "Original AABB volume-to-actor mapping does not cover source capacity");
    const auto* groups=update.getGroups();
    const auto check=[&](const physx::Bp::ShapeHandle* handles,
                         physx::PxU32 count,bool created) {
        if(count && (!handles || !groups))
            throw std::invalid_argument(
                "Source AABB handle deltas lack their original manager arrays");
        for(physx::PxU32 i=0;i<count;++i) {
            const auto handle=handles[i];
            if(handle>=capacity)
                throw std::out_of_range(
                    "Original SAP delta references an invalid source handle");
            const auto& volume=state.volume_data[handle];
            const bool aggregate_root=has_aggregate_roots && created &&
                volume.isAggregate();
            const bool retired_root=has_aggregate_roots && !created &&
                state.volume_to_actor[handle]==
                    std::numeric_limits<BodyId>::max();
            if(!aggregate_root && !retired_root &&
               state.volume_to_actor[handle]==
                   std::numeric_limits<BodyId>::max())
                throw std::out_of_range(
                    "Original SAP delta has no retained Sc actor identity");
            if(created) {
                if(groups[handle]==physx::Bp::FilterGroup::eINVALID ||
                   !volume.getUserData() ||
                   (!aggregate_root && !volume.isSingleActor()))
                    throw std::invalid_argument(
                        "Source GPU created handle is neither an original aggregate root nor a single actor");
            }else if(groups[handle]!=physx::Bp::FilterGroup::eINVALID ||
                     volume.getUserData()) {
                throw std::invalid_argument(
                    "Source GPU removed handle still owns a live volume");
            }
        }
    };
    check(update.getCreatedHandles(),update.getNumCreatedHandles(),true);
    check(update.getRemovedHandles(),update.getNumRemovedHandles(),false);
}

// Found GPU SAP reports MUST reference currently admitted original volumes.
// Lost reports may name volumes reset by PxgAABBManager::removeBounds and
// are resolved through the retained source mapping until this fetch ends.
inline void validate_source_sap_found_volume_lifetime_native(
    std::span<const physx::PxgBroadPhasePair> found,
    SourceSapVolumeActorLifetimeNative state) {
    for(const auto& pair:found) {
        if(pair.mVolA>=pair.mVolB ||
           pair.mVolB>=state.volume_data.size() ||
           pair.mVolB>=state.volume_to_actor.size())
            throw std::out_of_range(
                "Original GPU found pair references an invalid source volume");
        for(const auto handle:{pair.mVolA,pair.mVolB}) {
            const auto& volume=state.volume_data[handle];
            if(!volume.isSingleActor() || !volume.getUserData() ||
               state.volume_to_actor[handle]==
                   std::numeric_limits<BodyId>::max())
                throw std::invalid_argument(
                    "Original GPU SAP found a pair containing a retired volume");
        }
    }
}

// Called only AFTER source GPU report publication, fetch history rotation
// and enqueueing source post-fetch bitmap retirement. Failed fetch leaves
// the mapping unchanged; a later Sc admission may then safely reuse the
// retired handle without inheriting an unrelated actor's identity.
inline void retire_source_sap_volume_actor_lifetime_native(
    SourceSapVolumeActorLifetimeNative state,
    const physx::Bp::BroadPhaseUpdateData& update) noexcept {
    const auto* removed=update.getRemovedHandles();
    for(physx::PxU32 i=0;i<update.getNumRemovedHandles();++i)
        state.volume_to_actor[removed[i]]=
            std::numeric_limits<BodyId>::max();
}
} // namespace fyzz::physx_gpu
