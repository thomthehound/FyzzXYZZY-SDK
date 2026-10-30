// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "native/scene.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhasePairReport.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Consumes the ORIGINAL PxgCudaBroadPhaseSap::fetchBroadPhaseResults
// mFoundActorPairsMapped/mLostActorPairsMapped records after the required
// GPU fence and successful report capacity/abort validation.
//
// mVolA/mVolB are source broadphase volume handles, NOT native actor IDs.
// The PxgAABBManager owner retains handle -> actor ownership until after
// this call, including handles being retired this frame. We canonicalize
// into the single Sc stream for N1's existing contact-manager lifecycle.
// Caller retains both output allocations across source updates; no second
// SAP registry, GPU history or simulation clock is created here.
// Original PxgAABBManager::processFoundPairs/processLostPairs combine
// source SAP actor reports and postBPStage2 aggregate member reports. The
// existing Sc pair destination is their ONE publication path. No host
// broadphase, second registry or extra frame state is introduced.
inline void lower_source_sap_and_aggregate_reports_to_sc(
    std::span<const physx::PxgBroadPhasePair> actor_found,
    std::span<const physx::PxgBroadPhasePair> actor_lost,
    std::span<const physx::PxgBroadPhasePair> aggregate_found,
    std::span<const physx::PxgBroadPhasePair> aggregate_lost,
    std::span<const BodyId> volume_to_actor,
    std::vector<native::Pair>& found,
    std::vector<native::Pair>& lost) {
    if(&found==&lost)
        throw std::invalid_argument("Source GPU Sc found/lost outputs alias");

    const auto check=[&](std::span<const physx::PxgBroadPhasePair> pairs) {
        for(const auto& p:pairs) {
            if(p.mVolA>=p.mVolB ||
               p.mVolB>=volume_to_actor.size() ||
               volume_to_actor[p.mVolA]==std::numeric_limits<BodyId>::max() ||
               volume_to_actor[p.mVolB]==std::numeric_limits<BodyId>::max() ||
               volume_to_actor[p.mVolA]==volume_to_actor[p.mVolB])
                throw std::out_of_range(
                    "Original source GPU result references an invalid Sc volume handle");
        }
    };
    // Verify all four original report streams before changing either Sc
    // destination. Aggregate lost pairs may reference retired members,
    // whose original source actor identities survive until this fetch.
    check(actor_found);
    check(actor_lost);
    check(aggregate_found);
    check(aggregate_lost);
    found.reserve(actor_found.size()+aggregate_found.size());
    lost.reserve(actor_lost.size()+aggregate_lost.size());
    const auto lower=[&](std::span<const physx::PxgBroadPhasePair> actors,
                         std::span<const physx::PxgBroadPhasePair> aggregates,
                         std::vector<native::Pair>& destination) {
        destination.clear();
        const auto append=[&](std::span<const physx::PxgBroadPhasePair> pairs) {
            for(const auto& p:pairs) {
                const BodyId a=volume_to_actor[p.mVolA];
                const BodyId b=volume_to_actor[p.mVolB];
                destination.push_back({std::min(a,b),std::max(a,b)});
            }
        };
        append(actors);
        append(aggregates);
        // Sorting is already required by the original report
        // classification. Duplicate shape and root-member reports for
        // a single Sc actor pair must not cause a second contact manager.
        if(!std::is_sorted(destination.begin(),destination.end()))
            std::sort(destination.begin(),destination.end());
        destination.erase(
            std::unique(destination.begin(),destination.end()),
            destination.end());
    };
    lower(actor_found,aggregate_found,found);
    lower(actor_lost,aggregate_lost,lost);
}

inline void lower_source_sap_actor_reports_to_sc(
    std::span<const physx::PxgBroadPhasePair> source_found,
    std::span<const physx::PxgBroadPhasePair> source_lost,
    std::span<const BodyId> volume_to_actor,
    std::vector<native::Pair>& found,
    std::vector<native::Pair>& lost) {
    lower_source_sap_and_aggregate_reports_to_sc(
        source_found,source_lost,{},{},volume_to_actor,found,lost);
}
} // namespace fyzz::physx_gpu
