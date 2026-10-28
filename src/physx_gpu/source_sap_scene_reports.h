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
inline void lower_source_sap_actor_reports_to_sc(
    std::span<const physx::PxgBroadPhasePair> source_found,
    std::span<const physx::PxgBroadPhasePair> source_lost,
    std::span<const BodyId> volume_to_actor,
    std::vector<native::Pair>& found,
    std::vector<native::Pair>& lost) {
    if(&found==&lost)
        throw std::invalid_argument("Source GPU SAP pair outputs alias");

    // Verify BOTH result streams before modifying either consumer buffer.
    // PxgBroadPhasePair::mVolA/B are sorted by volume handle, which need
    // not preserve actor ID order across source volume recycling.
    const auto check=[&](std::span<const physx::PxgBroadPhasePair> pairs) {
        for(const auto& p:pairs) {
            if(p.mVolA>=p.mVolB ||
               p.mVolB>=volume_to_actor.size() ||
               volume_to_actor[p.mVolA]==std::numeric_limits<BodyId>::max() ||
               volume_to_actor[p.mVolB]==std::numeric_limits<BodyId>::max() ||
               volume_to_actor[p.mVolA]==volume_to_actor[p.mVolB])
                throw std::out_of_range(
                    "Original SAP result references an invalid source volume handle");
        }
    };
    check(source_found);
    check(source_lost);
    found.reserve(source_found.size());
    lost.reserve(source_lost.size());
    const auto lower=[&](std::span<const physx::PxgBroadPhasePair> source,
                         std::vector<native::Pair>& target) {
        target.clear();
        for(const auto& p:source) {
            const BodyId a=volume_to_actor[p.mVolA];
            const BodyId b=volume_to_actor[p.mVolB];
            target.push_back({std::min(a,b),std::max(a,b)});
        }
        // GPU report order is not necessarily actor order, but avoid an
        // O(n log n) sort when the lowered Sc keys are already ordered.
        if(!std::is_sorted(target.begin(),target.end()))
            std::sort(target.begin(),target.end());
        target.erase(std::unique(target.begin(),target.end()),target.end());
    };
    lower(source_found,found);
    lower(source_lost,lost);
}
} // namespace fyzz::physx_gpu
