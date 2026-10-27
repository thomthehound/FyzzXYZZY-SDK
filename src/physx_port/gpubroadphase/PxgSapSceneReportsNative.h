// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_sap_fetch_native.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// The native PxgAABBManager's original volume-handle -> Sc BodyId lookup
// is owned by the GPU scene, including retired slots until Lost has been
// consumed. This is ONLY report translation, not another BP pair manager,
// allocator, broadphase driver, or publication scheduler.
//
// Called after the original source completion gate and
// fetch_source_sap_results() have published mFound/LostActorPairsMapped.
// Output vectors are persistent working storage owned by the GPU scene.
// No additional GPU download/fence or CPU overlap computation takes place.
inline native::GpuSapPairReports source_sap_actor_reports_to_sc(
    SourceSapFetchedCounts counts,
    SourceSapActorPairDestination found,
    SourceSapActorPairDestination lost,
    std::span<const BodyId> source_volume_to_actor,
    std::vector<native::Pair>& native_found,
    std::vector<native::Pair>& native_lost) {

    if (counts.overflow)
        throw std::overflow_error("Source GPU SAP pair reports overflowed");
    if ((counts.found_actors && (!found.data ||
            counts.found_actors > found.capacity)) ||
        (counts.lost_actors && (!lost.data ||
            counts.lost_actors > lost.capacity)))
        throw std::out_of_range("Invalid source GPU mapped pair count");

    const auto convert=[&](const physx::PxgBroadPhasePair* pairs,
                           std::uint32_t count,
                           std::vector<native::Pair>& out) {
        out.clear();
        out.reserve(count);
        for (std::uint32_t i=0;i<count;++i) {
            const auto& report=pairs[i];
            if(report.mVolA>=report.mVolB ||
               report.mVolB>=source_volume_to_actor.size())
                throw std::invalid_argument(
                    "Source GPU SAP report has invalid volume indices");
            const BodyId a=source_volume_to_actor[report.mVolA];
            const BodyId b=source_volume_to_actor[report.mVolB];
            if(a==std::numeric_limits<BodyId>::max() ||
               b==std::numeric_limits<BodyId>::max() || a==b)
                throw std::invalid_argument(
                    "Source GPU SAP report resolves to invalid native actor");
            out.push_back(a<b?native::Pair{a,b}:native::Pair{b,a});
        }
        // Original purged reports are sorted by SOURCE volume handles.
        // Native actor IDs may differ from reordered/reused volume slots.
        // Translate first, then sort by the authoritative Sc pair key.
        std::sort(out.begin(),out.end());
        if(std::adjacent_find(out.begin(),out.end())!=out.end())
            throw std::invalid_argument(
                "GPU source volumes alias duplicate native Sc actor pairs");
    };
    // An error here invalidates this source frame and the owning
    // Scene aborts its GPU history before any Sc report is published.
    convert(found.data,counts.found_actors,native_found);
    convert(lost.data,counts.lost_actors,native_lost);
    return {native_found,native_lost};
}
} // namespace fyzz::physx_gpu
