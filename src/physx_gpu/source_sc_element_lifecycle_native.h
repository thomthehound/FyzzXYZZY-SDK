// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "ScObjectIDTracker.h"

#include <stdexcept>

namespace fyzz::physx_gpu {

// Original Sc::ElementSim / Sc::Scene element-ID lifecycle for the source
// single-shape GPU AABB path. Sc::ObjectIDTracker owns the IDs; the AABB
// manager owns the matching VolumeData and bitmaps. Actor IDs are NEVER
// treated as shape volume handles.
//
// The caller is the Sc scene's element-admission path and retains the
// ObjectIDTracker through postReportsCleanup. These transitions do not
// allocate an additional handle pool or retain a parallel live registry.
class SourceScElementLifecycleNative final {
public:
    static physx::Bp::ShapeHandle admit(
        physx::Sc::ObjectIDTracker& elements,
        PxgAabbSourceMembersNative& aabb,
        BodyId actor,
        physx::PxReal contact_distance,
        physx::Bp::FilterGroup::Enum group,
        void* element_sim) {
        if(!element_sim)
            throw std::invalid_argument(
                "Source Sc element requires its original ElementSim user data");
        // Sc::ElementSim::initID acquires the ID before BoundsArray::initEntry.
        const auto handle=elements.createID();
        try {
            aabb.admit(handle,actor,contact_distance,group,element_sim);
        } catch(...) {
            // Sc releases a failed element ID through its deferred pool.
            // Do not expose a recycled ID before postReportsCleanup.
            elements.releaseID(handle);
            throw;
        }
        return handle;
    }

    static bool remove(physx::Sc::ObjectIDTracker& elements,
                       PxgAabbSourceMembersNative& aabb,
                       physx::Bp::ShapeHandle handle) {
        // Sc::ElementSim::removeFromAABBMgr precedes its destructor's
        // releaseID; original ObjectIDTracker defers freeID until scene
        // postReportsCleanup. GPU lost reports still have the source
        // AABB manager's retained actor mapping through fetch.
        const bool canceled=aabb.remove(handle);
        elements.releaseID(handle);
        return canceled;
    }

    static void post_reports_cleanup(
        physx::Sc::ObjectIDTracker& elements) {
        // Sc::Scene::postReportsCleanup ordering: only after source SAP
        // GPU fetch and lost-pair publication, plus AABB bitmap/identity
        // retirement. This is the original IDPool deferred-free barrier.
        elements.processPendingReleases();
        elements.clearDeletedIDMap();
    }
};

} // namespace fyzz::physx_gpu
