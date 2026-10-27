// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "ScObjectIDTracker.h"
#include "BpFiltering.h"

#include <limits>
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
        native::GpuBroadphaseBox::Motion motion,
        void* element_sim) {
        if(!element_sim)
            throw std::invalid_argument(
                "Source Sc element requires its original ElementSim user data");
        // ScShapeSimBase::getBPGroup uses the original Bp filtering
        // encoding: all statics share eSTATICS; dynamic and kinematic
        // groups incorporate Sc's actor ID and element type. In
        // particular, different dynamics MUST NOT share eDYNAMICS_BASE.
        const bool is_static=motion==native::GpuBroadphaseBox::Motion::Static;
        if(!is_static &&
           actor>=((std::numeric_limits<physx::PxU32>::max() >>
                    BP_FILTERING_TYPE_SHIFT_BIT) -
                   physx::Bp::FilterGroup::eDYNAMICS_BASE))
            throw std::out_of_range(
                "Source Sc actor ID overflows original Bp filter group encoding");
        const auto group=physx::Bp::getFilterGroup(
            is_static,actor,
            motion==native::GpuBroadphaseBox::Motion::Kinematic);
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
