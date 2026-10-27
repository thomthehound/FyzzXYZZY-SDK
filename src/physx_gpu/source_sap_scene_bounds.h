// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "BpBroadPhaseUpdate.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxMat33.h"
#include "foundation/PxTransform.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Borrowed source PxgAABBManager/Bp::BroadPhaseUpdateData arrays, indexed
// by ORIGINAL source volume handles. The AABB manager owns handle admission,
// removal, bitmap histories and the arrays themselves. This does not assign
// handles, create a separate broadphase or inflate the source bounds.
struct SourceSapSceneBounds {
    std::span<physx::PxBounds3> bounds;
    std::span<physx::PxReal> contact_distances;
    std::span<physx::Bp::FilterGroup::Enum> groups;
};

// Per-producer staging, retained across frames. This is not an AABB-manager
// history: only the original manager owns current/past volume membership.
struct SourceSapSceneBoundsScratch {
    std::vector<std::uint32_t> visit_epochs;
    std::vector<physx::PxBounds3> prepared_bounds;
    std::uint32_t epoch=0;
};

// Borrowed, ascending source PxgAABBManager/Bp::BroadPhaseUpdateData
// delta lists. These are NOT reconstructed from Sc geometry: only the
// original AABB manager knows which volumes were admitted and retired.
struct SourceSapSceneFrameChanges {
    std::span<const physx::Bp::ShapeHandle> created;
    std::span<const physx::Bp::ShapeHandle> updated;
    std::span<const physx::Bp::ShapeHandle> removed;
};

// Translates one shape per presently admitted native rigid actor into the
// original PxBounds3/FilterGroup representation used by preBroadPhase().
// The caller supplies the live source volume handles in the same order as
// boxes, and retains a scratch vector across frames. All inputs and handle
// uniqueness are checked before modifying any source array. Retired handles
// and their pre-existing source array entries are deliberately untouched.
inline void write_source_sap_scene_bounds(
    std::span<const native::GpuBroadphaseBox> boxes,
    std::span<const physx::Bp::ShapeHandle> source_handles,
    SourceSapSceneBounds output,
    SourceSapSceneBoundsScratch& scratch,
    const SourceSapSceneFrameChanges* changes = nullptr) {

    if(boxes.size()!=source_handles.size() ||
       output.bounds.size()!=output.contact_distances.size() ||
       output.bounds.size()!=output.groups.size() ||
       output.bounds.size()>std::numeric_limits<physx::PxU32>::max())
        throw std::invalid_argument("Source SAP scene array capacities or live handles disagree");

    // Reuse linear-time membership stamps rather than sorting the full
    // source handle population each frame. All writes remain deferred
    // until every geometry and group value has passed validation.
    scratch.visit_epochs.resize(output.bounds.size(),0u);
    scratch.prepared_bounds.resize(boxes.size());
    if(++scratch.epoch==0u) {
        std::fill(scratch.visit_epochs.begin(),scratch.visit_epochs.end(),0u);
        scratch.epoch=1u;
    }

    for(std::size_t i=0;i<boxes.size();++i) {
        const auto handle=source_handles[i];
        const auto& box=boxes[i];
        if(handle>=output.bounds.size() ||
           handle>=BP_INVALID_BP_HANDLE)
            throw std::out_of_range("Source SAP shape volume handle outside AABB manager");
        if(scratch.visit_epochs[handle]==scratch.epoch)
            throw std::invalid_argument("Source SAP assigned a volume handle twice");
        scratch.visit_epochs[handle]=scratch.epoch;
        const auto finite=[](float x){return std::isfinite(x);};
        const auto& p=box.position;
        const auto& e=box.half_extents;
        const auto& q=box.orientation;
        if(!finite(p.x)||!finite(p.y)||!finite(p.z) ||
           !finite(e.x)||!finite(e.y)||!finite(e.z) ||
           e.x<=0.0f||e.y<=0.0f||e.z<=0.0f ||
           !finite(q.x)||!finite(q.y)||!finite(q.z)||!finite(q.w) ||
           !finite(box.contact_offset)||box.contact_offset<0.0f)
            throw std::invalid_argument("Invalid native Sc geometry in source SAP AABB input");
        const double norm=double(q.x)*q.x+double(q.y)*q.y+
                          double(q.z)*q.z+double(q.w)*q.w;
        if(std::abs(norm-1.0)>1.0e-3)
            throw std::invalid_argument("Source SAP box orientation is not normalized");
        // Source Bp::getFilterGroup_Dynamics shifts (rigidId+1) by 3.
        // Prevent overflow and collisions with reserved aggregate groups.
        if(box.motion!=native::GpuBroadphaseBox::Motion::Static &&
           box.id>=((std::numeric_limits<physx::PxU32>::max()>>3)-1u))
            throw std::out_of_range("Source SAP rigid group ID exceeds encoded range");
        const physx::PxTransform pose(
            physx::PxVec3(p.x,p.y,p.z),physx::PxQuat(q.x,q.y,q.z,q.w));
        const auto bounds=physx::PxBounds3::poseExtent(
            pose,physx::PxVec3(e.x,e.y,e.z));
        if(!bounds.isFinite() || bounds.isEmpty())
            throw std::out_of_range("Source SAP box produces invalid world bounds");
        scratch.prepared_bounds[i]=bounds;
    }

    if(changes) {
        // Bp::BroadPhaseUpdateData requires ascending, mutually exclusive
        // delta streams. Validate all lists against the actual source
        // volume handles BEFORE committing the staged box geometry.
        const auto valid_stream=[&](std::span<const physx::Bp::ShapeHandle> handles,
                                    bool must_be_live) {
            physx::Bp::ShapeHandle previous=0;
            bool first=true;
            for(const auto handle:handles) {
                if(handle>=output.bounds.size() || handle>=BP_INVALID_BP_HANDLE)
                    throw std::out_of_range("Source SAP delta references an invalid volume");
                if(!first && previous>=handle)
                    throw std::invalid_argument("Source SAP delta handles are not ascending");
                first=false;
                previous=handle;
                const bool live=scratch.visit_epochs[handle]==scratch.epoch;
                if(live!=must_be_live)
                    throw std::invalid_argument(
                        "Source SAP delta does not agree with live AABB manager volumes");
            }
        };
        valid_stream(changes->created,true);
        valid_stream(changes->updated,true);
        valid_stream(changes->removed,false);
        const auto intersects=[](std::span<const physx::Bp::ShapeHandle> a,
                                 std::span<const physx::Bp::ShapeHandle> b) {
            std::size_t i=0,j=0;
            while(i<a.size() && j<b.size()) {
                if(a[i]==b[j]) return true;
                if(a[i]<b[j]) ++i;
                else ++j;
            }
            return false;
        };
        if(intersects(changes->created,changes->updated) ||
           intersects(changes->created,changes->removed) ||
           intersects(changes->updated,changes->removed))
            throw std::invalid_argument("Source SAP handle occurs in multiple delta lists");
    }

    for(std::size_t i=0;i<boxes.size();++i) {
        const auto handle=source_handles[i];
        const auto& b=boxes[i];
        output.bounds[handle]=scratch.prepared_bounds[i];
        output.contact_distances[handle]=b.contact_offset;
        output.groups[handle]=b.motion==native::GpuBroadphaseBox::Motion::Static
            ? physx::Bp::getFilterGroup_Statics()
            : physx::Bp::getFilterGroup_Dynamics(
                b.id,b.motion==native::GpuBroadphaseBox::Motion::Kinematic);
    }
}

// One source-ordered input boundary: validate original volume deltas,
// publish Sc geometry into the borrowed PxgAABBManager arrays, and
// construct the ORIGINAL Bp::BroadPhaseUpdateData for preBroadPhase()
// and gpuDMAUp(). The returned object borrows every pointer and the filter;
// the manager/owner must retain them through both calls. In particular,
// gpu_state_changed comes from the GPU manager, not an inferred CPU diff.
inline physx::Bp::BroadPhaseUpdateData prepare_source_sap_scene_update(
    std::span<const native::GpuBroadphaseBox> live_boxes,
    std::span<const physx::Bp::ShapeHandle> live_source_handles,
    SourceSapSceneBounds arrays,
    SourceSapSceneFrameChanges changes,
    const physx::Bp::BpFilter& filter,
    bool state_changed,
    bool gpu_state_changed,
    SourceSapSceneBoundsScratch& scratch) {
    write_source_sap_scene_bounds(
        live_boxes,live_source_handles,arrays,scratch,&changes);
    return physx::Bp::BroadPhaseUpdateData(
        changes.created.data(),static_cast<physx::PxU32>(changes.created.size()),
        changes.updated.data(),static_cast<physx::PxU32>(changes.updated.size()),
        changes.removed.data(),static_cast<physx::PxU32>(changes.removed.size()),
        arrays.bounds.data(),arrays.groups.data(),
        arrays.contact_distances.data(),
        static_cast<physx::PxU32>(arrays.bounds.size()),
        filter,state_changed,gpu_state_changed);
}
} // namespace fyzz::physx_gpu
