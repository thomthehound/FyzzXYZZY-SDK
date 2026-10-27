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
    std::vector<physx::Bp::ShapeHandle>& handle_scratch) {

    if(boxes.size()!=source_handles.size() ||
       output.bounds.size()!=output.contact_distances.size() ||
       output.bounds.size()!=output.groups.size() ||
       output.bounds.size()>std::numeric_limits<physx::PxU32>::max())
        throw std::invalid_argument("Source SAP scene array capacities or live handles disagree");

    // Validate the complete batch before modifying even one source AABB.
    // This includes duplicate source handles if the AABB manager has
    // accidentally reissued a handle during the same publication epoch.
    handle_scratch.assign(source_handles.begin(),source_handles.end());
    std::sort(handle_scratch.begin(),handle_scratch.end());
    if(std::adjacent_find(handle_scratch.begin(),handle_scratch.end())
       !=handle_scratch.end())
        throw std::invalid_argument("Source SAP assigned a volume handle twice");

    for(std::size_t i=0;i<boxes.size();++i) {
        const auto handle=source_handles[i];
        const auto& box=boxes[i];
        if(handle>=output.bounds.size() ||
           handle>=physx::Bp::BP_INVALID_BP_HANDLE)
            throw std::out_of_range("Source SAP shape volume handle outside AABB manager");
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
    }

    for(std::size_t i=0;i<boxes.size();++i) {
        const auto handle=source_handles[i];
        const auto& b=boxes[i];
        const physx::PxTransform pose(
            physx::PxVec3(b.position.x,b.position.y,b.position.z),
            physx::PxQuat(b.orientation.x,b.orientation.y,
                          b.orientation.z,b.orientation.w));
        output.bounds[handle]=physx::PxBounds3::poseExtent(
            pose,physx::PxVec3(b.half_extents.x,b.half_extents.y,
                               b.half_extents.z));
        output.contact_distances[handle]=b.contact_offset;
        output.groups[handle]=b.motion==native::GpuBroadphaseBox::Motion::Static
            ? physx::Bp::getFilterGroup_Statics()
            : physx::Bp::getFilterGroup_Dynamics(
                b.id,b.motion==native::GpuBroadphaseBox::Motion::Kinematic);
    }
}
} // namespace fyzz::physx_gpu
