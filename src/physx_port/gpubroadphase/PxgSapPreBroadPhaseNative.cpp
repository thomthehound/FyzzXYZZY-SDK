// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"

#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fyzz::physx_gpu {

void validate_pre_broadphase_native(
    const physx::Bp::BroadPhaseUpdateData& update) {
    // PxgCudaBroadPhaseSap::preBroadPhase uses PxU32 byte counts. Reject
    // unsupported capacities before allocations instead of wrapping them.
    static_assert(sizeof(physx::Bp::FilterGroup::Enum) == sizeof(physx::PxU32),
                  "PhysX group upload must retain the original 32-bit layout");
    const auto capacity = update.getCapacity();
    if (capacity > std::numeric_limits<physx::PxU32>::max() /
                       sizeof(physx::PxBounds3))
        throw std::overflow_error("PhysX preBroadPhase bounds byte count exceeds PxU32");
    if (capacity && (!update.getContactDistance() || !update.getGroups() ||
                     (update.getStateChanged() && !update.getAABBs())))
        throw std::invalid_argument("PhysX preBroadPhase missing source input array");

}

namespace {

// The source AABB manager supplies the changed bitmap, while the original
// Bp update supplies newly admitted and explicitly updated handles.
// Restrict the transfer to a single contiguous interval: one smaller HtoD
// replaces one full HtoD, without extra queue records or a shadow dirty list.
std::optional<std::pair<physx::PxU32,physx::PxU32>>
contiguous_dirty_bounds(const physx::Bp::BroadPhaseUpdateData& update,
                        std::span<const physx::PxU32> changed) {
    const auto capacity=update.getCapacity();
    if(changed.size()<(std::uint64_t(capacity)+31u)/32u)
        throw std::invalid_argument(
            "Source GPU changed map cannot address all AABB volumes");
    const auto* created=update.getCreatedHandles();
    const auto* updated=update.getUpdatedHandles();
    const auto created_count=update.getNumCreatedHandles();
    const auto updated_count=update.getNumUpdatedHandles();
    if((created_count && !created) || (updated_count && !updated))
        throw std::invalid_argument("Source GPU AABB delta handle array missing");
    physx::PxU32 first=capacity,last=0u;
    physx::PxU32 created_index=0u,updated_index=0u;
    bool gap=false;
    for(physx::PxU32 i=0u;i<capacity;++i) {
        const bool in_created=created_index<created_count &&
                              created[created_index]==i;
        const bool in_updated=updated_index<updated_count &&
                              updated[updated_index]==i;
        if(in_created)++created_index;
        if(in_updated)++updated_index;
        const bool dirty=(changed[i>>5u] & (1u<<(i&31u)))!=0u ||
                          in_created || in_updated;
        if(dirty) {
            if(gap)return std::nullopt;
            if(first==capacity)first=i;
            last=i+1u;
        }else if(first!=capacity)gap=true;
    }
    if(created_index!=created_count || updated_index!=updated_count)
        throw std::out_of_range("Source GPU AABB delta handle outside capacity");
    return std::pair{first,last};
}

} // namespace

void append_pre_broadphase_native_uploads(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    std::vector<SourceSapUpload>& uploads,
    std::span<const physx::PxU32> changed_handles) {
    validate_pre_broadphase_native(update);
    const auto capacity = update.getCapacity();
    const auto bounds_bytes = capacity * sizeof(physx::PxBounds3);
    const auto distances_bytes = capacity * sizeof(physx::PxReal);
    const auto groups_bytes = capacity * sizeof(physx::PxU32);
    const auto env_bytes = capacity * sizeof(physx::PxU32);

    // The source PxgCudaBuffer can reserve more bytes than the last
    // published logical array. Growth into that spare allocation still
    // needs a full initialization of the newly addressable volume slots.
    const auto previously_published=buffers.boxes_capacity;
    buffers.boxes_capacity = capacity;
    const bool bounds_resident=update.getStateChanged() &&
        capacity<=previously_published &&
        buffers.bounds.getDevicePtr()!=0u &&
        buffers.bounds.getSize()>=bounds_bytes;
    // Compute the interval before changing any source heap allocation.
    // An uninitialized or grown buffer still needs the complete upload.
    const auto dirty_interval=bounds_resident && !changed_handles.empty()
        ? contiguous_dirty_bounds(update,changed_handles)
        : std::nullopt;

    // Retain the original allocation conditions and free-first buffer
    // growth semantics. Inactive bounds keep their previous GPU contents;
    // absence of environment IDs does not reallocate or upload their buffer.
    buffers.contact_distances.allocate(distances_bytes, PX_FL);
    buffers.groups.allocate(groups_bytes, PX_FL);
    if (update.getEnvIDs())
        buffers.environment_ids.allocate(env_bytes, PX_FL);
    if (update.getStateChanged())
        buffers.bounds.allocate(bounds_bytes, PX_FL);

    // Adjacent source HtoD records have no intervening GPU work.
    // The enclosing AABB-manager pre-update owns their submission.
    uploads.reserve(uploads.size()+4u);
    if(update.getStateChanged()) {
        if(!bounds_resident || changed_handles.empty() || !dirty_interval) {
            uploads.push_back({buffers.bounds.getDevicePtr(),
                               update.getAABBs(),bounds_bytes});
        }else if(dirty_interval->first!=capacity) {
            const auto first=dirty_interval->first;
            const auto size=dirty_interval->second-first;
            uploads.push_back({
                buffers.bounds.getDevicePtr()+
                    std::uint64_t(first)*sizeof(physx::PxBounds3),
                update.getAABBs()+first,
                std::size_t(size)*sizeof(physx::PxBounds3)});
        }
    }
    uploads.push_back({buffers.contact_distances.getDevicePtr(),
                       update.getContactDistance(), distances_bytes});
    uploads.push_back({buffers.groups.getDevicePtr(),
                       update.getGroups(), groups_bytes});
    if (update.getEnvIDs())
        uploads.push_back({buffers.environment_ids.getDevicePtr(),
                           update.getEnvIDs(), env_bytes});
}

void pre_broadphase_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    SourceSapDmaNative& stream) {
    auto& uploads=stream.upload_scratch();
    append_pre_broadphase_native_uploads(update,buffers,uploads);
    stream.queue_uploads(uploads);
}

} // namespace fyzz::physx_gpu
