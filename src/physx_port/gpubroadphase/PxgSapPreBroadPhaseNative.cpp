// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"

#include <algorithm>
#include <array>
#include <bit>
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

// The source AABB manager supplies the changed bitmap and the original
// Bp update supplies created/updated handles. Bound sparse copies to 32
// runs so scattered changes avoid full transfers without an unbounded
// command list. More fragmented updates use the original full transfer.
// Enumerate changed source handles in ascending order.
std::optional<SourceSapDirtyRangesNative>
source_sap_dirty_ranges_native(const physx::Bp::BroadPhaseUpdateData& update,
                               std::span<const physx::PxU32> changed,
                               bool include_removed) {
    const auto capacity=update.getCapacity();
    const auto* created=update.getCreatedHandles();
    const auto* updated=update.getUpdatedHandles();
    const auto* removed=update.getRemovedHandles();
    const auto created_count=update.getNumCreatedHandles();
    const auto updated_count=update.getNumUpdatedHandles();
    const auto removed_count=include_removed?update.getNumRemovedHandles():0u;
    if((created_count && !created) || (updated_count && !updated) ||
       (removed_count && !removed))
        throw std::invalid_argument("Source GPU AABB delta handle array missing");

    SourceSapDirtyRangesNative runs{};
    bool overflow=false;
    physx::PxU32 ci=0u,ui=0u,ri=0u;
    const auto word_count=(std::uint64_t(capacity)+31u)/32u;
    for(std::size_t word=0;word<word_count;++word) {
        physx::PxU32 bits=word<changed.size()?changed[word]:0u;
        if(word+1u==word_count && (capacity&31u))
            bits&=(1u<<(capacity&31u))-1u;
        const auto collect=[&](const physx::Bp::ShapeHandle* handles,
                               physx::PxU32 count,physx::PxU32& cursor) {
            while(cursor<count && (std::uint64_t(handles[cursor])>>5u)==word) {
                const auto handle=handles[cursor];
                if(handle>=capacity || (cursor && handles[cursor-1u]>=handle))
                    throw std::invalid_argument(
                        "Source GPU AABB delta handles are not ordered");
                bits|=1u<<(handle&31u);
                ++cursor;
            }
        };
        collect(created,created_count,ci);
        collect(updated,updated_count,ui);
        if(include_removed)collect(removed,removed_count,ri);
        while(bits) {
            const auto handle=static_cast<physx::PxU32>(
                word*32u+std::countr_zero(bits));
            if(!overflow) {
                if(runs.count && runs.spans[runs.count-1u].second==handle)
                    ++runs.spans[runs.count-1u].second;
                else if(runs.count<runs.spans.size())
                    runs.spans[runs.count++]={handle,handle+1u};
                else overflow=true;
            }
            bits&=bits-1u;
        }
    }
    if(ci!=created_count || ui!=updated_count || ri!=removed_count)
        throw std::out_of_range(
            "Source GPU AABB delta handle outside capacity or unsorted");
    if(overflow)return std::nullopt;
    return runs;
}

void append_pre_broadphase_native_uploads(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    std::vector<SourceSapUpload>& uploads,
    std::span<const physx::PxU32> changed_handles,
    bool source_owned_changes,
    bool device_generated_bounds,
    bool aggregate_metadata_changes) {
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
    const bool metadata_resident=update.getStateChanged() &&
        capacity<=previously_published &&
        buffers.contact_distances.getDevicePtr()!=0u &&
        buffers.contact_distances.getSize()>=distances_bytes &&
        buffers.groups.getDevicePtr()!=0u &&
        buffers.groups.getSize()>=groups_bytes;
    // Retired handles and aggregate member admissions can change Bp
    // metadata outside the top-level changed-handle bitmap. Keep their
    // complete source transfer; standalone live edits can use a delta.
    const bool sparse_metadata=source_owned_changes && metadata_resident &&
        !aggregate_metadata_changes;
    const bool sparse_bounds=source_owned_changes && bounds_resident &&
        !device_generated_bounds;
    // Determine the interval before modifying any source heap allocation.
    // The same original Sc changed handles own bounds and Bp metadata.
    const auto dirty_ranges=(sparse_bounds || sparse_metadata)
        ? source_sap_dirty_ranges_native(update,changed_handles)
        : std::nullopt;
    const auto metadata_ranges=sparse_metadata && update.getNumRemovedHandles()
        ?source_sap_dirty_ranges_native(update,changed_handles,true)
        :dirty_ranges;

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
    if(update.getStateChanged() && !device_generated_bounds) {
        if(!sparse_bounds || !dirty_ranges) {
            uploads.push_back({buffers.bounds.getDevicePtr(),
                               update.getAABBs(),bounds_bytes});
        }else for(std::size_t i=0u;i<dirty_ranges->count;++i) {
            const auto [first,last]=dirty_ranges->spans[i];
            uploads.push_back({
                buffers.bounds.getDevicePtr()+
                    std::uint64_t(first)*sizeof(physx::PxBounds3),
                update.getAABBs()+first,
                std::size_t(last-first)*sizeof(physx::PxBounds3)});
        }
    }
    if(sparse_metadata && metadata_ranges) {
        for(std::size_t i=0u;i<metadata_ranges->count;++i) {
            const auto [first,last]=metadata_ranges->spans[i];
            uploads.push_back({
                buffers.contact_distances.getDevicePtr()+
                    std::uint64_t(first)*sizeof(physx::PxReal),
                update.getContactDistance()+first,
                std::size_t(last-first)*sizeof(physx::PxReal)});
        }
        for(std::size_t i=0u;i<metadata_ranges->count;++i) {
            const auto [first,last]=metadata_ranges->spans[i];
            uploads.push_back({
                buffers.groups.getDevicePtr()+
                    std::uint64_t(first)*sizeof(physx::PxU32),
                update.getGroups()+first,
                std::size_t(last-first)*sizeof(physx::PxU32)});
        }
    }else {
        uploads.push_back({buffers.contact_distances.getDevicePtr(),
                           update.getContactDistance(),distances_bytes});
        uploads.push_back({buffers.groups.getDevicePtr(),
                           update.getGroups(),groups_bytes});
    }
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
