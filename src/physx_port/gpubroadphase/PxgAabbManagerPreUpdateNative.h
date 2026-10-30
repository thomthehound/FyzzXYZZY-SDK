// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
#include "physx_port/lowlevelaabb/include/BpVolumeData.h"

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

// Borrowed PxgAABBManager bitmaps and their ORIGINAL device buffers.
// The manager owns handle admission, bitmap mutation and buffer lifetime.
struct SourceSapAabbManagerBitmapNative {
    physx::PxgCudaBuffer& device;
    std::span<const physx::PxU32> words;
};

// PxgAABBManager::preBpUpdate_GPU inputs. This does not construct a new
// AABB manager or derive its bitmaps from Scene geometry.
struct SourceSapAabbManagerPreUpdateNative {
    SourceSapPreBroadPhaseBuffers sap;
    SourceSapAabbManagerBitmapNative added;
    SourceSapAabbManagerBitmapNative removed;
    SourceSapAabbManagerBitmapNative aggregated;
    SourceSapAabbManagerBitmapNative changed;
    physx::PxgCudaBuffer& volume_data_device;
    std::span<const physx::Bp::VolumeData> volume_data;
    bool persistent_state_changed = false;
    bool gpu_state_changed = false;
    bool has_aggregates = false;
    // Borrow the same original GPU AABB owner for aggregate source stages.
    // Absent on ordinary single-volume scenes; no parallel aggregate path.
    const SourceAggregateGpuNativeStage* aggregate_stage = nullptr;
    const CudaDeviceAddressMap* aggregate_addresses = nullptr;
    physx::PxgCudaBuffer* aggregate_slots_device = nullptr;
    physx::PxgCudaBuffer* aggregate_members_device = nullptr;
    physx::PxU32 aggregate_slots_count = 0;
    physx::PxU32 aggregate_member_capacity = 0;
};

inline SourceSapAabbManagerDescriptorView
source_sap_aabb_descriptor_native(
    const SourceSapAabbManagerPreUpdateNative& manager) noexcept {
    return {
        manager.changed.device.getDevicePtr(),
        static_cast<physx::PxU32>(manager.changed.words.size()),
        manager.added.device.getDevicePtr(),
        manager.removed.device.getDevicePtr(),
        manager.aggregated.device.getDevicePtr(),
        manager.volume_data_device.getDevicePtr()
    };
}

// The original AABB-manager first-pass GPU prefix. The original source
// bitmaps must already reflect add/remove/dirty operations. Preserve:
// persistent added/removed maps -> aggregated map + volume data ->
// preBroadPhase -> changed map. All transfers use the same source stream;
// no host wait, second scheduler or CPU broadphase is introduced.
inline void pre_update_source_sap_aabb_manager_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapAabbManagerPreUpdateNative& manager,
    SourceSapDmaNative& stream) {
    if(manager.has_aggregates &&
       (!manager.aggregate_stage || !manager.aggregate_addresses ||
        !manager.aggregate_slots_device || !manager.aggregate_members_device ||
        !manager.aggregate_slots_count || !manager.aggregate_member_capacity))
        throw std::logic_error(
            "GPU aggregate update requires the source aggregate GPU stage and owner");
    const bool state_changed =
        manager.persistent_state_changed || update.getStateChanged();
    if(!state_changed && !manager.gpu_state_changed)return;

    // The AABB manager's bitmap uploads occur before preBroadPhase.
    // Reject bad bounds/group inputs first, while the source stream and
    // its persistent GPU histories are still untouched.
    if(state_changed)validate_pre_broadphase_native(update);
    const auto capacity=update.getCapacity();
    if(state_changed && manager.volume_data.size()<capacity)
        throw std::invalid_argument(
            "Source AABB manager volume data does not match bounds capacity");
    const auto checked_bytes=[](std::size_t elements,std::size_t stride){
        if(elements>std::numeric_limits<physx::PxU32>::max()/stride)
            throw std::overflow_error(
                "Source AABB manager GPU transfer exceeds PxU32 byte range");
        return elements*stride;
    };
    const auto validate_bitmap=[&](const SourceSapAabbManagerBitmapNative& bitmap){
        return checked_bytes(bitmap.words.size(),sizeof(physx::PxU32));
    };
    // Validate ALL borrowed host array lengths before the first source
    // heap allocation or transfer. Source bitmap word counts are preserved.
    const auto added_bytes=validate_bitmap(manager.added);
    const auto removed_bytes=validate_bitmap(manager.removed);
    const auto aggregate_bytes=validate_bitmap(manager.aggregated);
    const auto changed_bytes=validate_bitmap(manager.changed);
    const auto volume_bytes=checked_bytes(
        state_changed?capacity:0u,sizeof(physx::Bp::VolumeData));

    const auto append_bitmap=[](
        std::vector<SourceSapUpload>& uploads,
        const SourceSapAabbManagerBitmapNative& bitmap,
        std::size_t bytes) {
        if(bytes)
            uploads.push_back({
                bitmap.device.getDevicePtr(),bitmap.words.data(),bytes});
    };

    // These source DMA groups are adjacent on mStream. Allocate each
    // source-owned destination in original order, then snapshot the
    // complete HtoD sequence in one native queue submission.
    auto& uploads=stream.upload_scratch();
    uploads.reserve(10u);
    if(manager.persistent_state_changed) {
        manager.added.device.allocate(added_bytes,PX_FL);
        manager.removed.device.allocate(removed_bytes,PX_FL);
        append_bitmap(uploads,manager.added,added_bytes);
        append_bitmap(uploads,manager.removed,removed_bytes);
    }

    if(state_changed) {
        manager.aggregated.device.allocate(aggregate_bytes,PX_FL);
        manager.volume_data_device.allocate(volume_bytes,PX_FL);
        append_bitmap(uploads,manager.aggregated,aggregate_bytes);
        if(volume_bytes)
            uploads.push_back({manager.volume_data_device.getDevicePtr(),
                               manager.volume_data.data(),volume_bytes});

        // preBroadPhase follows the manager's original bitmap/volume
        // records without a kernel, copy or fence between them.
        append_pre_broadphase_native_uploads(update,manager.sap,uploads);
    }

    if(state_changed || manager.gpu_state_changed) {
        manager.changed.device.allocate(changed_bytes,PX_FL);
        append_bitmap(uploads,manager.changed,changed_bytes);
    }
    stream.queue_uploads(uploads);

    // PxgAABBManager::computeAggregateBounds (first pass), followed by
    // markAggregateBoundsBitmap (second pass before SAP update). The
    // already-uploaded source bounds, distances and dirty bitmap, plus
    // the owner's aggregate/member buffers, are passed unchanged.
    if(manager.has_aggregates) {
        const auto gpu=SourceAggregateGpuNativeBuffers{
            manager.sap.bounds.getDevicePtr(),
            manager.sap.contact_distances.getDevicePtr(),
            manager.aggregate_slots_device->getDevicePtr(),
            manager.aggregate_members_device->getDevicePtr(),
            manager.changed.device.getDevicePtr(),
            capacity,manager.aggregate_slots_count,
            manager.aggregate_member_capacity,
            static_cast<physx::PxU32>(manager.changed.words.size()),
            manager.aggregate_slots_count};
        manager.aggregate_stage->append(
            stream,*manager.aggregate_addresses,gpu);
    }
}

} // namespace fyzz::physx_gpu
