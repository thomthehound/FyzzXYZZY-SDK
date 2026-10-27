// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "BpBroadPhaseUpdate.h"
#include "BpVolumeData.h"

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
    if(manager.has_aggregates)
        throw std::logic_error(
            "GPU aggregate update requires the source aggregate owner");
    const bool state_changed =
        manager.persistent_state_changed || update.getStateChanged();
    if(!state_changed && !manager.gpu_state_changed)return;

    const auto capacity=update.getCapacity();
    if(state_changed && manager.volume_data.size()!=capacity)
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
        manager.volume_data.size(),sizeof(physx::Bp::VolumeData));

    const auto append_bitmap=[](
        std::vector<SourceSapUpload>& uploads,
        const SourceSapAabbManagerBitmapNative& bitmap,
        std::size_t bytes) {
        if(bytes)
            uploads.push_back({
                bitmap.device.getDevicePtr(),bitmap.words.data(),bytes});
    };

    if(manager.persistent_state_changed) {
        manager.added.device.allocate(added_bytes,PX_FL);
        manager.removed.device.allocate(removed_bytes,PX_FL);
        std::vector<SourceSapUpload> uploads;
        uploads.reserve(2u);
        append_bitmap(uploads,manager.added,added_bytes);
        append_bitmap(uploads,manager.removed,removed_bytes);
        stream.queue_uploads(uploads);
    }

    if(state_changed) {
        manager.aggregated.device.allocate(aggregate_bytes,PX_FL);
        manager.volume_data_device.allocate(volume_bytes,PX_FL);
        std::vector<SourceSapUpload> uploads;
        uploads.reserve(2u);
        append_bitmap(uploads,manager.aggregated,aggregate_bytes);
        if(volume_bytes)
            uploads.push_back({manager.volume_data_device.getDevicePtr(),
                               manager.volume_data.data(),volume_bytes});
        stream.queue_uploads(uploads);

        // preBroadPhase is conditional on manager persistent/bounds changes,
        // but it still uploads groups and contact distances if invoked.
        pre_broadphase_native(update,manager.sap,stream);
    }

    if(state_changed || manager.gpu_state_changed) {
        manager.changed.device.allocate(changed_bytes,PX_FL);
        std::vector<SourceSapUpload> uploads;
        uploads.reserve(1u);
        append_bitmap(uploads,manager.changed,changed_bytes);
        stream.queue_uploads(uploads);
    }
}

} // namespace fyzz::physx_gpu
