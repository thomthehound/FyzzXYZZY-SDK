// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {
namespace {
// Checked once at the entrance and again before the asynchronous upload;
// source Bp::BroadPhaseUpdateData is a borrowed view and can be reused.
void check_handle_inputs(const physx::Bp::BroadPhaseUpdateData& update) {
    if ((update.getNumCreatedHandles() && !update.getCreatedHandles()) ||
        (update.getNumRemovedHandles() && !update.getRemovedHandles()))
        throw std::invalid_argument("PhysX gpuDMAUp missing nonempty handle input");
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    if (update.getNumUpdatedHandles() && !update.getUpdatedHandles())
        throw std::invalid_argument("PhysX gpuDMAUp missing nonempty updated-handle input");
#endif
}
}

physx::PxU64 checked_sap_region_slots_native(physx::PxU64 projections) {
    // Original SAP computes (nbProjections * 64 + 3) & ~3, then stores
    // the result in PxU32. Use 64-bit arithmetic and reject overflow
    // BEFORE committing any owning SAP counters or GPU history copies.
    if (projections > (std::numeric_limits<physx::PxU64>::max() - 3u)
                          / kSapRegionSlotsPerProjection)
        throw std::overflow_error("PhysX SAP region slot computation overflows PxU64");
    const auto expanded = projections * kSapRegionSlotsPerProjection;
    const auto slots = (expanded + 3u) & ~physx::PxU64{3u};
    if (slots > std::numeric_limits<physx::PxU32>::max())
        throw std::overflow_error("PhysX SAP region slot count exceeds PxU32");
    return slots;
}

SourceSapGpuDMAUpCounts prepare_gpu_dma_up_counts_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpCountMembers members) {
    check_handle_inputs(update);
    const auto created = update.getNumCreatedHandles();
    const auto removed = update.getNumRemovedHandles();
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    const auto updated = update.getNumUpdatedHandles();
#endif
    const auto capacity = update.getCapacity();
    const auto previous = members.num_boxes;

    // Original mNumOfBoxes += created - removed; projection history
    // temporarily includes removed handles and hence covers old + created.
    // Guard C++ unsigned wrap BEFORE mutating the borrowed source members.
    if (removed > previous)
        throw std::invalid_argument("PhysX gpuDMAUp removes more handles than are live");
    const std::uint64_t live = std::uint64_t(previous) + created - removed;
    const std::uint64_t total = std::uint64_t(previous) + created;
    const std::uint64_t projections = total * 2u;
    if (live > capacity ||
        projections > std::numeric_limits<physx::PxU32>::max() - 3u)
        throw std::overflow_error("PhysX gpuDMAUp box/projection sizing exceeds source range");
    // The post-axis allocator requires 64 region slots per projection.
    // Reject its known source PxU32 overflow before changing any borrowed
    // counters; otherwise retrying this frame applies creation twice.
    (void)checked_sap_region_slots_native(projections);
    const auto unpadded = static_cast<physx::PxU32>(projections);
    const auto padded = (unpadded + 3u) & ~physx::PxU32{3u};

    members.created_size = created;
    members.removed_size = removed;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    members.updated_size = updated;
#endif
    members.boxes_capacity = capacity;
    members.num_boxes = static_cast<physx::PxU32>(live);
    return {previous, members.num_boxes, capacity,
            created, removed,
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
            updated,
#endif
            unpadded, padded};
}

void allocate_gpu_dma_up_handles_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers) {
    // Original GPU allocator preserves capacity when requests shrink.
    // This takes place AFTER history growth and BEFORE updateDescriptor.
    buffers.created.allocate(std::uint64_t(counts.created) * sizeof(physx::PxU32), PX_FL);
    buffers.removed.allocate(std::uint64_t(counts.removed) * sizeof(physx::PxU32), PX_FL);
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    buffers.updated.allocate(std::uint64_t(counts.updated) * sizeof(physx::PxU32), PX_FL);
#endif
}

void append_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    std::vector<SourceSapUpload>& uploads) {
    check_handle_inputs(update);
    if (counts.created != update.getNumCreatedHandles() ||
        counts.removed != update.getNumRemovedHandles() ||
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        counts.updated != update.getNumUpdatedHandles() ||
#endif
        counts.capacity != update.getCapacity())
        throw std::invalid_argument("PhysX gpuDMAUp source handle counts changed after preparation");
    if (buffers.created.getSize() < std::uint64_t(counts.created)*sizeof(physx::PxU32) ||
        buffers.removed.getSize() < std::uint64_t(counts.removed)*sizeof(physx::PxU32))
        throw std::logic_error("PhysX gpuDMAUp handle buffers were not allocated");
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    if (buffers.updated.getSize() < std::uint64_t(counts.updated)*sizeof(physx::PxU32))
        throw std::logic_error("PhysX gpuDMAUp updated-handle buffer was not allocated");
#endif

    // Original source executes these adjacent HtoD calls after building the
    // descriptors. The stream snapshots the borrowed inputs before return.
    uploads.insert(uploads.end(), {
        {buffers.created.getDevicePtr(), update.getCreatedHandles(),
         std::size_t(counts.created)*sizeof(physx::PxU32)},
        {buffers.removed.getDevicePtr(), update.getRemovedHandles(),
         std::size_t(counts.removed)*sizeof(physx::PxU32)},
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        {buffers.updated.getDevicePtr(), update.getUpdatedHandles(),
         std::size_t(counts.updated)*sizeof(physx::PxU32)}
#endif
    });
}

void queue_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    SourceSapDmaNative& stream) {
    std::vector<SourceSapUpload> uploads;
    uploads.reserve(3u);
    append_gpu_dma_up_handles_native(update, counts, buffers, uploads);
    stream.queue_uploads(uploads);
}

} // namespace fyzz::physx_gpu
