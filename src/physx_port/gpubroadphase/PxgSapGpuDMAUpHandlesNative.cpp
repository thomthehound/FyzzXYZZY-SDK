// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "BpBroadPhaseUpdate.h"

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
        (update.getNumRemovedHandles() && !update.getRemovedHandles()) ||
        (update.getNumUpdatedHandles() && !update.getUpdatedHandles()))
        throw std::invalid_argument("PhysX gpuDMAUp missing nonempty handle input");
}
}

SourceSapGpuDMAUpCounts prepare_gpu_dma_up_counts_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpCountMembers members) {
    check_handle_inputs(update);
    const auto created = update.getNumCreatedHandles();
    const auto removed = update.getNumRemovedHandles();
    const auto updated = update.getNumUpdatedHandles();
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
    const auto unpadded = static_cast<physx::PxU32>(projections);
    const auto padded = (unpadded + 3u) & ~physx::PxU32{3u};

    members.created_size = created;
    members.removed_size = removed;
    members.updated_size = updated;
    members.boxes_capacity = capacity;
    members.num_boxes = static_cast<physx::PxU32>(live);
    return {previous, members.num_boxes, capacity,
            created, removed, updated, unpadded, padded};
}

void allocate_gpu_dma_up_handles_native(
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers) {
    // Original GPU allocator preserves capacity when requests shrink.
    // This takes place AFTER history growth and BEFORE updateDescriptor.
    buffers.created.allocate(std::uint64_t(counts.created) * sizeof(physx::PxU32), PX_FL);
    buffers.removed.allocate(std::uint64_t(counts.removed) * sizeof(physx::PxU32), PX_FL);
    buffers.updated.allocate(std::uint64_t(counts.updated) * sizeof(physx::PxU32), PX_FL);
}

void queue_gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    SourceSapGpuDMAUpHandleBuffers buffers,
    SourceSapDmaNative& stream) {
    check_handle_inputs(update);
    if (counts.created != update.getNumCreatedHandles() ||
        counts.removed != update.getNumRemovedHandles() ||
        counts.updated != update.getNumUpdatedHandles() ||
        counts.capacity != update.getCapacity())
        throw std::invalid_argument("PhysX gpuDMAUp source handle counts changed after preparation");
    if (buffers.created.getSize() < std::uint64_t(counts.created)*sizeof(physx::PxU32) ||
        buffers.removed.getSize() < std::uint64_t(counts.removed)*sizeof(physx::PxU32) ||
        buffers.updated.getSize() < std::uint64_t(counts.updated)*sizeof(physx::PxU32))
        throw std::logic_error("PhysX gpuDMAUp handle buffers were not allocated");

    // Original source executes these adjacent HtoD calls after building the
    // descriptors. The stream snapshots the borrowed inputs before return.
    const std::vector<SourceSapUpload> uploads{
        {buffers.created.getDevicePtr(), update.getCreatedHandles(),
         std::size_t(counts.created)*sizeof(physx::PxU32)},
        {buffers.removed.getDevicePtr(), update.getRemovedHandles(),
         std::size_t(counts.removed)*sizeof(physx::PxU32)},
        {buffers.updated.getDevicePtr(), update.getUpdatedHandles(),
         std::size_t(counts.updated)*sizeof(physx::PxU32)}
    };
    stream.queue_uploads(uploads);
}

} // namespace fyzz::physx_gpu
