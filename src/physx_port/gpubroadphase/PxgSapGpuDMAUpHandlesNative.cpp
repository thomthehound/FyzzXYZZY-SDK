// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "BpBroadPhaseUpdate.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fyzz::physx_gpu {

SourceSapGpuDMAUpCounts gpu_dma_up_handles_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapGpuDMAUpHandleMembers members,
    SourceSapDmaNative& stream) {
    const auto created = update.getNumCreatedHandles();
    const auto removed = update.getNumRemovedHandles();
    const auto updated = update.getNumUpdatedHandles();
    const auto capacity = update.getCapacity();
    const auto previous = members.num_boxes;

    // Source formula: mNumOfBoxes += created - removed;
    // nbProjections = 2 * (mNumOfBoxes + removed).
    // Apply validity/overflow checks before touching GPU allocations, stream
    // submissions or any of the borrowed original member counters.
    if ((created && !update.getCreatedHandles()) ||
        (removed && !update.getRemovedHandles()) ||
        (updated && !update.getUpdatedHandles()))
        throw std::invalid_argument("PhysX gpuDMAUp missing nonempty handle input");
    if (removed > previous)
        throw std::invalid_argument("PhysX gpuDMAUp removes more handles than are live");

    const auto live = std::uint64_t(previous) + created - removed;
    const auto sourceProjections = (std::uint64_t(previous) + created) * 2u;
    if (live > capacity || sourceProjections > std::numeric_limits<physx::PxU32>::max() - 3u)
        throw std::overflow_error("PhysX gpuDMAUp box/projection sizing exceeds source range");
    const auto projections = static_cast<physx::PxU32>(sourceProjections);
    const auto padded = (projections + 3u) & ~physx::PxU32{3u};

    // These arrays are uploaded AFTER source allocation and construction of
    // PxgBroadPhaseDesc/PxgRadixSortDesc, but BEFORE descriptor HtoD. The
    // caller can now use the returned sizes for intermediate GPU buffers.
    members.created.allocate(std::uint64_t(created) * sizeof(physx::PxU32), PX_FL);
    members.removed.allocate(std::uint64_t(removed) * sizeof(physx::PxU32), PX_FL);
    members.updated.allocate(std::uint64_t(updated) * sizeof(physx::PxU32), PX_FL);

    // Adjacent HtoD commands share the original ordered native stream and
    // preserve host input snapshot semantics even if source arrays are reused.
    const std::vector<SourceSapUpload> uploads{
        {members.created.getDevicePtr(), update.getCreatedHandles(),
         std::size_t(created) * sizeof(physx::PxU32)},
        {members.removed.getDevicePtr(), update.getRemovedHandles(),
         std::size_t(removed) * sizeof(physx::PxU32)},
        {members.updated.getDevicePtr(), update.getUpdatedHandles(),
         std::size_t(updated) * sizeof(physx::PxU32)}
    };
    stream.queue_uploads(uploads);

    members.created_size = created;
    members.removed_size = removed;
    members.updated_size = updated;
    members.boxes_capacity = capacity;
    members.num_boxes = static_cast<physx::PxU32>(live);
    return {previous, members.num_boxes, capacity,
            created, removed, updated, projections, padded};
}

} // namespace fyzz::physx_gpu
