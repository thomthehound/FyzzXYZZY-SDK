// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "BpBroadPhaseUpdate.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <span>

namespace fyzz::physx_gpu {

void pre_broadphase_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    SourceSapDmaNative& stream) {
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

    const auto bounds_bytes = capacity * sizeof(physx::PxBounds3);
    const auto distances_bytes = capacity * sizeof(physx::PxReal);
    const auto groups_bytes = capacity * sizeof(physx::PxU32);
    const auto env_bytes = capacity * sizeof(physx::PxU32);

    buffers.boxes_capacity = capacity;

    // Retain the original allocation conditions and free-first buffer
    // growth semantics. Inactive bounds keep their previous GPU contents;
    // absence of environment IDs does not reallocate or upload their buffer.
    buffers.contact_distances.allocate(distances_bytes, PX_FL);
    buffers.groups.allocate(groups_bytes, PX_FL);
    if (update.getEnvIDs())
        buffers.environment_ids.allocate(env_bytes, PX_FL);
    if (update.getStateChanged())
        buffers.bounds.allocate(bounds_bytes, PX_FL);

    // Four consecutive upstream memcpyHtoDAsync calls are one ordered
    // native upload group. Context snapshots the host bytes at enqueue,
    // and its queue/fences retain staging until PhysX's fetch boundary.
    // The four original transfers are a bounded source batch. Build
    // descriptors on the stack; stream-owned scratch retains capacity
    // across source frames without allocating per preBroadPhase call.
    std::array<SourceSapUpload,4> uploads{};
    std::size_t count=0;
    if (update.getStateChanged())
        uploads[count++]={buffers.bounds.getDevicePtr(),
                          update.getAABBs(), bounds_bytes};
    uploads[count++]={buffers.contact_distances.getDevicePtr(),
                      update.getContactDistance(), distances_bytes};
    uploads[count++]={buffers.groups.getDevicePtr(),
                      update.getGroups(), groups_bytes};
    if (update.getEnvIDs())
        uploads[count++]={buffers.environment_ids.getDevicePtr(),
                          update.getEnvIDs(), env_bytes};
    stream.queue_uploads(std::span<const SourceSapUpload>(uploads.data(),count));
}

} // namespace fyzz::physx_gpu
