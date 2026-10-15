// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/source_overlap_work_native.h"
#include "physx_gpu/source_overlap_prefix_native.h"
#include "physx_gpu/source_newbounds_pairs_native.h"

namespace fyzz::physx_gpu {

// Capacities come from the original PxgCudaBroadPhaseSap allocation model.
// The GPU descriptor and all source virtual pointers remain caller-owned.
struct SourceNewBoundsSequenceNativeLayout {
    CudaDeviceAddressMap::Address descriptor = 0;
    std::uint32_t current_boxes = 0;
    std::uint32_t box_capacity = 0;
    std::uint32_t projection_slots = 0;       // retained removed projections included
    std::uint32_t region_work_capacity = 0;   // overlapCounts and handleRegions
};

// Only lower PxgCudaBroadPhaseSap::generateNewPairsKernel's four original
// launches into the caller's existing command list. The preceding region
// producer and the following PhysX scene/stream tasks remain authoritative.
// No submit, allocations, copies, CPU waits, or independent frame scheduler.
class SourceNewBoundsSequenceNativeStage final {
public:
    SourceNewBoundsSequenceNativeStage(
        const SourceOverlapWorkNativeStage& work,
        const SourceOverlapPrefixNativeStage& prefix,
        const SourceNewBoundsPairsNativeStage& pairs) noexcept
        : work_(work), prefix_(prefix), pairs_(pairs) {}

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceNewBoundsSequenceNativeLayout layout) const;
private:
    const SourceOverlapWorkNativeStage& work_;
    const SourceOverlapPrefixNativeStage& prefix_;
    const SourceNewBoundsPairsNativeStage& pairs_;
};
} // namespace fyzz::physx_gpu
