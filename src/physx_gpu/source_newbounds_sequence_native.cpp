// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_sequence_native.h"

#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

void SourceNewBoundsSequenceNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceNewBoundsSequenceNativeLayout layout) const {
    // Exactly the gate on PxgCudaBroadPhaseSap::generateNewPairsKernel().
    if (!source.numCreatedHandles) return;

    // Prepare all four commands off-list. A late failure resolving the
    // found-report pointer must not commit the preceding three commands.
    std::vector<compute::BatchCommand> pending;
    work_.append(pending, source, addresses,
        {layout.descriptor, layout.current_boxes, layout.box_capacity,
         layout.projection_slots, layout.region_work_capacity,
         layout.region_work_capacity});
    prefix_.append(pending, source, addresses,
        {layout.descriptor, layout.current_boxes, layout.region_work_capacity});
    pairs_.append(pending, source, addresses,
        {layout.descriptor, layout.current_boxes, layout.box_capacity,
         layout.projection_slots, layout.region_work_capacity,
         layout.region_work_capacity});
    if (pending.size() != 4u)
        throw std::logic_error("PhysX source new-bounds kernel launch sequence changed");

    commands.reserve(commands.size() + pending.size());
    for (auto& command : pending)
        commands.emplace_back(std::move(command));
}

} // namespace fyzz::physx_gpu
