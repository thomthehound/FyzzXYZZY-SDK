// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/sap_newbounds_sequence_arena.h"
#include "PxgBroadPhaseDesc.h"

namespace fyzz::physx_gpu {

// EXPERIMENTAL, TEST-ONLY fast path for the actual PxgBroadPhaseDesc
// fields consumed by the new-bounds GPU chain. This works only when every
// required source CUDA allocation is a slice of ONE caller-owned backing.
// The original PhysX buddy heap normally admits MULTIPLE roots and large
// independent allocations, so this is NOT the production binding strategy
// or an acceptance test for a live GPU scene. No copies are hidden here.
struct SourceNewBoundsArenaLayout {
    compute::BufferId backing = 0;
    std::uint32_t backing_bytes = 0;
    CudaDeviceAddressMap::Address descriptor = 0; // original mBPDescBuf
    std::uint32_t current_boxes = 0;
    std::uint32_t box_capacity = 0;
    // Original host uses (current boxes + removed handles)*2 when
    // constructing sorted/histogram arrays; never infer from capacity.
    std::uint32_t projection_slots = 0;
    // These are SOURCE ALLOCATION CAPACITIES, not d.regionAccumTotal.
    // The latter is written by GPU kernels to the resident descriptor and
    // must never be read back merely to rebuild its native binding views.
    std::uint32_t overlap_prefix_capacity = 0; // uint64 elements
    std::uint32_t handle_region_capacity = 0; // {handle, region} entries
};

// Throws if any required source pointer refers to another backing resource,
// an invalid/freed slice, an unaligned address, or a range beyond 32-bit arena
// addressing. Never silently copies individual PhysX allocations into a
// temporary staging buffer merely to satisfy a translated shader.
SapNewBoundsSequenceArenaDescriptors lower_source_newbounds_descriptor(
    const physx::PxgBroadPhaseDesc& original,
    const CudaDeviceAddressMap& addresses,
    const SourceNewBoundsArenaLayout& layout);

} // namespace fyzz::physx_gpu
