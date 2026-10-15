// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/sap_newbounds_sequence_arena.h"
#include "PxgBroadPhaseDesc.h"

namespace fyzz::physx_gpu {

// Lossless, source-field-based lowering of the actual PxgBroadPhaseDesc
// members consumed by the new-bounds GPU chain. The source CUDA allocations
// are slices of one *persistent* caller-owned native broadphase backing
// resource; no frame-time packing, per-kernel transfer, or shadow simulation
// state. Other source kernels will need their own explicit field contracts.
struct SourceNewBoundsArenaLayout {
    compute::BufferId backing = 0;
    std::uint32_t backing_bytes = 0;
    CudaDeviceAddressMap::Address descriptor = 0; // original mBPDescBuf
    std::uint32_t current_boxes = 0;
    std::uint32_t box_capacity = 0;
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
