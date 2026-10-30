// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"

#include <cstdint>

namespace fyzz::physx_gpu {

// Source postBPStage2 continuation. The original GPU SAP descriptor owns
// the semantic current/previous integer AABB addresses across host swaps.
inline void append_source_sap_post_aggregate_member_history_native(
    const physx::PxgBroadPhaseDesc& descriptor,
    PxgAabbSourceMembersNative& aabb,
    const SourceAggregateMemberHistoryNativeStage& stage,
    SourceSapDmaNative& stream,
    const CudaDeviceAddressMap& addresses) {
    const auto gpu_address=[](auto pointer) {
        return static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(pointer));
    };
    const auto buffers=aabb.aggregate_member_history_buffers(
        gpu_address(descriptor.newIntegerBounds),
        gpu_address(descriptor.oldIntegerBounds));
    stage.append(stream,addresses,buffers);
}

} // namespace fyzz::physx_gpu
