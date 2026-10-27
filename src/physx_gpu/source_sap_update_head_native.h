// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_translate_aabbs_native.h"
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_sap_updated_native.h"
#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap::update, before the incremental endpoint
// histograms. The caller must have completed preBroadPhase/gpuDMAUp on
// this source stream and must retain the original scene/AABB manager,
// descriptor and six radix descriptors for the whole frame.
struct SourceSapUpdateHeadNative {
    SourceSapTranslateAabbsNativeInput translation{};
    SourceSapRemovedNativeLayout removed{};
    SourceSapUpdatedNativeLayout updated{};
    std::uint32_t previous_boxes=0u;
    bool gpu_state_changed=false;
    bool execute_gpu_stages=false; // source mForceUpdate/idle update decision
};

// translateAABBs -> markRemovedPairs -> markUpdatedPairs (GPU state gate)
// -> sortProjectionAndHandlesWR: rank init, stable eight-pass radix,
// previous-handle gather. The incremental histograms and pair producer
// append AFTER this function, on the very same SourceSapDmaNative stream.
// No new allocations, CPU upload/readback, submission or wait.
inline void append_source_sap_update_head_native(
    std::vector<compute::BatchCommand>& commands,
    const CudaAabbLaunchLowering& translate,
    const SourceSapRemovedNativeStage& deletion,
    const SourceSapUpdatedNativeStage& changed,
    const SourceSapRadixNativeStage& sort,
    const physx::PxgBroadPhaseDesc& source,
    std::span<const physx::PxgRadixSortDesc,6> wr_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateHeadNative& frame) {
    if(!frame.execute_gpu_stages)return;
    if(frame.previous_boxes!=source.numPreviousHandles)
        throw std::invalid_argument("Original source SAP previous-volume count disagrees");
    std::vector<compute::BatchCommand> prepared;
    prepared.reserve(22u);
    append_source_sap_translate_aabbs_native(
        prepared,translate,frame.translation,true);
    deletion.append_mark_removed(prepared,source,addresses,frame.removed);
    if(frame.gpu_state_changed)
        changed.append(prepared,source,addresses,frame.updated);
    sort.append_wr(prepared,source,wr_radix,addresses,frame.previous_boxes);
    // If any late source descriptor is invalid, none of the commands
    // above are committed to mStream. Existing queued source DMA stays
    // untouched, preserving PhysX's original transfer/dependency order.
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
        std::make_move_iterator(prepared.begin()),
        std::make_move_iterator(prepared.end()));
}

inline void append_source_sap_update_head_native(
    SourceSapDmaNative& stream,
    const CudaAabbLaunchLowering& translate,
    const SourceSapRemovedNativeStage& deletion,
    const SourceSapUpdatedNativeStage& changed,
    const SourceSapRadixNativeStage& sort,
    const physx::PxgBroadPhaseDesc& source,
    std::span<const physx::PxgRadixSortDesc,6> wr_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateHeadNative& frame) {
    append_source_sap_update_head_native(stream.commands(),translate,deletion,
        changed,sort,source,wr_radix,addresses,frame);
}
} // namespace fyzz::physx_gpu
