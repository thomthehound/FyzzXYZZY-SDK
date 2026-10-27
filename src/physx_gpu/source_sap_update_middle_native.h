// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sapbox_indices_native.h"
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// PxgCudaBroadPhaseSap::update AFTER performIncrementalSapKernel and
// BEFORE the second endpoint histogram and final SAP index rebuild.
// This is source-owned continuation of the same stream, not a new
// broadphase/simulation scheduler.
struct SourceSapUpdateMiddleNative {
    SourceSapBoxIndicesNativeLayout previous_indices{};
    SourceSapRemovedNativeLayout removed{};
    SourceSapCreatedNativeLayout created{};
    bool execute_gpu_stages=false; // original update/mForceUpdate gate
};

inline void append_source_sap_update_middle_native(
    SourceSapDmaNative& stream,
    const SourceSapBoxIndicesNativeStage& indices,
    const SourceSapRemovedNativeStage& deletion,
    const SourceSapCreatedNativeStage& insertion,
    const SourceSapRadixNativeStage& sort,
    const physx::PxgBroadPhaseDesc& source,
    std::span<const physx::PxgRadixSortDesc,6> wor_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateMiddleNative& frame) {
    if(!frame.execute_gpu_stages)return;
    const auto previous=source.numPreviousHandles;
    const auto created=source.numCreatedHandles;
    const auto removed=source.numRemovedHandles;
    const std::uint64_t retained=std::uint64_t(previous)+created;
    if(retained<removed || retained>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Source SAP created/removed population inconsistent");

    std::vector<compute::BatchCommand> prepared;
    if(removed) {
        // Source initializeSapBoxKernel(previous, false), then
        // markRemovedPairsProjectionsKernel. Invalid projection indices
        // go to the end of the subsequent WOR sort.
        indices.append(prepared,source,addresses,
                       frame.previous_indices,previous,false);
        deletion.append_remove_projections(
            prepared,source,addresses,frame.removed);
    }
    if(created || removed) {
        // Source markCreatedPairsKernel is gated by created handles.
        // The second radix sort is gated by created OR removed handles,
        // and sorts the CURRENT handle payloads, never WR rank indices.
        insertion.append(prepared,source,addresses,frame.created);
        sort.append_wor(prepared,wor_radix,addresses,previous,created);
    }

    // Do not submit or wait here. Atomic batch construction preserves
    // the original source array identities and all later fetch semantics.
    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
        std::make_move_iterator(prepared.begin()),
        std::make_move_iterator(prepared.end()));
}
} // namespace fyzz::physx_gpu
