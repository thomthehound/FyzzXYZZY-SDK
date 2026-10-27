// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sap_endpoint_histograms_native.h"
#include "physx_gpu/source_sap_incremental_counts_native.h"
#include <iterator>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Between WR sort and BP_INCREMENTAL_SAP, source update() always
// builds current (and, for removals, prior) endpoint histograms.
// Comparison prefixes are gated by getGpuStateChanged(), exactly as
// in PxgCudaBroadPhaseSap::performIncrementalSapKernel.
// The subsequent pair kernel consumes this same descriptor and stream.
inline void append_source_sap_incremental_histograms_native(
    SourceSapDmaNative& stream,
    const SourceSapEndpointHistogramsNativeStage& histograms,
    const SourceSapIncrementalCountsNativeStage& comparisons,
    const physx::PxgBroadPhaseDesc& descriptor,
    CudaDeviceAddressMap::Address device_descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout histogram_layout,
    SourceSapIncrementalCountsLayout comparison_layout,
    bool gpu_state_changed,bool execute_gpu_stages) {
    if(!execute_gpu_stages)return;
    std::vector<compute::BatchCommand> prepared;
    prepared.reserve(gpu_state_changed?4u:2u);
    histograms.append(prepared,descriptor,addresses,histogram_layout,true);
    if(gpu_state_changed)
        comparisons.append(prepared,descriptor,device_descriptor,addresses,
                           comparison_layout);
    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
                    std::make_move_iterator(prepared.begin()),
                    std::make_move_iterator(prepared.end()));
}

// Called after created/removed WOR sort and before the source final
// SAP index rebuild. The source runs the second histogram only if
// creation or removal was present. No duplicate histogram on quiet,
// motion-only or forced-flush-only frames.
inline void append_source_sap_final_histograms_native(
    SourceSapDmaNative& stream,
    const SourceSapEndpointHistogramsNativeStage& histograms,
    const physx::PxgBroadPhaseDesc& descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout layout,
    bool execute_gpu_stages) {
    if(!execute_gpu_stages ||
       (!descriptor.numCreatedHandles && !descriptor.numRemovedHandles))
        return;
    std::vector<compute::BatchCommand> prepared;
    prepared.reserve(2u);
    histograms.append(prepared,descriptor,addresses,layout,false);
    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
                    std::make_move_iterator(prepared.begin()),
                    std::make_move_iterator(prepared.end()));
}
} // namespace fyzz::physx_gpu
