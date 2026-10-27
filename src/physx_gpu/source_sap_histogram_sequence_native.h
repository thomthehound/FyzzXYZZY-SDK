// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sap_endpoint_histograms_native.h"
#include "physx_gpu/source_sap_incremental_counts_native.h"
#include "physx_gpu/source_sap_incremental_pairs_native.h"
#include <iterator>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// The source update() interval between WR sorting and the
// post-incremental removal/creation handling. The HISTOGRAM operation
// runs on every update. GPU-changed frames append both incremental
// comparison passes AND the original pair producer as one transaction.
// The caller has already queued WR sorting on the same mStream.
inline void append_source_sap_incremental_phase_native(
    std::vector<compute::BatchCommand>& commands,
    const SourceSapEndpointHistogramsNativeStage& histograms,
    const SourceSapIncrementalCountsNativeStage& comparisons,
    const SourceSapIncrementalPairsNativeStage& pairs,
    const physx::PxgBroadPhaseDesc& descriptor,
    CudaDeviceAddressMap::Address device_descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout histogram_layout,
    SourceSapIncrementalCountsLayout comparison_layout,
    SourceSapIncrementalPairsNativeLayout pair_layout,
    bool gpu_state_changed,bool execute_gpu_stages) {
    if(!execute_gpu_stages)return;
    std::vector<compute::BatchCommand> prepared;
    prepared.reserve(gpu_state_changed?5u:2u);
    histograms.append(prepared,descriptor,addresses,histogram_layout,true);
    if(gpu_state_changed) {
        comparisons.append(prepared,descriptor,device_descriptor,addresses,
                           comparison_layout);
        pairs.append(prepared,descriptor,device_descriptor,addresses,
                     pair_layout);
    }
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
    std::vector<compute::BatchCommand>& commands,
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
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
                    std::make_move_iterator(prepared.begin()),
                    std::make_move_iterator(prepared.end()));
}

inline void append_source_sap_incremental_phase_native(
    SourceSapDmaNative& stream,
    const SourceSapEndpointHistogramsNativeStage& histograms,
    const SourceSapIncrementalCountsNativeStage& comparisons,
    const SourceSapIncrementalPairsNativeStage& pairs,
    const physx::PxgBroadPhaseDesc& descriptor,
    CudaDeviceAddressMap::Address device_descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout histogram_layout,
    SourceSapIncrementalCountsLayout comparison_layout,
    SourceSapIncrementalPairsNativeLayout pair_layout,
    bool gpu_state_changed,bool execute_gpu_stages) {
    append_source_sap_incremental_phase_native(
        stream.commands(),histograms,comparisons,pairs,descriptor,
        device_descriptor,addresses,histogram_layout,comparison_layout,
        pair_layout,gpu_state_changed,execute_gpu_stages);
}

inline void append_source_sap_final_histograms_native(
    SourceSapDmaNative& stream,
    const SourceSapEndpointHistogramsNativeStage& histograms,
    const physx::PxgBroadPhaseDesc& descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout layout,
    bool execute_gpu_stages) {
    append_source_sap_final_histograms_native(
        stream.commands(),histograms,descriptor,addresses,layout,
        execute_gpu_stages);
}
} // namespace fyzz::physx_gpu
