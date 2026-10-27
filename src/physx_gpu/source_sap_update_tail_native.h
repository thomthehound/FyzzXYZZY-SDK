// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/source_region_newbounds_chain_native.h"
#include "physx_gpu/source_clear_new_flags_native.h"
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
namespace fyzz::physx_gpu {
// Original source update() TAIL after initializeSapBoxKernel:
// regions -> active lists -> new pairs -> integer bounds swap ->
// clearNewFlagKernel -> runCopyResultsKernel. The enclosing source owner
// must have appended all preceding GPU SAP operations to this SAME stream.
// No scheduler, buffer owner, submission, blocking wait or fetch here.
inline void append_source_sap_update_tail_native(
    SourceSapDmaNative& stream,
    const SourceRegionNewBoundsNativeStage& region,
    const SourceClearNewFlagsNativeStage& clear,
    const SourceSapReportNativeStage& reports,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceRegionNewBoundsNativeLayout region_layout,
    SourceClearNewFlagsNativeLayout clear_layout,
    SourceSapReportNativeLayout report_layout,
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds,
    bool execute_gpu_stages) {
    if(!execute_gpu_stages)return;
    std::vector<compute::BatchCommand> prepared;
    region.append(prepared,source,addresses,region_layout);
    const auto after_regions=prepared.size();
    clear.append(prepared,source,addresses,clear_layout);
    if(prepared.size()!=after_regions+(source.numCreatedHandles?1u:0u))
        throw std::logic_error("Original source BP_CLEAR_NEWFLAG dispatch missing");
    reports.append(prepared,source,addresses,report_layout);
    if(prepared.size()!=after_regions+(source.numCreatedHandles?1u:0u)+3u)
        throw std::logic_error("Original source SAP report dispatch count changed");
    // Resolve every later view BEFORE touching the original bounds owners
    // or committing any commands to the source mStream.
    auto& commands=stream.commands();
    commands.reserve(commands.size()+prepared.size());
    commands.insert(commands.end(),
        std::make_move_iterator(prepared.begin()),
        std::make_move_iterator(prepared.end()));
    // Original source swaps after new-pair generation and before flag
    // clearing/report dispatch. All GPU pointers have already been captured
    // from the current source descriptor, not from these owning wrappers.
    physx::PxgCudaBuffer::swapBuffer(new_integer_bounds,old_integer_bounds);
}
} // namespace fyzz::physx_gpu
