// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_update_head_native.h"
#include "physx_gpu/source_sap_histogram_sequence_native.h"
#include "physx_gpu/source_sap_update_middle_native.h"
#include "physx_gpu/source_sap_update_tail_native.h"
#include "physx_port/gpubroadphase/PxgSapUpdateFetchNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"

#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {
struct SourceSapGpuBoxTransformsNative;

// Borrowed GPU stages corresponding to the original sequence of calls in
// PxgCudaBroadPhaseSap::update(). The source owner retains all pipelines,
// source buffers, descriptor lifetimes, and the native mStream.
struct SourceSapUpdateNativeStages {
    const CudaAabbLaunchLowering& translate;
    const SourceSapRemovedNativeStage& removed;
    const SourceSapUpdatedNativeStage& updated;
    const SourceSapRadixNativeStage& radix;
    const SourceSapEndpointHistogramsNativeStage& histograms;
    const SourceSapIncrementalCountsNativeStage& comparisons;
    const SourceSapIncrementalPairsNativeStage& incremental_pairs;
    const SourceSapBoxIndicesNativeStage& indices;
    const SourceSapCreatedNativeStage& created;
    const SourceRegionNewBoundsNativeStage& new_pairs;
    const SourceClearNewFlagsNativeStage& clear;
    const SourceSapReportNativeStage& reports;
};

// Extents and stage descriptors come directly from the source gpuDMAUp
// allocations. The caller must finish preBroadPhase/gpuDMAUp before this
// update and must pass the CURRENT source descriptor for every frame.
struct SourceSapUpdateNativeFrame {
    SourceSapTranslateAabbsNativeInput translation{};
    SourceSapRemovedNativeLayout removed{};
    SourceSapUpdatedNativeLayout updated{};
    SourceSapEndpointHistogramsNativeLayout histograms{};
    SourceSapIncrementalCountsLayout comparisons{};
    SourceSapIncrementalPairsNativeLayout incremental_pairs{};
    SourceSapBoxIndicesNativeLayout indices{};
    SourceSapCreatedNativeLayout created{};
    SourceRegionNewBoundsNativeLayout new_pairs{};
    SourceClearNewFlagsNativeLayout clear{};
    SourceSapReportNativeLayout reports{};
    CudaDeviceAddressMap::Address device_descriptor=0;
};

// Derive stage views from this frame's actual source gpuDMAUp counts and
// resident descriptor. The production owner never supplies test-sized
// addresses or capacities for its native GPU execution.
SourceSapUpdateNativeFrame make_source_sap_update_frame_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    const SourceSapGpuDMAUpCounts& counts,
    const physx::PxgBroadPhaseDesc& descriptor,
    CudaDeviceAddressMap::Address device_descriptor);

// A single native lowering of the source update() call, AFTER gpuDMAUp.
// The complete GPU command sequence is built before ANY command is
// appended to mStream, and before PhysX ownership/force-update state
// advances. There is no submission, device readback or second scheduler.
//
// GPU descriptor pointers are captured from this frame BEFORE the
// PxgCudaBuffer integer-bounds ownership swap. The source performs that
// swap after generating new pairs; clear/report commands follow in the
// same ordered stream. The next frame's gpuDMAUp will construct fresh
// descriptors from the exchanged owners. Fetch handles the seven
// endpoint/history swaps separately, after successful publication.
bool append_source_sap_update_native(
    SourceSapDmaNative& stream,
    const SourceSapUpdateNativeStages& stages,
    const physx::Bp::BroadPhaseUpdateData& update,
    bool& force_update,
    const physx::PxgBroadPhaseDesc& descriptor,
    std::span<const physx::PxgRadixSortDesc,6> wr_radix,
    std::span<const physx::PxgRadixSortDesc,6> wor_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateNativeFrame& frame,
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds);

// PhysX's gpuDMAUp update decision has ALREADY been evaluated by
// begin_sap_update_native(). Consume that admission without evaluating
// mForceUpdate again. In particular, a forced final quiet frame cannot
// be converted into a second state transition at this boundary.
bool append_source_sap_admitted_update_native(
    SourceSapDmaNative& stream,
    const SourceSapUpdateNativeStages& stages,
    bool execute_gpu_stages,
    bool gpu_state_changed,
    const physx::Bp::BroadPhaseUpdateData& update,
    const physx::PxgBroadPhaseDesc& descriptor,
    std::span<const physx::PxgRadixSortDesc,6> wr_radix,
    std::span<const physx::PxgRadixSortDesc,6> wor_radix,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateNativeFrame& frame,
    physx::PxgCudaBuffer& new_integer_bounds,
    physx::PxgCudaBuffer& old_integer_bounds);

// Contiguous source update() call: gpuDMAUp (including six original
// HtoD record uploads) -> the SINGLE PhysX forced/quiet decision ->
// all native GPU stages. Fetch remains the existing source continuation.
// Original members, stream, allocations and command ordering are borrowed.
SourceSapUpdateAdmission advance_source_sap_frame_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapDmaUpFrameMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    const SourceSapUpdateNativeFrame& frame,
    const SourceSapGpuBoxTransformsNative* gpu_box_transforms=nullptr);
} // namespace fyzz::physx_gpu
