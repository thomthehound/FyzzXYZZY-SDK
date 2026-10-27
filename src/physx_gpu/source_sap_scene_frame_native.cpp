// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_scene_frame_native.h"

#include <stdexcept>

namespace fyzz::physx_gpu {
SourceSapSceneFrameNativeResult advance_source_sap_scene_frame_native(
    SourceSapSceneFrameNativeMembers members,
    SourceSapDmaNative& stream,
    bool& force_update,
    const SourceSapUpdateNativeStages& stages,
    const CudaDeviceAddressMap& addresses,
    compute::Context& context) {

    // The original AABB manager has already consumed the prepared Sc
    // arrays in preBroadPhase(), on its own task chain. No scene geometry
    // is mutated after its GPU snapshot at this boundary.

    // Original PxgCudaBroadPhaseSap::update: gpuDMAUp always precedes
    // mForceUpdate gating. The same owner retains the stream and all
    // persistent allocations until its fetch continuation has completed.
    const auto admission=advance_source_sap_frame_native(
        members.aabb_update,members.dma,stream,force_update,stages,
        addresses,members.update);
    stream.wait_at_fetch();

    // Fetch publishes the ORIGINAL CPU destination arrays and rotates
    // all seven PxgCudaBufferN<6> endpoint histories exactly once.
    // Failed source-buffer validation does not publish either report.
    const auto fetched=fetch_source_sap_results(
        context,addresses,members.update.device_descriptor,
        members.dma.broadphase_descriptor,
        members.found_pairs,members.lost_pairs,
        members.abort_mode,members.histories,members.mapped_reports);
    if(fetched.overflow)
        throw std::overflow_error("Original GPU SAP found/lost report capacity overflow");
    lower_source_sap_actor_reports_to_sc(
        std::span<const physx::PxgBroadPhasePair>(
            members.found_pairs.data,fetched.found_actors),
        std::span<const physx::PxgBroadPhasePair>(
            members.lost_pairs.data,fetched.lost_actors),
        members.volume_to_actor,members.found,members.lost);
    return {admission,fetched};
}
} // namespace fyzz::physx_gpu
