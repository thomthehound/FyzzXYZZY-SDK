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

    // The source manager's handle/bitmap first pass has already created
    // this frame's authoritative Bp update. Validate its GPU dirty flag
    // before queuing any source heap allocation or transfer.
    if(members.aabb_manager.gpu_state_changed!=
       members.aabb_update.getGpuStateChanged())
        throw std::invalid_argument(
            "Source AABB manager GPU dirty flag disagrees with Bp frame");

    // PxgAABBManager::preBpUpdate_GPU executes before SAP update().
    // Its original device bitmap and VolumeData buffers are used by
    // the SAP descriptor, not a parallel scene-owned history.
    pre_update_source_sap_aabb_manager_native(
        members.aabb_update,members.aabb_manager,stream);
    const auto manager_descriptor=
        source_sap_aabb_descriptor_native(members.aabb_manager);
    auto dma=members.dma;
    dma.descriptor_inputs.aabb_manager=&manager_descriptor;

    // Original PxgCudaBroadPhaseSap::update: gpuDMAUp always precedes
    // mForceUpdate gating. The same owner retains the stream and all
    // persistent allocations until its fetch continuation has completed.
    const auto admission=advance_source_sap_frame_native(
        members.aabb_update,dma,stream,force_update,stages,
        addresses,members.update);
    stream.wait_at_fetch();

    // Fetch publishes the ORIGINAL CPU destination arrays and rotates
    // all seven PxgCudaBufferN<6> endpoint histories exactly once.
    // Failed source-buffer validation does not publish either report.
    const auto fetched=fetch_source_sap_results_native(
        context,addresses,members.update.device_descriptor,
        members.dma.broadphase_descriptor,
        members.found_pairs,members.lost_pairs,
        members.abort_mode,
        [histories=members.histories]() noexcept {
            swap_sap_endpoint_histories_after_fetch_native(histories);
        },
        members.mapped_reports,
        [&](const SourceSapFetchedCounts& counts) {
            if(counts.overflow)
                throw std::overflow_error(
                    "Original GPU SAP found/lost report capacity overflow");
            lower_source_sap_actor_reports_to_sc(
                std::span<const physx::PxgBroadPhasePair>(
                    members.found_pairs.data,counts.found_actors),
                std::span<const physx::PxgBroadPhasePair>(
                    members.lost_pairs.data,counts.lost_actors),
                members.volume_to_actor,members.found,members.lost);
        });
    return {admission,fetched};
}
} // namespace fyzz::physx_gpu
