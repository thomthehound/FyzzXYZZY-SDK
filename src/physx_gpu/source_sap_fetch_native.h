// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/source_sap_fetch_core_native.h"
#include "physx_gpu/sap_host_frame_roles.h"
#include "physx_port/gpubroadphase/PxgSapUpdateFetchNative.h"

namespace fyzz::physx_gpu {

// Legacy Context-side frame role view retained for source-independent
// broadphase tests and non-PhysX callers.
inline SourceSapFetchedCounts fetch_source_sap_results(
    compute::Context& context,
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address descriptor_address,
    physx::PxgBroadPhaseDesc& host_descriptor,
    SourceSapActorPairDestination found,
    SourceSapActorPairDestination lost,
    bool abort_mode,
    SapFrameBufferRoles& roles,
    SourceSapMappedReportView* mapped_roots = nullptr) {
    return fetch_source_sap_results_native(context,addresses,
        descriptor_address,host_descriptor,found,lost,abort_mode,
        [&roles]() noexcept {roles.swap_histories_after_fetch();},
        mapped_roots);
}

// Production source fetch rotates the actual seven PxgCudaBufferN<6>
// owners after GPU completion and BOTH pair destinations have been
// validated/published. There is no parallel BufferId history to keep
// synchronized with the real PhysX allocator.
inline SourceSapFetchedCounts fetch_source_sap_results(
    compute::Context& context,
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address descriptor_address,
    physx::PxgBroadPhaseDesc& host_descriptor,
    SourceSapActorPairDestination found,
    SourceSapActorPairDestination lost,
    bool abort_mode,
    SourceSapFetchHistoryBuffers histories,
    SourceSapMappedReportView* mapped_roots = nullptr) {
    return fetch_source_sap_results_native(context,addresses,
        descriptor_address,host_descriptor,found,lost,abort_mode,
        [histories]() noexcept {
            swap_sap_endpoint_histories_after_fetch_native(histories);
        },mapped_roots);
}

} // namespace fyzz::physx_gpu
