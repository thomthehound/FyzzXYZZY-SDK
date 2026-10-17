// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapDmaUpUploadsNative.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <stdexcept>

namespace fyzz::physx_gpu {
void validate_sap_dma_up_descriptor_buffers_native(
    SourceSapDmaUpDescriptorBuffers buffers) {
    constexpr std::size_t radix_bytes = sizeof(physx::PxgRadixSortDesc) * 6u;
    if (buffers.broadphase.getSize() < sizeof(physx::PxgBroadPhaseDesc) ||
        buffers.radix.getSize() < radix_bytes ||
        buffers.radix_without_ranks.getSize() < radix_bytes)
        throw std::logic_error("PhysX gpuDMAUp original GPU descriptor buffers not allocated");
}

void append_sap_dma_up_descriptors_native(
    const physx::PxgBroadPhaseDesc& bp_desc,
    const physx::PxgRadixSortDesc* radix_descs,
    const physx::PxgRadixSortDesc* radix_wor_descs,
    SourceSapDmaUpDescriptorBuffers buffers,
    std::vector<SourceSapUpload>& uploads) {
    // Validate the complete source upload group before submitting ANY
    // descriptor HtoD. Keep the six-entry source record ABI unchanged.
    constexpr std::size_t radix_bytes = sizeof(physx::PxgRadixSortDesc) * 6u;
    if (!radix_descs || !radix_wor_descs)
        throw std::invalid_argument("PhysX gpuDMAUp has null radix descriptor records");
    validate_sap_dma_up_descriptor_buffers_native(buffers);

    // The original three memcpyHtoDAsync calls are adjacent and ordered.
    // SourceSapDmaNative snapshots host data immediately and submits
    // them on the already-owning GPU queue, without an invented fence.
    uploads.insert(uploads.end(), {
        {buffers.broadphase.getDevicePtr(), &bp_desc, sizeof(bp_desc)},
        {buffers.radix.getDevicePtr(), radix_descs, radix_bytes},
        {buffers.radix_without_ranks.getDevicePtr(), radix_wor_descs, radix_bytes}
    });
}
void queue_sap_dma_up_descriptors_native(
    const physx::PxgBroadPhaseDesc& bp_desc,
    const physx::PxgRadixSortDesc* radix_descs,
    const physx::PxgRadixSortDesc* radix_wor_descs,
    SourceSapDmaUpDescriptorBuffers buffers,
    SourceSapDmaNative& stream) {
    std::vector<SourceSapUpload> uploads;
    uploads.reserve(3u);
    append_sap_dma_up_descriptors_native(
        bp_desc, radix_descs, radix_wor_descs, buffers, uploads);
    stream.queue_uploads(uploads);
}

} // namespace fyzz::physx_gpu
