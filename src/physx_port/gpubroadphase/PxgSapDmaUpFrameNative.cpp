// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <stdexcept>

namespace fyzz::physx_gpu {

SourceSapGpuDMAUpCounts gpu_dma_up_sap_frame_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapDmaUpFrameMembers members,
    SourceSapDmaNative& stream) {
    if (!members.radix_descriptors || !members.radix_wor_descriptors)
        throw std::invalid_argument("PhysX SAP gpuDMAUp requires source radix host records");
    // The source constructor allocates these GPU buffers. Reject missing
    // descriptor storage BEFORE source box counters or GPU allocations change.
    validate_sap_dma_up_descriptor_buffers_native(members.descriptor_gpu_buffers);

    // Exactly the original source sequence. Keep each prefix, inner j
    // remainder, then four per-axis scratch allocations adjacent before
    // proceeding to the next axis. Reordering these can change source
    // deferred frees, buddy heap demands and DtoD history dependencies.
    const auto counts = prepare_gpu_dma_up_counts_native(update, members.counts);
    grow_sap_old_bounds_native(counts, members.old_integer_bounds, stream);
    for (physx::PxU32 i = 0; i < 3u; ++i) {
        grow_sap_history_axis_prefix_native(counts, members.axes[i], stream);
        grow_sap_history_axis_remainder_native(
            counts, i, members.remainder, stream);
    }
    allocate_sap_post_axis_buffers_native(counts, members.post_axis);
    allocate_gpu_dma_up_handles_native(counts, members.handles);
    allocate_sap_new_integer_bounds_native(counts, members.new_integer_bounds);

    // Source updateDescriptor() reads current post-allocation pointers.
    // Scalar snapshots and mapped/manager aliases are supplied by the
    // owning broadphase and its AABB manager at this exact call boundary.
    auto inputs = members.descriptor_inputs;
    inputs.num_boxes = counts.current_boxes;
    inputs.created_count = counts.created;
    inputs.removed_count = counts.removed;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
    inputs.updated_count = counts.updated;
#else
    inputs.updated_count = 0;
#endif
    prepare_sap_broadphase_descriptor_native(
        members.broadphase_descriptor, members.broadphase_buffers, inputs);
    prepare_sap_radix_descriptors_native(
        members.radix_descriptors, members.radix_wor_descriptors,
        members.radix_buffers);

    // Original adjacent HtoD calls: created, removed, optional updated,
    // then one BP descriptor, six regular radix and six WOR radix records,
    // all enqueued on the original mStream with no extra host wait.
    // These six consecutive source HtoD calls have no intervening CUDA
    // copy, launch or wait. Resolve/validate both groups BEFORE one native
    // submission. The Context snapshots CPU inputs before return; source
    // owner allocation/retirement semantics outside this exact boundary
    // remain unchanged.
    std::vector<SourceSapUpload> uploads;
    uploads.reserve(6u);
    append_gpu_dma_up_handles_native(update, counts, members.handles, uploads);
    append_sap_dma_up_descriptors_native(
        members.broadphase_descriptor, members.radix_descriptors,
        members.radix_wor_descriptors, members.descriptor_gpu_buffers,
        uploads);
    stream.queue_uploads(uploads);
    return counts;
}

} // namespace fyzz::physx_gpu
