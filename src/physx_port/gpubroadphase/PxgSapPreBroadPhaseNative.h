// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_PRE_BROADPHASE_NATIVE_H
#define FYZZXYZZY_PXG_SAP_PRE_BROADPHASE_NATIVE_H

#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "foundation/PxBounds3.h"
#include <span>
#include <vector>

namespace physx { namespace Bp { class BroadPhaseUpdateData; } }

namespace fyzz::physx_gpu {
class SourceSapDmaNative;
struct SourceSapUpload;

// Original PxgCudaBroadPhaseSap member buffers used in preBroadPhase().
// The enclosing PhysX broadphase remains their owner. This borrowed view
// allows the original host method to use the native stream without adopting
// a second broadphase controller or independent frame state.
struct SourceSapPreBroadPhaseBuffers {
    physx::PxgTypedCudaBuffer<physx::PxBounds3>& bounds;
    physx::PxgTypedCudaBuffer<physx::PxReal>& contact_distances;
    physx::PxgTypedCudaBuffer<physx::PxU32>& groups;
    physx::PxgTypedCudaBuffer<physx::PxU32>& environment_ids;
    physx::PxU32& boxes_capacity;
};

// Preflight source bounds and host-array capacity before any preceding
// AABB-manager bitmap transfer can enter the same GPU frame.
void validate_pre_broadphase_native(
    const physx::Bp::BroadPhaseUpdateData& update);

// Append the source preBroadPhase allocations and HtoD records to the
// current AABB-manager transfer group. The owning caller commits all
// adjacent source uploads as one ordered native submission.
void append_pre_broadphase_native_uploads(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    std::vector<SourceSapUpload>& uploads,
    // The source AABB manager's original changed-handle bitmap. When
    // provided, unchanged resident bounds may be retained on the device.
    std::span<const physx::PxU32> changed_handles = {});

// Source: PxgCudaBroadPhaseSap::preBroadPhase, including its conditional
// bounds upload and unconditional contact-distance/group uploads.
// The original Bp::BroadPhaseUpdateData is consumed directly; GPU buffers
// remain owned by PhysX and transfers are ordered by its existing mStream.
void pre_broadphase_native(
    const physx::Bp::BroadPhaseUpdateData& update,
    SourceSapPreBroadPhaseBuffers buffers,
    SourceSapDmaNative& stream);

} // namespace fyzz::physx_gpu
#endif
