// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#ifndef FYZZXYZZY_PXG_SAP_RADIX_DESCRIPTORS_NATIVE_H
#define FYZZXYZZY_PXG_SAP_RADIX_DESCRIPTORS_NATIVE_H

#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "PxgRadixSortDesc.h"

namespace fyzz::physx_gpu {

// Borrowed references to the EXACT source members consumed by
// PxgCudaBroadPhaseSap::updateRadixSortDesc, including mRadixCountBuf
// initialized in the original broadphase constructor. The owning
// PhysX SAP object remains responsible for all buffer lifetimes.
struct SourceSapRadixDescriptorBuffers {
    const physx::PxgCudaBufferN<3>& projections;
    const physx::PxgCudaBufferN<3>& projection_ranks;
    const physx::PxgCudaBufferN<3>& temporary_projections;
    const physx::PxgCudaBufferN<3>& temporary_handles;
    const physx::PxgCudaBufferN<3>& radix_counts;
    const physx::PxgCudaBufferN<6>& projection_handles;
};

// Faithfully lower the original six forward/reverse descriptors and six
// WOR (without-ranks) descriptors, using PhysX's actual PxgRadixSortDesc
// ABI. The caller provides the source mRSDescs and mRSDescsWOR storage.
// The original method does NOT assign the descriptor 'count' member;
// preserve it rather than resetting or inventing a value.
void prepare_sap_radix_descriptors_native(
    physx::PxgRadixSortDesc* descriptors,
    physx::PxgRadixSortDesc* descriptors_without_ranks,
    SourceSapRadixDescriptorBuffers buffers);

} // namespace fyzz::physx_gpu
#endif
