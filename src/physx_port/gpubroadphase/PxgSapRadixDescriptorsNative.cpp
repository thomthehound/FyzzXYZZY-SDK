// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapRadixDescriptorsNative.h"

#include <stdexcept>

namespace fyzz::physx_gpu {

void prepare_sap_radix_descriptors_native(
    physx::PxgRadixSortDesc* rsDescs,
    physx::PxgRadixSortDesc* rsDescsWOR,
    SourceSapRadixDescriptorBuffers buffers) {
    if (!rsDescs || !rsDescsWOR)
        throw std::invalid_argument("PhysX SAP radix sort descriptor storage is null");

    // Source PxgCudaBroadPhaseSap::updateRadixSortDesc: i and i+3
    // are opposite directions of the same axis; the WOR descriptors
    // deliberately bind original box handles instead of ranks.
    // This function only writes the five original pointer fields,
    // retaining the source descriptors' existing count values.
    for (physx::PxU32 i = 0; i < 3u; ++i) {
        const physx::PxU32 offIndex = i + 3u;
        const CUdeviceptr inputKeyd = buffers.projections[i].getDevicePtr();
        const CUdeviceptr inputRankd = buffers.projection_ranks[i].getDevicePtr();
        const CUdeviceptr outputKeyd = buffers.temporary_projections[i].getDevicePtr();
        const CUdeviceptr outputRankd = buffers.temporary_handles[i].getDevicePtr();
        const CUdeviceptr radixCountd = buffers.radix_counts[i].getDevicePtr();

        rsDescs[i].inputKeys = reinterpret_cast<physx::PxU32*>(inputKeyd);
        rsDescs[i].inputRanks = reinterpret_cast<physx::PxU32*>(inputRankd);
        rsDescs[i].outputKeys = reinterpret_cast<physx::PxU32*>(outputKeyd);
        rsDescs[i].outputRanks = reinterpret_cast<physx::PxU32*>(outputRankd);
        rsDescs[i].radixBlockCounts = reinterpret_cast<physx::PxU32*>(radixCountd);

        rsDescs[offIndex].outputKeys = reinterpret_cast<physx::PxU32*>(inputKeyd);
        rsDescs[offIndex].outputRanks = reinterpret_cast<physx::PxU32*>(inputRankd);
        rsDescs[offIndex].inputKeys = reinterpret_cast<physx::PxU32*>(outputKeyd);
        rsDescs[offIndex].inputRanks = reinterpret_cast<physx::PxU32*>(outputRankd);
        rsDescs[offIndex].radixBlockCounts = reinterpret_cast<physx::PxU32*>(radixCountd);

        const CUdeviceptr inputVald = buffers.projection_handles[i].getDevicePtr();
        rsDescsWOR[i].inputKeys = reinterpret_cast<physx::PxU32*>(inputKeyd);
        rsDescsWOR[i].inputRanks = reinterpret_cast<physx::PxU32*>(inputVald);
        rsDescsWOR[i].outputKeys = reinterpret_cast<physx::PxU32*>(outputKeyd);
        rsDescsWOR[i].outputRanks = reinterpret_cast<physx::PxU32*>(outputRankd);
        rsDescsWOR[i].radixBlockCounts = reinterpret_cast<physx::PxU32*>(radixCountd);

        rsDescsWOR[offIndex].outputKeys = reinterpret_cast<physx::PxU32*>(inputKeyd);
        rsDescsWOR[offIndex].outputRanks = reinterpret_cast<physx::PxU32*>(inputVald);
        rsDescsWOR[offIndex].inputKeys = reinterpret_cast<physx::PxU32*>(outputKeyd);
        rsDescsWOR[offIndex].inputRanks = reinterpret_cast<physx::PxU32*>(outputRankd);
        rsDescsWOR[offIndex].radixBlockCounts = reinterpret_cast<physx::PxU32*>(radixCountd);
    }
}

} // namespace fyzz::physx_gpu
