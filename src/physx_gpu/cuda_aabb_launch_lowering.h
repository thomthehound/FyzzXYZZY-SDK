// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/aabb_encoding.h"
#include "physx_gpu/cuda_device_address_map.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// One *real upstream* PxgCudaBroadPhaseSap::translateAABBsKernel launch
// lowered into its existing translated AABB shader. A temporary bridge,
// internal to the source-derived broadphase host. Not a new public CUDA
// runtime or an arbitrary kernel interpreter.
class CudaAabbLaunchLowering final {
public:
    CudaAabbLaunchLowering(
        const CudaDeviceAddressMap& addresses,
        const AabbEncodingStage& native_shader) noexcept;

    // Matches _launch(PxgKernelIds::BP_TRANSLATE_AABBS, gridX,1,1,
    // PxgBPKernelBlockDim::BP_TRANSLATE_AABBS,1,1,0,stream,kernelParams).
    // kernel_params points to the eleven CUDA argument value addresses
    // from PX_CUDA_KERNEL_PARAM2. The containing PhysX host owns task and
    // stream order and appends to its existing native command batch.
    void append(std::vector<compute::BatchCommand>& commands,
                std::uint32_t kernel_id,
                std::uint32_t cuda_grid_x,
                std::uint32_t cuda_block_x,
                const void* const* kernel_params,
                std::size_t kernel_param_count) const;
private:
    const CudaDeviceAddressMap& addresses_;
    const AabbEncodingStage& native_shader_;
};

} // namespace fyzz::physx_gpu
