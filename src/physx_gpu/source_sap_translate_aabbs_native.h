// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_aabb_launch_lowering.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/gpucommon/include/PxgKernelIndices.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// Original PxgCudaBroadPhaseSap::translateAABBsKernel arguments borrowed
// from its four existing source buffers and PxGpuBroadPhaseDesc bit fields.
// In particular, the optional environment pointer is the original buffer
// address, not an independent scene/environment allocation.
struct SourceSapTranslateAabbsNativeInput {
    CudaDeviceAddressMap::Address fp_bounds=0;
    CudaDeviceAddressMap::Address new_integer_bounds=0;
    CudaDeviceAddressMap::Address contact_distances=0;
    CudaDeviceAddressMap::Address environment_ids=0;
    AabbEncodingParameters encoding{};
};

// Called after begin_sap_update_native() on the original source mStream,
// before markRemovedPairsKernel(). The source update gate must already
// have completed gpuDMAUp; no separate submission or wait occurs here.
// This uses the existing source-CUDA ABI translator and AABB shader.
inline void append_source_sap_translate_aabbs_native(
    std::vector<compute::BatchCommand>& commands,
    const CudaAabbLaunchLowering& translate,
    const SourceSapTranslateAabbsNativeInput& source,
    bool execute_gpu_stages) {
    if(!execute_gpu_stages || !source.encoding.count)return;
    const auto& p=source.encoding;
    // PxGpuBroadPhaseDesc declares six PxU8 shifts. Never narrow silently
    // when constructing the original eleven-argument CUDA launch record.
    if(p.shift_x>std::numeric_limits<std::uint8_t>::max() ||
       p.shift_y>std::numeric_limits<std::uint8_t>::max() ||
       p.shift_z>std::numeric_limits<std::uint8_t>::max() ||
       p.env_bits_x>std::numeric_limits<std::uint8_t>::max() ||
       p.env_bits_y>std::numeric_limits<std::uint8_t>::max() ||
       p.env_bits_z>std::numeric_limits<std::uint8_t>::max())
        throw std::invalid_argument("Source GPU SAP AABB bit shift exceeds PxU8");
    const std::array<std::uint8_t,6> bits{
        static_cast<std::uint8_t>(p.shift_x),
        static_cast<std::uint8_t>(p.shift_y),
        static_cast<std::uint8_t>(p.shift_z),
        static_cast<std::uint8_t>(p.env_bits_x),
        static_cast<std::uint8_t>(p.env_bits_y),
        static_cast<std::uint8_t>(p.env_bits_z)};
    // CUDA kernel arguments are host addresses to the actual parameter
    // values, copied synchronously by CudaAabbLaunchLowering::append().
    const auto box=source.fp_bounds;
    const auto encoded=source.new_integer_bounds;
    const auto distance=source.contact_distances;
    const auto env=source.environment_ids;
    const std::array<const void*,11> args{
        &box,&encoded,&distance,&env,&p.count,
        &bits[0],&bits[1],&bits[2],&bits[3],&bits[4],&bits[5]};
    const auto grid=(std::uint64_t(p.count)+31u)/32u;
    if(grid>std::numeric_limits<std::uint32_t>::max())
        throw std::out_of_range("Source SAP translateAABBs grid exceeds PxU32");
    translate.append(commands,physx::PxgKernelIds::BP_TRANSLATE_AABBS,
                     static_cast<std::uint32_t>(grid),
                     256u, // original PxgBPKernelBlockDim::BP_TRANSLATE_AABBS
                     args.data(),args.size());
}


// Retain the public per-stage stream form; the full source head can
// instead stage every kernel transactionally before appending them.
inline void append_source_sap_translate_aabbs_native(
    SourceSapDmaNative& stream,
    const CudaAabbLaunchLowering& translate,
    const SourceSapTranslateAabbsNativeInput& source,
    bool execute_gpu_stages) {
    append_source_sap_translate_aabbs_native(
        stream.commands(),translate,source,execute_gpu_stages);
}
} // namespace fyzz::physx_gpu
