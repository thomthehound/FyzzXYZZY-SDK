// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/source_fill_d32_native.h"
#include "PxgBroadPhaseDesc.h"

namespace fyzz::physx_gpu {

// Sizes of original PxgCudaBroadPhaseSap allocations, not a substitute
// descriptor or an independently owned frame. Projection slots include
// temporarily retained removed handles.
struct SourceRegionCreateNativeLayout {
    std::uint32_t box_capacity = 0;
    std::uint32_t projection_slots = 0;
};

// The first THREE original operations of
// PxgCudaBroadPhaseSap::computeRegionHistogramKernel:
// memsetD32Async(activeRegionsHistogram), memsetD32Async(startRegionsHistogram),
// and BP_CREATE_REGIONS (32x256). All use original PxgBroadPhaseDesc CUDA
// addresses, borrowed GPU resources, and the CALLER'S existing command list.
// The following source region/start histogram scans are intentionally separate.
class SourceRegionCreateNativeStage final {
public:
    SourceRegionCreateNativeStage(compute::Context& context,
                                  compute::ShaderBinary shader,
                                  const SourceFillD32NativeStage& fill);
    ~SourceRegionCreateNativeStage();
    SourceRegionCreateNativeStage(const SourceRegionCreateNativeStage&) = delete;
    SourceRegionCreateNativeStage& operator=(const SourceRegionCreateNativeStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceRegionCreateNativeLayout layout) const;
private:
    compute::Context& context_;
    const SourceFillD32NativeStage& fill_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
