// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>
namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap::performIncrementalSapKernel's two
// comparison-prefix launches. All 24 input/output allocations belong to
// PxgCudaBroadPhaseSap; the 25th view is the ORIGINAL DEVICE descriptor
// holding totalIncrementalComparisons[3], not a host-side shadow counter.
struct SourceSapIncrementalCountsLayout {
    std::uint32_t endpoint_slots=0;
};

class SourceSapIncrementalCountsNativeStage final {
public:
    SourceSapIncrementalCountsNativeStage(
        compute::Context& context, compute::ShaderBinary first,
        compute::ShaderBinary second);
    ~SourceSapIncrementalCountsNativeStage();
    SourceSapIncrementalCountsNativeStage(const SourceSapIncrementalCountsNativeStage&)=delete;
    SourceSapIncrementalCountsNativeStage& operator=(const SourceSapIncrementalCountsNativeStage&)=delete;

    // Original source 32x256 + 32x256 sequence. Both append to the
    // caller's ordered GPU stream, without submission or readback.
    // source_device_descriptor is mBPDescBuf.getDevicePtr() (CUDA VA).
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                CudaDeviceAddressMap::Address source_device_descriptor,
                const CudaDeviceAddressMap& addresses,
                SourceSapIncrementalCountsLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId first_=0,second_=0;
    std::size_t alignment_=4u;
    std::uint64_t max_view_=0u;
};
} // namespace fyzz::physx_gpu
