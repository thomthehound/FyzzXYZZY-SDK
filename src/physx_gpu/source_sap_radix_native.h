// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include "PxgRadixSortDesc.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace fyzz::physx_gpu {

// Borrowed original mRSDescs/mRSDescsWOR[0..5], prepared by
// PxgCudaBroadPhaseSap::updateRadixSortDesc. Each parity owns three
// source radix descriptors. Temporary projections, ranks and histogram
// allocations remain with the source SAP instance.
class SourceSapRadixNativeStage final {
public:
    SourceSapRadixNativeStage(compute::Context& context,
                              compute::ShaderBinary initialize_ranks,
                              compute::ShaderBinary count,
                              compute::ShaderBinary scatter,
                              compute::ShaderBinary gather_handles);
    ~SourceSapRadixNativeStage();
    SourceSapRadixNativeStage(const SourceSapRadixNativeStage&)=delete;
    SourceSapRadixNativeStage& operator=(const SourceSapRadixNativeStage&)=delete;

    // Original source sortProjectionAndHandlesWRKernel. Initializes
    // current projection ranks, runs eight stable 4-bit radix passes,
    // then gathers previous-frame handles through the final rank array.
    void append_wr(std::vector<compute::BatchCommand>& commands,
                   const physx::PxgBroadPhaseDesc& source,
                   std::span<const physx::PxgRadixSortDesc,6> radix,
                   const CudaDeviceAddressMap& addresses,
                   std::uint32_t previous_boxes) const;

    // Original sortProjectionAndHandlesWORKernel: sort current
    // projections and CURRENT handle payloads (not previous-frame ranks).
    void append_wor(std::vector<compute::BatchCommand>& commands,
                    std::span<const physx::PxgRadixSortDesc,6> radix,
                    const CudaDeviceAddressMap& addresses,
                    std::uint32_t previous_boxes,
                    std::uint32_t created_boxes) const;
private:
    void append_sort(std::vector<compute::BatchCommand>& staged,
                     std::span<const physx::PxgRadixSortDesc,6> radix,
                     const CudaDeviceAddressMap& addresses,
                     std::uint32_t padded_keys) const;
    compute::Context& context_;
    compute::PipelineId init_pipeline_=0;
    compute::PipelineId count_pipeline_=0;
    compute::PipelineId scatter_pipeline_=0;
    compute::PipelineId gather_pipeline_=0;
    std::size_t view_alignment_=4u;
    std::uint64_t max_view_bytes_=0u;
};
} // namespace fyzz::physx_gpu
