// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Source PxgCudaBroadPhaseSap allocation extents; volumes and endpoints are
// borrowed from the original AABB/SAP owner, not a second broadphase arena.
struct SourceSapRemovedNativeLayout {
    std::uint32_t box_capacity=0;
    std::uint32_t endpoint_slots=0;
};

// The original two *distinct* 32x256 kernels:
// - BP_MARK_DELETEDPAIRS: after translateAABBs, before ranked WR sort.
// - BP_UPDATE_DELETEDPAIRS: after incremental SAP, before created-pair sort.
// Source descriptor pointers remain authoritative across both calls.
class SourceSapRemovedNativeStage final {
public:
    SourceSapRemovedNativeStage(compute::Context& context,
                                compute::ShaderBinary mark_shader,
                                compute::ShaderBinary projection_shader);
    ~SourceSapRemovedNativeStage();
    SourceSapRemovedNativeStage(const SourceSapRemovedNativeStage&)=delete;
    SourceSapRemovedNativeStage& operator=(const SourceSapRemovedNativeStage&)=delete;

    void append_mark_removed(std::vector<compute::BatchCommand>& commands,
                             const physx::PxgBroadPhaseDesc& source,
                             const CudaDeviceAddressMap& addresses,
                             SourceSapRemovedNativeLayout layout) const;
    void append_remove_projections(std::vector<compute::BatchCommand>& commands,
                                   const physx::PxgBroadPhaseDesc& source,
                                   const CudaDeviceAddressMap& addresses,
                                   SourceSapRemovedNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId mark_pipeline_=0;
    compute::PipelineId projection_pipeline_=0;
};
} // namespace fyzz::physx_gpu
