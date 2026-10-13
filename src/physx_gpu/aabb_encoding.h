// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Port of the input/output contract of PhysX's translateAABBsLaunch.
// Upstream: physx/source/gpubroadphase/src/CUDA/broadphase.cu
// Host dispatch: physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp
//
// Bounds and encoded bounds each contain six tightly packed 32-bit elements:
// min.xyz followed by max.xyz, matching PxBounds3 and PxgIntegerAABB.
// All buffers belong to the calling scene; the stage owns no simulation state.
struct AabbEncodingParameters {
    std::uint32_t count = 0;
    std::uint32_t shift_x = 4;
    std::uint32_t shift_y = 4;
    std::uint32_t shift_z = 4;
    std::uint32_t env_bits_x = 0;
    std::uint32_t env_bits_y = 0;
    std::uint32_t env_bits_z = 0;
};

class AabbEncodingStage final {
public:
    AabbEncodingStage(compute::Context& context, compute::ShaderBinary shader);
    ~AabbEncodingStage();

    AabbEncodingStage(const AabbEncodingStage&) = delete;
    AabbEncodingStage& operator=(const AabbEncodingStage&) = delete;

    // Appends the PhysX AABB encoding stage to the caller's ordered GPU work.
    // Pass zero for environment_ids when the source array is absent.
    // Buffer sizes and identifier validity are checked upon submission.
    void append(std::vector<compute::BatchCommand>& commands,
                compute::BufferId bounds, compute::BufferId contact_distances,
                compute::BufferId environment_ids, compute::BufferId encoded_bounds,
                const AabbEncodingParameters& parameters) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};

} // namespace fyzz::physx_gpu
