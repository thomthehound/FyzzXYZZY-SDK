// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "PxgBroadPhaseDesc.h"
#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Original mBPDescBuf address and original Bp::VolumeData layout. All other
// resources are resolved directly from the source PxgBroadPhaseDesc.
struct SourceSapReportNativeLayout {
    CudaDeviceAddressMap::Address descriptor = 0;
    std::uint32_t volume_capacity = 0;
    std::uint32_t volume_stride = 0;
    std::uint32_t aggregate_member_offset = 0;
};

// The original BP_ACCUMULATE_REPORT_STAGE_{1,2} and BP_COPY_REPORTS launches.
// Every input and output may belong to a different native PhysX heap root;
// shared-root slices work without a copied arena. The original PxgBroadPhaseDesc
// fields are the GPU counters and overflow flag, not shadow descriptors.
// Appends all three commands atomically to the caller's source stream.
class SourceSapReportNativeStage final {
public:
    SourceSapReportNativeStage(
        compute::Context& context,
        compute::ShaderBinary first_portable,
        compute::ShaderBinary second_portable,
        compute::ShaderBinary copy_shader,
        compute::ShaderBinary first_wave,
        compute::ShaderBinary second_wave,
        bool force_portable = false);
    ~SourceSapReportNativeStage();
    SourceSapReportNativeStage(const SourceSapReportNativeStage&) = delete;
    SourceSapReportNativeStage& operator=(const SourceSapReportNativeStage&) = delete;

    bool using_wave_reports() const noexcept { return using_wave_; }
    void append(std::vector<compute::BatchCommand>& commands,
                const physx::PxgBroadPhaseDesc& source,
                const CudaDeviceAddressMap& addresses,
                SourceSapReportNativeLayout layout) const;
private:
    compute::Context& context_;
    compute::PipelineId first_=0, second_=0, copy_=0;
    bool using_wave_=false;
};

} // namespace fyzz::physx_gpu
