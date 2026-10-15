// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Lower a source PxCudaContext::memsetD32Async(device_address, value, words,
// mStream) into an existing ordered native compute command list. Does not
// allocate, submit, synchronize, or take ownership of the source allocation.
// May be appended immediately before the original histogram kernels.
class SourceFillD32NativeStage final {
public:
    SourceFillD32NativeStage(compute::Context& context,
                             compute::ShaderBinary shader);
    ~SourceFillD32NativeStage();
    SourceFillD32NativeStage(const SourceFillD32NativeStage&) = delete;
    SourceFillD32NativeStage& operator=(const SourceFillD32NativeStage&) = delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address device_address,
                std::uint32_t value, std::uint32_t words) const;

private:
    compute::Context& context_;
    compute::PipelineId pipeline_ = 0;
};
} // namespace fyzz::physx_gpu
