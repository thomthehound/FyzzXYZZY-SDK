// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "physx_gpu/cuda_device_address_map.h"

#include <cstdint>
#include <vector>

namespace fyzz::physx_gpu {

// Pure per-view D32 capacity calculation. Exposed only inside the PhysX
// host-lowering implementation so tests can force a small descriptor-view
// limit without allocating gigabytes or adding a test-only GPU runtime path.
// Returns a nonzero word count that fits both the resolved CUDA allocation
// and the physically aligned native RAW buffer view.
std::uint32_t source_d32_fill_chunk_words(
    const CudaDeviceAddressMap& addresses,
    CudaDeviceAddressMap::Address address,
    std::uint32_t remaining_words,
    std::size_t alignment, std::uint64_t max_view_bytes);

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
