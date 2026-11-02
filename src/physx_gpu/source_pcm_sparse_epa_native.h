// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// The source Sc edge/result arrays remain the sole authority. This is a
// transient, device-owned EPA worklist; it never owns contact managers.
class SourcePcmSparseEpaNativeStage final {
public:
    SourcePcmSparseEpaNativeStage(
            compute::Context& context,compute::ShaderBinary compact_shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source PCM EPA workgroup unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.storage_buffer_binding_count=3u;
        desc.storage_buffer_views={
            compute::StorageBufferViewKind::RawRead,
            compute::StorageBufferViewKind::RawReadWrite,
            compute::StorageBufferViewKind::RawReadWrite};
        desc.push_constant_bytes=16u;
        if(compact_shader.format==compute::ShaderFormat::Spirv)
            compact_shader.entry_point="compact";
        desc.shader=std::move(compact_shader);
        compact_=context_.create_compute_pipeline(desc);
    }
    SourcePcmSparseEpaNativeStage(
        const SourcePcmSparseEpaNativeStage&)=delete;
    SourcePcmSparseEpaNativeStage& operator=(
        const SourcePcmSparseEpaNativeStage&)=delete;
    ~SourcePcmSparseEpaNativeStage() {
        if(compact_)context_.destroy_compute_pipeline(compact_);
    }

    // dispatch_args is the native 12-byte {x,y,z} indirect launch record.
    // A following one-thread-per-EPA-pair kernel consumes x GPU-side.
    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                std::uint32_t pair_count) const {
        const auto& limits=context_.device_info();
        if(pair_count>limits.max_workgroup_count[0])
            throw std::out_of_range(
                "Source EPA work count exceeds indirect dispatch limit");
        // Empty manager streams have no EPA consumer or indirect dispatch.
        if(!pair_count)return;
        const auto groups=(std::uint64_t(pair_count)+127u)/128u;
        std::array<std::uint32_t,4> params{pair_count,0u,0u,0u};
        const auto view=[&](CudaDeviceAddressMap::Address addr,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access) {
            return source_native_view(addresses,addr,bytes,
                alignment_,slot,access,params[slot+1u],max_view_);
        };
        const auto inputs=std::array<compute::BufferBinding,3>{
            view(results,source_allocation_bytes(pair_count,32u),
                 0u,compute::BufferAccess::Read),
            view(work,source_allocation_bytes(pair_count,4u),
                 1u,compute::BufferAccess::Write),
            view(dispatch_args,12u,2u,compute::BufferAccess::ReadWrite)};
        if(inputs[2].buffer==inputs[0].buffer ||
           inputs[2].buffer==inputs[1].buffer)
            throw std::invalid_argument(
                "Source EPA indirect arguments must have a separate GPU buffer");
        compute::DispatchCommand compact{};
        compact.pipeline=compact_;
        compact.groups={static_cast<std::uint32_t>(groups),1u,1u};
        compact.buffers.assign(inputs.begin(),inputs.end());
        compact.push_constants.resize(sizeof(params));
        std::memcpy(compact.push_constants.data(),
                    params.data(),sizeof(params));
        commands.emplace_back(std::move(compact));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                std::uint32_t pair_count) const {
        append(stream.commands(),addresses,results,work,dispatch_args,pair_count);
    }
private:
    compute::Context& context_;
    compute::PipelineId compact_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
