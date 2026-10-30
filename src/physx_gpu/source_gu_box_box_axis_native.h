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

// Gu box-box mPairData axis is retained by its existing contact manager.
// These are transient pair-work records keyed by original Sc element handles.
struct SourceGuBoxPairAxisNative {
    std::uint32_t first_element;
    std::uint32_t second_element;
    std::uint32_t previous_axis;
    std::uint32_t reserved=0;
};
static_assert(sizeof(SourceGuBoxPairAxisNative)==16u);

class SourceGuBoxBoxAxisNativeStage final {
public:
    SourceGuBoxBoxAxisNativeStage(compute::Context& context,
                                  compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source Gu box pair GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(std::max<std::uint64_t>(
            4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=4u;
        desc.storage_buffer_views.assign(
            4u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=24u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourceGuBoxBoxAxisNativeStage(const SourceGuBoxBoxAxisNativeStage&)=delete;
    SourceGuBoxBoxAxisNativeStage& operator=(const SourceGuBoxBoxAxisNativeStage&)=delete;
    ~SourceGuBoxBoxAxisNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address axes,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity) const {
        if(!pair_count)return;
        if(!shape_capacity)
            throw std::invalid_argument("Source Gu box pairs require shape capacity");
        const std::uint64_t groups=(std::uint64_t(pair_count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source Gu box axis dispatch exceeds device limit");
        std::array<std::uint32_t,6> params{pair_count,0u,0u,0u,0u,shape_capacity};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,params[slot+1u],max_view_);
        };
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
        dispatch.buffers={
            view(transforms,source_allocation_bytes(shape_capacity,32u),
                 0u,compute::BufferAccess::Read),
            view(geometry,source_allocation_bytes(shape_capacity,16u),
                 1u,compute::BufferAccess::Read),
            view(pairs,source_allocation_bytes(pair_count,16u),
                 2u,compute::BufferAccess::Read),
            view(axes,source_allocation_bytes(pair_count,4u),
                 3u,compute::BufferAccess::Write)};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(dispatch));
    }

    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address axes,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity) const {
        append(stream.commands(),addresses,transforms,geometry,
               pairs,axes,pair_count,shape_capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
