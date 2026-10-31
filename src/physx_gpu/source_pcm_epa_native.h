// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// A GuEPA continuation into existing Sc EdgeIndex and GuPCM candidate
// storage. The preceding GPU worklist supplies the indirect group count.
class SourcePcmEpaNativeStage final {
public:
    SourcePcmEpaNativeStage(compute::Context& context,
                            compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(!device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source GPU EPA dispatch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=7u;
        desc.storage_buffer_views.assign(
            7u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=44u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourcePcmEpaNativeStage(const SourcePcmEpaNativeStage&)=delete;
    SourcePcmEpaNativeStage& operator=(const SourcePcmEpaNativeStage&)=delete;
    ~SourcePcmEpaNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                std::uint32_t pair_count,std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity) const {
        if(!pair_count)return;
        if(!shape_capacity || !candidate_capacity || !edge_capacity ||
           pair_count>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Source GPU EPA exceeds source or native dispatch capacity");
        std::array<std::uint32_t,11> params{
            pair_count,0u,0u,0u,0u,0u,0u,0u,
            shape_capacity,candidate_capacity,edge_capacity};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,params[slot+1u],max_view_);
        };
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={1u,1u,1u};
        dispatch.buffers={
            view(transforms,source_allocation_bytes(shape_capacity,32u),
                 0u,compute::BufferAccess::Read),
            view(geometry,source_allocation_bytes(shape_capacity,16u),
                 1u,compute::BufferAccess::Read),
            view(pairs,source_allocation_bytes(pair_count,20u),
                 2u,compute::BufferAccess::Read),
            view(results,source_allocation_bytes(pair_count,32u),
                 3u,compute::BufferAccess::ReadWrite),
            view(candidates,source_allocation_bytes(candidate_capacity,40u),
                 4u,compute::BufferAccess::Write),
            view(edge_cache,source_allocation_bytes(edge_capacity,240u),
                 5u,compute::BufferAccess::ReadWrite),
            view(work,source_allocation_bytes(pair_count,4u),
                 6u,compute::BufferAccess::Read)};
        const auto arguments=addresses.resolve(dispatch_args,12u);
        if(arguments.offset%4u)
            throw std::invalid_argument("Source EPA indirect arguments misaligned");
        for(const auto& binding:dispatch.buffers)
            if(binding.buffer==arguments.buffer)
                throw std::invalid_argument(
                    "Source EPA indirect arguments alias Sc data");
        dispatch.indirect=compute::DispatchIndirectArgs{
            arguments.buffer,arguments.offset};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),
                    params.data(),sizeof(params));
        commands.emplace_back(std::move(dispatch));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address work,
                CudaDeviceAddressMap::Address dispatch_args,
                std::uint32_t pair_count,std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity,
                std::uint32_t edge_capacity) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,candidates,edge_cache,work,dispatch_args,
               pair_count,shape_capacity,candidate_capacity,edge_capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
