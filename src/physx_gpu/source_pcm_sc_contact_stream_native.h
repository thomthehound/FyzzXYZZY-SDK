// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_pcm_box_candidates_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {

// Source-oriented Gu world contacts are the input to DyContactPrep.
// The original Sc EdgeIndex retains the persistent B-local witnesses.
struct SourcePcmScWorldContactNative {
    float position[3];
    float separation;
};
static_assert(sizeof(SourcePcmScWorldContactNative)==16u);
struct SourcePcmScContactStreamNative {
    std::uint32_t edge_handle;
    std::uint32_t cached_count;
    std::uint32_t status;
    // Lower 31 bits: selected Gu count; high bit: regenerated flag.
    std::uint32_t selected_and_regenerated;
    float normal[3];
    std::uint32_t emitted_count;
    SourcePcmScWorldContactNative contacts[4];
};
static_assert(sizeof(SourcePcmScContactStreamNative)==96u);

class SourcePcmScContactStreamNativeStage final {
public:
    SourcePcmScContactStreamNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source PCM contact stream GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(std::max<std::uint64_t>(
            4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=6u;
        desc.storage_buffer_views.assign(
            6u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=36u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourcePcmScContactStreamNativeStage(const SourcePcmScContactStreamNativeStage&)=delete;
    SourcePcmScContactStreamNativeStage& operator=(
        const SourcePcmScContactStreamNativeStage&)=delete;
    ~SourcePcmScContactStreamNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address output,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t edge_capacity) const {
        if(!pair_count)return;
        if(!shape_capacity || !edge_capacity)
            throw std::invalid_argument(
                "Source PCM contact stream requires live Sc shapes and edges");
        const auto groups=(std::uint64_t(pair_count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source PCM contact stream dispatch exceeds device limit");
        std::array<std::uint32_t,9> params{
            pair_count,0u,0u,0u,0u,0u,0u,shape_capacity,edge_capacity};
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
            view(pairs,source_allocation_bytes(pair_count,20u),
                 2u,compute::BufferAccess::Read),
            view(results,source_allocation_bytes(pair_count,32u),
                 3u,compute::BufferAccess::Read),
            view(edge_cache,source_allocation_bytes(edge_capacity,240u),
                 4u,compute::BufferAccess::Read),
            view(output,source_allocation_bytes(pair_count,96u),
                 5u,compute::BufferAccess::Write)};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(dispatch));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address edge_cache,
                CudaDeviceAddressMap::Address output,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t edge_capacity) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,edge_cache,output,pair_count,shape_capacity,
               edge_capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
};

} // namespace fyzz::physx_gpu
