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

// Source Sc contact-manager identity and GuPCM candidate destination.
struct SourcePcmBoxPairNative {
    std::uint32_t first_element;
    std::uint32_t second_element;
    std::uint32_t edge_handle;
    std::uint32_t candidate_offset;
    std::uint32_t candidate_capacity;
};
static_assert(sizeof(SourcePcmBoxPairNative)==20u);

// GuPCMContactBoxBox::pcmContactBoxBox requires local A/B witnesses and a
// B-local normal before the source persistent-manifold reduction.
struct SourcePcmBoxCandidateNative {
    float local_a[3];
    float local_b[3];
    float normal_b[3];
    float separation;
};
static_assert(sizeof(SourcePcmBoxCandidateNative)==40u);

struct SourcePcmBoxPairResultNative {
    std::uint32_t candidate_count;
    std::uint32_t overlapping;
    // 0 valid, 1 invalid manager/geometry/window, 2 contact-window overflow.
    std::uint32_t status;
    std::uint32_t edge_handle;
    std::uint32_t candidate_offset;
};
static_assert(sizeof(SourcePcmBoxPairResultNative)==20u);

class SourcePcmBoxCandidatesNativeStage final {
public:
    SourcePcmBoxCandidatesNativeStage(compute::Context& context,
                                      compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source PCM box GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(std::max<std::uint64_t>(
            4u,device.storage_buffer_offset_alignment));
        max_view_=device.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=5u;
        desc.storage_buffer_views.assign(
            5u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=32u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourcePcmBoxCandidatesNativeStage(const SourcePcmBoxCandidatesNativeStage&)=delete;
    SourcePcmBoxCandidatesNativeStage& operator=(
        const SourcePcmBoxCandidatesNativeStage&)=delete;
    ~SourcePcmBoxCandidatesNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address candidates,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity) const {
        if(!pair_count)return;
        if(!shape_capacity || !candidate_capacity)
            throw std::invalid_argument("Source PCM GPU requires shapes and candidate capacity");
        const auto groups=(std::uint64_t(pair_count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source PCM GPU dispatch exceeds device limit");
        std::array<std::uint32_t,8> params{
            pair_count,0u,0u,0u,0u,0u,shape_capacity,candidate_capacity};
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
            view(results,source_allocation_bytes(pair_count,20u),
                 3u,compute::BufferAccess::Write),
            view(candidates,source_allocation_bytes(candidate_capacity,40u),
                 4u,compute::BufferAccess::Write)};
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
                CudaDeviceAddressMap::Address candidates,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t candidate_capacity) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,candidates,pair_count,shape_capacity,candidate_capacity);
    }

private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
