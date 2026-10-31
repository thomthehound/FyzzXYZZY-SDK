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

// Source Gu contact-manager pair work, keyed by original Sc element handles.
struct SourceGuBoxPairNative {
    std::uint32_t first_element;
    std::uint32_t second_element;
    std::uint32_t previous_axis;
    std::uint32_t contact_offset=0;
    std::uint32_t contact_capacity=0;
    std::uint32_t reserved=0;
};
static_assert(sizeof(SourceGuBoxPairNative)==24u);

struct SourceGuBoxContactNative {
    float position[3];
    float normal[3];
    float separation;
    std::uint32_t reserved=0;
};
static_assert(sizeof(SourceGuBoxContactNative)==32u);

struct SourceGuBoxPairResultNative {
    std::uint32_t count;
    std::uint32_t collision_axis;
    // 0 valid, 1 invalid source handle/window, 2 insufficient contact storage.
    std::uint32_t status;
    std::uint32_t reserved;
};
static_assert(sizeof(SourceGuBoxPairResultNative)==16u);

class SourceGuBoxBoxContactsNativeStage final {
public:
    SourceGuBoxBoxContactsNativeStage(compute::Context& context,
                                     compute::ShaderBinary shader)
        : context_(context) {
        const auto& device=context.device_info();
        if(device.max_workgroup_invocations<128u ||
           device.max_workgroup_size[0]<128u ||
           !device.max_workgroup_count[0] ||
           !device.max_storage_buffer_range)
            throw std::runtime_error("Source Gu contact GPU workgroup unavailable");
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
    SourceGuBoxBoxContactsNativeStage(const SourceGuBoxBoxContactsNativeStage&)=delete;
    SourceGuBoxBoxContactsNativeStage& operator=(
        const SourceGuBoxBoxContactsNativeStage&)=delete;
    ~SourceGuBoxBoxContactsNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address transforms,
                CudaDeviceAddressMap::Address geometry,
                CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address results,
                CudaDeviceAddressMap::Address contacts,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t contact_capacity) const {
        if(!pair_count)return;
        if(!shape_capacity || !contact_capacity)
            throw std::invalid_argument("Gu GPU contacts require shape and contact capacity");
        const auto groups=(std::uint64_t(pair_count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Gu GPU contact dispatch exceeds device limit");
        std::array<std::uint32_t,8> params{
            pair_count,0u,0u,0u,0u,0u,shape_capacity,contact_capacity};
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
            view(pairs,source_allocation_bytes(pair_count,24u),
                 2u,compute::BufferAccess::Read),
            view(results,source_allocation_bytes(pair_count,16u),
                 3u,compute::BufferAccess::Write),
            view(contacts,source_allocation_bytes(contact_capacity,32u),
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
                CudaDeviceAddressMap::Address contacts,
                std::uint32_t pair_count,
                std::uint32_t shape_capacity,
                std::uint32_t contact_capacity) const {
        append(stream.commands(),addresses,transforms,geometry,pairs,
               results,contacts,pair_count,shape_capacity,contact_capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
