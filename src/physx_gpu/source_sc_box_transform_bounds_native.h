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

// Source Sc's absolute box pose and local extents, keyed by the original
// ElementSim handle. Its GPU form is a transfer record, not a new shape ID.
struct SourceScBoxTransformInputNative {
    std::uint32_t element_index;
    float position[3];
    float orientation[4];
    float half_extents[3];
    float contact_distance=0.0f;
};
static_assert(sizeof(SourceScBoxTransformInputNative)==48u);

struct SourceScBoxGeometryNative {
    float half_extents[3];
    float contact_distance;
};
static_assert(sizeof(SourceScBoxGeometryNative)==16u);

// GPU outputs use the original PxBounds3 (24 bytes) and PxsCachedTransform
// (32 bytes) element strides. The cache is shared by BP and future NP.
class SourceScBoxTransformBoundsNativeStage final {
public:
    SourceScBoxTransformBoundsNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        : context_(context) {
        const auto& info=context.device_info();
        if(info.max_workgroup_invocations<128u ||
           info.max_workgroup_size[0]<128u ||
           !info.max_workgroup_count[0] ||
           !info.max_storage_buffer_range)
            throw std::runtime_error(
                "Source Sc box transform GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(std::max<std::uint64_t>(
            4u,info.storage_buffer_offset_alignment));
        max_view_=info.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=4u;
        desc.storage_buffer_views.assign(
            4u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=20u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourceScBoxTransformBoundsNativeStage(
        const SourceScBoxTransformBoundsNativeStage&)=delete;
    SourceScBoxTransformBoundsNativeStage& operator=(
        const SourceScBoxTransformBoundsNativeStage&)=delete;
    ~SourceScBoxTransformBoundsNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address input,
                CudaDeviceAddressMap::Address bounds,
                CudaDeviceAddressMap::Address cache,
                CudaDeviceAddressMap::Address geometry,
                std::uint32_t count,
                std::uint32_t capacity) const {
        if(!count)return;
        if(count>capacity)
            throw std::invalid_argument(
                "Source Sc shape transform count exceeds source capacity");
        const auto groups=(std::uint64_t(count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Source Sc box transform GPU dispatch exceeds device limit");
        std::array<std::uint32_t,5> params{count};
        auto view=[&](CudaDeviceAddressMap::Address address,
                      std::size_t bytes,std::uint32_t slot,
                      compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,params[slot+1u],max_view_);
        };
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
        dispatch.buffers={
            view(input,source_allocation_bytes(count,48u),
                 0u,compute::BufferAccess::Read),
            view(bounds,source_allocation_bytes(capacity,24u),
                 1u,compute::BufferAccess::Write),
            view(cache,source_allocation_bytes(capacity,32u),
                 2u,compute::BufferAccess::Write),
            view(geometry,source_allocation_bytes(capacity,16u),
                 3u,compute::BufferAccess::Write)};
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),
                    params.data(),sizeof(params));
        commands.emplace_back(std::move(dispatch));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address input,
                CudaDeviceAddressMap::Address bounds,
                CudaDeviceAddressMap::Address cache,
                CudaDeviceAddressMap::Address geometry,
                std::uint32_t count,
                std::uint32_t capacity) const {
        append(stream.commands(),addresses,input,bounds,cache,geometry,count,capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
