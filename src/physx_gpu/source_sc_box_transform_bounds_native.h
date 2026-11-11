// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <bit>
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
    float local_position[3]{};
    std::uint32_t geometry_kind=0;
    float local_orientation[4]{0,0,0,1};
    std::uint32_t next_element=0xffffffffu;
    std::uint32_t reserved[3]{};
};
static_assert(sizeof(SourceScBoxTransformInputNative)==96u &&
    offsetof(SourceScBoxTransformInputNative,local_position)==48u &&
    offsetof(SourceScBoxTransformInputNative,local_orientation)==64u &&
    offsetof(SourceScBoxTransformInputNative,next_element)==80u);

struct SourceScBoxGeometryNative {
    float half_extents[3];
    float contact_distance;
    float local_position[3]{};
    std::uint32_t geometry_kind=0;
    float local_orientation[4]{0,0,0,1};
    std::uint32_t next_element=0xffffffffu;
    std::uint32_t reserved[3]{};
};
static_assert(sizeof(SourceScBoxGeometryNative)==64u &&
    offsetof(SourceScBoxGeometryNative,local_position)==16u &&
    offsetof(SourceScBoxGeometryNative,local_orientation)==32u &&
    offsetof(SourceScBoxGeometryNative,next_element)==48u);

inline SourceScBoxTransformInputNative source_shape_transform_input(
    const native::GpuBroadphaseBox& shape,std::uint32_t handle) {
    const auto p=shape.local_pose.position;
    const auto q=shape.local_pose.orientation;
    return {handle,{shape.position.x,shape.position.y,shape.position.z},
        {shape.orientation.x,shape.orientation.y,shape.orientation.z,shape.orientation.w},
        {shape.half_extents.x,shape.half_extents.y,shape.half_extents.z},shape.contact_offset,
        {p.x,p.y,p.z},shape.geometry_kind,{q.x,q.y,q.z,q.w},shape.next_shape,
        {shape.convex_vertex_offset,shape.convex_vertex_count,
         std::bit_cast<std::uint32_t>(shape.convex_margin_half_extent)}};
}

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
        desc.storage_buffer_binding_count=5u;
        desc.storage_buffer_views.assign(
            5u,compute::StorageBufferViewKind::RawReadWrite);
        desc.storage_buffer_views[0]=compute::StorageBufferViewKind::RawRead;
        desc.storage_buffer_views[4]=compute::StorageBufferViewKind::RawRead;
        desc.push_constant_bytes=24u;
        pipeline_=context_.create_compute_pipeline(desc);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,"sc.box_bounds");
#endif
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
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t vertex_count,
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
        std::array<std::uint32_t,6> params{count};
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
            view(input,source_allocation_bytes(count,sizeof(SourceScBoxTransformInputNative)),
                 0u,compute::BufferAccess::Read),
            view(bounds,source_allocation_bytes(capacity,24u),
                 1u,compute::BufferAccess::Write),
            view(cache,source_allocation_bytes(capacity,32u),
                 2u,compute::BufferAccess::Write),
            view(geometry,source_allocation_bytes(capacity,sizeof(SourceScBoxGeometryNative)),
                 3u,compute::BufferAccess::Write),
            view(vertex_count?convex_vertices:input,
                 vertex_count?source_allocation_bytes(vertex_count,12u):
                              source_allocation_bytes(count,sizeof(SourceScBoxTransformInputNative)),
                 4u,compute::BufferAccess::Read)};
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
                CudaDeviceAddressMap::Address convex_vertices,
                std::uint32_t vertex_count,
                std::uint32_t count,
                std::uint32_t capacity) const {
        append(stream.commands(),addresses,input,bounds,cache,geometry,
               convex_vertices,vertex_count,count,capacity);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
