// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_clear_new_flags_native.h"
#include "physx_gpu/source_sap_update_tail_native.h"
#include "physx_gpu/source_native_view.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
namespace fyzz::physx_gpu {
namespace {
struct alignas(16) Parameters {
    std::uint32_t created_bias=0;
    std::uint32_t sap_x_bias=0,sap_y_bias=0,sap_z_bias=0;
    std::uint32_t handles_x_bias=0,handles_y_bias=0,handles_z_bias=0;
    std::uint32_t created_count=0;
};
static_assert(sizeof(Parameters)==32u);
template<typename T>
CudaDeviceAddressMap::Address address(T* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
}
SourceClearNewFlagsNativeStage::SourceClearNewFlagsNativeStage(
    compute::Context& context,compute::ShaderBinary shader)
    :context_(context) {
    const auto& device=context_.device_info();
    if(device.max_workgroup_invocations<256u ||
       device.max_workgroup_size[0]<256u ||
       device.max_workgroup_count[0]<32u)
        throw std::runtime_error("Original PhysX BP_CLEAR_NEWFLAG grid unavailable");
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(shader);
    desc.storage_buffer_binding_count=7u;
    desc.storage_buffer_views.assign(
        7u,compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=sizeof(Parameters);
    pipeline_=context_.create_compute_pipeline(desc);
}
SourceClearNewFlagsNativeStage::~SourceClearNewFlagsNativeStage(){
    if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
}
void SourceClearNewFlagsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceClearNewFlagsNativeLayout layout) const {
    if(!source.numCreatedHandles)return;
    const auto retained=std::uint64_t(source.numPreviousHandles)+
                        source.numCreatedHandles;
    if(retained<source.numRemovedHandles ||
       retained-source.numRemovedHandles>layout.box_capacity ||
       !layout.box_capacity ||
       layout.box_capacity>std::numeric_limits<std::uint32_t>::max()/8u ||
       layout.endpoint_slots<2u*(retained-source.numRemovedHandles) ||
       layout.endpoint_slots>std::numeric_limits<std::uint32_t>::max()/4u)
        throw std::invalid_argument("Invalid original SAP clear-new capacities");
    const auto& device=context_.device_info();
    const auto align=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,device.storage_buffer_offset_alignment));
    if(!device.max_storage_buffer_range)
        throw std::runtime_error("Missing native source storage view limit");
    Parameters params{};
    params.created_count=source.numCreatedHandles;
    compute::DispatchCommand cmd;
    cmd.pipeline=pipeline_;
    cmd.groups={32u,1u,1u};
    cmd.buffers.reserve(7u);
    cmd.buffers.emplace_back(source_native_view(
        addresses,address(source.updateData_createdHandles),
        std::size_t(source.numCreatedHandles)*4u,align,0u,
        compute::BufferAccess::ReadWrite,params.created_bias,
        device.max_storage_buffer_range));
    std::array<std::uint32_t*,6> biases{
        &params.sap_x_bias,&params.sap_y_bias,&params.sap_z_bias,
        &params.handles_x_bias,&params.handles_y_bias,&params.handles_z_bias};
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.emplace_back(source_native_view(
            addresses,address(source.boxSapBox1D[axis]),
            std::size_t(layout.box_capacity)*8u,align,axis+1u,
            compute::BufferAccess::ReadWrite,*biases[axis],
            device.max_storage_buffer_range));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.emplace_back(source_native_view(
            addresses,address(source.boxHandles[0][axis]),
            std::size_t(layout.endpoint_slots)*4u,align,axis+4u,
            compute::BufferAccess::ReadWrite,*biases[axis+3u],
            device.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
