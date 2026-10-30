// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_native_view.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGrid=32u,kThreads=256u;
template<class T> CudaDeviceAddressMap::Address ptr(T* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
}
SourceSapCreatedNativeStage::SourceSapCreatedNativeStage(
    compute::Context& context,compute::ShaderBinary shader) :context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGrid ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original BP_UPDATE_CREATEDPAIRS 32x256 grid unavailable");
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(shader);
    desc.storage_buffer_binding_count=9u;
    desc.storage_buffer_views.assign(
        9u,compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=12u*sizeof(std::uint32_t);
    pipeline_=context_.create_compute_pipeline(desc);
}
SourceSapCreatedNativeStage::~SourceSapCreatedNativeStage(){
    if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
}
void SourceSapCreatedNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapCreatedNativeLayout layout) const {
    if(!source.numCreatedHandles)return;
    const auto population=std::uint64_t(source.numPreviousHandles)+
                          source.numCreatedHandles;
    const auto padded=population+(population&1u);
    if(!layout.box_capacity || !layout.endpoint_slots ||
       population>std::numeric_limits<std::uint32_t>::max()/2u ||
       padded*2u>layout.endpoint_slots ||
       layout.box_capacity>std::numeric_limits<std::size_t>::max()/24u ||
       layout.endpoint_slots>std::numeric_limits<std::size_t>::max()/4u)
        throw std::invalid_argument("Original source created-pair population exceeds SAP buffers");
    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    const std::array<CudaDeviceAddressMap::Address,9> pointers{
        ptr(source.updateData_createdHandles),
        ptr(source.newIntegerBounds),
        ptr(source.oldIntegerBounds),
        ptr(source.boxProjections[0]),
        ptr(source.boxProjections[1]),
        ptr(source.boxProjections[2]),
        ptr(source.boxHandles[0][0]),
        ptr(source.boxHandles[0][1]),
        ptr(source.boxHandles[0][2])
    };
    const std::array<std::size_t,9> lengths{
        source_allocation_bytes(source.numCreatedHandles,4u),
        source_allocation_bytes(layout.box_capacity,24u),
        source_allocation_bytes(layout.box_capacity,24u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u)
    };
    std::array<std::uint32_t,12> params{};
    params[9]=source.numPreviousHandles;
    params[10]=source.numCreatedHandles;
    params[11]=layout.box_capacity;
    compute::DispatchCommand cmd;
    cmd.pipeline=pipeline_;
    cmd.groups={kGrid,1u,1u};
    cmd.buffers.reserve(pointers.size());
    for(std::uint32_t i=0;i<pointers.size();++i)
        cmd.buffers.push_back(source_native_view(
            addresses,pointers[i],lengths[i],alignment,i,
            i<2u ? compute::BufferAccess::Read:
                   compute::BufferAccess::ReadWrite,
            params[i],gpu.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),params.data(),sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
