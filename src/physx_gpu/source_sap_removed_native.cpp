// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_native_view.h"
#include "physx_port/gpubroadphase/include/PxgSapBox1D.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
template<class T>
CudaDeviceAddressMap::Address addr(T* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
constexpr std::uint32_t kGrid=32u, kThreads=256u;
static_assert(sizeof(physx::PxgSapBox1D)==8u);
void validate(const physx::PxgBroadPhaseDesc& source,
              SourceSapRemovedNativeLayout layout,bool projections) {
    if(!source.numRemovedHandles)return;
    const auto population=std::uint64_t(source.numPreviousHandles)+
                          source.numCreatedHandles;
    if(!layout.box_capacity || !layout.endpoint_slots ||
       source.numRemovedHandles>source.numPreviousHandles ||
       source.numPreviousHandles>layout.box_capacity ||
       population>std::numeric_limits<std::uint32_t>::max()/2u ||
       layout.endpoint_slots<2u*(projections?population:
                                 std::uint64_t(source.numPreviousHandles)) ||
       layout.box_capacity>std::numeric_limits<std::size_t>::max()/24u ||
       layout.endpoint_slots>std::numeric_limits<std::size_t>::max()/4u)
        throw std::invalid_argument("Invalid source SAP removal population or capacity");
}
} // namespace

SourceSapRemovedNativeStage::SourceSapRemovedNativeStage(
    compute::Context& context,compute::ShaderBinary mark_shader,
    compute::ShaderBinary projection_shader) : context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGrid ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original SAP 32x256 deletion grid unavailable");
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(mark_shader);
    desc.storage_buffer_binding_count=8u;
    desc.storage_buffer_views.assign(
        8u,compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=12u*sizeof(std::uint32_t);
    mark_pipeline_=context_.create_compute_pipeline(desc);
    try {
        desc.shader=std::move(projection_shader);
        desc.storage_buffer_binding_count=7u;
        desc.storage_buffer_views.assign(
            7u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=8u*sizeof(std::uint32_t);
        projection_pipeline_=context_.create_compute_pipeline(desc);
    } catch(...) {
        context_.destroy_compute_pipeline(mark_pipeline_);
        mark_pipeline_=0;
        throw;
    }
}
SourceSapRemovedNativeStage::~SourceSapRemovedNativeStage() {
    if(projection_pipeline_) context_.destroy_compute_pipeline(projection_pipeline_);
    if(mark_pipeline_) context_.destroy_compute_pipeline(mark_pipeline_);
}

void SourceSapRemovedNativeStage::append_mark_removed(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapRemovedNativeLayout layout) const {
    if(!source.numRemovedHandles)return;
    validate(source,layout,false);
    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    std::array<std::uint32_t,12> params{};
    params[8]=source.numRemovedHandles;
    compute::DispatchCommand cmd;
    cmd.pipeline=mark_pipeline_;
    cmd.groups={kGrid,1u,1u};
    cmd.buffers.reserve(8u);
    cmd.buffers.push_back(source_native_view(
        addresses,addr(source.updateData_removedHandles),
        source_allocation_bytes(source.numRemovedHandles,4u),alignment,0u,
        compute::BufferAccess::Read,params[0],gpu.max_storage_buffer_range));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.push_back(source_native_view(
            addresses,addr(source.boxSapBox1D[axis]),
            source_allocation_bytes(layout.box_capacity,8u),alignment,axis+1u,
            compute::BufferAccess::Read,params[axis+1u],gpu.max_storage_buffer_range));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.push_back(source_native_view(
            addresses,addr(source.boxHandles[1][axis]),
            source_allocation_bytes(layout.endpoint_slots,4u),alignment,axis+4u,
            compute::BufferAccess::ReadWrite,params[axis+4u],gpu.max_storage_buffer_range));
    cmd.buffers.push_back(source_native_view(
        addresses,addr(source.oldIntegerBounds),
        source_allocation_bytes(layout.box_capacity,24u),alignment,7u,
        compute::BufferAccess::ReadWrite,params[7],gpu.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),params.data(),sizeof(params));
    commands.emplace_back(std::move(cmd));
}

void SourceSapRemovedNativeStage::append_remove_projections(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapRemovedNativeLayout layout) const {
    if(!source.numRemovedHandles)return;
    validate(source,layout,true);
    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    std::array<std::uint32_t,8> params{};
    params[7]=source.numRemovedHandles;
    compute::DispatchCommand cmd;
    cmd.pipeline=projection_pipeline_;
    cmd.groups={kGrid,1u,1u};
    cmd.buffers.reserve(7u);
    cmd.buffers.push_back(source_native_view(
        addresses,addr(source.updateData_removedHandles),
        source_allocation_bytes(source.numRemovedHandles,4u),alignment,0u,
        compute::BufferAccess::Read,params[0],gpu.max_storage_buffer_range));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.push_back(source_native_view(
            addresses,addr(source.boxSapBox1D[axis]),
            source_allocation_bytes(layout.box_capacity,8u),alignment,axis+1u,
            compute::BufferAccess::ReadWrite,params[axis+1u],gpu.max_storage_buffer_range));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.push_back(source_native_view(
            addresses,addr(source.boxProjections[axis]),
            source_allocation_bytes(layout.endpoint_slots,4u),alignment,axis+4u,
            compute::BufferAccess::ReadWrite,params[axis+4u],gpu.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),params.data(),sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
