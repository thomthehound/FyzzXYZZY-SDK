// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_updated_native.h"
#include "physx_gpu/source_native_view.h"
#include "physx_port/gpubroadphase/include/PxgSapBox1D.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGrid=32u, kThreads=256u;
static_assert(sizeof(physx::PxgSapBox1D)==8u);
template<class T> CudaDeviceAddressMap::Address ptr(T* p) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
}
SourceSapUpdatedNativeStage::SourceSapUpdatedNativeStage(
    compute::Context& context,compute::ShaderBinary shader)
    :context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGrid ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original source BP_UPDATE_UPDATEDPAIRS 32x256 grid unavailable");
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(shader);
    desc.storage_buffer_binding_count=10u;
    desc.storage_buffer_views.assign(
        10u,compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=12u*sizeof(std::uint32_t);
    pipeline_=context_.create_compute_pipeline(desc);
}
SourceSapUpdatedNativeStage::~SourceSapUpdatedNativeStage() {
    if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
}
void SourceSapUpdatedNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapUpdatedNativeLayout layout) const {
    const auto words=source.aabbMngr_changedHandleBitMapWordCounts;
    if(!words)return;
    if(!layout.box_capacity || !layout.endpoint_slots ||
       words>std::numeric_limits<std::uint32_t>::max()/32u ||
       layout.box_capacity>words*32u ||
       layout.box_capacity>std::numeric_limits<std::size_t>::max()/24u ||
       layout.endpoint_slots>std::numeric_limits<std::size_t>::max()/4u)
        throw std::invalid_argument("Invalid original AABB manager bitmap or SAP capacities");
    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    std::array<std::uint32_t,12> params{};
    params[10]=words;
    params[11]=layout.box_capacity;
    const std::array<CudaDeviceAddressMap::Address,10> pointers{
        ptr(source.aabbMngr_changedHandleMap),
        ptr(source.aabbMngr_addedHandleMap),
        ptr(source.aabbMngr_removedHandleMap),
        ptr(source.boxSapBox1D[0]),
        ptr(source.boxSapBox1D[1]),
        ptr(source.boxSapBox1D[2]),
        ptr(source.boxProjections[0]),
        ptr(source.boxProjections[1]),
        ptr(source.boxProjections[2]),
        ptr(source.newIntegerBounds)
    };
    const std::array<std::size_t,10> lengths{
        source_allocation_bytes(words,4u),
        source_allocation_bytes(words,4u),
        source_allocation_bytes(words,4u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.box_capacity,8u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.endpoint_slots,4u),
        source_allocation_bytes(layout.box_capacity,24u)
    };
    compute::DispatchCommand cmd;
    cmd.pipeline=pipeline_;
    cmd.groups={kGrid,1u,1u};
    cmd.buffers.reserve(pointers.size());
    for(std::uint32_t i=0;i<pointers.size();++i)
        cmd.buffers.push_back(source_native_view(
            addresses,pointers[i],lengths[i],alignment,i,
            i>=6u&&i<=8u ? compute::BufferAccess::ReadWrite:
                           compute::BufferAccess::Read,
            params[i],gpu.max_storage_buffer_range));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),params.data(),sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
