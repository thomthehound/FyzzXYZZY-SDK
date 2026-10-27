// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sapbox_indices_native.h"
#include "physx_gpu/source_native_view.h"
#include "PxgSapBox1D.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
struct alignas(16) Parameters {
    std::uint32_t biases[6]{};
    std::uint32_t num_projections=0;
    std::uint32_t reserved=0;
};
static_assert(sizeof(Parameters)==32u);
static_assert(sizeof(physx::PxgSapBox1D)==8u,
              "Original PxgSapBox1D source stride changed");
template<class T>
CudaDeviceAddressMap::Address addr(T* value) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(value));
}
}
SourceSapBoxIndicesNativeStage::SourceSapBoxIndicesNativeStage(
    compute::Context& context,compute::ShaderBinary shader)
    : context_(context) {
    const auto& device=context.device_info();
    if(device.max_workgroup_invocations<256u ||
       device.max_workgroup_size[0]<256u ||
       !device.max_workgroup_count[0] ||
       !device.max_storage_buffer_range)
        throw std::runtime_error("Original BP_INITIALIZE_SAPBOX grid unavailable");
    view_alignment_=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,device.storage_buffer_offset_alignment));
    max_view_bytes_=device.max_storage_buffer_range;
    max_groups_=device.max_workgroup_count[0];
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(shader);
    desc.storage_buffer_binding_count=6u;
    desc.storage_buffer_views.assign(
        6u,compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=sizeof(Parameters);
    pipeline_=context_.create_compute_pipeline(desc);
}
SourceSapBoxIndicesNativeStage::~SourceSapBoxIndicesNativeStage(){
    if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
}
void SourceSapBoxIndicesNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapBoxIndicesNativeLayout layout,
    std::uint32_t num_handles,bool is_new) const {
    // Source initializeSapBoxKernel returns before dispatch for zero blocks.
    if(!num_handles)return;
    if(!layout.box_capacity || num_handles>layout.box_capacity ||
       num_handles>std::numeric_limits<std::uint32_t>::max()/2u ||
       layout.endpoint_slots<num_handles*2u ||
       layout.box_capacity>std::numeric_limits<std::size_t>::max()/8u ||
       layout.endpoint_slots>std::numeric_limits<std::size_t>::max()/4u)
        throw std::invalid_argument("Invalid PhysX SAP box-index source capacities");
    const std::uint32_t projections=num_handles*2u;
    const std::uint32_t groups=(projections+255u)/256u;
    if(groups>max_groups_)
        throw std::out_of_range("Original BP_INITIALIZE_SAPBOX grid exceeds device");
    Parameters params{};
    params.num_projections=projections;
    compute::DispatchCommand cmd;
    cmd.pipeline=pipeline_;
    cmd.groups={groups,1u,1u};
    cmd.buffers.reserve(6u);
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.emplace_back(source_native_view(
            addresses,addr(source.boxHandles[0][axis]),
            std::size_t(layout.endpoint_slots)*4u,view_alignment_,axis,
            compute::BufferAccess::Read,params.biases[axis],
            max_view_bytes_));
    for(std::uint32_t axis=0;axis<3u;++axis)
        cmd.buffers.emplace_back(source_native_view(
            addresses,addr(is_new?source.boxNewSapBox1D[axis]:
                                  source.boxSapBox1D[axis]),
            std::size_t(layout.box_capacity)*sizeof(physx::PxgSapBox1D),
            view_alignment_,axis+3u,compute::BufferAccess::ReadWrite,
            params.biases[axis+3u],max_view_bytes_));
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
