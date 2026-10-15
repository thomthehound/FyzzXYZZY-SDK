// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_region_histograms_native.h"
#include "physx_gpu/source_native_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups=32u, kThreads=256u;
template<class T> CudaDeviceAddressMap::Address address(T* ptr) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(ptr));
}
struct Parameters {
    std::uint32_t boxes, data_bias, blocks_bias, descriptor_bias;
    std::uint32_t total_field_offset, reserved[3] = {};
};
static_assert(sizeof(Parameters)==32u, "PhysX source region prefix ABI changed");
} // namespace

SourceRegionHistogramsNativeStage::SourceRegionHistogramsNativeStage(
    compute::Context& context,
    compute::ShaderBinary first_portable,
    compute::ShaderBinary first_wave,
    compute::ShaderBinary second, bool force_portable)
    : context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGroups)
        throw std::runtime_error("PhysX source region histogram requires 32x256");

    using_wave_=!force_portable && gpu.supports_wave_arithmetic &&
        gpu.subgroup_size>=4u && 256u%gpu.subgroup_size==0u;
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count=3u;
    desc.push_constant_bytes=sizeof(Parameters);
    // GPU source allocations can all be slices of a single native root.
    // Consistent RAW UAV binding avoids mixed D3D12 subresource states.
    desc.storage_buffer_views.assign(3u,
        compute::StorageBufferViewKind::RawReadWrite);
    try{
        desc.shader=using_wave_ ? std::move(first_wave) : std::move(first_portable);
        first_=context_.create_compute_pipeline(desc);
        desc.shader=std::move(second);
        second_=context_.create_compute_pipeline(desc);
    }catch(...){
        if(first_) context_.destroy_compute_pipeline(first_);
        throw;
    }
}
SourceRegionHistogramsNativeStage::~SourceRegionHistogramsNativeStage(){
    if(first_) context_.destroy_compute_pipeline(first_);
    if(second_) context_.destroy_compute_pipeline(second_);
}

void SourceRegionHistogramsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceRegionHistogramsNativeLayout layout) const {
    // Precisely the source created-handle gate; an idle update enqueues none.
    if(!source.numCreatedHandles)return;
    const std::uint64_t preRemove =
        std::uint64_t(source.numPreviousHandles)+source.numCreatedHandles;
    if(preRemove<source.numRemovedHandles)
        throw std::invalid_argument("Invalid original PhysX region population");
    const auto boxes=preRemove-source.numRemovedHandles;
    if(!boxes || !layout.descriptor ||
       boxes>(std::numeric_limits<std::uint32_t>::max()-255u)/2u ||
       !layout.projection_slots ||
       preRemove*2u>layout.projection_slots)
        throw std::invalid_argument("Invalid PhysX region histogram projection capacity");
    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(gpu.storage_buffer_offset_alignment);
    if(!alignment || alignment%4u || !gpu.max_storage_buffer_range)
        throw std::invalid_argument("Missing native PhysX region histogram limits");
    const auto data_bytes=source_allocation_bytes(layout.projection_slots,4u);
    const auto desc_bytes=sizeof(physx::PxgBroadPhaseDesc);
    const auto desc_offset=layout.descriptor;
    constexpr std::size_t start_scalar=offsetof(physx::PxgBroadPhaseDesc,startRegionAccumTotal);
    constexpr std::size_t region_scalar=offsetof(physx::PxgBroadPhaseDesc,regionAccumTotal);
    static_assert(start_scalar%4u==0u && region_scalar%4u==0u,
                  "Source PxgBroadPhaseDesc totals require D32 alignment");
    std::vector<compute::BatchCommand> pending;
    pending.reserve(4u);
    for(std::uint32_t which=0u;which<2u;++which){
        const auto data=which==0u ? address(source.startRegionAccum) :
                                     address(source.regionAccum);
        const auto block=which==0u ? address(source.blockStartRegionAccum) :
                                      address(source.blockRegionAccum);
        const std::uint32_t scalar=static_cast<std::uint32_t>(
            which==0u ? start_scalar:region_scalar);
        // Both original passes bind the same source allocations.
        // First computes local exclusive sums and GPU block totals,
        // second finishes the scan and writes the GPU descriptor scalar.
        for(std::uint32_t pass=0u;pass<2u;++pass){
            compute::DispatchCommand cmd;
            cmd.pipeline=pass==0u?first_:second_;
            cmd.groups={kGroups,1u,1u};
            cmd.buffers.reserve(3u);
            Parameters params{};
            params.boxes=static_cast<std::uint32_t>(boxes);
            params.total_field_offset=scalar;
            cmd.buffers.push_back(source_native_view(addresses,data,data_bytes,
                alignment,0u,compute::BufferAccess::ReadWrite,
                params.data_bias,gpu.max_storage_buffer_range));
            cmd.buffers.push_back(source_native_view(addresses,block,32u*4u,
                alignment,1u,pass==0u?compute::BufferAccess::Write:
                                     compute::BufferAccess::Read,
                params.blocks_bias,gpu.max_storage_buffer_range));
            cmd.buffers.push_back(source_native_view(addresses,desc_offset,desc_bytes,
                alignment,2u,pass==0u?compute::BufferAccess::Read:
                                     compute::BufferAccess::Write,
                params.descriptor_bias,gpu.max_storage_buffer_range));
            cmd.push_constants.resize(sizeof(params));
            std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
            pending.emplace_back(std::move(cmd));
        }
    }
    commands.reserve(commands.size()+pending.size());
    for(auto& cmd:pending)commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
