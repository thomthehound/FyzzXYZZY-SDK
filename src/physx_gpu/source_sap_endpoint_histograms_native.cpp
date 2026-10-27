// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_endpoint_histograms_native.h"
#include "physx_gpu/source_native_view.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fyzz::physx_gpu {
namespace {
constexpr std::uint32_t kGroups=32u,kAxes=3u,kThreads=256u,kBuffers=48u;
struct alignas(16) Params {
    std::array<std::uint32_t,28> packed{};
};
static_assert(sizeof(Params)==112u);
template<class T> CudaDeviceAddressMap::Address ptr(T* p){
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
}
SourceSapEndpointHistogramsNativeStage::SourceSapEndpointHistogramsNativeStage(
    compute::Context& context,compute::ShaderBinary first,
    compute::ShaderBinary second) :context_(context){
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGroups ||
       gpu.max_workgroup_count[1]<kAxes ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original PhysX 32x3x256 endpoint histogram grid unavailable");
    alignment_=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    max_view_=gpu.max_storage_buffer_range;
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count=kBuffers;
    desc.storage_buffer_views.assign(kBuffers,
        compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=sizeof(Params);
    desc.shader=std::move(first);
    first_=context_.create_compute_pipeline(desc);
    try{
        desc.shader=std::move(second);
        second_=context_.create_compute_pipeline(desc);
    }catch(...){
        context_.destroy_compute_pipeline(first_);
        first_=0;
        throw;
    }
}
SourceSapEndpointHistogramsNativeStage::~SourceSapEndpointHistogramsNativeStage(){
    if(second_)context_.destroy_compute_pipeline(second_);
    if(first_)context_.destroy_compute_pipeline(first_);
}
void SourceSapEndpointHistogramsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapEndpointHistogramsNativeLayout layout,
    bool is_incremental) const {
    const auto previous=source.numPreviousHandles;
    const auto created=source.numCreatedHandles;
    const auto removed=source.numRemovedHandles;
    const auto population=std::uint64_t(previous)+created;
    if(population<removed || population>std::numeric_limits<std::uint32_t>::max() ||
       population>std::numeric_limits<std::uint32_t>::max()/2u ||
       previous>std::numeric_limits<std::uint32_t>::max()/2u ||
       !layout.endpoint_slots || !layout.ordered_handle_slots)
        throw std::invalid_argument("Invalid source SAP endpoint population");
    const auto current=static_cast<std::uint32_t>(population-removed);
    const auto active=is_incremental?previous:current;
    const auto largest=(is_incremental&&removed)?previous:active;
    if(std::uint64_t(largest)*2u>layout.endpoint_slots ||
       largest>layout.ordered_handle_slots)
        throw std::invalid_argument("Insufficient source endpoint or ordered-handle capacity");

    const auto endpoints=source_allocation_bytes(layout.endpoint_slots,4u);
    const auto ordered=source_allocation_bytes(layout.ordered_handle_slots,4u);
    const auto blocks=source_allocation_bytes(kGroups,4u);
    std::array<CudaDeviceAddressMap::Address,kBuffers> ptrs{};
    std::array<std::size_t,kBuffers> lengths{};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto base=axis*16u;
        for(std::uint32_t frame=0u;frame<2u;++frame){
            const auto slot=base+frame*7u;
            ptrs[slot+0u]=ptr(source.boxHandles[frame][axis]);
            ptrs[slot+1u]=ptr(source.endPtHistogram[frame][axis]);
            ptrs[slot+2u]=ptr(source.startPtHistogram[frame][axis]);
            ptrs[slot+3u]=ptr(source.endPointHandles[frame][axis]);
            ptrs[slot+4u]=ptr(source.startPointHandles[frame][axis]);
            ptrs[slot+5u]=ptr(source.blockEndPtHistogram[frame][axis]);
            ptrs[slot+6u]=ptr(source.blockStartPtHistogram[frame][axis]);
            for(std::uint32_t i=0u;i<7u;++i)
                lengths[slot+i]=(i>=5u)?blocks:((i==3u||i==4u)?ordered:endpoints);
        }
        ptrs[base+14u]=ptr(source.totalEndPtHistogram[axis]);
        ptrs[base+15u]=ptr(source.blockTotalEndPtHistogram[axis]);
        lengths[base+14u]=endpoints;
        lengths[base+15u]=blocks;
    }
    Params params{};
    params.packed[24u]=previous;
    params.packed[25u]=created;
    params.packed[26u]=removed;
    params.packed[27u]=is_incremental?1u:0u;
    std::array<compute::BufferBinding,kBuffers> bindings{};
    // Source CUDA base views normally have <=128B alignment deltas.
    // Pack two independent 16-bit shader-local offsets into one DWORD,
    // keeping all 48 original resources and the exact TWO source launches
    // inside Vulkan's guaranteed 128-byte push constant floor.
    for(std::uint32_t slot=0u;slot<kBuffers;++slot){
        std::uint32_t offset=0u;
        const auto field=slot%16u;
        const bool write=field!=0u&&field!=7u;
        bindings[slot]=source_native_view(
            addresses,ptrs[slot],lengths[slot],alignment_,slot,
            write?compute::BufferAccess::ReadWrite:compute::BufferAccess::Read,
            offset,max_view_);
        if(offset>0xffffu)
            throw std::out_of_range("Original source SAP view requires more than 16-bit aligned offset");
        params.packed[slot>>1u]|=offset<<((slot&1u)*16u);
    }
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2u);
    for(auto pipeline:{first_,second_}){
        compute::DispatchCommand cmd;
        cmd.pipeline=pipeline;
        cmd.groups={kGroups,kAxes,1u};
        cmd.buffers.assign(bindings.begin(),bindings.end());
        cmd.push_constants.resize(sizeof(params));
        std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
        staged.emplace_back(std::move(cmd));
    }
    commands.reserve(commands.size()+staged.size());
    commands.insert(commands.end(),std::make_move_iterator(staged.begin()),
                    std::make_move_iterator(staged.end()));
}
} // namespace fyzz::physx_gpu
