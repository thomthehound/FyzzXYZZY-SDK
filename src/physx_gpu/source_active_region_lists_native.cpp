// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_active_region_lists_native.h"
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
template<class T> CudaDeviceAddressMap::Address address(T* ptr) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(ptr));
}
struct Parameters {
    std::uint32_t boxes, sorted_bias, sap_bias, ranges_bias;
    std::uint32_t prefix_bias, descriptor_bias, active_bias, starts_bias;
    std::uint32_t ordered_active_bias, ordered_starts_bias;
    std::uint32_t total_field_offset, reserved;
};
static_assert(sizeof(Parameters)==48u,"PhysX source active-region ABI changed");
constexpr std::uint32_t kFirstGroups=32u, kOtherGroups=64u;
} // namespace

SourceActiveRegionListsNativeStage::SourceActiveRegionListsNativeStage(
    compute::Context& context, compute::ShaderBinary writeout,
    compute::ShaderBinary compute_wave, compute::ShaderBinary compute_portable,
    compute::ShaderBinary output, bool force_portable)
    : context_(context) {
    const auto& gpu=context.device_info();
    if (gpu.max_workgroup_invocations<512u || gpu.max_workgroup_size[0]<512u ||
        gpu.max_workgroup_count[0]<kOtherGroups)
        throw std::runtime_error("Original PhysX active region lists require 64x512 dispatch");
    using_wave_=!force_portable && gpu.supports_wave_arithmetic &&
        gpu.subgroup_size>=4u && 512u%gpu.subgroup_size==0u;
    const std::array<std::uint32_t,3> counts{6u,2u,9u};
    compute::ShaderBinary binaries[]{
        std::move(writeout),
        using_wave_?std::move(compute_wave):std::move(compute_portable),
        std::move(output)};
    try {
        for(std::uint32_t i=0u;i<3u;++i){
            compute::ComputePipelineDesc p;
            p.shader=std::move(binaries[i]);
            p.storage_buffer_binding_count=counts[i];
            p.push_constant_bytes=sizeof(Parameters);
            // PhysX buddy heaps may alias logical CUDA arrays within a
            // single D3D12 resource. RAW UAV states remain compatible.
            p.storage_buffer_views.assign(counts[i],
                compute::StorageBufferViewKind::RawReadWrite);
            pipelines_[i]=context_.create_compute_pipeline(p);
        }
    }catch(...){
        for(const auto id:pipelines_)
            if(id)context_.destroy_compute_pipeline(id);
        throw;
    }
}
SourceActiveRegionListsNativeStage::~SourceActiveRegionListsNativeStage(){
    for(const auto id:pipelines_)
        if(id)context_.destroy_compute_pipeline(id);
}

void SourceActiveRegionListsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceActiveRegionListsNativeLayout layout) const {
    if(!source.numCreatedHandles)return;
    const auto prior=std::uint64_t(source.numPreviousHandles)+source.numCreatedHandles;
    if(prior<source.numRemovedHandles)
        throw std::invalid_argument("Invalid source PhysX active-region population");
    const auto boxes=prior-source.numRemovedHandles;
    const auto n=boxes*2u;
    const auto retained=prior*2u;
    // Source histograms have 64 region slots, with 32-bit shader
    // addresses. These bounds also protect the host's CUDA rounding.
    if(!boxes || boxes>layout.box_capacity || !layout.descriptor ||
       boxes>(std::numeric_limits<std::uint32_t>::max()-511u)/2u ||
       boxes>std::numeric_limits<std::uint32_t>::max()/(2u*64u*4u) ||
       retained>layout.projection_slots ||
       retained*64u>layout.region_table_words)
        throw std::invalid_argument("PhysX source active-region capacity overflow");

    const auto& gpu=context_.device_info();
    const auto alignment=static_cast<std::size_t>(gpu.storage_buffer_offset_alignment);
    if(!alignment || alignment%4u || !gpu.max_storage_buffer_range)
        throw std::invalid_argument("Invalid PhysX native active-region GPU storage limits");
    const auto denseBytes=source_allocation_bytes(static_cast<std::size_t>(n*64u),4u);
    const auto projectionBytes=source_allocation_bytes(layout.projection_slots,4u);
    const auto sapBytes=source_allocation_bytes(layout.box_capacity,8u);
    const auto regionBytes=source_allocation_bytes(layout.box_capacity,32u);
    const auto descriptorBytes=sizeof(physx::PxgBroadPhaseDesc);
    constexpr auto totalField=offsetof(physx::PxgBroadPhaseDesc,startRegionAccumTotal);
    static_assert(totalField%4u==0u,"PhysX active-region GPU descriptor total needs D32 alignment");

    // Give every original CUDA allocation a native view. The order below
    // matches the source kernels' resource slots; no intermediate upload,
    // copy or CPU lookup of the GPU-published histogram total is allowed.
    struct Slice{
        CudaDeviceAddressMap::Address ptr;
        std::size_t bytes;
        std::uint32_t* bias;
        compute::BufferAccess access;
    };
    std::vector<compute::BatchCommand> pending;
    pending.reserve(3u);
    for(std::uint32_t pass=0u;pass<3u;++pass){
        compute::DispatchCommand cmd;
        cmd.pipeline=pipelines_[pass];
        cmd.groups={pass==0u?kFirstGroups:kOtherGroups,1u,1u};
        Parameters params{};
        params.boxes=static_cast<std::uint32_t>(boxes);
        params.total_field_offset=static_cast<std::uint32_t>(totalField);
        const auto sorted=Slice{address(source.boxHandles[0][0]),projectionBytes,
            &params.sorted_bias,compute::BufferAccess::Read};
        const auto sap=Slice{address(source.boxSapBox1D[0]),sapBytes,
            &params.sap_bias,compute::BufferAccess::Read};
        const auto ranges=Slice{address(source.regionRange),regionBytes,
            &params.ranges_bias,compute::BufferAccess::Read};
        const auto prefix=Slice{address(source.startRegionAccum),projectionBytes,
            &params.prefix_bias,compute::BufferAccess::Read};
        const auto gpuDesc=Slice{layout.descriptor,descriptorBytes,
            &params.descriptor_bias,compute::BufferAccess::Read};
        const auto active=Slice{address(source.activeRegionsHistogram),denseBytes,
            &params.active_bias,pass==1u?compute::BufferAccess::ReadWrite:
                                    pass==0u?compute::BufferAccess::Write:
                                              compute::BufferAccess::Read};
        const auto starts=Slice{address(source.startRegionsHistogram),denseBytes,
            &params.starts_bias,pass==1u?compute::BufferAccess::ReadWrite:
                                    pass==0u?compute::BufferAccess::Write:
                                              compute::BufferAccess::Read};
        const auto orderedActive=Slice{address(source.orderedActiveRegionHandles),denseBytes,
            &params.ordered_active_bias,compute::BufferAccess::Write};
        const auto orderedStarts=Slice{address(source.orderedStartRegionHandles),denseBytes,
            &params.ordered_starts_bias,compute::BufferAccess::Write};
        const std::array<Slice,6> writeout{
            sorted,ranges,prefix,gpuDesc,active,starts};
        const std::array<Slice,2> scan{active,starts};
        const std::array<Slice,9> output{
            sorted,ranges,prefix,gpuDesc,sap,starts,active,orderedStarts,orderedActive};
        const auto bind=[&](const Slice& slice,std::uint32_t index){
            cmd.buffers.push_back(source_native_view(
                addresses,slice.ptr,slice.bytes,alignment,index,slice.access,
                *slice.bias,gpu.max_storage_buffer_range));
        };
        if(pass==0u){
            cmd.buffers.reserve(writeout.size());
            for(std::uint32_t i=0;i<writeout.size();++i)bind(writeout[i],i);
        }else if(pass==1u){
            cmd.buffers.reserve(scan.size());
            for(std::uint32_t i=0;i<scan.size();++i)bind(scan[i],i);
        }else{
            cmd.buffers.reserve(output.size());
            for(std::uint32_t i=0;i<output.size();++i)bind(output[i],i);
        }
        cmd.push_constants.resize(sizeof(params));
        std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
        pending.emplace_back(std::move(cmd));
    }
    // Failures resolving a late original allocation must not leave the
    // previous two original source launches in the caller's work list.
    commands.reserve(commands.size()+pending.size());
    for(auto& cmd:pending)commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
