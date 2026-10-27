// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_incremental_pairs_native.h"
#include "physx_gpu/source_native_view.h"
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
constexpr std::uint32_t kBlocks=256u,kAxes=3u,kThreads=256u,kViews=41u;
struct alignas(16) Params {std::array<std::uint32_t,28> words{};};
static_assert(sizeof(Params)==112u);
template<class T>CudaDeviceAddressMap::Address ptr(T* p){
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
}
SourceSapIncrementalPairsNativeStage::SourceSapIncrementalPairsNativeStage(
    compute::Context& context,compute::ShaderBinary shader):context_(context){
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kBlocks ||
       gpu.max_workgroup_count[1]<kAxes ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original PhysX BP_INCREMENTAL_SAP 256x3x256 unavailable");
    alignment_=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    max_view_=gpu.max_storage_buffer_range;
    compute::ComputePipelineDesc desc;
    desc.shader=std::move(shader);
    desc.storage_buffer_binding_count=kViews;
    desc.storage_buffer_views.assign(kViews,
        compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=sizeof(Params);
    pipeline_=context_.create_compute_pipeline(desc);
}
SourceSapIncrementalPairsNativeStage::~SourceSapIncrementalPairsNativeStage(){
    if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
}
void SourceSapIncrementalPairsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    CudaDeviceAddressMap::Address source_device_descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapIncrementalPairsNativeLayout layout) const {
    if(!source.numPreviousHandles)return;
    const auto boxes=source.numPreviousHandles;
    if(!layout.box_capacity || !layout.endpoint_slots ||
       !layout.ordered_handle_slots || !layout.report_capacity ||
       boxes>layout.box_capacity ||
       boxes>(std::numeric_limits<std::uint32_t>::max()-255u)/2u ||
       layout.endpoint_slots<std::uint64_t(boxes)*2u ||
       layout.ordered_handle_slots<boxes ||
       !source_device_descriptor ||
       source_device_descriptor>
           std::numeric_limits<CudaDeviceAddressMap::Address>::max()-
           offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex))
        throw std::invalid_argument("Source incremental SAP buffer capacities invalid");
    const auto endpoints=source_allocation_bytes(layout.endpoint_slots,4u);
    const auto ordered=source_allocation_bytes(layout.ordered_handle_slots,4u);
    const auto bounds=source_allocation_bytes(layout.box_capacity,24u);
    const auto objects=source_allocation_bytes(layout.box_capacity,4u);
    const auto reports=source_allocation_bytes(layout.report_capacity,8u);
    std::array<CudaDeviceAddressMap::Address,kViews> pointers{};
    std::array<std::size_t,kViews> lengths{};
    for(std::uint32_t axis=0u;axis<kAxes;++axis){
        const auto i=axis*11u;
        const std::array<CudaDeviceAddressMap::Address,11> inputs{
            ptr(source.boxHandles[0][axis]),
            ptr(source.boxProjectionRanks[axis]),
            ptr(source.incrementalComparisons[axis]),
            ptr(source.startPtHistogram[0][axis]),
            ptr(source.endPtHistogram[0][axis]),
            ptr(source.startPtHistogram[1][axis]),
            ptr(source.endPtHistogram[1][axis]),
            ptr(source.startPointHandles[0][axis]),
            ptr(source.endPointHandles[0][axis]),
            ptr(source.startPointHandles[1][axis]),
            ptr(source.endPointHandles[1][axis])};
        for(std::uint32_t field=0u;field<11u;++field){
            pointers[i+field]=inputs[field];
            lengths[i+field]=field>=7u?ordered:endpoints;
        }
    }
    pointers[33u]=ptr(source.oldIntegerBounds);lengths[33u]=bounds;
    pointers[34u]=ptr(source.newIntegerBounds);lengths[34u]=bounds;
    pointers[35u]=ptr(source.updateData_groups);lengths[35u]=objects;
    pointers[36u]=ptr(source.updateData_envIDs?
        source.updateData_envIDs:source.updateData_groups);
    lengths[36u]=objects;
    pointers[37u]=ptr(source.foundPairReport);lengths[37u]=reports;
    pointers[38u]=ptr(source.lostPairReport);lengths[38u]=reports;
    pointers[39u]=source_device_descriptor+
        offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex);
    lengths[39u]=2u*sizeof(std::uint32_t);
    pointers[40u]=source_device_descriptor+
        offsetof(physx::PxgBroadPhaseDesc,totalIncrementalComparisons);
    lengths[40u]=3u*sizeof(std::uint32_t);
    Params params{};
    params.words[24u]=boxes;
    params.words[25u]=layout.box_capacity;
    params.words[26u]=layout.report_capacity;
    params.words[27u]=source.updateData_envIDs?1u:0u;
    compute::DispatchCommand cmd;
    cmd.pipeline=pipeline_;
    cmd.groups={kBlocks,kAxes,1u};
    cmd.buffers.reserve(kViews);
    for(std::uint32_t slot=0u;slot<kViews;++slot){
        std::uint32_t offset=0u;
        const auto access=slot==37u||slot==38u||slot==39u?
            compute::BufferAccess::ReadWrite:compute::BufferAccess::Read;
        const auto view=source_native_view(
            addresses,pointers[slot],lengths[slot],alignment_,slot,
            access,offset,max_view_);
        if(offset>0xffffu)
            throw std::out_of_range("Source incremental SAP view offset exceeds 16 bits");
        params.words[slot>>1u]|=offset<<((slot&1u)*16u);
        cmd.buffers.push_back(view);
    }
    cmd.push_constants.resize(sizeof(params));
    std::memcpy(cmd.push_constants.data(),&params,sizeof(params));
    commands.emplace_back(std::move(cmd));
}
} // namespace fyzz::physx_gpu
