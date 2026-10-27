// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_incremental_counts_native.h"
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
constexpr std::uint32_t kGroups=32u,kThreads=256u;
constexpr std::uint32_t kFieldsPerAxis=8u;
constexpr std::uint32_t kBindings=25u;
struct alignas(16) Params {
    std::array<std::uint32_t,28u> words{};
};
static_assert(sizeof(Params)==112u);
template<class T> CudaDeviceAddressMap::Address ptr(T* p){
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(p));
}
void append_commands(std::vector<compute::BatchCommand>& commands,
                     std::vector<compute::BatchCommand>& staged) {
    commands.reserve(commands.size()+staged.size());
    commands.insert(commands.end(),std::make_move_iterator(staged.begin()),
                    std::make_move_iterator(staged.end()));
}
}
SourceSapIncrementalCountsNativeStage::SourceSapIncrementalCountsNativeStage(
    compute::Context& context,compute::ShaderBinary first,
    compute::ShaderBinary second) : context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kThreads ||
       gpu.max_workgroup_size[0]<kThreads ||
       gpu.max_workgroup_count[0]<kGroups ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Source incremental comparison 32x256 grid unavailable");
    alignment_=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    max_view_=gpu.max_storage_buffer_range;
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_binding_count=kBindings;
    desc.storage_buffer_views.assign(kBindings,
        compute::StorageBufferViewKind::RawReadWrite);
    desc.push_constant_bytes=sizeof(Params);
    desc.shader=std::move(first);
    first_=context_.create_compute_pipeline(desc);
    try {
        desc.shader=std::move(second);
        second_=context_.create_compute_pipeline(desc);
    }catch(...) {
        context_.destroy_compute_pipeline(first_);
        first_=0;
        throw;
    }
}
SourceSapIncrementalCountsNativeStage::~SourceSapIncrementalCountsNativeStage(){
    if(second_)context_.destroy_compute_pipeline(second_);
    if(first_)context_.destroy_compute_pipeline(first_);
}
void SourceSapIncrementalCountsNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    CudaDeviceAddressMap::Address source_device_descriptor,
    const CudaDeviceAddressMap& addresses,
    SourceSapIncrementalCountsLayout layout) const {
    const auto boxes=source.numPreviousHandles;
    if(boxes>(std::numeric_limits<std::uint32_t>::max()-255u)/2u ||
       layout.endpoint_slots<std::uint64_t(boxes)*2u ||
       (boxes && !layout.endpoint_slots))
        throw std::invalid_argument("Original incremental SAP projection capacity invalid");

    // No new totals allocation. The output alias is exactly the original
    // GPU PxgBroadPhaseDesc's three contiguous scalar fields.
    if(!source_device_descriptor ||
       source_device_descriptor >
           std::numeric_limits<CudaDeviceAddressMap::Address>::max()-
           offsetof(physx::PxgBroadPhaseDesc,totalIncrementalComparisons))
        throw std::invalid_argument("Original source GPU descriptor address invalid");
    std::array<CudaDeviceAddressMap::Address,kBindings> pointers{};
    std::array<std::size_t,kBindings> lengths{};
    const auto projection_bytes=source_allocation_bytes(
        std::max<std::uint32_t>(layout.endpoint_slots,1u),4u);
    const auto block_bytes=source_allocation_bytes(kGroups,4u);
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto slot=axis*kFieldsPerAxis;
        pointers[slot+0u]=ptr(source.boxHandles[0][axis]);
        pointers[slot+1u]=ptr(source.boxProjectionRanks[axis]);
        pointers[slot+2u]=ptr(source.startPtHistogram[0][axis]);
        pointers[slot+3u]=ptr(source.endPtHistogram[0][axis]);
        pointers[slot+4u]=ptr(source.startPtHistogram[1][axis]);
        pointers[slot+5u]=ptr(source.endPtHistogram[1][axis]);
        pointers[slot+6u]=ptr(source.incrementalComparisons[axis]);
        pointers[slot+7u]=ptr(source.incrementalBlockComparisons[axis]);
        for(std::uint32_t i=0u;i<kFieldsPerAxis;++i)
            lengths[slot+i]=(i==7u)?block_bytes:projection_bytes;
    }
    pointers[24u]=source_device_descriptor+
        offsetof(physx::PxgBroadPhaseDesc,totalIncrementalComparisons);
    lengths[24u]=3u*sizeof(std::uint32_t);
    Params params{};
    params.words[25u]=boxes;
    std::array<compute::BufferBinding,kBindings> bindings{};
    // Resolve every source allocation before adding EITHER kernel.
    // The preceding WR-ranked sort stays queued if this validation fails.
    for(std::uint32_t slot=0u;slot<kBindings;++slot) {
        bindings[slot]=source_native_view(
            addresses,pointers[slot],lengths[slot],alignment_,slot,
            slot%kFieldsPerAxis==6u || slot%kFieldsPerAxis==7u || slot==24u
                ?compute::BufferAccess::ReadWrite:compute::BufferAccess::Read,
            params.words[slot],max_view_);
    }
    // Keep two original dispatches adjacent in the SAME mStream.
    std::vector<compute::BatchCommand> staged;
    staged.reserve(2u);
    for(const auto pipeline:{first_,second_}){
        compute::DispatchCommand dispatch;
        dispatch.pipeline=pipeline;
        dispatch.groups={kGroups,1u,1u};
        dispatch.buffers.assign(bindings.begin(),bindings.end());
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),&params,sizeof(params));
        staged.emplace_back(std::move(dispatch));
    }
    append_commands(commands,staged);
}
} // namespace fyzz::physx_gpu
