// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_sap_radix_native.h"
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
template<class T> CudaDeviceAddressMap::Address addr(T* value) {
    return static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(value));
}
constexpr std::uint32_t kRadixBlocks=32u, kSortAxes=3u, kRadixThreads=1024u;
constexpr std::uint32_t kRankBlocks=64u, kRankThreads=256u;
constexpr std::size_t kCountWords=16u*kRadixBlocks;
struct alignas(16) RankParams {
    std::uint32_t biases[3]{};
    std::uint32_t boxes=0;
    std::uint32_t pad[4]{};
};
struct alignas(16) GatherParams {
    std::uint32_t biases[9]{};
    std::uint32_t boxes=0;
    std::uint32_t pad[2]{};
};
struct alignas(16) CountParams {
    std::uint32_t biases[6]{};
    std::uint32_t keys=0;
    std::uint32_t shift=0;
};
struct alignas(16) ScatterParams {
    std::uint32_t biases[15]{};
    std::uint32_t keys=0,shift=0,pad[3]{};
};
static_assert(sizeof(RankParams)==32u);
static_assert(sizeof(GatherParams)==48u);
static_assert(sizeof(CountParams)==32u);
static_assert(sizeof(ScatterParams)==80u);

std::uint32_t padded_keys(std::uint64_t boxes) {
    const auto count=boxes*2u;
    if(count>std::numeric_limits<std::uint32_t>::max()-3u)
        throw std::out_of_range("PhysX source SAP radix projection count overflow");
    return static_cast<std::uint32_t>((count+3u)&~std::uint64_t{3u});
}
void commit(std::vector<compute::BatchCommand>& commands,
            std::vector<compute::BatchCommand>& staged) {
    commands.reserve(commands.size()+staged.size());
    commands.insert(commands.end(),std::make_move_iterator(staged.begin()),
                    std::make_move_iterator(staged.end()));
}
}

SourceSapRadixNativeStage::SourceSapRadixNativeStage(
    compute::Context& context,compute::ShaderBinary initialize,
    compute::ShaderBinary count,compute::ShaderBinary scatter,
    compute::ShaderBinary gather,
    compute::ShaderBinary count_wave,bool force_portable,
    compute::ShaderBinary scatter_wave) : context_(context) {
    const auto& gpu=context_.device_info();
    if(gpu.max_workgroup_invocations<kRadixThreads ||
       gpu.max_workgroup_size[0]<kRadixThreads ||
       gpu.max_workgroup_count[0]<kRankBlocks ||
       gpu.max_workgroup_count[1]<kSortAxes ||
       !gpu.max_storage_buffer_range)
        throw std::runtime_error("Original source SAP radix 32x3x1024/64x256 GPU launch unsupported");
    using_wave_count_=!force_portable && !count_wave.bytes.empty() &&
        gpu.supports_wave_arithmetic && gpu.subgroup_size>=4u &&
        kRadixThreads%gpu.subgroup_size==0u;
    using_wave_scatter_=!force_portable && !scatter_wave.bytes.empty() &&
        gpu.supports_wave_arithmetic && gpu.subgroup_size>=4u &&
        kRadixThreads%gpu.subgroup_size==0u;
    view_alignment_=static_cast<std::size_t>(
        std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
    max_view_bytes_=gpu.max_storage_buffer_range;
    compute::ComputePipelineDesc desc;
    desc.storage_buffer_views.assign(3u,compute::StorageBufferViewKind::RawReadWrite);
    desc.storage_buffer_binding_count=3u;
    desc.push_constant_bytes=sizeof(RankParams);
    try {
        desc.shader=std::move(initialize);
        init_pipeline_=context_.create_compute_pipeline(desc);
        desc.shader=using_wave_count_?std::move(count_wave):std::move(count);
        desc.storage_buffer_binding_count=6u;
        desc.storage_buffer_views.assign(6u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=sizeof(CountParams);
        count_pipeline_=context_.create_compute_pipeline(desc);
        desc.shader=using_wave_scatter_?std::move(scatter_wave):std::move(scatter);
        desc.storage_buffer_binding_count=15u;
        desc.storage_buffer_views.assign(15u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=sizeof(ScatterParams);
        scatter_pipeline_=context_.create_compute_pipeline(desc);
        desc.shader=std::move(gather);
        desc.storage_buffer_binding_count=9u;
        desc.storage_buffer_views.assign(9u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=sizeof(GatherParams);
        gather_pipeline_=context_.create_compute_pipeline(desc);
    } catch(...) {
        if(gather_pipeline_)context_.destroy_compute_pipeline(gather_pipeline_);
        if(scatter_pipeline_)context_.destroy_compute_pipeline(scatter_pipeline_);
        if(count_pipeline_)context_.destroy_compute_pipeline(count_pipeline_);
        if(init_pipeline_)context_.destroy_compute_pipeline(init_pipeline_);
        throw;
    }
}
SourceSapRadixNativeStage::~SourceSapRadixNativeStage() {
    if(gather_pipeline_)context_.destroy_compute_pipeline(gather_pipeline_);
    if(scatter_pipeline_)context_.destroy_compute_pipeline(scatter_pipeline_);
    if(count_pipeline_)context_.destroy_compute_pipeline(count_pipeline_);
    if(init_pipeline_)context_.destroy_compute_pipeline(init_pipeline_);
}

void SourceSapRadixNativeStage::append_sort(
    std::vector<compute::BatchCommand>& staged,
    std::span<const physx::PxgRadixSortDesc,6> radix,
    const CudaDeviceAddressMap& addresses,std::uint32_t keys) const {
    if(!keys) return;
    const auto key_bytes=source_allocation_bytes(keys,4u);
    const auto histogram_bytes=kCountWords*sizeof(std::uint32_t);
    staged.reserve(staged.size()+16u);
    // Keep the source's eight four-bit LSD passes. All 16 dispatches
    // append to the caller's existing native queue; no HtoD staging,
    // DtoH sync, allocation or inter-pass CPU processing is introduced.
    for(std::uint32_t pass=0u;pass<8u;++pass) {
        const auto parity=pass&1u;
        const auto shift=pass*4u;
        const auto axis_desc=radix.subspan(parity*3u,3u);
        CountParams count{};
        count.keys=keys;count.shift=shift;
        compute::DispatchCommand counter;
        counter.pipeline=count_pipeline_;
        counter.groups={kRadixBlocks,kSortAxes,1u};
        counter.buffers.reserve(6u);
        for(std::uint32_t axis=0u;axis<3u;++axis) {
            counter.buffers.push_back(source_native_view(
                addresses,addr(axis_desc[axis].inputKeys),
                key_bytes,view_alignment_,axis,compute::BufferAccess::Read,
                count.biases[axis],max_view_bytes_));
            counter.buffers.push_back(source_native_view(
                addresses,addr(axis_desc[axis].radixBlockCounts),
                histogram_bytes,view_alignment_,axis+3u,compute::BufferAccess::Write,
                count.biases[axis+3u],max_view_bytes_));
        }
        // Bind order must match the six source shader registrations:
        // keys X/Y/Z, then histogram X/Y/Z.
        std::array<compute::BufferBinding,6> sorted{
            counter.buffers[0],counter.buffers[2],counter.buffers[4],
            counter.buffers[1],counter.buffers[3],counter.buffers[5]};
        for(std::uint32_t slot=0;slot<6u;++slot)
            sorted[slot].slot=slot;
        counter.buffers.assign(sorted.begin(),sorted.end());
        counter.push_constants.resize(sizeof(count));
        std::memcpy(counter.push_constants.data(),&count,sizeof(count));
        staged.emplace_back(std::move(counter));

        ScatterParams scatter{};
        scatter.keys=keys;scatter.shift=shift;
        compute::DispatchCommand permute;
        permute.pipeline=scatter_pipeline_;
        permute.groups={kRadixBlocks,kSortAxes,1u};
        permute.buffers.reserve(15u);
        for(std::uint32_t axis=0u;axis<3u;++axis) {
            const auto& desc=axis_desc[axis];
            const std::array<CudaDeviceAddressMap::Address,5> pointers{
                addr(desc.inputKeys),addr(desc.inputRanks),
                addr(desc.outputKeys),addr(desc.outputRanks),
                addr(desc.radixBlockCounts)};
            for(std::uint32_t field=0u;field<5u;++field) {
                const auto slot=axis*5u+field;
                const auto length=field==4u?histogram_bytes:key_bytes;
                const auto access=field==2u||field==3u
                    ? compute::BufferAccess::Write:compute::BufferAccess::Read;
                permute.buffers.push_back(source_native_view(
                    addresses,pointers[field],length,view_alignment_,slot,
                    access,scatter.biases[slot],max_view_bytes_));
            }
        }
        permute.push_constants.resize(sizeof(scatter));
        std::memcpy(permute.push_constants.data(),&scatter,sizeof(scatter));
        staged.emplace_back(std::move(permute));
    }
}

void SourceSapRadixNativeStage::append_wr(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    std::span<const physx::PxgRadixSortDesc,6> radix,
    const CudaDeviceAddressMap& addresses,
    std::uint32_t previous_boxes) const {
    if(!previous_boxes)return;
    const auto keys=padded_keys(previous_boxes);
    std::vector<compute::BatchCommand> staged;
    staged.reserve(18u);
    RankParams init{};
    init.boxes=previous_boxes;
    compute::DispatchCommand initialize;
    initialize.pipeline=init_pipeline_;
    initialize.groups={kRankBlocks,1u,1u};
    initialize.buffers.reserve(3u);
    for(std::uint32_t axis=0u;axis<3u;++axis)
        initialize.buffers.push_back(source_native_view(
            addresses,addr(source.boxProjectionRanks[axis]),
            source_allocation_bytes(keys,4u),view_alignment_,axis,
            compute::BufferAccess::Write,init.biases[axis],max_view_bytes_));
    initialize.push_constants.resize(sizeof(init));
    std::memcpy(initialize.push_constants.data(),&init,sizeof(init));
    staged.emplace_back(std::move(initialize));
    append_sort(staged,radix,addresses,keys);
    GatherParams gather{};
    gather.boxes=previous_boxes;
    compute::DispatchCommand update;
    update.pipeline=gather_pipeline_;
    update.groups={kRankBlocks,1u,1u};
    update.buffers.reserve(9u);
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const std::array<CudaDeviceAddressMap::Address,3> pointers{
            addr(source.boxProjectionRanks[axis]),
            addr(source.boxHandles[1][axis]),
            addr(source.boxHandles[0][axis])};
        for(std::uint32_t field=0u;field<3u;++field) {
            const auto slot=axis+field*3u;
            update.buffers.push_back(source_native_view(
                addresses,pointers[field],source_allocation_bytes(keys,4u),
                view_alignment_,slot,field==2u?
                    compute::BufferAccess::Write:compute::BufferAccess::Read,
                gather.biases[slot],max_view_bytes_));
        }
    }
    std::array<compute::BufferBinding,9> sorted{
        update.buffers[0],update.buffers[3],update.buffers[6],
        update.buffers[1],update.buffers[4],update.buffers[7],
        update.buffers[2],update.buffers[5],update.buffers[8]};
    for(std::uint32_t i=0u;i<9u;++i)sorted[i].slot=i;
    update.buffers.assign(sorted.begin(),sorted.end());
    update.push_constants.resize(sizeof(gather));
    std::memcpy(update.push_constants.data(),&gather,sizeof(gather));
    staged.emplace_back(std::move(update));
    commit(commands,staged);
}
void SourceSapRadixNativeStage::append_wor(
    std::vector<compute::BatchCommand>& commands,
    std::span<const physx::PxgRadixSortDesc,6> radix,
    const CudaDeviceAddressMap& addresses,
    std::uint32_t previous_boxes,
    std::uint32_t created_boxes) const {
    if(!previous_boxes && !created_boxes)return;
    const auto keys=padded_keys(std::uint64_t(previous_boxes)+created_boxes);
    std::vector<compute::BatchCommand> staged;
    append_sort(staged,radix,addresses,keys);
    commit(commands,staged);
}
} // namespace fyzz::physx_gpu
