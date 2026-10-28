// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_native_view.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhasePairReport.h"

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

using Address = CudaDeviceAddressMap::Address;

template<class T> Address address(T* p) {
    return static_cast<Address>(reinterpret_cast<std::uintptr_t>(p));
}

struct alignas(16) Parameters {
    std::uint32_t bias[13] = {};
    std::uint32_t raw_capacity = 0;
    std::uint32_t agg_capacity = 0;
    std::uint32_t volume_capacity = 0;
    std::uint32_t volume_stride = 0;
    std::uint32_t aggregate_offset = 0;
    std::uint32_t has_volumes = 0;
    std::uint32_t reserved[13] = {};
};
static_assert(sizeof(Parameters) == 128u,
              "Source multi-root report root constants changed");

struct Resource {
    Address pointer = 0;
    std::size_t bytes = 0;
    compute::BufferAccess access = compute::BufferAccess::Read;
};

} // namespace

SourceSapReportNativeStage::SourceSapReportNativeStage(
    compute::Context& context,
    compute::ShaderBinary first_portable,
    compute::ShaderBinary second_portable,
    compute::ShaderBinary copy_shader,
    compute::ShaderBinary first_wave,
    compute::ShaderBinary second_wave,
    bool force_portable) : context_(context) {
    const auto& gpu = context_.device_info();
    if (gpu.max_workgroup_invocations < 256u ||
        gpu.max_workgroup_size[0] < 256u ||
        gpu.max_workgroup_count[0] < 64u ||
        gpu.max_workgroup_count[1] < 4u)
        throw std::runtime_error("Original PhysX report grid unavailable");
    using_wave_ = !force_portable && gpu.supports_wave_arithmetic &&
                  gpu.subgroup_size >= 4u && 256u % gpu.subgroup_size == 0u;
    compute::ComputePipelineDesc desc;
    desc.push_constant_bytes = sizeof(Parameters);
    try {
        desc.storage_buffer_binding_count = 8u;
        desc.storage_buffer_views.assign(8u,
            compute::StorageBufferViewKind::RawReadWrite);
        desc.shader = using_wave_ ? std::move(first_wave) :
                                    std::move(first_portable);
        first_ = context_.create_compute_pipeline(desc);

        desc.storage_buffer_binding_count = 13u;
        desc.storage_buffer_views.assign(13u,
            compute::StorageBufferViewKind::RawReadWrite);
        desc.shader = using_wave_ ? std::move(second_wave) :
                                    std::move(second_portable);
        second_ = context_.create_compute_pipeline(desc);

        desc.storage_buffer_binding_count = 7u;
        desc.storage_buffer_views.assign(7u,
            compute::StorageBufferViewKind::RawReadWrite);
        desc.shader = std::move(copy_shader);
        copy_ = context_.create_compute_pipeline(desc);
    } catch (...) {
        if (copy_) context_.destroy_compute_pipeline(copy_);
        if (second_) context_.destroy_compute_pipeline(second_);
        if (first_) context_.destroy_compute_pipeline(first_);
        throw;
    }
}

SourceSapReportNativeStage::~SourceSapReportNativeStage() {
    if (copy_) context_.destroy_compute_pipeline(copy_);
    if (second_) context_.destroy_compute_pipeline(second_);
    if (first_) context_.destroy_compute_pipeline(first_);
}

void SourceSapReportNativeStage::append(
    std::vector<compute::BatchCommand>& commands,
    const physx::PxgBroadPhaseDesc& source,
    const CudaDeviceAddressMap& addresses,
    SourceSapReportNativeLayout layout) const {
    if (!layout.descriptor || !source.max_found_lost_pairs ||
        !source.max_found_lost_agg_pairs)
        throw std::invalid_argument("PhysX GPU report descriptor/capacities missing");
    if (source.aabbMngr_volumeData &&
        (!layout.volume_capacity || layout.volume_stride % 4u ||
         !layout.volume_stride || layout.aggregate_member_offset % 4u ||
         layout.aggregate_member_offset > layout.volume_stride ||
         4u > layout.volume_stride - layout.aggregate_member_offset))
        throw std::invalid_argument("Invalid native PhysX VolumeData layout");

    constexpr std::size_t rawField =
        offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex);
    constexpr std::size_t aggField =
        offsetof(physx::PxgBroadPhaseDesc, sharedFoundAggPairIndex);
    constexpr std::size_t overflowField =
        offsetof(physx::PxgBroadPhaseDesc, found_lost_pairs_overflow_flags);
    static_assert(offsetof(physx::PxgBroadPhaseDesc, sharedLostPairIndex) ==
                      rawField + 4u &&
                  offsetof(physx::PxgBroadPhaseDesc, sharedLostAggPairIndex) ==
                      aggField + 4u, "PhysX GPU report source counter ABI changed");
    static_assert(overflowField % 4u == 0u &&
                  overflowField + 4u <= sizeof(physx::PxgBroadPhaseDesc),
                  "PhysX GPU report overflow scalar ABI changed");

    const auto pairs = source_allocation_bytes(
        source.max_found_lost_pairs, sizeof(physx::PxgBroadPhasePair));
    const auto aggPairs = source_allocation_bytes(
        source.max_found_lost_agg_pairs, sizeof(physx::PxgBroadPhasePair));
    const auto vol = source.aabbMngr_volumeData
        ? source_allocation_bytes(layout.volume_capacity, layout.volume_stride)
        : 8u;
    const auto& gpu = context_.device_info();
    const auto align = static_cast<std::size_t>(
        std::max<std::uint64_t>(4u, gpu.storage_buffer_offset_alignment));
    if (!gpu.max_storage_buffer_range || align % 4u)
        throw std::runtime_error("Native PhysX storage buffer view limits missing");

    Parameters params{};
    params.raw_capacity = source.max_found_lost_pairs;
    params.agg_capacity = source.max_found_lost_agg_pairs;
    params.volume_capacity = layout.volume_capacity;
    params.volume_stride = layout.volume_stride;
    params.aggregate_offset = layout.aggregate_member_offset;
    params.has_volumes = source.aabbMngr_volumeData ? 1u : 0u;

    const Resource raw = {layout.descriptor + rawField, 8u};
    const Resource aggregate = {layout.descriptor + aggField, 8u};
    const Resource rawFound = {address(source.foundPairReport), pairs};
    const Resource rawLost = {address(source.lostPairReport), pairs};
    const Resource volume = {source.aabbMngr_volumeData
        ? address(source.aabbMngr_volumeData) : raw.pointer, vol};
    const std::array<Resource, 4> blocks{{
        {address(source.aggReportBlock[0]), 32u*4u},
        {address(source.actorReportBlock[0]), 32u*4u},
        {address(source.aggReportBlock[1]), 32u*4u},
        {address(source.actorReportBlock[1]), 32u*4u}
    }};
    const Resource foundAggregate = {address(source.foundAggPairReport), aggPairs};
    const Resource foundActor = {address(source.foundActorPairReport), pairs};
    const Resource lostAggregate = {address(source.lostAggPairReport), aggPairs};
    const Resource lostActor = {address(source.lostActorPairReport), pairs};
    const Resource mapFound = {address(source.foundPairReportMap), pairs};
    const Resource mapLost = {address(source.lostPairReportMap), pairs};
    const Resource overflow = {layout.descriptor + overflowField, 4u};

    const std::array<std::vector<Resource>,3> passes{{
        {raw,rawFound,rawLost,volume,
         {blocks[0].pointer,blocks[0].bytes,compute::BufferAccess::Write},
         {blocks[1].pointer,blocks[1].bytes,compute::BufferAccess::Write},
         {blocks[2].pointer,blocks[2].bytes,compute::BufferAccess::Write},
         {blocks[3].pointer,blocks[3].bytes,compute::BufferAccess::Write}},
        {raw,{aggregate.pointer,aggregate.bytes,compute::BufferAccess::Write},
         rawFound,rawLost,volume,blocks[0],blocks[1],blocks[2],blocks[3],
         {foundAggregate.pointer,foundAggregate.bytes,compute::BufferAccess::Write},
         {foundActor.pointer,foundActor.bytes,compute::BufferAccess::Write},
         {lostAggregate.pointer,lostAggregate.bytes,compute::BufferAccess::Write},
         {lostActor.pointer,lostActor.bytes,compute::BufferAccess::Write}},
        {raw,aggregate,foundActor,lostActor,
         {mapFound.pointer,mapFound.bytes,compute::BufferAccess::Write},
         {mapLost.pointer,mapLost.bytes,compute::BufferAccess::Write},
         {overflow.pointer,overflow.bytes,compute::BufferAccess::Write}}
    }};

    std::vector<compute::BatchCommand> pending;
    pending.reserve(3u);
    for (std::uint32_t pass=0u;pass<3u;++pass) {
        auto constants = params;
        compute::DispatchCommand cmd;
        cmd.pipeline = pass==0u ? first_ : pass==1u ? second_ : copy_;
        cmd.groups = pass==2u ? compute::DispatchSize{64u,1u,1u} :
                                compute::DispatchSize{32u,4u,1u};
        cmd.buffers.reserve(passes[pass].size());
        for (std::uint32_t i=0u;i<passes[pass].size();++i) {
            const auto& item=passes[pass][i];
            cmd.buffers.emplace_back(source_native_view(
                addresses,item.pointer,item.bytes,align,i,item.access,
                constants.bias[i],gpu.max_storage_buffer_range));
        }
        cmd.push_constants.resize(sizeof(constants));
        std::memcpy(cmd.push_constants.data(),&constants,sizeof(constants));
        pending.emplace_back(std::move(cmd));
    }
    commands.reserve(commands.size()+pending.size());
    for (auto& cmd:pending)commands.emplace_back(std::move(cmd));
}

} // namespace fyzz::physx_gpu
