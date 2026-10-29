// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"
#include "physx_port/lowlevelaabb/include/BpVolumeData.h"
#include "physx_port/gpubroadphase/include/PxgAggregate.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {
static_assert(sizeof(physx::Bp::VolumeData)==16u);
static_assert(sizeof(physx::PxgAggregatePair)==12u);
static_assert(offsetof(physx::PxgAggregatePair,isNew)==8u);
static_assert(offsetof(physx::PxgAggregatePair,isDead)==9u);

// All 21 original source GPU allocations are borrowed; the AABB owner
// owns VolumeData, current/previous projections, pair slots, remove bits
// and the aggregate found/lost reports. No second pair identity map.
struct SourceAggregatePersistentCollisionNativeBuffers {
    CudaDeviceAddressMap::Address current_bounds=0;
    CudaDeviceAddressMap::Address previous_bounds=0;
    CudaDeviceAddressMap::Address volume_data=0;
    CudaDeviceAddressMap::Address aggregate_slots=0;
    CudaDeviceAddressMap::Address current_keys=0;
    CudaDeviceAddressMap::Address current_handles=0;
    CudaDeviceAddressMap::Address current_members=0;
    CudaDeviceAddressMap::Address current_masks=0;
    CudaDeviceAddressMap::Address previous_keys=0;
    CudaDeviceAddressMap::Address previous_handles=0;
    CudaDeviceAddressMap::Address previous_members=0;
    CudaDeviceAddressMap::Address previous_masks=0;
    CudaDeviceAddressMap::Address previous_sizes=0;
    CudaDeviceAddressMap::Address group_ids=0;
    CudaDeviceAddressMap::Address aggregate_pairs=0;
    CudaDeviceAddressMap::Address aggregate_pair_count=0;
    CudaDeviceAddressMap::Address removed_root_bitmap=0;
    CudaDeviceAddressMap::Address remove_pair_bitmap=0;
    CudaDeviceAddressMap::Address found_pairs=0;
    CudaDeviceAddressMap::Address lost_pairs=0;
    CudaDeviceAddressMap::Address found_lost_counts=0;
    std::uint32_t pair_capacity=0;
    std::uint32_t aggregate_capacity=0;
    std::uint32_t member_capacity=0;
    std::uint32_t bounds_capacity=0;
    std::uint32_t report_capacity=0;
};

class SourceAggregatePersistentCollisionNativeStage final {
public:
    SourceAggregatePersistentCollisionNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<128u ||
           gpu.max_workgroup_size[0]<32u ||
           gpu.max_workgroup_size[1]<4u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original persistent aggregate GPU pair launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=21u;
        pipeline.storage_buffer_views.assign(
            21u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=112u;
        pipeline.shader=std::move(shader);
        pipeline_=context.create_compute_pipeline(pipeline);
    }
    ~SourceAggregatePersistentCollisionNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }
    SourceAggregatePersistentCollisionNativeStage(
        const SourceAggregatePersistentCollisionNativeStage&)=delete;
    SourceAggregatePersistentCollisionNativeStage& operator=(
        const SourceAggregatePersistentCollisionNativeStage&)=delete;
    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePersistentCollisionNativeBuffers& src) const {
        if(!src.pair_capacity)return;
        if(!src.aggregate_capacity || !src.member_capacity ||
           !src.bounds_capacity || !src.report_capacity ||
           src.pair_capacity>std::numeric_limits<std::uint32_t>::max()/12u ||
           src.member_capacity>std::numeric_limits<std::uint32_t>::max()/8u ||
           src.bounds_capacity>std::numeric_limits<std::uint32_t>::max()/24u ||
           src.report_capacity>std::numeric_limits<std::uint32_t>::max()/8u ||
           src.aggregate_pairs==src.remove_pair_bitmap ||
           src.current_bounds==src.previous_bounds ||
           src.current_keys==src.previous_keys ||
           src.current_members==src.previous_members ||
           src.current_handles==src.previous_handles ||
           src.current_masks==src.previous_masks)
            throw std::invalid_argument(
                "Original persistent aggregate GPU pair histories or capacities invalid");
        const auto groups=std::min<std::uint64_t>(
            64u,(std::uint64_t(src.pair_capacity)+3u)/4u);
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original persistent aggregate pair grid exceeds GPU limit");
        std::array<std::uint32_t,28> params{
            src.pair_capacity,src.aggregate_capacity,
            src.member_capacity,src.bounds_capacity,src.report_capacity};
        const auto view=[&](std::uint32_t slot,
                            CudaDeviceAddressMap::Address address,
                            std::size_t bytes,
                            compute::BufferAccess access) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,params[5u+slot],max_view_);
        };
        const auto bounds=source_allocation_bytes(src.bounds_capacity,24u);
        const auto members=source_allocation_bytes(src.member_capacity,4u);
        const auto projection=source_allocation_bytes(src.member_capacity,8u);
        const auto reports=source_allocation_bytes(src.report_capacity,8u);
        const std::array<compute::BufferBinding,21> bindings{{
            view(0u,src.current_bounds,bounds,compute::BufferAccess::Read),
            view(1u,src.previous_bounds,bounds,compute::BufferAccess::Read),
            view(2u,src.volume_data,
                 source_allocation_bytes(src.bounds_capacity,16u),
                 compute::BufferAccess::Read),
            view(3u,src.aggregate_slots,
                 source_allocation_bytes(src.aggregate_capacity,16u),
                 compute::BufferAccess::Read),
            view(4u,src.current_keys,projection,compute::BufferAccess::Read),
            view(5u,src.current_handles,projection,compute::BufferAccess::Read),
            view(6u,src.current_members,members,compute::BufferAccess::Read),
            view(7u,src.current_masks,members,compute::BufferAccess::Read),
            view(8u,src.previous_keys,projection,compute::BufferAccess::Read),
            view(9u,src.previous_handles,projection,compute::BufferAccess::Read),
            view(10u,src.previous_members,members,compute::BufferAccess::Read),
            view(11u,src.previous_masks,members,compute::BufferAccess::Read),
            view(12u,src.previous_sizes,
                 source_allocation_bytes(src.aggregate_capacity,4u),
                 compute::BufferAccess::Read),
            view(13u,src.group_ids,
                 source_allocation_bytes(src.bounds_capacity,4u),
                 compute::BufferAccess::Read),
            view(14u,src.aggregate_pairs,
                 source_allocation_bytes(src.pair_capacity,12u),
                 compute::BufferAccess::ReadWrite),
            view(15u,src.aggregate_pair_count,4u,compute::BufferAccess::Read),
            view(16u,src.removed_root_bitmap,
                 source_allocation_bytes((std::uint64_t(src.bounds_capacity)+31u)/32u,4u),
                 compute::BufferAccess::Read),
            view(17u,src.remove_pair_bitmap,
                 source_allocation_bytes(src.pair_capacity,4u),
                 compute::BufferAccess::Write),
            view(18u,src.found_pairs,reports,compute::BufferAccess::ReadWrite),
            view(19u,src.lost_pairs,reports,compute::BufferAccess::ReadWrite),
            view(20u,src.found_lost_counts,8u,compute::BufferAccess::ReadWrite)
        }};
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
        dispatch.buffers.assign(bindings.begin(),bindings.end());
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(dispatch));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePersistentCollisionNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
