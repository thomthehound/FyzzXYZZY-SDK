// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Original PxgAABBManager::postBPStage2/doSelfCollision source GPU views.
// Every address belongs to the existing PhysX AABB/SAP source allocation.
// The two 32-bit pair counters are the original shared found/lost indices;
// values greater than report_capacity are overflow, never silent truncation.
struct SourceAggregateSelfCollisionNativeBuffers {
    CudaDeviceAddressMap::Address new_integer_bounds=0;
    CudaDeviceAddressMap::Address old_integer_bounds=0;
    CudaDeviceAddressMap::Address aggregates=0;
    CudaDeviceAddressMap::Address current_handles=0;
    CudaDeviceAddressMap::Address current_sap=0;
    CudaDeviceAddressMap::Address current_member_indices=0;
    CudaDeviceAddressMap::Address current_start_masks=0;
    CudaDeviceAddressMap::Address previous_handles=0;
    CudaDeviceAddressMap::Address previous_sap=0;
    CudaDeviceAddressMap::Address previous_member_indices=0;
    CudaDeviceAddressMap::Address previous_start_masks=0;
    CudaDeviceAddressMap::Address group_ids=0;
    CudaDeviceAddressMap::Address previous_sizes=0;
    CudaDeviceAddressMap::Address found_pairs=0;
    CudaDeviceAddressMap::Address lost_pairs=0;
    CudaDeviceAddressMap::Address found_lost_counts=0;
    std::uint32_t aggregate_count=0;
    std::uint32_t aggregate_capacity=0;
    std::uint32_t member_capacity=0;
    std::uint32_t bounds_capacity=0;
    std::uint32_t report_capacity=0;
};

class SourceAggregateSelfCollisionNativeStage final {
public:
    SourceAggregateSelfCollisionNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<128u ||
           gpu.max_workgroup_size[0]<32u ||
           gpu.max_workgroup_size[1]<4u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Source aggregate self-collision four-warp GPU launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=16u;
        pipeline.storage_buffer_views.assign(
            16u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=80u;
        pipeline.shader=std::move(shader);
        pipeline_=context.create_compute_pipeline(pipeline);
    }
    SourceAggregateSelfCollisionNativeStage(
        const SourceAggregateSelfCollisionNativeStage&)=delete;
    SourceAggregateSelfCollisionNativeStage& operator=(
        const SourceAggregateSelfCollisionNativeStage&)=delete;
    ~SourceAggregateSelfCollisionNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateSelfCollisionNativeBuffers& src) const {
        if(!src.aggregate_count)return;
        if(src.aggregate_count>src.aggregate_capacity ||
           !src.member_capacity || !src.bounds_capacity ||
           !src.report_capacity ||
           src.member_capacity>std::numeric_limits<std::uint32_t>::max()/8u ||
           src.bounds_capacity>std::numeric_limits<std::uint32_t>::max()/24u ||
           src.report_capacity>std::numeric_limits<std::uint32_t>::max()/8u)
            throw std::invalid_argument(
                "Original aggregate GPU self-collision capacity invalid");
        if(src.new_integer_bounds==src.old_integer_bounds ||
           src.current_handles==src.previous_handles ||
           src.current_member_indices==src.previous_member_indices ||
           src.current_sap==src.previous_sap ||
           src.current_start_masks==src.previous_start_masks ||
           src.found_pairs==src.lost_pairs)
            throw std::invalid_argument(
                "Original aggregate GPU current/previous or reports are aliased");
        const auto groups=(std::uint64_t(src.aggregate_count)+3u)/4u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate self-collision grid exceeds GPU limit");
        std::array<std::uint32_t,20> params{
            src.aggregate_count,src.member_capacity,
            src.bounds_capacity,src.report_capacity};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t offset_index) {
            return source_native_view(addresses,address,bytes,alignment_,
                slot,access,params[offset_index],max_view_);
        };
        auto current=view(src.new_integer_bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            0u,compute::BufferAccess::Read,4u);
        auto previous=view(src.old_integer_bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            1u,compute::BufferAccess::Read,5u);
        auto aggregates=view(src.aggregates,
            source_allocation_bytes(src.aggregate_capacity,16u),
            2u,compute::BufferAccess::Read,6u);
        auto currentHandles=view(src.current_handles,
            source_allocation_bytes(src.member_capacity,8u),
            3u,compute::BufferAccess::Read,7u);
        auto currentSap=view(src.current_sap,
            source_allocation_bytes(src.member_capacity,8u),
            4u,compute::BufferAccess::Read,8u);
        auto currentMembers=view(src.current_member_indices,
            source_allocation_bytes(src.member_capacity,4u),
            5u,compute::BufferAccess::Read,9u);
        auto currentMasks=view(src.current_start_masks,
            source_allocation_bytes(src.member_capacity,4u),
            6u,compute::BufferAccess::Read,10u);
        auto previousHandles=view(src.previous_handles,
            source_allocation_bytes(src.member_capacity,8u),
            7u,compute::BufferAccess::Read,11u);
        auto previousSap=view(src.previous_sap,
            source_allocation_bytes(src.member_capacity,8u),
            8u,compute::BufferAccess::Read,12u);
        auto previousMembers=view(src.previous_member_indices,
            source_allocation_bytes(src.member_capacity,4u),
            9u,compute::BufferAccess::Read,13u);
        auto previousMasks=view(src.previous_start_masks,
            source_allocation_bytes(src.member_capacity,4u),
            10u,compute::BufferAccess::Read,14u);
        auto groupsView=view(src.group_ids,
            source_allocation_bytes(src.bounds_capacity,4u),
            11u,compute::BufferAccess::Read,15u);
        auto previousSizes=view(src.previous_sizes,
            source_allocation_bytes(src.aggregate_capacity,4u),
            12u,compute::BufferAccess::ReadWrite,16u);
        auto found=view(src.found_pairs,
            source_allocation_bytes(src.report_capacity,8u),
            13u,compute::BufferAccess::Write,17u);
        auto lost=view(src.lost_pairs,
            source_allocation_bytes(src.report_capacity,8u),
            14u,compute::BufferAccess::Write,18u);
        auto counters=view(src.found_lost_counts,8u,
            15u,compute::BufferAccess::ReadWrite,19u);

        compute::DispatchCommand command{};
        command.pipeline=pipeline_;
        command.groups={static_cast<std::uint32_t>(groups),1u,1u};
        command.buffers={current,previous,aggregates,currentHandles,
            currentSap,currentMembers,currentMasks,previousHandles,
            previousSap,previousMembers,previousMasks,groupsView,
            previousSizes,found,lost,counters};
        command.push_constants.resize(sizeof(params));
        std::memcpy(command.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(command));
    }

    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateSelfCollisionNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }

private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
