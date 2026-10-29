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

// PxgAggregate double-buffered sortedProjections, sortedHandles, sapBox1D,
// startMasks and boundIndices, lowered to source-heap-backed reserved slices.
// Each current/previous slot contains exactly the original per-aggregate
// arrays. The caller owns slot rotation at the successful report fetch.
struct SourceAggregateProjectionNativeBuffers {
    CudaDeviceAddressMap::Address integer_bounds=0;
    CudaDeviceAddressMap::Address aggregate_slots=0;
    CudaDeviceAddressMap::Address source_members=0;
    CudaDeviceAddressMap::Address sorted_projections=0;
    CudaDeviceAddressMap::Address sorted_handles=0;
    CudaDeviceAddressMap::Address scratch_projections=0;
    CudaDeviceAddressMap::Address scratch_handles=0;
    CudaDeviceAddressMap::Address sap_boxes=0;
    CudaDeviceAddressMap::Address start_masks=0;
    CudaDeviceAddressMap::Address bound_index_history=0;
    std::uint32_t bounds_capacity=0;
    std::uint32_t aggregate_slots_capacity=0;
    std::uint32_t member_capacity=0;
    std::uint32_t aggregate_count=0;
    std::uint32_t current_slot=0;
};

class SourceAggregateProjectionNativeStage final {
public:
    SourceAggregateProjectionNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<128u ||
           gpu.max_workgroup_size[0]<32u ||
           gpu.max_workgroup_size[1]<4u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original aggregate projection sort four-warp launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=10u;
        pipeline.storage_buffer_views.assign(
            10u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=64u;
        pipeline.shader=std::move(shader);
        pipeline_=context.create_compute_pipeline(pipeline);
    }
    SourceAggregateProjectionNativeStage(
        const SourceAggregateProjectionNativeStage&)=delete;
    SourceAggregateProjectionNativeStage& operator=(
        const SourceAggregateProjectionNativeStage&)=delete;
    ~SourceAggregateProjectionNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateProjectionNativeBuffers& src) const {
        if(!src.aggregate_count)return;
        if(src.aggregate_count>src.aggregate_slots_capacity ||
           !src.member_capacity || !src.bounds_capacity ||
           src.member_capacity>std::numeric_limits<std::uint32_t>::max()/16u ||
           src.current_slot>1u)
            throw std::invalid_argument(
                "Original aggregate projection history has invalid capacities or slot");
        const auto groups=(std::uint64_t(src.aggregate_count)+3u)/4u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate projection launch exceeds GPU grid");
        std::array<std::uint32_t,16> params{
            src.aggregate_count,src.member_capacity,
            src.bounds_capacity,src.current_slot};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t bias) {
            return source_native_view(addresses,address,bytes,alignment_,
                slot,access,params[bias],max_view_);
        };
        const auto bounds=view(src.integer_bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            0u,compute::BufferAccess::Read,4u);
        const auto aggregates=view(src.aggregate_slots,
            source_allocation_bytes(src.aggregate_slots_capacity,16u),
            1u,compute::BufferAccess::Read,5u);
        const auto source_members=view(src.source_members,
            source_allocation_bytes(src.member_capacity,4u),
            2u,compute::BufferAccess::Read,6u);
        const auto projections=view(src.sorted_projections,
            source_allocation_bytes(src.member_capacity,16u),
            3u,compute::BufferAccess::ReadWrite,7u);
        const auto handles=view(src.sorted_handles,
            source_allocation_bytes(src.member_capacity,16u),
            4u,compute::BufferAccess::ReadWrite,8u);
        const auto scratch_projections=view(src.scratch_projections,
            source_allocation_bytes(src.member_capacity,8u),
            5u,compute::BufferAccess::ReadWrite,9u);
        const auto scratch_handles=view(src.scratch_handles,
            source_allocation_bytes(src.member_capacity,8u),
            6u,compute::BufferAccess::ReadWrite,10u);
        const auto sap_boxes=view(src.sap_boxes,
            source_allocation_bytes(src.member_capacity,16u),
            7u,compute::BufferAccess::Write,11u);
        const auto start_masks=view(src.start_masks,
            source_allocation_bytes(src.member_capacity,8u),
            8u,compute::BufferAccess::Write,12u);
        const auto history=view(src.bound_index_history,
            source_allocation_bytes(src.member_capacity,8u),
            9u,compute::BufferAccess::Write,13u);

        compute::DispatchCommand command{};
        command.pipeline=pipeline_;
        command.groups={static_cast<std::uint32_t>(groups),1u,1u};
        command.buffers={bounds,aggregates,source_members,
            projections,handles,scratch_projections,scratch_handles,
            sap_boxes,start_masks,history};
        command.push_constants.resize(sizeof(params));
        std::memcpy(command.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(command));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateProjectionNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }

private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};

} // namespace fyzz::physx_gpu
