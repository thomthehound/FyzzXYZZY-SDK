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
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Original PxgAABBManager::computeAggregateBounds/markAggregateBoundsBitmap
// GPU resources. Every pointer borrows an allocation owned by the source
// AABB manager; the descriptor records are pointer-lowered PxgAggregate
// entries, not an independent aggregate registry or shape history.
struct SourceAggregateGpuNativeBuffers {
    CudaDeviceAddressMap::Address bounds=0;
    CudaDeviceAddressMap::Address contact_distances=0;
    CudaDeviceAddressMap::Address aggregate_slots=0;
    CudaDeviceAddressMap::Address member_indices=0;
    CudaDeviceAddressMap::Address changed_bitmap=0;
    std::uint32_t bounds_capacity=0;
    std::uint32_t aggregate_slots_capacity=0;
    std::uint32_t member_indices_capacity=0;
    std::uint32_t changed_bitmap_words=0;
    std::uint32_t aggregate_count=0;
};

// PxgAggregate::mIndex, index into flattened original
// updateBoundIndices, size and reserved. isValid() remains exactly
// size > 0 && mIndex != PX_INVALID_U32.
struct SourceAggregateGpuNativeRecord {
    std::uint32_t root;
    std::uint32_t first_member;
    std::uint32_t size;
    std::uint32_t reserved;
};
static_assert(sizeof(SourceAggregateGpuNativeRecord)==16u);

class SourceAggregateGpuNativeStage final {
public:
    SourceAggregateGpuNativeStage(
        compute::Context& context,compute::ShaderBinary bounds,
        compute::ShaderBinary changed) : context_(context) {
        const auto& gpu=context_.device_info();
        if(gpu.max_workgroup_invocations<128u ||
           gpu.max_workgroup_size[0]<32u ||
           gpu.max_workgroup_size[1]<4u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original aggregate source GPU four-warp launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=4u;
        pipeline.storage_buffer_views.assign(
            4u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=32u;
        pipeline.shader=std::move(bounds);
        bounds_=context_.create_compute_pipeline(pipeline);
        try {
            pipeline.storage_buffer_binding_count=3u;
            pipeline.storage_buffer_views.assign(
                3u,compute::StorageBufferViewKind::RawReadWrite);
            pipeline.shader=std::move(changed);
            changed_=context_.create_compute_pipeline(pipeline);
        } catch(...) {
            context_.destroy_compute_pipeline(bounds_);
            bounds_=0;
            throw;
        }
    }
    SourceAggregateGpuNativeStage(const SourceAggregateGpuNativeStage&)=delete;
    SourceAggregateGpuNativeStage& operator=(
        const SourceAggregateGpuNativeStage&)=delete;
    ~SourceAggregateGpuNativeStage() {
        if(changed_)context_.destroy_compute_pipeline(changed_);
        if(bounds_)context_.destroy_compute_pipeline(bounds_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateGpuNativeBuffers& src) const {
        if(!src.aggregate_count)return;
        if(src.aggregate_count>src.aggregate_slots_capacity ||
           !src.bounds_capacity || !src.member_indices_capacity ||
           src.changed_bitmap_words<(std::uint64_t(src.bounds_capacity)+31u)/32u)
            throw std::invalid_argument(
                "Source aggregate GPU member/root capacities invalid");
        const auto groups=(std::uint64_t(src.aggregate_count)+3u)/4u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source aggregate GPU workgroup count exceeded");

        // All five original source buffers are validated before either
        // dispatch becomes visible to the caller's mStream.
        std::array<std::uint32_t,8> params{src.aggregate_count};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t offset_index) {
            return source_native_view(
                addresses,address,bytes,alignment_,slot,access,
                params[offset_index],max_view_);
        };
        const auto bounds=view(src.bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            0u,compute::BufferAccess::ReadWrite,1u);
        const auto distances=view(src.contact_distances,
            source_allocation_bytes(src.bounds_capacity,4u),
            1u,compute::BufferAccess::Read,2u);
        const auto aggregates=view(src.aggregate_slots,
            source_allocation_bytes(src.aggregate_slots_capacity,16u),
            2u,compute::BufferAccess::Read,3u);
        const auto members=view(src.member_indices,
            source_allocation_bytes(src.member_indices_capacity,4u),
            3u,compute::BufferAccess::Read,4u);
        const auto bitmap=view(src.changed_bitmap,
            source_allocation_bytes(src.changed_bitmap_words,4u),
            2u,compute::BufferAccess::ReadWrite,5u);

        std::array<compute::DispatchCommand,2> staged{};
        staged[0].pipeline=bounds_;
        staged[0].groups={static_cast<std::uint32_t>(groups),1u,1u};
        staged[0].buffers={bounds,distances,aggregates,members};
        staged[1].pipeline=changed_;
        staged[1].groups=staged[0].groups;
        // Changed-map shader registers are aggregate, member, bitmap.
        auto aggregate_view=aggregates;
        auto member_view=members;
        aggregate_view.slot=0u;
        member_view.slot=1u;
        staged[1].buffers={aggregate_view,member_view,bitmap};
        for(auto& command:staged) {
            command.push_constants.resize(sizeof(params));
            std::memcpy(command.push_constants.data(),
                        params.data(),sizeof(params));
        }
        commands.reserve(commands.size()+staged.size());
        for(auto& command:staged)
            commands.emplace_back(std::move(command));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateGpuNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }
private:
    compute::Context& context_;
    compute::PipelineId bounds_=0,changed_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
