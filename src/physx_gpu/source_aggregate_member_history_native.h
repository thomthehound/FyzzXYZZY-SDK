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

// PxgAABBManager::updateBPSecondPass, aggregate.cu:markAddedAndDeletedAggregatedBounds.
// Device aliases belong to the original SAP integer-history buffers and AABB
// manager member-admission lists. No host overlap work or history copies.
struct SourceAggregateMemberHistoryNativeBuffers {
    CudaDeviceAddressMap::Address new_integer_bounds=0;
    CudaDeviceAddressMap::Address old_integer_bounds=0;
    CudaDeviceAddressMap::Address added_members=0;
    CudaDeviceAddressMap::Address removed_members=0;
    std::uint32_t bounds_capacity=0;
    std::uint32_t num_added=0;
    std::uint32_t num_removed=0;
};

class SourceAggregateMemberHistoryNativeStage final {
public:
    SourceAggregateMemberHistoryNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        : context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<256u ||
           gpu.max_workgroup_size[0]<128u ||
           gpu.max_workgroup_size[1]<2u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original aggregate added/deleted source GPU workgroup unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=4u;
        pipeline.storage_buffer_views.assign(
            4u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=32u;
        pipeline.shader=std::move(shader);
        pipeline_=context.create_compute_pipeline(pipeline);
    }
    SourceAggregateMemberHistoryNativeStage(
        const SourceAggregateMemberHistoryNativeStage&)=delete;
    SourceAggregateMemberHistoryNativeStage& operator=(
        const SourceAggregateMemberHistoryNativeStage&)=delete;
    ~SourceAggregateMemberHistoryNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateMemberHistoryNativeBuffers& src) const {
        const auto count=std::max(src.num_added,src.num_removed);
        if(!count)return;
        if(!src.bounds_capacity || !src.new_integer_bounds ||
           !src.old_integer_bounds ||
           (src.num_added && !src.added_members) ||
           (src.num_removed && !src.removed_members))
            throw std::invalid_argument(
                "Original aggregate member history has incomplete source buffers");
        const auto groups=(std::uint64_t(count)+127u)/128u;
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate history launch grid exceeds device");
        std::array<std::uint32_t,8> params{
            src.num_added,src.num_removed};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,std::uint32_t bias) {
            return source_native_view(addresses,address,bytes,alignment_,
                slot,access,params[bias],max_view_);
        };
        // A zero-count list is not dereferenced; bind the other live
        // list's first word so the native API never receives a null UAV.
        const auto list_address=src.num_added
            ? src.added_members:src.removed_members;
        const auto added=view(src.num_added?src.added_members:list_address,
            source_allocation_bytes(std::max(1u,src.num_added),4u),
            2u,compute::BufferAccess::Read,4u);
        const auto removed=view(src.num_removed?src.removed_members:list_address,
            source_allocation_bytes(std::max(1u,src.num_removed),4u),
            3u,compute::BufferAccess::Read,5u);
        const auto new_bounds=view(src.new_integer_bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            0u,compute::BufferAccess::ReadWrite,2u);
        const auto old_bounds=view(src.old_integer_bounds,
            source_allocation_bytes(src.bounds_capacity,24u),
            1u,compute::BufferAccess::ReadWrite,3u);
        params[6]=src.bounds_capacity;

        compute::DispatchCommand command{};
        command.pipeline=pipeline_;
        command.groups={static_cast<std::uint32_t>(groups),1u,1u};
        command.buffers={new_bounds,old_bounds,added,removed};
        command.push_constants.resize(sizeof(params));
        std::memcpy(command.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(command));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateMemberHistoryNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};

} // namespace fyzz::physx_gpu
