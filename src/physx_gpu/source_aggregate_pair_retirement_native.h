// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// PxgAggregateDesc::removeBitmap/removeHistogram/nbRemoved borrow the
// same original AABB owner's persistent PxgAggregatePair and free-ID heap.
struct SourceAggregatePairRetirementNativeBuffers {
    CudaDeviceAddressMap::Address aggregate_pairs=0;
    CudaDeviceAddressMap::Address remove_bitmap=0;
    CudaDeviceAddressMap::Address remove_histogram=0;
    CudaDeviceAddressMap::Address free_indices=0;
    CudaDeviceAddressMap::Address free_list=0;
    CudaDeviceAddressMap::Address aggregate_pair_count=0;
    CudaDeviceAddressMap::Address removed_count=0;
    std::uint32_t aggregate_pair_capacity=0;
};

class SourceAggregatePairRetirementNativeStage final {
public:
    SourceAggregatePairRetirementNativeStage(
        compute::Context& context,
        compute::ShaderBinary histogram_shader,
        compute::ShaderBinary mark_shader,
        compute::ShaderBinary publish_shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<256u ||
           gpu.max_workgroup_size[0]<256u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original aggregate pair retirement GPU launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc descriptor{};
        descriptor.storage_buffer_binding_count=7u;
        descriptor.storage_buffer_views.assign(
            7u,compute::StorageBufferViewKind::RawReadWrite);
        descriptor.push_constant_bytes=48u;
        // SPIR-V preserves the three source entrypoint names; DXIL
        // already contains the selected entry and the native D3D12
        // pipeline contract requires its default "main".
        if(histogram_shader.format==compute::ShaderFormat::Spirv) {
            histogram_shader.entry_point="histogram";
            mark_shader.entry_point="mark";
            publish_shader.entry_point="publish";
        }
        try {
            descriptor.shader=std::move(histogram_shader);
            histogram_=context.create_compute_pipeline(descriptor);
            descriptor.shader=std::move(mark_shader);
            mark_=context.create_compute_pipeline(descriptor);
            descriptor.shader=std::move(publish_shader);
            publish_=context.create_compute_pipeline(descriptor);
        }catch(...) {
            if(publish_)context_.destroy_compute_pipeline(publish_);
            if(mark_)context_.destroy_compute_pipeline(mark_);
            if(histogram_)context_.destroy_compute_pipeline(histogram_);
            throw;
        }
    }
    ~SourceAggregatePairRetirementNativeStage() {
        if(publish_)context_.destroy_compute_pipeline(publish_);
        if(mark_)context_.destroy_compute_pipeline(mark_);
        if(histogram_)context_.destroy_compute_pipeline(histogram_);
    }
    SourceAggregatePairRetirementNativeStage(
        const SourceAggregatePairRetirementNativeStage&)=delete;
    SourceAggregatePairRetirementNativeStage& operator=(
        const SourceAggregatePairRetirementNativeStage&)=delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePairRetirementNativeBuffers& src) const {
        if(!src.aggregate_pair_capacity ||
           src.aggregate_pair_capacity>
               std::numeric_limits<std::uint32_t>::max()/12u ||
           src.aggregate_pairs==src.remove_bitmap ||
           src.remove_bitmap==src.remove_histogram ||
           src.free_list==src.aggregate_pair_count ||
           src.free_indices==src.aggregate_pairs)
            throw std::invalid_argument(
                "Original aggregate pair-retirement GPU buffer aliases or capacity invalid");
        const auto chunks=(std::uint64_t(src.aggregate_pair_capacity)+255u)/256u;
        const auto groups=std::min<std::uint64_t>(64u,chunks);
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate pair retirement GPU grid unavailable");

        std::array<std::uint32_t,12> params{
            src.aggregate_pair_capacity,static_cast<std::uint32_t>(groups)};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t bias_index) {
            return source_native_view(addresses,address,bytes,
                alignment_,slot,access,params[bias_index],max_view_);
        };
        const auto pairs=view(src.aggregate_pairs,
            source_allocation_bytes(src.aggregate_pair_capacity,12u),
            0u,compute::BufferAccess::ReadWrite,2u);
        const auto bitmap=view(src.remove_bitmap,
            source_allocation_bytes(src.aggregate_pair_capacity,4u),
            1u,compute::BufferAccess::Read,3u);
        const auto histogram=view(src.remove_histogram,
            source_allocation_bytes(chunks,4u),
            2u,compute::BufferAccess::ReadWrite,4u);
        const auto indices=view(src.free_indices,
            source_allocation_bytes(src.aggregate_pair_capacity,4u),
            3u,compute::BufferAccess::Write,5u);
        const auto list=view(src.free_list,8u,
            4u,compute::BufferAccess::ReadWrite,6u);
        const auto count=view(src.aggregate_pair_count,4u,
            5u,compute::BufferAccess::Read,7u);
        const auto removed=view(src.removed_count,4u,
            6u,compute::BufferAccess::ReadWrite,8u);
        const std::array<compute::BufferBinding,7> bindings{
            pairs,bitmap,histogram,indices,list,count,removed};
        const auto append_one=[&](compute::PipelineId pipeline,
                                  std::uint32_t dispatch_groups) {
            compute::DispatchCommand command{};
            command.pipeline=pipeline;
            command.groups={dispatch_groups,1u,1u};
            command.buffers.assign(bindings.begin(),bindings.end());
            command.push_constants.resize(sizeof(params));
            std::memcpy(command.push_constants.data(),
                        params.data(),sizeof(params));
            commands.emplace_back(std::move(command));
        };
        commands.reserve(commands.size()+3u);
        append_one(histogram_,static_cast<std::uint32_t>(groups));
        append_one(mark_,static_cast<std::uint32_t>(groups));
        append_one(publish_,1u);
    }

    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePairRetirementNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }

private:
    compute::Context& context_;
    compute::PipelineId histogram_=0,mark_=0,publish_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
