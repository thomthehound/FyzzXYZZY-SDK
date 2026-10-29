// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhaseDesc.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// PxgAggregateDesc::aggPairs, freeIndices, freeBufferList, aggPairCount,
// and aggregate-pair overflow counters are borrowed from the original
// source AABB GPU owner; the SAP descriptor supplies its authoritative
// sharedFoundAggPairIndex on the GPU, never a CPU-fetched approximation.
struct SourceAggregatePairAdmissionNativeBuffers {
    CudaDeviceAddressMap::Address broadphase_descriptor=0;
    CudaDeviceAddressMap::Address found_aggregate_pairs=0;
    CudaDeviceAddressMap::Address aggregate_pairs=0;
    CudaDeviceAddressMap::Address free_indices=0;
    CudaDeviceAddressMap::Address free_list=0;
    CudaDeviceAddressMap::Address aggregate_pair_count=0;
    CudaDeviceAddressMap::Address overflow_reports=0;
    std::uint32_t found_root_report_capacity=0;
    std::uint32_t aggregate_pair_capacity=0;
};

class SourceAggregatePairAdmissionNativeStage final {
public:
    SourceAggregatePairAdmissionNativeStage(
        compute::Context& context,
        compute::ShaderBinary admit_shader,
        compute::ShaderBinary publish_shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<256u ||
           gpu.max_workgroup_size[0]<256u ||
           !gpu.max_workgroup_count[0] ||
           !gpu.max_storage_buffer_range)
            throw std::runtime_error(
                "Original source aggregate pair GPU admission launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc descriptor{};
        descriptor.storage_buffer_binding_count=7u;
        descriptor.storage_buffer_views.assign(
            7u,compute::StorageBufferViewKind::RawReadWrite);
        descriptor.push_constant_bytes=48u;
        // Both binaries are compiled from named HLSL entrypoints. Vulkan
        // must request the same SPIR-V entrypoint, not ShaderBinary's
        // default "main"; DXIL already contains its selected entry.
        admit_shader.entry_point="admit";
        publish_shader.entry_point="publish";
        descriptor.shader=std::move(admit_shader);
        try {
            admit_=context.create_compute_pipeline(descriptor);
        }catch(const std::exception& error) {
            throw std::runtime_error(
                std::string("Source aggregate pair admit GPU pipeline: ")+
                error.what());
        }
        try {
            descriptor.shader=std::move(publish_shader);
            publish_=context.create_compute_pipeline(descriptor);
        }catch(const std::exception& error) {
            context.destroy_compute_pipeline(admit_);
            admit_=0;
            throw std::runtime_error(
                std::string("Source aggregate pair publish GPU pipeline: ")+
                error.what());
        }
    }
    ~SourceAggregatePairAdmissionNativeStage() {
        if(publish_)context_.destroy_compute_pipeline(publish_);
        if(admit_)context_.destroy_compute_pipeline(admit_);
    }
    SourceAggregatePairAdmissionNativeStage(
        const SourceAggregatePairAdmissionNativeStage&)=delete;
    SourceAggregatePairAdmissionNativeStage& operator=(
        const SourceAggregatePairAdmissionNativeStage&)=delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePairAdmissionNativeBuffers& src) const {
        if(!src.found_root_report_capacity ||
           !src.aggregate_pair_capacity ||
           src.aggregate_pair_capacity>
                std::numeric_limits<std::uint32_t>::max()/12u ||
           src.found_root_report_capacity>
                std::numeric_limits<std::uint32_t>::max()/8u ||
           src.found_aggregate_pairs==src.aggregate_pairs)
            throw std::invalid_argument(
                "Original source aggregate root pair GPU capacities or aliases invalid");
        const auto blocks=std::min<std::uint64_t>(
            64u,(std::uint64_t(src.found_root_report_capacity)+255u)/256u);
        if(blocks>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate pair admission exceeds GPU grid limit");
        std::array<std::uint32_t,12> params{
            src.found_root_report_capacity,src.aggregate_pair_capacity,
            static_cast<std::uint32_t>(
                offsetof(physx::PxgBroadPhaseDesc,sharedFoundAggPairIndex)),
            static_cast<std::uint32_t>(blocks)};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t index) {
            return source_native_view(addresses,address,bytes,alignment_,
                slot,access,params[index],max_view_);
        };
        const auto descriptor=view(src.broadphase_descriptor,
            sizeof(physx::PxgBroadPhaseDesc),
            0u,compute::BufferAccess::Read,4u);
        const auto root_pairs=view(src.found_aggregate_pairs,
            source_allocation_bytes(src.found_root_report_capacity,8u),
            1u,compute::BufferAccess::Read,5u);
        const auto pair_slots=view(src.aggregate_pairs,
            source_allocation_bytes(src.aggregate_pair_capacity,12u),
            2u,compute::BufferAccess::Write,6u);
        const auto free_indices=view(src.free_indices,
            source_allocation_bytes(src.aggregate_pair_capacity,4u),
            3u,compute::BufferAccess::Read,7u);
        const auto free_list=view(src.free_list,8u,
            4u,compute::BufferAccess::ReadWrite,8u);
        const auto pair_count=view(src.aggregate_pair_count,4u,
            5u,compute::BufferAccess::Write,9u);
        const auto overflow=view(src.overflow_reports,12u,
            6u,compute::BufferAccess::Write,10u);
        const std::array<compute::BufferBinding,7> bindings{
            descriptor,root_pairs,pair_slots,free_indices,
            free_list,pair_count,overflow};
        const auto append_one=[&](compute::PipelineId pipeline,
                                  std::uint32_t groups) {
            compute::DispatchCommand command{};
            command.pipeline=pipeline;
            command.groups={groups,1u,1u};
            command.buffers.assign(bindings.begin(),bindings.end());
            command.push_constants.resize(sizeof(params));
            std::memcpy(command.push_constants.data(),
                        params.data(),sizeof(params));
            commands.emplace_back(std::move(command));
        };
        commands.reserve(commands.size()+2u);
        append_one(admit_,static_cast<std::uint32_t>(blocks));
        append_one(publish_,1u);
    }

    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregatePairAdmissionNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }
private:
    compute::Context& context_;
    compute::PipelineId admit_=0,publish_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
