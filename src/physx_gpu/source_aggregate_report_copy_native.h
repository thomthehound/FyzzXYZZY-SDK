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

// PxgAggregateDesc::foundPairReport/lostPairReport -> original mapped
// foundPairReportMap/lostPairReportMap, with shared unsaturated indices.
// The source GPU allocator and mapped-report heap own all five resources.
struct SourceAggregateReportCopyNativeBuffers {
    CudaDeviceAddressMap::Address found_source=0;
    CudaDeviceAddressMap::Address lost_source=0;
    CudaDeviceAddressMap::Address found_mapped=0;
    CudaDeviceAddressMap::Address lost_mapped=0;
    CudaDeviceAddressMap::Address found_lost_counts=0;
    std::uint32_t report_capacity=0;
};
class SourceAggregateReportCopyNativeStage final {
public:
    SourceAggregateReportCopyNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        :context_(context) {
        const auto& gpu=context.device_info();
        if(gpu.max_workgroup_invocations<256u ||
           gpu.max_workgroup_size[0]<256u ||
           !gpu.max_workgroup_count[0])
            throw std::runtime_error(
                "Original aggregate report-copy GPU launch unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,gpu.storage_buffer_offset_alignment));
        max_view_=gpu.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.storage_buffer_binding_count=5u;
        pipeline.storage_buffer_views.assign(
            5u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=24u;
        pipeline.shader=std::move(shader);
        pipeline_=context.create_compute_pipeline(pipeline);
    }
    ~SourceAggregateReportCopyNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }
    SourceAggregateReportCopyNativeStage(
        const SourceAggregateReportCopyNativeStage&)=delete;
    SourceAggregateReportCopyNativeStage& operator=(
        const SourceAggregateReportCopyNativeStage&)=delete;

    void append(std::vector<compute::BatchCommand>& commands,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateReportCopyNativeBuffers& src) const {
        if(!src.report_capacity ||
           src.report_capacity>std::numeric_limits<std::uint32_t>::max()/8u ||
           src.found_source==src.lost_source ||
           src.found_mapped==src.lost_mapped ||
           src.found_source==src.found_mapped ||
           src.lost_source==src.lost_mapped)
            throw std::invalid_argument(
                "Original aggregate report-copy GPU buffers or capacity invalid");
        const auto groups=std::min<std::uint64_t>(
            64u,(std::uint64_t(src.report_capacity)*2u+255u)/256u);
        if(groups>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Original aggregate report-copy GPU grid exceeds device limits");
        std::array<std::uint32_t,6> params{src.report_capacity};
        const auto view=[&](CudaDeviceAddressMap::Address address,
                            std::size_t bytes,std::uint32_t slot,
                            compute::BufferAccess access,
                            std::uint32_t offset_index) {
            return source_native_view(addresses,address,bytes,alignment_,
                slot,access,params[offset_index],max_view_);
        };
        const auto reportBytes=source_allocation_bytes(src.report_capacity,8u);
        auto found=view(src.found_source,reportBytes,
            0u,compute::BufferAccess::Read,1u);
        auto lost=view(src.lost_source,reportBytes,
            1u,compute::BufferAccess::Read,2u);
        auto mappedFound=view(src.found_mapped,reportBytes,
            2u,compute::BufferAccess::Write,3u);
        auto mappedLost=view(src.lost_mapped,reportBytes,
            3u,compute::BufferAccess::Write,4u);
        auto counts=view(src.found_lost_counts,8u,
            4u,compute::BufferAccess::Read,5u);
        compute::DispatchCommand command{};
        command.pipeline=pipeline_;
        command.groups={static_cast<std::uint32_t>(groups),1u,1u};
        command.buffers={found,lost,mappedFound,mappedLost,counts};
        command.push_constants.resize(sizeof(params));
        std::memcpy(command.push_constants.data(),params.data(),sizeof(params));
        commands.emplace_back(std::move(command));
    }
    void append(SourceSapDmaNative& stream,
                const CudaDeviceAddressMap& addresses,
                const SourceAggregateReportCopyNativeBuffers& src) const {
        append(stream.commands(),addresses,src);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0;
    std::uint64_t max_view_=0;
};
} // namespace fyzz::physx_gpu
