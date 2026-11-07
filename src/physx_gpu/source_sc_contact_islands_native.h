// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_native_view.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace fyzz::physx_gpu {

struct SourceScContactGraphPairNative {
    native::BodyId body_a,body_b;
    std::uint32_t dynamic_mask,edge;
};
static_assert(sizeof(SourceScContactGraphPairNative)==16u);

// The Sc PCM EdgeIndex is the source identity. The topology buffer
// materializes only current rigid connectivity and has no pair allocator.
class SourceScContactIslandsNativeStage final {
public:
    SourceScContactIslandsNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        : context_(context) {
        const auto& info=context.device_info();
        if(info.max_workgroup_invocations<128u ||
           info.max_workgroup_size[0]<128u ||
           !info.max_workgroup_count[0])
            throw std::runtime_error("Sc island GPU workgroup unavailable");
        alignment_=std::max<std::uint64_t>(4u,info.storage_buffer_offset_alignment);
        max_view_=info.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.shader=std::move(shader);
        pipeline.storage_buffer_binding_count=2u;
        pipeline.storage_buffer_views.assign(
            2u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=16u;
        pipeline_=context_.create_compute_pipeline(pipeline);
    }
    SourceScContactIslandsNativeStage(const SourceScContactIslandsNativeStage&)=delete;
    SourceScContactIslandsNativeStage& operator=(const SourceScContactIslandsNativeStage&)=delete;
    ~SourceScContactIslandsNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(SourceSapDmaNative& stream,const CudaDeviceAddressMap& addresses,
                CudaDeviceAddressMap::Address graph,std::size_t graphBytes,
                CudaDeviceAddressMap::Address contacts,std::size_t contactsBytes,
                std::uint32_t pairCount,std::uint32_t rigidCount) const {
        if(!pairCount || !rigidCount)return;
        const auto maxGroups=context_.device_info().max_workgroup_count[0];
        std::array<std::uint32_t,4> args{};
        compute::DispatchCommand command{};
        command.pipeline=pipeline_;
        command.buffers={
            source_native_view(addresses,graph,graphBytes,alignment_,0u,
                compute::BufferAccess::ReadWrite,args[0],max_view_),
            source_native_view(addresses,contacts,contactsBytes,alignment_,1u,
                compute::BufferAccess::Read,args[1],max_view_)};
        command.push_constants.resize(sizeof(args));
        const auto queue=[&](std::uint32_t phase,std::uint32_t count) {
            args[2]=phase;
            for(std::uint32_t first=0u;first<count;) {
                const auto groups=std::min<std::uint64_t>(
                    maxGroups,(std::uint64_t(count-first)+127u)/128u);
                // graph phases use a global id offset only when the
                // device limit requires more than one command.
                if(first!=0u)
                    throw std::overflow_error(
                        "Sc island graph exceeds one device dispatch domain");
                command.groups={static_cast<std::uint32_t>(groups),1u,1u};
                std::memcpy(command.push_constants.data(),args.data(),sizeof(args));
                stream.commands().emplace_back(command);
                first+=static_cast<std::uint32_t>(groups*128u);
            }
        };
        queue(0u,rigidCount);
        queue(1u,pairCount);
        queue(2u,rigidCount);
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0u;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
};

} // namespace fyzz::physx_gpu
