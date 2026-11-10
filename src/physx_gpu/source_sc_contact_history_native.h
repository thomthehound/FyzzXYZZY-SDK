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

namespace fyzz::physx_gpu {

// The journal is indexed by the already-admitted Sc PCM pair sequence;
// it is not another EdgeIndex allocator or a contact-manifold owner.
class SourceScContactHistoryNativeStage final {
public:
    enum class Phase:std::uint32_t {
        Initialize=0u,Snapshot=1u,Restore=2u,
        PublishHostTouch=3u,PublishHostWithSnapshot=4u,RestoreHost=5u,
        PublishGuTouch=6u,RestoreGuTouch=7u
    };
    SourceScContactHistoryNativeStage(
        compute::Context& context,compute::ShaderBinary shader)
        : context_(context) {
        const auto& info=context_.device_info();
        if(info.max_workgroup_invocations<128u ||
           info.max_workgroup_size[0]<128u ||
           !info.max_workgroup_count[0] ||
           !info.max_storage_buffer_range)
            throw std::runtime_error("Native Sc PCM journal compute unavailable");
        alignment_=std::max<std::uint64_t>(
            4u,info.storage_buffer_offset_alignment);
        max_view_=info.max_storage_buffer_range;
        compute::ComputePipelineDesc desc{};
        desc.shader=std::move(shader);
        desc.storage_buffer_binding_count=3u;
        desc.storage_buffer_views.assign(
            3u,compute::StorageBufferViewKind::RawReadWrite);
        desc.push_constant_bytes=32u;
        pipeline_=context_.create_compute_pipeline(desc);
    }
    SourceScContactHistoryNativeStage(const SourceScContactHistoryNativeStage&)=delete;
    SourceScContactHistoryNativeStage& operator=(const SourceScContactHistoryNativeStage&)=delete;
    ~SourceScContactHistoryNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
    }

    void append(SourceSapDmaNative& stream,const CudaDeviceAddressMap& addresses,
                Phase phase,CudaDeviceAddressMap::Address pairs,
                CudaDeviceAddressMap::Address edges,
                CudaDeviceAddressMap::Address journal,
                std::uint32_t pairCount,std::uint32_t edgeCapacity,
                std::uint32_t clearBegin=0u) const {
        if(clearBegin>edgeCapacity)
            throw std::out_of_range("Sc contact journal clear exceeds EdgeIndex capacity");
        const auto records=phase==Phase::Initialize
            ?edgeCapacity-clearBegin:pairCount;
        if(!records)return;
        const bool compact_gu=phase==Phase::PublishGuTouch ||
            phase==Phase::RestoreGuTouch;
        const bool host_stream=compact_gu ||
            phase==Phase::PublishHostTouch ||
            phase==Phase::PublishHostWithSnapshot ||
            phase==Phase::RestoreHost;
        const std::uint64_t word_count=host_stream
            ?records:std::uint64_t(records)*60u;
        if(word_count>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Sc contact journal word count overflow");
        const auto words=static_cast<std::uint32_t>(word_count);
        std::array<std::uint32_t,8> args{};
        const auto edgeBytes=source_allocation_bytes(edgeCapacity,240u);
        const auto pairBytes=phase==Phase::Initialize
            ?edgeBytes:source_allocation_bytes(pairCount,
                compact_gu?8u:(host_stream?84u:20u));
        const auto journalBytes=phase==Phase::Initialize
            ?edgeBytes:source_allocation_bytes(pairCount,
                host_stream?8u:240u);
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.buffers={
            source_native_view(addresses,
                phase==Phase::Initialize?edges:pairs,pairBytes,
                alignment_,0u,compute::BufferAccess::Read,args[0],max_view_),
            source_native_view(addresses,edges,edgeBytes,
                alignment_,1u,compute::BufferAccess::ReadWrite,args[1],max_view_),
            source_native_view(addresses,
                phase==Phase::Initialize?edges:journal,journalBytes,
                alignment_,2u,
                    phase==Phase::Restore || phase==Phase::RestoreHost ||
                        phase==Phase::RestoreGuTouch ||
                        phase==Phase::PublishHostTouch
                        ?compute::BufferAccess::Read
                        :compute::BufferAccess::ReadWrite,args[2],max_view_)};
        args[3]=static_cast<std::uint32_t>(phase);
        args[5]=words;
        args[6]=edgeCapacity;
        args[7]=clearBegin;
        dispatch.push_constants.resize(sizeof(args));
        const auto maxGroups=context_.device_info().max_workgroup_count[0];
        for(std::uint32_t first=0u;first<words;) {
            const auto groups=std::min<std::uint64_t>(
                maxGroups,(std::uint64_t(words-first)+127u)/128u);
            args[4]=first;
            dispatch.groups={static_cast<std::uint32_t>(groups),1u,1u};
            std::memcpy(dispatch.push_constants.data(),args.data(),sizeof(args));
            stream.commands().emplace_back(dispatch);
            first+=static_cast<std::uint32_t>(
                std::min<std::uint64_t>(std::uint64_t(words)-first,groups*128u));
        }
    }
private:
    compute::Context& context_;
    compute::PipelineId pipeline_=0u;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
};

} // namespace fyzz::physx_gpu
