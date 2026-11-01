// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Dy's original per-task conflict partitions and prepared 1D rows are
// transferred as one transient source-heap allocation. The native Scene
// remains the sole source state owner; one ordered GPU dispatch executes
// its actual PGS impulses, including saved position-phase motion deltas.
class SourceDyDistanceNativeStage final {
public:
    SourceDyDistanceNativeStage(compute::Context& context,
        CudaDeviceAddressMap& addresses,SourceSapDmaNative& stream,
        physx::PxgHeapMemoryAllocator& allocator,compute::ShaderBinary shader)
        : context_(context),addresses_(addresses),stream_(stream),
          arena_(allocator,physx::PxsHeapStats::eSOLVER) {
        const auto& info=context_.device_info();
        if(!info.supports_float64_shader)
            throw std::runtime_error(
                "Source Dy 1D GPU requires enabled shader FP64 arithmetic");
        if(info.max_workgroup_invocations<128u ||
           info.max_workgroup_size[0]<128u ||
           !info.max_workgroup_count[0] ||
           !info.max_storage_buffer_range)
            throw std::runtime_error("Source Dy GPU partition workgroup unavailable");
        alignment_=static_cast<std::size_t>(
            std::max<std::uint64_t>(4u,info.storage_buffer_offset_alignment));
        max_view_=info.max_storage_buffer_range;
        compute::ComputePipelineDesc pipeline{};
        pipeline.shader=std::move(shader);
        pipeline.storage_buffer_binding_count=6u;
        pipeline.storage_buffer_views.assign(
            6u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=48u;
        pipeline_=context_.create_compute_pipeline(pipeline);
    }
    SourceDyDistanceNativeStage(const SourceDyDistanceNativeStage&)=delete;
    SourceDyDistanceNativeStage& operator=(const SourceDyDistanceNativeStage&)=delete;
    ~SourceDyDistanceNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
        arena_.deallocate();
    }

    void solve(std::span<native::GpuDyDistanceBody> bodies,
               std::span<native::GpuDyDistanceRow> rows,
               std::span<const native::GpuDyDistanceDescriptor> descriptors,
               std::span<const std::uint32_t> partition_ends,
               std::span<const std::uint32_t> body_ordinals,
               std::span<native::GpuDyDistanceTask> tasks) {
        if(tasks.empty() || descriptors.empty())
            throw std::invalid_argument("Source Dy GPU requires active 1D tasks");
        if(tasks.size()>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source Dy GPU task dispatch exceeds device limit");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU input exceeds 32-bit offset ABI");
            return static_cast<std::uint32_t>(value);
        };
        constexpr std::size_t parts=6u;
        const std::array<std::size_t,parts> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_ends.size_bytes(),body_ordinals.size_bytes(),
            tasks.size_bytes()};
        std::array<std::size_t,parts> offsets{};
        std::size_t total=0u;
        const auto align=[](std::size_t value) {
            return (value+127u)&~std::size_t(127u);
        };
        // Group all GPU-mutated ranges at the front of the source arena.
        // The single GPU readback need not copy read-only descriptor,
        // partition or body-ordinal inputs back to the host.
        constexpr std::array<std::size_t,parts> allocation_order{
            0u,1u,5u,2u,3u,4u};
        std::size_t output_bytes=0u;
        for(const auto i:allocation_order) {
            if(bytes[i]==0u)
                throw std::invalid_argument("Source Dy GPU input array is empty");
            if(total>std::numeric_limits<std::uint32_t>::max()-128u ||
               bytes[i]>std::numeric_limits<std::uint32_t>::max()-total)
                throw std::overflow_error("Source Dy GPU arena exceeds 32-bit range");
            offsets[i]=total;
            total=align(total+bytes[i]);
            if(i==5u)output_bytes=offsets[i]+bytes[i];
        }
        narrow(total);
        narrow(output_bytes);
        host_.resize(total);
        std::memcpy(host_.data()+offsets[0],bodies.data(),bytes[0]);
        std::memcpy(host_.data()+offsets[1],rows.data(),bytes[1]);
        std::memcpy(host_.data()+offsets[2],descriptors.data(),bytes[2]);
        std::memcpy(host_.data()+offsets[3],partition_ends.data(),bytes[3]);
        std::memcpy(host_.data()+offsets[4],body_ordinals.data(),bytes[4]);
        std::memcpy(host_.data()+offsets[5],tasks.data(),bytes[5]);
        arena_.allocate(total,PX_FL);
        const auto address=arena_.getDevicePtr();
        const std::vector<SourceSapUpload> uploads{
            {address,host_.data(),host_.size()}};
        stream_.queue_uploads(uploads);

        std::array<std::uint32_t,12> params{
            0u,0u,0u,0u,0u,0u,
            narrow(bodies.size()),narrow(rows.size()),
            narrow(descriptors.size()),narrow(partition_ends.size()),
            narrow(body_ordinals.size()),narrow(tasks.size())};
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={narrow(tasks.size()),1u,1u};
        dispatch.buffers.reserve(parts);
        for(std::uint32_t i=0;i<parts;++i)
            dispatch.buffers.push_back(source_native_view(
                addresses_,address+offsets[i],bytes[i],
                alignment_,i,compute::BufferAccess::ReadWrite,
                params[i],max_view_));
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        stream_.commands().emplace_back(std::move(dispatch));
        // A single ordered readback covers precisely the mutable body,
        // distance impulse and task-status prefix; immutable inputs stay
        // on the device until the source heap's normal retirement.
        const auto mapped=addresses_.resolve(address,output_bytes);
        const std::vector<compute::DownloadSpan> downloads{
            {mapped.buffer,mapped.offset,host_.data(),output_bytes}};
        stream_.submit();
        try {
            context_.download_batch(downloads);
        } catch(...) {
            stream_.wait_at_fetch();
            throw;
        }
        stream_.wait_at_fetch();
        std::memcpy(bodies.data(),host_.data()+offsets[0],bytes[0]);
        std::memcpy(rows.data(),host_.data()+offsets[1],bytes[1]);
        std::memcpy(tasks.data(),host_.data()+offsets[5],bytes[5]);
        for(const auto& task:tasks)
            if(task.error!=0u)
                throw std::overflow_error("Source Dy 1D GPU solver rejected a row");
        last_transfer_bytes_={total,output_bytes};
    }
    std::pair<std::size_t,std::size_t> last_transfer_bytes() const noexcept {
        return last_transfer_bytes_;
    }
private:
    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& stream_;
    physx::PxgCudaBuffer arena_;
    compute::PipelineId pipeline_=0;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
    std::vector<std::uint8_t> host_;
    // Actual packed device upload and ordered readback byte counts.
    std::pair<std::size_t,std::size_t> last_transfer_bytes_{};
};

} // namespace fyzz::physx_gpu
