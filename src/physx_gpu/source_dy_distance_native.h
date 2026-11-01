// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <algorithm>
#include <array>
#include <cstddef>
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
        pipeline.storage_buffer_binding_count=12u;
        pipeline.storage_buffer_views.assign(
            12u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=116u;
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
               std::span<native::GpuDyDistanceTask> tasks,
               std::span<native::GpuDyContactPoint> contacts,
               std::span<native::GpuDyFrictionRow> friction,
               std::span<native::GpuDyContactPatch> patches,
               std::span<const native::GpuDyContactManifold> manifolds,
               std::span<const std::uint32_t> contact_order) {
        if(tasks.empty() || descriptors.empty())
            throw std::invalid_argument("Source Dy GPU requires active source tasks");
        if(tasks.size()>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source Dy task dispatch exceeds device limits");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        constexpr std::size_t count=11u;
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_ends.size_bytes(),body_ordinals.size_bytes(),
            tasks.size_bytes(),contacts.size_bytes(),friction.size_bytes(),
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes()};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptors.size(),partition_ends.size(),
            body_ordinals.size(),tasks.size(),contacts.size(),friction.size(),
            patches.size(),manifolds.size(),contact_order.size()};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),descriptors.data(),partition_ends.data(),
            body_ordinals.data(),tasks.data(),contacts.data(),friction.data(),
            patches.data(),manifolds.data(),contact_order.data()};
        // Download only the source writeback fields. The GPU still
        // receives the original packed row inputs in one ordered upload.
        constexpr std::array<std::size_t,6> output_inputs{
            0u,1u,5u,6u,7u,8u};
        constexpr std::array<std::size_t,6> output_strides{
            48u,8u,4u,8u,8u,4u};
        constexpr std::array<std::size_t,6> input_strides{
            52u,80u,36u,88u,72u,32u};
        constexpr std::array<std::size_t,6> field_offsets{
            0u,72u,32u,72u,64u,12u};
        static_assert(offsetof(native::GpuDyDistanceRow,applied_force)==72u &&
                      offsetof(native::GpuDyDistanceTask,error)==32u &&
                      offsetof(native::GpuDyContactPoint,applied_force)==72u &&
                      offsetof(native::GpuDyFrictionRow,applied_force)==64u &&
                      offsetof(native::GpuDyContactPatch,broken)==12u);
        std::array<std::size_t,6> result_offsets{};
        std::size_t output_bytes=0u;
        for(std::size_t i=0;i<output_inputs.size();++i) {
            const auto n=elements[output_inputs[i]];
            if(n>(std::numeric_limits<std::uint32_t>::max()-output_bytes)/
                output_strides[i])
                throw std::overflow_error("Source Dy writeback exceeds 32-bit view ABI");
            result_offsets[i]=output_bytes;
            output_bytes+=n*output_strides[i];
        }
        constexpr std::array<std::size_t,count> order{
            0u,1u,5u,6u,7u,8u,2u,3u,4u,9u,10u};
        std::array<std::size_t,count> offsets{},views{};
        // Leave the source input addresses 128-byte aligned. One packed
        // output view precedes them, so readback needs only a single copy.
        std::size_t total=(output_bytes+127u)&~std::size_t(127u);
        for(const auto i:order) {
            views[i]=std::max<std::size_t>(bytes[i],4u);
            if(total>std::numeric_limits<std::uint32_t>::max()-128u ||
               views[i]>std::numeric_limits<std::uint32_t>::max()-total)
                throw std::overflow_error("Source Dy solver arena too large");
            offsets[i]=total;
            total=(total+views[i]+127u)&~std::size_t(127u);
        }
        narrow(total);
        host_.resize(total);
        for(std::size_t i=0;i<count;++i)
            if(bytes[i])
                std::memcpy(host_.data()+offsets[i],sources[i],bytes[i]);
        // Seed unmatched source slots with their original values. A
        // descriptor skipped by source preparation must retain its input
        // writeback, not uninitialized device memory.
        for(std::size_t i=0;i<output_inputs.size();++i) {
            const auto index=output_inputs[i];
            const auto* source=static_cast<const std::uint8_t*>(sources[index]);
            for(std::size_t j=0;j<elements[index];++j)
                std::memcpy(host_.data()+result_offsets[i]+
                            j*output_strides[i],
                            source+j*input_strides[i]+field_offsets[i],
                            output_strides[i]);
        }
        arena_.allocate(total,PX_FL);
        const auto address=arena_.getDevicePtr();
        stream_.queue_uploads({{address,host_.data(),host_.size()}});
        std::array<std::uint32_t,29> params{};
        for(std::size_t i=0;i<count;++i)
            params[i<6u?6u+i:17u+i-6u]=narrow(elements[i]);
        for(std::size_t i=0;i<result_offsets.size();++i)
            params[22u+i]=narrow(result_offsets[i]);
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={narrow(tasks.size()),1u,1u};
        dispatch.buffers.reserve(count+1u);
        for(std::uint32_t i=0;i<count;++i)
            dispatch.buffers.push_back(source_native_view(
                addresses_,address+offsets[i],views[i],alignment_,i,
                compute::BufferAccess::ReadWrite,
                params[i<6u?i:i+6u],max_view_));
        dispatch.buffers.push_back(source_native_view(
            addresses_,address,output_bytes,alignment_,11u,
            compute::BufferAccess::ReadWrite,params[28],max_view_));
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        stream_.commands().emplace_back(std::move(dispatch));
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
        for(std::size_t i=0;i<tasks.size();++i)
            std::memcpy(&tasks[i].error,
                host_.data()+result_offsets[2]+i*4u,4u);
        for(const auto& task:tasks)
            if(task.error!=0u)
                throw std::overflow_error("Source Dy GPU solver rejected a row");
        for(std::size_t i=0;i<bodies.size();++i)
            std::memcpy(&bodies[i],
                host_.data()+result_offsets[0]+i*48u,48u);
        for(std::size_t i=0;i<rows.size();++i)
            std::memcpy(&rows[i].applied_force,
                host_.data()+result_offsets[1]+i*8u,8u);
        for(std::size_t i=0;i<contacts.size();++i)
            std::memcpy(&contacts[i].applied_force,
                host_.data()+result_offsets[3]+i*8u,8u);
        for(std::size_t i=0;i<friction.size();++i)
            std::memcpy(&friction[i].applied_force,
                host_.data()+result_offsets[4]+i*8u,8u);
        for(std::size_t i=0;i<patches.size();++i)
            std::memcpy(&patches[i].broken,
                host_.data()+result_offsets[5]+i*4u,4u);
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
