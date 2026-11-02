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
        pipeline.storage_buffer_binding_count=2u;
        pipeline.storage_buffer_views={
            compute::StorageBufferViewKind::RawReadWrite,
            compute::StorageBufferViewKind::RawReadWrite};
        pipeline.push_constant_bytes=128u;
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
               std::span<const std::uint32_t> contact_order,
               std::span<const native::GpuDyContactPrepPair> contact_prep,
               CudaDeviceAddressMap::Address pcm_contacts,
               std::uint32_t pcm_pair_count) {
        if(tasks.empty() || descriptors.empty())
            throw std::invalid_argument("Source Dy GPU requires active source tasks");
        if(tasks.size()>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source Dy task dispatch exceeds device limits");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        std::size_t reused_pcm_rows=0u;
        for(const auto& pair:contact_prep)
            if(pair.pcm_pair_index!=0xffffffffu) {
                if(!pcm_contacts || pair.pcm_pair_index>=pcm_pair_count ||
                   pair.pcm_edge==0xffffffffu)
                    throw std::logic_error(
                        "Source Dy PCM contact provenance is invalid");
            }
        for(const auto& manager:manifolds)
            if(manager.patch_count) {
                if(manager.prep_index>=contact_prep.size())
                    throw std::logic_error(
                        "Source Dy PCM preparation index is invalid");
                if(contact_prep[manager.prep_index].pcm_pair_index!=0xffffffffu)
                    reused_pcm_rows+=manager.contact_count;
            }
        constexpr std::size_t count=12u;
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_ends.size_bytes(),body_ordinals.size_bytes(),
            tasks.size_bytes(),contacts.size_bytes(),friction.size_bytes(),
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes(),
            contact_prep.size_bytes()};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptors.size(),partition_ends.size(),
            body_ordinals.size(),tasks.size(),contacts.size(),friction.size(),
            patches.size(),manifolds.size(),contact_order.size(),
            contact_prep.size()};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),descriptors.data(),partition_ends.data(),
            body_ordinals.data(),tasks.data(),contacts.data(),friction.data(),
            patches.data(),manifolds.data(),contact_order.data(),
            contact_prep.data()};
        // Download only the source writeback fields. The GPU still
        // receives the original packed row inputs in one ordered upload.
        constexpr std::array<std::size_t,6> output_inputs{
            0u,1u,5u,6u,7u,8u};
        constexpr std::array<std::size_t,6> output_strides{
            48u,8u,4u,8u,20u,4u};
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
            0u,1u,5u,6u,7u,8u,2u,3u,4u,9u,10u,11u};
        std::array<std::size_t,count> offsets{};
        // Leave the source input addresses 128-byte aligned. One packed
        // output view precedes them, so readback needs only a single copy.
        const std::size_t upload_begin=(output_bytes+127u)&~std::size_t(127u);
        std::size_t total=upload_begin;
        for(const auto i:order) {
            const auto view_bytes=std::max<std::size_t>(bytes[i],4u);
            if(total>std::numeric_limits<std::uint32_t>::max()-128u ||
               view_bytes>std::numeric_limits<std::uint32_t>::max()-total)
                throw std::overflow_error("Source Dy solver arena too large");
            offsets[i]=total;
            total=(total+view_bytes+127u)&~std::size_t(127u);
        }
        narrow(total);
        host_.resize(total);
        for(std::size_t i=0;i<count;++i)
            if(bytes[i])
                std::memcpy(host_.data()+offsets[i],sources[i],bytes[i]);
        // The CPU retains unexecuted source rows. Only ranges owned by
        // active Dy tasks consume GPU writeback; the compact output region
        // needs no HtoD initialization.
        arena_.allocate(total,PX_FL);
        const auto address=arena_.getDevicePtr();
        stream_.queue_uploads({{
            address+upload_begin,host_.data()+upload_begin,
            total-upload_begin}});
        std::array<std::uint32_t,32> params{};
        for(std::size_t i=0;i<11u;++i)
            params[i<6u?6u+i:17u+i-6u]=narrow(elements[i]);
        params[30]=narrow(elements[11]);
        for(std::size_t i=0;i<result_offsets.size();++i)
            params[22u+i]=narrow(result_offsets[i]);
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=pipeline_;
        dispatch.groups={narrow(tasks.size()),1u,1u};
        // All source spans are subranges of this one source heap block.
        // Bind its original allocation once; shader offsets are relative
        // to the native view base, including the required RAW alignment.
        dispatch.buffers.push_back(source_native_view(
            addresses_,address,total,alignment_,0u,
            compute::BufferAccess::ReadWrite,params[28],max_view_));
        // A Dy-only scene has no PCM contact buffer. Its source rows
        // never index this slot; reuse the existing solver allocation.
        dispatch.buffers.push_back(source_native_view(
            addresses_,pcm_pair_count?pcm_contacts:address,
            pcm_pair_count
                ? source_allocation_bytes(pcm_pair_count,96u):total,
            alignment_,1u,compute::BufferAccess::Read,
            params[31],max_view_));
        for(std::size_t i=0;i<11u;++i)
            params[i<6u?i:i+6u]=narrow(
                static_cast<std::size_t>(params[28])+offsets[i]);
        params[29]=narrow(
            static_cast<std::size_t>(params[28])+offsets[11]);
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        stream_.commands().emplace_back(std::move(dispatch));
        const auto mapped=addresses_.resolve(address,output_bytes);
        const std::array<compute::DownloadSpan,1> downloads{
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
        // The source task graph identifies every GPU-written slot. Rows
        // excluded from its descriptor/ordinal ranges retain their original
        // CPU values rather than requiring device-side output seeding.
        for(const auto& task:tasks) {
            for(std::size_t i=task.body_begin;i<task.body_end;++i) {
                const auto index=body_ordinals[i];
                if(index>=bodies.size())
                    throw std::logic_error("Source Dy body writeback index invalid");
                std::memcpy(&bodies[index],
                    host_.data()+result_offsets[0]+index*48u,48u);
            }
            for(std::size_t i=task.descriptor_begin;
                i<task.descriptor_end;++i) {
                const auto& desc=descriptors[i];
                if(desc.distance==1u) {
                    if(desc.index>=rows.size())
                        throw std::logic_error("Source Dy row writeback index invalid");
                    std::memcpy(&rows[desc.index].applied_force,
                        host_.data()+result_offsets[1]+desc.index*8u,8u);
                    continue;
                }
                if(desc.index>=manifolds.size())
                    throw std::logic_error("Source Dy manager writeback index invalid");
                const auto& manager=manifolds[desc.index];
                if(!manager.patch_count)continue;
                if(manager.order_offset>contact_order.size() ||
                   manager.contact_count>
                       contact_order.size()-manager.order_offset ||
                   manager.patch_offset>patches.size() ||
                   manager.patch_count>patches.size()-manager.patch_offset)
                    throw std::logic_error("Source Dy contact writeback range invalid");
                for(std::size_t j=manager.order_offset;
                    j<manager.order_offset+manager.contact_count;++j) {
                    const auto index=contact_order[j];
                    if(index>=contacts.size())
                        throw std::logic_error("Source Dy contact writeback index invalid");
                    std::memcpy(&contacts[index].applied_force,
                        host_.data()+result_offsets[3]+index*8u,8u);
                }
                for(std::size_t j=manager.patch_offset;
                    j<manager.patch_offset+manager.patch_count;++j) {
                    const auto& patch=patches[j];
                    // The final source velocity pass writes only live
                    // friction anchors; zero-anchor patches retain history.
                    if(!patch.anchor_count)continue;
                    if(patch.anchor_count>2u ||
                       patch.row_offset>friction.size() ||
                       patch.anchor_count*2u>friction.size()-patch.row_offset)
                        throw std::logic_error("Source Dy friction writeback range invalid");
                    for(std::size_t k=patch.row_offset;
                        k<patch.row_offset+patch.anchor_count*2u;++k) {
                        const auto* result=
                            host_.data()+result_offsets[4]+k*20u;
                        std::memcpy(&friction[k].tangent,result,12u);
                        std::memcpy(&friction[k].applied_force,result+12u,8u);
                    }
                    std::memcpy(&patches[j].broken,
                        host_.data()+result_offsets[5]+j*4u,4u);
                }
            }
        }
        last_transfer_bytes_={total-upload_begin,output_bytes};
        last_reused_pcm_contact_rows_=reused_pcm_rows;
    }
    std::pair<std::size_t,std::size_t> last_transfer_bytes() const noexcept {
        return last_transfer_bytes_;
    }
    std::size_t last_reused_pcm_contact_rows() const noexcept {
        return last_reused_pcm_contact_rows_;
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
    std::size_t last_reused_pcm_contact_rows_=0u;
};

} // namespace fyzz::physx_gpu
