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

// The GPU prep descriptor borrows active Dy bodies from the already uploaded
// integrateCore input. Only immovable or nonintegrated participants require
// an inline body. The source pair's 56-byte material/contact tail is intact.
struct SourceDyPackedContactPrepNative {
    std::uint32_t body0_source=0u,body1_source=0u;
    std::array<std::uint8_t,56> constants{};
};
static_assert(sizeof(SourceDyPackedContactPrepNative)==64u &&
              offsetof(SourceDyPackedContactPrepNative,constants)==8u);

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
        pipeline.storage_buffer_binding_count=5u;
        pipeline.storage_buffer_views.assign(
            5u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=128u;
        pipeline_=context_.create_compute_pipeline(pipeline);
    }
    SourceDyDistanceNativeStage(const SourceDyDistanceNativeStage&)=delete;
    SourceDyDistanceNativeStage& operator=(const SourceDyDistanceNativeStage&)=delete;
    ~SourceDyDistanceNativeStage() {
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
        arena_.deallocate();
    }
    void invalidate_after_rollback() noexcept {
        if(++history_epoch_==0u) {
            std::fill(history_epochs_.begin(),history_epochs_.end(),0u);
            history_epoch_=1u;
        }
    }

    void solve(std::span<native::GpuDyDistanceBody> bodies,
               std::span<native::GpuDyDistanceRow> rows,
               std::span<const native::GpuDyDistanceDescriptor> descriptors,
               std::span<const std::uint32_t> partition_ends,
               std::span<const std::uint32_t> body_ordinals,
               std::span<native::GpuDyDistanceTask> tasks,
               std::span<float> contact_forces,
               std::span<const native::GpuDyGuContactInput> cpu_gu,
               std::span<native::GpuDyFrictionWriteback> friction,
               std::span<const native::GpuDyFrictionAnchorInput> anchors,
               std::span<native::GpuDyContactPatch> patches,
               std::span<const native::GpuDyContactManifold> manifolds,
               std::span<const std::uint32_t> contact_order,
               std::span<const native::GpuDyContactPrepPair> contact_prep,
               std::span<const native::GpuDyBodyIntegrationInput> integration,
               std::span<native::GpuDyIntegratedPose> integrated_poses,
               CudaDeviceAddressMap::Address pcm_contacts,
               std::uint32_t pcm_pair_count,
               CudaDeviceAddressMap::Address shape_cache,
               CudaDeviceAddressMap::Address shape_bounds,
               CudaDeviceAddressMap::Address shape_geometry,
               std::uint32_t shape_capacity) {
        if(tasks.empty() || descriptors.empty())
            throw std::invalid_argument("Source Dy GPU requires active source tasks");
        if(integration.size()!=bodies.size() ||
           integrated_poses.size()!=bodies.size())
            throw std::invalid_argument(
                "Source Dy GPU integrateCore inputs do not match solver bodies");
        last_integrated_body_count_=0u;
        if(tasks.size()>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range("Source Dy task dispatch exceeds device limits");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        if(contact_forces.size()>
           std::numeric_limits<std::size_t>::max()/sizeof(native::GpuDyContactPoint))
            throw std::overflow_error("Source Dy contact row byte count overflow");
        const auto contact_bytes=
            contact_forces.size()*sizeof(native::GpuDyContactPoint);
        if(friction.size()>std::numeric_limits<std::size_t>::max()/
                              sizeof(native::GpuDyFrictionRow) ||
           friction.size()%2u || anchors.size()!=friction.size()/2u)
            throw std::logic_error("Source friction rows lack paired anchors");
        const auto friction_bytes=
            friction.size()*sizeof(native::GpuDyFrictionRow);
        std::size_t reused_pcm_rows=0u;
        std::size_t cpu_gu_rows=0u;
        for(const auto& pair:contact_prep)
            if(pair.pcm_pair_index!=0xffffffffu &&
               (!pcm_contacts || pair.pcm_pair_index>=pcm_pair_count ||
                pair.pcm_edge==0xffffffffu))
                throw std::logic_error(
                    "Source Dy PCM contact provenance is invalid");
        std::size_t next_history_capacity_=friction_history_capacity_;
        for(const auto& pair:contact_prep)
            if(pair.pcm_pair_index!=0xffffffffu)
                next_history_capacity_=std::max(next_history_capacity_,
                    std::size_t(pair.pcm_edge)+1u);
        if(next_history_capacity_>
           (std::numeric_limits<std::uint32_t>::max()-127u)/64u)
            throw std::overflow_error(
                "Source GPU friction history exceeds address space");
        const std::size_t history_bytes=(
            next_history_capacity_*64u+127u)&~std::size_t(127u);
        history_epochs_.resize(next_history_capacity_,0u);
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            if(manager.prep_index>=contact_prep.size() ||
               manager.patch_offset>patches.size() ||
               manager.patch_count>patches.size()-manager.patch_offset)
                throw std::logic_error(
                    "Source friction history has an invalid Sc patch");
            const auto& pair=contact_prep[manager.prep_index];
            if(pair.pcm_pair_index==0xffffffffu)continue;
            for(std::size_t j=manager.patch_offset;
                j<manager.patch_offset+manager.patch_count;++j) {
                auto& patch=patches[j];
                if((patch.broken&0x80000000u)==0u ||
                   patch.anchor_count==0u)continue;
                if(history_epochs_[pair.pcm_edge]!=history_epoch_) {
                    patch.broken|=0x40000000u;
                }
            }
        }
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            if(manager.prep_index>=contact_prep.size() ||
               manager.order_offset>contact_forces.size() ||
               manager.contact_count>
                   contact_forces.size()-manager.order_offset)
                throw std::logic_error(
                    "Source Dy Gu contact preparation range invalid");
            const auto& pair=contact_prep[manager.prep_index];
            if(pair.pcm_pair_index!=0xffffffffu) {
                if(pair.cpu_gu_offset!=0xffffffffu)
                    throw std::logic_error(
                        "Source PCM pair unexpectedly owns CPU Gu geometry");
                reused_pcm_rows+=manager.contact_count;
            } else {
                if(pair.cpu_gu_offset==0xffffffffu ||
                   pair.cpu_gu_offset>cpu_gu.size() ||
                   manager.contact_count>
                       cpu_gu.size()-pair.cpu_gu_offset)
                    throw std::logic_error(
                        "Source CPU Gu contact inputs are incomplete");
                cpu_gu_rows+=manager.contact_count;
            }
        }
        if(cpu_gu_rows!=cpu_gu.size() ||
           reused_pcm_rows+cpu_gu_rows!=contact_forces.size())
            throw std::logic_error(
                "Source Gu contact preparation does not cover solver rows");
        // Original Sc contact managers keep their 256-byte CPU input
        // records. Lower their GPU transfer into 72-byte descriptors and
        // reuse the already uploaded active Dy body/integrateCore inputs.
        constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
        if(contact_prep.size()>
           std::numeric_limits<std::uint32_t>::max()/
               sizeof(SourceDyPackedContactPrepNative))
            throw std::overflow_error("Source Dy contact prep descriptor overflow");
        const std::size_t prep_bytes=contact_prep.size()*
            sizeof(SourceDyPackedContactPrepNative);
        if(cpu_gu.size_bytes()>
           std::numeric_limits<std::uint32_t>::max()-prep_bytes)
            throw std::overflow_error(
                "Source Gu contact-prep arena exceeds address space");
        const std::size_t static_begin=prep_bytes+cpu_gu.size_bytes();
        packed_.resize(contact_prep.size());
        inline_bodies_.clear();
        assigned_.assign(contact_prep.size(),0u);
        const auto ref_body=[&](const native::GpuDyContactPrepBody& body,
                                std::uint32_t ordinal) -> std::uint32_t {
            if(body.inverse_mass>0.f) {
                if(ordinal==invalid || ordinal>=integration.size() ||
                   ordinal>=bodies.size() || ordinal>=invalid/2u)
                    throw std::logic_error(
                        "Active source Dy contact body has no solver ordinal");
                if(!integration[ordinal].active ||
                   body.inverse_mass!=bodies[ordinal].inverse_mass)
                    throw std::logic_error(
                        "Sc contact preparation has no matching active Dy body");
                return (ordinal<<1u)|1u;
            }
            if(inline_bodies_.size()>=(
                std::numeric_limits<std::uint32_t>::max()-static_begin)/
                    sizeof(native::GpuDyContactPrepBody))
                throw std::overflow_error("Source Dy inline contact body overflow");
            const auto offset=static_begin+inline_bodies_.size()*
                sizeof(native::GpuDyContactPrepBody);
            inline_bodies_.push_back(body);
            return narrow(offset); // An aligned offset; low bit clear.
        };
        for(const auto& desc:descriptors) {
            if(desc.distance==1u)continue;
            if(desc.index>=manifolds.size())
                throw std::logic_error("Source contact descriptor has no manager");
            const auto& manager=manifolds[desc.index];
            if(!manager.patch_count)continue;
            if(manager.prep_index>=contact_prep.size() ||
               assigned_[manager.prep_index]++)
                throw std::logic_error("Source Dy contact prep is not unique");
            const auto& source=contact_prep[manager.prep_index];
            auto& target=packed_[manager.prep_index];
            target.body0_source=ref_body(source.body0,desc.row0);
            target.body1_source=ref_body(source.body1,desc.row1);
            std::memcpy(target.constants.data(),
                reinterpret_cast<const std::uint8_t*>(&source)+192u,
                target.constants.size());
        }
        // Only CPU Gu friction patches and invalidated GPU PCM history
        // require host anchor input. Compact those source-owned anchors
        // in patch order; each patch records its own position relative
        // to the next contact-row arena. Ordinary PCM patches read their
        // already-resident EdgeIndex history instead.
        compact_anchors_.clear();
        patch_anchor_starts_.assign(patches.size(),
            std::numeric_limits<std::size_t>::max());
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            const auto& pair=contact_prep[manager.prep_index];
            const bool pcm=pair.pcm_pair_index!=invalid;
            for(std::size_t j=manager.patch_offset;
                j<manager.patch_offset+manager.patch_count;++j) {
                auto& patch=patches[j];
                if(patch.anchor_count>2u || patch.row_offset%2u ||
                   patch.row_offset/2u>anchors.size() ||
                   patch.anchor_count>anchors.size()-patch.row_offset/2u)
                    throw std::logic_error(
                        "Source GPU friction patch has invalid anchor rows");
                if(pcm && (patch.broken&0x40000000u)==0u)continue;
                if(!patch.anchor_count)continue;
                patch_anchor_starts_[j]=compact_anchors_.size();
                const auto begin=anchors.begin()+patch.row_offset/2u;
                compact_anchors_.insert(compact_anchors_.end(),begin,
                                        begin+patch.anchor_count);
            }
        }
        const auto compact_anchor_bytes=compact_anchors_.size()*
            sizeof(native::GpuDyFrictionAnchorInput);
        if(compact_anchor_bytes>
           std::numeric_limits<std::uint32_t>::max()-prep_bytes-
               cpu_gu.size_bytes())
            throw std::overflow_error(
                "Source GPU compact friction inputs exceed address space");
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            const auto& pair=contact_prep[manager.prep_index];
            const bool pcm=pair.pcm_pair_index!=invalid;
            for(std::size_t j=manager.patch_offset;
                j<manager.patch_offset+manager.patch_count;++j) {
                auto& patch=patches[j];
                const auto start=patch_anchor_starts_[j];
                if(start==std::numeric_limits<std::size_t>::max())continue;
                const auto distance=narrow((compact_anchors_.size()-start)*
                    sizeof(native::GpuDyFrictionAnchorInput));
                if(pcm) {
                    // The GPU source descriptor uses the CPU-Gu offset
                    // slot for recovery anchors only when PCM is resident.
                    auto& target=packed_[manager.prep_index];
                    std::memcpy(target.constants.data()+
                        offsetof(native::GpuDyContactPrepPair,cpu_gu_offset)-
                        offsetof(native::GpuDyContactPrepPair,rest_distance),
                                &distance,sizeof(distance));
                }else patch.reserved=distance;
            }
        }
        // A manager excluded by the source island/partition pass has no
        // GPU reader; only scheduled descriptors require body references.
        const std::size_t inline_body_bytes=
            inline_bodies_.size()*sizeof(native::GpuDyContactPrepBody);
        const std::size_t compact_prep_bytes=static_begin+inline_body_bytes;
        constexpr std::size_t count=14u;
        // One 16-byte header describes the three original Sc/SAP GPU
        // allocations, followed by exactly one source body record per row.
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_ends.size_bytes(),body_ordinals.size_bytes(),
            tasks.size_bytes(),contact_bytes,friction_bytes,
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes(),
            compact_prep_bytes,
            integration.size_bytes()+16u,compact_anchor_bytes};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptors.size(),partition_ends.size(),
            body_ordinals.size(),tasks.size(),contact_forces.size(),friction.size(),
            patches.size(),manifolds.size(),contact_order.size(),
            contact_prep.size(),integration.size(),compact_anchors_.size()};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),descriptors.data(),partition_ends.data(),
            body_ordinals.data(),tasks.data(),nullptr,nullptr,
            patches.data(),manifolds.data(),contact_order.data(),
            contact_prep.data(),integration.data(),nullptr};
        // The host retains only source normal impulses and friction
        // tangent/force writeback. Both full row ABIs live only on GPU.
        constexpr std::array<std::size_t,7> output_inputs{
            0u,1u,5u,6u,7u,8u,12u};
        constexpr std::array<std::size_t,7> output_strides{
            48u,4u,4u,4u,16u,4u,32u};
        static_assert(offsetof(native::GpuDyDistanceRow,applied_force)==60u &&
                      offsetof(native::GpuDyDistanceTask,error)==32u &&
                      offsetof(native::GpuDyContactPoint,applied_force)==56u &&
                      offsetof(native::GpuDyFrictionRow,applied_force)==52u &&
                      offsetof(native::GpuDyContactPatch,broken)==12u);
        std::array<std::size_t,7> result_offsets{};
        std::size_t output_bytes=0u;
        for(std::size_t i=0;i<output_inputs.size();++i) {
            const auto n=elements[output_inputs[i]];
            if(n>(std::numeric_limits<std::uint32_t>::max()-output_bytes)/
                output_strides[i])
                throw std::overflow_error("Source Dy writeback exceeds 32-bit view ABI");
            result_offsets[i]=output_bytes;
            output_bytes+=n*output_strides[i];
        }
        // Both writable contact and friction arenas follow all source
        // inputs. The existing GPU preparation owns the complete rows.
        constexpr std::array<std::size_t,count> order{
            0u,1u,5u,8u,2u,3u,4u,9u,10u,11u,12u,13u,6u,7u};
        std::array<std::size_t,count> offsets{};
        // The original Dy heap retains an EdgeIndex-keyed Sc anchor
        // prefix. Solver output follows it and remains the sole readback;
        // transient input and output locations may grow independently.
        const std::size_t upload_begin=history_bytes+
            ((output_bytes+127u)&~std::size_t(127u));
        std::size_t total=upload_begin;
        for(const auto i:order) {
            const auto view_bytes=std::max<std::size_t>(bytes[i],4u);
            if(total>std::numeric_limits<std::uint32_t>::max()-128u ||
               view_bytes>std::numeric_limits<std::uint32_t>::max()-total)
                throw std::overflow_error("Source Dy solver arena too large");
            offsets[i]=total;
            // Anchor input ends exactly at the writable contact arena.
            // GPU-only PCM can omit that trailing region while retaining
            // one contiguous source upload and the original dispatch.
            total=i==13u?total+view_bytes:
                (total+view_bytes+127u)&~std::size_t(127u);
        }
        narrow(total);
        // Host staging contains only compact readback and real upload
        // inputs. Writable GPU contact rows have no CPU mirror.
        host_.resize(offsets[6]);
        for(std::size_t i=0;i<count;++i)
            if(i!=6u && i!=7u && i!=11u && i!=12u &&
               i!=13u && bytes[i])
                std::memcpy(host_.data()+offsets[i],sources[i],bytes[i]);
        if(prep_bytes)
            std::memcpy(host_.data()+offsets[11],packed_.data(),prep_bytes);
        if(!cpu_gu.empty())
            std::memcpy(host_.data()+offsets[11]+prep_bytes,
                        cpu_gu.data(),cpu_gu.size_bytes());
        if(!compact_anchors_.empty())
            std::memcpy(host_.data()+offsets[13],
                        compact_anchors_.data(),compact_anchor_bytes);
        if(!inline_bodies_.empty())
            std::memcpy(host_.data()+offsets[11]+static_begin,
                        inline_bodies_.data(),inline_bodies_.size()*
                            sizeof(native::GpuDyContactPrepBody));
        if(!integration.empty())
            std::memcpy(host_.data()+offsets[12]+16u,
                        integration.data(),integration.size_bytes());
        // The source GPU kernel constructs all writable Gu contact rows.
        // One ordered source upload covers every other active input and
        // stops before the contact-row arena, without scattered HtoD spans.
        // Only the Sc EdgeIndex friction mirror survives arena growth;
        // every other Dy input and writeback is rebuilt for this frame.
        arena_.allocateCopyPrefixAsync(
            total,friction_history_capacity_*64u,stream_,PX_FL);
        friction_history_capacity_=next_history_capacity_;
        const auto address=arena_.getDevicePtr();
        const std::size_t uploaded_bytes=
            (compact_anchors_.empty()?offsets[13]:offsets[6])-upload_begin;
        std::array<std::uint32_t,32> params{};
        for(std::size_t i=0;i<11u;++i)
            params[i<6u?6u+i:17u+i-6u]=narrow(elements[i]);
        params[30]=narrow(elements[11]);
        for(std::size_t i=0;i<6u;++i)
            params[22u+i]=narrow(history_bytes+result_offsets[i]);
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
                ? source_allocation_bytes(pcm_pair_count,84u):total,
            alignment_,1u,compute::BufferAccess::Read,
            params[31],max_view_));
        if(shape_capacity && (!shape_cache || !shape_bounds || !shape_geometry))
            throw std::logic_error("Source Sc GPU integration shape buffers are missing");
        const auto empty_shape=std::numeric_limits<std::uint32_t>::max();
        for(const auto& input:integration)
            if(input.active && input.source_element!=empty_shape &&
               input.source_element>=shape_capacity)
                throw std::out_of_range(
                    "Source Dy integrated ElementSim handle exceeds AABB capacity");
        for(const auto& input:integration)
            if(input.active && (input.source_shape_count>shape_capacity ||
               (input.source_element==empty_shape && input.source_shape_count)))
                throw std::out_of_range("Source Dy attached shape span exceeds AABB capacity");
        std::array<std::uint32_t,4> shape_header{0u,0u,0u,shape_capacity};
        const auto bind_shape=[&](CudaDeviceAddressMap::Address source,
                                  std::uint32_t stride,std::uint32_t slot,
                                  compute::BufferAccess access,
                                  std::uint32_t& bias) {
            const bool empty=shape_capacity==0u;
            return source_native_view(
                addresses_,empty?address:source,
                empty?total:source_allocation_bytes(shape_capacity,stride),
                alignment_,slot,access,bias,max_view_);
        };
        dispatch.buffers.push_back(bind_shape(
            shape_cache,32u,2u,compute::BufferAccess::ReadWrite,
            shape_header[0]));
        dispatch.buffers.push_back(bind_shape(
            shape_bounds,24u,3u,compute::BufferAccess::ReadWrite,
            shape_header[1]));
        dispatch.buffers.push_back(bind_shape(
            shape_geometry,64u,4u,compute::BufferAccess::Read,
            shape_header[2]));
        std::memcpy(host_.data()+offsets[12],
                    shape_header.data(),sizeof(shape_header));
        const auto previous_spans=stream_.submission_stats().upload_spans;
        stream_.queue_uploads({{
            address+upload_begin,host_.data()+upload_begin,uploaded_bytes}});
        last_upload_spans_=stream_.submission_stats().upload_spans-
                           previous_spans;
        for(std::size_t i=0;i<11u;++i)
            params[i<6u?i:i+6u]=narrow(
                static_cast<std::size_t>(params[28])+offsets[i]);
        params[29]=narrow(
            static_cast<std::size_t>(params[28])+offsets[11]);
        // The launch group count is already exactly tasks.size().
        // Reuse the redundant taskCount root constant for integration
        // input bias rather than exceeding Vulkan's 128-byte minimum.
        params[11]=narrow(
            static_cast<std::size_t>(params[28])+offsets[12]);
        dispatch.push_constants.resize(sizeof(params));
        std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
        stream_.commands().emplace_back(std::move(dispatch));
        const auto mapped=addresses_.resolve(address+history_bytes,output_bytes);
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
        for(const auto& task:tasks)
            for(std::size_t i=task.body_begin;i<task.body_end;++i) {
                const auto index=body_ordinals[i];
                if(index>=integration.size())
                    throw std::logic_error(
                        "Source Dy GPU integration body index invalid");
                if(!integration[index].active)continue;
                auto& pose=integrated_poses[index];
                std::memcpy(&pose,host_.data()+result_offsets[6]+
                    std::size_t(index)*sizeof(pose),sizeof(pose));
                if(pose.valid!=1u)
                    throw std::logic_error(
                        "Source Dy GPU integrateCore pose not published");
                ++last_integrated_body_count_;
            }
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
                        host_.data()+result_offsets[1]+desc.index*4u,4u);
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
                    if(index>=contact_forces.size())
                        throw std::logic_error("Source Dy contact writeback index invalid");
                    std::memcpy(&contact_forces[index],
                        host_.data()+result_offsets[3]+index*4u,4u);
                }
                for(std::size_t j=manager.patch_offset;
                    j<manager.patch_offset+manager.patch_count;++j) {
                    auto& patch=patches[j];
                    const bool pcm_growth=(patch.broken&0x80000000u)!=0u;
                    // The GPU selects new PCM anchors before source PGS.
                    // One original four-byte patch report conveys its
                    // resulting count and changed Gu indices, without
                    // transferring a second device anchor array.
                    std::uint32_t report=0u;
                    if(pcm_growth || patch.anchor_count)
                        std::memcpy(&report,
                            host_.data()+result_offsets[5]+j*4u,4u);
                    const auto count=pcm_growth?(report>>1u)&3u:
                        patch.anchor_count;
                    if(count>2u || patch.row_offset>friction.size() ||
                       count*2u>friction.size()-patch.row_offset)
                        throw std::logic_error(
                            "Source Dy friction writeback range invalid");
                    for(std::size_t k=patch.row_offset;
                        k<patch.row_offset+count*2u;++k) {
                        const auto* result=
                            host_.data()+result_offsets[4]+k*16u;
                        std::memcpy(&friction[k].tangent,result,12u);
                        std::memcpy(&friction[k].applied_force,result+12u,4u);
                    }
                    if(pcm_growth) {
                        patch.anchor_count=count;
                        patch.reserved=report;
                        patch.broken=report&1u;
                    }else if(count)patch.broken=report;
                }
            }
        }
        last_transfer_bytes_={uploaded_bytes,output_bytes};
        last_reused_pcm_contact_rows_=reused_pcm_rows;
        last_omitted_pcm_upload_bytes_=
            reused_pcm_rows*sizeof(native::GpuDyContactPoint);
        last_cpu_gu_geometry_bytes_=cpu_gu.size_bytes();
        last_friction_anchor_upload_bytes_=compact_anchor_bytes;
        for(const auto& manager:manifolds) {
            if(!manager.patch_count || manager.prep_index>=contact_prep.size())
                continue;
            const auto& pair=contact_prep[manager.prep_index];
            if(pair.pcm_pair_index!=0xffffffffu)
                history_epochs_[pair.pcm_edge]=history_epoch_;
        }
        last_friction_prepared_rows_=friction.size();
        last_contact_prep_upload_saving_=
            static_cast<std::int64_t>(contact_prep.size_bytes())-
            static_cast<std::int64_t>(prep_bytes+inline_body_bytes);
    }
    std::size_t last_integrated_body_count() const noexcept {
        return last_integrated_body_count_;
    }
    std::pair<std::size_t,std::size_t> last_transfer_bytes() const noexcept {
        return last_transfer_bytes_;
    }
    std::size_t last_reused_pcm_contact_rows() const noexcept {
        return last_reused_pcm_contact_rows_;
    }
    std::size_t last_omitted_pcm_upload_bytes() const noexcept {
        return last_omitted_pcm_upload_bytes_;
    }
    std::size_t last_cpu_gu_geometry_bytes() const noexcept {
        return last_cpu_gu_geometry_bytes_;
    }
    std::size_t last_upload_spans() const noexcept {
        return last_upload_spans_;
    }
    std::size_t last_friction_anchor_upload_bytes() const noexcept {
        return last_friction_anchor_upload_bytes_;
    }
    std::size_t last_friction_prepared_rows() const noexcept {
        return last_friction_prepared_rows_;
    }
    std::int64_t last_contact_prep_upload_saving() const noexcept {
        return last_contact_prep_upload_saving_;
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
    std::vector<SourceDyPackedContactPrepNative> packed_;
    std::vector<native::GpuDyContactPrepBody> inline_bodies_;
    std::vector<native::GpuDyFrictionAnchorInput> compact_anchors_;
    std::vector<std::size_t> patch_anchor_starts_;
    std::vector<std::uint8_t> assigned_;
    // Actual packed device upload and ordered readback byte counts.
    std::pair<std::size_t,std::size_t> last_transfer_bytes_{};
    std::size_t last_integrated_body_count_=0u;
    std::size_t last_reused_pcm_contact_rows_=0u;
    std::size_t last_omitted_pcm_upload_bytes_=0u;
    std::size_t last_cpu_gu_geometry_bytes_=0u;
    std::size_t last_upload_spans_=0u;
    std::size_t last_friction_anchor_upload_bytes_=0u;
    std::size_t friction_history_capacity_=0u;
    std::vector<std::uint32_t> history_epochs_;
    std::uint32_t history_epoch_=1u;
    std::size_t last_friction_prepared_rows_=0u;
    std::int64_t last_contact_prep_upload_saving_=0;
};

} // namespace fyzz::physx_gpu
