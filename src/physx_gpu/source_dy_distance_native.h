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
#include <optional>
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

// Dy integrateCore reads compact transient inputs. Box-shaped bodies with
// authorized Sc device poses omit the host pose entirely; authored edits
// and bodies without eligible device shapes retain an explicit 28-byte pose.
struct SourceDyPackedIntegrationNative {
    float linear_velocity[3]{},angular_velocity[3]{};
    float sqrt_inverse_inertia[9]{};
    std::uint32_t lock_flags=0u;
    float seconds=0.0f;
    std::uint32_t active=0u,source_element=0xffffffffu;
    float mass_local_position[3]{},mass_local_orientation[4]{};
    std::uint32_t source_shape_count=0u,pose_index=0xffffffffu;
};
static_assert(sizeof(SourceDyPackedIntegrationNative)==112u &&
              offsetof(SourceDyPackedIntegrationNative,pose_index)==108u);
struct SourceDyHostPoseNative {
    float position[3]{},orientation[4]{};
};
static_assert(sizeof(SourceDyHostPoseNative)==28u);

struct SourceDyLaunchStats {
    std::size_t wide_tasks=0u;
    std::size_t max_partition_groups=0u;
    std::size_t dispatches=0u;
    std::size_t tile_bytes=0u;
};

// Dy's original per-task conflict partitions and prepared 1D rows are
// transferred as one transient source-heap allocation. The native Scene
// remains the sole source state owner. One submission orders all solver
// dispatches and preserves the position-phase motion snapshot.
class SourceDyDistanceNativeStage final {
public:
    SourceDyDistanceNativeStage(compute::Context& context,
        CudaDeviceAddressMap& addresses,SourceSapDmaNative& stream,
        physx::PxgHeapMemoryAllocator& allocator,compute::ShaderBinary shader,
        std::optional<compute::ShaderBinary> convex_shader=std::nullopt)
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
        if(convex_shader) {
            try {
                pipeline.shader=std::move(*convex_shader);
                pipeline.storage_buffer_binding_count=6u;
                pipeline.storage_buffer_views.assign(
                    6u,compute::StorageBufferViewKind::RawReadWrite);
                convex_pipeline_=context_.create_compute_pipeline(pipeline);
            }catch(...) {
                context_.destroy_compute_pipeline(pipeline_);
                pipeline_=0u;
                throw;
            }
        }
    }
    SourceDyDistanceNativeStage(const SourceDyDistanceNativeStage&)=delete;
    SourceDyDistanceNativeStage& operator=(const SourceDyDistanceNativeStage&)=delete;
    ~SourceDyDistanceNativeStage() {
        if(convex_pipeline_)context_.destroy_compute_pipeline(convex_pipeline_);
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
               std::span<native::GpuDyJointRow> rows,
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
               std::uint32_t shape_capacity,
               CudaDeviceAddressMap::Address convex_vertices,
               std::uint32_t convex_vertex_count,
               bool device_partitions=false) {
        if(integration.size()<bodies.size() ||
           integrated_poses.size()!=integration.size() ||
           (tasks.empty() && integration.size()==bodies.size()))
            throw std::invalid_argument(
                "Source Dy GPU integrateCore inputs do not match solver bodies");
        last_integrated_body_count_=0u;
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        const auto free_bodies=integration.size()-bodies.size();
        if(device_partitions) {
            build_device_launch_plan(tasks,descriptors,partition_ends,
                                     narrow(body_ordinals.size()),
                                     narrow(bodies.size()));
        } else {
            build_launch_plan(tasks,partition_ends,narrow(descriptors.size()),
                              narrow(body_ordinals.size()));
        }
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
            if((desc.row_count&255u)!=0u)continue;
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
        // Contact/friction prep and integrateCore consume the same
        // authorized Sc mass pose. No CPU pose mirror for contact bodies.
        packed_integration_.clear();
        host_poses_.clear();
        last_resident_pose_count_=0u;
        packed_integration_.reserve(integration.size());
        host_poses_.reserve(integration.size());
        for(const auto& input:integration) {
            SourceDyPackedIntegrationNative record{};
            std::memcpy(record.linear_velocity,input.linear_velocity,
                        sizeof(record.linear_velocity));
            std::memcpy(record.angular_velocity,input.angular_velocity,
                        sizeof(record.angular_velocity));
            std::memcpy(record.sqrt_inverse_inertia,input.sqrt_inverse_inertia,
                        sizeof(record.sqrt_inverse_inertia));
            record.lock_flags=input.lock_flags;
            record.seconds=input.seconds;
            record.active=input.active&1u;
            record.source_element=input.source_element;
            std::memcpy(record.mass_local_position,input.mass_local_position,
                        sizeof(record.mass_local_position));
            std::memcpy(record.mass_local_orientation,input.mass_local_orientation,
                        sizeof(record.mass_local_orientation));
            record.source_shape_count=input.source_shape_count;
            if(record.active && (input.active&2u)) {
                if(!record.source_shape_count ||
                   record.source_element==invalid)
                    throw std::logic_error("Source Dy resident pose lacks Sc ElementSim");
                record.pose_index=invalid;
                ++last_resident_pose_count_;
            } else if(record.active) {
                record.pose_index=narrow(host_poses_.size());
                SourceDyHostPoseNative pose{};
                std::memcpy(pose.position,input.position,sizeof(pose.position));
                std::memcpy(pose.orientation,input.orientation,sizeof(pose.orientation));
                host_poses_.push_back(pose);
            }
            packed_integration_.push_back(record);
        }
        constexpr std::size_t integration_header_bytes=36u;
        if(integration.size()>
           (std::numeric_limits<std::uint32_t>::max()-integration_header_bytes)/
               sizeof(SourceDyPackedIntegrationNative))
            throw std::overflow_error("Source Dy packed body inputs exceed address space");
        const std::size_t packed_input_bytes=integration_header_bytes+
            packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative);
        if(host_poses_.size()>
           (std::numeric_limits<std::uint32_t>::max()-packed_input_bytes)/
               sizeof(SourceDyHostPoseNative))
            throw std::overflow_error("Source Dy authored poses exceed address space");
        const std::size_t integration_bytes=packed_input_bytes+
            host_poses_.size()*sizeof(SourceDyHostPoseNative);
        last_host_pose_upload_bytes_=host_poses_.size()*sizeof(SourceDyHostPoseNative);
        // Transient source-order descriptors and conflict-coloring state
        // share the existing Dy arena; no separate scene state is built.
        if(device_partitions &&
           (descriptors.size()>
                (std::numeric_limits<std::uint32_t>::max()-20u)/24u ||
            bodies.size()>
                (std::numeric_limits<std::uint32_t>::max()-20u-
                    descriptors.size()*24u)/12u))
            throw std::overflow_error("Device Dy partition scratch exceeds address space");
        const std::size_t partition_scratch_bytes=device_partitions
            ? 20u+descriptors.size()*24u+bodies.size()*12u:0u;
        constexpr std::size_t count=16u;
        // The integration prefix describes Sc/SAP views and the launch
        // table. Solver-body ordinals precede the free-body suffix.
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_ends.size_bytes(),body_ordinals.size_bytes(),
            tasks.size_bytes(),contact_bytes,friction_bytes,
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes(),
            compact_prep_bytes,
            integration_bytes,compact_anchor_bytes,
            tiles_.size()*sizeof(WorkTile),partition_scratch_bytes};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptors.size(),partition_ends.size(),
            body_ordinals.size(),tasks.size(),contact_forces.size(),friction.size(),
            patches.size(),manifolds.size(),contact_order.size(),
            contact_prep.size(),integration.size(),compact_anchors_.size(),
            tiles_.size(),device_partitions?1u:0u};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),descriptors.data(),partition_ends.data(),
            body_ordinals.data(),tasks.data(),nullptr,nullptr,
            patches.data(),manifolds.data(),contact_order.data(),
            contact_prep.data(),integration.data(),nullptr,tiles_.data(),
            nullptr};
        // The host retains only source normal impulses and friction
        // tangent/force writeback. Both full row ABIs live only on GPU.
        constexpr std::array<std::size_t,6> output_inputs{
            0u,5u,6u,7u,8u,12u};
        constexpr std::array<std::size_t,6> output_strides{
            48u,4u,4u,16u,4u,32u};
        static_assert(offsetof(native::GpuDyJointRow,applied_force)==60u &&
                      offsetof(native::GpuDyDistanceTask,error)==32u &&
                      offsetof(native::GpuDyContactPoint,applied_force)==56u &&
                      offsetof(native::GpuDyFrictionRow,applied_force)==52u &&
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
        // Both writable contact and friction arenas follow all source
        // inputs. The existing GPU preparation owns the complete rows.
        constexpr std::array<std::size_t,count> order{
            0u,1u,5u,8u,2u,3u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u};
        std::array<std::size_t,count> offsets{};
        // The original Dy heap retains an EdgeIndex-keyed Sc anchor
        // prefix. Solver output follows it and remains the sole readback;
        // transient input and output locations may grow independently.
        const std::size_t upload_begin=history_bytes+
            ((output_bytes+127u)&~std::size_t(127u));
        std::size_t total=upload_begin;
        for(const auto i:order) {
            if((i==14u || i==15u) && !bytes[i])continue;
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
               i!=13u && i!=15u && bytes[i])
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
        if(device_partitions) {
            const auto n=descriptors.size();
            const std::array<std::uint32_t,5> scratch_header{
                20u,narrow(20u+n*16u),narrow(20u+n*20u),
                narrow(20u+n*24u),
                tiles_.empty()?0u:narrow(offsets[14])};
            std::memcpy(host_.data()+offsets[15],scratch_header.data(),
                        sizeof(scratch_header));
            if(n)
                std::memcpy(host_.data()+offsets[15]+20u,
                            descriptors.data(),descriptors.size_bytes());
        }
        if(!packed_integration_.empty())
            std::memcpy(host_.data()+offsets[12]+integration_header_bytes,
                        packed_integration_.data(),
                        packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative));
        if(!host_poses_.empty())
            std::memcpy(host_.data()+offsets[12]+packed_input_bytes,
                        host_poses_.data(),last_host_pose_upload_bytes_);
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
        for(std::size_t i=0;i<5u;++i)
            params[22u+i]=narrow(history_bytes+result_offsets[i]);
        if(convex_vertex_count && !convex_pipeline_)
            throw std::logic_error(
                "Cooked convex Dy integration requires the hull-capable solver shader");
        compute::DispatchCommand dispatch{};
        dispatch.pipeline=convex_vertex_count?convex_pipeline_:pipeline_;
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
        const auto empty_shape=std::numeric_limits<std::uint32_t>::max();
        bool has_shape_targets=false;
        for(const auto& input:integration) {
            if(!input.active || !input.source_shape_count)continue;
            if(input.source_element==empty_shape ||
               input.source_element>=shape_capacity ||
               input.source_shape_count>shape_capacity)
                throw std::out_of_range("Source Dy attached shape span exceeds AABB capacity");
            has_shape_targets=true;
        }
        if(has_shape_targets && (!shape_cache || !shape_bounds || !shape_geometry))
            throw std::logic_error("Source Sc GPU integration shape buffers are missing");
        std::array<std::uint32_t,9> integration_header{
            0u,0u,0u,shape_capacity,narrow(body_ordinals.size()),
            narrow(descriptors.size()),narrow(tasks.size()),
            device_partitions?narrow(std::size_t(params[28])+offsets[15]):
                (tiles_.empty()?0u:narrow(std::size_t(params[28])+offsets[14])),
            narrow(integration.size())};
        const auto bind_shape=[&](CudaDeviceAddressMap::Address source,
                                  std::uint32_t stride,std::uint32_t slot,
                                  compute::BufferAccess access,
                                  std::uint32_t& bias) {
            return source_native_view(
                addresses_,has_shape_targets?source:address,
                has_shape_targets?source_allocation_bytes(shape_capacity,stride):total,
                alignment_,slot,access,bias,max_view_);
        };
        dispatch.buffers.push_back(bind_shape(
            shape_cache,32u,2u,compute::BufferAccess::ReadWrite,
            integration_header[0]));
        dispatch.buffers.push_back(bind_shape(
            shape_bounds,24u,3u,compute::BufferAccess::ReadWrite,
            integration_header[1]));
        dispatch.buffers.push_back(bind_shape(
            shape_geometry,64u,4u,compute::BufferAccess::Read,
            integration_header[2]));
        // Primitive-only scenes execute the original five-view Dy kernel.
        // Hull geometry adds one read-only source view only to the convex
        // specialization; no extra descriptors or dynamic vertex loops
        // enter ordinary rigid constraint solves.
        if(convex_vertex_count)
            dispatch.buffers.push_back(source_native_view(
                addresses_,convex_vertices,
                source_allocation_bytes(convex_vertex_count,12u),
                alignment_,5u,compute::BufferAccess::Read,
                params[27],max_view_));
        std::memcpy(host_.data()+offsets[12],
                    integration_header.data(),sizeof(integration_header));
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
        params[11]=narrow(
            static_cast<std::size_t>(params[28])+offsets[12]);
        dispatch.push_constants.resize(sizeof(params));
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        const auto launch=[&](Launch phase,TileRange range,std::uint32_t iteration=0u) {
            const bool integrates=phase==Launch::Fused || phase==Launch::Integrate ||
                                  phase==Launch::FreeBodies;
            dispatch.buffers[2].access=integrates ? compute::BufferAccess::ReadWrite
                                                 : compute::BufferAccess::Read;
            dispatch.buffers[3].access=integrates ? compute::BufferAccess::Write
                                                 : compute::BufferAccess::Read;
            params[10]=(iteration<<4u)|static_cast<std::uint32_t>(phase)|
                (device_partitions?0x80000000u:0u)|
                (device_partitions && !wide_tasks_.empty() &&
                 phase!=Launch::Color ?0x40000000u:0u);
            for(std::size_t group=0;group<range.count;) {
                const auto count=std::min<std::size_t>(max_groups,range.count-group);
                params[8]=narrow(range.begin+group);
                dispatch.groups={narrow(count),1u,1u};
                std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
                stream_.commands().emplace_back(dispatch);
                ++last_schedule_.dispatches;
                group+=count;
            }
        };
        if(device_partitions && !wide_tasks_.empty())
            launch(Launch::Color,{0u,tasks.size()});
        launch(Launch::Fused,fused_tiles_);
        if(!wide_tasks_.empty()) {
            // The existing arena write dependencies order each dispatch
            // across workgroups. A relative partition from every wide
            // task can execute together because their body writes are disjoint.
            launch(Launch::Normals,prep_tiles_);
            launch(Launch::Friction,prep_tiles_);
            for(std::uint32_t iteration=0u;iteration<wide_iterations_;++iteration) {
                for(const auto range:partition_tiles_)
                    if(iteration<range.iterations)
                        launch(Launch::Partition,range,iteration);
                if(iteration<position_ends_.size() && position_ends_[iteration])
                    launch(Launch::Motion,body_tiles_,iteration);
            }
            launch(Launch::Integrate,body_tiles_);
            launch(Launch::PublishErrors,{0u,(tasks.size()+127u)/128u});
        }
        // The integration extent fits in launchControl's upper bits
        // because the packed arena is limited to 32-bit byte offsets.
        launch(Launch::FreeBodies,{0u,(free_bodies+127u)/128u},
               narrow(integration.size()));
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
                host_.data()+result_offsets[1]+i*4u,4u);
        for(const auto& task:tasks)
            if(task.error!=0u)
                throw std::overflow_error("Source Dy GPU solver rejected a row");
        const auto fetch_pose=[&](std::size_t index) {
            auto& pose=integrated_poses[index];
            std::memcpy(&pose,host_.data()+result_offsets[5]+
                index*sizeof(pose),sizeof(pose));
            if(pose.valid!=1u)
                throw std::overflow_error("Source Dy GPU integrateCore rejected a body");
            ++last_integrated_body_count_;
        };
        for(const auto index:body_ordinals) {
            if(index>=bodies.size())
                throw std::logic_error("Source Dy GPU integration body index invalid");
            if(integration[index].active)fetch_pose(index);
        }
        for(std::size_t index=bodies.size();index<integration.size();++index)
            fetch_pose(index);
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
                // Joint impulses remain local to the GPU PGS row.
                // No Sc task consumes a host copy after solve.
                if((desc.row_count&255u)!=0u)continue;
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
                        host_.data()+result_offsets[2]+index*4u,4u);
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
                            host_.data()+result_offsets[4]+j*4u,4u);
                    const auto count=pcm_growth?(report>>1u)&3u:
                        patch.anchor_count;
                    if(count>2u || patch.row_offset>friction.size() ||
                       count*2u>friction.size()-patch.row_offset)
                        throw std::logic_error(
                            "Source Dy friction writeback range invalid");
                    for(std::size_t k=patch.row_offset;
                        k<patch.row_offset+count*2u;++k) {
                        const auto* result=
                            host_.data()+result_offsets[3]+k*16u;
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
    std::size_t last_resident_pose_count() const noexcept {
        return last_resident_pose_count_;
    }
    std::size_t last_host_pose_upload_bytes() const noexcept {
        return last_host_pose_upload_bytes_;
    }
    std::size_t last_integrated_body_count() const noexcept {
        return last_integrated_body_count_;
    }
    SourceDyLaunchStats last_schedule() const noexcept { return last_schedule_; }
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
    enum class Launch : std::uint32_t {
        Fused, Normals, Friction, Partition, Motion, Integrate, PublishErrors,
        FreeBodies, Color
    };
    struct WorkTile {
        std::uint32_t task,begin,end,reserved=0u;
    };
    static_assert(sizeof(WorkTile)==16u);
    struct TileRange {
        std::size_t begin=0u,count=0u;
        std::uint32_t iterations=0u;
    };

    void build_device_launch_plan(
        std::span<const native::GpuDyDistanceTask> tasks,
        std::span<const native::GpuDyDistanceDescriptor> descriptors,
        std::span<const std::uint32_t> partitions,
        std::uint32_t ordinals,std::uint32_t bodies) {
        constexpr std::uint32_t threads=128u;
        constexpr std::uint32_t prep_managers=threads/4u;
        constexpr std::uint32_t wide_min=4u*threads;
        wide_tasks_.clear();
        tiles_.clear();
        partition_tiles_.clear();
        position_ends_.fill(false);
        wide_iterations_=0u;
        last_schedule_={};
        last_schedule_.max_partition_groups=tasks.empty()?0u:1u;
        degree_dynamic_.assign(bodies,0u);
        degree_static_.assign(bodies,0u);
        color_bounds_.assign(tasks.size(),0u);
        fused_tiles_={0u,tasks.size()};
        prep_tiles_={};
        body_tiles_={};
        std::uint32_t descriptor_end=0u,partition_end=0u,body_end=0u;
        std::size_t partition_levels=0u;
        for(std::size_t task_id=0;task_id<tasks.size();++task_id) {
            const auto& task=tasks[task_id];
            if(task.descriptor_begin!=descriptor_end ||
               task.descriptor_end<descriptor_end ||
               task.descriptor_end>descriptors.size() ||
               task.partition_begin!=partition_end ||
               task.partition_end<partition_end ||
               task.partition_end>partitions.size() ||
               task.partition_end-partition_end!=
                   task.descriptor_end-descriptor_end ||
               task.body_begin!=body_end ||
               task.body_end<body_end || task.body_end>ordinals ||
               !task.position_iterations || task.position_iterations>255u ||
               task.velocity_iterations>255u)
                throw std::invalid_argument(
                    "Source device Dy task ranges are inconsistent");
            std::uint32_t max_dynamic_degree=0u,max_static_degree=0u;
            for(std::uint32_t i=task.descriptor_begin;
                i<task.descriptor_end;++i) {
                const auto& desc=descriptors[i];
                const auto flags=desc.row_count&0x300u;
                if((desc.row_count&~0x3ffu)!=0u || !flags ||
                   (flags&0x100u && desc.row0>=bodies) ||
                   (flags&0x200u && desc.row1>=bodies))
                    throw std::invalid_argument(
                        "Sc device descriptor has invalid endpoint classification");
                if(flags==0x300u) {
                    max_dynamic_degree=std::max(max_dynamic_degree,
                        ++degree_dynamic_[desc.row0]);
                    max_dynamic_degree=std::max(max_dynamic_degree,
                        ++degree_dynamic_[desc.row1]);
                }else {
                    const auto body=flags&0x100u?desc.row0:desc.row1;
                    max_static_degree=std::max(max_static_degree,
                        ++degree_static_[body]);
                }
            }
            for(std::uint32_t i=task.descriptor_begin;
                i<task.descriptor_end;++i) {
                const auto& desc=descriptors[i];
                if(desc.row_count&0x100u) {
                    degree_dynamic_[desc.row0]=0u;
                    degree_static_[desc.row0]=0u;
                }
                if(desc.row_count&0x200u) {
                    degree_dynamic_[desc.row1]=0u;
                    degree_static_[desc.row1]=0u;
                }
            }
            const auto count=task.descriptor_end-task.descriptor_begin;
            const std::size_t bound=std::min<std::size_t>(count,
                (max_dynamic_degree?2u*std::size_t(max_dynamic_degree)-1u:0u)+
                max_static_degree);
            color_bounds_[task_id]=static_cast<std::uint32_t>(bound);
            descriptor_end=task.descriptor_end;
            partition_end=task.partition_end;
            body_end=task.body_end;
            // A 512-way disjoint partition requires at least 512
            // independently writable dynamic solver bodies. This keeps
            // compact connected islands inside their one-group PGS path.
            if(task.body_end-task.body_begin<wide_min || count<wide_min)
                continue;
            wide_tasks_.push_back(static_cast<std::uint32_t>(task_id));
            last_schedule_.max_partition_groups=std::max(
                last_schedule_.max_partition_groups,
                (std::size_t(count)+threads-1u)/threads);
            partition_levels=std::max(partition_levels,bound);
            wide_iterations_=std::max(wide_iterations_,
                task.position_iterations+
                    std::max(task.velocity_iterations,1u));
            position_ends_[task.position_iterations-1u]=true;
        }
        if(descriptor_end!=descriptors.size() ||
           partition_end!=partitions.size() || body_end!=ordinals)
            throw std::invalid_argument(
                "Source device Dy task input stream is incomplete");
        if(wide_tasks_.empty())return;
        last_schedule_.wide_tasks=wide_tasks_.size();
        std::size_t wide_index=0u;
        for(std::size_t task=0;task<tasks.size();++task) {
            if(wide_index<wide_tasks_.size() &&
               task==wide_tasks_[wide_index])++wide_index;
            else tiles_.push_back({
                static_cast<std::uint32_t>(task),0u,0u});
        }
        fused_tiles_.count=tiles_.size();
        const auto append_tiles=[&](std::uint32_t task,
                                    std::uint32_t begin,
                                    std::uint32_t end,std::uint32_t step) {
            while(begin<end) {
                const auto next=begin+std::min(step,end-begin);
                tiles_.push_back({task,begin,next});
                begin=next;
            }
        };
        prep_tiles_.begin=tiles_.size();
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            append_tiles(id,task.descriptor_begin,task.descriptor_end,
                         prep_managers);
        }
        prep_tiles_.count=tiles_.size()-prep_tiles_.begin;
        body_tiles_.begin=tiles_.size();
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            append_tiles(id,task.body_begin,task.body_end,threads);
        }
        body_tiles_.count=tiles_.size()-body_tiles_.begin;
        // Device colors have not been computed yet. Upper bounds come
        // from endpoint degrees, not a host conflict-colored schedule.
        // Each tile resolves its actual color interval in the shader.
        partition_tiles_.resize(partition_levels);
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            const auto count=task.descriptor_end-task.descriptor_begin;
            const auto tile_count=(std::size_t(count)+threads-1u)/threads;
            for(std::uint32_t color=0u;color<color_bounds_[id];++color) {
                auto& range=partition_tiles_[color];
                range.count+=tile_count;
                range.iterations=std::max(range.iterations,
                    task.position_iterations+
                        std::max(task.velocity_iterations,1u));
            }
        }
        std::size_t end=tiles_.size();
        for(auto& range:partition_tiles_) {
            range.begin=end;
            end+=range.count;
            range.count=0u;
        }
        tiles_.resize(end);
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            const auto count=task.descriptor_end-task.descriptor_begin;
            for(std::uint32_t color=0u;color<color_bounds_[id];++color) {
                auto& range=partition_tiles_[color];
                for(std::uint32_t offset=0u;offset<count;) {
                    const auto next=offset+std::min(threads,count-offset);
                    tiles_[range.begin+range.count++]={id,offset,next,color};
                    offset=next;
                }
            }
        }
        last_schedule_.tile_bytes=tiles_.size()*sizeof(WorkTile);
    }

    void build_launch_plan(std::span<const native::GpuDyDistanceTask> tasks,
                           std::span<const std::uint32_t> partitions,
                           std::uint32_t descriptors,std::uint32_t ordinals) {
        constexpr std::uint32_t threads=128u;
        constexpr std::uint32_t prep_managers=threads/4u;
        // Tile only when one partition can supply four full groups.
        // Smaller tasks keep their partition dependencies in one group.
        constexpr std::uint32_t wide_min=4u*threads;
        wide_tasks_.clear();
        tiles_.clear();
        partition_tiles_.clear();
        position_ends_.fill(false);
        wide_iterations_=0u;
        last_schedule_={};
        last_schedule_.max_partition_groups=tasks.empty()?0u:1u;
        fused_tiles_={0u,tasks.size()};
        prep_tiles_={};
        body_tiles_={};
        std::uint32_t descriptor_end=0u, partition_end=0u, body_end=0u;
        std::size_t partition_levels=0u;
        for(std::size_t index=0;index<tasks.size();++index) {
            const auto& task=tasks[index];
            if(task.descriptor_begin!=descriptor_end || task.descriptor_end<descriptor_end ||
               task.descriptor_end>descriptors || task.partition_begin!=partition_end ||
               task.partition_end<partition_end || task.partition_end>partitions.size() ||
               task.body_begin!=body_end || task.body_end<body_end || task.body_end>ordinals ||
               !task.position_iterations || task.position_iterations>255u ||
               task.velocity_iterations>255u)
                throw std::invalid_argument("Source Dy task ranges are invalid");
            std::uint32_t width=0u;
            for(std::uint32_t i=task.partition_begin;i<task.partition_end;++i) {
                const auto end=partitions[i];
                if(end<descriptor_end || end>task.descriptor_end)
                    throw std::invalid_argument("Source Dy partition exceeds its task");
                width=std::max(width,end-descriptor_end);
                descriptor_end=end;
            }
            if(descriptor_end!=task.descriptor_end)
                throw std::invalid_argument("Source Dy partitions do not cover their task");
            partition_end=task.partition_end;
            body_end=task.body_end;
            if(width<wide_min)continue;
            wide_tasks_.push_back(static_cast<std::uint32_t>(index));
            last_schedule_.max_partition_groups=std::max(
                last_schedule_.max_partition_groups,(std::size_t(width)+threads-1u)/threads);
            partition_levels=std::max(partition_levels,
                std::size_t(task.partition_end-task.partition_begin));
            wide_iterations_=std::max(wide_iterations_,
                task.position_iterations+std::max(task.velocity_iterations,1u));
            position_ends_[task.position_iterations-1u]=true;
        }
        if(descriptor_end!=descriptors || body_end!=ordinals || partition_end!=partitions.size())
            throw std::invalid_argument("Source Dy tasks do not cover their inputs");
        if(wide_tasks_.empty())return;
        last_schedule_.wide_tasks=wide_tasks_.size();

        std::size_t wide_index=0u;
        for(std::size_t index=0;index<tasks.size();++index) {
            if(wide_index<wide_tasks_.size() && index==wide_tasks_[wide_index])
                ++wide_index;
            else
                tiles_.push_back({static_cast<std::uint32_t>(index),0u,0u});
        }
        fused_tiles_.count=tiles_.size();
        const auto append_tiles=[&](std::uint32_t task,std::uint32_t begin,
                                    std::uint32_t end,std::uint32_t stride) {
            while(begin<end) {
                const auto next=begin+std::min(stride,end-begin);
                tiles_.push_back({task,begin,next});
                begin=next;
            }
        };
        prep_tiles_.begin=tiles_.size();
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            append_tiles(id,task.descriptor_begin,task.descriptor_end,prep_managers);
        }
        prep_tiles_.count=tiles_.size()-prep_tiles_.begin;
        body_tiles_.begin=tiles_.size();
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            append_tiles(id,task.body_begin,task.body_end,threads);
        }
        body_tiles_.count=tiles_.size()-body_tiles_.begin;

        // Count and scatter once in relative partition order. Every
        // iteration reuses these same work tiles and task ordinals.
        partition_tiles_.resize(partition_levels);
        const auto for_partitions=[&](auto&& visit) {
            for(const auto id:wide_tasks_) {
                const auto& task=tasks[id];
                auto begin=task.descriptor_begin;
                for(auto i=task.partition_begin;i<task.partition_end;++i) {
                    visit(id,partition_tiles_[i-task.partition_begin],begin,partitions[i]);
                    begin=partitions[i];
                }
            }
        };
        for_partitions([&](std::uint32_t id,TileRange& range,
                           std::uint32_t begin,std::uint32_t end) {
            range.count+=(std::size_t(end-begin)+threads-1u)/threads;
            const auto& task=tasks[id];
            range.iterations=std::max(range.iterations,
                task.position_iterations+std::max(task.velocity_iterations,1u));
        });
        auto end=tiles_.size();
        for(auto& range:partition_tiles_) {
            range.begin=end;
            end+=range.count;
            range.count=0u;
        }
        tiles_.resize(end);
        for_partitions([&](std::uint32_t id,TileRange& range,
                           std::uint32_t begin,std::uint32_t end) {
            while(begin<end) {
                const auto next=begin+std::min(threads,end-begin);
                tiles_[range.begin+range.count++]={id,begin,next};
                begin=next;
            }
        });
        last_schedule_.tile_bytes=tiles_.size()*sizeof(WorkTile);
    }

    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& stream_;
    physx::PxgCudaBuffer arena_;
    compute::PipelineId pipeline_=0;
    compute::PipelineId convex_pipeline_=0;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
    std::vector<std::uint8_t> host_;
    std::vector<SourceDyPackedContactPrepNative> packed_;
    std::vector<SourceDyPackedIntegrationNative> packed_integration_;
    std::vector<SourceDyHostPoseNative> host_poses_;
    std::vector<native::GpuDyContactPrepBody> inline_bodies_;
    std::vector<native::GpuDyFrictionAnchorInput> compact_anchors_;
    std::vector<std::size_t> patch_anchor_starts_;
    std::vector<std::uint8_t> assigned_;
    std::vector<std::uint32_t> wide_tasks_;
    std::vector<std::uint32_t> degree_dynamic_,degree_static_,color_bounds_;
    std::vector<WorkTile> tiles_;
    std::vector<TileRange> partition_tiles_;
    TileRange fused_tiles_,prep_tiles_,body_tiles_;
    std::array<bool,255> position_ends_{};
    std::uint32_t wide_iterations_=0u;
    SourceDyLaunchStats last_schedule_{};
    // Actual packed device upload and ordered readback byte counts.
    std::pair<std::size_t,std::size_t> last_transfer_bytes_{};
    std::size_t last_integrated_body_count_=0u;
    std::size_t last_resident_pose_count_=0u;
    std::size_t last_host_pose_upload_bytes_=0u;
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
