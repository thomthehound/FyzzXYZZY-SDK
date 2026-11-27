// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_native_view.h"
#include "physx_gpu/source_frame_input_arena_native.h"
#include "physx_gpu/frame_working_set.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sc_device_islands_native.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz::physx_gpu {

// Device-authored PCM contact extent travels with the existing patch report.
inline constexpr std::uint32_t source_dy_pcm_extent_valid_bit=1u<<30u;
inline constexpr std::uint32_t source_dy_pcm_extent_shift=16u;
inline constexpr std::uint32_t source_dy_pcm_extent_count_mask=7u;
inline constexpr std::uint32_t source_dy_pcm_compacted_bit=1u<<28u;

// Dy contact preparation shares timestep coefficients across a frame.
// Per-manager records retain only body references and numerical material.
struct SourceDyPackedContactPrepNative {
    std::uint32_t body0_source=0u,body1_source=0u;
    std::array<std::uint8_t,40> constants{};
};
static_assert(sizeof(SourceDyPackedContactPrepNative)==48u &&
              offsetof(SourceDyPackedContactPrepNative,constants)==8u);
inline constexpr std::size_t source_dy_contact_frame_bytes=16u;

// Dy integrateCore reads compact transient inputs. Box-shaped bodies with
// authorized Sc device poses omit the host pose entirely; authored edits
// and bodies without eligible device shapes retain an explicit 28-byte pose.
struct SourceDyPackedIntegrationNative {
    std::uint32_t lock_flags=0u;
    float seconds=0.0f;
    std::uint32_t active=0u,source_element=0xffffffffu;
    std::uint32_t source_shape_count=0u,pose_index=0xffffffffu;
    std::uint32_t actor_local_offset=0xffffffffu;
    // Prepared velocities and active force modifications are compact
    // scalar streams; exact component masks live in active flags.
    std::uint32_t velocity_offset=0xffffffffu;
    std::uint32_t force_offset=0xffffffffu;
    std::uint32_t counted_interactions=0u;
};
static_assert(sizeof(SourceDyPackedIntegrationNative)==40u &&
              offsetof(SourceDyPackedIntegrationNative,seconds)==4u &&
              offsetof(SourceDyPackedIntegrationNative,pose_index)==20u &&
              offsetof(SourceDyPackedIntegrationNative,actor_local_offset)==24u &&
              offsetof(SourceDyPackedIntegrationNative,velocity_offset)==28u &&
              offsetof(SourceDyPackedIntegrationNative,force_offset)==32u &&
              offsetof(SourceDyPackedIntegrationNative,counted_interactions)==36u);
inline constexpr std::uint32_t source_dy_device_inertia_flag=0x4000u;
inline constexpr std::uint32_t source_dy_conditional_integration_flag=0x8000u;
inline constexpr std::uint32_t source_dy_wake_integration_flag=0x10000000u;
static_assert(((source_dy_device_inertia_flag|source_dy_conditional_integration_flag|
                source_dy_wake_integration_flag)&
    ((63u<<8u)|(4095u<<16u)|1u|8u|16u|32u|64u|128u))==0u);
struct SourceDyPreparedVelocityNative {
    float linear[3]{},angular[3]{};
};
static_assert(sizeof(SourceDyPreparedVelocityNative)==24u);
struct SourceDyHostPoseNative {
    float position[3]{},orientation[4]{};
};
static_assert(sizeof(SourceDyHostPoseNative)==28u);
struct SourceDyActorLocalNative {
    float position[3]{},orientation[4]{};
    float inverse_mass=0.0f;
    // Stable authored dynamics precede GPU-owned motion and sleep state.
    float linear_damping=0.0f,angular_damping=0.0f;
    float max_linear_speed_squared=0.0f,max_angular_speed_squared=0.0f;
    float inverse_inertia_local[3]{};
    std::uint32_t dynamics_flags=0u;
    float linear_velocity[3]{},angular_velocity[3]{};
    // Dy owns wake arbitration across frames. Sc mirrors this state
    // after a validated device solve for rollback and public queries.
    float wake_counter=0.4f;
    float sleep_linear_acc[3]{},sleep_angular_acc[3]{};
    // Dy integrates the actor frame independently of attached shape storage.
    float world_position[3]{},world_orientation[4]{};
};
static_assert(sizeof(SourceDyActorLocalNative)==144u &&
              offsetof(SourceDyActorLocalNative,inverse_mass)==28u &&
              offsetof(SourceDyActorLocalNative,linear_damping)==32u &&
              offsetof(SourceDyActorLocalNative,dynamics_flags)==60u &&
              offsetof(SourceDyActorLocalNative,linear_velocity)==64u &&
              offsetof(SourceDyActorLocalNative,angular_velocity)==76u &&
              offsetof(SourceDyActorLocalNative,wake_counter)==88u &&
              offsetof(SourceDyActorLocalNative,sleep_linear_acc)==92u &&
              offsetof(SourceDyActorLocalNative,sleep_angular_acc)==104u &&
              offsetof(SourceDyActorLocalNative,world_position)==116u &&
              offsetof(SourceDyActorLocalNative,world_orientation)==128u);

struct SourceDyLaunchStats {
    std::size_t device_colored_tasks=0u;
    std::size_t device_colored_descriptors=0u;
    // Only wide-task candidates require a host degree analysis.
    std::size_t host_degree_scanned_descriptors=0u;
    std::size_t wide_tasks=0u;
    std::size_t max_partition_groups=0u;
    std::size_t dispatches=0u;
    std::size_t constraint_preparation_dispatches=0u;
    std::size_t motion_capture_dispatches=0u;
    std::size_t device_island_indirect_dispatches=0u;
    std::size_t host_lane_packed_tasks=0u;
    std::size_t device_admitted_tasks=0u;
    std::size_t device_admitted_bodies=0u;
    std::size_t device_admitted_descriptors=0u;
    std::size_t device_task_readback_bytes=0u;
    std::size_t device_epoch_error_readback_bytes=0u;
    std::size_t device_contact_admission_readback_bytes=0u;
    std::size_t tile_bytes=0u;
    std::size_t device_partition_tile_bytes=0u;
    std::size_t host_staging_bytes=0u;
    std::size_t mapped_input_bytes=0u;
    std::size_t numerical_arena_capacity=0u;
    std::size_t numerical_arena_bytes=0u;
    std::size_t numerical_arena_padding_bytes=0u;
    std::size_t host_result_capacity=0u;
    std::size_t device_partition_indirect_dispatches=0u;
    std::size_t resident_fused_workgroups=0u;
    bool resident_wide_schedule=false;
    bool resident_schedule_reuse_candidate=false;
    bool resident_schedule_reused=false;
    // Device-written descriptors, partition ranges, tasks, and ordinals
    // omitted from the host numerical input packet.
    std::size_t device_generated_bytes=0u;
    std::size_t omitted_partition_scratch_bytes=0u;
    std::size_t device_source_copy_bytes=0u;
    std::size_t device_source_descriptors=0u;
    std::size_t host_descriptor_inputs=0u;
    std::size_t source_descriptor_upload_bytes=0u;
    // Native Sc initializes transient solver bodies from the resident
    // actor inverse masses; standalone Dy retains explicit input support.
    std::size_t device_initialized_bodies=0u;
    std::size_t device_body_upload_bytes=0u;
    std::size_t device_initialization_dispatches=0u;
    std::size_t device_world_inertia_bodies=0u;
    std::size_t host_inertia_upload_bytes=0u;
    std::size_t device_inertia_generated_bytes=0u;
    std::size_t device_world_motion_bodies=0u;
    std::size_t actor_local_upload_bytes=0u;
    std::size_t host_contact_source_bytes=0u;
    std::size_t retained_friction_source_bytes=0u;
    std::size_t prewitness_retained_friction_upload_bytes=0u;
    std::size_t late_retained_friction_upload_bytes=0u;
    std::size_t prewitness_retained_friction_direct_bytes=0u;
    std::size_t late_retained_friction_direct_bytes=0u;
    std::size_t resident_immovable_contact_bodies=0u;
    std::size_t inline_contact_body_bytes=0u;
    std::size_t actor_local_upload_spans=0u;
    std::size_t prewitness_actor_local_upload_bytes=0u;
    std::size_t prewitness_actor_local_upload_spans=0u;
    std::size_t device_upload_batches=0u;
    std::size_t reused_actor_local_poses=0u;
    std::size_t resident_velocity_reuses=0u;
    std::size_t prepared_velocity_updated_bodies=0u;
    std::size_t prepared_velocity_upload_bytes=0u;
    std::size_t device_joint_input_rows=0u;
    std::size_t device_joint_geometry_count=0u;
    std::size_t device_joint_metadata_upload_bytes=0u;
    std::size_t prewitness_joint_source_upload_bytes=0u;
    std::size_t resident_joint_source_bytes=0u;
    std::size_t device_joint_metadata_upload_spans=0u;
    std::size_t device_joint_metadata_reuses=0u;
    std::size_t device_preintegrated_bodies=0u;
    std::size_t force_mod_upload_bytes=0u;
    std::size_t free_velocity_readback_bytes=0u;
    std::size_t device_sleep_bodies=0u;
    std::size_t sleep_readback_bytes=0u;
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
        std::optional<compute::ShaderBinary> convex_shader=std::nullopt,
        std::optional<compute::ShaderBinary> contact_endpoints_shader=std::nullopt)
        : context_(context),addresses_(addresses),stream_(stream),
          arena_(allocator,physx::PxsHeapStats::eSOLVER),
          actor_locals_(allocator,physx::PxsHeapStats::eSOLVER),
          joint_sources_(allocator,physx::PxsHeapStats::eSOLVER),
          retained_sources_(allocator,physx::PxsHeapStats::eSOLVER),
          retained_frame_input_(context,addresses,stream) {
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
        pipeline.storage_buffer_binding_count=9u;
        pipeline.storage_buffer_views.assign(
            9u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=128u;
        pipeline_=context_.create_compute_pipeline(pipeline);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,"dy.distance_pgs");
#endif
        if(convex_shader) {
            try {
                pipeline.shader=std::move(*convex_shader);
                pipeline.storage_buffer_binding_count=10u;
                pipeline.storage_buffer_views.assign(
                    10u,compute::StorageBufferViewKind::RawReadWrite);
                convex_pipeline_=context_.create_compute_pipeline(pipeline);
#if defined(FYZZXYZZY_PROFILE_GPU)
                context_.set_pipeline_profile_label(convex_pipeline_,"dy.convex");
#endif
            }catch(...) {
                context_.destroy_compute_pipeline(pipeline_);
                pipeline_=0u;
                throw;
            }
        }
        if(contact_endpoints_shader) {
            try {
                compute::ComputePipelineDesc refs{};
                refs.shader=std::move(*contact_endpoints_shader);
                refs.storage_buffer_binding_count=2u;
                refs.storage_buffer_views.assign(
                    2u,compute::StorageBufferViewKind::RawReadWrite);
                refs.push_constant_bytes=40u;
                contact_endpoints_pipeline_=context_.create_compute_pipeline(refs);
#if defined(FYZZXYZZY_PROFILE_GPU)
                context_.set_pipeline_profile_label(
                    contact_endpoints_pipeline_,"dy.contact_endpoints");
#endif
            }catch(...) {
                if(convex_pipeline_)context_.destroy_compute_pipeline(convex_pipeline_);
                if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
                convex_pipeline_=0u;
                pipeline_=0u;
                throw;
            }
        }
    }
    SourceDyDistanceNativeStage(const SourceDyDistanceNativeStage&)=delete;
    SourceDyDistanceNativeStage& operator=(const SourceDyDistanceNativeStage&)=delete;
    ~SourceDyDistanceNativeStage() {
        if(contact_endpoints_pipeline_)
            context_.destroy_compute_pipeline(contact_endpoints_pipeline_);
        if(convex_pipeline_)context_.destroy_compute_pipeline(convex_pipeline_);
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
        if(partition_dispatch_buffer_)
            context_.retire_buffer(partition_dispatch_buffer_);
        if(input_upload_buffer_)context_.retire_buffer(input_upload_buffer_);
        arena_.deallocate();
        actor_locals_.deallocate();
        joint_sources_.deallocate();
        retained_sources_.deallocate();
    }
    void retire_rigid_actor(BodyId id) noexcept {
        if(id<actor_state_.size())actor_state_[id]=0u;
    }
    void invalidate_after_rollback() noexcept {
        resident_schedule_committed_=false;
        pending_solve_.reset();
        deferred_result_downloads_.clear();
        std::fill(actor_state_.begin(),actor_state_.end(),0u);
        std::fill(joint_source_valid_.begin(),joint_source_valid_.end(),0u);
        actor_local_live_slots_=0u;
        joint_source_live_slots_=0u;
        friction_history_live_slots_=0u;
        prewitness_joint_source_slots_.clear();
        prewitness_joint_source_upload_bytes_=0u;
        prewitness_joint_source_upload_spans_=0u;
        prewitness_actor_local_upload_bytes_=0u;
        prewitness_actor_local_upload_spans_=0u;
        prewitness_actor_slots_.clear();
        prewitness_retained_source_data_=nullptr;
        prewitness_retained_source_count_=0u;
        retained_source_address_=0u;
        if(++history_epoch_==0u) {
            std::fill(history_epochs_.begin(),history_epochs_.end(),0u);
            history_epoch_=1u;
        }
    }

    void discard_unconsumed_retained_friction_sources() noexcept {
        prewitness_retained_source_data_=nullptr;
        prewitness_retained_source_count_=0u;
        retained_source_address_=0u;
    }

    void enqueue_retained_friction_sources(
        std::span<const native::GpuDyRetainedFrictionCandidate> candidates) {
        if(pending_solve_ || prewitness_retained_source_data_)
            throw std::logic_error(
                "Dy retained friction source overlaps a numerical epoch");
        if(candidates.empty())return;
        const auto bytes=source_allocation_bytes(
            candidates.size(),sizeof(native::GpuDyRetainedFrictionCandidate));
        if(bytes>max_view_ ||
           bytes>std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("Retained friction source exceeds device view");
        if(retained_frame_input_.direct())
            retained_source_address_=retained_frame_input_.write(
                candidates.data(),bytes);
        else {
            retained_sources_.allocate(bytes,PX_FL);
            retained_source_address_=retained_sources_.getDevicePtr();
            auto& uploads=stream_.upload_scratch();
            uploads.push_back({
                retained_source_address_,candidates.data(),bytes});
            stream_.queue_uploads(uploads);
        }
        prewitness_retained_source_data_=candidates.data();
        prewitness_retained_source_count_=candidates.size();
    }

    void enqueue_joint_sources(
        std::span<const native::GpuDyJointInput> inputs) {
        if(pending_solve_)
            throw std::logic_error("Dy joint source overlaps a numerical epoch");
        prewitness_joint_source_slots_.clear();
        prewitness_joint_source_upload_bytes_=0u;
        prewitness_joint_source_upload_spans_=0u;
        if(inputs.empty())return;
        std::size_t capacity=joint_source_capacity_;
        for(const auto& joint:inputs) {
            if(joint.source_slot==0xffffffffu)
                throw std::logic_error("Native Dy joint has no live Sc slot");
            capacity=std::max(capacity,std::size_t(joint.source_slot)+1u);
        }
        if(capacity>
               (std::numeric_limits<std::uint32_t>::max()-127u)/
                   sizeof(native::GpuDyJointInput) ||
           capacity>max_view_/sizeof(native::GpuDyJointInput))
            throw std::overflow_error("Native Dy joint sources exceed device view");
        joint_source_host_.resize(capacity);
        joint_source_valid_.resize(capacity,0u);
        if(capacity>joint_source_capacity_) {
            joint_sources_.allocateCopyPrefixAsync(
                capacity*sizeof(native::GpuDyJointInput)+127u,
                joint_source_live_slots_*sizeof(native::GpuDyJointInput),
                stream_,PX_FL);
            joint_source_capacity_=capacity;
        }
        for(const auto& joint:inputs) {
            const auto slot=joint.source_slot;
            if(!joint_source_valid_[slot] ||
               std::memcmp(&joint_source_host_[slot],&joint,sizeof(joint))!=0) {
                joint_source_valid_[slot]=0u;
                joint_source_host_[slot]=joint;
                prewitness_joint_source_slots_.push_back(slot);
            }
        }
        std::sort(prewitness_joint_source_slots_.begin(),
                  prewitness_joint_source_slots_.end());
        if(std::adjacent_find(prewitness_joint_source_slots_.begin(),
                              prewitness_joint_source_slots_.end())!=
           prewitness_joint_source_slots_.end())
            throw std::logic_error("Native Dy joint source admitted twice");
        auto& uploads=stream_.upload_scratch();
        uploads.clear();
        for(std::size_t i=0u;i<prewitness_joint_source_slots_.size();) {
            const auto first=prewitness_joint_source_slots_[i];
            auto last=first+1u;
            ++i;
            while(i<prewitness_joint_source_slots_.size() &&
                  prewitness_joint_source_slots_[i]==last) {
                ++last;
                ++i;
            }
            const auto bytes=std::size_t(last-first)*
                sizeof(native::GpuDyJointInput);
            uploads.push_back({
                joint_sources_.getDevicePtr()+std::size_t(first)*
                    sizeof(native::GpuDyJointInput),
                joint_source_host_.data()+first,bytes});
            prewitness_joint_source_upload_bytes_+=bytes;
        }
        prewitness_joint_source_upload_spans_=uploads.size();
        stream_.queue_uploads(uploads);
        if(!prewitness_joint_source_slots_.empty())
            joint_source_live_slots_=std::max(joint_source_live_slots_,
                std::size_t(prewitness_joint_source_slots_.back())+1u);
    }

    void discard_unconsumed_actor_sources() noexcept {
        for(const auto id:prewitness_actor_slots_)
            actor_state_[id]&=~actor_prewitness_mask;
        prewitness_actor_slots_.clear();
        prewitness_actor_local_upload_bytes_=0u;
        prewitness_actor_local_upload_spans_=0u;
    }

    void enqueue_actor_sources(
        std::span<const native::GpuDyBodyIntegrationInput> inputs) {
        if(pending_solve_)
            throw std::logic_error("Dy actor source overlaps a numerical epoch");
        if(inputs.empty())return;
        const std::size_t first_dirty_slot=prewitness_actor_slots_.size();
        pending_actor_locals_.clear();
        std::size_t needed_actor_slots=0u;
        for(const auto& input:inputs) {
            const auto id=input.source_actor;
            if(!(input.active&1u) || id==0xffffffffu ||
               id>(std::numeric_limits<std::uint32_t>::max()-128u)/
                      sizeof(SourceDyActorLocalNative))
                throw std::overflow_error(
                    "Pre-witness Dy actor lacks a valid source slot");
            needed_actor_slots=std::max(needed_actor_slots,std::size_t(id)+1u);
        }
        reserve_actor_local_slots(needed_actor_slots);
        constexpr auto metadata_bytes=
            offsetof(SourceDyActorLocalNative,linear_velocity);
        constexpr std::size_t velocity_bytes=24u,pose_bytes=28u,sleep_bytes=28u;
        for(const auto& input:inputs) {
            const auto id=input.source_actor;
            SourceDyActorLocalNative local{};
            std::memcpy(local.position,input.mass_local_position,
                        sizeof(local.position));
            std::memcpy(local.orientation,input.mass_local_orientation,
                        sizeof(local.orientation));
            local.inverse_mass=input.source_inverse_mass;
            local.linear_damping=input.linear_damping;
            local.angular_damping=input.angular_damping;
            local.max_linear_speed_squared=input.max_linear_speed_squared;
            local.max_angular_speed_squared=input.max_angular_speed_squared;
            std::memcpy(local.inverse_inertia_local,input.inverse_inertia_local,
                        sizeof(local.inverse_inertia_local));
            local.dynamics_flags=input.dynamics_flags;
            std::memcpy(&local.wake_counter,&input.wake_counter,sleep_bytes);
            std::memcpy(local.world_position,input.actor_position,
                        sizeof(local.world_position));
            std::memcpy(local.world_orientation,input.actor_orientation,
                        sizeof(local.world_orientation));
            auto& resident=actor_local_host_[id];
            auto& state=actor_state_[id];
            const bool metadata_changed=!(state&actor_metadata_valid) ||
                std::memcmp(&resident,&local,metadata_bytes)!=0;
            const bool pose_changed=!(state&actor_pose_valid) ||
                std::memcmp(resident.world_position,local.world_position,
                            pose_bytes)!=0;
            const bool sleep_changed=!(state&actor_metadata_valid) ||
                std::memcmp(&resident.wake_counter,
                            &local.wake_counter,sleep_bytes)!=0;
            std::uint8_t dirty=(metadata_changed?1u:0u)|
                (sleep_changed?4u:0u)|(pose_changed?8u:0u);
            if(metadata_changed)
                std::memcpy(&resident,&local,metadata_bytes);
            if(pose_changed)
                std::memcpy(resident.world_position,local.world_position,
                            pose_bytes);
            if(!(state&(actor_metadata_valid|actor_velocity_valid)) &&
               (input.active&0x100u)==0u) {
                std::memcpy(resident.linear_velocity,input.linear_velocity,
                            velocity_bytes);
                dirty|=2u;
            }
            if((input.active&0x100u)!=0u) {
                if(input.source_inverse_mass!=0.0f)
                    throw std::logic_error(
                        "Immovable Dy source has responding inverse mass");
                if(!(state&actor_metadata_valid) ||
                   std::memcmp(resident.linear_velocity,
                               input.linear_velocity,velocity_bytes)!=0) {
                    std::memcpy(resident.linear_velocity,
                                input.linear_velocity,velocity_bytes);
                    dirty|=2u;
                }
                state&=~actor_velocity_valid;
            }
            if(sleep_changed)
                std::memcpy(&resident.wake_counter,
                            &local.wake_counter,sleep_bytes);
            state|=(metadata_changed?actor_prewitness_metadata:0u)|
                   (pose_changed?actor_prewitness_pose:0u);
            if(dirty) {
                prewitness_actor_slots_.push_back(id);
                pending_actor_locals_.push_back({id,dirty});
            }
        }
        std::sort(prewitness_actor_slots_.begin()+first_dirty_slot,
                  prewitness_actor_slots_.end());
        if(std::adjacent_find(prewitness_actor_slots_.begin()+first_dirty_slot,
                             prewitness_actor_slots_.end())!=
           prewitness_actor_slots_.end())
            throw std::logic_error("Pre-witness Dy actor source admitted twice");
        build_actor_local_uploads();
        auto& uploads=stream_.upload_scratch();
        uploads.assign(actor_local_uploads_.begin(),actor_local_uploads_.end());
        for(const auto& upload:uploads)
            prewitness_actor_local_upload_bytes_+=upload.bytes;
        prewitness_actor_local_upload_spans_+=uploads.size();
        stream_.queue_uploads(uploads);
        actor_local_live_slots_=std::max(actor_local_live_slots_,needed_actor_slots);
        for(std::size_t i=first_dirty_slot;i<prewitness_actor_slots_.size();++i) {
            const auto id=prewitness_actor_slots_[i];
            actor_state_[id]|=actor_metadata_valid|actor_pose_valid;
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
               std::span<native::GpuDyBodyIntegrationInput> integration,
               std::span<native::GpuDyIntegratedPose> integrated_poses,
               CudaDeviceAddressMap::Address pcm_contacts,
               std::uint32_t pcm_pair_count,
               CudaDeviceAddressMap::Address shape_cache,
               CudaDeviceAddressMap::Address shape_bounds,
               CudaDeviceAddressMap::Address shape_geometry,
               std::uint32_t shape_capacity,
               CudaDeviceAddressMap::Address convex_vertices,
               std::uint32_t convex_vertex_count,
               bool device_partitions=false,
               std::span<native::GpuDyFrictionAnchorImpulse> anchor_impulses={},
               bool device_initial_bodies=false,
               bool device_world_motion=false,
               bool device_unconstrained_preparation=false,
               std::span<native::GpuDySleepState> sleep_states={},
               std::span<const native::GpuDyJointInput> joint_inputs={}) {
        enqueue(bodies,rows,descriptors,partition_ends,body_ordinals,
                tasks,contact_forces,cpu_gu,friction,anchors,patches,
                manifolds,contact_order,contact_prep,integration,
                integrated_poses,pcm_contacts,pcm_pair_count,shape_cache,
                shape_bounds,shape_geometry,shape_capacity,convex_vertices,
                convex_vertex_count,device_partitions,anchor_impulses,
                device_initial_bodies,device_world_motion,
                device_unconstrained_preparation,sleep_states,joint_inputs);
        finish();
    }
    void enqueue(std::span<native::GpuDyDistanceBody> bodies,
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
               std::span<native::GpuDyBodyIntegrationInput> integration,
               std::span<native::GpuDyIntegratedPose> integrated_poses,
               CudaDeviceAddressMap::Address pcm_contacts,
               std::uint32_t pcm_pair_count,
               CudaDeviceAddressMap::Address shape_cache,
               CudaDeviceAddressMap::Address shape_bounds,
               CudaDeviceAddressMap::Address shape_geometry,
               std::uint32_t shape_capacity,
               CudaDeviceAddressMap::Address convex_vertices,
               std::uint32_t convex_vertex_count,
               bool device_partitions=false,
               std::span<native::GpuDyFrictionAnchorImpulse> anchor_impulses={},
               bool device_initial_bodies=false,
               bool device_world_motion=false,
               bool device_unconstrained_preparation=false,
               std::span<native::GpuDySleepState> sleep_states={},
               std::span<const native::GpuDyJointInput> joint_inputs={},
               std::optional<SourceScDeviceIslandDyView> device_admission=std::nullopt,
               std::span<const compute::DownloadSpan> extra_downloads={},
               bool readback_device_task_records=true,
               bool compact_contact_admission=false,
               std::span<const native::GpuDyCompactContactPrep> compact_contact_prep={},
        std::span<const native::GpuDyRetainedFrictionCandidate> retained_candidates={},
        bool defer_result_dma=false,
        native::GpuDyWideScheduleBounds wide_bounds={},
        native::GpuDyResidentScheduleCapacity resident_capacity={}) {
        if(pending_solve_)
            throw std::logic_error("Source Dy numerical epoch already queued");
        const bool prior_schedule=std::exchange(resident_schedule_committed_,false);
        const auto prior_tasks=last_schedule_.device_admitted_tasks;
        if(!contact_prep.empty() && !compact_contact_prep.empty())
            throw std::invalid_argument(
                "Native Dy contact preparation has multiple source representations");
        const std::size_t prep_count=
            contact_prep.size()+compact_contact_prep.size();
        std::array<float,4> contact_frame{};
        bool first_contact_source=true;
        std::size_t next_history_capacity_=friction_history_capacity_;
        const auto append_source=[&](const auto& source) {
            const std::array<float,4> coefficients{
                source.inv_dt,source.seconds,source.rigid_contact_bias,
                source.bounce_threshold};
            if(first_contact_source) {
                contact_frame=coefficients;
                first_contact_source=false;
            }else if(std::memcmp(contact_frame.data(),coefficients.data(),
                                 sizeof(contact_frame))!=0)
                throw std::invalid_argument(
                    "Dy contact sources disagree on frame timestep coefficients");
            if(source.pcm_pair_index==0xffffffffu)return;
            if(!pcm_contacts || source.pcm_pair_index>=pcm_pair_count ||
               source.pcm_edge==0xffffffffu)
                throw std::logic_error(
                    "Source Dy PCM contact provenance is invalid");
            next_history_capacity_=std::max(next_history_capacity_,
                std::size_t(source.pcm_edge)+1u);
        };
        const auto contact_origin=[&](std::size_t index) {
            return source_contact_origin(index,contact_prep,
                                         compact_contact_prep);
        };
        for(const auto& source:contact_prep)append_source(source);
        for(const auto& source:compact_contact_prep)append_source(source);
        if(resident_capacity.tasks || resident_capacity.body_ordinals) {
            if(!device_admission || !resident_capacity.tasks ||
               !tasks.empty() || !body_ordinals.empty() ||
               readback_device_task_records)
                throw std::invalid_argument("Dy resident capacity conflicts with host schedule");
        }
        const auto task_capacity=resident_capacity.tasks
            ?resident_capacity.tasks:tasks.size();
        const auto ordinal_capacity=resident_capacity.tasks
            ?resident_capacity.body_ordinals:body_ordinals.size();
        const auto task_bytes=task_capacity*sizeof(native::GpuDyDistanceTask);
        const auto ordinal_bytes=ordinal_capacity*sizeof(std::uint32_t);
        if(device_admission &&
           task_capacity>source_sc_indirect_task_limit(context_.device_info()))
            throw std::out_of_range(
                "Device Sc compact tasks exceed indirect GPU Dy launch capacity");

        if(integration.size()<bodies.size() ||
           integrated_poses.size()!=integration.size() ||
           (task_capacity==0u && integration.size()==bodies.size()))
            throw std::invalid_argument(
                "Source Dy GPU integrateCore inputs do not match solver bodies");
        last_integrated_body_count_=0u;
        last_schedule_.resident_immovable_contact_bodies=0u;
        last_schedule_.inline_contact_body_bytes=0u;
        if(device_initial_bodies && !device_partitions)
            throw std::invalid_argument(
                "Source Dy device body initialization requires device partitions");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        const auto free_bodies=integration.size()-bodies.size();
        bool source_indexed_integration=false;
        bool device_generated_inertia=device_world_motion;
        bool has_shape_targets=false;
        std::size_t active_bodies=0u,needed_actor_slots=0u;
        constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
        for(std::size_t ordinal=0u;ordinal<integration.size();++ordinal) {
            const auto& input=integration[ordinal];
            const bool active=(input.active&1u)!=0u;
            if(ordinal<bodies.size()) {
                source_indexed_integration|=
                    (input.active&native::gpu_dy_source_indexed_integration)!=0u;
                if(device_initial_bodies && device_world_motion &&
                   !active && input.source_inverse_mass!=0.0f)
                    throw std::logic_error(
                        "Native Dy inactive solver row has nonzero inverse mass");
                if(active && (input.active&32u)==0u)
                    device_generated_inertia=false;
            }
            if(input.active && input.source_shape_count) {
                if(input.source_element==invalid ||
                   input.source_element>=shape_capacity ||
                   input.source_shape_count>shape_capacity)
                    throw std::out_of_range(
                        "Source Dy attached shape span exceeds AABB capacity");
                has_shape_targets=true;
            }
            if(!active)continue;
            ++active_bodies;
            const auto id=device_world_motion?input.source_actor:narrow(ordinal);
            if(id==invalid ||
               id>(invalid-128u)/sizeof(SourceDyActorLocalNative))
                throw std::overflow_error(
                    "Source Dy rigid actor local-pose index exceeds GPU address space");
            needed_actor_slots=std::max(needed_actor_slots,std::size_t(id)+1u);
        }
        if(has_shape_targets && (!shape_cache || !shape_bounds || !shape_geometry))
            throw std::logic_error("Source Sc GPU integration shape buffers are missing");
        if(source_indexed_integration &&
           (!device_admission || device_admission->deferred_body_remap ||
            !compact_contact_admission || !device_unconstrained_preparation))
            throw std::logic_error("Source-indexed integration requires resident Sc admission");
        const auto active_integration_count=source_indexed_integration
            ?active_bodies:ordinal_capacity+free_bodies;
        const auto descriptor_capacity=device_admission
            ?std::size_t(device_admission->response_capacity)
            :descriptors.size();
        if(compact_contact_admission &&
           (!device_admission ||
            device_admission->contact_manager_capacity!=manifolds.size()))
            throw std::invalid_argument(
                "Source Dy compact admission requires the complete Sc candidate table");
        if(device_unconstrained_preparation &&
           sleep_states.size()!=integration.size())
            throw std::invalid_argument(
                "Source GPU Dy sleep-state outputs do not match body ordinals");
        const auto partition_count=device_admission
            ? descriptor_capacity
            : device_partitions
                ? (task_capacity==0u?0u:tasks.back().partition_end)
                : partition_ends.size();
        if(partition_count>std::numeric_limits<std::uint32_t>::max()/4u)
            throw std::overflow_error("Source Dy partition arena exceeds 32-bit offsets");
        if(device_partitions) {
            build_device_launch_plan(tasks,descriptors,narrow(partition_count),
                                     narrow(descriptor_capacity),
                                     narrow(ordinal_capacity),
                                     narrow(bodies.size()),
                                     device_admission.has_value(),task_capacity);
        } else {
            build_launch_plan(tasks,partition_ends,narrow(descriptors.size()),
                              narrow(ordinal_capacity));
        }
        const bool resident_wide=wide_bounds.stages!=0u;
        if(resident_wide) {
            if(!source_indexed_integration || !device_admission ||
               !device_partitions || !device_initial_bodies || !descriptors.empty())
                throw std::logic_error("Resident wide Dy requires source-indexed admission");
            build_resident_launch_plan(wide_bounds,bodies.size(),
                                       descriptor_capacity,task_capacity);
        }
        last_schedule_.host_descriptor_inputs=descriptors.size();
        last_schedule_.host_contact_source_bytes=compact_contact_prep.empty()
            ?contact_prep.size_bytes():compact_contact_prep.size_bytes();
        last_schedule_.retained_friction_source_bytes=
            retained_candidates.size_bytes();
        last_schedule_.device_initialized_bodies=device_initial_bodies?bodies.size():0u;
        last_schedule_.device_body_upload_bytes=device_initial_bodies
            ? (device_world_motion?0u:bodies.size()*sizeof(float))
            : bodies.size_bytes();
        last_schedule_.device_world_motion_bodies=
            device_world_motion?ordinal_capacity:0u;
        if(device_unconstrained_preparation &&
           (!device_world_motion || !device_initial_bodies))
            throw std::invalid_argument(
                "GPU preintegration requires native device body ownership");
        last_schedule_.device_preintegrated_bodies=
            device_unconstrained_preparation
                ?active_integration_count:0u;
        last_schedule_.device_sleep_bodies=
            last_schedule_.device_preintegrated_bodies;
        last_schedule_.device_joint_input_rows=
            std::count_if(rows.begin(),rows.end(),
                [](const auto& row){return row.reserved==1u || row.reserved==2u;});
        last_schedule_.device_joint_geometry_count=joint_inputs.size();
        last_schedule_.device_joint_metadata_upload_bytes=
            prewitness_joint_source_upload_bytes_;
        last_schedule_.device_joint_metadata_upload_spans=
            prewitness_joint_source_upload_spans_;
        last_schedule_.prewitness_joint_source_upload_bytes=
            prewitness_joint_source_upload_bytes_;
        last_schedule_.prewitness_actor_local_upload_bytes=
            prewitness_actor_local_upload_bytes_;
        last_schedule_.prewitness_actor_local_upload_spans=
            prewitness_actor_local_upload_spans_;
        last_schedule_.device_joint_metadata_reuses=0u;
        if(!joint_inputs.empty() && !device_world_motion)
            throw std::invalid_argument(
                "Source device joint geometry requires native resident actor motion");
        for(const auto& descriptor:descriptors) {
            const auto row_count=descriptor.row_count&255u;
            if(!row_count)continue;
            if(descriptor.index>=rows.size() ||
               row_count>rows.size()-descriptor.index)
                throw std::logic_error("Source Dy joint descriptor exceeds row storage");
            const auto& first=rows[descriptor.index];
            if(first.reserved!=2u)continue;
            const auto index=std::bit_cast<std::uint32_t>(first.applied_force);
            if(index>=joint_inputs.size())
                throw std::logic_error("Source Dy joint metadata index is invalid");
            const auto kind=joint_inputs[index].kind;
            const auto expected=kind==0u?1u:kind==2u?3u:
                kind==1u?6u:5u;
            if(kind>4u || row_count!=expected)
                throw std::logic_error("Source Dy joint kind and row count disagree");
        }
        if(contact_forces.size()>
           std::numeric_limits<std::size_t>::max()/sizeof(native::GpuDyContactPoint))
            throw std::overflow_error("Source Dy contact row byte count overflow");
        const auto contact_bytes=
            contact_forces.size()*sizeof(native::GpuDyContactPoint);
        const bool compact_friction=!anchor_impulses.empty();
        if(compact_friction && !friction.empty())
            throw std::invalid_argument(
                "Source Dy friction cannot return rows and anchors together");
        if(compact_friction && anchor_impulses.size()!=anchors.size())
            throw std::invalid_argument("Source Dy anchor report count mismatch");
        if(anchors.size()>std::numeric_limits<std::size_t>::max()/2u)
            throw std::overflow_error("Source Dy friction count overflow");
        const auto friction_count=compact_friction
            ? anchor_impulses.size()*2u:friction.size();
        if(friction_count>std::numeric_limits<std::size_t>::max()/
                             sizeof(native::GpuDyFrictionRow) ||
           friction_count%2u || anchors.size()!=friction_count/2u)
            throw std::logic_error("Source friction rows lack paired anchors");
        const auto friction_bytes=
            friction_count*sizeof(native::GpuDyFrictionRow);
        last_friction_writeback_bytes_=compact_friction
            ? anchor_impulses.size()*sizeof(native::GpuDyFrictionAnchorImpulse)
            : friction.size()*sizeof(native::GpuDyFrictionWriteback);
        std::size_t reused_pcm_rows=0u;
        std::size_t cpu_gu_rows=0u;
        if(next_history_capacity_>
           (std::numeric_limits<std::uint32_t>::max()-127u)/64u)
            throw std::overflow_error(
                "Source GPU friction history exceeds address space");
        const std::size_t history_bytes=(
            next_history_capacity_*64u+127u)&~std::size_t(127u);
        std::size_t next_joint_capacity=joint_source_capacity_;
        for(const auto& joint:joint_inputs) {
            if(joint.source_slot==0xffffffffu)
                throw std::logic_error("Native Dy joint has no live Sc constraint slot");
            next_joint_capacity=std::max(next_joint_capacity,
                                         std::size_t(joint.source_slot)+1u);
        }
        if(next_joint_capacity>
               (std::numeric_limits<std::uint32_t>::max()-127u)/
                   sizeof(native::GpuDyJointInput) ||
           next_joint_capacity>max_view_/sizeof(native::GpuDyJointInput))
            throw std::overflow_error("Native Dy resident joints exceed device view");
        last_schedule_.resident_joint_source_bytes=
            next_joint_capacity*sizeof(native::GpuDyJointInput);
        joint_source_host_.resize(next_joint_capacity);
        joint_source_valid_.resize(next_joint_capacity,0u);
        joint_source_dirty_.clear();
        joint_source_late_upload_slots_.clear();
        for(const auto& joint:joint_inputs) {
            const auto slot=joint.source_slot;
            const bool matches_staged=
                std::binary_search(prewitness_joint_source_slots_.begin(),
                                   prewitness_joint_source_slots_.end(),slot) &&
                std::memcmp(&joint_source_host_[slot],&joint,sizeof(joint))==0;
            if(!joint_source_valid_[slot] ||
               std::memcmp(&joint_source_host_[slot],&joint,sizeof(joint))!=0) {
                // A rejected frame never commits source-slot residency.
                joint_source_valid_[slot]=0u;
                joint_source_host_[slot]=joint;
                joint_source_dirty_.push_back(slot);
                if(!matches_staged)
                    joint_source_late_upload_slots_.push_back(slot);
            }else
                ++last_schedule_.device_joint_metadata_reuses;
        }
        std::sort(joint_source_dirty_.begin(),joint_source_dirty_.end());
        std::sort(joint_source_late_upload_slots_.begin(),
                  joint_source_late_upload_slots_.end());
        if(std::adjacent_find(joint_source_dirty_.begin(),
                              joint_source_dirty_.end())!=joint_source_dirty_.end())
            throw std::logic_error("Native Dy joint slot admitted more than once");
        history_epochs_.resize(next_history_capacity_,0u);
        // Validate source ownership and determine host friction inputs together.
        std::size_t compact_anchor_count=0u;
        constexpr auto anchor_stride=sizeof(native::GpuDyFrictionAnchorInput);
        constexpr auto max_anchor_count=
            std::numeric_limits<std::uint32_t>::max()/anchor_stride;
        const bool early_retained=prewitness_retained_source_data_!=nullptr;
        if(early_retained &&
           (retained_candidates.data()!=prewitness_retained_source_data_ ||
            retained_candidates.size()!=prewitness_retained_source_count_))
            throw std::logic_error(
                "Pre-witness retained friction source changed before Dy admission");
        retained_patch_inputs_.clear();
        packed_retained_candidates_.clear();
        std::size_t selected_retained=0u;
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            if(manager.prep_index>=prep_count ||
               manager.patch_offset>patches.size() ||
               manager.patch_count>patches.size()-manager.patch_offset)
                throw std::logic_error(
                    "Source Dy contact preparation has invalid source bounds");
            const auto pair=contact_origin(manager.prep_index);
            const bool pcm=pair.pcm_pair_index!=0xffffffffu;
            if(pcm && (manager.patch_count!=1u ||
                       manager.patch_offset>=patches.size() ||
                       manager.contact_count!=0u ||
                       manager.order_offset!=0xffffffffu))
                throw std::logic_error(
                    "Source PCM numerical count must be device-authored");
            const auto capacity=pcm
                ?patches[manager.patch_offset].contact_count
                :manager.contact_count;
            const auto source_begin=pcm
                ?manager.source_order_offset:manager.order_offset;
            if(source_begin>contact_forces.size() ||
               capacity>contact_forces.size()-source_begin)
                throw std::logic_error(
                    "Source Dy Gu contact preparation range invalid");
            if(pcm) {
                if(pair.cpu_gu_offset!=0xffffffffu)
                    throw std::logic_error(
                        "Source PCM pair unexpectedly owns CPU Gu geometry");
                reused_pcm_rows+=capacity;
            } else {
                if(pair.cpu_gu_offset==0xffffffffu ||
                   pair.cpu_gu_offset>cpu_gu.size() ||
                   capacity>cpu_gu.size()-pair.cpu_gu_offset)
                    throw std::logic_error(
                        "Source CPU Gu contact inputs are incomplete");
                cpu_gu_rows+=capacity;
            }
            for(std::size_t i=manager.patch_offset;
                i<manager.patch_offset+manager.patch_count;++i) {
                auto& patch=patches[i];
                if(pcm && (patch.broken&0x80000000u)!=0u &&
                   patch.anchor_count!=0u &&
                   history_epochs_[pair.pcm_edge]!=history_epoch_)
                    patch.broken|=0x40000000u;
                if(patch.anchor_count>2u || patch.row_offset%2u ||
                   patch.row_offset/2u>anchors.size() ||
                   patch.anchor_count>anchors.size()-patch.row_offset/2u)
                    throw std::logic_error(
                        "Source GPU friction patch has invalid anchor rows");
                if(!pcm || (patch.broken&0x40000000u)!=0u) {
                    if(patch.anchor_count>max_anchor_count-compact_anchor_count)
                        throw std::overflow_error(
                            "Source GPU compact friction input count exceeds address space");
                    compact_anchor_count+=patch.anchor_count;
                }
                if(!retained_candidates.empty() && (patch.broken&0x10000000u)!=0u) {
                    const auto count=(patch.broken>>8u)&63u;
                    const auto first=patch.reserved;
                    if((patch.broken&0x80000000u)==0u ||
                       count==0u || count>32u ||
                       first>retained_candidates.size() ||
                       count>retained_candidates.size()-first)
                        throw std::logic_error(
                            "Device PCM retained candidates have invalid bounds");
                    const auto packed_first=early_retained
                        ?std::size_t(first):packed_retained_candidates_.size();
                    std::uint32_t prior=0xffffffffu;
                    for(std::size_t j=first;j<first+count;++j) {
                        const auto& candidate=retained_candidates[j];
                        if(candidate.source_ordinal>=32u ||
                           (prior!=0xffffffffu &&
                            candidate.source_ordinal<=prior) ||
                           candidate.anchor_count==0u ||
                           candidate.anchor_count>2u)
                            throw std::logic_error(
                                "Device PCM retained candidate is invalid");
                        prior=candidate.source_ordinal;
                        if(!early_retained)
                            packed_retained_candidates_.push_back(candidate);
                    }
                    selected_retained+=count;
                    retained_patch_inputs_.push_back({i,narrow(packed_first)});
                }
            }
        }
        // Device-admitted PCM source windows may contain non-touching
        // candidates. Only admitted responses use working rows.
        const auto source_rows=reused_pcm_rows+cpu_gu_rows;
        if(cpu_gu_rows!=cpu_gu.size() ||
           source_rows>contact_forces.size() ||
           ((!device_admission || !compact_contact_admission) &&
            source_rows!=contact_forces.size()))
            throw std::logic_error(
                "Source Gu contact preparation exceeds numerical row capacity");
        // Reuse resident Dy body inputs and share identical timestep
        // coefficients across the source contact-preparation records.
        if(prep_count>
           std::numeric_limits<std::uint32_t>::max()/
               sizeof(SourceDyPackedContactPrepNative))
            throw std::overflow_error("Source Dy contact prep descriptor overflow");
        const std::size_t prep_bytes=prep_count
            ?source_dy_contact_frame_bytes+
                prep_count*sizeof(SourceDyPackedContactPrepNative):0u;
        if(cpu_gu.size_bytes()>
           std::numeric_limits<std::uint32_t>::max()-prep_bytes)
            throw std::overflow_error(
                "Source Gu contact-prep arena exceeds address space");
        const std::size_t static_begin=prep_bytes+cpu_gu.size_bytes();
        packed_.resize(prep_count);
        inline_bodies_.clear();
        if(++inline_body_epoch_==0u) {
            std::fill(inline_body_slots_.begin(),inline_body_slots_.end(),
                      InlineBodySlot{});
            inline_body_epoch_=1u;
        }
        assigned_.assign(prep_count,0u);
        bool pending_device_contact_references=false;
        const auto ref_body=[&](const native::GpuDyContactPrepBody& body,
                                std::uint32_t ordinal,
                                BodyId actor) -> std::uint32_t {
            if(ordinal!=invalid && body.solver_ordinal!=invalid &&
               ordinal!=body.solver_ordinal)
                throw std::logic_error(
                    "Source Dy descriptor disagrees with prepared contact body");
            if(ordinal==invalid)ordinal=body.solver_ordinal;
            if(ordinal==native::gpu_dy_unresolved_solver_ordinal) {
                if(!device_admission || !device_world_motion ||
                   device_admission->deferred_body_remap ||
                   !contact_endpoints_pipeline_ || actor==invalid ||
                   !std::isfinite(body.inverse_mass) || body.inverse_mass<0.f)
                    throw std::logic_error(
                        "Device contact source cannot defer its numerical body");
                // The device patches this tagged reference before PGS.
                pending_device_contact_references=true;
                return 0xfffffffdu;
            }
            // An awake zero-inverse-mass dynamic still has source motion.
            // GPU preintegration owns it, unlike static/kinematic inline
            // participants whose velocities Sc prepares independently.
            const bool active_dynamic=ordinal!=invalid &&
                ordinal<integration.size() &&
                (integration[ordinal].active&1u)!=0u;
            if(active_dynamic) {
                if(ordinal>=bodies.size() || ordinal>=invalid/4u ||
                   body.inverse_mass!=(device_world_motion
                       ?integration[ordinal].source_inverse_mass
                       :bodies[ordinal].inverse_mass) ||
                   (device_world_motion && actor!=invalid &&
                    integration[ordinal].source_actor!=actor))
                    throw std::logic_error(
                        "Active source Dy contact body has no matching solver row");
                return (ordinal<<2u)|1u;
            }
            if(body.inverse_mass>0.f)
                throw std::logic_error(
                    "Responding source Dy contact body has no solver ordinal");
            // Fixed and kinematic actors use the staged actor-local pool.
            if(device_world_motion && actor!=invalid &&
               actor<actor_state_.size() &&
               (actor_state_[actor]&actor_metadata_valid) &&
               actor_local_host_[actor].inverse_mass==0.0f) {
                if(actor>=invalid/4u)
                    throw std::overflow_error(
                        "Resident contact actor source exceeds tagged address space");
                ++last_schedule_.resident_immovable_contact_bodies;
                return (actor<<2u)|3u;
            }
            if(!compact_contact_prep.empty())
                throw std::logic_error(
                    "Scene Dy contact source omitted its resident actor-local input");
            if(actor!=invalid && actor<inline_body_slots_.size()) {
                const auto& slot=inline_body_slots_[actor];
                if(slot.epoch==inline_body_epoch_) {
                    const auto& saved=inline_bodies_[slot.index];
                    auto expected=body;
                    expected.solver_ordinal=ordinal;
                    if(std::memcmp(&saved,&expected,sizeof(expected))!=0)
                        throw std::logic_error(
                            "Source Dy immovable actor has inconsistent contact inputs");
                    return narrow(static_begin+slot.index*
                        sizeof(native::GpuDyContactPrepBody));
                }
            }
            if(inline_bodies_.size()>=(
                std::numeric_limits<std::uint32_t>::max()-static_begin)/
                    sizeof(native::GpuDyContactPrepBody))
                throw std::overflow_error("Source Dy inline contact body overflow");
            const auto offset=static_begin+inline_bodies_.size()*
                sizeof(native::GpuDyContactPrepBody);
            inline_bodies_.push_back(body);
            inline_bodies_.back().solver_ordinal=ordinal;
            if(actor!=invalid) {
                if(actor>=inline_body_slots_.size())
                    inline_body_slots_.resize(std::size_t(actor)+1u);
                inline_body_slots_[actor]={
                    inline_body_epoch_,
                    static_cast<std::uint32_t>(inline_bodies_.size()-1u)};
            }
            return narrow(offset); // An aligned offset; low bit clear.
        };
        const auto pack_manager=[&](const native::GpuDyContactManifold& manager,
                                std::uint32_t row0,std::uint32_t row1) {
            if(manager.prep_index>=prep_count ||
               assigned_[manager.prep_index]++)
                throw std::logic_error("Source Dy contact prep is not unique");
            auto& target=packed_[manager.prep_index];
            if(!compact_contact_prep.empty()) {
                const auto& source=compact_contact_prep[manager.prep_index];
                native::GpuDyContactPrepBody first{},second{};
                first.inverse_mass=source.inverse_mass0;
                second.inverse_mass=source.inverse_mass1;
                first.solver_ordinal=source.solver_ordinal0;
                second.solver_ordinal=source.solver_ordinal1;
                target.body0_source=ref_body(
                    first,row0,source.source_body0);
                target.body1_source=ref_body(
                    second,row1,source.source_body1);
                const auto* coeffs=
                    reinterpret_cast<const std::uint8_t*>(&source)+
                    offsetof(native::GpuDyCompactContactPrep,rest_distance);
                std::memcpy(target.constants.data(),coeffs,20u);
                std::memcpy(target.constants.data()+20u,coeffs+36u,20u);
            }else {
                const auto& source=contact_prep[manager.prep_index];
                target.body0_source=ref_body(
                    source.body0,row0,source.source_body0);
                target.body1_source=ref_body(
                    source.body1,row1,source.source_body1);
                const auto* coeffs=
                    reinterpret_cast<const std::uint8_t*>(&source)+
                    offsetof(native::GpuDyContactPrepPair,rest_distance);
                std::memcpy(target.constants.data(),coeffs,20u);
                std::memcpy(target.constants.data()+20u,coeffs+36u,20u);
            }
        };
        if(device_admission) {
            // Device contact tokens use stable source identities.
            // Responding references resolve from Sc's resident topology.
            for(const auto& manager:manifolds) {
                if(!manager.patch_count)continue;
                if(manager.prep_index>=prep_count)
                    throw std::logic_error("Device contact source has no preparation");
                const auto first=compact_contact_prep.empty()
                    ?contact_prep[manager.prep_index].body0.solver_ordinal
                    :compact_contact_prep[manager.prep_index].solver_ordinal0;
                const auto second=compact_contact_prep.empty()
                    ?contact_prep[manager.prep_index].body1.solver_ordinal
                    :compact_contact_prep[manager.prep_index].solver_ordinal1;
                pack_manager(manager,first,second);
            }
        }else {
            for(const auto& desc:descriptors) {
                if((desc.row_count&255u)!=0u)continue;
                if(desc.index>=manifolds.size())
                    throw std::logic_error("Source contact descriptor has no manager");
                const auto& manager=manifolds[desc.index];
                if(manager.patch_count)
                    pack_manager(manager,desc.row0,desc.row1);
            }
        }
        if(device_admission)
            for(std::size_t i=0u;i<packed_.size();++i)
                if(!assigned_[i])packed_[i]={};
        const auto compact_anchor_bytes=compact_anchor_count*anchor_stride;
        if(compact_anchor_bytes>
           std::numeric_limits<std::uint32_t>::max()-prep_bytes-
               cpu_gu.size_bytes())
            throw std::overflow_error(
                "Source GPU compact friction inputs exceed address space");
        // A manager excluded by the source island/partition pass has no
        // GPU reader; only scheduled descriptors require body references.
        const std::size_t inline_body_bytes=
            inline_bodies_.size()*sizeof(native::GpuDyContactPrepBody);
        const std::size_t retained_begin=static_begin+inline_body_bytes;
        if(retained_begin>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(
                "Retained GPU friction input exceeds address space");
        const auto stride=sizeof(native::GpuDyRetainedFrictionCandidate);
        const std::size_t retained_bytes=selected_retained*stride;
        const std::size_t retained_source_count=early_retained
            ?prewitness_retained_source_count_:packed_retained_candidates_.size();
        if(retained_source_count && !early_retained) {
            const auto bytes=source_allocation_bytes(retained_source_count,stride);
            if(bytes>max_view_ || bytes>std::numeric_limits<std::uint32_t>::max())
                throw std::out_of_range("Retained friction source exceeds device view");
            if(retained_frame_input_.direct())
                retained_source_address_=retained_frame_input_.write(
                    packed_retained_candidates_.data(),bytes);
            else {
                retained_sources_.allocate(bytes,PX_FL);
                retained_source_address_=retained_sources_.getDevicePtr();
                auto& uploads=stream_.upload_scratch();
                uploads.push_back({
                    retained_source_address_,
                    packed_retained_candidates_.data(),bytes});
                stream_.queue_uploads(uploads);
            }
        }
        const bool direct=retained_frame_input_.direct();
        last_schedule_.retained_friction_source_bytes=retained_bytes;
        last_schedule_.prewitness_retained_friction_upload_bytes=
            early_retained && !direct?prewitness_retained_source_count_*stride:0u;
        last_schedule_.prewitness_retained_friction_direct_bytes=
            early_retained && direct?prewitness_retained_source_count_*stride:0u;
        last_schedule_.late_retained_friction_upload_bytes=
            !early_retained && !direct?packed_retained_candidates_.size()*stride:0u;
        last_schedule_.late_retained_friction_direct_bytes=
            !early_retained && direct?packed_retained_candidates_.size()*stride:0u;
        const std::size_t compact_prep_bytes=retained_begin;
        // Contact/friction prep and integrateCore share one actor world pose.
        // Sc retains the completed-frame mirror for publication and rollback.
        packed_integration_.clear();
        host_poses_.clear();
        last_resident_pose_count_=0u;
        last_schedule_.device_world_inertia_bodies=0u;
        last_schedule_.actor_local_upload_bytes=0u;
        last_schedule_.actor_local_upload_spans=0u;
        last_schedule_.reused_actor_local_poses=0u;
        last_schedule_.resident_velocity_reuses=0u;
        last_schedule_.prepared_velocity_updated_bodies=0u;
        last_schedule_.prepared_velocity_upload_bytes=0u;
        last_schedule_.force_mod_upload_bytes=0u;
        last_schedule_.free_velocity_readback_bytes=0u;
        pending_actor_locals_.clear();
        prepared_velocity_components_.clear();
        force_mod_components_.clear();
        packed_integration_.reserve(integration.size());
        prepared_velocity_components_.reserve(integration.size()*6u);
        force_mod_components_.reserve(integration.size()*12u);
        host_poses_.reserve(integration.size());
        reserve_actor_local_slots(needed_actor_slots);
        for(std::size_t ordinal=0u;ordinal<integration.size();++ordinal) {
            const auto& input=integration[ordinal];
            SourceDyPackedIntegrationNative record{};
            record.lock_flags=input.lock_flags;
            record.seconds=input.seconds;
            record.active=(input.active&1u)|
                (device_world_motion && (input.active&1u) ?(8u|128u):0u)|
                (device_world_motion && (input.active&32u) ?source_dy_device_inertia_flag:0u)|
                ((input.active&native::gpu_dy_conditional_integration)
                    ?source_dy_conditional_integration_flag:0u)|
                ((input.active&native::gpu_dy_wake_integration)
                    ?source_dy_wake_integration_flag:0u);
            if((record.active&source_dy_wake_integration_flag) &&
               !(record.active&source_dy_conditional_integration_flag))
                throw std::logic_error("Dormant Dy source lacks conditional integration");
            if((record.active&source_dy_conditional_integration_flag) &&
               (!source_indexed_integration || ordinal>=bodies.size() ||
                !(input.active&1u)))
                throw std::logic_error("Conditional Dy integration lacks a Sc source");
            // Bits [8:13] are exact authored velocity-change components;
            // Bits 14 and 15 select device inertia and conditional integration.
            if((record.active&source_dy_device_inertia_flag)!=0u)
                ++last_schedule_.device_world_inertia_bodies;
            record.source_element=input.source_element;
            record.source_shape_count=input.source_shape_count;
            record.counted_interactions=input.counted_interactions;
            if(device_unconstrained_preparation && (input.active&1u))
                record.active|=64u;
            if(input.active&1u) {
                const auto id=device_world_motion
                    ?input.source_actor:narrow(ordinal);
                record.actor_local_offset=id;
                auto& state=actor_state_[id];
                const bool resident_valid=(state&actor_metadata_valid)!=0u;
                const bool resident_velocity_valid=(state&actor_velocity_valid)!=0u;
                SourceDyActorLocalNative local{};
                std::memcpy(local.position,input.mass_local_position,
                            sizeof(local.position));
                std::memcpy(local.orientation,input.mass_local_orientation,
                            sizeof(local.orientation));
                local.inverse_mass=input.source_inverse_mass;
                local.linear_damping=input.linear_damping;
                local.angular_damping=input.angular_damping;
                local.max_linear_speed_squared=input.max_linear_speed_squared;
                local.max_angular_speed_squared=input.max_angular_speed_squared;
                std::memcpy(local.inverse_inertia_local,input.inverse_inertia_local,
                            sizeof(local.inverse_inertia_local));
                local.dynamics_flags=input.dynamics_flags;
                // Only the immutable mass-frame prefix participates in
                // metadata invalidation; Dy owns the trailing velocities.
                constexpr auto metadata_bytes=
                    offsetof(SourceDyActorLocalNative,linear_velocity);
                if(!resident_valid ||
                   std::memcmp(&actor_local_host_[id],&local,metadata_bytes)!=0) {
                    std::memcpy(&actor_local_host_[id],&local,metadata_bytes);
                    state&=~actor_metadata_valid;
                    pending_actor_locals_.push_back({id,1u});
                }else if((state&actor_prewitness_metadata)==0u)
                    ++last_schedule_.reused_actor_local_poses;
                // Authored and restored poses re-seed only their actor slot.
                // Unchanged integrated poses remain the next device input.
                if(device_world_motion) {
                    const bool reused=(state&actor_pose_valid) &&
                        std::memcmp(actor_local_host_[id].world_position,
                            input.actor_position,
                            sizeof(local.world_position))==0 &&
                        std::memcmp(actor_local_host_[id].world_orientation,
                            input.actor_orientation,
                            sizeof(local.world_orientation))==0;
                    if(reused) {
                        if((state&actor_prewitness_pose)==0u)
                            ++last_resident_pose_count_;
                    }else {
                        std::memcpy(actor_local_host_[id].world_position,
                                    input.actor_position,
                                    sizeof(local.world_position));
                        std::memcpy(actor_local_host_[id].world_orientation,
                                    input.actor_orientation,
                                    sizeof(local.world_orientation));
                        state&=~actor_metadata_valid;
                        pending_actor_locals_.push_back({id,8u});
                    }
                }
                std::uint32_t changed=0u;
                const bool prior=device_world_motion &&
                    (state&actor_velocity_valid);
                const auto& resident=actor_local_host_[id];
                for(unsigned axis=0u;axis<6u;++axis) {
                    const float* prepared=axis<3u
                        ?input.linear_velocity+axis
                        :input.angular_velocity+(axis-3u);
                    const float* previous=axis<3u
                        ?resident.linear_velocity+axis
                        :resident.angular_velocity+(axis-3u);
                    if(!prior || std::memcmp(prepared,previous,sizeof(float))!=0)
                        changed|=1u<<axis;
                }
                if(changed==0u) {
                    ++last_schedule_.resident_velocity_reuses;
                }else {
                    record.active|=changed<<8u;
                    record.velocity_offset=narrow(
                        prepared_velocity_components_.size());
                    for(unsigned axis=0u;axis<6u;++axis)
                        if(changed&(1u<<axis))
                            prepared_velocity_components_.push_back(
                                axis<3u?input.linear_velocity[axis]:
                                         input.angular_velocity[axis-3u]);
                    ++last_schedule_.prepared_velocity_updated_bodies;
                }
                // Dy retains sleep accumulators; Sc may also edit wake
                // state between device frames and must re-authorize it.
                auto& cached=actor_local_host_[id];
                constexpr auto sleep_bytes=28u;
                if(!resident_valid ||
                   std::memcmp(&cached.wake_counter,
                               &input.wake_counter,sleep_bytes)!=0) {
                    std::memcpy(&cached.wake_counter,
                                &input.wake_counter,sleep_bytes);
                    state&=~actor_metadata_valid;
                    pending_actor_locals_.push_back({id,4u});
                }
                // Every published body must explicitly reestablish its
                // committed velocity after a successful device solve.
                state&=~actor_velocity_valid;
                if(!resident_valid && !resident_velocity_valid)
                    pending_actor_locals_.push_back({id,15u});
                if(device_unconstrained_preparation) {
                    record.active|=32u;
                    const float* streams[4]{
                        input.pending_linear_velocity_delta,
                        input.pending_linear_acceleration,
                        input.pending_angular_velocity_delta,
                        input.pending_angular_acceleration};
                    std::uint32_t mask=0u;
                    for(unsigned j=0u;j<12u;++j)
                        if(std::bit_cast<std::uint32_t>(streams[j/3u][j%3u])!=0u)
                            mask|=1u<<j;
                    record.active|=mask<<16u;
                    if(mask) {
                        record.force_offset=narrow(force_mod_components_.size());
                        for(unsigned j=0u;j<12u;++j)
                            if(mask&(1u<<j))
                                force_mod_components_.push_back(
                                    streams[j/3u][j%3u]);
                    }
                }
            }else if(input.active&16u) {
                // Infinite-mass kinematic joints contribute current
                // velocity without creating a GPU-integrated actor.
                record.active=16u|(63u<<8u);
                record.velocity_offset=narrow(
                    prepared_velocity_components_.size());
                prepared_velocity_components_.insert(
                    prepared_velocity_components_.end(),
                    input.linear_velocity,input.linear_velocity+3u);
                prepared_velocity_components_.insert(
                    prepared_velocity_components_.end(),
                    input.angular_velocity,input.angular_velocity+3u);
                ++last_schedule_.prepared_velocity_updated_bodies;
            }
            if((input.active&1u) && device_world_motion) {
                // The BodyId pool is the pose source for every native rigid body.
                record.pose_index=invalid;
            }else if((input.active&1u) && (input.active&2u)) {
                if(!record.source_shape_count ||
                   record.source_element==invalid)
                    throw std::logic_error("Source Dy resident pose lacks Sc ElementSim");
                record.pose_index=invalid;
                ++last_resident_pose_count_;
            } else if(input.active&1u) {
                record.pose_index=narrow(host_poses_.size());
                SourceDyHostPoseNative pose{};
                std::memcpy(pose.position,input.position,sizeof(pose.position));
                std::memcpy(pose.orientation,input.orientation,sizeof(pose.orientation));
                host_poses_.push_back(pose);
            }
            packed_integration_.push_back(record);
        }
        constexpr std::size_t integration_header_bytes=44u;
        if(integration.size()>
           (std::numeric_limits<std::uint32_t>::max()-integration_header_bytes)/
               sizeof(SourceDyPackedIntegrationNative))
            throw std::overflow_error("Source Dy packed body inputs exceed address space");
        const std::size_t packed_input_bytes=integration_header_bytes+
            packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative);
        if(prepared_velocity_components_.size()>
           (std::numeric_limits<std::uint32_t>::max()-packed_input_bytes)/
               sizeof(float))
            throw std::overflow_error(
                "Source Dy prepared velocity updates exceed address space");
        const std::size_t velocity_bytes=prepared_velocity_components_.size()*
            sizeof(float);
        const std::size_t force_begin=packed_input_bytes+velocity_bytes;
        if(force_mod_components_.size()>
           (std::numeric_limits<std::uint32_t>::max()-force_begin)/
               sizeof(float))
            throw std::overflow_error(
                "Source Dy velocity modification stream exceeds address space");
        const std::size_t force_bytes=force_mod_components_.size()*
            sizeof(float);
        last_schedule_.force_mod_upload_bytes=force_bytes;
        const std::size_t host_pose_begin=force_begin+force_bytes;
        if(host_poses_.size()>
           (std::numeric_limits<std::uint32_t>::max()-host_pose_begin)/
               sizeof(SourceDyHostPoseNative))
            throw std::overflow_error("Source Dy authored poses exceed address space");
        const std::size_t mass_offset=host_pose_begin+
            host_poses_.size()*sizeof(SourceDyHostPoseNative);
        last_schedule_.prepared_velocity_upload_bytes=velocity_bytes;
        const bool upload_body_masses=device_initial_bodies &&
            !device_world_motion;
        if(upload_body_masses &&
           bodies.size()>(std::numeric_limits<std::uint32_t>::max()-
                          mass_offset)/sizeof(float))
            throw std::overflow_error(
                "Source Dy device solver-body masses exceed address space");
        const std::size_t mass_bytes=upload_body_masses
            ? bodies.size()*sizeof(float):0u;
        const std::size_t integration_bytes=mass_offset+mass_bytes;
        // Contact-only native rigid bodies derive their world inertia from
        // resident actor mass properties. Separate that GPU-written matrix
        // from the compact authored input without adding a dispatch.
        if(bodies.size()>
           std::numeric_limits<std::uint32_t>::max()/(9u*sizeof(float)))
            throw std::overflow_error("Source Dy inertia work exceeds address space");
        const std::size_t inertia_bytes=bodies.size()*9u*sizeof(float);
        last_schedule_.host_inertia_upload_bytes=
            device_generated_inertia?0u:inertia_bytes;
        last_schedule_.device_inertia_generated_bytes=
            device_generated_inertia?bodies.size()*9u*sizeof(float):0u;
        last_host_pose_upload_bytes_=host_poses_.size()*sizeof(SourceDyHostPoseNative);
        // Native Sc transfers source-order manager and joint identities.
        // Contact prep already encodes active body ordinals; device-owned
        // 1D geometry carries the same ordinals in its first row.
        const bool source_descriptors=device_partitions &&
            device_initial_bodies && device_world_motion;
        const auto lane_task=[](const native::GpuDyDistanceTask& task) {
            return task.descriptor_end==task.descriptor_begin+1u &&
                   task.partition_end==task.partition_begin+1u &&
                   task.body_end>task.body_begin &&
                   task.body_end-task.body_begin<=2u;
        };
        const bool host_lane_packing=source_descriptors && !device_admission &&
            task_capacity>1u && wide_tasks_.empty();
        last_schedule_.host_lane_packed_tasks=host_lane_packing
            ?std::count_if(tasks.begin(),tasks.end(),lane_task):0u;
        const bool packed_only=host_lane_packing &&
            last_schedule_.host_lane_packed_tasks==task_capacity;
        const auto raw_descriptor_count=source_descriptors && !packed_only
            ?descriptor_capacity:0u;
        const auto device_color_count=device_admission || !source_descriptors
            ?descriptor_capacity:0u;
        const auto resident_stage_words=resident_wide
            ?3ull*(descriptor_capacity/native::gpu_dy_cross_group_color_min):0ull;
        const auto color_body_count=packed_only?0u:bodies.size();
        const auto host_token_count=source_descriptors && !device_admission
            ?descriptor_capacity:0u;
        const std::size_t partition_header_bytes=source_descriptors
            ?(device_admission?64u:24u):20u;
        if(device_admission &&
           (!source_descriptors || !device_admission->work ||
            !device_admission->task_dispatch.buffer || task_capacity==0u))
            throw std::invalid_argument(
                "Source Dy device island admission requires resident numerical task launch");
        if(device_admission) {
            const auto& layout=device_admission->layout;
            if(compact_contact_admission &&
               (layout.contact_admissions>layout.bytes ||
                ((std::size_t(device_admission->contact_manager_capacity)+31u)/32u)*
                    sizeof(std::uint32_t)>
                    layout.bytes-layout.contact_admissions))
                throw std::out_of_range(
                    "Source Sc device contact admission exceeds its view");
            if(device_admission->deferred_body_remap &&
               (layout.body_remap<=layout.body_order ||
                layout.body_remap>layout.bytes ||
                layout.bytes-layout.body_remap<sizeof(std::uint32_t)))
                throw std::out_of_range(
                    "Dy deferred body remap exceeds the Sc work allocation");
            if(device_admission->deferred_joint_remap &&
               (layout.joint_remap<=layout.body_remap ||
                layout.joint_remap>layout.bytes ||
                device_admission->joint_source_count>
                    (layout.bytes-layout.joint_remap)/sizeof(std::uint32_t)))
                throw std::out_of_range(
                    "Dy deferred joint remap exceeds the Sc work allocation");
            if(layout.tasks>layout.block_offsets ||
               task_bytes>layout.block_offsets-layout.tasks ||
               layout.body_order>layout.contact_endpoints ||
               layout.contact_endpoints>layout.response_tokens ||
               layout.response_tokens>layout.bytes ||
               std::size_t(device_admission->contact_manager_capacity)*8u>
                   layout.response_tokens-layout.contact_endpoints ||
               ordinal_bytes>
                   layout.response_tokens-layout.body_order ||
               descriptor_capacity>
                   (layout.bytes-layout.response_tokens)/
                       sizeof(std::uint32_t))
                throw std::out_of_range(
                    "Source Sc resident body and response views exceed capacity");
        }
        source_descriptor_tokens_.clear();
        source_descriptor_full_.clear();
        if(source_descriptors) {
            if(!device_admission)
                source_descriptor_tokens_.reserve(descriptors.size());
            for(const auto& desc:descriptors) {
                if(desc.index>=0x40000000u)
                    throw std::overflow_error(
                        "Sc source descriptor index exceeds device token range");
                // Resident Sc admission supplies these source tokens.
                // The CPU still authors full 1D descriptors that are
                // not constructed from the device joint-row table.
                if((desc.row_count&255u)==0u) {
                    if(!device_admission)
                        source_descriptor_tokens_.push_back(
                            0xc0000000u|desc.index);
                }else if(rows[desc.index].reserved==2u) {
                    if(!device_admission)
                        source_descriptor_tokens_.push_back(
                            0x80000000u|desc.index);
                }else {
                    if(source_descriptor_full_.size()>=0x40000000u)
                        throw std::overflow_error(
                            "Dy source descriptor table exceeds device token range");
                    if(!device_admission)
                        source_descriptor_tokens_.push_back(
                            static_cast<std::uint32_t>(source_descriptor_full_.size()));
                    source_descriptor_full_.push_back(desc);
                }
            }
        }
        const std::uint64_t partition_input_bytes=device_partitions
            ?partition_header_bytes+
                std::uint64_t(host_token_count)*4ull+
                (source_descriptors
                    ?4ull+std::uint64_t(source_descriptor_full_.size())*16ull
                    :std::uint64_t(descriptor_capacity)*16ull)
            :0ull;
        const std::uint64_t partition_workspace_bytes=device_partitions
            ?std::uint64_t(device_color_count)*4ull+
                std::uint64_t(raw_descriptor_count)*16ull+
                resident_stage_words*4ull+
                std::uint64_t(color_body_count)*12ull
            :0ull;
        if(partition_input_bytes+partition_workspace_bytes>
           std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(
                "Device Dy partition storage exceeds address space");
        last_schedule_.device_source_descriptors=
            source_descriptors?descriptor_capacity:0u;
        last_schedule_.source_descriptor_upload_bytes=source_descriptors
            ?source_descriptor_tokens_.size()*sizeof(std::uint32_t)+
                source_descriptor_full_.size()*sizeof(native::GpuDyDistanceDescriptor)
            :descriptors.size_bytes();
        if(rows.size_bytes()>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Device Dy joint rows exceed 32-bit view");
        constexpr std::size_t count=18u;
        // The integration prefix describes Sc/SAP views and the launch
        // table. Solver-body ordinals precede the free-body suffix.
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),
            (packed_only?0u:descriptor_capacity)*sizeof(native::GpuDyDistanceDescriptor),
            (packed_only?0u:partition_count)*sizeof(std::uint32_t),
            device_admission?0u:ordinal_bytes,
            device_admission?0u:task_bytes,
            contact_bytes,friction_bytes,
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes(),
            compact_prep_bytes,
            integration_bytes,compact_anchor_bytes,
            tiles_.size()*sizeof(WorkTile),
            static_cast<std::size_t>(partition_input_bytes),inertia_bytes,
            static_cast<std::size_t>(partition_workspace_bytes)};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptor_capacity,partition_count,
            ordinal_capacity,task_capacity,contact_forces.size(),friction_count,
            patches.size(),manifolds.size(),contact_order.size(),
            prep_count,integration.size(),compact_anchor_count,
            tiles_.size(),device_partitions?1u:0u,bodies.size(),0u};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),
            device_partitions?nullptr:descriptors.data(),
            device_partitions?nullptr:partition_ends.data(),
            body_ordinals.data(),tasks.data(),nullptr,nullptr,
            patches.data(),
            manifolds.data(),contact_order.data(),
            contact_prep.data(),integration.data(),nullptr,tiles_.data(),
            nullptr,nullptr,nullptr};
        // Both full contact and friction solver rows live on device.
        // Sc returns normal impulses and compact per-anchor friction;
        // standalone conformance may return complete tangent/force rows.
        constexpr std::array<std::size_t,6> output_inputs{
            0u,5u,6u,7u,8u,12u};
        const std::array<std::size_t,6> output_strides{
            device_unconstrained_preparation?52u:48u,
            4u,4u,compact_friction?12u:16u,4u,32u};
        static_assert(offsetof(native::GpuDyJointRow,applied_force)==60u &&
                      offsetof(native::GpuDyDistanceTask,error)==32u &&
                      offsetof(native::GpuDyContactPoint,applied_force)==52u &&
                      offsetof(native::GpuDyFrictionRow,applied_force)==48u &&
                      offsetof(native::GpuDyContactPatch,broken)==12u);
        std::array<std::size_t,6> result_offsets{};
        std::size_t output_bytes=0u;
        for(std::size_t i=0;i<output_inputs.size();++i) {
            const auto n=i==1u && device_admission
                ?0u:i==3u && compact_friction
                    ?anchor_impulses.size():elements[output_inputs[i]];
            if(n>(std::numeric_limits<std::uint32_t>::max()-output_bytes)/
                output_strides[i])
                throw std::overflow_error("Source Dy writeback exceeds 32-bit view ABI");
            result_offsets[i]=output_bytes;
            output_bytes+=n*output_strides[i];
        }
        const std::size_t free_velocity_offset=output_bytes;
        const std::size_t free_velocity_bytes=
            device_unconstrained_preparation?free_bodies*24u:0u;
        const std::size_t free_sleep_bytes=
            device_unconstrained_preparation?free_bodies*28u:0u;
        if(free_bodies>(std::numeric_limits<std::uint32_t>::max()-
                       output_bytes)/52u)
            throw std::overflow_error(
                "Source GPU free-body velocity writeback exceeds 32-bit view");
        output_bytes+=free_velocity_bytes+free_sleep_bytes;
        if(resident_wide) {
            constexpr auto schedule_bytes=2u*sizeof(std::uint32_t);
            if(output_bytes>std::numeric_limits<std::uint32_t>::max()-schedule_bytes)
                throw std::overflow_error("Resident Dy schedule output exceeds view");
            output_bytes+=schedule_bytes;
        }
        // Source-indexed integration publishes by numerical address;
        // only remapped or diagnostic consumers need the ordinal list.
        const bool readback_body_ordinals=device_admission &&
            (!source_indexed_integration || readback_device_task_records);
        const std::size_t device_header_offset=output_bytes;
        const std::size_t device_task_offset=output_bytes+
            (device_admission?16u:0u);
        const std::size_t device_task_bytes=
            device_admission && readback_device_task_records
                ?task_bytes:0u;
        const std::size_t device_error_offset=device_task_offset+
            device_task_bytes;
        const std::size_t host_result_bytes=device_error_offset+
            (device_admission?8u:0u);
        if(device_admission &&
           (host_result_bytes<device_task_offset ||
            host_result_bytes>std::numeric_limits<std::uint32_t>::max() ||
            (readback_body_ordinals && ordinal_bytes>
                std::numeric_limits<std::uint32_t>::max()-host_result_bytes)))
            throw std::overflow_error(
                "Resident Sc Dy scheduling result exceeds address space");
        last_schedule_.device_task_readback_bytes=device_task_bytes;
        last_schedule_.device_epoch_error_readback_bytes=
            device_admission?8u:0u;
        last_schedule_.device_contact_admission_readback_bytes=
            compact_contact_admission
                ?((std::size_t(device_admission->contact_manager_capacity)+31u)/32u)*
                    sizeof(std::uint32_t):0u;
        last_schedule_.free_velocity_readback_bytes=free_velocity_bytes;
        last_schedule_.sleep_readback_bytes=
            device_unconstrained_preparation
                ?active_integration_count*28u:0u;
        // Both writable contact and friction arenas follow all source
        // inputs. The existing GPU preparation owns the complete rows.
        // Device coloring owns its workspace and sorted descriptors.
        // Place those ranges after the upload cutoff; compact anchors
        // must remain adjacent to contact rows.
        // Resident Sc tasks and body ordinals are GPU copy destinations,
        // not CPU packet inputs. Keep them beyond the host upload cutoff,
        // alongside Dy's other device-written numerical ranges.
        const std::array<std::size_t,count-2u> source_order=device_admission
            ? std::array<std::size_t,count-2u>{
                1u,8u,9u,10u,11u,12u,14u,15u,13u,6u,7u,0u,2u,3u,4u,5u}
            : device_initial_bodies
                ? std::array<std::size_t,count-2u>{
                1u,5u,8u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u,0u,2u,3u}
            : device_partitions
                ? std::array<std::size_t,count-2u>{
                    0u,1u,5u,8u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u,2u,3u}
                : std::array<std::size_t,count-2u>{
                    0u,1u,5u,8u,2u,3u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u};
        std::array<std::size_t,count> order{};
        std::size_t order_count=0u;
        for(const auto slot:source_order) {
            if(!device_generated_inertia && slot==13u)
                order[order_count++]=16u;
            order[order_count++]=slot;
        }
        if(device_generated_inertia)order[order_count++]=16u;
        order[order_count++]=17u;
        if(order_count!=count)
            throw std::logic_error("Source Dy arena placement omitted a view");
        last_schedule_.device_generated_bytes=device_partitions
            ? bytes[2]+bytes[3]+
                (device_admission
                    ?ordinal_bytes+task_bytes:0u):0u;
        std::array<std::size_t,count> offsets{};
        // The original Dy heap retains an EdgeIndex-keyed Sc anchor
        // prefix. Solver output follows it and remains the sole readback;
        // transient input and output locations may grow independently.
        // Internal raw-buffer tables need only 16-byte alignment.
        constexpr std::size_t table_alignment=16u;
        const auto align_table=[](std::size_t bytes) {
            return (bytes+table_alignment-1u)&~(table_alignment-1u);
        };
        const std::size_t upload_begin=history_bytes+align_table(output_bytes);
        std::size_t padding_bytes=upload_begin-history_bytes-output_bytes;
        std::size_t total=upload_begin;
        for(const auto i:order) {
            if((i==14u || i==15u || i==17u) && !bytes[i])continue;
            if(packed_only && (i==2u || i==3u))continue;
            const auto view_bytes=std::max<std::size_t>(bytes[i],4u);
            if(total>std::numeric_limits<std::uint32_t>::max()-table_alignment ||
               view_bytes>std::numeric_limits<std::uint32_t>::max()-total)
                throw std::overflow_error("Source Dy solver arena too large");
            offsets[i]=total;
            // Anchor input ends exactly at the writable contact arena.
            // GPU-only PCM can omit that trailing region while retaining
            // one contiguous source upload and the original dispatch.
            const auto next=i==13u?total+view_bytes:
                align_table(total+view_bytes);
            padding_bytes+=next-total-bytes[i];
            total=next;
        }
        narrow(total);
        const auto partition_output_offset=total;
        if(partition_output_count_>
           (std::numeric_limits<std::uint32_t>::max()-total)/sizeof(WorkTile))
            throw std::overflow_error("Device Dy partition tiles exceed arena");
        total+=partition_output_count_*sizeof(WorkTile);
        const auto mode_output_offset=total;
        if(resident_wide) {
            // Per task: current mode, retained descriptor/body ranges,
            // and the exact end of the conflict partition stream.
            constexpr std::size_t schedule_stride=24u;
            if(total>std::numeric_limits<std::uint32_t>::max() ||
               task_capacity>(std::numeric_limits<std::uint32_t>::max()-total)/
                    schedule_stride)
                throw std::overflow_error("Resident Dy task schedule exceeds arena");
            total+=task_capacity*schedule_stride;
        }
        narrow(total);
        last_schedule_.numerical_arena_bytes=total;
        last_schedule_.numerical_arena_padding_bytes=padding_bytes;
        // The contact shader addresses anchors backward from contactBias.
        // Host staging stops at the writable contact arena.
        const auto input_end=offsets[6];
        last_schedule_.omitted_partition_scratch_bytes=bytes[17];
        const std::size_t uploaded_bytes=input_end-upload_begin;
        // Numerical input is packed directly into transfer memory; host_
        // owns only the unpublished numerical result.
        const bool contract_result=result_working_set_.should_contract(
            host_result_bytes,host_result_capacity_);
        if(host_result_bytes>host_result_capacity_ || contract_result) {
            const auto capacity=contract_result?host_result_bytes:
                std::max(host_result_bytes,std::min(
                    host_result_capacity_*2u,
                    std::size_t(std::numeric_limits<std::uint32_t>::max())));
            host_=std::make_unique_for_overwrite<std::uint8_t[]>(capacity);
            host_result_capacity_=capacity;
            result_working_set_.reset();
        }
        last_schedule_.host_staging_bytes=host_result_bytes;
        last_schedule_.host_result_capacity=host_result_capacity_;
        last_schedule_.mapped_input_bytes=uploaded_bytes;
        const bool contract_input=input_working_set_.should_contract(
            uploaded_bytes,input_upload_capacity_);
        if(uploaded_bytes>input_upload_capacity_ || contract_input ||
           (input_upload_buffer_ && stream_.buffer_in_use(input_upload_buffer_))) {
            const auto capacity=contract_input?uploaded_bytes:
                std::max(uploaded_bytes,input_upload_capacity_);
            const auto replacement=context_.create_buffer(
                capacity,compute::MemoryPlacement::Upload);
            try {
                if(input_upload_buffer_)stream_.retire_root(input_upload_buffer_);
            }catch(...) {
                context_.retire_buffer(replacement);
                throw;
            }
            input_upload_buffer_=replacement;
            input_upload_capacity_=capacity;
            input_upload_initialized_=0u;
            input_working_set_.reset();
        }
        struct InputMap {
            compute::Context& context;
            compute::BufferId buffer;
            std::uint8_t* data;
            ~InputMap() { if(data)context.unmap_buffer(buffer); }
            void close() { context.unmap_buffer(buffer); data=nullptr; }
        } input_map{context_,input_upload_buffer_,static_cast<std::uint8_t*>(
            context_.map_buffer(input_upload_buffer_,compute::HostMapAccess::Write))};
        if(uploaded_bytes>input_upload_initialized_)
            std::memset(input_map.data+input_upload_initialized_,0,
                        uploaded_bytes-input_upload_initialized_);
        input_upload_initialized_=uploaded_bytes;
        const auto write_input=[&](std::size_t offset,const void* data,
                                   std::size_t size) {
            if(!size)return;
            if(offset<upload_begin || offset>input_end || size>input_end-offset)
                throw std::out_of_range("Dy host input has no upload span");
            std::memcpy(input_map.data+(offset-upload_begin),data,size);
        };
        // Pack only host-owned anchors directly into the mapped input.
        // Their backward offsets address the adjacent contact-row arena.
        std::size_t anchor_cursor=0u;
        for(const auto& manager:manifolds) {
            if(!manager.patch_count)continue;
            const auto pair=contact_origin(manager.prep_index);
            const bool pcm=pair.pcm_pair_index!=invalid;
            for(std::size_t j=manager.patch_offset;
                j<manager.patch_offset+manager.patch_count;++j) {
                auto& patch=patches[j];
                if(!patch.anchor_count ||
                   (pcm && (patch.broken&0x40000000u)==0u))continue;
                const auto distance=narrow(
                    (compact_anchor_count-anchor_cursor)*anchor_stride);
                write_input(offsets[13]+anchor_cursor*anchor_stride,
                            anchors.data()+patch.row_offset/2u,
                            patch.anchor_count*anchor_stride);
                anchor_cursor+=patch.anchor_count;
                if(pcm)
                    std::memcpy(packed_[manager.prep_index].constants.data()+36u,
                                &distance,sizeof(distance));
                else patch.reserved=distance;
            }
        }
        for(std::size_t i=0;i<count;++i)
            if(i!=6u && i!=7u && i!=11u && i!=12u &&
               i!=13u && i!=15u && i!=16u && i!=17u && bytes[i] &&
               (!device_initial_bodies || i!=0u) &&
               (!device_partitions || (i!=2u && i!=3u)) &&
               (!device_admission || (i!=4u && i!=5u)))
                write_input(offsets[i],sources[i],bytes[i]);
        if(device_admission)
            for(std::size_t i=0u;i<manifolds.size();++i) {
                const auto& manager=manifolds[i];
                if(!manager.patch_count ||
                   manager.prep_index>=prep_count ||
                   contact_origin(manager.prep_index).pcm_pair_index!=invalid)
                    continue;
                const auto source=manager.order_offset;
                write_input(offsets[9]+
                    i*sizeof(native::GpuDyContactManifold)+
                    offsetof(native::GpuDyContactManifold,source_order_offset),
                    &source,sizeof(source));
            }
        if(prep_bytes) {
            write_input(offsets[11],contact_frame.data(),
                        source_dy_contact_frame_bytes);
            write_input(offsets[11]+source_dy_contact_frame_bytes,
                        packed_.data(),
                        packed_.size()*sizeof(SourceDyPackedContactPrepNative));
        }
        if(!cpu_gu.empty())
            write_input(offsets[11]+prep_bytes,
                        cpu_gu.data(),cpu_gu.size_bytes());
        if(!inline_bodies_.empty())
            write_input(offsets[11]+static_begin,
                        inline_bodies_.data(),inline_bodies_.size()*
                            sizeof(native::GpuDyContactPrepBody));
        last_schedule_.inline_contact_body_bytes=
            inline_bodies_.size()*sizeof(native::GpuDyContactPrepBody);
        if(device_partitions) {
            const auto n=descriptor_capacity;
            const auto workspace=bytes[17]?offsets[17]-offsets[15]:0u;
            const auto color_begin=source_descriptors && !device_admission
                ?partition_header_bytes:workspace;
            const auto body_state_begin=workspace+device_color_count*4u+
                raw_descriptor_count*16u;
            const auto stage_list_begin=body_state_begin+color_body_count*12u;
            // Host tokens alias color slots. Raw descriptors precede
            // endpoint coloring state; resident tiles retain stage ends.
            const std::array<std::uint32_t,5> scratch_header{
                narrow(partition_header_bytes),narrow(color_begin),
                narrow(stage_list_begin),narrow(body_state_begin),
                tiles_.empty()?0u:narrow(offsets[14])};
            write_input(offsets[15],scratch_header.data(),
                        sizeof(scratch_header));
            if(source_descriptors) {
                // Sc's response tokens follow its body ordinals in the
                // same allocation; Dy writes only its own color scratch.
                const auto token_offset=device_admission
                    ?narrow(device_admission->layout.response_tokens-
                            device_admission->layout.body_order):0u;
                write_input(offsets[15]+20u,
                            &token_offset,sizeof(token_offset));
                if(device_admission) {
                    const auto task_offset=narrow(
                        device_admission->layout.tasks);
                    write_input(offsets[15]+24u,
                                &task_offset,sizeof(task_offset));
                    const auto extent_offset=narrow(
                        device_admission->layout.response_extents-
                        device_admission->layout.body_order);
                    write_input(offsets[15]+28u,
                                &extent_offset,sizeof(extent_offset));
                    const std::uint32_t retained_task_count=invalid;
                    write_input(offsets[15]+32u,
                                &retained_task_count,sizeof(retained_task_count));
                    const auto contact_prefix_offset=
                        device_admission->layout.device_contact_prefix
                        ?narrow(device_admission->layout.response_prefixes-
                                device_admission->layout.body_order):0u;
                    write_input(offsets[15]+44u,
                                &contact_prefix_offset,
                                sizeof(contact_prefix_offset));
                    const auto body_remap_offset=
                        device_admission->deferred_body_remap
                        ?narrow(device_admission->layout.body_remap-
                                device_admission->layout.body_order):0u;
                    write_input(offsets[15]+48u,
                                &body_remap_offset,
                                sizeof(body_remap_offset));
                    const auto joint_remap_offset=
                        device_admission->deferred_joint_remap
                        ?narrow(device_admission->layout.joint_remap-
                                device_admission->layout.body_order):0u;
                    write_input(offsets[15]+52u,
                                &joint_remap_offset,
                                sizeof(joint_remap_offset));
                    const auto joint_count=
                        device_admission->joint_source_count;
                    write_input(offsets[15]+56u,
                                &joint_count,sizeof(joint_count));
                    const auto counts=source_indexed_integration
                        ?narrow(device_admission->layout.counts):0u;
                    write_input(offsets[15]+60u,
                                &counts,sizeof(counts));
                    const std::array<std::uint32_t,2> epoch_errors{
                        0u,std::numeric_limits<std::uint32_t>::max()};
                    write_input(offsets[15]+36u,
                                epoch_errors.data(),sizeof(epoch_errors));
                }
                if(n && !device_admission)
                    write_input(offsets[15]+scratch_header[1],
                                source_descriptor_tokens_.data(),
                                n*sizeof(std::uint32_t));
                const auto table=offsets[15]+partition_header_bytes+
                    host_token_count*4u;
                const auto total=narrow(source_descriptor_full_.size());
                if(table<offsets[15] ||
                   table-offsets[15]>bytes[15] ||
                   sizeof(total)+std::size_t(total)*
                       sizeof(native::GpuDyDistanceDescriptor)>
                       bytes[15]-(table-offsets[15]))
                    throw std::logic_error(
                        "Source Dy descriptor table exceeds scratch allocation");
                write_input(table,&total,sizeof(total));
                if(total)
                    write_input(table+4u,
                                source_descriptor_full_.data(),
                                source_descriptor_full_.size()*
                                    sizeof(native::GpuDyDistanceDescriptor));
            }else if(n)
                write_input(offsets[15]+partition_header_bytes,
                            descriptors.data(),descriptors.size_bytes());
        }
        if(!prepared_velocity_components_.empty())
            write_input(offsets[12]+packed_input_bytes,
                        prepared_velocity_components_.data(),velocity_bytes);
        if(!force_mod_components_.empty())
            write_input(offsets[12]+force_begin,
                        force_mod_components_.data(),force_bytes);
        if(!host_poses_.empty())
            write_input(offsets[12]+host_pose_begin,
                        host_poses_.data(),last_host_pose_upload_bytes_);
        if(upload_body_masses)
            for(std::size_t i=0;i<bodies.size();++i)
                write_input(offsets[12]+mass_offset+i*4u,
                            &bodies[i].inverse_mass,sizeof(float));
        // Preserve the EdgeIndex history prefix when replacing the
        // transient numerical working set.
        const auto live_history=friction_history_live_slots_*64u;
        if(arena_working_set_.should_contract(total,arena_.getSize())) {
            arena_.resizeCopyPrefixAsync(total,live_history,stream_,PX_FL);
            arena_working_set_.reset();
        }else {
            const auto previous_capacity=arena_.getSize();
            arena_.allocateCopyPrefixAsync(total,live_history,stream_,PX_FL);
            if(arena_.getSize()!=previous_capacity)arena_working_set_.reset();
        }
        last_schedule_.numerical_arena_capacity=arena_.getSize();
        friction_history_capacity_=next_history_capacity_;
        if(next_joint_capacity>joint_source_capacity_) {
            joint_sources_.allocateCopyPrefixAsync(
                next_joint_capacity*sizeof(native::GpuDyJointInput)+127u,
                joint_source_live_slots_*sizeof(native::GpuDyJointInput),
                stream_,PX_FL);
            joint_source_capacity_=next_joint_capacity;
        }
        const auto address=arena_.getDevicePtr();
        if(resident_wide) {
            // Retained endpoints, scatter positions, partition ends
            // and tiles occupy device-only ranges of this exact layout.
            const std::array<std::uint64_t,16> layout{
                address,offsets[15],offsets[17],offsets[2],offsets[3],offsets[14],
                partition_output_offset,mode_output_offset,task_capacity,
                descriptor_capacity,bodies.size(),wide_bounds.nodes,
                wide_bounds.stages,partition_output_count_,
                partition_dispatch_count_,total};
            const bool reuse=prior_schedule && layout==resident_schedule_layout_ &&
                partition_dispatch_buffer_ &&
                partition_dispatch_count_<=partition_dispatch_capacity_/16u;
            resident_schedule_layout_=layout;
            last_schedule_.resident_schedule_reuse_candidate=reuse;
            const auto retained_task_count=reuse?narrow(prior_tasks):invalid;
            write_input(offsets[15]+32u,
                        &retained_task_count,sizeof(retained_task_count));
        }
        // Adjacent Sc constraint slot edits share one DMA into the
        // resident Dy allocation, not the transient solver-row upload.
        joint_source_uploads_.clear();
        for(std::size_t i=0u;i<joint_source_late_upload_slots_.size();) {
            const auto first=joint_source_late_upload_slots_[i];
            auto last=first+1u;
            ++i;
            while(i<joint_source_late_upload_slots_.size() &&
                  joint_source_late_upload_slots_[i]==last) {
                ++last;
                ++i;
            }
            joint_source_uploads_.push_back({
                joint_sources_.getDevicePtr()+std::size_t(first)*
                    sizeof(native::GpuDyJointInput),
                joint_source_host_.data()+first,
                std::size_t(last-first)*sizeof(native::GpuDyJointInput)});
            last_schedule_.device_joint_metadata_upload_bytes+=
                std::size_t(last-first)*sizeof(native::GpuDyJointInput);
        }
        last_schedule_.device_joint_metadata_upload_spans+=
            joint_source_uploads_.size();
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
        std::array<std::uint32_t,11> integration_header{
            0u,0u,0u,shape_capacity,narrow(ordinal_capacity),
            narrow(descriptor_capacity),narrow(task_capacity),
            device_partitions?narrow(std::size_t(params[28])+offsets[15]):
                (tiles_.empty()?0u:narrow(std::size_t(params[28])+offsets[14])),
            narrow(integration.size()),
            device_initial_bodies
                ? (device_world_motion?invalid
                    :narrow(std::size_t(params[28])+offsets[12]+mass_offset))
                :0u,
            narrow(std::size_t(params[28])+offsets[16])};
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
        // Stable BodyIds index one actor-local and world-motion state pool.
        // The transient body stream carries only the per-actor byte offset.
        std::uint32_t actor_local_bias=0u;
        dispatch.buffers.push_back(source_native_view(
            addresses_,actor_local_capacity_
                ?actor_locals_.getDevicePtr():address,
            actor_local_capacity_
                ?actor_local_capacity_*sizeof(SourceDyActorLocalNative):total,
            alignment_,5u,compute::BufferAccess::Read,
            actor_local_bias,max_view_));
        // Packed immovable body sources are stable actor IDs until the
        // binding's byte offset is known. Resolve them in the final
        // upload packet; no inline pose or velocity payload is needed.
        for(std::size_t i=0u;i<packed_.size();++i) {
            const auto resolve=[&](std::uint32_t source,
                                   std::size_t field) {
                if((source&3u)!=3u)return;
                const auto actor=source>>2u;
                if(actor>=actor_state_.size() ||
                   !(actor_state_[actor]&actor_metadata_valid) ||
                   actor_local_host_[actor].inverse_mass!=0.0f)
                    throw std::logic_error(
                        "Source Dy immovable actor has no resident input");
                const std::uint64_t address=
                    std::uint64_t(actor_local_bias)+
                    std::uint64_t(actor)*sizeof(SourceDyActorLocalNative);
                if(address>std::numeric_limits<std::uint32_t>::max()-144u)
                    throw std::overflow_error(
                        "Dy immovable actor-local reference exceeds 32-bit view");
                const auto tagged=static_cast<std::uint32_t>(address)|3u;
                write_input(offsets[11]+
                    source_dy_contact_frame_bytes+
                    i*sizeof(SourceDyPackedContactPrepNative)+field,
                    &tagged,sizeof(tagged));
            };
            resolve(packed_[i].body0_source,0u);
            resolve(packed_[i].body1_source,4u);
        }
        if(convex_vertex_count)
            dispatch.buffers.push_back(source_native_view(
                addresses_,convex_vertices,
                source_allocation_bytes(convex_vertex_count,12u),
                alignment_,6u,compute::BufferAccess::Read,
                params[27],max_view_));
        // All Sc admission views share one device allocation. Dy writes
        // only transient task fields; Sc owns the input identities.
        std::uint32_t resident_ordinal_bias=0u;
        std::uint32_t resident_task_bias=0u;
        dispatch.buffers.push_back(source_native_view(
            addresses_,device_admission
                ?device_admission->work:address,
            device_admission?device_admission->layout.bytes:total,
            alignment_,convex_vertex_count?7u:6u,
            compute::BufferAccess::ReadWrite,
            resident_task_bias,max_view_));
        if(device_admission) {
            const auto base=std::size_t(resident_task_bias);
            resident_ordinal_bias=narrow(
                base+device_admission->layout.body_order);
            resident_task_bias=narrow(
                base+device_admission->layout.tasks);
        }
        std::uint32_t joint_source_bias=0u;
        dispatch.buffers.push_back(source_native_view(
            addresses_,joint_source_capacity_
                ?joint_sources_.getDevicePtr():address,
            joint_source_capacity_
                ?joint_source_capacity_*sizeof(native::GpuDyJointInput):total,
            alignment_,convex_vertex_count?8u:7u,
            compute::BufferAccess::Read,joint_source_bias,max_view_));
        std::uint32_t retained_bias=0u;
        if(retained_source_count) {
            const auto retained_view=source_native_view(
                addresses_,retained_source_address_,
                retained_source_count*sizeof(native::GpuDyRetainedFrictionCandidate),
                alignment_,convex_vertex_count?9u:8u,
                compute::BufferAccess::Read,retained_bias,max_view_);
            dispatch.buffers.push_back(retained_view);
            for(const auto& input:retained_patch_inputs_) {
                const auto index=std::size_t(input.candidate);
                if(index>=retained_source_count ||
                   std::size_t(retained_bias)+index*
                       sizeof(native::GpuDyRetainedFrictionCandidate)>
                       std::numeric_limits<std::uint32_t>::max())
                    throw std::out_of_range(
                        "Retained friction device source offset is invalid");
                const auto offset=static_cast<std::uint32_t>(
                    std::size_t(retained_bias)+index*
                        sizeof(native::GpuDyRetainedFrictionCandidate));
                write_input(offsets[8]+
                    input.patch*sizeof(native::GpuDyContactPatch)+
                    offsetof(native::GpuDyContactPatch,reserved),
                    &offset,sizeof(offset));
            }
        }else {
            std::uint32_t dummy_bias=0u;
            dispatch.buffers.push_back(source_native_view(
                addresses_,address,total,alignment_,
                convex_vertex_count?9u:8u,compute::BufferAccess::Read,
                dummy_bias,max_view_));
        }
        for(auto& record:packed_integration_) {
            if(record.actor_local_offset!=invalid) {
                const auto offset=std::size_t(actor_local_bias)+
                    std::size_t(record.actor_local_offset)*
                        sizeof(SourceDyActorLocalNative);
                record.actor_local_offset=narrow(offset);
            }
            if(record.velocity_offset!=invalid)
                record.velocity_offset=narrow(
                    std::size_t(params[28])+offsets[12]+packed_input_bytes+
                    std::size_t(record.velocity_offset)*sizeof(float));
            if(record.force_offset!=invalid)
                record.force_offset=narrow(
                    std::size_t(params[28])+offsets[12]+force_begin+
                    std::size_t(record.force_offset)*sizeof(float));
            if(record.pose_index!=invalid)
                record.pose_index=narrow(
                    std::size_t(params[28])+offsets[12]+host_pose_begin+
                    std::size_t(record.pose_index)*
                        sizeof(SourceDyHostPoseNative));
        }
        if(!packed_integration_.empty())
            write_input(offsets[12]+integration_header_bytes,
                        packed_integration_.data(),
                        packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative));
        if(!device_generated_inertia)
            for(std::size_t i=0u;i<bodies.size();++i)
                write_input(offsets[16]+i*9u*sizeof(float),
                            integration[i].sqrt_inverse_inertia,
                            9u*sizeof(float));
        // Joint source slots are stable; Dy rows retain task order.
        // Each first-row descriptor selects its source independently.
        joint_row_source_seen_.assign(joint_inputs.size(),0u);
        std::size_t joint_row_begin=0u;
        for(std::size_t i=0u;i<joint_inputs.size();++i) {
            if(joint_row_begin>=rows.size())
                throw std::logic_error(
                    "Source Dy joint input exceeds prepared row storage");
            const auto& first=rows[joint_row_begin];
            const auto source=std::bit_cast<std::uint32_t>(first.applied_force);
            if(first.reserved!=2u || source>=joint_inputs.size() ||
               joint_row_source_seen_[source]++)
                throw std::logic_error(
                    "Source Dy joint input lacks a unique first-row descriptor");
            const auto& input=joint_inputs[source];
            const auto kind=input.kind;
            const std::size_t count=kind==0u?1u:kind==2u?3u:
                kind==1u?6u:5u;
            if(kind>4u || count>rows.size()-joint_row_begin)
                throw std::logic_error(
                    "Source Dy joint input exceeds prepared row storage");
            const auto flags=std::bit_cast<std::uint32_t>(first.axis[2]);
            if((flags&255u)!=count || (flags&~0x3ffu)!=0u)
                throw std::logic_error(
                    "Source Dy joint input lacks its first-row descriptor");
            const auto address=narrow(std::size_t(joint_source_bias)+
                std::size_t(input.source_slot)*sizeof(native::GpuDyJointInput));
            write_input(offsets[1]+
                            joint_row_begin*sizeof(native::GpuDyJointRow)+
                            offsetof(native::GpuDyJointRow,applied_force),
                        &address,sizeof(address));
            joint_row_begin+=count;
        }
        if(!joint_inputs.empty() && joint_row_begin!=rows.size())
            throw std::logic_error(
                "Source Dy joint inputs omitted numerical rows");
        // Changed actor-local fields share contiguous source DMA spans.
        build_actor_local_uploads();
        for(const auto& upload:actor_local_uploads_)
            last_schedule_.actor_local_upload_bytes+=upload.bytes;
        last_schedule_.actor_local_upload_spans=actor_local_uploads_.size();
        write_input(offsets[12],
                    integration_header.data(),sizeof(integration_header));
        const auto previous_spans=stream_.submission_stats().upload_spans;
        const auto previous_batches=stream_.submission_stats().upload_batches;
        auto& uploads=stream_.upload_scratch();
        uploads.reserve(joint_source_uploads_.size()+actor_local_uploads_.size());
        uploads.insert(uploads.end(),joint_source_uploads_.begin(),
                       joint_source_uploads_.end());
        uploads.insert(uploads.end(),actor_local_uploads_.begin(),
                       actor_local_uploads_.end());
        input_map.close();
        const auto target=addresses_.resolve(address+upload_begin,uploaded_bytes);
        const compute::CopyCommand staged{input_upload_buffer_,0u,
            target.buffer,target.offset,uploaded_bytes};
        stream_.queue_uploads(uploads,std::span(&staged,1u));
        actor_local_live_slots_=std::max(actor_local_live_slots_,needed_actor_slots);
        if(!joint_source_dirty_.empty())
            joint_source_live_slots_=std::max(joint_source_live_slots_,
                std::size_t(joint_source_dirty_.back())+1u);
        for(const auto id:prewitness_actor_slots_)
            actor_state_[id]&=~actor_prewitness_mask;
        prewitness_joint_source_slots_.clear();
        prewitness_joint_source_upload_bytes_=0u;
        prewitness_joint_source_upload_spans_=0u;
        prewitness_actor_local_upload_bytes_=0u;
        prewitness_actor_local_upload_spans_=0u;
        prewitness_actor_slots_.clear();
        last_schedule_.device_upload_batches=
            stream_.submission_stats().upload_batches-previous_batches;
        // Sc and Dy use one ordered source stream. The admission header,
        // task, ordinal and response-token inputs remain in the same
        // transient Sc device allocation.
        if(device_admission) {
            const auto& view=*device_admission;
            const auto& layout=view.layout;
            const auto in_range=[&](std::size_t offset,std::size_t bytes) {
                return offset<=layout.bytes && bytes<=layout.bytes-offset;
            };
            if(!in_range(0u,16u) ||
               layout.tasks>layout.block_offsets ||
               layout.body_order>layout.response_tokens ||
               layout.response_tokens>layout.response_extents ||
               layout.response_extents>layout.response_prefixes ||
               layout.response_prefixes>layout.contact_admissions ||
               layout.contact_admissions>layout.bytes ||
               task_bytes>layout.block_offsets-layout.tasks ||
               ordinal_bytes>
                   layout.response_tokens-layout.body_order ||
               descriptor_capacity>
                   (layout.response_extents-layout.response_tokens)/
                       sizeof(std::uint32_t) ||
               descriptor_capacity>
                   (layout.response_prefixes-layout.response_extents)/
                       sizeof(std::uint32_t) ||
               (layout.device_contact_prefix &&
                descriptor_capacity>
                   (layout.contact_admissions-layout.response_prefixes)/
                       sizeof(std::uint32_t)))
                throw std::out_of_range(
                    "Source Sc device admission capacity exceeds its typed stream");
        }
        last_upload_spans_=stream_.submission_stats().upload_spans-
                           previous_spans-joint_source_uploads_.size()-
                           actor_local_uploads_.size();
        for(std::size_t i=0;i<11u;++i)
            params[i<6u?i:i+6u]=narrow(
                static_cast<std::size_t>(params[28])+offsets[i]);
        if(device_admission) {
            params[4]=resident_ordinal_bias;
            params[5]=resident_task_bias;
        }
        params[29]=narrow(
            static_cast<std::size_t>(params[28])+offsets[11]);
        params[11]=narrow(
            static_cast<std::size_t>(params[28])+offsets[12]);
        const auto parameter_storage=std::make_shared<compute::DispatchParameterStorage>();
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        const auto partition_output_bias=narrow(
            std::size_t(params[28])+partition_output_offset);
        const auto resident_mode_bias=resident_wide
            ?narrow(std::size_t(params[28])+mode_output_offset):params[23];
        if(resident_wide)params[23]=resident_mode_bias;
        if(partition_dispatch_count_) {
            if((device_admission && !resident_wide) ||
               partition_dispatch_count_>max_view_/16u ||
               partition_dispatch_count_>std::numeric_limits<std::uint32_t>::max()/16u)
                throw std::out_of_range("Device Dy partition dispatch extent exceeds view");
            const auto dispatch_bytes=partition_dispatch_count_*16u;
            if(dispatch_bytes>partition_dispatch_capacity_) {
                const auto next=context_.create_buffer(
                    dispatch_bytes,compute::MemoryPlacement::DeviceLocal);
                if(partition_dispatch_buffer_)
                    context_.retire_buffer(partition_dispatch_buffer_);
                partition_dispatch_buffer_=next;
                partition_dispatch_capacity_=dispatch_bytes;
            }
        }
        if(pending_device_contact_references) {
            if(!contact_endpoints_pipeline_ ||
               device_admission->contact_manager_capacity!=manifolds.size())
                throw std::logic_error(
                    "Sc device contact endpoints require a complete Dy source table");
            const auto& layout=device_admission->layout;
            std::uint32_t work_base=0u;
            compute::DispatchCommand refs{};
            refs.pipeline=contact_endpoints_pipeline_;
            refs.buffers.push_back(dispatch.buffers[0]);
            refs.buffers.push_back(source_native_view(
                addresses_,device_admission->work,layout.bytes,
                alignment_,1u,compute::BufferAccess::ReadWrite,
                work_base,max_view_));
            std::array<std::uint32_t,10> arguments{
                params[15],params[29],work_base,
                narrow(std::size_t(work_base)+layout.contact_endpoints),
                narrow(std::size_t(work_base)+layout.contact_admissions),
                narrow(manifolds.size()),narrow(prep_count),narrow(bodies.size()),
                0u,params[11]};
            const auto ref_access=[&](std::uint32_t slot,std::size_t offset,
                                      std::size_t size,compute::BufferAccess access) {
                if(size)
                    refs.accesses.push_back({slot,
                        {refs.buffers[slot].accessed_extent().offset+offset,size},
                        access});
            };
            using A=compute::BufferAccess;
            ref_access(0u,offsets[9],bytes[9],A::Read);
            ref_access(0u,offsets[11]+source_dy_contact_frame_bytes,
                prep_count*sizeof(SourceDyPackedContactPrepNative),A::ReadWrite);
            ref_access(0u,offsets[12]+integration_header_bytes,
                packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative),
                A::Read);
            ref_access(1u,0u,sizeof(std::uint32_t),A::ReadWrite);
            ref_access(1u,layout.contact_endpoints,
                manifolds.size()*2u*sizeof(std::uint32_t),A::Read);
            ref_access(1u,layout.contact_admissions,
                ((manifolds.size()+31u)/32u)*sizeof(std::uint32_t),A::Read);
            refs.share_resources();
            for(std::size_t first=0u;first<manifolds.size();) {
                const auto groups=std::min<std::size_t>(
                    max_groups,(manifolds.size()-first+127u)/128u);
                arguments[8]=narrow(first);
                refs.groups={narrow(groups),1u,1u};
                refs.append_parameters(parameter_storage,arguments);
                stream_.commands().emplace_back(refs);
                ++last_schedule_.dispatches;
                first+=groups*128u;
            }
        }
        const auto declare_phase_accesses=[&](compute::DispatchResources& declaration,
                                              Launch phase,bool packed) {
            using A=compute::BufferAccess;
            auto& accesses=declaration.accesses;
            const bool fused=phase==Launch::Fused;
            const bool free=phase==Launch::FreeBodies;
            const bool integrates=fused || free || phase==Launch::Integrate;
            const bool solves=fused || phase==Launch::Partition;
            const bool snapshots=fused || phase==Launch::Motion;
            const bool prepares=fused || phase==Launch::Prepare;
            const bool colors=phase==Launch::Color ||
                (fused && !packed && device_partitions &&
                 wide_tasks_.empty() && !resident_wide);
            const bool expands=source_descriptors && (colors || packed);
            const bool admits_extents=expands || (prepares && !source_descriptors);
            const bool initializes=phase==Launch::InitializeBodies ||
                (phase==Launch::Color && device_initial_bodies && !resident_wide) ||
                (colors && fused && device_initial_bodies &&
                 task_capacity==1u && bodies.size()<=128u);
            const bool publishes=fused || phase==Launch::Color ||
                phase==Launch::PublishErrors;
            const auto region=[&](std::uint32_t slot,std::size_t offset,
                                  std::size_t size,A access) {
                if(size)
                    accesses.push_back({slot,
                        {declaration.buffers[slot].accessed_extent().offset+offset,size},
                        access});
            };
            const auto arena_region=[&](std::size_t offset,std::size_t size,A access) {
                region(0u,offset,size,access);
            };
            const auto table=[&](std::size_t index,A access) {
                arena_region(offsets[index],bytes[index],access);
            };
            const auto binding=[&](std::uint32_t slot,A access) {
                accesses.push_back({slot,declaration.buffers[slot].accessed_extent(),access});
            };
            const std::uint32_t island_slot=convex_vertex_count?7u:6u;
            const auto sc_region=[&](std::size_t offset,std::size_t size,A access) {
                region(island_slot,offset,size,access);
            };
            const bool reads_ordinals=snapshots || (integrates && !free);
            const bool task_access=!free && phase!=Launch::InitializeBodies;
            const bool task_writes=task_access && phase!=Launch::PublishErrors;
            arena_region(offsets[12],
                initializes || prepares || integrates
                    ?bytes[12]:integration_header_bytes,A::Read);
            if(device_partitions) {
                if(colors)table(15u,A::ReadWrite);
                else {
                    arena_region(offsets[15],
                        packed?bytes[15]:partition_header_bytes,A::Read);
                    if(publishes && device_admission)
                        arena_region(offsets[15]+36u,8u,A::ReadWrite);
                }
            }
            if(device_admission) {
                const auto& layout=device_admission->layout;
                sc_region(0u,sizeof(SourceScDeviceIslandHeader),A::Read);
                if(initializes || free) {
                    sc_region(layout.roots,layout.counts-layout.roots,A::Read);
                    sc_region(layout.counts,layout.admissions-layout.counts,A::Read);
                }
                if(task_access)
                    sc_region(layout.tasks,task_bytes,
                        task_writes?A::ReadWrite:A::Read);
                if(expands)
                    sc_region(layout.response_tokens,
                        layout.contact_admissions-layout.response_tokens,A::Read);
                if(expands && device_admission->deferred_joint_remap)
                    sc_region(layout.joint_remap,
                        device_admission->joint_source_count*sizeof(std::uint32_t),
                        A::Read);
                if(initializes || reads_ordinals || colors) {
                    sc_region(layout.body_order,ordinal_bytes,A::Read);
                    if(device_admission->deferred_body_remap)
                        sc_region(layout.body_remap,
                                  layout.joint_remap-layout.body_remap,A::Read);
                }
            }else {
                if(task_access)table(5u,task_writes?A::ReadWrite:A::Read);
                if(reads_ordinals)table(4u,A::Read);
            }

            // Fused and packed launches combine these numerical phases.
            if(solves || snapshots)table(0u,A::ReadWrite);
            else if(initializes)table(0u,A::Write);
            else if(prepares || (integrates && !free))table(0u,A::Read);
            if(prepares || solves)table(1u,A::ReadWrite);
            else if(colors)table(1u,A::Read);
            if(!packed) {
                if(colors) {
                    table(2u,A::ReadWrite);
                    table(3u,A::ReadWrite);
                    table(17u,A::ReadWrite);
                }else {
                    if(prepares || solves)table(2u,A::Read);
                    if(solves)table(3u,A::Read);
                }
                if(prepares || solves || snapshots || (integrates && !free)) {
                    table(14u,A::Read);
                    arena_region(partition_output_offset,
                        partition_output_count_*sizeof(WorkTile),A::Read);
                }
            }
            if(solves) {
                table(6u,A::ReadWrite);
                table(7u,A::ReadWrite);
            }else if(prepares) {
                table(6u,A::Write);
                table(7u,A::Write);
            }
            if(admits_extents || prepares || solves) {
                table(8u,A::ReadWrite);
                table(9u,admits_extents?A::ReadWrite:A::Read);
                table(10u,expands && device_admission?A::ReadWrite:A::Read);
                table(11u,A::Read);
            }
            if(prepares) {
                table(13u,A::Read);
                if(pcm_pair_count)
                    arena_region(0u,next_history_capacity_*64u,A::ReadWrite);
            }
            if(initializes)table(16u,
                prepares || integrates?A::ReadWrite:A::Write);
            else if(prepares || (integrates && !free))table(16u,A::Read);
            if(colors && resident_wide)
                arena_region(mode_output_offset,task_capacity*24u,A::ReadWrite);
            else if(fused && resident_wide)
                arena_region(mode_output_offset,task_capacity*4u,A::Read);

            // Solver impulse and integration results occupy disjoint
            // streams. Free flight never accesses numerical solver rows.
            if(solves)
                arena_region(history_bytes+result_offsets[2],
                    result_offsets[5]-result_offsets[2],A::ReadWrite);
            else {
                if(admits_extents)
                    arena_region(history_bytes+result_offsets[2],
                        result_offsets[3]-result_offsets[2],A::Write);
                if(admits_extents || (prepares && pcm_pair_count))
                    arena_region(history_bytes+result_offsets[4],
                        result_offsets[5]-result_offsets[4],A::ReadWrite);
            }
            if(integrates) {
                if(!free || source_indexed_integration)
                    arena_region(history_bytes+result_offsets[0],
                        result_offsets[1]-result_offsets[0],A::Write);
                arena_region(history_bytes+result_offsets[5],
                    elements[12]*output_strides[5],A::Write);
                if(free)
                    arena_region(history_bytes+free_velocity_offset,
                        free_velocity_bytes+free_sleep_bytes,A::Write);
            }
            if(publishes && !device_admission)
                arena_region(history_bytes+result_offsets[1],
                    task_capacity*sizeof(std::uint32_t),A::Write);

            if((admits_extents || prepares) && pcm_pair_count)binding(1u,A::Read);
            if((initializes || prepares || integrates) && has_shape_targets) {
                binding(2u,integrates?A::ReadWrite:A::Read);
                if(integrates)binding(3u,A::Write);
                binding(4u,A::Read);
                if(integrates && convex_vertex_count)binding(6u,A::Read);
            }
            if((initializes || prepares || integrates) && actor_local_capacity_)
                binding(5u,initializes || integrates?A::ReadWrite:A::Read);
            if(prepares && !joint_inputs.empty())binding(island_slot+1u,A::Read);
            if(prepares && retained_source_count)binding(island_slot+2u,A::Read);
        };
        // Phase declarations are frame-local; launches own only their parameters.
        constexpr auto packed_phase=static_cast<std::size_t>(Launch::BuildPartitionTiles)+1u;
        std::array<std::shared_ptr<const compute::DispatchResources>,packed_phase+1u> phases{};
        const auto phase_resources=[&](Launch phase,bool packed) {
            auto& entry=phases[packed?packed_phase:static_cast<std::size_t>(phase)];
            if(!entry) {
                compute::DispatchResources declaration{dispatch.buffers,{}};
                const bool integrates=phase==Launch::Fused || phase==Launch::Integrate ||
                                      phase==Launch::FreeBodies;
                declaration.buffers[2].access=integrates ? compute::BufferAccess::ReadWrite
                                                     : compute::BufferAccess::Read;
                declaration.buffers[3].access=integrates ? compute::BufferAccess::Write
                                                     : compute::BufferAccess::Read;
                const bool writes_actor_velocities=integrates ||
                    phase==Launch::InitializeBodies ||
                    (phase==Launch::Color && device_initial_bodies && !resident_wide);
                declaration.buffers[5].access=writes_actor_velocities
                    ?compute::BufferAccess::ReadWrite
                    :compute::BufferAccess::Read;
                declare_phase_accesses(declaration,phase,packed);
                entry=std::make_shared<const compute::DispatchResources>(std::move(declaration));
            }
            return entry;
        };
        compute::DispatchCommand launch_command{};
        launch_command.pipeline=dispatch.pipeline;
        const auto launch=[&](Launch phase,TileRange range,std::uint32_t iteration=0u) {
            if(!range.count)return;
            if(resident_wide)params[23]=resident_mode_bias;
            launch_command.resources=phase_resources(phase,false);
            params[10]=(iteration<<4u)|static_cast<std::uint32_t>(phase)|
                (compact_friction?0x20000000u:0u)|
                (device_partitions?0x80000000u:0u)|
                (device_admission && phase!=Launch::FreeBodies
                    ?0x08000000u:0u)|
                (device_partitions && (!wide_tasks_.empty() || resident_wide) &&
                 phase!=Launch::Color ?0x40000000u:0u)|
                (resident_wide && phase!=Launch::FreeBodies?0x04000000u:0u);
            // Compact tasks have no cross-workgroup dependencies. Their
            // workgroup count is published by the Sc device prefix, so
            // dispatch directly from that result instead of recopying a
            // host task count into the launch command.
            if(device_admission && phase==Launch::Fused &&
               wide_tasks_.empty() && !resident_wide) {
                if(range.begin!=0u || range.count!=task_capacity ||
                   range.count>source_sc_indirect_task_limit(context_.device_info()))
                    throw std::out_of_range(
                        "Source device Sc task launch exceeds compact dispatch bounds");
                params[8]=0u;
                launch_command.groups={1u,1u,1u};
                launch_command.indirect=device_admission->task_dispatch;
                launch_command.append_parameters(parameter_storage,params);
                stream_.commands().emplace_back(launch_command);
                launch_command.indirect.reset();
                ++last_schedule_.dispatches;
                ++last_schedule_.device_island_indirect_dispatches;
                return;
            }
            const bool resident_tiles=resident_wide &&
                (phase==Launch::Fused || phase==Launch::Prepare ||
                 phase==Launch::Motion ||
                 phase==Launch::Integrate);
            if(resident_tiles ||
               (phase==Launch::Partition && partition_dispatch_count_)) {
                if(phase==Launch::Partition)params[10]|=0x10000000u;
                for(std::size_t group=0u;group<range.count;group+=max_groups) {
                    if(resident_wide && phase==Launch::Partition) {
                        // The device assigns the stage's tile base;
                        // each indirect chunk supplies its local offset.
                        params[8]=narrow(std::size_t(params[28])+offsets[14]+
                            range.begin*sizeof(WorkTile));
                        params[23]=narrow(group*sizeof(WorkTile));
                    }else params[8]=narrow(std::size_t(partition_output_bias)+
                        (range.begin+group)*sizeof(WorkTile));
                    launch_command.groups={1u,1u,1u};
                    launch_command.indirect=compute::DispatchIndirectArgs{
                        partition_dispatch_buffer_,
                        (range.dispatch_begin+group/max_groups)*16u};
                    launch_command.append_parameters(parameter_storage,params);
                    stream_.commands().emplace_back(launch_command);
                    ++last_schedule_.dispatches;
                    if(phase==Launch::Partition)
                        ++last_schedule_.device_partition_indirect_dispatches;
                    if(phase==Launch::Fused)
                        ++last_schedule_.device_island_indirect_dispatches;
                }
                launch_command.indirect.reset();
                return;
            }
            launch_command.indirect.reset();
            for(std::size_t group=0;group<range.count;) {
                const auto count=std::min<std::size_t>(max_groups,range.count-group);
                params[8]=narrow(range.begin+group);
                launch_command.groups={narrow(count),1u,1u};
                launch_command.append_parameters(parameter_storage,params);
                stream_.commands().emplace_back(launch_command);
                ++last_schedule_.dispatches;
                group+=count;
            }
        };
        const auto launch_packed=[&](std::size_t first,std::size_t count) {
            launch_command.resources=phase_resources(Launch::Fused,true);
            launch_command.indirect.reset();
            const auto maximum=std::min<std::size_t>(
                std::size_t(max_groups)*128u,0x00ffffffu);
            for(std::size_t offset=0u;offset<count;) {
                const auto task_count=std::min(maximum,count-offset);
                params[8]=narrow(first+offset);
                params[10]=(narrow(task_count)<<4u)|
                    static_cast<std::uint32_t>(Launch::Fused)|
                    0x10000000u|0x80000000u|
                    (compact_friction?0x20000000u:0u);
                launch_command.groups={narrow((task_count+127u)/128u),1u,1u};
                launch_command.append_parameters(parameter_storage,params);
                stream_.commands().emplace_back(launch_command);
                ++last_schedule_.dispatches;
                offset+=task_count;
            }
        };
        // The coloring dispatch reserves additional workgroups for
        // parallel body initialization, without another submission.
        // Small single-task scenes initialize within their fused group.
        // Large compact tasks and independent compact islands use a
        // separate ordered dispatch rather than serializing body writes.
        if(device_initial_bodies && !bodies.empty() &&
           wide_tasks_.empty() &&
           (task_capacity!=1u || bodies.size()>128u)) {
            const auto before=last_schedule_.dispatches;
            launch(Launch::InitializeBodies,
                {0u,(bodies.size()+127u)/128u});
            last_schedule_.device_initialization_dispatches=
                last_schedule_.dispatches-before;
        }
        if(device_partitions && (!wide_tasks_.empty() || resident_wide))
            launch(Launch::Color,{0u,task_capacity+
                (device_initial_bodies && !resident_wide
                    ? (bodies.size()+127u)/128u : 0u)});
        if(partition_dispatch_count_) {
            // Color ends now own the exact cross-workgroup tile ranges.
            auto build=dispatch;
            build.accesses.clear();
            const auto slot=convex_vertex_count?9u:8u;
            build.buffers[slot]={slot,partition_dispatch_buffer_,0u,
                partition_dispatch_count_*16u,compute::BufferAccess::ReadWrite};
            build.indirect.reset();
            build.buffers[2].access=compute::BufferAccess::Read;
            build.buffers[3].access=compute::BufferAccess::Read;
            build.buffers[5].access=compute::BufferAccess::Read;
            using A=compute::BufferAccess;
            const auto build_region=[&](std::uint32_t binding,std::size_t offset,
                                        std::size_t size,A access) {
                if(size)
                    build.accesses.push_back({binding,
                        {build.buffers[binding].accessed_extent().offset+offset,size},
                        access});
            };
            build_region(0u,offsets[12],integration_header_bytes,A::Read);
            build_region(0u,offsets[15],
                source_descriptors?(device_admission?64u:24u):20u,
                resident_wide?A::ReadWrite:A::Read);
            build_region(0u,offsets[3],bytes[3],A::Read);
            build_region(0u,offsets[14],bytes[14],
                resident_wide?A::ReadWrite:A::Read);
            build_region(0u,partition_output_offset,
                partition_output_count_*sizeof(WorkTile),A::Write);
            build_region(slot,0u,partition_dispatch_count_*16u,
                resident_wide?A::ReadWrite:A::Write);
            if(resident_wide) {
                const auto& layout=device_admission->layout;
                const auto island_slot=convex_vertex_count?7u:6u;
                build_region(island_slot,0u,sizeof(SourceScDeviceIslandHeader),A::Read);
                build_region(island_slot,layout.tasks,task_bytes,A::ReadWrite);
                build_region(0u,mode_output_offset,task_capacity*24u,A::Read);
                const auto stage_bytes=static_cast<std::size_t>(
                    resident_stage_words*sizeof(std::uint32_t));
                build_region(0u,offsets[17]+bytes[17]-stage_bytes,stage_bytes,A::Read);
                build_region(0u,history_bytes+output_bytes-8u,8u,A::Write);
            }else
                build_region(0u,offsets[5],bytes[5],A::ReadWrite);
            auto arguments=params;
            arguments[7]=partition_output_bias;
            arguments[9]=max_groups;
            arguments[10]=static_cast<std::uint32_t>(Launch::BuildPartitionTiles);
            arguments[17]=narrow(partition_header_begin_);
            arguments[18]=narrow(partition_task_begin_);
            arguments[19]=narrow(wide_tasks_.size());
            const auto reservations=partition_tiles_.size()+(resident_wide?3u:0u);
            if(resident_wide) {
                arguments[10]|=0x0c000000u;
                arguments[18]=wide_bounds.stages;
                arguments[19]=narrow(task_capacity);
                arguments[21]=narrow(descriptor_capacity);
                arguments[24]=narrow(history_bytes+output_bytes-8u);
            }
            build.share_resources();
            for(std::size_t first=0u;first<reservations;first+=max_groups) {
                arguments[8]=narrow(first);
                build.groups={narrow(std::min<std::size_t>(
                    max_groups,reservations-first)),1u,1u};
                build.append_parameters(parameter_storage,arguments);
                stream_.commands().emplace_back(build);
                ++last_schedule_.dispatches;
            }
        }
        if(last_schedule_.host_lane_packed_tasks) {
            for(std::size_t first=0u;first<task_capacity;) {
                const bool packed=lane_task(tasks[first]);
                std::size_t end=first+1u;
                while(end<task_capacity && lane_task(tasks[end])==packed)
                    ++end;
                if(packed)launch_packed(first,end-first);
                else launch(Launch::Fused,{first,end-first});
                first=end;
            }
        }else launch(Launch::Fused,fused_tiles_);
        if(!wide_tasks_.empty() || resident_wide) {
            // The existing arena write dependencies order each dispatch
            // across workgroups. A relative partition from every wide
            // task can execute together because their body writes are disjoint.
            const auto before_preparation=last_schedule_.dispatches;
            launch(Launch::Prepare,prep_tiles_);
            last_schedule_.constraint_preparation_dispatches=
                last_schedule_.dispatches-before_preparation;
            for(std::uint32_t iteration=0u;iteration<wide_iterations_;++iteration) {
                for(const auto range:partition_tiles_)
                    if(iteration<range.iterations)
                        launch(Launch::Partition,range,iteration);
                if(!resident_wide && iteration<position_ends_.size() &&
                   position_ends_[iteration]) {
                    const auto before=last_schedule_.dispatches;
                    launch(Launch::Motion,body_tiles_,iteration);
                    last_schedule_.motion_capture_dispatches+=last_schedule_.dispatches-before;
                }
            }
            launch(Launch::Integrate,body_tiles_);
            launch(Launch::PublishErrors,{0u,(task_capacity+127u)/128u});
        }else if(device_admission) {
            // Even compact tasks must report a failed resident Sc epoch,
            // including an indirect launch with a mismatched task count.
            launch(Launch::PublishErrors,{0u,(task_capacity+127u)/128u});
        }
        // The integration extent fits in launchControl's upper bits
        // because the packed arena is limited to 32-bit byte offsets.
        const auto free_scan_count=source_indexed_integration
            ?integration.size():free_bodies;
        launch(Launch::FreeBodies,{0u,(free_scan_count+127u)/128u},
               narrow(integration.size()));
        const auto mapped=addresses_.resolve(address+history_bytes,output_bytes);
        compute::BufferId device_meta_buffer=0u,device_task_buffer=0u,
                          device_ordinal_buffer=0u;
        std::size_t device_meta_offset=0u,device_task_source=0u,
                    device_ordinal_source=0u;
        compute::BufferId device_response_buffer=0u;
        std::size_t device_response_source=0u;
        admitted_body_ordinals_.resize(
            readback_body_ordinals?ordinal_capacity:0u);
        admitted_response_tokens_.clear();
        admitted_contact_sources_.clear();
        if(device_admission) {
            const auto metadata=addresses_.resolve(
                device_admission->work,16u);
            device_meta_buffer=metadata.buffer;
            device_meta_offset=metadata.offset;
            if(device_task_bytes) {
                const auto device_tasks=addresses_.resolve(
                    device_admission->work+
                        device_admission->layout.tasks,device_task_bytes);
                device_task_buffer=device_tasks.buffer;
                device_task_source=device_tasks.offset;
            }
            if(readback_body_ordinals && ordinal_capacity!=0u) {
                const auto device_ordinals=addresses_.resolve(
                    device_admission->work+
                        device_admission->layout.body_order,
                    ordinal_bytes);
                device_ordinal_buffer=device_ordinals.buffer;
                device_ordinal_source=device_ordinals.offset;
            }
            if(descriptor_capacity || compact_contact_admission) {
                const auto& view=*device_admission;
                const auto bytes=compact_contact_admission
                    ?last_schedule_.device_contact_admission_readback_bytes
                    :descriptor_capacity*sizeof(std::uint32_t);
                if(bytes) {
                    const auto responses=addresses_.resolve(
                        view.work+(compact_contact_admission
                            ?view.layout.contact_admissions
                            :view.layout.response_tokens),bytes);
                    device_response_buffer=responses.buffer;
                    device_response_source=responses.offset;
                }
                if(compact_contact_admission)
                    admitted_contact_sources_.resize(
                        (std::size_t(view.contact_manager_capacity)+31u)/32u);
                else admitted_response_tokens_.resize(descriptor_capacity);
            }
        }
        // The final readback shares the numerical dispatch submission.
        // Neither its staging nor Sc's preceding witness DMA is joined.
        pending_solve_.emplace(PendingSolve{
            .bodies=bodies,.tasks=tasks,.contact_forces=contact_forces,
            .friction=friction,.anchor_impulses=anchor_impulses,
            .patches=patches,.integration=integration,
            .integrated_poses=integrated_poses,.sleep_states=sleep_states,
            .descriptors=descriptors,.body_ordinals=body_ordinals,
            .source_to_solver_row=device_admission
                ?device_admission->source_to_solver_row:
                 std::span<const std::uint32_t>{},
            .compact_contact_admission=compact_contact_admission,
            .source_indexed_integration=source_indexed_integration,
            .response_capacity=descriptor_capacity,
            .task_capacity=task_capacity,.ordinal_capacity=ordinal_capacity,
            .manifolds=manifolds,.contact_order=contact_order,
            .contact_prep=contact_prep,
            .compact_contact_prep=compact_contact_prep,
            .result_offsets=result_offsets,
            .output_bytes=output_bytes,
            .device_admission=device_admission.has_value(),
            .readback_device_task_records=readback_device_task_records,
            .readback_body_ordinals=readback_body_ordinals,
            .source_descriptors=source_descriptors,
            .device_header_offset=device_header_offset,
            .device_task_offset=device_task_offset,
            .device_error_offset=device_error_offset,
            .free_velocity_offset=free_velocity_offset,
            .friction_count=friction_count,
            .uploaded_bytes=uploaded_bytes,
            .cpu_gu_bytes=cpu_gu.size_bytes(),
            .reused_pcm_rows=reused_pcm_rows,
            .compact_anchor_bytes=compact_anchor_bytes,
            .prep_bytes=prep_bytes,.inline_body_bytes=inline_body_bytes,
            .compact_friction=compact_friction,
            .device_unconstrained_preparation=device_unconstrained_preparation,
            .device_world_motion=device_world_motion});
        std::array<compute::DownloadSpan,6u> downloads{};
        std::size_t download_count=0u;
        downloads[download_count++]={
            mapped.buffer,mapped.offset,host_.get(),output_bytes};
        if(device_admission) {
            downloads[download_count++]={
                device_meta_buffer,device_meta_offset,
                host_.get()+device_header_offset,16u};
            if(device_task_bytes)
                downloads[download_count++]={
                    device_task_buffer,device_task_source,
                    host_.get()+device_task_offset,device_task_bytes};
            if(readback_body_ordinals && ordinal_capacity!=0u)
                downloads[download_count++]={
                    device_ordinal_buffer,device_ordinal_source,
                    admitted_body_ordinals_.data(),ordinal_bytes};
            const auto errors=addresses_.resolve(
                address+offsets[15]+36u,8u);
            downloads[download_count++]={
                errors.buffer,errors.offset,
                host_.get()+device_error_offset,8u};
        }
        if(!admitted_contact_sources_.empty())
            downloads[download_count++]={
                device_response_buffer,device_response_source,
                admitted_contact_sources_.data(),
                admitted_contact_sources_.size()*sizeof(std::uint32_t)};
        else if(!admitted_response_tokens_.empty())
            downloads[download_count++]={
                device_response_buffer,device_response_source,
                admitted_response_tokens_.data(),
                admitted_response_tokens_.size()*sizeof(std::uint32_t)};
        if(defer_result_dma) {
            if(!extra_downloads.empty())
                throw std::logic_error("Deferred Dy readback already has PCM spans");
            deferred_result_downloads_.assign(
                downloads.begin(),downloads.begin()+download_count);
            stream_.submit();
        }else if(extra_downloads.empty())
            stream_.begin_numerical_fetch(
                std::span(downloads.data(),download_count));
        else {
            std::vector<compute::DownloadSpan> joined;
            joined.reserve(download_count+extra_downloads.size());
            joined.insert(joined.end(),downloads.begin(),
                          downloads.begin()+download_count);
            joined.insert(joined.end(),extra_downloads.begin(),
                          extra_downloads.end());
            stream_.begin_numerical_fetch(joined);
        }
        prewitness_retained_source_data_=nullptr;
        prewitness_retained_source_count_=0u;
        retained_source_address_=0u;
    }
    bool has_queued_solve() const noexcept {
        return pending_solve_.has_value();
    }
    bool has_deferred_result_dma() const noexcept {
        return !deferred_result_downloads_.empty();
    }
    void enqueue_deferred_result_dma(
            std::span<const compute::DownloadSpan> extra_downloads) {
        if(!pending_solve_ || deferred_result_downloads_.empty())
            throw std::logic_error("Source Dy result DMA has no submitted solve");
        deferred_result_downloads_.insert(deferred_result_downloads_.end(),
            extra_downloads.begin(),extra_downloads.end());
        stream_.begin_numerical_fetch(deferred_result_downloads_);
        deferred_result_downloads_.clear();
    }
    void finish() {
        if(!pending_solve_ || has_deferred_result_dma())
            throw std::logic_error("No queued source Dy numerical result DMA");
        // Sc owns the PCM witness publication. Numerical results can
        // finish after their device dependency without observing it.
        auto pending=std::move(*pending_solve_);
        pending_solve_.reset();
        const auto bodies=pending.bodies;
        auto tasks=pending.tasks;
        const auto contact_forces=pending.contact_forces;
        const auto friction=pending.friction;
        const auto anchor_impulses=pending.anchor_impulses;
        const auto patches=pending.patches;
        const auto integration=pending.integration;
        const auto integrated_poses=pending.integrated_poses;
        const auto sleep_states=pending.sleep_states;
        const auto descriptors=pending.descriptors;
        std::span<const std::uint32_t> body_ordinals=pending.body_ordinals;
        const auto manifolds=pending.manifolds;
        const auto contact_order=pending.contact_order;
        // Numerical sources share the pending solve's borrowed lifetime.
        const auto prep_count=pending.contact_prep.size()+
                              pending.compact_contact_prep.size();
        const auto contact_origin=[&](std::size_t index) {
            return source_contact_origin(index,pending.contact_prep,
                                         pending.compact_contact_prep);
        };
        const auto result_offsets=pending.result_offsets;
        const auto output_bytes=pending.output_bytes;
        const auto free_velocity_offset=pending.free_velocity_offset;
        const auto friction_count=pending.friction_count;
        const auto uploaded_bytes=pending.uploaded_bytes;
        const auto reused_pcm_rows=pending.reused_pcm_rows;
        std::size_t admitted_pcm_rows=0u;
        const auto compact_anchor_bytes=pending.compact_anchor_bytes;
        const auto prep_bytes=pending.prep_bytes;
        const auto inline_body_bytes=pending.inline_body_bytes;
        const auto compact_friction=pending.compact_friction;
        const auto device_unconstrained_preparation=
            pending.device_unconstrained_preparation;
        const auto device_world_motion=pending.device_world_motion;
        stream_.finish_numerical_fetch();
        bool invalid_device_admission=false;
        if(pending.device_admission) {
            std::array<std::uint32_t,4> admitted{};
            std::memcpy(admitted.data(),
                host_.get()+pending.device_header_offset,sizeof(admitted));
            invalid_device_admission=
                admitted[0]!=0u || admitted[1]>pending.task_capacity ||
                admitted[2]>pending.ordinal_capacity ||
                admitted[3]>pending.response_capacity;
            if(!invalid_device_admission) {
                last_schedule_.device_admitted_tasks=admitted[1];
                last_schedule_.device_admitted_bodies=admitted[2];
                last_schedule_.device_admitted_descriptors=admitted[3];
                if(last_schedule_.resident_wide_schedule) {
                    std::array<std::uint32_t,2> schedule{};
                    std::memcpy(schedule.data(),host_.get()+output_bytes-8u,8u);
                    const auto wide=schedule[0]&0x7fffffffu;
                    last_schedule_.resident_schedule_reused=(schedule[0]&0x80000000u)!=0u;
                    if(wide>admitted[1] || schedule[1]>admitted[1]-wide)
                        throw std::logic_error("Resident Dy task or tile count is invalid");
                    last_schedule_.wide_tasks=wide;
                    last_schedule_.resident_fused_workgroups=schedule[1];
                }
                if(pending.readback_device_task_records) {
                    if(admitted[1])
                        std::memcpy(tasks.data(),
                            host_.get()+pending.device_task_offset,
                            admitted[1]*sizeof(native::GpuDyDistanceTask));
                    for(std::size_t i=admitted[1];i<tasks.size();++i)
                        tasks[i].error=0u;
                    tasks=tasks.first(admitted[1]);
                }
                if(pending.readback_body_ordinals) {
                    admitted_body_ordinals_.resize(admitted[2]);
                    if(!pending.source_to_solver_row.empty())
                        for(auto& index:admitted_body_ordinals_) {
                            if(index>=pending.source_to_solver_row.size())
                                throw std::logic_error(
                                    "Source Dy GPU body source ordinal invalid");
                            index=pending.source_to_solver_row[index];
                            if(index>=bodies.size())
                                throw std::logic_error(
                                    "Source Dy GPU mapped solver row invalid");
                        }
                    body_ordinals=admitted_body_ordinals_;
                }else if(device_world_motion)
                    last_schedule_.device_world_motion_bodies=admitted[2];
            }
        }
        if(pending.device_admission) {
            std::array<std::uint32_t,2> errors{};
            std::memcpy(errors.data(),
                host_.get()+pending.device_error_offset,sizeof(errors));
            if(errors[0]!=0u)
                throw std::overflow_error(
                    "Source Dy GPU solver task "+std::to_string(errors[1])+
                    " rejected flags "+std::to_string(errors[0]));
        }else {
            for(std::size_t i=0;i<tasks.size();++i)
                std::memcpy(&tasks[i].error,
                    host_.get()+result_offsets[1]+i*4u,4u);
            for(std::size_t i=0u;i<tasks.size();++i)
                if(tasks[i].error!=0u)
                    throw std::overflow_error(
                        "Source Dy GPU solver task "+std::to_string(i)+
                        " rejected flags "+std::to_string(tasks[i].error));
        }
        if(invalid_device_admission)
            throw std::overflow_error(
                "Resident Sc Dy scheduling result exceeds declared capacities");
        const auto fetch_pose=[&](std::size_t index) {
            auto& pose=integrated_poses[index];
            std::memcpy(&pose,host_.get()+result_offsets[5]+
                index*sizeof(pose),sizeof(pose));
            if(pose.valid==native::gpu_dy_integration_inactive &&
               (integration[index].active&native::gpu_dy_conditional_integration))
                return;
            if(pose.valid!=1u)
                throw std::overflow_error("Source Dy GPU integrateCore rejected a body");
            ++last_integrated_body_count_;
        };
        const auto each_body_source=[&](auto&& visit) {
            if(pending.source_indexed_integration) {
                for(std::size_t index=0u;index<bodies.size();++index)
                    if(integration[index].active&1u)visit(index);
            }else for(const auto index:body_ordinals) {
                if(index>=bodies.size())
                    throw std::logic_error("Source Dy GPU body index invalid");
                visit(index);
            }
        };
        each_body_source([&](std::size_t index) {
            if(integration[index].active&1u)fetch_pose(index);
        });
        for(std::size_t index=bodies.size();index<integration.size();++index)
            fetch_pose(index);
        const auto each_body_result=[&](auto&& visit) {
            each_body_source([&](std::size_t index) {
                if(!(integration[index].active&native::gpu_dy_conditional_integration) ||
                   integrated_poses[index].valid==1u)
                    visit(index);
            });
        };
        if(device_unconstrained_preparation && pending.source_indexed_integration) {
            last_schedule_.device_preintegrated_bodies=last_integrated_body_count_;
            last_schedule_.device_sleep_bodies=last_integrated_body_count_;
        }
        if(device_unconstrained_preparation)
            for(std::size_t index=bodies.size();index<integration.size();++index) {
                auto& input=integration[index];
                const auto* output=host_.get()+free_velocity_offset+
                    (index-bodies.size())*52u;
                std::memcpy(input.linear_velocity,output,12u);
                std::memcpy(input.angular_velocity,output+12u,12u);
                std::memcpy(&sleep_states[index],output+24u,
                            sizeof(native::GpuDySleepState));
            }
        // The GPU task stream defines body writeback ownership; its
        // descriptor ordinals need not match the CPU's public Gu index
        // permutation. Source-manager identity owns contact/patch output.
        const auto publish_descriptor=[&](const native::GpuDyDistanceDescriptor& desc) {
            // Joint impulses remain local to the GPU PGS row.
            // No Sc task consumes a host copy after solve.
            if((desc.row_count&255u)!=0u)return;
            if(desc.index>=manifolds.size())
                throw std::logic_error("Source Dy manager writeback index invalid");
            const auto& manager=manifolds[desc.index];
            if(!manager.patch_count)return;
            if(manager.patch_offset>patches.size() ||
               manager.patch_count>patches.size()-manager.patch_offset)
                throw std::logic_error("Source Dy contact writeback range invalid");
            const bool resident_extent=pending.source_descriptors &&
                manager.patch_count==1u &&
                manager.prep_index<prep_count &&
                contact_origin(manager.prep_index).pcm_pair_index!=0xffffffffu;
            const std::size_t capacity=resident_extent
                ?patches[manager.patch_offset].contact_count
                :manager.contact_count;
            const auto source_begin=resident_extent
                ?manager.source_order_offset:manager.order_offset;
            if(source_begin>contact_forces.size() ||
               capacity>contact_forces.size()-source_begin)
                throw std::logic_error("Source Dy contact writeback range invalid");
            // PCM's patch reserves a window; its exact count comes
            // from the GPU result rather than the host manifold.
            std::size_t exact_contacts=capacity;
            if(resident_extent) {
                admitted_pcm_rows+=capacity;
                std::uint32_t report=0u;
                std::memcpy(&report,host_.get()+result_offsets[4]+
                            manager.patch_offset*4u,sizeof(report));
                if((report&source_dy_pcm_extent_valid_bit)==0u)
                    throw std::logic_error(
                        "Source Dy resident PCM contact extent was not published");
                exact_contacts=(report>>source_dy_pcm_extent_shift)&
                    source_dy_pcm_extent_count_mask;
                if(exact_contacts>capacity || exact_contacts>4u)
                    throw std::logic_error(
                        "Source Dy resident PCM contact extent exceeds capacity");
            }
            std::size_t written=0u;
            for(std::size_t patch_index=manager.patch_offset;
                patch_index<manager.patch_offset+manager.patch_count;
                ++patch_index) {
                auto& patch=patches[patch_index];
                constexpr std::uint32_t direct=0x80000000u;
                const bool resident=
                    manager.prep_index<prep_count &&
                    contact_origin(manager.prep_index).pcm_pair_index!=0xffffffffu;
                const std::size_t begin=resident
                    ?manager.source_order_offset:patch.order_offset&~direct;
                if(!resident && (patch.order_offset&direct)!=0u)
                    throw std::logic_error(
                        "Source CPU Gu patch unexpectedly has PCM contact order");
                const std::size_t limit=resident
                    ?contact_forces.size():contact_order.size();
                const auto count=resident_extent
                    ?exact_contacts:std::size_t(patch.contact_count);
                if(begin>limit || count>limit-begin ||
                   written>exact_contacts ||
                   count>exact_contacts-written)
                    throw std::logic_error(
                        "Source Dy contact patch order is invalid");
                for(std::size_t k=0u;k<count;++k) {
                    const auto index=resident
                        ?begin+k:contact_order[begin+k];
                    if(index<source_begin ||
                       index>=source_begin+exact_contacts)
                        throw std::logic_error(
                            "Source Dy contact writeback index invalid");
                    std::memcpy(&contact_forces[index],
                        host_.get()+result_offsets[2]+index*4u,4u);
                }
                if(resident_extent)
                    patch.contact_count=static_cast<std::uint32_t>(count);
                written+=count;
            }
            if(written!=exact_contacts)
                throw std::logic_error(
                    "Source Dy contact writeback omitted Gu rows");
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
                        host_.get()+result_offsets[4]+j*4u,4u);
                const auto count=pcm_growth?(report>>1u)&3u:
                    patch.anchor_count;
                if(count>2u || patch.row_offset>friction_count ||
                   count*2u>friction_count-patch.row_offset)
                    throw std::logic_error(
                        "Source Dy friction writeback range invalid");
                if(compact_friction) {
                    if(patch.row_offset&1u)
                        throw std::logic_error(
                            "Source Dy friction anchor range is misaligned");
                    for(std::size_t k=patch.row_offset/2u;
                        k<patch.row_offset/2u+count;++k)
                        std::memcpy(&anchor_impulses[k],
                            host_.get()+result_offsets[3]+k*12u,12u);
                }else {
                    for(std::size_t k=patch.row_offset;
                        k<patch.row_offset+count*2u;++k) {
                        const auto* result=
                            host_.get()+result_offsets[3]+k*16u;
                        std::memcpy(&friction[k].tangent,result,12u);
                        std::memcpy(&friction[k].applied_force,result+12u,4u);
                    }
                }
                if(pcm_growth) {
                    patch.anchor_count=count;
                    patch.reserved=report;
                    patch.broken=report&1u;
                }else if(count)patch.broken=report&1u;
            }
        };
        each_body_result([&](std::size_t index) {
            if(device_unconstrained_preparation) {
                const auto* output=host_.get()+result_offsets[0]+index*52u;
                std::memcpy(&bodies[index],output,24u);
                std::memcpy(&sleep_states[index],output+24u,
                            sizeof(native::GpuDySleepState));
            }else
                std::memcpy(&bodies[index],
                    host_.get()+result_offsets[0]+index*48u,48u);
        });
        if(pending.device_admission) {
            if(pending.compact_contact_admission) {
                // Device Sc exposes the original candidate ownership.
                // Numerical publication does not reconstruct an edge map.
                if(admitted_contact_sources_.size()!=
                    (manifolds.size()+31u)/32u)
                    throw std::logic_error(
                        "Resident Sc returned an invalid candidate admission mask");
                std::size_t admitted=0u;
                for(std::size_t word=0u;word<admitted_contact_sources_.size();
                    ++word) {
                    std::uint32_t bits=admitted_contact_sources_[word];
                    while(bits) {
                        const auto source=word*32u+std::countr_zero(bits);
                        if(source>=manifolds.size() ||
                           !manifolds[source].patch_count)
                            throw std::logic_error(
                                "Resident Sc contact admission has no prepared manager");
                        publish_descriptor({static_cast<std::uint32_t>(source),
                                            0u,0u,0u});
                        ++admitted;
                        bits&=bits-1u;
                    }
                }
                if(admitted>last_schedule_.device_admitted_descriptors)
                    throw std::logic_error(
                        "Resident Sc contact admission exceeds numerical responses");
            } else {
                // Diagnostic callers retain the full typed response view.
                for(std::size_t i=0u;
                    i<last_schedule_.device_admitted_descriptors;++i) {
                    const std::uint32_t token=admitted_response_tokens_[i];
                    const auto type=token>>30u;
                    if(type==0u || type==2u)continue;
                    if(type!=3u)
                        throw std::logic_error(
                            "Resident Sc returned an invalid response token");
                    const auto index=token&0x3fffffffu;
                    if(index>=manifolds.size())
                        throw std::logic_error(
                            "Resident Sc contact token exceeds its manager table");
                    publish_descriptor({index,0u,0u,0u});
                }
            }
        }else for(const auto& task:tasks)
            for(std::size_t i=task.descriptor_begin;
                i<task.descriptor_end;++i)
                publish_descriptor(descriptors[i]);
        // Commit all numerical comparison fields for an actor together.
        // Conditional inactive sources retain their previous device state.
        if(device_world_motion || device_unconstrained_preparation) {
            const auto mirror_actor=[&](std::size_t index,bool free_body) {
                const auto& input=integration[index];
                if(!(input.active&1u) &&
                   (free_body || !device_world_motion || !input.active))return;
                const auto id=input.source_actor;
                auto& resident=actor_local_host_[id];
                auto& state=actor_state_[id];
                if(device_world_motion) {
                    if(free_body) {
                        for(unsigned axis=0u;axis<3u;++axis)
                            resident.linear_velocity[axis]=
                                (input.lock_flags&(1u<<axis))
                                    ?0.0f:input.linear_velocity[axis];
                        std::memcpy(resident.angular_velocity,
                                    input.angular_velocity,
                                    sizeof(resident.angular_velocity));
                    }else {
                        std::memcpy(resident.linear_velocity,
                                    bodies[index].linear_delta,
                                    sizeof(resident.linear_velocity));
                        std::memcpy(resident.angular_velocity,
                                    bodies[index].angular_state,
                                    sizeof(resident.angular_velocity));
                    }
                    state|=actor_velocity_valid;
                }
                if(!(input.active&1u))return;
                if(device_unconstrained_preparation) {
                    const auto& sleep=sleep_states[index];
                    for(const auto component:sleep.linear_acc)
                        if(!std::isfinite(component))
                            throw std::overflow_error(
                                "Source GPU sleep linear accumulator is not finite");
                    for(const auto component:sleep.angular_acc)
                        if(!std::isfinite(component))
                            throw std::overflow_error(
                                "Source GPU sleep angular accumulator is not finite");
                    if(!std::isfinite(sleep.wake_counter) || sleep.wake_counter<0.0f)
                        throw std::overflow_error(
                            "Source GPU wake counter is not finite");
                    resident.wake_counter=sleep.wake_counter;
                    std::memcpy(resident.sleep_linear_acc,sleep.linear_acc,
                                sizeof(sleep.linear_acc));
                    std::memcpy(resident.sleep_angular_acc,sleep.angular_acc,
                                sizeof(sleep.angular_acc));
                }
                if(device_world_motion) {
                    const auto& pose=integrated_poses[index];
                    std::memcpy(resident.world_position,pose.position,
                                sizeof(resident.world_position));
                    std::memcpy(resident.world_orientation,pose.orientation,
                                sizeof(resident.world_orientation));
                    state|=actor_pose_valid;
                }
            };
            each_body_result([&](std::size_t index) { mirror_actor(index,false); });
            for(std::size_t index=bodies.size();index<integration.size();++index)
                mirror_actor(index,true);
        }
        for(const auto& dirty:pending_actor_locals_)
            actor_state_[dirty.first]|=actor_metadata_valid;
        last_transfer_bytes_={uploaded_bytes,
            output_bytes+(pending.device_admission
                ?16u+(pending.readback_device_task_records
                        ?pending.task_capacity*sizeof(native::GpuDyDistanceTask):0u)+
                    (pending.readback_body_ordinals
                        ?pending.ordinal_capacity*sizeof(std::uint32_t):0u)+8u+
                    (pending.compact_contact_admission
                        ?admitted_contact_sources_.size()*sizeof(std::uint32_t)
                        :pending.response_capacity*sizeof(std::uint32_t))
                    :0u)};
        last_reused_pcm_contact_rows_=pending.compact_contact_admission
            ?admitted_pcm_rows:reused_pcm_rows;
        last_omitted_pcm_upload_bytes_=
            last_reused_pcm_contact_rows_*sizeof(native::GpuDyContactPoint);
        last_cpu_gu_geometry_bytes_=pending.cpu_gu_bytes;
        last_friction_anchor_upload_bytes_=compact_anchor_bytes;
        for(std::size_t i=0u;i<manifolds.size();++i) {
            const auto& manager=manifolds[i];
            if(!manager.patch_count || manager.prep_index>=prep_count)
                continue;
            const auto pair=contact_origin(manager.prep_index);
            if(pair.pcm_pair_index==0xffffffffu)continue;
            if(pending.compact_contact_admission &&
               (admitted_contact_sources_[i/32u]&(1u<<(i&31u)))==0u) {
                history_epochs_[pair.pcm_edge]=0u;
                continue;
            }
            history_epochs_[pair.pcm_edge]=history_epoch_;
            friction_history_live_slots_=std::max(friction_history_live_slots_,
                std::size_t(pair.pcm_edge)+1u);
        }
        last_friction_prepared_rows_=friction_count;
        // The device accepted each changed joint only after all source
        // task results and their lifetime checks were published.
        for(const auto slot:joint_source_dirty_)
            joint_source_valid_[slot]=1u;
        last_contact_prep_upload_saving_=
            static_cast<std::int64_t>(prep_count*
                sizeof(native::GpuDyContactPrepPair))-
            static_cast<std::int64_t>(
                prep_bytes+inline_body_bytes+
                last_schedule_.retained_friction_source_bytes);
        resident_schedule_committed_=last_schedule_.resident_wide_schedule;
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
    std::span<const std::uint32_t> admitted_response_tokens() const noexcept {
        return {admitted_response_tokens_.data(),
            std::min(admitted_response_tokens_.size(),
                     last_schedule_.device_admitted_descriptors)};
    }
    std::span<const std::uint32_t> admitted_contact_sources() const noexcept {
        return {admitted_contact_sources_.data(),
                admitted_contact_sources_.size()};
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
    std::size_t last_friction_writeback_bytes() const noexcept {
        return last_friction_writeback_bytes_;
    }
    std::size_t last_friction_prepared_rows() const noexcept {
        return last_friction_prepared_rows_;
    }
    std::int64_t last_contact_prep_upload_saving() const noexcept {
        return last_contact_prep_upload_saving_;
    }
private:
    // Host updates preserve the GPU-owned dynamic velocity interval.
    void build_actor_local_uploads() {
        std::sort(pending_actor_locals_.begin(),pending_actor_locals_.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        std::size_t packed=0u;
        for(const auto dirty:pending_actor_locals_) {
            if(packed && pending_actor_locals_[packed-1u].first==dirty.first)
                pending_actor_locals_[packed-1u].second|=dirty.second;
            else pending_actor_locals_[packed++]=dirty;
        }
        pending_actor_locals_.resize(packed);
        actor_local_uploads_.clear();
        constexpr std::array<std::size_t,5> boundary{
            0u,offsetof(SourceDyActorLocalNative,linear_velocity),
            offsetof(SourceDyActorLocalNative,wake_counter),
            offsetof(SourceDyActorLocalNative,world_position),
            sizeof(SourceDyActorLocalNative)};
        for(const auto& dirty:pending_actor_locals_) {
            const auto* host=reinterpret_cast<const std::byte*>(
                actor_local_host_.data()+dirty.first);
            const auto base=actor_locals_.getDevicePtr()+
                std::size_t(dirty.first)*sizeof(SourceDyActorLocalNative);
            for(std::size_t field=0u;field<4u;++field) {
                if((dirty.second&(1u<<field))==0u)continue;
                const auto offset=boundary[field];
                const auto bytes=boundary[field+1u]-offset;
                const auto address=base+offset;
                const auto* source=host+offset;
                if(!actor_local_uploads_.empty()) {
                    auto& previous=actor_local_uploads_.back();
                    if(previous.destination+previous.bytes==address &&
                       static_cast<const std::byte*>(previous.data)+
                           previous.bytes==source) {
                        previous.bytes+=bytes;
                        continue;
                    }
                }
                actor_local_uploads_.push_back({address,source,bytes});
            }
        }
    }
    void reserve_actor_local_slots(std::size_t needed_actor_slots) {
        if(needed_actor_slots>actor_local_capacity_) {
            const auto capacity=std::max(needed_actor_slots,
                actor_local_capacity_+
                    std::max(actor_local_capacity_,std::size_t(8u)));
            if(capacity>(std::numeric_limits<std::uint32_t>::max()-128u)/
                            sizeof(SourceDyActorLocalNative) ||
               capacity>max_view_/sizeof(SourceDyActorLocalNative))
                throw std::overflow_error(
                    "Source Dy rigid actor local-pose pool exceeds GPU view");
            actor_local_host_.resize(capacity);
            actor_state_.resize(capacity,0u);
            actor_locals_.allocateCopyPrefixAsync(
                capacity*sizeof(SourceDyActorLocalNative)+127u,
                actor_local_live_slots_*sizeof(SourceDyActorLocalNative),
                stream_,PX_FL);
            actor_local_capacity_=capacity;
        }
    }
    struct PendingSolve {
        std::span<native::GpuDyDistanceBody> bodies{};
        std::span<native::GpuDyDistanceTask> tasks{};
        std::span<float> contact_forces{};
        std::span<native::GpuDyFrictionWriteback> friction{};
        std::span<native::GpuDyFrictionAnchorImpulse> anchor_impulses{};
        std::span<native::GpuDyContactPatch> patches{};
        std::span<native::GpuDyBodyIntegrationInput> integration{};
        std::span<native::GpuDyIntegratedPose> integrated_poses{};
        std::span<native::GpuDySleepState> sleep_states{};
        std::span<const native::GpuDyDistanceDescriptor> descriptors{};
        std::span<const std::uint32_t> body_ordinals{};
        std::span<const std::uint32_t> source_to_solver_row{};
        bool compact_contact_admission=false;
        bool source_indexed_integration=false;
        std::size_t response_capacity=0u;
        std::size_t task_capacity=0u,ordinal_capacity=0u;
        std::span<const native::GpuDyContactManifold> manifolds{};
        std::span<const std::uint32_t> contact_order{};
        std::span<const native::GpuDyContactPrepPair> contact_prep{};
        std::span<const native::GpuDyCompactContactPrep> compact_contact_prep{};

        std::array<std::size_t,6> result_offsets{};
        std::size_t output_bytes=0u;
        bool device_admission=false;
        bool readback_device_task_records=true;
        bool readback_body_ordinals=true;
        bool source_descriptors=false;
        std::size_t device_header_offset=0u,device_task_offset=0u,
                    device_error_offset=0u;
        std::size_t free_velocity_offset=0u,friction_count=0u;
        std::size_t uploaded_bytes=0u,cpu_gu_bytes=0u,reused_pcm_rows=0u;
        std::size_t compact_anchor_bytes=0u,prep_bytes=0u;
        std::size_t inline_body_bytes=0u;
        bool compact_friction=false;
        bool device_unconstrained_preparation=false;
        bool device_world_motion=false;
    };
    std::optional<PendingSolve> pending_solve_{};
    std::vector<compute::DownloadSpan> deferred_result_downloads_;
    enum class Launch : std::uint32_t {
        Fused, Prepare, Partition=3u, Motion, Integrate, PublishErrors,
        FreeBodies, Color, InitializeBodies, BuildPartitionTiles
    };
    struct WorkTile {
        std::uint32_t task,begin,end,reserved=0u;
    };
    static_assert(sizeof(WorkTile)==16u);
    struct TileRange {
        std::size_t begin=0u,count=0u;
        std::uint32_t iterations=0u;
        std::size_t dispatch_begin=0u;
    };

    void build_device_launch_plan(
        std::span<const native::GpuDyDistanceTask> tasks,
        std::span<const native::GpuDyDistanceDescriptor> descriptors,
        std::uint32_t partition_count,std::uint32_t response_capacity,
        std::uint32_t ordinals,std::uint32_t bodies,
        bool resident_admission,std::size_t task_capacity) {
        constexpr std::uint32_t threads=128u;
        constexpr std::uint32_t prep_managers=threads/4u;
        constexpr std::uint32_t wide_min=native::gpu_dy_wide_task_threshold;
        wide_tasks_.clear();
        tiles_.clear();
        partition_tiles_.clear();
        partition_output_count_=partition_dispatch_count_=0u;
        position_ends_.fill(false);
        wide_iterations_=0u;
        last_schedule_={};
        last_schedule_.device_colored_tasks=task_capacity;
        last_schedule_.device_colored_descriptors=response_capacity;
        last_schedule_.max_partition_groups=task_capacity?1u:0u;
        if(resident_admission) {
            // The host supplies numerical capacities, not runnable task
            // ranges. GPU Sc writes the compact tasks and their actual
            // count before device Dy dispatch, so unused slots cannot
            // enter the host's sequential task-range validation or
            // endpoint-degree analysis.
            degree_dynamic_.clear();
            degree_static_.clear();
            color_bounds_.clear();
            partition_endpoint_counts_.clear();
            fused_tiles_={0u,task_capacity};
            prep_tiles_={};
            body_tiles_={};
            return;
        }
        const bool has_wide_candidate=std::any_of(
            tasks.begin(),tasks.end(),[&](const auto& task) {
                return task.descriptor_end>=task.descriptor_begin &&
                    task.body_end>=task.body_begin &&
                    task.descriptor_end-task.descriptor_begin>=wide_min &&
                    task.body_end-task.body_begin>=wide_min;
            });
        if(has_wide_candidate) {
            degree_dynamic_.assign(bodies,0u);
            degree_static_.assign(bodies,0u);
            color_bounds_.assign(tasks.size(),0u);
            partition_endpoint_counts_.assign(tasks.size(),0u);
        }else {
            // Compact tasks are already GPU-colored. Do not zero or
            // rescan dense host endpoint-degree arrays for their launch.
            degree_dynamic_.clear();
            degree_static_.clear();
            color_bounds_.clear();
            partition_endpoint_counts_.clear();
        }
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
               task.partition_end>partition_count ||
               task.partition_end-partition_end!=
                   task.descriptor_end-descriptor_end ||
               task.body_begin!=body_end ||
               task.body_end<body_end || task.body_end>ordinals ||
               !task.position_iterations || task.position_iterations>255u ||
               task.velocity_iterations>255u)
                throw std::invalid_argument(
                    "Source device Dy task ranges are inconsistent");
            const auto count=task.descriptor_end-task.descriptor_begin;
            const auto writable=task.body_end-task.body_begin;
            // A task below either wide threshold always uses the
            // device-colored fused path; its host endpoint analysis
            // cannot change launch selection or partition semantics.
            if(count<wide_min || writable<wide_min) {
                descriptor_end=task.descriptor_end;
                partition_end=task.partition_end;
                body_end=task.body_end;
                continue;
            }
            last_schedule_.host_degree_scanned_descriptors+=count;
            std::size_t dynamic_edges=0u,static_edges=0u;
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
                    ++dynamic_edges;
                    ++degree_dynamic_[desc.row0];
                    ++degree_dynamic_[desc.row1];
                }else {
                    ++static_edges;
                    const auto body=flags&0x100u?desc.row0:desc.row1;
                    ++degree_static_[body];
                }
            }
            // First-fit dynamic coloring can skip at most the colors
            // occupied by an edge's two endpoints. After that body
            // reserves its own static constraints in encounter order.
            // Taking the per-edge sum avoids combining unrelated global
            // maximum degrees, without computing any host color.
            std::size_t bound=0u;
            std::size_t max_dynamic_conflicts=0u;
            std::size_t max_static_degree=0u;
            for(std::uint32_t i=task.descriptor_begin;
                i<task.descriptor_end;++i) {
                const auto& desc=descriptors[i];
                const auto flags=desc.row_count&0x300u;
                if(flags==0x300u) {
                    // Dynamic edges color before static reservations.
                    // Count only dynamic-edge adjacency when estimating
                    // their guaranteed conflict-free first partition.
                    max_dynamic_conflicts=std::max(
                        max_dynamic_conflicts,
                        std::size_t(degree_dynamic_[desc.row0])+
                            degree_dynamic_[desc.row1]-1u);
                    const auto degree=std::size_t(degree_dynamic_[desc.row0])+
                                      degree_dynamic_[desc.row1]-1u;
                    const auto static_degree=std::max(
                        degree_static_[desc.row0],degree_static_[desc.row1]);
                    bound=std::max(bound,degree+static_degree);
                }else {
                    const auto body=flags&0x100u?desc.row0:desc.row1;
                    max_static_degree=std::max(
                        max_static_degree,
                        std::size_t(degree_static_[body]));
                    bound=std::max(bound,std::size_t(degree_static_[body]));
                }
            }
            // The integration ordinal list can omit bodies with
            // zero-response contact managers. Partitioning still
            // addresses those bodies, so count distinct dynamic
            // descriptor endpoints before retiring their degree data.
            std::size_t partition_endpoints=0u;
            const auto retire_endpoint=[&](std::uint32_t body) {
                if(degree_dynamic_[body] || degree_static_[body])
                    ++partition_endpoints;
                degree_dynamic_[body]=0u;
                degree_static_[body]=0u;
            };
            for(std::uint32_t i=task.descriptor_begin;
                i<task.descriptor_end;++i) {
                const auto& desc=descriptors[i];
                if(desc.row_count&0x100u)
                    retire_endpoint(desc.row0);
                if(desc.row_count&0x200u)
                    retire_endpoint(desc.row1);
            }
            partition_endpoint_counts_[task_id]=
                static_cast<std::uint32_t>(partition_endpoints);
            bound=std::min<std::size_t>(count,bound);
            color_bounds_[task_id]=static_cast<std::uint32_t>(bound);
            descriptor_end=task.descriptor_end;
            partition_end=task.partition_end;
            body_end=task.body_end;
            // The first dynamic partition is a maximal matching, so it
            // contains at least dynamic_edges/max_dynamic_conflicts
            // descriptors. Static constraints reserve colors afterward,
            // potentially skipping otherwise free low colors; they
            // contribute to the guarantee only for static-only tasks.
            // High-degree stars have many bodies but one useful lane per
            // color and must not create a quadratic tiled schedule.
            const std::size_t parallel_width=dynamic_edges
                ?(dynamic_edges-1u)/max_dynamic_conflicts+1u
                :static_edges?(static_edges-1u)/max_static_degree+1u:0u;
            const auto groups=(partition_endpoints+threads-1u)/threads;
            // Color count multiplied by conservative workgroups must not
            // exceed one tile per source descriptor. In particular, a
            // collection of connected high-degree hubs cannot amplify a
            // compact source graph into a quadratic upload and dispatch
            // schedule merely because its first matching is wide.
            if(writable<wide_min || count<wide_min ||
               parallel_width<threads || bound>std::size_t(count)/groups)
                continue;
            wide_tasks_.push_back(static_cast<std::uint32_t>(task_id));
            // Each color has at most one descriptor per dynamic
            // endpoint, including zero-response contact participants
            // omitted from the integration writeback ordinal stream.
            last_schedule_.max_partition_groups=std::max(
                last_schedule_.max_partition_groups,
                (partition_endpoints+threads-1u)/threads);
            partition_levels=std::max(partition_levels,bound);
            wide_iterations_=std::max(wide_iterations_,
                task.position_iterations+
                    std::max(task.velocity_iterations,1u));
            position_ends_[task.position_iterations-1u]=true;
        }
        if(descriptor_end!=descriptors.size() ||
           partition_end!=partition_count || body_end!=ordinals)
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
        // Endpoint degrees bound storage and command count. The GPU
        // generates exact tiles and dispatch arguments from color ends.
        partition_tiles_.resize(partition_levels);
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            const auto endpoints=partition_endpoint_counts_[id];
            const auto tile_count=(std::size_t(endpoints)+threads-1u)/threads;
            for(std::uint32_t color=0u;color<color_bounds_[id];++color) {
                auto& range=partition_tiles_[color];
                range.count+=tile_count;
                range.iterations=std::max(range.iterations,
                    task.position_iterations+
                        std::max(task.velocity_iterations,1u));
            }
        }
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        const auto narrow_tile=[](std::size_t value) {
            if(value>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Device Dy partition reservation exceeds range");
            return static_cast<std::uint32_t>(value);
        };
        partition_header_begin_=tiles_.size();
        for(auto& range:partition_tiles_) {
            range.begin=partition_output_count_;
            range.dispatch_begin=partition_dispatch_count_;
            const auto chunks=(range.count-1u)/max_groups+1u;
            partition_output_count_+=range.count;
            partition_dispatch_count_+=chunks;
            tiles_.push_back({narrow_tile(range.begin),narrow_tile(range.count),
                              narrow_tile(range.dispatch_begin),narrow_tile(chunks)});
        }
        partition_task_begin_=tiles_.size();
        for(const auto task:wide_tasks_)
            tiles_.push_back({task,color_bounds_[task],
                narrow_tile((std::size_t(partition_endpoint_counts_[task])+
                             threads-1u)/threads),0u});
        last_schedule_.tile_bytes=tiles_.size()*sizeof(WorkTile);
        last_schedule_.device_partition_tile_bytes=
            partition_output_count_*sizeof(WorkTile);
    }

    void build_resident_launch_plan(native::GpuDyWideScheduleBounds bounds,
            std::size_t nodes,std::size_t responses,std::size_t task_capacity) {
        const auto max_groups=context_.device_info().max_workgroup_count[0];
        const auto wide_capacity=bounds.nodes/native::gpu_dy_wide_task_threshold;
        const auto body_groups=(std::size_t(bounds.nodes)+127u)/128u+wide_capacity;
        if(bounds.nodes>nodes || !wide_capacity || !bounds.stages || !bounds.iterations ||
           bounds.iterations>510u || bounds.stages>responses ||
           !task_capacity)
            throw std::logic_error("Resident Dy wide reservation is invalid");
        const auto narrow=[](std::size_t value) {
            if(value>std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Resident Dy schedule exceeds address space");
            return static_cast<std::uint32_t>(value);
        };
        const auto reserve=[&](TileRange& range,std::size_t count) {
            range={partition_output_count_,count,bounds.iterations,
                   partition_dispatch_count_};
            const auto chunks=(count-1u)/max_groups+1u;
            partition_output_count_+=count;
            partition_dispatch_count_+=chunks;
            tiles_.push_back({narrow(range.begin),narrow(count),
                              narrow(range.dispatch_begin),narrow(chunks)});
        };
        partition_header_begin_=0u;
        reserve(fused_tiles_,task_capacity);
        reserve(prep_tiles_,(responses+31u)/32u+wide_capacity);
        reserve(body_tiles_,body_groups);
        // Wide colors and intervening local runs share a response-bounded
        // tile pool. Each stage reserves only its indirect commands.
        partition_tiles_.resize(bounds.stages);
        for(auto& range:partition_tiles_) {
            range={tiles_.size(),body_groups,bounds.iterations,
                   partition_dispatch_count_};
            const auto chunks=(body_groups-1u)/max_groups+1u;
            partition_dispatch_count_+=chunks;
            tiles_.push_back({0u,narrow(body_groups),
                              narrow(range.dispatch_begin),narrow(chunks)});
        }
        // A wide color of width w emits at most 2*floor(w/128) tiles.
        // Narrow runs contribute at most two tiles per wide color.
        const auto pool_capacity=static_cast<std::size_t>(
            std::min<std::uint64_t>(4ull*(responses/128u),
                std::uint64_t(bounds.stages)*body_groups));
        tiles_.push_back({narrow(partition_output_count_),narrow(pool_capacity),
                          0u,0u});
        partition_output_count_+=pool_capacity;
        wide_iterations_=bounds.iterations;
        last_schedule_.resident_wide_schedule=true;
        last_schedule_.max_partition_groups=body_groups;
        last_schedule_.tile_bytes=tiles_.size()*sizeof(WorkTile);
        last_schedule_.device_partition_tile_bytes=
            pool_capacity*sizeof(WorkTile);
    }

    void build_launch_plan(std::span<const native::GpuDyDistanceTask> tasks,
                           std::span<const std::uint32_t> partitions,
                           std::uint32_t descriptors,std::uint32_t ordinals) {
        constexpr std::uint32_t threads=128u;
        constexpr std::uint32_t prep_managers=threads/4u;
        // Tile only when one partition can supply four full groups.
        // Smaller tasks keep their partition dependencies in one group.
        constexpr std::uint32_t wide_min=native::gpu_dy_wide_task_threshold;
        wide_tasks_.clear();
        tiles_.clear();
        partition_tiles_.clear();
        partition_output_count_=partition_dispatch_count_=0u;
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
    physx::PxgCudaBuffer actor_locals_;
    physx::PxgCudaBuffer joint_sources_;
    physx::PxgCudaBuffer retained_sources_;
    SourceFrameInputArenaNative retained_frame_input_;
    compute::PipelineId pipeline_=0;
    compute::PipelineId contact_endpoints_pipeline_=0;
    compute::PipelineId convex_pipeline_=0;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
    compute::BufferId input_upload_buffer_=0u;
    std::size_t input_upload_capacity_=0u,input_upload_initialized_=0u;
    FrameWorkingSet input_working_set_,arena_working_set_,result_working_set_;
    std::unique_ptr<std::uint8_t[]> host_;
    std::size_t host_result_capacity_=0u;
    std::vector<std::uint32_t> admitted_body_ordinals_;
    std::vector<std::uint32_t> admitted_response_tokens_;
    std::vector<std::uint32_t> admitted_contact_sources_;
    struct ContactOrigin {
        std::uint32_t pcm_pair_index=0xffffffffu;
        std::uint32_t pcm_edge=0xffffffffu;
        std::uint32_t cpu_gu_offset=0xffffffffu;
    };
    static ContactOrigin source_contact_origin(
            std::size_t index,
            std::span<const native::GpuDyContactPrepPair> pairs,
            std::span<const native::GpuDyCompactContactPrep> compact) {
        if(!compact.empty()) {
            if(index>=compact.size())
                throw std::logic_error("Compact Dy contact source index invalid");
            const auto& source=compact[index];
            return {source.pcm_pair_index,source.pcm_edge,source.cpu_gu_offset};
        }
        if(index>=pairs.size())
            throw std::logic_error("Dy contact source index invalid");
        const auto& source=pairs[index];
        return {source.pcm_pair_index,source.pcm_edge,source.cpu_gu_offset};
    }
    std::vector<SourceDyPackedContactPrepNative> packed_;
    struct RetainedPatchInput {
        std::size_t patch;
        std::uint32_t candidate;
    };
    std::vector<RetainedPatchInput> retained_patch_inputs_;
    std::vector<native::GpuDyRetainedFrictionCandidate> packed_retained_candidates_;
    const native::GpuDyRetainedFrictionCandidate* prewitness_retained_source_data_=nullptr;
    std::size_t prewitness_retained_source_count_=0u;
    CudaDeviceAddressMap::Address retained_source_address_=0u;
    std::vector<std::uint32_t> source_descriptor_tokens_;
    std::vector<native::GpuDyDistanceDescriptor> source_descriptor_full_;
    std::vector<SourceDyPackedIntegrationNative> packed_integration_;
    std::vector<SourceDyHostPoseNative> host_poses_;
    std::vector<SourceDyActorLocalNative> actor_local_host_;
    enum ActorState : std::uint8_t {
        actor_metadata_valid=1u,
        actor_velocity_valid=2u,
        actor_pose_valid=4u,
        actor_prewitness_metadata=8u,
        actor_prewitness_pose=16u,
        actor_prewitness_mask=actor_prewitness_metadata|actor_prewitness_pose
    };
    std::vector<std::uint8_t> actor_state_;
    std::vector<float> prepared_velocity_components_,force_mod_components_;
    std::vector<std::pair<std::uint32_t,std::uint8_t>> pending_actor_locals_;
    std::vector<SourceSapUpload> actor_local_uploads_;
    std::size_t actor_local_capacity_=0u,actor_local_live_slots_=0u;
    std::vector<native::GpuDyJointInput> joint_source_host_;
    std::vector<std::uint8_t> joint_source_valid_,joint_row_source_seen_;
    std::vector<std::uint32_t> joint_source_dirty_;
    std::vector<std::uint32_t> joint_source_late_upload_slots_;
    std::vector<std::uint32_t> prewitness_joint_source_slots_;
    std::vector<std::uint32_t> prewitness_actor_slots_;
    std::size_t prewitness_actor_local_upload_bytes_=0u;
    std::size_t prewitness_actor_local_upload_spans_=0u;
    std::size_t prewitness_joint_source_upload_bytes_=0u;
    std::size_t prewitness_joint_source_upload_spans_=0u;
    std::vector<SourceSapUpload> joint_source_uploads_;
    std::size_t joint_source_capacity_=0u,joint_source_live_slots_=0u;
    struct InlineBodySlot {
        std::uint32_t epoch=0u,index=0u;
    };
    std::vector<native::GpuDyContactPrepBody> inline_bodies_;
    std::vector<InlineBodySlot> inline_body_slots_;
    std::uint32_t inline_body_epoch_=0u;
    std::vector<std::uint8_t> assigned_;
    std::vector<std::uint32_t> wide_tasks_;
    std::vector<std::uint32_t> degree_dynamic_,degree_static_,color_bounds_;
    std::vector<std::uint32_t> partition_endpoint_counts_;
    std::vector<WorkTile> tiles_;
    std::vector<TileRange> partition_tiles_;
    TileRange fused_tiles_,prep_tiles_,body_tiles_;
    std::size_t partition_output_count_=0u,partition_dispatch_count_=0u;
    std::size_t partition_header_begin_=0u,partition_task_begin_=0u;
    compute::BufferId partition_dispatch_buffer_=0u;
    std::size_t partition_dispatch_capacity_=0u;
    std::array<std::uint64_t,16> resident_schedule_layout_{};
    bool resident_schedule_committed_=false;
    std::array<bool,255> position_ends_{};
    std::uint32_t wide_iterations_=0u;
    SourceDyLaunchStats last_schedule_{};
    // Actual packed device upload and ordered readback byte counts.
    std::pair<std::size_t,std::size_t> last_transfer_bytes_{};
    std::size_t last_friction_writeback_bytes_=0u;
    std::size_t last_integrated_body_count_=0u;
    std::size_t last_resident_pose_count_=0u;
    std::size_t last_host_pose_upload_bytes_=0u;
    std::size_t last_reused_pcm_contact_rows_=0u;
    std::size_t last_omitted_pcm_upload_bytes_=0u;
    std::size_t last_cpu_gu_geometry_bytes_=0u;
    std::size_t last_upload_spans_=0u;
    std::size_t last_friction_anchor_upload_bytes_=0u;
    std::size_t friction_history_capacity_=0u,friction_history_live_slots_=0u;
    std::vector<std::uint32_t> history_epochs_;
    std::uint32_t history_epoch_=1u;
    std::size_t last_friction_prepared_rows_=0u;
    std::int64_t last_contact_prep_upload_saving_=0;
};

} // namespace fyzz::physx_gpu
