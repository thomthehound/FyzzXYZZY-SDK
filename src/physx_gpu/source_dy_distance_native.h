// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "native/scene.h"
#include "physx_gpu/source_native_view.h"
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
    float sqrt_inverse_inertia[9]{};
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
static_assert(sizeof(SourceDyPackedIntegrationNative)==76u &&
              offsetof(SourceDyPackedIntegrationNative,pose_index)==56u &&
              offsetof(SourceDyPackedIntegrationNative,actor_local_offset)==60u &&
              offsetof(SourceDyPackedIntegrationNative,velocity_offset)==64u &&
              offsetof(SourceDyPackedIntegrationNative,force_offset)==68u &&
              offsetof(SourceDyPackedIntegrationNative,counted_interactions)==72u);
inline constexpr std::uint32_t source_dy_device_inertia_flag=0x4000u;
static_assert((source_dy_device_inertia_flag&
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
    std::size_t device_island_indirect_dispatches=0u;
    std::size_t host_lane_packed_tasks=0u;
    std::size_t device_admitted_tasks=0u;
    std::size_t device_admitted_bodies=0u;
    std::size_t device_admitted_descriptors=0u;
    std::size_t tile_bytes=0u;
    // Device-written descriptors, partition ranges, tasks, and ordinals
    // omitted from the host numerical input packet.
    std::size_t device_generated_bytes=0u;
    std::size_t device_source_descriptors=0u;
    std::size_t source_descriptor_upload_bytes=0u;
    // Native Sc initializes transient solver bodies from the resident
    // actor inverse masses; standalone Dy retains explicit input support.
    std::size_t device_initialized_bodies=0u;
    std::size_t device_body_upload_bytes=0u;
    std::size_t device_initialization_dispatches=0u;
    std::size_t device_world_inertia_bodies=0u;
    std::size_t device_world_motion_bodies=0u;
    std::size_t actor_local_upload_bytes=0u;
    std::size_t actor_local_upload_spans=0u;
    std::size_t device_upload_batches=0u;
    std::size_t reused_actor_local_poses=0u;
    std::size_t resident_velocity_reuses=0u;
    std::size_t prepared_velocity_updated_bodies=0u;
    std::size_t prepared_velocity_upload_bytes=0u;
    std::size_t device_joint_input_rows=0u;
    std::size_t device_joint_geometry_count=0u;
    std::size_t device_joint_metadata_upload_bytes=0u;
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
        std::optional<compute::ShaderBinary> convex_shader=std::nullopt)
        : context_(context),addresses_(addresses),stream_(stream),
          arena_(allocator,physx::PxsHeapStats::eSOLVER),
          actor_locals_(allocator,physx::PxsHeapStats::eSOLVER) {
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
        pipeline.storage_buffer_binding_count=6u;
        pipeline.storage_buffer_views.assign(
            6u,compute::StorageBufferViewKind::RawReadWrite);
        pipeline.push_constant_bytes=128u;
        pipeline_=context_.create_compute_pipeline(pipeline);
#if defined(FYZZXYZZY_PROFILE_GPU)
        context_.set_pipeline_profile_label(pipeline_,"dy.distance_pgs");
#endif
        if(convex_shader) {
            try {
                pipeline.shader=std::move(*convex_shader);
                pipeline.storage_buffer_binding_count=7u;
                pipeline.storage_buffer_views.assign(
                    7u,compute::StorageBufferViewKind::RawReadWrite);
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
    }
    SourceDyDistanceNativeStage(const SourceDyDistanceNativeStage&)=delete;
    SourceDyDistanceNativeStage& operator=(const SourceDyDistanceNativeStage&)=delete;
    ~SourceDyDistanceNativeStage() {
        if(convex_pipeline_)context_.destroy_compute_pipeline(convex_pipeline_);
        if(pipeline_)context_.destroy_compute_pipeline(pipeline_);
        arena_.deallocate();
        actor_locals_.deallocate();
    }
    void retire_rigid_actor(BodyId id) noexcept {
        if(id<actor_local_valid_.size()) {
            actor_local_valid_[id]=0u;
            actor_velocity_valid_[id]=0u;
            actor_pose_valid_[id]=0u;
        }
    }
    void invalidate_after_rollback() noexcept {
        pending_solve_.reset();
        std::fill(actor_local_valid_.begin(),actor_local_valid_.end(),0u);
        std::fill(actor_velocity_valid_.begin(),actor_velocity_valid_.end(),0u);
        std::fill(actor_pose_valid_.begin(),actor_pose_valid_.end(),0u);
        std::fill(joint_source_valid_.begin(),joint_source_valid_.end(),0u);
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
               std::optional<SourceScDeviceIslandDyView> device_admission=std::nullopt) {
        if(pending_solve_)
            throw std::logic_error("Source Dy numerical epoch already queued");
        if(device_admission &&
           tasks.size()>context_.device_info().max_workgroup_count[0])
            throw std::out_of_range(
                "Device Sc compact tasks exceed indirect GPU Dy launch capacity");

        if(integration.size()<bodies.size() ||
           integrated_poses.size()!=integration.size() ||
           (tasks.empty() && integration.size()==bodies.size()))
            throw std::invalid_argument(
                "Source Dy GPU integrateCore inputs do not match solver bodies");
        last_integrated_body_count_=0u;
        if(device_initial_bodies && !device_partitions)
            throw std::invalid_argument(
                "Source Dy device body initialization requires device partitions");
        const auto narrow=[](std::size_t value) {
            if(value>=std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Source Dy GPU exceeds 32-bit view ABI");
            return static_cast<std::uint32_t>(value);
        };
        const auto free_bodies=integration.size()-bodies.size();
        if(device_unconstrained_preparation &&
           sleep_states.size()!=integration.size())
            throw std::invalid_argument(
                "Source GPU Dy sleep-state outputs do not match body ordinals");
        if(device_initial_bodies && device_world_motion)
            for(std::size_t i=0u;i<bodies.size();++i)
                if(!(integration[i].active&1u) &&
                   bodies[i].inverse_mass!=0.0f)
                    throw std::logic_error(
                        "Native Dy inactive solver row has nonzero inverse mass");
        const auto partition_count=device_admission
            ? descriptors.size()
            : device_partitions
                ? (tasks.empty()?0u:tasks.back().partition_end)
                : partition_ends.size();
        if(partition_count>std::numeric_limits<std::uint32_t>::max()/4u)
            throw std::overflow_error("Source Dy partition arena exceeds 32-bit offsets");
        if(device_partitions) {
            build_device_launch_plan(tasks,descriptors,narrow(partition_count),
                                     narrow(body_ordinals.size()),
                                     narrow(bodies.size()),
                                     device_admission.has_value());
        } else {
            build_launch_plan(tasks,partition_ends,narrow(descriptors.size()),
                              narrow(body_ordinals.size()));
        }
        last_schedule_.device_initialized_bodies=device_initial_bodies?bodies.size():0u;
        last_schedule_.device_body_upload_bytes=device_initial_bodies
            ? (device_world_motion?0u:bodies.size()*sizeof(float))
            : bodies.size_bytes();
        last_schedule_.device_world_motion_bodies=
            device_world_motion?body_ordinals.size():0u;
        if(device_unconstrained_preparation &&
           (!device_world_motion || !device_initial_bodies))
            throw std::invalid_argument(
                "GPU preintegration requires native device body ownership");
        last_schedule_.device_preintegrated_bodies=
            device_unconstrained_preparation
                ?body_ordinals.size()+free_bodies:0u;
        last_schedule_.device_sleep_bodies=
            last_schedule_.device_preintegrated_bodies;
        last_schedule_.device_joint_input_rows=
            std::count_if(rows.begin(),rows.end(),
                [](const auto& row){return row.reserved==1u || row.reserved==2u;});
        last_schedule_.device_joint_geometry_count=joint_inputs.size();
        last_schedule_.device_joint_metadata_upload_bytes=0u;
        last_schedule_.device_joint_metadata_upload_spans=0u;
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
        const std::size_t joint_base=(
            next_history_capacity_*64u+127u)&~std::size_t(127u);
        // EdgeIndex friction history occupies the first part of Dy's
        // allocation; stable Sc constraint records follow that prefix.
        std::size_t next_joint_capacity=joint_source_capacity_;
        for(const auto& joint:joint_inputs) {
            if(joint.source_slot==0xffffffffu)
                throw std::logic_error("Native Dy joint has no live Sc constraint slot");
            next_joint_capacity=std::max(next_joint_capacity,
                                         std::size_t(joint.source_slot)+1u);
        }
        if(joint_base>std::numeric_limits<std::uint32_t>::max()-127u ||
           next_joint_capacity>
               (std::numeric_limits<std::uint32_t>::max()-joint_base-127u)/
               sizeof(native::GpuDyJointInput))
            throw std::overflow_error("Native Dy resident joints exceed 32-bit view");
        const std::size_t history_bytes=(joint_base+
            next_joint_capacity*sizeof(native::GpuDyJointInput)+127u)&
            ~std::size_t(127u);
        // Friction growth moves the joint base. Re-admit the source
        // record spans instead of preserving bytes at obsolete offsets.
        if(next_history_capacity_!=friction_history_capacity_)
            std::fill(joint_source_valid_.begin(),joint_source_valid_.end(),0u);
        joint_source_host_.resize(next_joint_capacity);
        joint_source_valid_.resize(next_joint_capacity,0u);
        joint_source_dirty_.clear();
        for(const auto& joint:joint_inputs) {
            const auto slot=joint.source_slot;
            if(!joint_source_valid_[slot] ||
               std::memcmp(&joint_source_host_[slot],&joint,sizeof(joint))!=0) {
                // An attempted edit is not resident until the full frame
                // completes. A failed submission must reupload on retry.
                joint_source_valid_[slot]=0u;
                joint_source_host_[slot]=joint;
                joint_source_dirty_.push_back(slot);
            }else
                ++last_schedule_.device_joint_metadata_reuses;
        }
        std::sort(joint_source_dirty_.begin(),joint_source_dirty_.end());
        if(std::adjacent_find(joint_source_dirty_.begin(),
                              joint_source_dirty_.end())!=joint_source_dirty_.end())
            throw std::logic_error("Native Dy joint slot admitted more than once");
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
            if(ordinal!=invalid && body.solver_ordinal!=invalid &&
               ordinal!=body.solver_ordinal)
                throw std::logic_error(
                    "Source Dy descriptor disagrees with prepared contact body");
            if(ordinal==invalid)ordinal=body.solver_ordinal;
            // An awake zero-inverse-mass dynamic still has source motion.
            // GPU preintegration owns it, unlike static/kinematic inline
            // participants whose velocities Sc prepares independently.
            const bool active_dynamic=ordinal!=invalid &&
                ordinal<integration.size() &&
                (integration[ordinal].active&1u)!=0u;
            if(active_dynamic) {
                if(ordinal>=bodies.size() || ordinal>=invalid/2u ||
                   body.inverse_mass!=bodies[ordinal].inverse_mass)
                    throw std::logic_error(
                        "Active source Dy contact body has no matching solver row");
                return (ordinal<<1u)|1u;
            }
            if(body.inverse_mass>0.f)
                throw std::logic_error(
                    "Responding source Dy contact body has no solver ordinal");
            if(inline_bodies_.size()>=(
                std::numeric_limits<std::uint32_t>::max()-static_begin)/
                    sizeof(native::GpuDyContactPrepBody))
                throw std::overflow_error("Source Dy inline contact body overflow");
            const auto offset=static_begin+inline_bodies_.size()*
                sizeof(native::GpuDyContactPrepBody);
            inline_bodies_.push_back(body);
            inline_bodies_.back().solver_ordinal=ordinal;
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
        std::size_t needed_actor_slots=0u;
        for(std::size_t ordinal=0u;ordinal<integration.size();++ordinal) {
            const auto& input=integration[ordinal];
            if(!(input.active&1u))continue;
            const auto id=device_world_motion
                ?input.source_actor:narrow(ordinal);
            if(id==invalid ||
               id>(std::numeric_limits<std::uint32_t>::max()-128u)/
                      sizeof(SourceDyActorLocalNative))
                throw std::overflow_error(
                    "Source Dy rigid actor local-pose index exceeds GPU address space");
            needed_actor_slots=std::max(needed_actor_slots,std::size_t(id)+1u);
        }
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
            actor_local_valid_.resize(capacity,0u);
            actor_velocity_valid_.resize(capacity,0u);
            actor_pose_valid_.resize(capacity,0u);
            actor_locals_.allocateCopyPrefixAsync(
                capacity*sizeof(SourceDyActorLocalNative)+127u,
                actor_local_capacity_*sizeof(SourceDyActorLocalNative),
                stream_,PX_FL);
            actor_local_capacity_=capacity;
        }
        for(std::size_t ordinal=0u;ordinal<integration.size();++ordinal) {
            const auto& input=integration[ordinal];
            SourceDyPackedIntegrationNative record{};
            std::memcpy(record.sqrt_inverse_inertia,input.sqrt_inverse_inertia,
                        sizeof(record.sqrt_inverse_inertia));
            record.lock_flags=input.lock_flags;
            record.seconds=input.seconds;
            record.active=(input.active&1u)|
                (device_world_motion && (input.active&1u) ?(8u|128u):0u)|
                (device_world_motion && (input.active&32u) ?source_dy_device_inertia_flag:0u);
            // Bits [8:13] are exact authored velocity-change components;
            // bit 14 is reserved for device inertia preparation.
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
                if(device_world_motion && ordinal<bodies.size() &&
                   local.inverse_mass!=bodies[ordinal].inverse_mass)
                    throw std::logic_error(
                        "Sc actor-local inverse mass differs from Dy admission");
                // Only the immutable mass-frame prefix participates in
                // metadata invalidation; Dy owns the trailing velocities.
                constexpr auto metadata_bytes=
                    offsetof(SourceDyActorLocalNative,linear_velocity);
                if(!actor_local_valid_[id] ||
                   std::memcmp(&actor_local_host_[id],&local,metadata_bytes)!=0) {
                    std::memcpy(&actor_local_host_[id],&local,metadata_bytes);
                    actor_local_valid_[id]=0u;
                    pending_actor_locals_.push_back(id);
                }else
                    ++last_schedule_.reused_actor_local_poses;
                // Authored and restored poses re-seed only their actor slot.
                // Unchanged integrated poses remain the next device input.
                if(device_world_motion) {
                    const bool reused=actor_pose_valid_[id] &&
                        std::memcmp(actor_local_host_[id].world_position,
                            input.actor_position,
                            sizeof(local.world_position))==0 &&
                        std::memcmp(actor_local_host_[id].world_orientation,
                            input.actor_orientation,
                            sizeof(local.world_orientation))==0;
                    if(reused)++last_resident_pose_count_;
                    else {
                        std::memcpy(actor_local_host_[id].world_position,
                                    input.actor_position,
                                    sizeof(local.world_position));
                        std::memcpy(actor_local_host_[id].world_orientation,
                                    input.actor_orientation,
                                    sizeof(local.world_orientation));
                        actor_local_valid_[id]=0u;
                        pending_actor_locals_.push_back(id);
                    }
                }
                std::uint32_t changed=0u;
                const bool prior=device_world_motion &&
                    actor_velocity_valid_[id];
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
                if(!actor_local_valid_[id] ||
                   std::memcmp(&cached.wake_counter,
                               &input.wake_counter,sleep_bytes)!=0) {
                    std::memcpy(&cached.wake_counter,
                                &input.wake_counter,sleep_bytes);
                    actor_local_valid_[id]=0u;
                    pending_actor_locals_.push_back(id);
                }
                // Every published body must explicitly reestablish its
                // committed velocity after a successful device solve.
                actor_velocity_valid_[id]=0u;
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
        constexpr std::size_t integration_header_bytes=40u;
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
            tasks.size()>1u && wide_tasks_.empty();
        last_schedule_.host_lane_packed_tasks=host_lane_packing
            ?std::count_if(tasks.begin(),tasks.end(),lane_task):0u;
        if(device_admission &&
           (!source_descriptors || !device_admission->work ||
            !device_admission->task_dispatch.buffer || tasks.empty()))
            throw std::invalid_argument(
                "Source Dy device island admission requires resident numerical task launch");
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
        const std::uint64_t scratch_bytes=device_partitions
            ?(source_descriptors?(device_admission?44ull:28ull):20ull)+
                std::uint64_t(descriptors.size())*24ull+
                std::uint64_t(bodies.size())*12ull+
                std::uint64_t(source_descriptor_full_.size())*16ull
            :0ull;
        if(scratch_bytes>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(
                "Device Dy partition scratch exceeds address space");
        const auto partition_scratch_bytes=
            static_cast<std::size_t>(scratch_bytes);
        last_schedule_.device_source_descriptors=
            source_descriptors?descriptors.size():0u;
        last_schedule_.source_descriptor_upload_bytes=source_descriptors
            ?source_descriptor_tokens_.size()*sizeof(std::uint32_t)+
                source_descriptor_full_.size()*sizeof(native::GpuDyDistanceDescriptor)
            :descriptors.size_bytes();
        if(rows.size_bytes()>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Device Dy joint rows exceed 32-bit view");
        constexpr std::size_t count=16u;
        // The integration prefix describes Sc/SAP views and the launch
        // table. Solver-body ordinals precede the free-body suffix.
        const std::array<std::size_t,count> bytes{
            bodies.size_bytes(),rows.size_bytes(),descriptors.size_bytes(),
            partition_count*sizeof(std::uint32_t),body_ordinals.size_bytes(),
            tasks.size_bytes(),contact_bytes,friction_bytes,
            patches.size_bytes(),manifolds.size_bytes(),contact_order.size_bytes(),
            compact_prep_bytes,
            integration_bytes,compact_anchor_bytes,
            tiles_.size()*sizeof(WorkTile),partition_scratch_bytes};
        const std::array<std::size_t,count> elements{
            bodies.size(),rows.size(),descriptors.size(),partition_count,
            body_ordinals.size(),tasks.size(),contact_forces.size(),friction_count,
            patches.size(),manifolds.size(),contact_order.size(),
            contact_prep.size(),integration.size(),compact_anchors_.size(),
            tiles_.size(),device_partitions?1u:0u};
        const std::array<const void*,count> sources{
            bodies.data(),rows.data(),
            device_partitions?nullptr:descriptors.data(),
            device_partitions?nullptr:partition_ends.data(),
            body_ordinals.data(),tasks.data(),nullptr,nullptr,
            patches.data(),manifolds.data(),contact_order.data(),
            contact_prep.data(),integration.data(),nullptr,tiles_.data(),
            nullptr};
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
                      offsetof(native::GpuDyContactPoint,applied_force)==56u &&
                      offsetof(native::GpuDyFrictionRow,applied_force)==52u &&
                      offsetof(native::GpuDyContactPatch,broken)==12u);
        std::array<std::size_t,6> result_offsets{};
        std::size_t output_bytes=0u;
        for(std::size_t i=0;i<output_inputs.size();++i) {
            const auto n=i==3u && compact_friction
                ? anchor_impulses.size():elements[output_inputs[i]];
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
        // Actual compact tasks and body ordinals are separate spans of
        // the same ordered Dy result fetch, not GPU copies to a mirror
        // region. Their destinations follow numerical output in host
        // staging; the device arena retains only its real output extent.
        const std::size_t device_header_offset=output_bytes;
        const std::size_t device_task_offset=output_bytes+
            (device_admission?16u:0u);
        const std::size_t device_ordinal_offset=device_task_offset+
            (device_admission?tasks.size_bytes():0u);
        const std::size_t host_result_bytes=device_ordinal_offset+
            (device_admission?body_ordinals.size_bytes():0u);
        if(device_admission &&
           (host_result_bytes<device_ordinal_offset ||
            host_result_bytes>std::numeric_limits<std::uint32_t>::max()))
            throw std::overflow_error(
                "Resident Sc Dy scheduling result exceeds address space");
        last_schedule_.free_velocity_readback_bytes=free_velocity_bytes;
        last_schedule_.sleep_readback_bytes=
            device_unconstrained_preparation
                ?(body_ordinals.size()+free_bodies)*28u:0u;
        // Both writable contact and friction arenas follow all source
        // inputs. The existing GPU preparation owns the complete rows.
        // Device coloring consumes raw descriptors in the scratch input
        // and writes both sorted descriptors and partition ends itself.
        // Place those output-only ranges after the contact/friction rows;
        // compact anchors must remain adjacent to contact rows.
        // Resident Sc tasks and body ordinals are GPU copy destinations,
        // not CPU packet inputs. Keep them beyond the host upload cutoff,
        // alongside Dy's other device-written numerical ranges.
        const std::array<std::size_t,count> order=device_admission
            ? std::array<std::size_t,count>{
                1u,8u,9u,10u,11u,12u,14u,15u,13u,6u,7u,0u,2u,3u,4u,5u}
            : device_initial_bodies
                ? std::array<std::size_t,count>{
                1u,5u,8u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u,0u,2u,3u}
            : device_partitions
                ? std::array<std::size_t,count>{
                    0u,1u,5u,8u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u,2u,3u}
                : std::array<std::size_t,count>{
                    0u,1u,5u,8u,2u,3u,4u,9u,10u,11u,12u,14u,15u,13u,6u,7u};
        last_schedule_.device_generated_bytes=device_partitions
            ? bytes[2]+bytes[3]+
                (device_admission?bytes[4]+bytes[5]:0u):0u;
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
        // The contact shader addresses anchors backward from contactBias.
        // Keep the compact anchor region immediately before contact rows.
        const auto input_end=offsets[6];
        host_.resize(std::max(input_end,host_result_bytes));
        for(std::size_t i=0;i<count;++i)
            if(i!=6u && i!=7u && i!=11u && i!=12u &&
               i!=13u && i!=15u && bytes[i] &&
               (!device_initial_bodies || i!=0u) &&
               (!device_partitions || (i!=2u && i!=3u)) &&
               (!device_admission || (i!=4u && i!=5u)))
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
            const std::size_t header_bytes=source_descriptors
                ?(device_admission?44u:24u):20u;
            const std::array<std::uint32_t,5> scratch_header{
                narrow(header_bytes),narrow(header_bytes+n*16u),
                narrow(header_bytes+n*20u),
                narrow(header_bytes+n*24u),
                tiles_.empty()?0u:narrow(offsets[14])};
            std::memcpy(host_.data()+offsets[15],scratch_header.data(),
                        sizeof(scratch_header));
            if(source_descriptors) {
                const std::uint32_t mode=device_admission?2u:1u;
                std::memcpy(host_.data()+offsets[15]+20u,
                            &mode,sizeof(mode));
                if(n && !device_admission)
                    std::memcpy(host_.data()+offsets[15]+scratch_header[1],
                                source_descriptor_tokens_.data(),
                                n*sizeof(std::uint32_t));
                const auto table=offsets[15]+scratch_header[3]+
                    bodies.size()*12u;
                const auto total=narrow(source_descriptor_full_.size());
                std::memcpy(host_.data()+table,&total,sizeof(total));
                if(total)
                    std::memcpy(host_.data()+table+4u,
                                source_descriptor_full_.data(),
                                source_descriptor_full_.size()*
                                    sizeof(native::GpuDyDistanceDescriptor));
            }else if(n)
                std::memcpy(host_.data()+offsets[15]+20u,
                            descriptors.data(),descriptors.size_bytes());
        }
        if(!prepared_velocity_components_.empty())
            std::memcpy(host_.data()+offsets[12]+packed_input_bytes,
                        prepared_velocity_components_.data(),velocity_bytes);
        if(!force_mod_components_.empty())
            std::memcpy(host_.data()+offsets[12]+force_begin,
                        force_mod_components_.data(),force_bytes);
        if(!host_poses_.empty())
            std::memcpy(host_.data()+offsets[12]+host_pose_begin,
                        host_poses_.data(),last_host_pose_upload_bytes_);
        if(upload_body_masses)
            for(std::size_t i=0;i<bodies.size();++i)
                std::memcpy(host_.data()+offsets[12]+mass_offset+i*4u,
                            &bodies[i].inverse_mass,sizeof(float));
        // The source GPU kernel constructs all writable Gu contact rows.
        // One ordered source upload covers every other active input and
        // stops before the contact-row arena, without scattered HtoD spans.
        // Retain both the source friction prefix and stable joint slots.
        // A friction growth moves the joint base and reuploads its records.
        const auto old_joint_base=(
            friction_history_capacity_*64u+127u)&~std::size_t(127u);
        const auto preserved_bytes=friction_history_capacity_==
                next_history_capacity_
            ?old_joint_base+joint_source_capacity_*
                sizeof(native::GpuDyJointInput)
            :friction_history_capacity_*64u;
        arena_.allocateCopyPrefixAsync(
            total,preserved_bytes,stream_,PX_FL);
        friction_history_capacity_=next_history_capacity_;
        joint_source_capacity_=next_joint_capacity;
        const auto address=arena_.getDevicePtr();
        // Sc supplies the descriptor tokens and Dy computes the raw
        // descriptors, colors, cursors, and body partition state. None
        // of those scratch arrays has a host source in a device-owned
        // island epoch. Skip their entire contiguous upload window.
        const auto device_skip_begin=device_admission
            ?offsets[15]+44u:0u;
        const auto device_skip_end=device_admission
            ?device_skip_begin+descriptors.size()*24u+bodies.size()*12u:0u;
        if(device_admission &&
           (device_skip_begin<upload_begin ||
            device_skip_end<device_skip_begin ||
            device_skip_end>input_end))
            throw std::out_of_range(
                "Source Sc numerical scratch upload window is invalid");
        const std::size_t uploaded_bytes=input_end-upload_begin-
            (device_admission?device_skip_end-device_skip_begin:0u);
        // Adjacent Sc constraint slot edits share one DMA into the
        // resident Dy allocation, not the transient solver-row upload.
        joint_source_uploads_.clear();
        for(std::size_t i=0u;i<joint_source_dirty_.size();) {
            const auto first=joint_source_dirty_[i];
            auto last=first+1u;
            ++i;
            while(i<joint_source_dirty_.size() &&
                  joint_source_dirty_[i]==last) {
                ++last;
                ++i;
            }
            joint_source_uploads_.push_back({
                address+joint_base+std::size_t(first)*
                    sizeof(native::GpuDyJointInput),
                joint_source_host_.data()+first,
                std::size_t(last-first)*sizeof(native::GpuDyJointInput)});
            last_schedule_.device_joint_metadata_upload_bytes+=
                std::size_t(last-first)*sizeof(native::GpuDyJointInput);
        }
        last_schedule_.device_joint_metadata_upload_spans=
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
        std::array<std::uint32_t,10> integration_header{
            0u,0u,0u,shape_capacity,narrow(body_ordinals.size()),
            narrow(descriptors.size()),narrow(tasks.size()),
            device_partitions?narrow(std::size_t(params[28])+offsets[15]):
                (tiles_.empty()?0u:narrow(std::size_t(params[28])+offsets[14])),
            narrow(integration.size()),
            device_initial_bodies
                ? (device_world_motion?invalid
                    :narrow(std::size_t(params[28])+offsets[12]+mass_offset))
                :0u};
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
        if(convex_vertex_count)
            dispatch.buffers.push_back(source_native_view(
                addresses_,convex_vertices,
                source_allocation_bytes(convex_vertex_count,12u),
                alignment_,6u,compute::BufferAccess::Read,
                params[27],max_view_));
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
            std::memcpy(host_.data()+offsets[12]+integration_header_bytes,
                        packed_integration_.data(),
                        packed_integration_.size()*sizeof(SourceDyPackedIntegrationNative));
        // Device-authored joint rows carry their source endpoint ordinals
        // until geometric preparation replaces the transient first-row
        // axis. The retained constraint pointer remains at +60.
        for(const auto& descriptor:descriptors) {
            if((descriptor.row_count&255u)==0u)continue;
            const auto& first=rows[descriptor.index];
            if(first.reserved!=2u)continue;
            if(source_descriptors) {
                const std::array<std::uint32_t,3> source{
                    descriptor.row0,descriptor.row1,descriptor.row_count};
                std::memcpy(host_.data()+offsets[1]+
                        std::size_t(descriptor.index)*
                            sizeof(native::GpuDyJointRow),
                            source.data(),sizeof(source));
            }
            const auto index=std::bit_cast<std::uint32_t>(first.applied_force);
            const auto slot=joint_inputs[index].source_slot;
            const auto address=narrow(std::size_t(params[28])+joint_base+
                std::size_t(slot)*sizeof(native::GpuDyJointInput));
            std::memcpy(host_.data()+offsets[1]+
                            std::size_t(descriptor.index)*sizeof(native::GpuDyJointRow)+
                            offsetof(native::GpuDyJointRow,applied_force),
                        &address,sizeof(address));
        }
        // Only newly admitted or edited actor-local transforms are
        // uploaded. Neighboring BodyIds share one source DMA span.
        std::sort(pending_actor_locals_.begin(),pending_actor_locals_.end());
        pending_actor_locals_.erase(std::unique(
            pending_actor_locals_.begin(),pending_actor_locals_.end()),
            pending_actor_locals_.end());
        actor_local_uploads_.clear();
        for(std::size_t i=0u;i<pending_actor_locals_.size();) {
            const auto first=pending_actor_locals_[i];
            auto last=first+1u;
            ++i;
            while(i<pending_actor_locals_.size() &&
                  pending_actor_locals_[i]==last) {
                ++last;
                ++i;
            }
            const auto count=last-first;
            actor_local_uploads_.push_back({
                actor_locals_.getDevicePtr()+
                    first*sizeof(SourceDyActorLocalNative),
                actor_local_host_.data()+first,
                count*sizeof(SourceDyActorLocalNative)});
            last_schedule_.actor_local_upload_bytes+=
                count*sizeof(SourceDyActorLocalNative);
        }
        last_schedule_.actor_local_upload_spans=actor_local_uploads_.size();
        std::memcpy(host_.data()+offsets[12],
                    integration_header.data(),sizeof(integration_header));
        const auto previous_spans=stream_.submission_stats().upload_spans;
        const auto previous_batches=stream_.submission_stats().upload_batches;
        auto& uploads=stream_.upload_scratch();
        uploads.reserve(joint_source_uploads_.size()+
                        actor_local_uploads_.size()+
                        (device_admission?2u:1u));
        uploads.insert(uploads.end(),joint_source_uploads_.begin(),
                       joint_source_uploads_.end());
        uploads.insert(uploads.end(),actor_local_uploads_.begin(),
                       actor_local_uploads_.end());
        if(device_admission) {
            // The device-generated scratch window has no host input.
            uploads.push_back({
                address+upload_begin,host_.data()+upload_begin,
                device_skip_begin-upload_begin});
            uploads.push_back({
                address+device_skip_end,host_.data()+device_skip_end,
                input_end-device_skip_end});
        }else uploads.push_back({
            address+upload_begin,host_.data()+upload_begin,uploaded_bytes});
        // These disjoint source spans share one ordered DMA submission.
        stream_.queue_uploads(uploads);
        last_schedule_.device_upload_batches=
            stream_.submission_stats().upload_batches-previous_batches;
        // The Sc producer and Dy consumer use the same ordered source
        // stream. GPU-to-GPU copies replace the transient task/ordinal/
        // token inputs after the shared host upload; no Sc witness fetch
        // or host graph readback occurs between their submissions.
        if(device_admission) {
            const auto& view=*device_admission;
            const auto& layout=view.layout;
            const auto in_range=[&](std::size_t offset,std::size_t bytes) {
                return offset<=layout.bytes && bytes<=layout.bytes-offset;
            };
            if(!in_range(0u,16u) ||
               layout.tasks>layout.block_offsets ||
               layout.body_order>layout.response_tokens ||
               layout.response_tokens>layout.bytes ||
               tasks.size_bytes()>layout.block_offsets-layout.tasks ||
               body_ordinals.size_bytes()>
                   layout.response_tokens-layout.body_order ||
               descriptors.size()> (layout.bytes-layout.response_tokens)/
                   sizeof(std::uint32_t))
                throw std::out_of_range(
                    "Source Sc device admission capacity exceeds its typed stream");
            stream_.queue_device_copy(address+offsets[15]+24u,
                                      view.work,20u);
            stream_.queue_device_copy(address+offsets[5],
                                      view.work+layout.tasks,tasks.size_bytes());
            if(!body_ordinals.empty())
                stream_.queue_device_copy(address+offsets[4],
                    view.work+layout.body_order,body_ordinals.size_bytes());
            if(!descriptors.empty())
                stream_.queue_device_copy(address+offsets[15]+44u+
                    descriptors.size()*16u,
                    view.work+layout.response_tokens,
                    descriptors.size()*sizeof(std::uint32_t));
        }
        last_upload_spans_=stream_.submission_stats().upload_spans-
                           previous_spans-joint_source_uploads_.size()-
                           actor_local_uploads_.size();
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
            const bool writes_actor_velocities=integrates ||
                (device_unconstrained_preparation &&
                 (phase==Launch::Color || phase==Launch::InitializeBodies));
            dispatch.buffers[5].access=writes_actor_velocities
                ?compute::BufferAccess::ReadWrite
                :compute::BufferAccess::Read;
            params[10]=(iteration<<4u)|static_cast<std::uint32_t>(phase)|
                (compact_friction?0x20000000u:0u)|
                (device_partitions?0x80000000u:0u)|
                (device_partitions && !wide_tasks_.empty() &&
                 phase!=Launch::Color ?0x40000000u:0u);
            // Compact tasks have no cross-workgroup dependencies. Their
            // workgroup count is published by the Sc device prefix, so
            // dispatch directly from that result instead of recopying a
            // host task count into the launch command.
            if(device_admission && phase==Launch::Fused &&
               wide_tasks_.empty()) {
                if(range.begin!=0u || range.count!=tasks.size() ||
                   range.count>max_groups)
                    throw std::out_of_range(
                        "Source device Sc task launch exceeds compact dispatch bounds");
                params[8]=0u;
                dispatch.groups={1u,1u,1u};
                dispatch.indirect=device_admission->task_dispatch;
                std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
                stream_.commands().emplace_back(dispatch);
                dispatch.indirect.reset();
                ++last_schedule_.dispatches;
                ++last_schedule_.device_island_indirect_dispatches;
                return;
            }
            dispatch.indirect.reset();
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
        const auto launch_packed=[&](std::size_t first,std::size_t count) {
            dispatch.buffers[2].access=compute::BufferAccess::ReadWrite;
            dispatch.buffers[3].access=compute::BufferAccess::Write;
            dispatch.buffers[5].access=compute::BufferAccess::ReadWrite;
            dispatch.indirect.reset();
            const auto maximum=std::min<std::size_t>(
                std::size_t(max_groups)*128u,0x00ffffffu);
            for(std::size_t offset=0u;offset<count;) {
                const auto task_count=std::min(maximum,count-offset);
                params[8]=narrow(first+offset);
                params[10]=(narrow(task_count)<<4u)|
                    static_cast<std::uint32_t>(Launch::Fused)|
                    0x10000000u|0x80000000u|
                    (compact_friction?0x20000000u:0u);
                dispatch.groups={narrow((task_count+127u)/128u),1u,1u};
                std::memcpy(dispatch.push_constants.data(),params.data(),sizeof(params));
                stream_.commands().emplace_back(dispatch);
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
           (tasks.size()!=1u || bodies.size()>128u)) {
            const auto before=last_schedule_.dispatches;
            launch(Launch::InitializeBodies,
                {0u,(bodies.size()+127u)/128u});
            last_schedule_.device_initialization_dispatches=
                last_schedule_.dispatches-before;
        }
        if(device_partitions && !wide_tasks_.empty())
            launch(Launch::Color,{0u,tasks.size()+
                (device_initial_bodies ? (bodies.size()+127u)/128u : 0u)});
        if(last_schedule_.host_lane_packed_tasks) {
            for(std::size_t first=0u;first<tasks.size();) {
                const bool packed=lane_task(tasks[first]);
                std::size_t end=first+1u;
                while(end<tasks.size() && lane_task(tasks[end])==packed)
                    ++end;
                if(packed)launch_packed(first,end-first);
                else launch(Launch::Fused,{first,end-first});
                first=end;
            }
        }else launch(Launch::Fused,fused_tiles_);
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
        }else if(device_admission) {
            // Even compact tasks must report a failed resident Sc epoch,
            // including an indirect launch with a mismatched task count.
            launch(Launch::PublishErrors,{0u,(tasks.size()+127u)/128u});
        }
        // The integration extent fits in launchControl's upper bits
        // because the packed arena is limited to 32-bit byte offsets.
        launch(Launch::FreeBodies,{0u,(free_bodies+127u)/128u},
               narrow(integration.size()));
        const auto mapped=addresses_.resolve(address+history_bytes,output_bytes);
        compute::BufferId device_meta_buffer=0u;
        std::size_t device_meta_offset=0u,device_task_source=0u,
                    device_ordinal_source=0u;
        if(device_admission) {
            const auto metadata=addresses_.resolve(
                address+offsets[15]+24u,16u);
            const auto device_tasks=addresses_.resolve(
                address+offsets[5],tasks.size_bytes());
            const auto device_ordinals=addresses_.resolve(
                address+offsets[4],
                std::max<std::size_t>(4u,body_ordinals.size_bytes()));
            if(metadata.buffer!=device_tasks.buffer ||
               metadata.buffer!=device_ordinals.buffer)
                throw std::logic_error(
                    "Device Dy scheduling spans crossed the numerical allocation");
            device_meta_buffer=metadata.buffer;
            device_meta_offset=metadata.offset;
            device_task_source=device_tasks.offset;
            device_ordinal_source=device_ordinals.offset;
        }
        // Numerical dispatch is ordered after PCM and host Gu uploads,
        // but does not consume the outstanding Sc witness readback.
        // Result publication joins only when Sc explicitly needs Dy.
        stream_.submit();
        pending_solve_.emplace(PendingSolve{
            .bodies=bodies,.tasks=tasks,.contact_forces=contact_forces,
            .friction=friction,.anchor_impulses=anchor_impulses,
            .patches=patches,.integration=integration,
            .integrated_poses=integrated_poses,.sleep_states=sleep_states,
            .descriptors=descriptors,.body_ordinals=body_ordinals,
            .manifolds=manifolds,.contact_order=contact_order,
            .contact_prep=contact_prep,.result_offsets=result_offsets,
            .output_buffer=mapped.buffer,.output_offset=mapped.offset,
            .output_bytes=output_bytes,
            .device_meta_buffer=device_meta_buffer,
            .device_meta_source=device_meta_offset,
            .device_task_source=device_task_source,
            .device_ordinal_source=device_ordinal_source,
            .device_admission=device_admission.has_value(),
            .device_header_offset=device_header_offset,
            .device_task_offset=device_task_offset,
            .device_ordinal_offset=device_ordinal_offset,
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
    }
    bool has_queued_solve() const noexcept {
        return pending_solve_.has_value();
    }
    void finish() {
        if(!pending_solve_)
            throw std::logic_error("No queued source Dy numerical epoch");
        // The same Context owns PCM witness staging. A premature host
        // join must not discard Dy's already submitted numerical result.
        if(stream_.has_pending_fetch())
            throw std::logic_error(
                "Source Dy result join precedes pending PCM witness DMA");
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
        const auto contact_prep=pending.contact_prep;
        const auto result_offsets=pending.result_offsets;
        const auto output_bytes=pending.output_bytes;
        const auto free_velocity_offset=pending.free_velocity_offset;
        const auto friction_count=pending.friction_count;
        const auto uploaded_bytes=pending.uploaded_bytes;
        const auto reused_pcm_rows=pending.reused_pcm_rows;
        const auto compact_anchor_bytes=pending.compact_anchor_bytes;
        const auto prep_bytes=pending.prep_bytes;
        const auto inline_body_bytes=pending.inline_body_bytes;
        const auto compact_friction=pending.compact_friction;
        const auto device_unconstrained_preparation=
            pending.device_unconstrained_preparation;
        const auto device_world_motion=pending.device_world_motion;
        const std::array<compute::DownloadSpan,4> downloads{{
            {pending.output_buffer,pending.output_offset,
             host_.data(),output_bytes},
            {pending.device_meta_buffer,pending.device_meta_source,
             host_.data()+pending.device_header_offset,16u},
            {pending.device_meta_buffer,pending.device_task_source,
             host_.data()+pending.device_task_offset,tasks.size_bytes()},
            {pending.device_meta_buffer,pending.device_ordinal_source,
             host_.data()+pending.device_ordinal_offset,
             body_ordinals.size_bytes()}
        }};
        const std::size_t download_count=!pending.device_admission?1u:
            body_ordinals.empty()?3u:4u;
        stream_.fetch(std::span(downloads.data(),download_count));
        admitted_body_ordinals_.clear();
        bool invalid_device_admission=false;
        if(pending.device_admission) {
            std::array<std::uint32_t,4> admitted{};
            std::memcpy(admitted.data(),
                host_.data()+pending.device_header_offset,sizeof(admitted));
            invalid_device_admission=
                admitted[0]!=0u || admitted[1]>tasks.size() ||
                admitted[2]>body_ordinals.size() ||
                admitted[3]>descriptors.size();
            if(!invalid_device_admission) {
                last_schedule_.device_admitted_tasks=admitted[1];
                last_schedule_.device_admitted_bodies=admitted[2];
                last_schedule_.device_admitted_descriptors=admitted[3];
                if(admitted[1])
                    std::memcpy(tasks.data(),
                        host_.data()+pending.device_task_offset,
                        admitted[1]*sizeof(native::GpuDyDistanceTask));
                for(std::size_t i=admitted[1];i<tasks.size();++i)
                    tasks[i].error=0u;
                tasks=tasks.first(admitted[1]);
                admitted_body_ordinals_.resize(admitted[2]);
                if(admitted[2])
                    std::memcpy(admitted_body_ordinals_.data(),
                        host_.data()+pending.device_ordinal_offset,
                        admitted[2]*sizeof(std::uint32_t));
                body_ordinals=admitted_body_ordinals_;
            }
        }
        for(const auto id:pending_actor_locals_)
            actor_local_valid_[id]=1u;
        for(std::size_t i=0;i<tasks.size();++i)
            std::memcpy(&tasks[i].error,
                host_.data()+result_offsets[1]+i*4u,4u);
        for(std::size_t i=0u;i<tasks.size();++i)
            if(tasks[i].error!=0u)
                throw std::overflow_error(
                    "Source Dy GPU solver task "+std::to_string(i)+
                    " rejected flags "+std::to_string(tasks[i].error));
        if(invalid_device_admission)
            throw std::overflow_error(
                "Resident Sc Dy scheduling result exceeds declared capacities");
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
            if(integration[index].active&1u)fetch_pose(index);
        }
        for(std::size_t index=bodies.size();index<integration.size();++index)
            fetch_pose(index);
        if(device_unconstrained_preparation)
            for(std::size_t index=bodies.size();index<integration.size();++index) {
                auto& input=integration[index];
                const auto* output=host_.data()+free_velocity_offset+
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
            if(manager.order_offset>contact_forces.size() ||
               manager.contact_count>
                   contact_forces.size()-manager.order_offset ||
               manager.patch_offset>patches.size() ||
               manager.patch_count>patches.size()-manager.patch_offset)
                throw std::logic_error("Source Dy contact writeback range invalid");
            // The source contact order is implicit for PCM patches:
            // their Gu points already occupy the resident solver
            // arena consecutively. CPU Gu patches retain their
            // explicit correlation order, packed without PCM holes.
            std::size_t written=0u;
            for(std::size_t patch_index=manager.patch_offset;
                patch_index<manager.patch_offset+manager.patch_count;
                ++patch_index) {
                const auto& patch=patches[patch_index];
                constexpr std::uint32_t direct=0x80000000u;
                const bool resident=(patch.order_offset&direct)!=0u;
                const std::size_t begin=patch.order_offset&~direct;
                const std::size_t limit=resident
                    ?contact_forces.size():contact_order.size();
                if(begin>limit || patch.contact_count>limit-begin ||
                   written>manager.contact_count ||
                   patch.contact_count>manager.contact_count-written)
                    throw std::logic_error(
                        "Source Dy contact patch order is invalid");
                for(std::size_t k=0u;k<patch.contact_count;++k) {
                    const auto index=resident
                        ?begin+k:contact_order[begin+k];
                    if(index<manager.order_offset ||
                       index>=manager.order_offset+manager.contact_count)
                        throw std::logic_error(
                            "Source Dy contact writeback index invalid");
                    std::memcpy(&contact_forces[index],
                        host_.data()+result_offsets[2]+index*4u,4u);
                }
                written+=patch.contact_count;
            }
            if(written!=manager.contact_count)
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
                        host_.data()+result_offsets[4]+j*4u,4u);
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
                            host_.data()+result_offsets[3]+k*12u,12u);
                }else {
                    for(std::size_t k=patch.row_offset;
                        k<patch.row_offset+count*2u;++k) {
                        const auto* result=
                            host_.data()+result_offsets[3]+k*16u;
                        std::memcpy(&friction[k].tangent,result,12u);
                        std::memcpy(&friction[k].applied_force,result+12u,4u);
                    }
                }
                if(pcm_growth) {
                    patch.anchor_count=count;
                    patch.reserved=report;
                    patch.broken=report&1u;
                }else if(count)patch.broken=report;
            }
        };
        for(const auto& task:tasks) {
            for(std::size_t i=task.body_begin;i<task.body_end;++i) {
                const auto index=body_ordinals[i];
                if(index>=bodies.size())
                    throw std::logic_error("Source Dy body writeback index invalid");
                if(device_unconstrained_preparation) {
                    const auto* output=host_.data()+result_offsets[0]+index*52u;
                    std::memcpy(&bodies[index],output,24u);
                    std::memcpy(&sleep_states[index],output+24u,
                                sizeof(native::GpuDySleepState));
                }else
                    std::memcpy(&bodies[index],
                        host_.data()+result_offsets[0]+index*48u,48u);
            }
        }
        if(pending.device_admission) {
            for(const auto& desc:descriptors)
                publish_descriptor(desc);
        }else for(const auto& task:tasks)
            for(std::size_t i=task.descriptor_begin;
                i<task.descriptor_end;++i)
                publish_descriptor(descriptors[i]);
        // Sc's final world velocities are the device pool's next-frame
        // baseline. Update the host comparison mirror from the same
        // fetched result, never by replaying PGS on the CPU.
        if(device_world_motion) {
            for(const auto index:body_ordinals) {
                if(!integration[index].active)continue;
                const auto id=integration[index].source_actor;
                std::memcpy(actor_local_host_[id].linear_velocity,
                            bodies[index].linear_delta,
                            sizeof(actor_local_host_[id].linear_velocity));
                std::memcpy(actor_local_host_[id].angular_velocity,
                            bodies[index].angular_state,
                            sizeof(actor_local_host_[id].angular_velocity));
                actor_velocity_valid_[id]=1u;
            }
            for(std::size_t index=bodies.size();index<integration.size();++index) {
                const auto& input=integration[index];
                if(!(input.active&1u))continue;
                const auto id=input.source_actor;
                float linear[3]{
                    input.linear_velocity[0],input.linear_velocity[1],
                    input.linear_velocity[2]};
                for(unsigned axis=0u;axis<3u;++axis)
                    if(input.lock_flags&(1u<<axis))linear[axis]=0.0f;
                std::memcpy(actor_local_host_[id].linear_velocity,linear,
                            sizeof(linear));
                std::memcpy(actor_local_host_[id].angular_velocity,
                            input.angular_velocity,
                            sizeof(actor_local_host_[id].angular_velocity));
                actor_velocity_valid_[id]=1u;
            }
        }
        if(device_unconstrained_preparation) {
            const auto mirror_sleep=[&](std::size_t index) {
                const auto& state=sleep_states[index];
                for(const auto component:state.linear_acc)
                    if(!std::isfinite(component))
                        throw std::overflow_error(
                            "Source GPU sleep linear accumulator is not finite");
                for(const auto component:state.angular_acc)
                    if(!std::isfinite(component))
                        throw std::overflow_error(
                            "Source GPU sleep angular accumulator is not finite");
                if(!std::isfinite(state.wake_counter) ||
                   state.wake_counter<0.0f)
                    throw std::overflow_error(
                        "Source GPU wake counter is not finite");
                const auto id=integration[index].source_actor;
                auto& resident=actor_local_host_[id];
                resident.wake_counter=state.wake_counter;
                std::memcpy(resident.sleep_linear_acc,state.linear_acc,
                            sizeof(state.linear_acc));
                std::memcpy(resident.sleep_angular_acc,state.angular_acc,
                            sizeof(state.angular_acc));
            };
            for(const auto ordinal:body_ordinals)
                if(integration[ordinal].active&1u)mirror_sleep(ordinal);
            for(std::size_t index=bodies.size();
                index<integration.size();++index)
                if(integration[index].active&1u)mirror_sleep(index);
        }
        if(device_world_motion) {
            const auto mirror_pose=[&](std::size_t index) {
                const auto id=integration[index].source_actor;
                auto& mirror=actor_local_host_[id];
                const auto& result=integrated_poses[index];
                std::memcpy(mirror.world_position,result.position,
                            sizeof(mirror.world_position));
                std::memcpy(mirror.world_orientation,result.orientation,
                            sizeof(mirror.world_orientation));
                actor_pose_valid_[id]=1u;
            };
            for(const auto ordinal:body_ordinals)
                if(integration[ordinal].active&1u)mirror_pose(ordinal);
            for(std::size_t index=bodies.size();index<integration.size();++index)
                if(integration[index].active&1u)mirror_pose(index);
        }
        last_transfer_bytes_={uploaded_bytes,
            output_bytes+(pending.device_admission
                ?16u+pending.tasks.size_bytes()+
                    pending.body_ordinals.size_bytes():0u)};
        last_reused_pcm_contact_rows_=reused_pcm_rows;
        last_omitted_pcm_upload_bytes_=
            reused_pcm_rows*sizeof(native::GpuDyContactPoint);
        last_cpu_gu_geometry_bytes_=pending.cpu_gu_bytes;
        last_friction_anchor_upload_bytes_=compact_anchor_bytes;
        for(const auto& manager:manifolds) {
            if(!manager.patch_count || manager.prep_index>=contact_prep.size())
                continue;
            const auto& pair=contact_prep[manager.prep_index];
            if(pair.pcm_pair_index!=0xffffffffu)
                history_epochs_[pair.pcm_edge]=history_epoch_;
        }
        last_friction_prepared_rows_=friction_count;
        // The device accepted each changed joint only after all source
        // task results and their lifetime checks were published.
        for(const auto slot:joint_source_dirty_)
            joint_source_valid_[slot]=1u;
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
        std::span<const native::GpuDyContactManifold> manifolds{};
        std::span<const std::uint32_t> contact_order{};
        std::span<const native::GpuDyContactPrepPair> contact_prep{};
        std::array<std::size_t,6> result_offsets{};
        compute::BufferId output_buffer=0u;
        std::size_t output_offset=0u,output_bytes=0u;
        compute::BufferId device_meta_buffer=0u;
        std::size_t device_meta_source=0u,device_task_source=0u,
                    device_ordinal_source=0u;
        bool device_admission=false;
        std::size_t device_header_offset=0u,device_task_offset=0u,
                    device_ordinal_offset=0u;
        std::size_t free_velocity_offset=0u,friction_count=0u;
        std::size_t uploaded_bytes=0u,cpu_gu_bytes=0u,reused_pcm_rows=0u;
        std::size_t compact_anchor_bytes=0u,prep_bytes=0u;
        std::size_t inline_body_bytes=0u;
        bool compact_friction=false;
        bool device_unconstrained_preparation=false;
        bool device_world_motion=false;
    };
    std::optional<PendingSolve> pending_solve_{};
    enum class Launch : std::uint32_t {
        Fused, Normals, Friction, Partition, Motion, Integrate, PublishErrors,
        FreeBodies, Color, InitializeBodies
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
        std::uint32_t partition_count,
        std::uint32_t ordinals,std::uint32_t bodies,
        bool resident_admission) {
        constexpr std::uint32_t threads=128u;
        constexpr std::uint32_t prep_managers=threads/4u;
        constexpr std::uint32_t wide_min=native::gpu_dy_wide_task_threshold;
        wide_tasks_.clear();
        tiles_.clear();
        partition_tiles_.clear();
        position_ends_.fill(false);
        wide_iterations_=0u;
        last_schedule_={};
        last_schedule_.device_colored_tasks=tasks.size();
        last_schedule_.device_colored_descriptors=descriptors.size();
        last_schedule_.max_partition_groups=tasks.empty()?0u:1u;
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
            fused_tiles_={0u,tasks.size()};
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
        // Device colors have not been computed yet. Upper bounds come
        // from endpoint degrees, not a host conflict-colored schedule.
        // Each tile resolves its actual color interval in the shader.
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
        std::size_t end=tiles_.size();
        for(auto& range:partition_tiles_) {
            range.begin=end;
            end+=range.count;
            range.count=0u;
        }
        tiles_.resize(end);
        for(const auto id:wide_tasks_) {
            const auto& task=tasks[id];
            const auto endpoints=partition_endpoint_counts_[id];
            for(std::uint32_t color=0u;color<color_bounds_[id];++color) {
                auto& range=partition_tiles_[color];
                for(std::uint32_t offset=0u;offset<endpoints;) {
                    const auto next=offset+std::min(threads,endpoints-offset);
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
        constexpr std::uint32_t wide_min=native::gpu_dy_wide_task_threshold;
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
    physx::PxgCudaBuffer actor_locals_;
    compute::PipelineId pipeline_=0;
    compute::PipelineId convex_pipeline_=0;
    std::size_t alignment_=0u;
    std::uint64_t max_view_=0u;
    std::vector<std::uint8_t> host_;
    std::vector<std::uint32_t> admitted_body_ordinals_;
    std::vector<SourceDyPackedContactPrepNative> packed_;
    std::vector<std::uint32_t> source_descriptor_tokens_;
    std::vector<native::GpuDyDistanceDescriptor> source_descriptor_full_;
    std::vector<SourceDyPackedIntegrationNative> packed_integration_;
    std::vector<SourceDyHostPoseNative> host_poses_;
    std::vector<SourceDyActorLocalNative> actor_local_host_;
    std::vector<std::uint8_t> actor_local_valid_,actor_velocity_valid_,actor_pose_valid_;
    std::vector<float> prepared_velocity_components_,force_mod_components_;
    std::vector<std::uint32_t> pending_actor_locals_;
    std::vector<SourceSapUpload> actor_local_uploads_;
    std::size_t actor_local_capacity_=0u;
    std::vector<native::GpuDyJointInput> joint_source_host_;
    std::vector<std::uint8_t> joint_source_valid_;
    std::vector<std::uint32_t> joint_source_dirty_;
    std::vector<SourceSapUpload> joint_source_uploads_;
    std::size_t joint_source_capacity_=0u;
    std::vector<native::GpuDyContactPrepBody> inline_bodies_;
    std::vector<native::GpuDyFrictionAnchorInput> compact_anchors_;
    std::vector<std::size_t> patch_anchor_starts_;
    std::vector<std::uint8_t> assigned_;
    std::vector<std::uint32_t> wide_tasks_;
    std::vector<std::uint32_t> degree_dynamic_,degree_static_,color_bounds_;
    std::vector<std::uint32_t> partition_endpoint_counts_;
    std::vector<WorkTile> tiles_;
    std::vector<TileRange> partition_tiles_;
    TileRange fused_tiles_,prep_tiles_,body_tiles_;
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
    std::size_t friction_history_capacity_=0u;
    std::vector<std::uint32_t> history_epochs_;
    std::uint32_t history_epoch_=1u;
    std::size_t last_friction_prepared_rows_=0u;
    std::int64_t last_contact_prep_upload_saving_=0;
};

} // namespace fyzz::physx_gpu
