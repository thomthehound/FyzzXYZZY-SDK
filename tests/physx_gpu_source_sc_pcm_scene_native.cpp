// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"
#include "gpu_integration_metrics.h"
#include "compound_rigid_cases.h"
#include "convex_gpu_case.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fyzz::physx_gpu {
struct SourceScDeviceIslandFixture {
    static bool pcm_ready(const SourceScOwnedGpuBroadphaseNative& owner) {
        return owner.pcm_ && owner.pcm_->device_contact_ready();
    }
    static std::string contact_records(SourceScOwnedGpuBroadphaseNative& owner) {
        const auto count=owner.device_pcm_contact_count();
        const auto source=owner.device_pcm_contact_stream();
        if(!count || !source)return {};
        const auto bytes=std::size_t(count)*sizeof(native::GpuPcmBoxResult);
        const auto view=owner.addresses_.resolve(source,bytes);
        std::vector<native::GpuPcmBoxResult> records(count);
        owner.context_.download(view.buffer,view.offset,records.data(),bytes);
        std::string result;
        for(std::size_t i=0u;i<records.size();++i)
            result+=" [slot="+std::to_string(i)+
                " edge="+std::to_string(records[i].edge)+
                " count="+std::to_string(records[i].count())+
                " transition="+std::to_string(static_cast<std::uint32_t>(
                    records[i].touch_transition()))+"]";
        return result;
    }
    static std::array<std::uint32_t,2> touch(
            SourceScOwnedGpuBroadphaseNative& owner,std::uint32_t edge) {
        if(edge>=owner.device_pcm_touch_capacity())
            return {0xffffffffu,0xffffffffu};
        const auto address=owner.device_pcm_touch_history()+
            std::size_t(edge)*240u+72u;
        const auto view=owner.addresses_.resolve(address,8u);
        std::array<std::uint32_t,2> state{};
        owner.context_.download(view.buffer,view.offset,
                                state.data(),sizeof(state));
        return state;
    }
    static SourceScDeviceIslandHeader read(SourceScOwnedGpuBroadphaseNative& owner) {
        if(!owner.pcm_ || !owner.pcm_->device_island_work_address())
            throw std::runtime_error("Device Sc task work was not queued");
        // An ordered direct download observes this task result without
        // touching the pending PCM witness DMA or its source tokens.
        // Production Dy consumes the buffer without a host observation.
        const auto layout=owner.pcm_->device_island_layout();
        const auto resolved=owner.addresses_.resolve(
            owner.pcm_->device_island_work_address(),layout.bytes);
        std::vector<std::byte> work(layout.bytes);
        owner.context_.download(resolved.buffer,resolved.offset,
                                work.data(),work.size());
        SourceScDeviceIslandHeader header{};
        std::memcpy(&header,work.data(),sizeof(header));
        if(header.errors==0u) {
            if(layout.admissions+
                   std::uint64_t(header.source_edges)*
                       sizeof(SourceScDeviceIslandAdmission)>work.size() ||
               layout.response_order+
                   std::uint64_t(header.admitted_edges)*4u>work.size() ||
               layout.response_tokens+
                   std::uint64_t(header.admitted_edges)*4u>work.size() ||
               layout.body_order+
                   std::uint64_t(header.responding_nodes)*4u>work.size())
                throw std::runtime_error("Device Sc response work exceeds its allocation");
            std::uint32_t previous_root=0u,previous_edge=0u;
            std::uint32_t previous_kind=0u;
            std::uint32_t previous_body_root=0u,previous_body=0u;
            for(std::uint32_t i=0u;i<header.responding_nodes;++i) {
                std::uint32_t node=0u;
                std::memcpy(&node,work.data()+layout.body_order+i*4u,4u);
                if(node>=((layout.counts-layout.roots)/sizeof(std::uint32_t)))
                    throw std::runtime_error(
                        "Device Sc body order references an invalid node");
                std::uint32_t root=0u;
                std::memcpy(&root,work.data()+layout.roots+node*4u,4u);
                if(i!=0u && (root<previous_body_root ||
                   (root==previous_body_root && node<=previous_body)))
                    throw std::runtime_error(
                        "Device Sc body order differs from canonical task roots");
                previous_body_root=root;
                previous_body=node;
            }
            for(std::uint32_t i=0u;i<header.admitted_edges;++i) {
                std::uint32_t edge=0u;
                std::memcpy(&edge,work.data()+layout.response_order+i*4u,4u);
                if(edge>=header.source_edges)
                    throw std::runtime_error("Device Sc response referenced a missing source edge");
                SourceScDeviceIslandAdmission record{};
                std::memcpy(&record,work.data()+layout.admissions+
                            std::size_t(edge)*sizeof(record),sizeof(record));
                const std::uint32_t expected_tag=
                    (record.kind&source_sc_device_island_joint_bit)==0u
                        ?0xc0000000u:
                        ((record.kind&source_sc_device_island_device_joint_bit)!=0u
                            ?0x80000000u:0u);
                std::uint32_t token=0u;
                std::memcpy(&token,work.data()+layout.response_tokens+i*4u,4u);
                const std::uint32_t kind_rank=
                    (record.kind&source_sc_device_island_joint_bit)!=0u
                        ?0u:1u;
                if(record.island_root==0xffffffffu ||
                   (record.kind&source_sc_device_island_no_response_bit)!=0u ||
                   record.source_index>=0x40000000u ||
                   token!=(expected_tag|record.source_index) ||
                   (i!=0u && (record.island_root<previous_root ||
                    (record.island_root==previous_root &&
                     (kind_rank<previous_kind ||
                      (kind_rank==previous_kind && edge<=previous_edge))))))
                    throw std::runtime_error(
                        "Device Sc response identity/order differs from source admission");
                previous_root=record.island_root;
                previous_edge=edge;
                previous_kind=kind_rank;
            }
        }
        const auto launch=owner.pcm_->device_island_indirect_args();
        if(!launch)
            throw std::runtime_error("Device Sc indirect task launch unavailable");
        std::array<std::uint32_t,3> groups{};
        owner.context_.download(launch->buffer,launch->offset,
                                groups.data(),sizeof(groups));
        if(groups!=std::array<std::uint32_t,3>{
                header.dispatch_x,header.dispatch_y,header.dispatch_z})
            throw std::runtime_error(
                "Device Sc indirect launch differs from resident task header");
        return header;
    }
};
} // namespace fyzz::physx_gpu

namespace fyzz::native {
struct GpuPcmFrictionAnchorFixture {
    static void verify_early_speculative_admission(
        const Scene& reference,const Scene& gpu) {
        if(!gpu.speculative_connectivity_prepared_)
            throw std::runtime_error(
                "GPU PCM scene did not overlap speculative IslandSim admission");
        if(reference.speculative_island_count_!=
               gpu.speculative_island_count_ ||
           reference.accurate_island_count_!=
               gpu.accurate_island_count_)
            throw std::runtime_error(
                "Early speculative Sc admission changed source island counts: " +
                std::to_string(reference.speculative_island_count_) + "/" +
                std::to_string(gpu.speculative_island_count_) + " speculative, " +
                std::to_string(reference.accurate_island_count_) + "/" +
                std::to_string(gpu.accurate_island_count_) + " accurate");
        for(std::size_t a=0u;a<reference.actors_.size();++a) {
            if(!reference.actors_[a].alive ||
               reference.actors_[a].motion!=Scene::Motion::Dynamic)continue;
            for(std::size_t b=a+1u;b<reference.actors_.size();++b) {
                if(!reference.actors_[b].alive ||
                   reference.actors_[b].motion!=Scene::Motion::Dynamic)continue;
                const auto eq=[a,b](const auto& graph) {
                    return graph.node_island[a]==graph.node_island[b];
                };
                if(eq(reference.speculative_islands_)!=
                       eq(gpu.speculative_islands_) ||
                   eq(reference.accurate_islands_)!=
                       eq(gpu.accurate_islands_))
                    throw std::runtime_error(
                        "Early speculative Sc admission changed source island connectivity");
            }
        }
    }
    static std::uint32_t contact_manager_edge(const Scene& scene,Pair shapes) {
        return scene.contact_edge_index(shapes);
    }
    static std::string island_source_edges(const Scene& scene) {
        std::string result;
        for(const auto& edge:scene.gpu_dy_island_edges_) {
            result+=" ["+std::to_string(edge.edge_index)+":"+
                std::to_string(edge.source_index)+":"+
                std::to_string(edge.kind)+"]";
        }
        return result;
    }
    static bool source_bp_contains(const Scene& scene,Pair shapes) {
        return scene.sap_pairs_.find(shapes)!=Scene::SapPairManager::invalid;
    }
    static void verify_pcm_contact_arena(const Scene& scene) {
        const auto* base=scene.contact_points_.data();
        for(const auto& manager:scene.manifolds_) {
            if(manager.pcm_result_index==
               std::numeric_limits<std::uint32_t>::max())continue;
            if(manager.count==0u ||
               manager.contact_offset>scene.contact_points_.size() ||
               manager.count>scene.contact_points_.size()-manager.contact_offset ||
               manager.points.size()!=manager.count ||
               manager.points.data()!=base+manager.contact_offset)
                throw std::runtime_error(
                    "Source GPU PCM published outside its Sc Gu contact arena");
        }
    }
    static void verify_device_contact_order(const Scene& scene,
                                            bool require_pcm,
                                            bool require_cpu) {
        std::size_t pcm_count=0u,cpu_count=0u;
        for(const auto index:scene.indexed_managers_) {
            const auto& manager=scene.manifolds_[index];
            if(!manager.friction_patch_count)continue;
            if(manager.pcm_result_index==
               std::numeric_limits<std::uint32_t>::max())
                cpu_count+=manager.count;
            else pcm_count+=manager.count;
        }
        if((require_pcm && !pcm_count) ||
           (require_cpu && !cpu_count) ||
           scene.gpu_dy_contact_order_.size()!=cpu_count ||
           scene.gpu_dy_gu_inputs_.size()!=cpu_count ||
           scene.gpu_dy_contact_forces_.size()!=scene.prepared_contact_rows_ ||
           scene.gpu_dy_friction_anchors_.size()*2u!=
               scene.prepared_friction_rows_ ||
           !scene.solver_contact_points_.empty() ||
           !scene.solver_friction_rows_.empty())
            throw std::runtime_error(
                "GPU Dy duplicated CPU numerical contact rows or lost Gu order");
    }
    static bool had_early_friction_history(const Scene& scene) {
        const auto& masks=scene.gpu_pcm_friction_history_masks_;
        if(!masks.empty() && masks.size()!=scene.gpu_pcm_pairs_.size())
            throw std::runtime_error(
                "Source early friction history lost the PCM manager stream");
        return std::any_of(masks.begin(),masks.end(),
            [](std::uint32_t bits) {return bits!=0u;});
    }
    static void verify_compact_gpu_contact_admission(const Scene& scene) {
        std::size_t manager_begin=0u,joint_begin=0u;
        std::size_t body_begin=0u,descriptor_begin=0u,contact_count=0u;
        for(const auto& task:scene.solver_tasks_) {
            const auto potential=task.joint_end-joint_begin+
                task.manager_end-manager_begin;
            const bool compact=
                potential<gpu_dy_wide_task_threshold ||
                task.solver_body_end-body_begin<gpu_dy_wide_task_threshold;
            for(std::size_t i=descriptor_begin;i<task.constraint_end;++i) {
                const auto& descriptor=scene.ordered_constraints_[i];
                if((descriptor.row_count&255u)!=0u)continue;
                ++contact_count;
                const bool has_endpoints=
                    descriptor.row0!=Scene::invalid_solver_row ||
                    descriptor.row1!=Scene::invalid_solver_row;
                if(compact==has_endpoints)
                    throw std::runtime_error(compact
                        ? "Compact GPU contact retained host solver endpoints"
                        : "Wide GPU contact lost its scheduling endpoints");
            }
            manager_begin=task.manager_end;
            joint_begin=task.joint_end;
            body_begin=task.solver_body_end;
            descriptor_begin=task.constraint_end;
        }
        if(descriptor_begin!=scene.ordered_constraints_.size() ||
           !contact_count)
            throw std::runtime_error(
                "GPU task admission failed to cover source contact descriptors");
    }
    static std::size_t cpu_gu_anchor_bytes(const Scene& scene) {
        std::size_t count=0u;
        for(const auto index:scene.indexed_managers_) {
            const auto& manager=scene.manifolds_[index];
            if(manager.pcm_result_index!=
               std::numeric_limits<std::uint32_t>::max())continue;
            for(std::size_t i=manager.friction_patch_offset;
                i<manager.friction_patch_offset+
                  manager.friction_patch_count;++i)
                count+=scene.friction_patches_[i].anchor_count;
        }
        return count*sizeof(GpuDyFrictionAnchorInput);
    }
    static void verify(const Scene& cpu,const Scene& gpu) {
        const auto& reference=cpu.friction_patches_;
        const auto& actual=gpu.friction_patches_;
        if(reference.size()!=actual.size())
            throw std::runtime_error(
                "GPU PCM friction changed Sc source patch count");
        const auto near=[](Vec3 a,Vec3 b) {
            return std::fabs(a.x-b.x)<0.005f &&
                   std::fabs(a.y-b.y)<0.005f &&
                   std::fabs(a.z-b.z)<0.005f;
        };
        for(std::size_t i=0u;i<reference.size();++i) {
            const auto& a=reference[i];
            const auto& b=actual[i];
            // PCM friction extent belongs to the resident Gu producer
            // when GPU Dy owns preparation. Its CPU mirror deliberately
            // retains no numerical bounds; source touch, patch identity,
            // normals and local anchor history remain authoritative here.
            const auto manager=std::lower_bound(
                gpu.manifolds_.begin(),gpu.manifolds_.end(),b.pair,
                [](const auto& m,Pair pair) {return m.pair<pair;});
            const bool resident_bounds=
                gpu.gpu_broadphase_ &&
                gpu.gpu_broadphase_->has_gpu_dy_distance() &&
                manager!=gpu.manifolds_.end() &&
                manager->pair==b.pair &&
                manager->pcm_result_index!=
                    std::numeric_limits<std::uint32_t>::max();
            if(a.pair!=b.pair || a.contact_count!=b.contact_count ||
               a.anchor_count!=b.anchor_count ||
               !near(a.normal,b.normal) ||
               (resident_bounds
                  ?(!near(b.bounds_min,{}) || !near(b.bounds_max,{}))
                  :(!near(a.bounds_min,b.bounds_min) ||
                    !near(a.bounds_max,b.bounds_max))))
                throw std::runtime_error(
                    "GPU PCM changed source correlated friction topology or restored host-owned bounds");
            for(std::size_t k=0u;k<a.anchor_count;++k)
                if(!near(a.body0_anchors[k],b.body0_anchors[k]) ||
                   !near(a.body1_anchors[k],b.body1_anchors[k]))
                    throw std::runtime_error(
                        "GPU PCM friction anchor does not match source local witness");
        }
    }
};
} // namespace fyzz::native

namespace {
void require(bool good,const char* message) {
    if(!good)throw std::runtime_error(message);
}
// Inspect compiled requirements: software adapters can execute FP64 and
// would otherwise conceal an accidental increase of the hardware floor.
void verify_rigid_shader_requirements(
    fyzz::compute::Backend backend,const char* path) {
    std::ifstream stream(path,std::ios::binary);
    require(bool(stream),"Source Dy shader bundle missing");
    const std::vector<char> bytes(
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{});
    const auto word=[&](std::size_t offset) {
        require(offset<=bytes.size() && bytes.size()-offset>=4u,
                "Truncated rigid shader metadata");
        std::uint32_t value;
        std::memcpy(&value,bytes.data()+offset,sizeof(value));
        return value;
    };
    if(backend==fyzz::compute::Backend::Vulkan) {
        require(bytes.size()>=20u && bytes.size()%4u==0u &&
                word(0u)==0x07230203u,"Invalid rigid SPIR-V module");
        for(std::size_t offset=20u;offset<bytes.size();) {
            const auto instruction=word(offset);
            const auto count=instruction>>16u,opcode=instruction&0xffffu;
            require(count>0u && count<=(bytes.size()-offset)/4u,
                    "Invalid rigid SPIR-V instruction");
            // OpCapability Float64 and OpTypeFloat with a 64-bit width.
            require(!(opcode==17u && word(offset+4u)==10u) &&
                    !(opcode==22u && word(offset+8u)==64u),
                    "Rigid solver shader requires FP64 arithmetic");
            offset+=std::size_t(count)*4u;
        }
    }else {
        require(bytes.size()>=32u && word(0u)==0x43425844u,
                "Invalid rigid DXIL container");
        const auto count=word(28u);
        require(count<=(bytes.size()-32u)/4u,
                "Invalid rigid DXIL part table");
        bool feature_info=false;
        for(std::size_t i=0u;i<count;++i) {
            const std::size_t offset=word(32u+i*4u);
            if(word(offset)!=0x30494653u)continue; // SFI0.
            require(word(offset+4u)>=8u,"Invalid DXIL feature metadata");
            // D3D shader requirements: doubles and extended double ops.
            require((word(offset+8u)&0x21u)==0u,
                    "Rigid solver shader requires FP64 arithmetic");
            feature_info=true;
        }
        require(feature_info,"Rigid DXIL shader has no feature metadata");
    }
}
// Independent GPU broadphase volumes share one Sc rigid actor identity.
// Same-actor shape overlap is filtered; each cross-actor shape pair is kept.
void verify_compound_shape_gpu_pairs(
    fyzz::compute::Backend backend,const char* sap) {
    using namespace fyzz;
    using Motion=native::GpuBroadphaseBox::Motion;
    physx_gpu::SourceScOwnedGpuBroadphaseNative gpu(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u);
    gpu.admit_shape_element(0u,0u,Motion::Dynamic,0.02f);
    gpu.admit_shape_element(1u,0u,Motion::Dynamic,0.02f);
    gpu.admit_shape_element(2u,1u,Motion::Static,0.02f);
    std::array<native::GpuBroadphaseBox,3> shapes{};
    for(std::uint32_t i=0u;i<shapes.size();++i) {
        shapes[i].id=i;
        shapes[i].half_extents={0.5f,0.5f,0.5f};
        shapes[i].contact_offset=0.02f;
        shapes[i].motion=i==2u?Motion::Static:Motion::Dynamic;
    }
    shapes[0].position={0.f,0.f,0.f};
    shapes[1].position={0.1f,0.f,0.f};
    shapes[2].position={0.5f,0.f,0.f};
    const auto contains=[](std::span<const native::Pair> pairs,
                           native::Pair expected) {
        return std::find(pairs.begin(),pairs.end(),expected)!=pairs.end();
    };
    const auto created=gpu.advance(shapes,false,true);
    require(!created.overflow && created.found.size()==2u &&
            contains(created.found,{0u,2u}) &&
            contains(created.found,{1u,2u}) &&
            !contains(created.found,{0u,1u}),
        "Source GPU broadphase did not preserve compound shape pairs");
    const auto initial_fetches=gpu.source_sap_fetch_count();
    const auto converged=gpu.advance(shapes,false,false);
    require(converged.found.empty() && converged.lost.empty() &&
            gpu.source_sap_fetch_count()==initial_fetches+1u,
        "Source SAP skipped its required final history-convergence pass");
    const auto quiet=gpu.advance(shapes,false,false);
    require(quiet.found.empty() && quiet.lost.empty() &&
            gpu.source_sap_fetch_count()==initial_fetches+1u &&
            !gpu.last_frame().update.execute_gpu_stages,
        "Unchanged source AABB submitted or fetched another SAP frame");
    // ScBodySim changes one rigid actor's motion and refilters every attached
    // ElementSim, even when the shapes overlap each other.
    gpu.set_actor_motion(0u,Motion::Kinematic);
    shapes[0].motion=Motion::Kinematic;
    shapes[1].motion=Motion::Kinematic;
    const auto kinematic=gpu.advance(shapes,false,true);
    require(gpu.source_sap_fetch_count()==initial_fetches+2u,
        "Motion-class refilter did not advance source GPU SAP");
    require(kinematic.found.empty() && kinematic.lost.empty(),
        "Compound actor kinematic transition changed stable GPU shape pairs");
    gpu.set_actor_motion(0u,Motion::Dynamic);
    shapes[0].motion=Motion::Dynamic;
    shapes[1].motion=Motion::Dynamic;
    const auto restored=gpu.advance(shapes,false,true);
    require(restored.lost.empty() && restored.found.empty(),
        "Compound actor dynamic transition changed stable GPU shape pairs");
    // Sc edits this shape, not its parent rigid actor or sibling shape.
    shapes[0].position={4.f,0.f,0.f};
    gpu.mark_shape_bounds_changed(0u,0.02f);
    const auto changed=gpu.advance(shapes,false,true);
    require(gpu.source_sap_fetch_count()==initial_fetches+4u,
        "Shape-bound update failed to resume source GPU SAP");
    require(changed.found.empty() && changed.lost.size()==1u &&
            contains(changed.lost,{0u,2u}),
        "Edited Sc shape retired the wrong GPU contact candidate");
    gpu.retire_shape(0u);
    const std::array<native::GpuBroadphaseBox,2> survivors{{
        shapes[1],shapes[2]}};
    const auto retired=gpu.advance(survivors,false,true);
    require(retired.found.empty() && retired.lost.empty(),
        "Retiring one shape disturbed surviving source GPU pair history");
    gpu.post_reports_cleanup();
}

void verify(fyzz::compute::Backend backend,const char* sap,
            const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    require(producer->has_gpu_pcm_box(),
            "GPU scene has no installed native Gu PCM producer");
    gpu.set_gpu_broadphase(std::move(owner));
    // Authored joints need no contact-history allocation. Contact
    // managers, including CPU Gu, instead require resident EdgeIndex
    // physical-touch publication before device task admission.
    constexpr std::uint32_t invalid_edge=0xffffffffu;
    const std::array<physx_gpu::SourceScDeviceIslandEdge,1> joint_edge{{
        {0u,0u,invalid_edge,0u,
         physx_gpu::source_sc_device_island_joint_bit}
    }};
    const std::array<std::uint32_t,1> joint_flags{{
        4u|(1u<<8u)|physx_gpu::source_sc_device_island_active_bit
    }};
    require(producer->device_pcm_touch_capacity()==0u,
        "Fresh source scene unexpectedly owns contact history");
    producer->enqueue_gpu_dy_island_admission(joint_edge,joint_flags,
        physx_gpu::SourceScDeviceIslandSources{0u,0u,1u});
    const auto joint_task=physx_gpu::SourceScDeviceIslandFixture::read(*producer);
    require(joint_task.errors==0u &&
            joint_task.runnable_components==1u &&
            joint_task.responding_nodes==1u &&
            joint_task.admitted_edges==1u,
        "Joint-only Sc task admission required contact witnesses");
    producer->commit_gpu_step();
    require(producer->device_island_work_address()==0u,
        "Committed joint-only task survived its source epoch");
    const auto cpu_a=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
    const auto cpu_b=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto gpu_a=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
    const auto gpu_b=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto check=[&](bool contact,bool force=false) {
        cpu.advance_collision(force);
        gpu.advance_collision(force);
        require(cpu.pairs().size()==gpu.pairs().size() &&
                cpu.contacts().size()==gpu.contacts().size(),
                "Sc GPU PCM pair or contact manager publication differs");
        native::GpuPcmFrictionAnchorFixture::verify_pcm_contact_arena(gpu);
        require((gpu.contacts().size()==1u)==contact,
                "Source GPU PCM scene has unexpected contact manager count");
        if(!contact)return;
        const auto& c=cpu.contacts().front();
        const auto& g=gpu.contacts().front();
        require(c.count==g.count && c.count>0u &&
                c.edge_handle==g.edge_handle,
                "Source GPU PCM did not preserve Sc manager identity/count");
        for(std::uint8_t i=0;i<c.count;++i) {
            const auto& x=c.points[i];
            const auto& y=g.points[i];
            require(std::fabs(x.separation-y.separation)<0.002f &&
                    std::fabs(x.position.x-y.position.x)<0.002f &&
                    std::fabs(x.position.y-y.position.y)<0.002f &&
                    std::fabs(x.position.z-y.position.z)<0.002f &&
                    std::fabs(x.normal.x-y.normal.x)<0.002f &&
                    std::fabs(x.normal.y-y.normal.y)<0.002f &&
                    std::fabs(x.normal.z-y.normal.z)<0.002f,
                    "Source GPU Sc contact witness differs from CPU Gu PCM");
        }
    };
    check(true);
    native::GpuPcmFrictionAnchorFixture::verify_early_speculative_admission(
        cpu,gpu);
    const auto edge=gpu.contact_edge_index({gpu_a,gpu_b});
    // CPU Sc and GPU Dy consume the same 84-byte source record.
    // No expanded per-manager host staging record is needed.
    const std::array<native::GpuPcmBoxPair,1> request{{
        {gpu_a,gpu_b,edge,false}
    }};
    std::array<native::GpuPcmBoxResult,1> direct{};
    producer->enqueue_gpu_pcm_box_contacts(request,direct);
    require(direct[0].packed_counts==0u &&
            producer->device_pcm_contact_count()==1u &&
            producer->device_pcm_contact_stream()!=0u &&
            producer->device_pcm_touch_history()!=0u &&
            producer->device_pcm_touch_capacity()>edge,
        "GPU PCM enqueue did not expose resident Gu before the host join");
    bool rejected_early_commit=false;
    try {
        producer->commit_gpu_step();
    } catch(const std::logic_error&) {
        rejected_early_commit=true;
    }
    require(rejected_early_commit,
        "GPU PCM accepted a frame before Sc joined its numerical witnesses");
    // Use actual device PCM touch, not a fixture-initialized contact
    // history. Execute accurate connectivity before host Gu publication.
    constexpr std::uint32_t no_node=0xffffffffu;
    const std::array<physx_gpu::SourceScDeviceIslandEdge,1> source_edges{{
        {edge,0u,no_node,0u,0u}
    }};
    const std::array<std::uint32_t,1> source_nodes{{
        4u|(4u<<8u)|physx_gpu::source_sc_device_island_active_bit
    }};
    producer->enqueue_gpu_dy_island_admission(source_edges,source_nodes,
        physx_gpu::SourceScDeviceIslandSources{1u,0u,0u});
    const auto graph=physx_gpu::SourceScDeviceIslandFixture::read(*producer);
    require(graph.errors==0u &&
            graph.runnable_components==1u &&
            graph.responding_nodes==1u &&
            graph.admitted_edges==1u &&
            graph.dispatch_x==1u &&
            direct[0].packed_counts==0u,
        "Device Sc task admission waited for host PCM witness publication");
    producer->finish_gpu_pcm_box_contacts();
    producer->commit_gpu_step();
    require(producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
            "Direct GPU PCM readback did not use the compact device stream");
    require(direct[0].edge==edge && direct[0].status()==0u &&
            direct[0].cached_count()<=4u &&
            direct[0].count()<=direct[0].cached_count() &&
            (direct[0].selected_and_regenerated()&0x7fffffffu)<=4u &&
            direct[0].count()>0u &&
            direct[0].touch_transition()==native::GpuPcmTouchTransition::Persists &&
            std::isfinite(direct[0].normal.x) &&
            std::isfinite(direct[0].points[0].separation),
            "Direct Sc GPU PCM readback disagrees with the shader record");
    physx_gpu::SourcePcmGpuContactStreamNative raw{};
    std::memcpy(&raw,&direct[0],sizeof(raw));
    const auto header=physx_gpu::unpack_source_pcm_gpu_header_native(raw);
    require(header.edge_handle==direct[0].edge &&
            header.cached_count==direct[0].cached_count() &&
            header.emitted_count==direct[0].count() &&
            header.touch_transition==
                static_cast<std::uint32_t>(direct[0].touch_transition()),
        "Source Sc result does not share the 84-byte GPU contact ABI");
    // A quiet refresh must reuse Sc's original device pair descriptor:
    // the GPU still runs PCM, but no pair-list HtoD submission is needed.
    std::array<native::GpuPcmBoxResult,1> repeated{};
    producer->enqueue_gpu_pcm_box_contacts(request,repeated);
    producer->finish_gpu_pcm_box_contacts();
    producer->commit_gpu_step();
    require(producer->pcm_pair_upload_bytes()==0u &&
            repeated[0].status()==0u && repeated[0].edge==edge &&
            repeated[0].count()>0u &&
            repeated[0].touch_transition()==native::GpuPcmTouchTransition::Persists,
            "Stable Sc PCM manager unexpectedly re-uploaded source pair input");
    // Explicitly invalidate a live Sc EdgeIndex. A newly admitted or
    // rejected source frame uses the identical candidate-owned reset.
    const std::array<native::GpuPcmBoxPair,1> reset_request{{
        {gpu_a,gpu_b,edge,true,16u,false,0.f,0.f,true}
    }};
    std::array<native::GpuPcmBoxResult,1> reset{};
    producer->enqueue_gpu_pcm_box_contacts(reset_request,reset);
    producer->finish_gpu_pcm_box_contacts();
    producer->commit_gpu_step();
    require(producer->pcm_pair_upload_bytes()==
                sizeof(physx_gpu::SourcePcmBoxPairNative),
            "Source PCM reset did not refresh its GPU pair descriptor");
    require(reset[0].edge==edge && reset[0].status()==0u &&
            reset[0].count()>0u &&
            reset[0].touch_transition()==native::GpuPcmTouchTransition::Persists &&
            (reset[0].selected_and_regenerated()&0x80000000u)!=0u,
            "Source GPU PCM failed to regenerate a reset EdgeIndex");
    for(std::uint32_t i=0u;i<reset[0].count();++i)
        require(std::isfinite(reset[0].points[i].separation) &&
                std::fabs(reset[0].points[i].separation-
                          direct[0].points[i].separation)<0.002f,
                "Source GPU PCM reset altered the Gu witness");
    // Force the original AABB volume domain to grow while the first
    // contact manager retains GPU-local cached shape geometry.
    BodyId distant_gpu=std::numeric_limits<BodyId>::max();
    for(std::uint32_t i=0u;i<36u;++i) {
        const Vec3 position{20.f+3.f*float(i),1.f,0.f};
        cpu.add_static_box({0.5f,0.5f,0.5f},position);
        const auto id=gpu.add_static_box({0.5f,0.5f,0.5f},position);
        if(i==0u)distant_gpu=id;
    }
    check(true);
    // Static shape growth changes the GPU broadphase allocation, but
    // creates no dynamic IslandSim topology work to overlap.
    // Deliberately reject a speculative Gu manager refresh: it writes
    // an empty witness and a physical-touch loss into an existing
    // EdgeIndex. The device must restore the accepted full witness
    // before another source publication or island calculation.
    const std::array<native::GpuPcmBoxPair,1> failed_request{{
        {gpu_a,distant_gpu,edge,true,16u,false,0.f,0.f,true}
    }};
    std::array<native::GpuPcmBoxResult,1> rejected{};
    producer->enqueue_gpu_pcm_box_contacts(failed_request,rejected);
    producer->finish_gpu_pcm_box_contacts();
    require(rejected[0].count()==0u &&
            rejected[0].touch_transition()==native::GpuPcmTouchTransition::Lost,
        "Rejected Sc PCM refresh did not mutate speculative touch history");
    producer->invalidate_after_rollback();
    std::array<native::GpuPcmBoxResult,1> recovered{};
    producer->enqueue_gpu_pcm_box_contacts(request,recovered);
    producer->finish_gpu_pcm_box_contacts();
    require(recovered[0].status()==0u &&
            recovered[0].count()>0u &&
            recovered[0].touch_transition()==
                native::GpuPcmTouchTransition::Persists &&
            recovered[0].cached_count()<=4u,
        "Sc device contact journal failed to restore committed PCM history");
    producer->commit_gpu_step();
    // An unobserved GPU PCM epoch still modifies device witness history.
    // Rollback must retire that queued epoch before the next Sc contact
    // result can be joined, without requiring an abandoned host readback.
    std::array<native::GpuPcmBoxResult,1> abandoned{};
    producer->enqueue_gpu_pcm_box_contacts(failed_request,abandoned);
    producer->invalidate_after_rollback();
    require(abandoned[0].packed_counts==0u &&
            abandoned[0].edge==0u,
        "Abandoned GPU readback published rejected contact witnesses");
    std::array<native::GpuPcmBoxResult,1> retried{};
    producer->enqueue_gpu_pcm_box_contacts(request,retried);
    producer->finish_gpu_pcm_box_contacts();
    require(retried[0].status()==0u && retried[0].count()>0u &&
            retried[0].touch_transition()==
                native::GpuPcmTouchTransition::Persists,
        "Unjoined GPU PCM epoch contaminated a subsequent Sc publication");
    producer->commit_gpu_step();
    // A CPU-only Gu EdgeIndex is journaled independently of the GPU
    // PCM contact list. Rejecting the combined epoch must restore its
    // exact previous touch words without disturbing the PCM manager.
    if(edge==0xfffffffeu)
        throw std::overflow_error("Sc Gu rollback test EdgeIndex exhausted");
    const auto gu_edge=edge+1u;
    const std::array<native::GpuGuTouchRecord,1> gu_touch{{
        {gu_edge,1u}
    }};
    std::array<native::GpuPcmBoxResult,1> shared_pending{};
    producer->enqueue_gpu_pcm_box_contacts(
        request,shared_pending,gu_edge+1u);
    producer->enqueue_cpu_gu_touch_history(gu_touch);
    const std::array<std::uint32_t,1> one_active{{
        4u|physx_gpu::source_sc_device_island_active_bit
    }};
    producer->enqueue_gpu_dy_island_admission(
        std::span<const native::GpuDyIslandEdge>{},one_active,
        native::GpuDyIslandSources{});
    producer->finish_gpu_pcm_box_contacts();
    const auto changed=physx_gpu::SourceScDeviceIslandFixture::touch(
        *producer,gu_edge);
    require(changed[0]==1u && changed[1]==0u,
            "CPU-only Gu physical touch did not reach device history");
    producer->invalidate_after_rollback();
    std::array<native::GpuPcmBoxResult,1> restored_pcm{};
    producer->enqueue_gpu_pcm_box_contacts(
        request,restored_pcm,gu_edge+1u);
    producer->finish_gpu_pcm_box_contacts();
    const auto restored_touch=physx_gpu::SourceScDeviceIslandFixture::touch(
        *producer,gu_edge);
    require(restored_touch[0]==0u && restored_touch[1]==0u &&
            restored_pcm[0].count()>0u &&
            restored_pcm[0].touch_transition()==
                native::GpuPcmTouchTransition::Persists,
            "Rejected CPU Gu touch changed the committed device epoch");
    producer->commit_gpu_step();
    check(true,true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})==edge,
            "AABB capacity growth discarded an unchanged GPU PCM manifold");
    cpu.set_position(cpu_a,{0.02f,1.0f,0.0f});
    gpu.set_position(gpu_a,{0.02f,1.0f,0.0f});
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})==edge,
            "Quiet Sc GPU PCM invalidated a persistent contact edge");
    // Quiet advance_collision() does not rerun Gu PCM. Force one real
    // Sc/Gu refresh after the authored pose invalidated its cache, then
    // require the next identical forced refresh to reuse the GPU pair
    // buffer rather than reading a stale transfer counter.
    check(true,true);
    check(true,true);
    require(producer->pcm_pair_upload_bytes()==0u,
            "Unchanged Sc PCM refresh re-uploaded an admitted pair");
    // The device stream reconstructs the original Gu B-local witnesses
    // in world space, including rotated source-order second shapes.
    constexpr float angle=0.08f;
    const Quat orientation{0.f,0.f,std::sin(angle*0.5f),
                           std::cos(angle*0.5f)};
    cpu.set_pose(cpu_b,{0.8f,1.f,0.f},orientation);
    gpu.set_pose(gpu_b,{0.8f,1.f,0.f},orientation);
    check(true);
    cpu.set_pose(cpu_b,{0.8f,1.f,0.f},{});
    gpu.set_pose(gpu_b,{0.8f,1.f,0.f},{});
    check(true);
    cpu.set_position(cpu_a,{5.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{5.0f,1.0f,0.0f});
    check(false);
    cpu.set_position(cpu_a,{0.0f,1.0f,0.0f});
    gpu.set_position(gpu_a,{0.0f,1.0f,0.0f});
    check(true);
    require(gpu.contact_edge_index({gpu_a,gpu_b})!=
                std::numeric_limits<std::uint32_t>::max(),
            "Source Sc GPU PCM failed contact retirement and re-admission");
    // The source-owned Gu contacts must enter the existing Dy contact
    // preparation/solve path, not merely appear in collision queries.
    constexpr float dt=1.f/120.f;
    cpu.step(dt);
    gpu.step(dt);
    const auto cpu_body=cpu.body_state(cpu_a);
    const auto gpu_body=gpu.body_state(gpu_a);
    require(gpu.solved_result_frame()==gpu.frame_index() &&
            cpu.solved_result_frame()==cpu.frame_index() &&
            std::fabs(cpu_body.position.x-gpu_body.position.x)<0.002f &&
            std::fabs(cpu_body.position.y-gpu_body.position.y)<0.002f &&
            std::fabs(cpu_body.position.z-gpu_body.position.z)<0.002f,
            "Source GPU PCM did not feed the existing Sc/Dy rigid solve");

    // GPU general PCM starts before its independent CPU primitive
    // witness exists. The bounded Gu span is submitted before Sc's
    // numerical witness join; both preserve their original ordinals.
    const auto sphere_a=gpu.add_static_sphere(
        0.5f,{100.f,2.f,0.f});
    const auto sphere_b=gpu.add_dynamic_sphere(
        0.5f,{100.7f,2.f,0.f},1.f);
    gpu.advance_collision(true);
    const auto sphere_edge=gpu.contact_edge_index({sphere_a,sphere_b});
    require(sphere_edge!=std::numeric_limits<std::uint32_t>::max() &&
            sphere_edge!=edge,
            "Mixed late-Gu fixture lost its independent Sc edge");
    const std::array<native::GpuPcmBoxPair,2> mixed{{
        {gpu_a,gpu_b,edge,false},
        {sphere_a,sphere_b,sphere_edge,false,1u,false,0.f,0.f,
         true,true,true}
    }};
    std::array<native::GpuPcmBoxResult,2> mixed_results{};
    producer->enqueue_gpu_pcm_box_contacts(mixed,mixed_results);
    require(producer->pcm_host_contact_upload_bytes()==0u &&
            mixed_results[1].packed_counts==0u &&
            producer->device_pcm_contact_count()==0u &&
            producer->device_pcm_contact_stream()==0u &&
            producer->device_pcm_touch_history()==0u &&
            !physx_gpu::SourceScDeviceIslandFixture::pcm_ready(*producer),
        "Incomplete mixed Gu stream leaked into device task admission");
    std::array<native::ContactPoint,1> points{};
    const auto emitted=native::source_pcm_sphere_sphere(
        {100.f,2.f,0.f},0.5f,{100.7f,2.f,0.f},0.5f,0.04f,points);
    require(emitted==1u,"Late CPU Gu fixture failed to generate contact");
    auto& primitive=mixed_results[1];
    primitive.edge=sphere_edge;
    primitive.set_primitive(emitted,true);
    primitive.normal=points[0].normal;
    primitive.points[0]={points[0].position,points[0].separation};
    producer->enqueue_host_pcm_contacts(mixed,mixed_results);
    require(producer->pcm_host_contact_upload_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->device_pcm_contact_count()==mixed.size() &&
            producer->device_pcm_contact_stream()!=0u &&
            producer->device_pcm_touch_history()!=0u &&
            producer->device_pcm_touch_capacity()>sphere_edge &&
            physx_gpu::SourceScDeviceIslandFixture::pcm_ready(*producer),
        "Mixed Gu was not device-ready before Sc joined GPU witnesses");
    const std::array<physx_gpu::SourceScDeviceIslandEdge,2> mixed_edges{{
        {edge,0u,no_node,0u,0u},
        {sphere_edge,1u,no_node,1u,0u}
    }};
    const std::array<std::uint32_t,2> mixed_nodes{{
        source_nodes[0],source_nodes[0]
    }};
    producer->enqueue_gpu_dy_island_admission(mixed_edges,mixed_nodes,
        physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
    const auto mixed_graph=physx_gpu::SourceScDeviceIslandFixture::read(*producer);
    require(mixed_graph.errors==0u &&
            mixed_graph.runnable_components==2u &&
            mixed_graph.responding_nodes==2u &&
            mixed_graph.admitted_edges==2u &&
            mixed_results[0].packed_counts==0u,
        "Mixed CPU Gu/GPU PCM device connectivity required Sc witness join");
    producer->finish_gpu_pcm_box_contacts();
    require(producer->pcm_host_contact_upload_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            mixed_results[0].edge==edge &&
            mixed_results[0].status()==0u &&
            mixed_results[0].cached_count()<=4u &&
            mixed_results[0].count()<=mixed_results[0].cached_count() &&
            primitive.edge==sphere_edge &&
            primitive.count()==1u,
        "Late CPU Gu upload or general PCM readback lost source ordinals");
    // The early host upload remains ordered after the GPU witness DMA.
    // Collision-only commit must retire it even without a Dy solve.
    producer->commit_gpu_step();

    // Roll back while the Gu witness DMA and later primitive upload
    // are both outstanding. Neither rejected host data nor GPU contact
    // results may escape their common source transaction.
    std::array<native::GpuPcmBoxResult,2> abandoned_mixed{};
    abandoned_mixed[1]=primitive;
    producer->enqueue_gpu_pcm_box_contacts(mixed,abandoned_mixed);
    producer->enqueue_host_pcm_contacts(mixed,abandoned_mixed);
    producer->enqueue_gpu_dy_island_admission(mixed_edges,mixed_nodes,
        physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
    producer->invalidate_after_rollback();
    require(producer->device_pcm_contact_count()==0u &&
            producer->device_pcm_contact_stream()==0u &&
            producer->device_pcm_touch_history()==0u &&
            producer->device_island_work_address()==0u &&
            !producer->device_island_indirect_args().has_value() &&
            !physx_gpu::SourceScDeviceIslandFixture::pcm_ready(*producer),
        "Rejected PCM transaction retained an executable device task view");
    require(abandoned_mixed[0].packed_counts==0u,
        "Rejected mixed PCM DMA published an abandoned GPU witness");
    std::array<native::GpuPcmBoxResult,2> restored{};
    restored[1]=primitive;
    producer->enqueue_gpu_pcm_box_contacts(mixed,restored);
    producer->enqueue_host_pcm_contacts(mixed,restored);
    producer->enqueue_gpu_dy_island_admission(mixed_edges,mixed_nodes,
        physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
    const auto recovered_graph=physx_gpu::SourceScDeviceIslandFixture::read(*producer);
    require(recovered_graph.errors==0u &&
            recovered_graph.runnable_components==2u &&
            recovered_graph.admitted_edges==2u,
        "Rolled-back PCM regenerated an incorrect device island task");
    producer->finish_gpu_pcm_box_contacts();
    require(restored[0].edge==edge &&
            restored[0].status()==0u &&
            restored[1].edge==sphere_edge &&
            restored[1].count()==1u &&
            producer->pcm_host_contact_upload_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Rejected late primitive upload contaminated the next PCM epoch");
    producer->commit_gpu_step();
}

// A PCM readback may remain pending while independent Dy integration
// executes on the same Context queue. The Dy result is fetched only
// after Sc has explicitly joined the earlier witness DMA.
// Contact-free Sc admission exercises the exact Vulkan/D3D12 indirect
// binding and Dy task ordering without running GJK on software GPUs.
void verify_device_sc_dy_indirect_api(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader) {
    using namespace fyzz;
    std::ifstream file(dy_shader,std::ios::binary);
    require(bool(file),"Source indirect Dy shader unavailable");
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    native::Scene scene;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader));
    auto* producer=owner.get();
    native::GpuBroadphaseProducer& scene_producer=*producer;
    scene.set_gpu_broadphase(std::move(owner));
    const std::array<BodyId,2> actors{{
        scene.add_dynamic_box({0.5f,0.5f,0.5f},{2.f,1.f,0.f},1.f),
        scene.add_dynamic_box({0.5f,0.5f,0.5f},{10.f,1.f,0.f},1.f)
    }};
    const std::array<native::GpuPcmBoxPair,0> no_pairs{};
    std::array<native::GpuPcmBoxResult,0> no_witness{};
    producer->enqueue_gpu_pcm_box_contacts(no_pairs,no_witness);
    const std::array<std::uint32_t,3> nodes{{
        4u|(1u<<8u)|physx_gpu::source_sc_device_island_active_bit,
        physx_gpu::source_sc_device_island_nonresponding_bit,
        5u|(2u<<8u)|physx_gpu::source_sc_device_island_active_bit
    }};
    scene_producer.enqueue_gpu_dy_island_admission(
        std::span<const native::GpuDyIslandEdge>{},nodes,
        physx_gpu::SourceScDeviceIslandSources{});
    std::array<native::GpuDyDistanceBody,3> bodies{};
    bodies[0].inverse_mass=bodies[2].inverse_mass=1.f;
    const std::array<std::uint32_t,2> ordinals{{0u,2u}};
    std::array<native::GpuDyDistanceTask,2> tasks{{
        {0u,0u,0u,0u,0u,1u,4u,1u,0x4000u},
        {0u,0u,0u,0u,1u,2u,5u,2u,0x4000u}
    }};
    std::array<native::GpuDyBodyIntegrationInput,3> inputs{};
    std::array<native::GpuDyIntegratedPose,3> poses{};
    std::array<native::GpuDySleepState,3> sleep{};
    constexpr float seconds=1.f/120.f;
    for(std::size_t i=0u;i<actors.size();++i) {
        auto& in=inputs[ordinals[i]];
        in.active=1u;
        in.source_actor=actors[i];
        in.source_element=0xffffffffu;
        in.source_inverse_mass=1.f;
        in.position[0]=2.f+8.f*float(i);
        in.position[1]=1.f;
        in.orientation[3]=1.f;
        in.actor_position[0]=in.position[0];
        in.actor_position[1]=in.position[1];
        in.actor_orientation[3]=1.f;
        in.mass_local_orientation[3]=1.f;
        in.linear_velocity[0]=0.5f+0.25f*float(i);
        in.max_linear_speed_squared=100.f;
        in.max_angular_speed_squared=100.f;
        in.wake_counter=0.4f;
        in.seconds=seconds;
    }
    producer->enqueue_gpu_dy_distance(
        bodies,{},{},{},ordinals,tasks,
        {},{},{},{},{},{},{},{},
        inputs,poses,true,{},true,true,true,sleep);
    require(producer->has_queued_gpu_dy_distance() &&
            producer->dy_launch_stats().device_island_indirect_dispatches==1u &&
            producer->dy_launch_stats().device_generated_bytes==
                sizeof(tasks)+sizeof(ordinals) &&
            producer->dy_launch_stats().source_descriptor_upload_bytes==0u &&
            producer->dy_upload_span_count()==2u,
        "Source Sc task admission retained a host numerical task upload");
    producer->finish_gpu_dy_distance();
    require(tasks[0].error==0u && tasks[1].error==0u &&
            poses[0].valid==1u && poses[2].valid==1u &&
            poses[1].valid==0u &&
            std::fabs(poses[0].position[0]-(2.f+0.5f*seconds))<0.002f &&
            std::fabs(poses[2].position[0]-(10.f+0.75f*seconds))<0.002f,
        "Device-indirect Dy omitted an isolated responding body");
    producer->commit_gpu_step();

    // The declared task/body capacity is deliberately larger than the
    // source graph's one runnable island. GPU admission owns the valid
    // prefix; no CPU predicted task, body or descriptor count is read
    // back to choose a numerical launch or publish extra body motion.
    const std::array<std::uint32_t,3> partly_active{{
        nodes[0],nodes[1],
        nodes[2]&~physx_gpu::source_sc_device_island_active_bit
    }};
    producer->enqueue_gpu_pcm_box_contacts(no_pairs,no_witness);
    producer->enqueue_gpu_dy_island_admission(
        std::span<const physx_gpu::SourceScDeviceIslandEdge>{},
        partly_active,physx_gpu::SourceScDeviceIslandSources{});
    tasks[0].error=0x4000u;
    // The unused second slot is capacity, not a host-authored task.
    // It intentionally violates range and iteration invariants.
    tasks[1].descriptor_begin=37u;
    tasks[1].partition_end=999u;
    tasks[1].body_end=0u;
    tasks[1].position_iterations=0u;
    tasks[1].error=0x4000u;
    poses={};
    producer->enqueue_gpu_dy_distance(
        bodies,{},{},{},ordinals,tasks,
        {},{},{},{},{},{},{},{},
        inputs,poses,true,{},true,true,true,sleep);
    producer->finish_gpu_dy_distance();
    require(producer->dy_launch_stats().device_admitted_tasks==1u &&
            producer->dy_launch_stats().device_admitted_bodies==1u &&
            producer->dy_launch_stats().device_admitted_descriptors==0u &&
            tasks[0].body_begin==0u && tasks[0].body_end==1u &&
            tasks[0].error==0u && tasks[1].error==0u &&
            poses[0].valid==1u && poses[1].valid==0u &&
            poses[2].valid==0u,
        "Dy used the host task capacity instead of the device admission");
    producer->commit_gpu_step();
}

// Dy consumes the physical task and body-ordinal stream produced by
// Sc on the same source queue. The deliberately poisoned host task is
// not the numerical input. An inconsistent device header must reject
// the epoch before any motion or contact writeback is committed.
void verify_device_sc_dy_task_admission(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader) {
    using namespace fyzz;
    std::ifstream file(dy_shader,std::ios::binary);
    require(bool(file),"Device Sc/Dy shader unavailable");
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    native::Scene scene;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto moving=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.f,1.f,0.f},1.f);
    const auto fixed=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.f,0.f});
    const auto independent=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{8.f,1.f,0.f},1.f);
    scene.advance_collision(true);
    const auto edge=scene.contact_edge_index({moving,fixed});
    require(edge!=0xffffffffu,"Device Sc/Dy test lost its Sc manager");
    constexpr float seconds=1.f/120.f;
    const std::array<native::GpuPcmBoxPair,1> requests{{
        {moving,fixed,edge,false}
    }};
    std::array<native::GpuPcmBoxResult,1> witness{};
    std::array<native::GpuDyDistanceBody,1> body{};
    std::array<native::GpuDyDistanceDescriptor,1> descriptors{{
        {1u,0u,0xffffffffu,0x100u}
    }};
    // The host descriptor addresses manager 1; the resident Sc token
    // addresses manager 0. Only manager 0 has a responding endpoint.
    // A retained host token therefore rejects numerical admission.
    std::array<native::GpuDyContactManifold,2> managers{{
        {0u,0u,0xffffffffu,0u,0x100u},
        {0xffffffffu,0u,0xffffffffu,0u,0u}
    }};
    std::array<std::uint32_t,1> ordinals{0u};
    std::array<native::GpuDyDistanceTask,1> tasks{};
    std::array<native::GpuDyBodyIntegrationInput,1> inputs{};
    std::array<native::GpuDyIntegratedPose,1> poses{};
    std::array<native::GpuDySleepState,1> sleep{};
    const auto dispatch=[&] {
        body={};
        body[0].inverse_mass=1.f;
        tasks={};
        tasks[0]={0u,1u,0u,1u,0u,1u,4u,1u,0x4000u};
        inputs={};
        poses={};
        sleep={};
        auto& in=inputs[0];
        in.active=1u;
        in.source_actor=moving;
        in.source_element=0xffffffffu;
        in.source_inverse_mass=1.f;
        in.position[0]=2.f;
        in.position[1]=1.f;
        in.orientation[3]=1.f;
        in.actor_position[0]=2.f;
        in.actor_position[1]=1.f;
        in.actor_orientation[3]=1.f;
        in.mass_local_orientation[3]=1.f;
        in.linear_velocity[0]=0.5f;
        in.max_linear_speed_squared=100.f;
        in.max_angular_speed_squared=100.f;
        in.wake_counter=0.4f;
        in.seconds=seconds;
        producer->enqueue_gpu_dy_distance(
            body,{},descriptors,{},ordinals,tasks,
            {},{},{},{},{},managers,{},{},
            inputs,poses,true,{},true,true,true,sleep);
    };
    const std::array<std::uint32_t,1> single_node{{
        4u|(1u<<8u)|physx_gpu::source_sc_device_island_active_bit
    }};
    producer->enqueue_gpu_pcm_box_contacts(requests,witness);
    const std::array<physx_gpu::SourceScDeviceIslandEdge,1> touching{{
        {edge,0u,0xffffffffu,0u,0u}
    }};
    producer->enqueue_gpu_dy_island_admission(
        touching,single_node,physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
    dispatch();
    require(producer->has_queued_gpu_dy_distance() &&
            witness[0].packed_counts==0u,
        "Device Sc/Dy consumed the PCM witness before numerical dispatch");
    producer->finish_gpu_pcm_box_contacts();
    producer->finish_gpu_dy_distance();
    require(tasks[0].error==0u && poses[0].valid==1u &&
            producer->dy_launch_stats().device_island_indirect_dispatches==1u &&
            producer->dy_launch_stats().source_descriptor_upload_bytes==0u &&
            producer->dy_upload_span_count()==2u &&
            std::fabs(poses[0].position[0]-
                      (2.f+0.5f*seconds))<0.002f,
        "Dy did not indirectly execute the resident Sc task");
    producer->commit_gpu_step();

    // The indirect launch also has to execute every independent
    // source task, not merely the first workgroup from a host hint.
    // One physically touching manager belongs to node 0; node 1 is a
    // separate awake solver body with no numerical constraint.
    {
        std::array<native::GpuDyDistanceBody,2> bodies{};
        for(auto& item:bodies)item.inverse_mass=1.f;
        const std::array<std::uint32_t,2> body_order{{0u,1u}};
        std::array<native::GpuDyDistanceTask,2> active_tasks{{
            {0u,1u,0u,1u,0u,1u,4u,1u,0x4000u},
            {1u,1u,1u,1u,1u,2u,5u,1u,0x4000u}
        }};
        std::array<native::GpuDyBodyIntegrationInput,2> active_inputs{};
        std::array<native::GpuDyIntegratedPose,2> active_poses{};
        std::array<native::GpuDySleepState,2> active_sleep{};
        for(std::size_t i=0u;i<2u;++i) {
            auto& in=active_inputs[i];
            in.active=1u;
            in.source_actor=i==0u?moving:independent;
            in.source_element=0xffffffffu;
            in.source_inverse_mass=1.f;
            in.position[0]=2.f+8.f*float(i);
            in.position[1]=1.f;
            in.orientation[3]=1.f;
            in.actor_position[0]=in.position[0];
            in.actor_position[1]=in.position[1];
            in.actor_orientation[3]=1.f;
            in.mass_local_orientation[3]=1.f;
            in.linear_velocity[0]=0.5f+0.25f*float(i);
            in.max_linear_speed_squared=100.f;
            in.max_angular_speed_squared=100.f;
            in.wake_counter=0.4f;
            in.seconds=seconds;
        }
        const std::array<std::uint32_t,2> active_nodes{{
            single_node[0],
            5u|(1u<<8u)|physx_gpu::source_sc_device_island_active_bit
        }};
        witness={};
        producer->enqueue_gpu_pcm_box_contacts(requests,witness);
        producer->enqueue_gpu_dy_island_admission(
            touching,active_nodes,
            physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
        producer->enqueue_gpu_dy_distance(
            bodies,{},descriptors,{},body_order,active_tasks,
            {},{},{},{},{},managers,{},{},
            active_inputs,active_poses,true,{},true,true,true,active_sleep);
        producer->finish_gpu_pcm_box_contacts();
        producer->finish_gpu_dy_distance();
        require(active_tasks[0].error==0u && active_tasks[1].error==0u &&
                active_poses[0].valid==1u && active_poses[1].valid==1u &&
                producer->dy_launch_stats().device_island_indirect_dispatches==1u &&
                std::fabs(active_poses[0].position[0]-
                          (2.f+0.5f*seconds))<0.002f &&
                std::fabs(active_poses[1].position[0]-
                          (10.f+0.75f*seconds))<0.002f,
            "Resident Sc indirect launch omitted an independent Dy task");
        producer->commit_gpu_step();
    }

    // The work allocation has room for two nodes, but Sc admits two
    // responding nodes while the Dy caller provides only one. Runtime
    // validation must fail before any free body is integrated.
    witness={};
    const std::array<physx_gpu::SourceScDeviceIslandEdge,1> bridge{{
        {edge,0u,1u,0xffffffffu,
         physx_gpu::source_sc_device_island_joint_bit|
         physx_gpu::source_sc_device_island_no_response_bit}
    }};
    const std::array<std::uint32_t,2> two_nodes{{
        single_node[0],4u|(1u<<8u)
    }};
    producer->enqueue_gpu_pcm_box_contacts(requests,witness);
    producer->enqueue_gpu_dy_island_admission(
        bridge,two_nodes,physx_gpu::SourceScDeviceIslandSources{2u,0u,0u});
    dispatch();
    producer->finish_gpu_pcm_box_contacts();
    bool rejected=false;
    try {
        producer->finish_gpu_dy_distance();
    }catch(const std::overflow_error&) {
        rejected=true;
    }
    require(rejected && poses[0].valid==0u &&
            (tasks[0].error&0x01000000u)!=0u,
        "Dy accepted device Sc island counts inconsistent with its numerical tables");
    producer->invalidate_after_rollback();
}

void verify_queued_pcm_dy(fyzz::compute::Backend backend,const char* sap,
                          const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Source queued Dy shader unavailable");
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(std::istreambuf_iterator<char>{input},
                        std::istreambuf_iterator<char>{});
    native::Scene scene;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto moving=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.f,1.f,0.f},1.f);
    const auto fixed=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.f,0.f});
    scene.advance_collision(true);
    const auto edge=scene.contact_edge_index({moving,fixed});
    require(edge!=std::numeric_limits<std::uint32_t>::max(),
        "Queued PCM/Dy test could not find source Sc contact manager");

    constexpr float seconds=1.f/120.f;
    const std::array<native::GpuPcmBoxPair,1> request{{
        {moving,fixed,edge,false}
    }};
    const auto launch=[&](std::array<native::GpuPcmBoxResult,1>& witness,
                          std::array<native::GpuDyBodyIntegrationInput,1>& inputs,
                          std::array<native::GpuDyIntegratedPose,1>& poses) {
        witness={};
        inputs={};
        poses={};
        auto& body=inputs[0];
        body.active=1u;
        body.position[0]=2.f;
        body.position[1]=1.f;
        body.orientation[3]=1.f;
        body.actor_position[0]=2.f;
        body.actor_position[1]=1.f;
        body.actor_orientation[3]=1.f;
        body.linear_velocity[0]=0.5f;
        body.seconds=seconds;
        body.source_actor=0u;
        body.source_inverse_mass=1.f;
        body.max_linear_speed_squared=100.f;
        body.max_angular_speed_squared=100.f;
        body.wake_counter=0.4f;
        producer->enqueue_gpu_pcm_box_contacts(request,witness);
        producer->enqueue_gpu_dy_distance(
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},
            inputs,poses);
    };
    std::array<native::GpuPcmBoxResult,1> witness{};
    std::array<native::GpuDyBodyIntegrationInput,1> inputs{};
    std::array<native::GpuDyIntegratedPose,1> poses{};
    launch(witness,inputs,poses);
    require(producer->has_queued_gpu_dy_distance() &&
            witness[0].packed_counts==0u &&
            producer->device_pcm_contact_count()==1u,
        "Dy numerical dispatch joined PCM witness DMA on the host");
    bool duplicate_rejected=false;
    try {
        producer->enqueue_gpu_dy_distance(
            {},{},{},{},{},{},{},{},{},{},{},{},{},{},
            inputs,poses);
    } catch(const std::logic_error&) {
        duplicate_rejected=true;
    }
    require(duplicate_rejected && producer->has_queued_gpu_dy_distance(),
        "Dy reused active numerical output before its owning fetch");
    bool premature_join_rejected=false;
    try {
        producer->finish_gpu_dy_distance();
    } catch(const std::logic_error&) {
        premature_join_rejected=true;
    }
    require(premature_join_rejected &&
            producer->has_queued_gpu_dy_distance() &&
            witness[0].packed_counts==0u,
        "Premature host Dy join discarded its queued numerical result");
    bool premature_commit_rejected=false;
    try {
        producer->commit_gpu_step();
    } catch(const std::logic_error&) {
        premature_commit_rejected=true;
    }
    require(premature_commit_rejected &&
            producer->has_queued_gpu_dy_distance(),
        "Source Sc committed before GPU Dy published numerical results");
    producer->finish_gpu_pcm_box_contacts();
    require(witness[0].status()==0u && witness[0].count()>0u,
        "Queued PCM witness was invalid after early Dy dispatch");
    producer->finish_gpu_dy_distance();
    require(!producer->has_queued_gpu_dy_distance() &&
            poses[0].valid==1u &&
            std::fabs(poses[0].position[0]-
                      (2.f+0.5f*seconds))<0.002f &&
            std::fabs(poses[0].position[1]-1.f)<0.002f,
        "Queued GPU Dy failed to publish integrated body after PCM join");
    producer->commit_gpu_step();

    // Reject a numerical epoch before either source result is fetched.
    // Neither borrowed output may survive the common rollback boundary.
    std::array<native::GpuPcmBoxResult,1> rejected{};
    launch(rejected,inputs,poses);
    producer->invalidate_after_rollback();
    require(!producer->has_queued_gpu_dy_distance() &&
            producer->device_pcm_contact_count()==0u &&
            rejected[0].packed_counts==0u,
        "Rollback retained GPU PCM or numerical Dy publication");
    std::array<native::GpuPcmBoxResult,1> recovered{};
    launch(recovered,inputs,poses);
    producer->finish_gpu_pcm_box_contacts();
    producer->finish_gpu_dy_distance();
    require(recovered[0].status()==0u && recovered[0].count()>0u &&
            poses[0].valid==1u,
        "Rolled-back source GPU numerical epoch failed to regenerate");
    producer->commit_gpu_step();

    // Reject after Sc has already joined PCM, but while the Dy GPU
    // output remains outstanding. Its submitted commands must finish
    // before the next source arena allocation can reuse those bytes.
    std::array<native::GpuPcmBoxResult,1> joined{};
    launch(joined,inputs,poses);
    producer->finish_gpu_pcm_box_contacts();
    require(producer->has_queued_gpu_dy_distance() &&
            joined[0].status()==0u,
        "PCM witness join unexpectedly consumed deferred Dy numerical work");
    producer->invalidate_after_rollback();
    require(!producer->has_queued_gpu_dy_distance() &&
            producer->device_pcm_contact_count()==0u,
        "Rollback after PCM witness publication retained pending Dy work");
    std::array<native::GpuPcmBoxResult,1> retry{};
    launch(retry,inputs,poses);
    producer->finish_gpu_pcm_box_contacts();
    producer->finish_gpu_dy_distance();
    require(retry[0].status()==0u && retry[0].count()>0u &&
            poses[0].valid==1u,
        "Rejected joined-PCM GPU Dy epoch corrupted its subsequent retry");
    producer->commit_gpu_step();
}

// Sparse live Sc managers keep their original edge handles. GPU readback
// must scale with active managers, not the largest surviving edge index.
void verify_sparse_contact_stream(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    std::array<BodyId,12> moving_cpu{},moving_gpu{};
    std::array<BodyId,12> fixed_cpu{},fixed_gpu{};
    for(std::uint32_t i=0u;i<12u;++i) {
        const float x=5.f*float(i);
        moving_cpu[i]=cpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x,1.f,0.f},1.f);
        fixed_cpu[i]=cpu.add_static_box(
            {0.5f,0.5f,0.5f},{x+0.8f,1.f,0.f});
        moving_gpu[i]=gpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x,1.f,0.f},1.f);
        fixed_gpu[i]=gpu.add_static_box(
            {0.5f,0.5f,0.5f},{x+0.8f,1.f,0.f});
    }
    const auto check_transitions=[&](std::size_t found,std::size_t lost) {
        const auto cf=cpu.found_contacts(),gf=gpu.found_contacts();
        const auto cl=cpu.lost_contacts(),gl=gpu.lost_contacts();
        require(cf.size()==found && gf.size()==found &&
                cl.size()==lost && gl.size()==lost &&
                std::equal(cf.begin(),cf.end(),gf.begin()) &&
                std::equal(cl.begin(),cl.end(),gl.begin()) &&
                std::is_sorted(gf.begin(),gf.end()) &&
                std::is_sorted(gl.begin(),gl.end()),
            "Sc PCM single-pass touch transitions disagree with source order");
    };
    const auto check_ordinals=[&] {
        require(cpu.contacts().size()==gpu.contacts().size(),
            "Sc GPU PCM contact manager count diverged");
        for(std::size_t i=0;i<gpu.contacts().size();++i)
            require(gpu.contacts()[i].pcm_result_index==i &&
                    cpu.contacts()[i].pcm_result_index==
                        std::numeric_limits<std::uint32_t>::max(),
                "Source Sc contact manager lost its GPU PCM result ordinal");
    };
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==12u && gpu.contacts().size()==12u,
            "Sparse GPU PCM fixture did not admit original Sc managers");
    require(producer->pcm_pair_upload_bytes()==
                12u*sizeof(physx_gpu::SourcePcmBoxPairNative) &&
            producer->pcm_contact_readback_bytes()==
                12u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_edge_history_copy_bytes()==0u,
            "Initial sparse PCM manager admission did not use compact GPU records");
    check_transitions(12u,0u);
    check_ordinals();
    // Initial Sc admission sets reset flags on every GPU PCM manager.
    // Publish the first source refresh to retire those flags before
    // measuring a genuinely isolated middle-manager edit.
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_pair_upload_bytes()==
                12u*sizeof(physx_gpu::SourcePcmBoxPairNative),
        "Initial PCM reset-bit retirement did not republish all managers");
    check_ordinals();
    // Reset only a middle Sc manager in a twelve-pair device input.
    // Its original pair ordinal and EdgeIndex stay fixed; the GPU
    // refresh must transfer only that manager's 20-byte descriptor.
    cpu.set_position(moving_cpu[5],{25.02f,1.f,0.f});
    gpu.set_position(moving_gpu[5],{25.02f,1.f,0.f});
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==12u && gpu.contacts().size()==12u &&
            producer->pcm_pair_upload_bytes()==
                sizeof(physx_gpu::SourcePcmBoxPairNative),
        "One Sc PCM cache reset retransmitted the whole pair array");
    check_ordinals();
    // The reset bit is cleared in the following forced Sc publication,
    // which must also update just that pair before the next quiet input.
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_pair_upload_bytes()==
                sizeof(physx_gpu::SourcePcmBoxPairNative),
        "One Sc PCM reset-bit retirement retransmitted unrelated pairs");
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_pair_upload_bytes()==0u,
        "Stable PCM manager descriptors triggered a redundant upload");
    check_ordinals();
    // Physically retire disjoint managers before their BP EdgeIndex
    // handles are reused. The simultaneous lost and later found sets
    // must remain ordered without a second Sc manifold comparison.
    constexpr std::array<std::uint32_t,3> interrupted{1u,6u,10u};
    for(const auto i:interrupted) {
        const Vec3 away{5.f*float(i)+2.5f,1.f,0.f};
        cpu.set_position(moving_cpu[i],away);
        gpu.set_position(moving_gpu[i],away);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    check_transitions(0u,interrupted.size());
    check_ordinals();
    // Readmit two old contacts while losing a different touching pair.
    // Both event streams must remain independently sorted and disjoint.
    for(const auto i:std::array<std::uint32_t,2>{1u,10u}) {
        const Vec3 restored{5.f*float(i),1.f,0.f};
        cpu.set_position(moving_cpu[i],restored);
        gpu.set_position(moving_gpu[i],restored);
    }
    cpu.set_position(moving_cpu[4],{22.5f,1.f,0.f});
    gpu.set_position(moving_gpu[4],{22.5f,1.f,0.f});
    cpu.advance_collision();
    gpu.advance_collision();
    check_transitions(2u,1u);
    check_ordinals();
    for(const auto i:std::array<std::uint32_t,2>{4u,6u}) {
        const Vec3 restored{5.f*float(i),1.f,0.f};
        cpu.set_position(moving_cpu[i],restored);
        gpu.set_position(moving_gpu[i],restored);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    check_transitions(2u,0u);
    check_ordinals();
    const auto retained_edge=gpu.contact_edge_index(
        {moving_gpu.back(),fixed_gpu.back()});
    require(retained_edge!=std::numeric_limits<std::uint32_t>::max(),
            "Sparse GPU PCM fixture has no retained Sc edge");
    for(std::uint32_t i=0u;i<11u;++i) {
        cpu.remove(moving_cpu[i]);
        gpu.remove(moving_gpu[i]);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==1u && gpu.contacts().size()==1u &&
            producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            gpu.contacts()[0].edge_handle==retained_edge &&
            cpu.contacts()[0].count==gpu.contacts()[0].count,
            "Sparse Sc contact stream misidentified surviving edge handle");
    check_ordinals();
    const auto c=cpu.contacts()[0].points[0];
    const auto g=gpu.contacts()[0].points[0];
    require(std::fabs(c.separation-g.separation)<0.002f &&
            std::fabs(c.position.x-g.position.x)<0.002f &&
            std::fabs(c.normal.x-g.normal.x)<0.002f,
            "Sparse Sc GPU contact stream changed the retained Gu witness");

    // Re-grow the GPU contact readback after the 12-pair -> 1-pair
    // retirement. The source PCM commands are now still outstanding
    // when Vulkan enlarges its cached staging buffer for this fetch.
    constexpr std::uint32_t added=16u;
    for(std::uint32_t i=0u;i<added;++i) {
        const Vec3 position{200.f+4.f*float(i),1.f,0.f};
        cpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
        gpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
        const Vec3 other{position.x+0.8f,position.y,position.z};
        cpu.add_static_box({0.5f,0.5f,0.5f},other);
        gpu.add_static_box({0.5f,0.5f,0.5f},other);
    }
    cpu.advance_collision();
    gpu.advance_collision();
    require(cpu.contacts().size()==added+1u &&
            gpu.contacts().size()==cpu.contacts().size() &&
            gpu.contact_edge_index(
                {moving_gpu.back(),fixed_gpu.back()})==retained_edge,
            "Growing source GPU PCM readback lost an existing Sc contact manager");
    require(producer->pcm_pair_upload_bytes()==
                (added+1u)*sizeof(physx_gpu::SourcePcmBoxPairNative) &&
            producer->pcm_contact_readback_bytes()==
                (added+1u)*sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_edge_history_copy_bytes()==12u*240u,
            "PCM pair or packed readback growth lost source manager descriptors");
    check_ordinals();
    for(std::size_t i=0;i<cpu.contacts().size();++i) {
        const auto& prior=cpu.contacts()[i];
        const auto& current=gpu.contacts()[i];
        require(prior.pair==current.pair &&
                prior.count==current.count &&
                current.count>0u,
                "Expanded GPU PCM readback changed source Gu manager order/count");
        for(std::uint8_t j=0;j<prior.count;++j)
            require(std::fabs(prior.points[j].separation-
                              current.points[j].separation)<0.002f &&
                    std::fabs(prior.points[j].normal.x-
                              current.points[j].normal.x)<0.002f,
                    "Expanded source GPU PCM stream changed Gu contact geometry");
    }
}

// Sc's dormant contact managers retain their own Gu publication. They
// must not force PCM device fetches when an unrelated manager changes.
void verify_dormant_contact_stream(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    std::array<BodyId,2> dormant_cpu{},dormant_gpu{};
    for(std::uint32_t i=0;i<2u;++i) {
        const float x=4.0f*float(i);
        cpu.add_static_box({0.5f,0.5f,0.5f},{x,0.f,0.f});
        gpu.add_static_box({0.5f,0.5f,0.5f},{x,0.f,0.f});
        dormant_cpu[i]=cpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x+0.997f,0.f,0.f},1.f);
        dormant_gpu[i]=gpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},{x+0.997f,0.f,0.f},1.f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        cpu.set_dynamics(dormant_cpu[i],dynamics);
        gpu.set_dynamics(dormant_gpu[i],dynamics);
    }
    // Source Dy sleepCheck publishes readiness; accurate IslandSim
    // consumes it on the following step without retiring Sc managers.
    for(std::uint32_t i=0;i<3u &&
        (!gpu.body_is_sleeping(dormant_gpu[0]) ||
         !gpu.body_is_sleeping(dormant_gpu[1]));++i) {
        cpu.step(0.5f);
        gpu.step(0.5f);
    }
    require(gpu.body_is_sleeping(dormant_gpu[0]) &&
            gpu.body_is_sleeping(dormant_gpu[1]) &&
            cpu.body_is_sleeping(dormant_cpu[0]) &&
            cpu.body_is_sleeping(dormant_cpu[1]) &&
            gpu.contacts().size()==2u,
        "Source dormant PCM fixture did not retain two sleeping Gu managers");
    const auto check=[&](std::size_t expected) {
        require(cpu.contacts().size()==expected &&
                gpu.contacts().size()==expected,
            "Dormant Sc GPU PCM changed live manager count");
        for(std::size_t i=0;i<expected;++i) {
            const auto& a=cpu.contacts()[i];
            const auto& b=gpu.contacts()[i];
            require(a.pair==b.pair && a.count==b.count && a.count>0u,
                "Dormant Sc GPU PCM changed Gu touch state");
            for(std::uint8_t j=0;j<a.count;++j)
                require(std::fabs(a.points[j].separation-
                                  b.points[j].separation)<0.002f &&
                        std::fabs(a.points[j].position.x-
                                  b.points[j].position.x)<0.002f &&
                        std::fabs(a.points[j].normal.x-
                                  b.points[j].normal.x)<0.002f,
                    "Dormant Sc GPU PCM lost the prior Gu witness");
        }
        require(cpu.found_contacts().size()==gpu.found_contacts().size() &&
                cpu.lost_contacts().size()==gpu.lost_contacts().size(),
            "Dormant PCM fabricated physical-touch transitions");
    };
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_contact_readback_bytes()==0u &&
            producer->pcm_pair_upload_bytes()==0u,
        "Sleeping Sc contact managers triggered GPU PCM transfer");
    check(2u);
    // An independent awake pair must not make dormant pairs pay the
    // full Sc PCM readback. This exercises sparse original EdgeIndex
    // order while the prior Gu manifolds are still published.
    const BodyId fixed_cpu=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{8.f,0.f,0.f});
    const BodyId fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{8.f,0.f,0.f});
    const auto awake_cpu=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{8.997f,0.f,0.f},1.f);
    const auto awake_gpu=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{8.997f,0.f,0.f},1.f);
    require(fixed_cpu==fixed_gpu && awake_cpu==awake_gpu,
        "Sc sparse dormant fixture lost source actor admission order");
    cpu.advance_collision();
    gpu.advance_collision();
    require(producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_edge_history_copy_bytes()==2u*240u &&
            gpu.contacts().size()==3u &&
            !gpu.body_is_sleeping(awake_gpu),
        "Awake Sc manager retransferred dormant GPU PCM manifolds");
    check(3u);
    // NpRigidDynamic::setGlobalPose wakes and invalidates the original
    // Sc manager even on an idempotent pose write. It must reenter the
    // resident GPU Gu cache and fetch exactly one additional manager.
    const auto pose=cpu.body_state(dormant_cpu[0]);
    cpu.set_position(dormant_cpu[0],pose.position);
    gpu.set_position(dormant_gpu[0],pose.position);
    cpu.advance_collision();
    gpu.advance_collision();
    require(!gpu.body_is_sleeping(dormant_gpu[0]) &&
            gpu.body_is_sleeping(dormant_gpu[1]) &&
            producer->pcm_contact_readback_bytes()==
                2u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_edge_history_copy_bytes()==0u,
        "Woken Sc EdgeIndex failed to reenter GPU PCM while another slept");
    check(3u);
}

// Speculative SAP overlap is not a physical Gu touch. Once the dynamic
// actor sleeps, the original zero-touch EdgeIndex can retain that result
// without repeating GPU PCM or downloading an empty contact stream.
void verify_dormant_empty_pcm(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu,gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const Vec3 center{1.5f,0.7f,0.f};
    const Vec3 half{1.f,0.1f,0.1f};
    const Quat turn{0.f,0.f,-0.382683432f,0.923879533f};
    const auto fixed_cpu=cpu.add_static_box(half,{0.f,0.f,0.f});
    const auto fixed_gpu=gpu.add_static_box(half,{0.f,0.f,0.f});
    const auto moving_cpu=cpu.add_dynamic_box(half,center,1.f);
    const auto moving_gpu=gpu.add_dynamic_box(half,center,1.f);
    cpu.set_pose(moving_cpu,center,turn);
    gpu.set_pose(moving_gpu,center,turn);
    RigidBodyDynamics dynamics{};
    dynamics.gravity_enabled=false;
    cpu.set_dynamics(moving_cpu,dynamics);
    gpu.set_dynamics(moving_gpu,dynamics);
    const auto check_empty=[&] {
        require(cpu.pairs().size()==1u && gpu.pairs().size()==1u &&
                cpu.contacts().empty() && gpu.contacts().empty() &&
                cpu.found_contacts().empty() && gpu.found_contacts().empty() &&
                cpu.lost_contacts().empty() && gpu.lost_contacts().empty(),
            "Dormant zero-touch source pair lost speculative SAT separation");
    };
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    check_empty();
    const native::Pair separated_pair{fixed_gpu,moving_gpu};
    const auto separated_edge=gpu.contact_edge_index(separated_pair);
    require(separated_edge!=std::numeric_limits<std::uint32_t>::max() &&
            !gpu.contact_edge_connected(separated_pair),
        "Source speculative non-touch pair lost its original Sc EdgeIndex");
    require(producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Uninitialized zero-touch Sc edge did not run original GPU PCM");
    for(std::uint32_t i=0u;i<6u &&
        (!cpu.body_is_sleeping(moving_cpu) ||
         !gpu.body_is_sleeping(moving_gpu));++i) {
        cpu.step(0.5f);
        gpu.step(0.5f);
    }
    require(cpu.body_is_sleeping(moving_cpu) &&
            gpu.body_is_sleeping(moving_gpu),
        "Speculative zero-touch pair did not enter source Dy sleep");
    check_empty();
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    check_empty();
    require(producer->pcm_contact_readback_bytes()==0u &&
            producer->pcm_pair_upload_bytes()==0u,
        "Cached non-touching sleeping Sc pair repeated GPU PCM readback");
    // An authored static-shape edit invalidates PCM even when the
    // dynamic partner remains asleep. Resolve and cache the same zero
    // result, then verify the next dormant frame needs no GPU readback.
    cpu.set_position(fixed_cpu,{0.f,0.f,0.f});
    gpu.set_position(fixed_gpu,{0.f,0.f,0.f});
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    check_empty();
    require(gpu.body_is_sleeping(moving_gpu) &&
            producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Static Sc cache reset did not refresh zero-touch GPU PCM");
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_contact_readback_bytes()==0u,
        "Unchanged zero-touch Sc manager did not resume dormant reuse");
    // A separate active touching manager must not requeue the dormant
    // negative-contact manager or shift the source Gu edge ordinal.
    cpu.add_static_box({0.5f,0.5f,0.5f},{5.f,0.f,0.f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{5.f,0.f,0.f});
    cpu.add_dynamic_box({0.5f,0.5f,0.5f},{5.997f,0.f,0.f},1.f);
    gpu.add_dynamic_box({0.5f,0.5f,0.5f},{5.997f,0.f,0.f},1.f);
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(cpu.pairs().size()==2u && gpu.pairs().size()==2u &&
            cpu.contacts().size()==1u && gpu.contacts().size()==1u &&
            cpu.contacts()[0].pair==gpu.contacts()[0].pair &&
            producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Awake Sc manager requeued dormant zero-touch GPU PCM");
    // NpRigidDynamic::setGlobalPose wakes and invalidates the original
    // zero-touch manager even on an idempotent write. Both active pairs
    // now occupy the original ordered GPU PCM stream.
    cpu.set_position(moving_cpu,center);
    gpu.set_position(moving_gpu,center);
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(!gpu.body_is_sleeping(moving_gpu) &&
            cpu.pairs().size()==gpu.pairs().size() &&
            cpu.contacts().size()==gpu.contacts().size() &&
            cpu.contacts().size()==1u &&
            producer->pcm_contact_readback_bytes()==
                2u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Woken zero-touch Sc manager did not reenter ordered GPU PCM");
    require(gpu.contact_edge_index(separated_pair)==separated_edge &&
            !gpu.contact_edge_connected(separated_pair),
        "Dormant negative-contact reuse replaced the original Sc edge");
    // The same original manager must now publish a real Gu contact
    // after an authored pose change. A stale negative result would hide
    // the new physical touch despite the speculative edge surviving.
    cpu.set_pose(moving_cpu,{0.5f,0.f,0.f},{});
    gpu.set_pose(moving_gpu,{0.5f,0.f,0.f},{});
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(cpu.contacts().size()==2u && gpu.contacts().size()==2u &&
            gpu.contact_edge_index(separated_pair)==separated_edge &&
            gpu.contact_edge_connected(separated_pair) &&
            cpu.found_contacts().size()==gpu.found_contacts().size() &&
            producer->pcm_contact_readback_bytes()==
                2u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
        "Authored Sc shape motion failed to turn cached absence into GPU touch");
}

// Coupled box contacts exercise source Sc manager identity and Dy's
// real contact partition/PGS path across multiple dependent bodies.
void verify_coupled_rigid_stack(
    fyzz::compute::Backend backend,const char* sap,const char* pcm) {
    using namespace fyzz;
    native::Scene cpu;
    native::Scene gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        128u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    cpu.add_static_box({0.5f,0.5f,0.5f},{0.f,-0.5f,0.f});
    const BodyId fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,-0.5f,0.f});
    std::array<BodyId,3> dynamic_cpu{},dynamic_gpu{};
    for(std::size_t i=0;i<dynamic_cpu.size();++i) {
        const Vec3 position{0.f,0.4f+0.9f*float(i),0.f};
        dynamic_cpu[i]=cpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},position,1.f);
        dynamic_gpu[i]=gpu.add_dynamic_box(
            {0.5f,0.5f,0.5f},position,1.f);
    }
    constexpr float dt=1.f/120.f;
    bool coupled=false;
    for(std::uint32_t frame=0u;frame<12u;++frame) {
        cpu.step(dt);
        gpu.step(dt);
        if(frame==0u)
            native::GpuPcmFrictionAnchorFixture::verify_early_speculative_admission(
                cpu,gpu);
        require(cpu.pairs().size()==gpu.pairs().size() &&
                cpu.contacts().size()==gpu.contacts().size(),
                "GPU coupled rigid stack lost Sc contact-manager parity");
        for(const auto& contact:gpu.contacts())
            if(contact.pair.a!=fixed_gpu &&
               contact.pair.b!=fixed_gpu && contact.count>0u)
                coupled=true;
        for(std::size_t i=0;i<dynamic_cpu.size();++i) {
            const auto a=cpu.body_state(dynamic_cpu[i]);
            const auto b=gpu.body_state(dynamic_gpu[i]);
            require(std::fabs(a.position.x-b.position.x)<0.025f &&
                    std::fabs(a.position.y-b.position.y)<0.025f &&
                    std::fabs(a.position.z-b.position.z)<0.025f &&
                    std::fabs(a.velocity.x-b.velocity.x)<0.025f &&
                    std::fabs(a.velocity.y-b.velocity.y)<0.025f &&
                    std::fabs(a.velocity.z-b.velocity.z)<0.025f,
                    "GPU coupled rigid stack diverged from source Sc/Dy motion");
        }
        // CPU-owned island topology remains source-equivalent while
        // GPU Dy consumes its compact task and descriptor stream.
        for(std::size_t i=0;i<dynamic_cpu.size();++i)
            for(std::size_t j=i+1u;j<dynamic_cpu.size();++j) {
                const auto ca=dynamic_cpu[i],cb=dynamic_cpu[j];
                const auto ga=dynamic_gpu[i],gb=dynamic_gpu[j];
                require(
                    (cpu.accurate_island_id(ca)==cpu.accurate_island_id(cb))==
                    (gpu.accurate_island_id(ga)==gpu.accurate_island_id(gb)) &&
                    (cpu.speculative_island_id(ca)==cpu.speculative_island_id(cb))==
                    (gpu.speculative_island_id(ga)==gpu.speculative_island_id(gb)),
                    "GPU scene diverged from source Sc island membership");
            }
        require(gpu.solved_result_frame()==gpu.frame_index(),
                "GPU coupled rigid stack did not complete Dy solve");
    }
    require(coupled,
            "GPU coupled rigid stack never admitted dynamic-dynamic contacts");
}

void verify_gpu_single_island_hybrid_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream file(dy_shader,std::ios::binary);
    require(bool(file),"Source Dy shader bundle missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        128u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    cpu.add_static_box({0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    std::array<BodyId,2> a{},b{};
    for(std::size_t i=0;i<a.size();++i) {
        const Vec3 p{0.f,0.83f+float(i)*0.83f,0.f};
        a[i]=cpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
        b[i]=gpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
    }
    // The joint preserves one responding island while independent
    // free-flight actors exercise CPU task grouping without GPU graph work.
    cpu.create_center_distance_constraint(a[0],a[1],0.83f);
    gpu.create_center_distance_constraint(b[0],b[1],0.83f);
    constexpr unsigned free_count=8u;
    std::array<BodyId,free_count> free_cpu{},free_gpu{};
    RigidBodyDynamics free_dynamics{};
    free_dynamics.gravity_enabled=false;
    for(unsigned i=0u;i<free_count;++i) {
        const Vec3 p{20.f+4.f*float(i),5.f,0.f};
        free_cpu[i]=cpu.add_dynamic_box({0.3f,0.3f,0.3f},p,1.f);
        free_gpu[i]=gpu.add_dynamic_box({0.3f,0.3f,0.3f},p,1.f);
        cpu.set_dynamics(free_cpu[i],free_dynamics);
        gpu.set_dynamics(free_gpu[i],free_dynamics);
        cpu.set_velocity(free_cpu[i],{0.25f,0.f,0.f});
        gpu.set_velocity(free_gpu[i],{0.25f,0.f,0.f});
    }
    bool saw_pcm_contact=false;
    for(unsigned frame=0u;frame<6u;++frame) {
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        const auto schedule=producer->dy_launch_stats();
        saw_pcm_contact|=producer->dy_pcm_device_contact_rows()>0u;
        require(gpu.largest_accurate_island_size()==2u &&
                cpu.contacts().size()==gpu.contacts().size() &&
                gpu.active_accurate_island_count()>1u &&
                producer->dy_gpu_integrated_body_count()==2u+free_count &&
                schedule.device_colored_tasks>0u &&
                schedule.device_colored_descriptors>0u &&
                schedule.device_island_indirect_dispatches==0u,
            "Single accurate island redundantly rebuilt its numerical schedule");
        for(std::size_t i=0;i<a.size();++i)
            test::compound_compare_state(
                gpu.body_state(b[i]),cpu.body_state(a[i]),0.03f,
                "Single-island hybrid GPU Dy diverged from CPU PGS");
        for(unsigned i=0u;i<free_count;++i)
            test::compound_compare_state(
                gpu.body_state(free_gpu[i]),cpu.body_state(free_cpu[i]),
                0.003f,"Hybrid GPU free-flight integration diverged");
        if(frame==0u)
            std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm case=single_island"
                     <<" dy_dispatches="<<schedule.dispatches
                     <<" island_indirect_dispatches="
                     <<schedule.device_island_indirect_dispatches
                     <<" descriptor_upload_bytes="
                     <<schedule.source_descriptor_upload_bytes<<'\n';
    }
    require(saw_pcm_contact,
        "Single-island hybrid solver never processed GPU PCM contacts");
}
// Source Gu PCM rows, friction patches, and 1D joints share the same
// production Dy task partitions and device execution on both backends.
void verify_gpu_coupled_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader,
    std::uint32_t spring_mode) {
    using namespace fyzz;
    std::ifstream stream(dy_shader,std::ios::binary);
    require(bool(stream),"Source Dy shader bundle missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    require(producer->has_gpu_pcm_box() && producer->has_gpu_dy_distance(),
            "Source coupled GPU PCM/Dy stages were not installed");
    gpu.set_gpu_broadphase(std::move(owner));
    const auto floor_cpu=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.0f,0.0f,0.0f});
    const auto floor_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.0f,0.0f,0.0f});
    // Exercise non-identity static and dynamic anchor frames.
    constexpr Quat floor_turn{0.0f,0.0998334166f,0.0f,0.9950041653f};
    cpu.set_pose(floor_cpu,{0.0f,0.0f,0.0f},floor_turn);
    gpu.set_pose(floor_gpu,{0.0f,0.0f,0.0f},floor_turn);
    std::array<BodyId,2> a{},b{};
    for(std::size_t i=0;i<a.size();++i) {
        const Vec3 p{0.0f,0.83f+0.83f*float(i),0.0f};
        a[i]=cpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
        b[i]=gpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
    }
    constexpr Quat body_turn{0.0f,0.0499791693f,0.0f,0.9987502604f};
    cpu.set_pose(a[0],{0.0f,0.83f,0.0f},body_turn);
    gpu.set_pose(b[0],{0.0f,0.83f,0.0f},body_turn);
    cpu.create_center_distance_constraint(a[0],a[1],0.82f);
    gpu.create_center_distance_constraint(b[0],b[1],0.82f);
    if(spring_mode) {
        Material spring{};
        spring.restitution=-100.0f;
        spring.damping=2.0f;
        spring.compliant_acceleration_spring=spring_mode==2u;
        for(std::size_t i=0;i<a.size();++i) {
            cpu.set_material(a[i],spring);
            gpu.set_material(b[i],spring);
        }
    }
    // Zero-impulse Gu managers retain contact geometry but deliberately
    // prepare no Dy solver rows. Coexist with an independently live
    // contact and switch the same manager between zero/live responses.
    cpu.add_static_box({0.5f,0.5f,0.5f},{10.f,0.f,0.f});
    const auto capped_fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{10.f,0.f,0.f});
    const auto capped_cpu=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.f,0.84f,0.f},1.f);
    const auto capped_gpu=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.f,0.84f,0.f},1.f);
    RigidBodyDynamics capped_dynamics{};
    capped_dynamics.gravity_enabled=false;
    capped_dynamics.max_contact_impulse=0.f;
    cpu.set_dynamics(capped_cpu,capped_dynamics);
    gpu.set_dynamics(capped_gpu,capped_dynamics);
    // A vertical-only approach uses the source fallback tangent axis;
    // the other material modes exercise velocity-aligned tangents.
    const Vec3 initial_velocity=spring_mode==1u
        ? Vec3{0.0f,-0.1f,0.0f}
        : Vec3{0.25f,-0.1f,0.12f};
    cpu.set_velocity(a[0],initial_velocity);
    gpu.set_velocity(b[0],initial_velocity);
    constexpr float dt=1.f/120.f;
    bool saw_contact=false,saw_force=false,saw_friction=false;
    bool saw_correlated_multi_point=false;
    bool saw_early_pcm_history=false;
    bool saw_device_pcm_contact_rows=false;
    bool saw_compact_prep=false;
    std::int64_t saved_prep_bytes=0;
    double max_normal_error=0.0,max_friction_error=0.0;
    float max_position_error=0.f,max_velocity_error=0.f;
    std::size_t uploaded_bytes=0,downloaded_bytes=0;
    std::array<double,11> cpu_step_ms{},gpu_step_ms{};
    for(std::uint32_t frame=0;frame<12u;++frame) {
        if(frame==4u || frame==8u) {
            capped_dynamics.max_contact_impulse=frame==4u
                ? std::numeric_limits<float>::max():0.f;
            cpu.set_dynamics(capped_cpu,capped_dynamics);
            gpu.set_dynamics(capped_gpu,capped_dynamics);
        }
        if(frame==6u) {
            // An authored edit invalidates this contact body's device
            // pose without evicting the other active contact bodies.
            const auto original=cpu.body_state(capped_cpu);
            const auto current=gpu.body_state(capped_gpu);
            cpu.set_pose(capped_cpu,
                {original.position.x+0.12f,original.position.y,
                 original.position.z},original.orientation);
            gpu.set_pose(capped_gpu,
                {current.position.x+0.12f,current.position.y,
                 current.position.z},current.orientation);
        }
        const auto start_step=std::chrono::steady_clock::now();
        cpu.step(dt);
        const auto cpu_done=std::chrono::steady_clock::now();
        gpu.step(dt);
        if(frame>0u) {
            const auto gpu_done=std::chrono::steady_clock::now();
            cpu_step_ms[frame-1u]=
                std::chrono::duration<double,std::milli>(
                    cpu_done-start_step).count();
            gpu_step_ms[frame-1u]=
                std::chrono::duration<double,std::milli>(
                    gpu_done-cpu_done).count();
        }
        native::GpuPcmFrictionAnchorFixture::
            verify_compact_gpu_contact_admission(gpu);
        native::GpuPcmFrictionAnchorFixture::verify(cpu,gpu);
        saw_early_pcm_history |=
            native::GpuPcmFrictionAnchorFixture::
                had_early_friction_history(gpu);
        require(producer->source_handle_projection_count()==1u,
            "Source GPU SAP remapped unchanged Sc ElementSim handles");
        if(frame>0u)
            require(producer->last_frame().device_cached_box_count>0u,
                "Source Dy integrated Sc pose was reuploaded to GPU SAP");
        const auto integrated=producer->dy_gpu_integrated_body_count();
        const auto resident=producer->dy_resident_pose_count();
        const auto host_pose_bytes=producer->dy_host_pose_upload_bytes();
        require(integrated>=3u && resident<=integrated &&
                host_pose_bytes==0u,
            "GPU Dy resident contact poses have inconsistent transfer accounting");
        if(frame==0u)
            require(resident==0u,
                "First Sc contact solve reused an unauthorized device pose");
        else if(frame==6u)
            require(resident>=2u && resident<integrated &&
                    producer->dy_launch_stats().actor_local_upload_bytes>=
                        sizeof(physx_gpu::SourceDyActorLocalNative),
                "Authored Sc contact pose did not invalidate only its device owner");
        else
            require(resident>=2u && host_pose_bytes==0u,
                "Active GPU contact bodies still upload authored mass poses");
        const auto [uploaded,downloaded]=producer->dy_distance_transfer_bytes();
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_colored_tasks>0u &&
                schedule.device_colored_descriptors>0u &&
                schedule.host_degree_scanned_descriptors==0u &&
                schedule.device_island_indirect_dispatches==0u &&
                schedule.device_admitted_tasks==0u &&
                schedule.source_descriptor_upload_bytes==
                    schedule.device_colored_descriptors*sizeof(std::uint32_t),
            "Coupled Dy retained redundant host constraint-degree scanning");
        require(schedule.device_source_descriptors==
                    schedule.device_colored_descriptors &&
                schedule.source_descriptor_upload_bytes<
                    schedule.device_colored_descriptors*
                        sizeof(native::GpuDyDistanceDescriptor),
            "Source Sc still uploads complete contact/constraint descriptors");
        require(schedule.device_initialized_bodies>0u &&
                schedule.device_world_motion_bodies>0u &&
                schedule.device_world_motion_bodies<=
                    schedule.device_initialized_bodies &&
                schedule.device_body_upload_bytes==0u,
            "Coupled Dy failed device world-motion publication");
        const auto prep_saving=producer->dy_contact_prep_upload_saving();
        // Compare writeback with the source-sized upload baseline. The
        // GPU contact-prep payload is intentionally smaller now.
        require(downloaded>0u &&
                static_cast<std::int64_t>(downloaded)*2<
                    static_cast<std::int64_t>(uploaded)+prep_saving,
            "Source coupled GPU Dy writeback did not use compact output");
        uploaded_bytes+=uploaded;
        downloaded_bytes+=downloaded;
        const auto device_rows=producer->dy_pcm_device_contact_rows();
        saw_device_pcm_contact_rows |= device_rows>0u;
        saved_prep_bytes+=prep_saving;
        saw_compact_prep|=prep_saving>0u;
        require(producer->dy_upload_span_count()==
                    1u+schedule.device_island_indirect_dispatches &&
                producer->dy_cpu_gu_geometry_bytes()==0u,
            "PCM-only Dy source stage fragmented its HtoD upload");
        require(producer->dy_friction_prepared_rows()>0u &&
                producer->dy_friction_anchor_upload_bytes()==0u,
            "Resident PCM friction anchors were retransferred to GPU Dy");
        require(producer->dy_friction_writeback_bytes()*2u==
                    producer->dy_friction_prepared_rows()*12u,
            "Source GPU Dy downloaded tangent rows instead of anchor impulses");
        require(producer->dy_pcm_contact_upload_bytes_saved()==
                    device_rows*sizeof(native::GpuDyContactPoint),
                "Source PCM-derived Dy contacts still upload CPU shadow rows");
        if(device_rows)
            native::GpuPcmFrictionAnchorFixture::verify_device_contact_order(
                gpu,true,false);
        require(producer->dy_distance_dispatch_count()==frame+1u &&
                producer->dy_gpu_integrated_body_count()>0u,
            "Source coupled Sc scene never published GPU-integrated poses");
        require(cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size() &&
                cpu.solved_friction_patches().size()==
                    gpu.solved_friction_patches().size(),
            "GPU Dy contact/friction writeback has wrong source ownership");
        saw_contact |= !gpu.contacts().empty();
        bool found_capped=false;
        for(const auto& contact:gpu.contacts()) {
            const native::Pair capped_pair{capped_fixed_gpu,capped_gpu};
            if(!(contact.pair==capped_pair)) {
                saw_correlated_multi_point |= contact.count>1u;
                continue;
            }
            found_capped=true;
            if(frame<4u || frame>=8u)
                for(std::size_t k=0;k<contact.count;++k)
                    require(gpu.solved_normal_impulses()[
                                contact.contact_offset+k]==0.0,
                        "Zero-cap GPU Gu manager generated a Dy contact impulse");
        }
        require(found_capped,
            "Zero-cap source Gu manager lost its Sc contact identity");
        // The same manager alternates between excluded and live Dy rows.
        // Excluded writeback slots must retain Sc's original source state.
        const auto capped_expected=cpu.body_state(capped_cpu);
        const auto capped_actual=gpu.body_state(capped_gpu);
        require(std::fabs(capped_expected.position.y-
                          capped_actual.position.y)<0.02f &&
                std::fabs(capped_expected.velocity.y-
                          capped_actual.velocity.y)<0.02f,
                "GPU Dy wrote an excluded contact-manager result");
        for(std::size_t i=0;i<cpu.solved_normal_impulses().size();++i) {
            const auto force=cpu.solved_normal_impulses()[i];
            const auto actual=gpu.solved_normal_impulses()[i];
            saw_force |= force>0.0;
            max_normal_error=std::max(max_normal_error,std::fabs(force-actual));
            require(std::fabs(force-actual)<0.06,
                "Source GPU Dy normal impulse differs from CPU PGS");
        }
        for(std::size_t i=0;i<cpu.solved_friction_patches().size();++i) {
            const auto& ref=cpu.solved_friction_patches()[i];
            const auto& actual=gpu.solved_friction_patches()[i];
            require(ref.anchor_count==actual.anchor_count &&
                    ref.broken==actual.broken,
                "Source Dy GPU friction patch lost anchor or break state");
            saw_friction |= ref.anchor_count>0;
            for(std::size_t k=0;k<ref.anchor_count;++k) {
                const double dx=std::fabs(ref.anchor_impulses[k].x-
                                           actual.anchor_impulses[k].x);
                const double dy=std::fabs(ref.anchor_impulses[k].y-
                                           actual.anchor_impulses[k].y);
                max_friction_error=std::max({max_friction_error,dx,dy});
                if(!(dx<0.06 && dy<0.06))
                    std::cerr<<"Source Dy friction mismatch: spring="
                             <<spring_mode<<" frame="<<frame
                             <<" patch="<<i<<" anchor="<<k
                             <<" cpu=("<<ref.anchor_impulses[k].x<<","
                             <<ref.anchor_impulses[k].y<<") gpu=("
                             <<actual.anchor_impulses[k].x<<","
                             <<actual.anchor_impulses[k].y<<")\n";
                require(dx<0.06 && dy<0.06,
                    "Source Dy GPU friction impulses differ from CPU PGS");
            }
        }
        for(std::size_t i=0;i<a.size();++i) {
            const auto expected=cpu.body_state(a[i]);
            const auto actual=gpu.body_state(b[i]);
            const float position_error=std::max(
                std::fabs(expected.position.x-actual.position.x),
                std::fabs(expected.position.y-actual.position.y));
            const float velocity_error=std::max(
                std::fabs(expected.velocity.x-actual.velocity.x),
                std::fabs(expected.velocity.y-actual.velocity.y));
            max_position_error=std::max(max_position_error,position_error);
            max_velocity_error=std::max(max_velocity_error,velocity_error);
            require(position_error<0.03f && velocity_error<0.03f,
                "Coupled GPU Gu PCM/1D/contact/friction diverged from CPU Dy");
        }
    }
    require(saw_device_pcm_contact_rows,
        "Source Dy did not consume device-owned PCM contact rows");
    if(spring_mode==0u)
        require(saw_early_pcm_history,
            "Persistent PCM friction history was not validated during GPU Gu");
    require(saw_compact_prep,
        "Source Dy contact prep did not reuse GPU-resident body inputs");
    require(saw_contact && saw_force && saw_friction &&
            saw_correlated_multi_point,
        "Source coupled GPU test never exercised multi-point contact/friction");
    std::sort(cpu_step_ms.begin(),cpu_step_ms.end());
    std::sort(gpu_step_ms.begin(),gpu_step_ms.end());
    std::cout<<"FYZZXYZZY_METRIC suite=dy_pgs spring_mode="<<spring_mode
             <<" sampled_frames=11"
             <<" cpu_step_p50_ms="<<cpu_step_ms[5]
             <<" gpu_step_p50_ms="<<gpu_step_ms[5]
             <<" cpu_step_p90_ms="<<cpu_step_ms[9]
             <<" gpu_step_p90_ms="<<gpu_step_ms[9]<<'\n';
    std::cout<<"FYZZXYZZY_METRIC suite=dy_pgs spring_mode="<<spring_mode
             <<" frames=12 gpu_dispatches="
             <<producer->dy_distance_dispatch_count()
             <<" max_normal_abs_error="<<max_normal_error
             <<" max_friction_abs_error="<<max_friction_error
             <<" max_position_abs_error="<<max_position_error
             <<" max_velocity_abs_error="<<max_velocity_error
             <<" host_upload_bytes="<<uploaded_bytes
             <<" host_contact_prep_bytes_saved="<<saved_prep_bytes
             <<" host_download_bytes="<<downloaded_bytes<<'\n';
    if(spring_mode==0u) {
        // A moving kinematic contact supplies read-only solver velocity.
        // Its body ordinal must survive the native GPU descriptor
        // reconstruction even though it owns no writable PGS endpoint.
        const Vec3 kinematic_position{30.f,2.f,0.f};
        const Vec3 dynamic_position{30.8f,2.f,0.f};
        const Vec3 extents{0.5f,0.5f,0.5f};
        const auto kinematic_cpu=cpu.add_dynamic_box(
            extents,kinematic_position,1.f);
        const auto kinematic_gpu=gpu.add_dynamic_box(
            extents,kinematic_position,1.f);
        const auto responder_cpu=cpu.add_dynamic_box(
            extents,dynamic_position,1.f);
        const auto responder_gpu=gpu.add_dynamic_box(
            extents,dynamic_position,1.f);
        cpu.set_kinematic(kinematic_cpu,true);
        gpu.set_kinematic(kinematic_gpu,true);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        cpu.set_dynamics(responder_cpu,dynamics);
        gpu.set_dynamics(responder_gpu,dynamics);
        constexpr Quat identity{0.f,0.f,0.f,1.f};
        for(float target_x:{30.1f,30.2f}) {
            cpu.set_kinematic_target(
                kinematic_cpu,{target_x,2.f,0.f},identity);
            gpu.set_kinematic_target(
                kinematic_gpu,{target_x,2.f,0.f},identity);
            cpu.step(dt);
            gpu.step(dt);
            const auto schedule=producer->dy_launch_stats();
            require(schedule.device_source_descriptors>0u &&
                    schedule.device_island_indirect_dispatches==0u &&
                    schedule.source_descriptor_upload_bytes==
                        schedule.device_source_descriptors*sizeof(std::uint32_t),
                "Moving kinematic contact retained a host numerical descriptor upload");
            const native::Pair pair{kinematic_cpu,responder_cpu};
            const auto contains=[&](const native::Scene& scene) {
                const auto pairs=scene.pairs();
                return std::find(pairs.begin(),pairs.end(),pair)!=pairs.end();
            };
            require(contains(cpu) && contains(gpu) &&
                    std::any_of(gpu.contacts().begin(),gpu.contacts().end(),
                        [&](const auto& contact) {
                            return contact.pair==pair && contact.count>0u;
                        }),
                "Moving kinematic contact was omitted from Sc GPU admission");
            const auto actual=gpu.body_state(responder_gpu);
            const auto expected=cpu.body_state(responder_cpu);
            require(std::fabs(actual.position.x-expected.position.x)<0.03f &&
                    std::fabs(actual.velocity.x-expected.velocity.x)<0.03f &&
                    std::fabs(actual.velocity.y-expected.velocity.y)<0.03f,
                "GPU contact descriptor lost moving kinematic solver velocity");
        }

        // Refiltering a previously ineligible finite BP overlap does
        // not require invalidating or rebuilding the device SAP history.
        // Sc's sparse BP membership index retains that overlap while its
        // shared contact EdgeIndex follows the source motion-class rules.
        const Vec3 left{60.f,2.f,0.f},right{60.75f,2.f,0.f};
        const auto left_cpu=cpu.add_dynamic_box(extents,left,1.f);
        const auto left_gpu=gpu.add_dynamic_box(extents,left,1.f);
        const auto right_cpu=cpu.add_dynamic_box(extents,right,1.f);
        const auto right_gpu=gpu.add_dynamic_box(extents,right,1.f);
        cpu.set_kinematic(left_cpu,true);
        gpu.set_kinematic(left_gpu,true);
        cpu.set_kinematic(right_cpu,true);
        gpu.set_kinematic(right_gpu,true);
        const native::Pair pair{left_gpu,right_gpu};
        const auto admitted=[&](const native::Scene& scene) {
            const auto current=scene.pairs();
            return std::find(current.begin(),current.end(),pair)!=current.end();
        };
        cpu.advance_collision();
        gpu.advance_collision();
        require(!admitted(cpu) && !admitted(gpu) &&
                native::GpuPcmFrictionAnchorFixture::source_bp_contains(gpu,pair),
            "Sc BP registry lost a kinematic overlap before refiltering");
        const auto projection_count=producer->source_handle_projection_count();
        cpu.set_kinematic(left_cpu,false);
        gpu.set_kinematic(left_gpu,false);
        cpu.advance_collision();
        gpu.advance_collision();
        require(admitted(cpu) && admitted(gpu),
            "Sparse Sc refilter did not admit a previously ineligible overlap");
        const auto edge=native::GpuPcmFrictionAnchorFixture::
            contact_manager_edge(gpu,pair);
        require(edge!=std::numeric_limits<std::uint32_t>::max(),
            "Refiltered manager has no shared Sc EdgeIndex");
        cpu.set_kinematic(right_cpu,false);
        gpu.set_kinematic(right_gpu,false);
        cpu.advance_collision();
        gpu.advance_collision();
        require(admitted(cpu) && admitted(gpu) &&
                native::GpuPcmFrictionAnchorFixture::
                    contact_manager_edge(gpu,pair)==edge,
            "Refiltering replaced an unchanged Sc contact manager");
        cpu.set_kinematic(left_cpu,true);
        gpu.set_kinematic(left_gpu,true);
        cpu.advance_collision();
        gpu.advance_collision();
        require(admitted(cpu) && admitted(gpu) &&
                native::GpuPcmFrictionAnchorFixture::
                    contact_manager_edge(gpu,pair)==edge,
            "Motion-class edit retired a live Sc contact EdgeIndex");
        cpu.set_kinematic(right_cpu,true);
        gpu.set_kinematic(right_gpu,true);
        cpu.advance_collision();
        gpu.advance_collision();
        require(!admitted(cpu) && !admitted(gpu) &&
                native::GpuPcmFrictionAnchorFixture::source_bp_contains(gpu,pair),
            "Kinematic-only refilter lost retained BP overlap membership");
        require(producer->source_handle_projection_count()==projection_count,
            "Motion-class refilter rebuilt source SAP handles");
    }
}

// The optional legacy Gu generator remains the source authority when
// selected before admission. Its CPU-prepared rows must still be uploaded
// to the same actual GPU Dy solver, without PCM-origin row omission.
void verify_cpu_gu_gpu_dy_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled source Dy shader unavailable");
    std::vector<char> binary(
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());
    native::Scene cpu,gpu;
    cpu.set_pcm_box_box(false);
    gpu.set_pcm_box_box(false);
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    require(producer->has_gpu_pcm_box() && producer->has_gpu_dy_distance(),
        "Source legacy Gu/GPU Dy regression has no installed GPU solver");
    gpu.set_gpu_broadphase(std::move(owner));
    const auto fixed_cpu=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    const auto fixed_gpu=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    const auto moving_cpu=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.f,0.84f,0.f},1.f);
    const auto moving_gpu=gpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.f,0.84f,0.f},1.f);
    constexpr float dt=1.f/120.f;
    bool saw_contact=false;
    for(std::uint32_t frame=0;frame<4u;++frame) {
        cpu.step(dt);
        gpu.step(dt);
        const native::Pair expected_cpu{fixed_cpu,moving_cpu};
        const native::Pair expected_gpu{fixed_gpu,moving_gpu};
        const auto touching=[](auto contacts,native::Pair pair) {
            return std::any_of(contacts.begin(),contacts.end(),
                [pair](const auto& m) {
                    return m.pair==pair && m.count>0u;
                });
        };
        const bool reference=touching(cpu.contacts(),expected_cpu);
        const bool actual=touching(gpu.contacts(),expected_gpu);
        saw_contact|=actual;
        require(reference==actual,
            "Source GPU Dy legacy Gu touch differs from CPU");
        require(producer->dy_pcm_device_contact_rows()==0u &&
                producer->dy_pcm_contact_upload_bytes_saved()==0u &&
                producer->dy_upload_span_count()==1u &&
                producer->dy_cpu_gu_geometry_bytes()>0u &&
                producer->dy_cpu_gu_geometry_bytes()%sizeof(
                    native::GpuDyGuContactInput)==0u,
            "CPU Gu inputs did not use compact single-batch source upload");
        native::GpuPcmFrictionAnchorFixture::verify_device_contact_order(
            gpu,false,false);
        require(producer->dy_friction_anchor_upload_bytes()==
                    producer->dy_friction_prepared_rows()*
                        sizeof(native::GpuDyFrictionAnchorInput)/2u,
            "CPU Gu solver staged complete friction rows");
        require(producer->dy_friction_writeback_bytes()*2u==
                    producer->dy_friction_prepared_rows()*12u,
            "CPU Gu GPU Dy returned tangent rows instead of anchor impulses");
        require(producer->dy_contact_prep_upload_saving()==88,
            "CPU Gu contact-prep accounting includes friction anchors or geometry");
        require(producer->dy_distance_dispatch_count()==frame+1u,
            "CPU Gu contact rows did not enter source GPU Dy solver");
        const auto a=cpu.body_state(moving_cpu);
        const auto b=gpu.body_state(moving_gpu);
        require(std::fabs(a.position.y-b.position.y)<0.002f &&
                std::fabs(a.velocity.y-b.velocity.y)<0.002f,
            "Source GPU Dy legacy Gu contact solve diverged from CPU");
    }
    require(saw_contact,"Source legacy Gu fixture never generated a contact");
}

// An already-admitted native scene can change numerical executor while
// retaining its independent SAP, PCM and contact-edge publication.


void verify_integrated_aggregate_gpu_pipeline(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_path,const char* convex_path,const char* aggregate) {
    using namespace fyzz;
    const auto compiled=[backend](const char* path) {
        std::ifstream file(path,std::ios::binary);
        require(bool(file),"Integrated GPU scene shader is missing");
        std::vector<char> bytes(std::istreambuf_iterator<char>{file},
                                std::istreambuf_iterator<char>{});
        require(!bytes.empty(),"Integrated GPU scene shader is empty");
        compute::ShaderBinary binary{};
        binary.format=backend==compute::Backend::Vulkan
            ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
        binary.bytes.assign(bytes.begin(),bytes.end());
        return binary;
    };
    World gpu=create_native_gpu_world(
        backend,NativeGpuPipelineShaders{
            .sap=load_native_gpu_sap_shaders(backend,sap),
            .aggregates=load_native_gpu_aggregate_shaders(backend,aggregate),
            .pcm=load_native_gpu_pcm_shaders(backend,pcm),
            .dy=compiled(dy_path),.convex_dy=compiled(convex_path)},
        {.execution=ExecutionMode::NativePortableGpu,
         .contact_notifications=true});
    World cpu({.execution=ExecutionMode::NativeCpu,
               .contact_notifications=true});
    const auto group=gpu.create_aggregate(3u,false);
    const auto a=gpu.add_dynamic_box_to_aggregate(
        group,{0.5f,0.5f,0.5f},{0.f,0.83f,0.f},1.f);
    const auto ca=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.f,0.83f,0.f},1.f);
    const auto support=gpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    const auto csupport=cpu.add_static_box(
        {0.5f,0.5f,0.5f},{0.f,0.f,0.f});
    const auto b=gpu.add_dynamic_box_to_aggregate(
        group,{0.5f,0.5f,0.5f},{12.f,4.f,0.f},1.f);
    const auto cb=cpu.add_dynamic_box(
        {0.5f,0.5f,0.5f},{12.f,4.f,0.f},1.f);
    const auto contains=[](std::span<const ContactEvent> events,
                           BodyId x,BodyId y,ContactEventKind kind) {
        return std::any_of(events.begin(),events.end(),
            [&](const ContactEvent& e) {
                return e.kind==kind &&
                    ((e.body0==x && e.body1==y) ||
                     (e.body0==y && e.body1==x));
            });
    };
    bool found=false,lost=false,found_again=false;
    for(unsigned frame=0u;frame<6u;++frame) {
        if(frame==2u) {
            gpu.set_body_position(support,{20.f,0.f,0.f});
            cpu.set_body_position(csupport,{20.f,0.f,0.f});
        }
        if(frame==4u) {
            const float height=cpu.body_state(ca).position.y-0.83f;
            gpu.set_body_position(support,{0.f,height,0.f});
            cpu.set_body_position(csupport,{0.f,height,0.f});
        }
        gpu.set_gpu_dy_enabled(frame!=3u);
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        found|=contains(gpu.contact_events(),a,support,
                        ContactEventKind::Found) && frame<2u;
        lost|=contains(gpu.contact_events(),a,support,
                       ContactEventKind::Lost) && frame>=2u;
        found_again|=contains(gpu.contact_events(),a,support,
                              ContactEventKind::Found) && frame>=4u;
        require(gpu.contact_events().size()==cpu.contact_events().size(),
                "Integrated GPU aggregate changed Sc contact publication");
        test::compound_compare_state(
            gpu.body_state(a),cpu.body_state(ca),0.035f,
            "Integrated aggregate PCM/Dy responding body diverged");
        test::compound_compare_state(
            gpu.body_state(b),cpu.body_state(cb),0.035f,
            "Integrated aggregate free-body motion diverged");
    }
    require(found && lost && found_again,
            "Integrated aggregate PCM/Dy lost contact lifecycle transitions");
    gpu.remove_actor_from_aggregate(group,b);
    gpu.destroy_aggregate(group);
    gpu.step(1.f/120.f);
    require(gpu.body_state(a).position.y<3.f,
            "Integrated aggregate release invalidated live body state");
}

void verify_plane_pcm_without_finite_pcm(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream file(dy_shader,std::ios::binary);
    require(bool(file),"Plane-only PCM Dy shader missing");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    cpu.set_pcm_box_box(false);
    gpu.set_pcm_box_box(false);
    auto producer=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        128u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* source=producer.get();
    gpu.set_gpu_broadphase(std::move(producer));
    cpu.add_static_floor(0.f);
    gpu.add_static_floor(0.f);
    cpu.add_static_box({0.5f,0.5f,0.5f},{5.f,2.f,0.f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{5.f,2.f,0.f});
    cpu.add_static_sphere(0.5f,{10.f,2.f,0.f});
    gpu.add_static_sphere(0.5f,{10.f,2.f,0.f});
    const std::array<BodyId,3> c{
        cpu.add_dynamic_box({0.5f,0.5f,0.5f},{0.f,0.43f,0.f},1.f),
        cpu.add_dynamic_box({0.5f,0.5f,0.5f},{5.4f,2.f,0.f},1.f),
        cpu.add_dynamic_sphere(0.5f,{10.75f,2.f,0.f},1.f)};
    const std::array<BodyId,3> g{
        gpu.add_dynamic_box({0.5f,0.5f,0.5f},{0.f,0.43f,0.f},1.f),
        gpu.add_dynamic_box({0.5f,0.5f,0.5f},{5.4f,2.f,0.f},1.f),
        gpu.add_dynamic_sphere(0.5f,{10.75f,2.f,0.f},1.f)};
    for(unsigned frame=0u;frame<3u;++frame) {
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        require(source->pcm_host_contact_upload_bytes()>0u &&
                cpu.contacts().size()==3u &&
                cpu.contacts().size()==gpu.contacts().size(),
            "Plane-only PCM did not preserve finite legacy Gu contacts");
        for(std::size_t i=0u;i<c.size();++i)
            test::compound_compare_state(gpu.body_state(g[i]),
                cpu.body_state(c[i]),0.035f,
                "Plane-only GPU PCM and finite Gu motion diverged");
    }
}
void verify_dy_execution_switch(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream file(dy_shader,std::ios::binary);
    require(bool(file),"Compiled switchable Dy shader missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    cpu.set_pcm_plane_box(false);
    gpu.set_pcm_plane_box(false);
    cpu.set_contact_notifications(true);
    gpu.set_contact_notifications(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        128u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    cpu.add_static_floor(0.f);
    gpu.add_static_floor(0.f);
    cpu.add_static_box({0.5f,0.5f,0.5f},{5.f,10.f,0.f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{5.f,10.f,0.f});
    std::array<BodyId,3> a{},b{};
    for(std::size_t i=0u;i<2u;++i) {
        const Vec3 p=i==0u?Vec3{0.f,0.43f,0.f}:
                               Vec3{5.f,10.84f,0.f};
        a[i]=cpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
        b[i]=gpu.add_dynamic_box({0.5f,0.5f,0.5f},p,1.f);
    }
    a[2]=cpu.add_dynamic_box({0.25f,0.25f,0.25f},
                             {20.f,5.f,0.f},1.f);
    b[2]=gpu.add_dynamic_box({0.25f,0.25f,0.25f},
                             {20.f,5.f,0.f},1.f);
    constexpr std::array<bool,6> schedule{true,false,true,true,false,true};
    std::uint64_t gpu_dispatches=0u;
    for(std::size_t frame=0u;frame<schedule.size();++frame) {
        gpu.set_gpu_dy_enabled(schedule[frame]);
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        gpu_dispatches+=schedule[frame]?1u:0u;
        require(producer->dy_distance_dispatch_count()==gpu_dispatches &&
                (!schedule[frame] ||
                 producer->dy_pcm_device_contact_rows()>0u),
            "CPU/GPU Dy selection lost numerical ownership or GPU PCM input");
        require(cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size() &&
                cpu.contact_events().size()==gpu.contact_events().size(),
            "Switching Dy execution changed Sc contact publication");
        for(std::size_t i=0u;i<a.size();++i)
            test::compound_compare_state(
                gpu.body_state(b[i]),cpu.body_state(a[i]),0.03f,
                "GPU SAP/PCM and switched CPU/GPU Dy diverged");
    }
    // A rejected CPU Dy frame must roll back the same GPU SAP/PCM
    // transaction before numerical execution returns to the device.
    gpu.set_gpu_dy_enabled(false);
    const auto published=gpu.body_state(b[2]);
    const auto frame=gpu.frame_index();
    const auto solved=gpu.solved_result_frame();
    const auto prior_contacts=gpu.contacts().size();
    const float extreme=std::numeric_limits<float>::max()*0.75f;
    gpu.set_angular_velocity(b[2],{extreme,0.f,0.f});
    gpu.add_spatial_velocity_mod(
        b[2],{extreme,0.f,0.f},ForceMode::VelocityChange,true);
    bool rejected=false;
    try {gpu.step(1.f/120.f);}
    catch(const std::overflow_error&) {rejected=true;}
    require(rejected && gpu.frame_index()==frame &&
            gpu.solved_result_frame()==solved &&
            gpu.contacts().size()==prior_contacts &&
            producer->dy_distance_dispatch_count()==gpu_dispatches &&
            gpu.body_state(b[2]).position.x==published.position.x,
        "Rejected CPU Dy frame corrupted GPU contact or numerical lifetime");
    gpu.clear_spatial_velocity_mod(
        b[2],ForceMode::VelocityChange,true);
    gpu.set_angular_velocity(b[2],published.angular_velocity);
    gpu.set_gpu_dy_enabled(true);
    cpu.step(1.f/120.f);
    gpu.step(1.f/120.f);
    require(producer->dy_distance_dispatch_count()==gpu_dispatches+1u &&
            cpu.contacts().size()==gpu.contacts().size() &&
            cpu.contact_events().size()==gpu.contact_events().size(),
        "CPU Dy rollback prevented subsequent GPU PCM/Dy execution");
    for(std::size_t i=0u;i<a.size();++i)
        test::compound_compare_state(
            gpu.body_state(b[i]),cpu.body_state(a[i]),0.03f,
            "Recovered GPU Dy frame diverged after CPU Dy rollback");
}
 
// Explicit diagnostic plane Gu combines with device box PCM inside
// one Dy task. Default GPU plane PCM is verified separately; the
// diagnostic selection preserves CPU-origin Gu upload coverage.
void verify_mixed_gu_pcm_gpu_dy_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled mixed Gu/PCM Dy shader missing");
    std::vector<char> binary(
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{});
    for(const bool plane_first:{true,false}) {
        compute::ShaderBinary shader{};
        shader.format=backend==compute::Backend::Vulkan
            ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
        shader.bytes.assign(binary.begin(),binary.end());
        native::Scene cpu,gpu;
        cpu.set_enhanced_determinism(true);
        gpu.set_enhanced_determinism(true);
        // Retain a real CPU Gu producer for the mixed-input regression.
        cpu.set_pcm_plane_box(false);
        gpu.set_pcm_plane_box(false);
        auto owner=std::make_unique<
            physx_gpu::SourceScOwnedGpuBroadphaseNative>(
                backend,load_native_gpu_sap_shaders(backend,sap),
                128u,128u,std::nullopt,
                load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
        auto* producer=owner.get();
        gpu.set_gpu_broadphase(std::move(owner));
        std::array<BodyId,2> moving_cpu{},moving_gpu{};
        BodyId floor_cpu{},floor_gpu{},pcm_static_cpu{},pcm_static_gpu{};
        const auto add_plane_contact=[&] {
            floor_cpu=cpu.add_static_floor(0.f);
            floor_gpu=gpu.add_static_floor(0.f);
            moving_cpu[0]=cpu.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.f,0.43f,0.f},1.f);
            moving_gpu[0]=gpu.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.f,0.43f,0.f},1.f);
        };
        const auto add_pcm_contact=[&] {
            pcm_static_cpu=cpu.add_static_box(
                {0.5f,0.5f,0.5f},{5.f,10.f,0.f});
            pcm_static_gpu=gpu.add_static_box(
                {0.5f,0.5f,0.5f},{5.f,10.f,0.f});
            moving_cpu[1]=cpu.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5.f,10.84f,0.f},1.f);
            moving_gpu[1]=gpu.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5.f,10.84f,0.f},1.f);
        };
        if(plane_first) {
            add_plane_contact();
            add_pcm_contact();
        } else {
            add_pcm_contact();
            add_plane_contact();
        }
        constexpr float dt=1.f/120.f;
        for(std::uint32_t frame=0;frame<4u;++frame) {
            cpu.step(dt);
            gpu.step(dt);
            if(frame>0u)
                require(producer->last_frame().device_cached_box_count>0u,
                    "Mixed Gu/PCM Dy body still uploaded its solved Sc pose");
            require(producer->dy_distance_dispatch_count()==frame+1u &&
                    producer->dy_pcm_device_contact_rows()>0u &&
                    producer->dy_cpu_gu_geometry_bytes()>=
                        sizeof(native::GpuDyGuContactInput) &&
                    producer->dy_upload_span_count()==1u &&
                    producer->dy_launch_stats().device_island_indirect_dispatches==0u,
                "Mixed source Gu and PCM rows did not enter device-authored Dy tasks");
            native::GpuPcmFrictionAnchorFixture::verify_device_contact_order(
                gpu,true,true);
            const auto expected_cpu_anchors=
                native::GpuPcmFrictionAnchorFixture::cpu_gu_anchor_bytes(gpu);
            require(expected_cpu_anchors>0u &&
                    producer->dy_friction_anchor_upload_bytes()==
                        expected_cpu_anchors &&
                    expected_cpu_anchors<
                        producer->dy_friction_prepared_rows()*
                            sizeof(native::GpuDyFrictionAnchorInput)/2u,
                "Mixed Dy source upload still includes resident PCM anchors");
            require(producer->dy_friction_writeback_bytes()*2u==
                        producer->dy_friction_prepared_rows()*12u,
                "Mixed source Gu/PCM Dy returned tangent rows instead of anchors");
            native::GpuPcmFrictionAnchorFixture::verify(cpu,gpu);
            require(producer->dy_contact_prep_upload_saving()==176,
                "Mixed Gu/PCM contact-prep savings counted unrelated inputs");
            require(cpu.contacts().size()==gpu.contacts().size() &&
                    cpu.solved_normal_impulses().size()==
                        gpu.solved_normal_impulses().size() &&
                    cpu.solved_friction_patches().size()==
                        gpu.solved_friction_patches().size(),
                "Mixed source Gu/PCM managers lost Sc solver row identity");
            bool saw_plane=false,saw_pcm=false;
            for(const auto& m:gpu.contacts()) {
                saw_plane |= m.pcm_result_index==
                    std::numeric_limits<std::uint32_t>::max() && m.count>0u;
                saw_pcm |= m.pcm_result_index!=
                    std::numeric_limits<std::uint32_t>::max() && m.count>0u;
            }
            require(saw_plane && saw_pcm,
                "Mixed Dy task did not retain both source contact producers");
            // Both producers publish to the same persistent device Gu
            // history. CPU-only plane contacts must not disappear from
            // the GPU accurate graph just because they lack a PCM slot.
            for(const auto& manager:gpu.contacts()) {
                if(manager.pcm_result_index!=0xffffffffu || !manager.count)
                    continue;
                const auto touch=physx_gpu::SourceScDeviceIslandFixture::
                    touch(*producer,manager.edge_handle);
                require(touch[0]==1u && touch[1]==(frame>0u?1u:0u),
                    "CPU Gu touch was not journaled to the shared GPU EdgeIndex");
            }
            for(std::size_t i=0;i<cpu.solved_normal_impulses().size();++i) {
                const auto expected=cpu.solved_normal_impulses()[i];
                const auto actual=gpu.solved_normal_impulses()[i];
                if(std::fabs(expected-actual)>=0.08) {
                    const auto admission=producer->dy_launch_stats();
                    throw std::runtime_error(
                        "Mixed CPU Gu/GPU PCM normal impulse lost source ordinal"
                        " plane_first="+std::to_string(plane_first)+
                        " frame="+std::to_string(frame)+
                        " index="+std::to_string(i)+
                        " cpu="+std::to_string(expected)+
                        " gpu="+std::to_string(actual)+
                        " task_count="+std::to_string(admission.device_admitted_tasks)+
                        " body_count="+std::to_string(admission.device_admitted_bodies)+
                        " response_count="+std::to_string(admission.device_admitted_descriptors)+
                        " source_count="+std::to_string(admission.device_colored_descriptors)+
                        " input_edges="+native::GpuPcmFrictionAnchorFixture::
                            island_source_edges(gpu)+
                        " resident_contacts="+physx_gpu::SourceScDeviceIslandFixture::
                            contact_records(*producer)+
                        [&]{
                            std::string state;
                            for(const auto& manager:gpu.contacts()) {
                                const auto touch=physx_gpu::SourceScDeviceIslandFixture::
                                    touch(*producer,manager.edge_handle);
                                state+=" [edge="+std::to_string(manager.edge_handle)+
                                    " pcm="+std::to_string(
                                        manager.pcm_result_index!=0xffffffffu)+
                                    " rows="+std::to_string(manager.friction_patch_count)+
                                    " touch="+std::to_string(touch[0])+
                                    "/"+std::to_string(touch[1])+"]";
                            }
                            return state;
                        }());
                }
            }
            for(std::size_t i=0;i<cpu.solved_friction_patches().size();++i) {
                const auto& a=cpu.solved_friction_patches()[i];
                const auto& b=gpu.solved_friction_patches()[i];
                require(a.anchor_count==b.anchor_count && a.broken==b.broken,
                    "Mixed contact friction patch lost source correlation");
                for(std::size_t k=0;k<a.anchor_count;++k)
                    require(std::fabs(a.anchor_impulses[k].x-
                                      b.anchor_impulses[k].x)<0.08 &&
                            std::fabs(a.anchor_impulses[k].y-
                                      b.anchor_impulses[k].y)<0.08,
                        "Mixed source GPU friction impulse diverged from CPU Dy");
            }
            for(std::size_t i=0;i<moving_cpu.size();++i) {
                const auto a=cpu.body_state(moving_cpu[i]);
                const auto b=gpu.body_state(moving_gpu[i]);
                require(std::fabs(a.position.y-b.position.y)<0.02f &&
                        std::fabs(a.velocity.y-b.velocity.y)<0.02f,
                    "Mixed source Gu/PCM GPU Dy motion diverged from CPU");
            }
        }
        // The plane has no finite GPU handle. Its simulation-flag and
        // actor retirement must still retire the same Sc contact edge
        // without changing the independent GPU PCM box manager.
        const auto has_pair=[](auto pairs,native::Pair pair) {
            return std::find(pairs.begin(),pairs.end(),pair)!=pairs.end();
        };
        const native::Pair plane_pair_cpu{
            std::min(floor_cpu,moving_cpu[0]),
            std::max(floor_cpu,moving_cpu[0])};
        const native::Pair plane_pair_gpu{
            std::min(floor_gpu,moving_gpu[0]),
            std::max(floor_gpu,moving_gpu[0])};
        const native::Pair pcm_pair_cpu{
            std::min(pcm_static_cpu,moving_cpu[1]),
            std::max(pcm_static_cpu,moving_cpu[1])};
        const native::Pair pcm_pair_gpu{
            std::min(pcm_static_gpu,moving_gpu[1]),
            std::max(pcm_static_gpu,moving_gpu[1])};
        // Finite GPU lost/found deltas must preserve the independent
        // plane contact edge without rebuilding its candidate product.
        const auto saved_cpu=cpu.body_state(moving_cpu[1]);
        const auto saved_gpu=gpu.body_state(moving_gpu[1]);
        cpu.set_position(moving_cpu[1],
            {30.f,saved_cpu.position.y,saved_cpu.position.z});
        gpu.set_position(moving_gpu[1],
            {30.f,saved_gpu.position.y,saved_gpu.position.z});
        cpu.advance_collision();
        gpu.advance_collision();
        require(has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu) &&
                !has_pair(cpu.pairs(),pcm_pair_cpu) &&
                !has_pair(gpu.pairs(),pcm_pair_gpu),
            "Finite lost report modified an independent Sc plane contact");
        cpu.set_position(moving_cpu[1],saved_cpu.position);
        gpu.set_position(moving_gpu[1],saved_gpu.position);
        cpu.advance_collision();
        gpu.advance_collision();
        require(has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu),
            "Finite found report failed to preserve Sc plane membership");
        cpu.set_shape_simulation_enabled(floor_cpu,false);
        gpu.set_shape_simulation_enabled(floor_gpu,false);
        cpu.advance_collision();
        gpu.advance_collision();
        require(!has_pair(cpu.pairs(),plane_pair_cpu) &&
                !has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu) &&
                cpu.contacts().size()==gpu.contacts().size(),
            "Disabling a Sc plane modified finite GPU contact membership");
        cpu.set_shape_simulation_enabled(floor_cpu,true);
        gpu.set_shape_simulation_enabled(floor_gpu,true);
        cpu.advance_collision();
        gpu.advance_collision();
        require(has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu),
            "Reenabling a Sc plane modified finite GPU contact membership");
        // The source rigid group transition refilters the original
        // finite BP registry and the independent plane candidate set.
        cpu.set_kinematic(moving_cpu[0],true);
        gpu.set_kinematic(moving_gpu[0],true);
        cpu.advance_collision();
        gpu.advance_collision();
        require(!has_pair(cpu.pairs(),plane_pair_cpu) &&
                !has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu),
            "Rigid-to-kinematic Sc refilter retained plane contact");
        cpu.set_kinematic(moving_cpu[0],false);
        gpu.set_kinematic(moving_gpu[0],false);
        cpu.advance_collision();
        gpu.advance_collision();
        require(has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu),
            "Kinematic-to-rigid Sc refilter lost plane admission");
        // A plane admitted after completed GPU frames has its own Sc
        // identity but no finite AABB. Both dynamic actors become
        // candidates, irrespective of actual Gu touch.
        const BodyId other_cpu=cpu.add_static_floor(-20.f);
        const BodyId other_gpu=gpu.add_static_floor(-20.f);
        const native::Pair other_pair_cpu{
            std::min(other_cpu,moving_cpu[1]),
            std::max(other_cpu,moving_cpu[1])};
        const native::Pair other_pair_gpu{
            std::min(other_gpu,moving_gpu[1]),
            std::max(other_gpu,moving_gpu[1])};
        const native::Pair other_first_cpu{
            std::min(other_cpu,moving_cpu[0]),
            std::max(other_cpu,moving_cpu[0])};
        const native::Pair other_first_gpu{
            std::min(other_gpu,moving_gpu[0]),
            std::max(other_gpu,moving_gpu[0])};
        cpu.advance_collision();
        gpu.advance_collision();
        require(has_pair(cpu.pairs(),other_pair_cpu) &&
                has_pair(gpu.pairs(),other_pair_gpu) &&
                has_pair(cpu.pairs(),other_first_cpu) &&
                has_pair(gpu.pairs(),other_first_gpu) &&
                has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu),
            "Second live Sc plane failed independent candidate admission");
        cpu.remove(other_cpu);
        gpu.remove(other_gpu);
        cpu.advance_collision();
        gpu.advance_collision();
        require(!has_pair(cpu.pairs(),other_pair_cpu) &&
                !has_pair(gpu.pairs(),other_pair_gpu) &&
                !has_pair(cpu.pairs(),other_first_cpu) &&
                !has_pair(gpu.pairs(),other_first_gpu) &&
                has_pair(cpu.pairs(),plane_pair_cpu) &&
                has_pair(gpu.pairs(),plane_pair_gpu),
            "Retiring one Sc plane modified surviving plane membership");
        cpu.remove(floor_cpu);
        gpu.remove(floor_gpu);
        cpu.advance_collision();
        gpu.advance_collision();
        require(!has_pair(cpu.pairs(),plane_pair_cpu) &&
                !has_pair(gpu.pairs(),plane_pair_gpu) &&
                has_pair(cpu.pairs(),pcm_pair_cpu) &&
                has_pair(gpu.pairs(),pcm_pair_gpu),
            "Retiring Sc plane modified surviving finite PCM membership");
    }
}

// The explicitly installed device Dy 1D stage must replace the
// production PGS solve, not merely compute a disconnected fixture result.
void verify_gpu_distance_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled source Dy shader unavailable");
    std::vector<char> binary(
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());

    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    require(producer->has_gpu_dy_distance(),
            "Explicit source Dy distance stage was not installed");
    gpu.set_gpu_broadphase(std::move(owner));
    constexpr float dt=1.f/120.f;
    std::array<BodyId,4> ids_cpu{},ids_gpu{};
    for(std::size_t i=0;i<ids_cpu.size();++i) {
        const Vec3 position{float(i)*4.0f,3.0f,0.0f};
        const Vec3 extents=i==0u
            ? Vec3{0.2f,0.45f,0.3f}:Vec3{0.3f,0.3f,0.3f};
        ids_cpu[i]=cpu.add_dynamic_box(extents,position,1.f);
        ids_gpu[i]=gpu.add_dynamic_box(extents,position,1.f);
        if(i==0u) {
            constexpr Quat turn{0.0f,0.258819045f,0.0f,0.965925826f};
            cpu.set_pose(ids_cpu[i],position,turn);
            gpu.set_pose(ids_gpu[i],position,turn);
        }
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.position_iterations=5;
        d.velocity_iterations=2;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        if(i==0u) {
            d.lock_flags=LockAngularX;
            d.max_angular_velocity=1.0f;
            d.gyroscopic_forces_enabled=true;
        }
        cpu.set_dynamics(ids_cpu[i],d);
        gpu.set_dynamics(ids_gpu[i],d);
    }
    // Two disconnected accurate islands must advance through independent
    // source task workgroups. Off-center anchors exercise angular Jacobians.
    cpu.create_anchored_distance_constraint(ids_cpu[0],ids_cpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    gpu.create_anchored_distance_constraint(ids_gpu[0],ids_gpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    const auto recycled_cpu=
        cpu.create_center_distance_constraint(ids_cpu[2],ids_cpu[3],3.5f);
    const auto recycled_gpu=
        gpu.create_center_distance_constraint(ids_gpu[2],ids_gpu[3],3.5f);
    // A source 1D joint against an immovable body must use the
    // original static-world sentinel, never a fabricated solver body.
    const auto fixed_cpu=cpu.add_static_box(
        {0.3f,0.3f,0.3f},{40.f,3.f,0.f});
    const auto fixed_gpu=gpu.add_static_box(
        {0.3f,0.3f,0.3f},{40.f,3.f,0.f});
    const auto anchored_cpu=cpu.add_dynamic_box(
        {0.3f,0.3f,0.3f},{44.f,3.f,0.f},1.f);
    const auto anchored_gpu=gpu.add_dynamic_box(
        {0.3f,0.3f,0.3f},{44.f,3.f,0.f},1.f);
    RigidBodyDynamics anchored_dynamics{};
    anchored_dynamics.gravity_enabled=false;
    anchored_dynamics.linear_damping=0.f;
    anchored_dynamics.angular_damping=0.f;
    cpu.set_dynamics(anchored_cpu,anchored_dynamics);
    gpu.set_dynamics(anchored_gpu,anchored_dynamics);
    cpu.create_center_distance_constraint(
        fixed_cpu,anchored_cpu,3.8f);
    gpu.create_center_distance_constraint(
        fixed_gpu,anchored_gpu,3.8f);
    cpu.set_velocity(ids_cpu[0],{0.4f,0.2f,0.0f});
    gpu.set_velocity(ids_gpu[0],{0.4f,0.2f,0.0f});
    // The original X angular component affects speed clamping and
    // gyroscopic torque before the final LockAngularX projection.
    cpu.set_angular_velocity(ids_cpu[0],{10.0f,1.0f,0.0f});
    gpu.set_angular_velocity(ids_gpu[0],{10.0f,1.0f,0.0f});
    cpu.set_velocity(ids_cpu[2],{-0.6f,0.0f,0.0f});
    gpu.set_velocity(ids_gpu[2],{-0.6f,0.0f,0.0f});
    std::size_t previous_cached=0u;
    for(int frame=0;frame<12;++frame) {
        if(frame==4 || frame==8) {
            // Idempotent setGlobalPose preserves device-resident bounds;
            // an actual pose edit invalidates only that ElementSim.
            // The second edit changes every solver body and requires
            // a complete Sc-authored transform upload.
            const bool change_all=frame==8;
            for(std::size_t i=0;i<ids_cpu.size();++i) {
                auto expected=cpu.body_state(ids_cpu[i]);
                auto actual=gpu.body_state(ids_gpu[i]);
                if(change_all || i==0u) {
                    expected.position.x+=0.12f;
                    actual.position.x+=0.12f;
                }
                cpu.set_pose(ids_cpu[i],expected.position,expected.orientation);
                gpu.set_pose(ids_gpu[i],actual.position,actual.orientation);
            }
            auto expected=cpu.body_state(anchored_cpu);
            auto actual=gpu.body_state(anchored_gpu);
            if(change_all) {
                expected.position.x+=0.12f;
                actual.position.x+=0.12f;
            }
            cpu.set_pose(anchored_cpu,expected.position,expected.orientation);
            gpu.set_pose(anchored_gpu,actual.position,actual.orientation);
        }
        cpu.step(dt);
        gpu.step(dt);
        require(gpu.accurate_island_id(ids_gpu[0])==
                    gpu.accurate_island_id(ids_gpu[1]) &&
                gpu.accurate_island_id(ids_gpu[2])==
                    gpu.accurate_island_id(ids_gpu[3]) &&
                gpu.accurate_island_id(ids_gpu[0])!=
                    gpu.accurate_island_id(ids_gpu[2]) &&
                gpu.accurate_island_id(anchored_gpu)!=
                    gpu.accurate_island_id(ids_gpu[0]) &&
                gpu.accurate_island_id(anchored_gpu)!=
                    gpu.accurate_island_id(ids_gpu[2]),
            "Sc joint-only island membership lost authored constraints");
        const auto cached=producer->last_frame().device_cached_box_count;
        if(frame==0 || frame==8)
            require(cached==0u,
                "Source GPU SAP reused a newly admitted or authored Sc pose");
        else if(frame==4)
            require(cached>0u && cached<previous_cached,
                "Source GPU SAP did not invalidate only the edited Sc pose");
        else
            require(cached>0u,
                "Source GPU Dy shape residency did not resume after Sc edits");
        previous_cached=cached;
        require(cpu.contacts().empty() && gpu.contacts().empty(),
            "Source GPU Dy 1D fixture unexpectedly generated Gu contacts");
        require(producer->dy_distance_dispatch_count()==
                    static_cast<std::uint64_t>(frame+1) &&
                producer->dy_gpu_integrated_body_count()>0u &&
                producer->dy_launch_stats().device_joint_input_rows==3u &&
                producer->dy_launch_stats().device_joint_geometry_count==3u &&
                (frame==0?producer->dy_launch_stats().device_joint_metadata_upload_bytes==
                    3u*sizeof(native::GpuDyJointInput)
                    :producer->dy_launch_stats().device_joint_metadata_upload_bytes==0u) &&
                producer->dy_launch_stats().device_joint_metadata_reuses==
                    (frame==0?0u:3u),
            "Source GPU 1D joint rows did not reach device preparation");
        const auto [uploaded,downloaded]=
            producer->dy_distance_transfer_bytes();
        // Five dynamic solver bodies and three disjoint task reports.
        // Joint impulses stay in their device PGS rows, not the
        // completed Sc result stream.
        constexpr std::size_t numerical_output=
            5u*(52u+sizeof(native::GpuDyIntegratedPose))+
            3u*sizeof(std::uint32_t);
        const auto& admission=producer->dy_launch_stats();
        const std::size_t scheduling_output=admission.device_island_indirect_dispatches
            ?16u+admission.device_colored_tasks*sizeof(native::GpuDyDistanceTask)+
                 admission.device_world_motion_bodies*sizeof(std::uint32_t)
            :0u;
        require(downloaded==numerical_output+scheduling_output &&
                downloaded<uploaded/2u,
            "Source Dy GPU fetched transient joint row impulses");
        const auto fixed_expected=cpu.body_state(anchored_cpu);
        const auto fixed_actual=gpu.body_state(anchored_gpu);
        require(std::fabs(fixed_expected.position.x-
                          fixed_actual.position.x)<0.001f &&
                std::fabs(fixed_expected.velocity.x-
                          fixed_actual.velocity.x)<0.001f,
                "Static-world Dy GPU 1D constraint changed source dynamics");
        for(std::size_t i=0;i<ids_cpu.size();++i) {
            const auto expected=cpu.body_state(ids_cpu[i]);
            const auto actual=gpu.body_state(ids_gpu[i]);
            require(std::fabs(expected.position.x-actual.position.x)<0.001f &&
                    std::fabs(expected.position.y-actual.position.y)<0.001f &&
                    std::fabs(expected.velocity.x-actual.velocity.x)<0.001f &&
                    std::fabs(expected.angular_velocity.y-
                              actual.angular_velocity.y)<0.001f &&
                    std::fabs(expected.angular_velocity.z-
                              actual.angular_velocity.z)<0.001f &&
                    std::fabs(expected.orientation.x-actual.orientation.x)<0.001f &&
                    std::fabs(expected.orientation.y-actual.orientation.y)<0.001f &&
                    std::fabs(expected.orientation.z-actual.orientation.z)<0.001f &&
                    std::fabs(expected.orientation.w-actual.orientation.w)<0.001f,
                "Source GPU Dy locked-axis integration diverged from CPU PGS");
        }
    }

    // A moving kinematic joint endpoint contributes target velocity
    // without entering the dynamic integration or mass-response pool.
    const auto kinematic_cpu=cpu.add_dynamic_box(
        {0.25f,0.25f,0.25f},{80.f,3.f,0.f},1.f);
    const auto kinematic_gpu=gpu.add_dynamic_box(
        {0.25f,0.25f,0.25f},{80.f,3.f,0.f},1.f);
    const auto follower_cpu=cpu.add_dynamic_box(
        {0.25f,0.25f,0.25f},{83.f,3.f,0.f},1.f);
    const auto follower_gpu=gpu.add_dynamic_box(
        {0.25f,0.25f,0.25f},{83.f,3.f,0.f},1.f);
    cpu.set_kinematic(kinematic_cpu,true);
    gpu.set_kinematic(kinematic_gpu,true);
    RigidBodyDynamics follower_dynamics{};
    follower_dynamics.gravity_enabled=false;
    follower_dynamics.linear_damping=0.0f;
    follower_dynamics.angular_damping=0.0f;
    cpu.set_dynamics(follower_cpu,follower_dynamics);
    gpu.set_dynamics(follower_gpu,follower_dynamics);
    const auto moving_cpu=cpu.create_anchored_distance_constraint(
        kinematic_cpu,follower_cpu,
        {0.f,0.2f,0.f},{0.f,-0.2f,0.f},3.f);
    const auto moving_gpu=gpu.create_anchored_distance_constraint(
        kinematic_gpu,follower_gpu,
        {0.f,0.2f,0.f},{0.f,-0.2f,0.f},3.f);
    for(unsigned frame=0u;frame<4u;++frame) {
        const Vec3 target{80.1f+0.1f*float(frame),3.f,0.f};
        const float half_angle=0.005f*float(frame+1u);
        const Quat orientation{
            0.f,0.f,std::sin(half_angle),std::cos(half_angle)};
        cpu.set_kinematic_target(kinematic_cpu,target,orientation);
        gpu.set_kinematic_target(kinematic_gpu,target,orientation);
        cpu.step(dt);
        gpu.step(dt);
        require(producer->dy_launch_stats().device_joint_input_rows>=4u &&
                producer->dy_launch_stats().device_joint_geometry_count>=4u &&
                producer->dy_launch_stats().device_joint_metadata_upload_bytes<=
                    4u*sizeof(native::GpuDyJointInput),
            "Native GPU Dy skipped moving kinematic 1D joint preparation");
        const auto expected=cpu.body_state(follower_cpu);
        const auto actual=gpu.body_state(follower_gpu);
        require(std::abs(expected.position.x-actual.position.x)<0.003f &&
                std::abs(expected.velocity.x-actual.velocity.x)<0.003f &&
                std::abs(expected.angular_velocity.z-
                         actual.angular_velocity.z)<0.003f,
            "GPU 1D joint ignored kinematic target linear/angular velocity");
    }

    // End the moving-kinematic fixture before inspecting exact sparse
    // upload counts for an unrelated dynamic joint's source slot.
    cpu.retire_constraint(moving_cpu);
    gpu.retire_constraint(moving_gpu);
    // Sc IDTracker recycles the same constraint slot with a new generation.
    // The already-resident old definition must not alias a different
    // authored frame or distance, even when the bodies remain unchanged.
    cpu.retire_constraint(recycled_cpu);
    gpu.retire_constraint(recycled_gpu);
    cpu.step(dt);
    gpu.step(dt);
    require(gpu.accurate_island_id(ids_gpu[2])!=
                gpu.accurate_island_id(ids_gpu[3]),
        "Retired Sc joint retained a stale dynamic island link");
    cpu.create_anchored_distance_constraint(ids_cpu[2],ids_cpu[3],
        {0.0f,0.15f,0.0f},{0.0f,-0.15f,0.0f},3.2f);
    gpu.create_anchored_distance_constraint(ids_gpu[2],ids_gpu[3],
        {0.0f,0.15f,0.0f},{0.0f,-0.15f,0.0f},3.2f);
    for(unsigned frame=0u;frame<3u;++frame) {
        cpu.step(dt);
        gpu.step(dt);
        require(gpu.accurate_island_id(ids_gpu[2])==
                    gpu.accurate_island_id(ids_gpu[3]),
            "Recycled joint did not reconnect the accurate island");
        const auto stats=producer->dy_launch_stats();
        if(!(stats.device_joint_geometry_count==3u &&
             stats.device_joint_metadata_upload_bytes==
                 (frame==0u?sizeof(native::GpuDyJointInput):0u) &&
             stats.device_joint_metadata_reuses==(frame==0u?2u:3u)))
            throw std::runtime_error(
                "Recycled native joint slot residency mismatch: frame="+
                std::to_string(frame)+
                " joints="+std::to_string(stats.device_joint_geometry_count)+
                " upload="+std::to_string(stats.device_joint_metadata_upload_bytes)+
                " reuse="+std::to_string(stats.device_joint_metadata_reuses));
        for(const auto i:{2u,3u}) {
            const auto expected=cpu.body_state(ids_cpu[i]);
            const auto actual=gpu.body_state(ids_gpu[i]);
            require(std::abs(expected.position.x-actual.position.x)<0.002f &&
                    std::abs(expected.position.y-actual.position.y)<0.002f &&
                    std::abs(expected.velocity.x-actual.velocity.x)<0.002f &&
                    std::abs(expected.velocity.y-actual.velocity.y)<0.002f,
                "Recycled GPU joint diverged from source constraint response");
        }
    }
}
} // namespace

// Sphere/box and sphere/sphere witnesses execute in the device PCM
// contact stream, then feed the same GPU Dy task as box PCM contacts.
// Sphere/plane also uses device PCM without a finite plane volume.
void verify_sphere_gpu_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream stream(dy_shader,std::ios::binary);
    require(bool(stream),"Sphere GPU Dy shader bundle missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    auto owner=std::make_unique<
        physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto add_scene=[](native::Scene& scene) {
        std::array<BodyId,3> moving{};
        scene.add_static_floor(0.f);
        moving[0]=scene.add_dynamic_sphere(
            0.5f,{0.f,0.45f,0.f},1.f);
        scene.add_static_sphere(0.5f,{5.f,10.f,0.f});
        moving[1]=scene.add_dynamic_sphere(
            0.5f,{5.92f,10.f,0.f},1.f);
        scene.add_static_box({0.5f,0.5f,0.5f},{10.f,10.f,0.f});
        moving[2]=scene.add_dynamic_sphere(
            0.25f,{10.7f,10.f,0.f},1.f);
        RigidBodyDynamics dynamic{};
        dynamic.gravity_enabled=false;
        for(auto id:moving)scene.set_dynamics(id,dynamic);
        // Three closing impacts exercise contact impulses rather than
        // merely preserving speculative overlaps at zero velocity.
        scene.set_velocity(moving[0],{0.f,-2.f,0.f});
        scene.set_velocity(moving[1],{-2.f,0.f,0.f});
        scene.set_velocity(moving[2],{-2.f,0.f,0.f});
        return moving;
    };
    const auto cpu_ids=add_scene(cpu);
    const auto gpu_ids=add_scene(gpu);
    constexpr float dt=1.f/120.f;
    bool touched_all=false,saw_device_bounds=false;
    for(std::uint32_t frame=0u;frame<8u;++frame) {
        if(frame==3u) {
            constexpr Quat rotate{0.f,0.0998334166f,0.f,0.9950041653f};
            for(std::size_t i=0;i<cpu_ids.size();++i) {
                auto a=cpu.body_state(cpu_ids[i]);
                auto b=gpu.body_state(gpu_ids[i]);
                cpu.set_pose(cpu_ids[i],a.position,rotate);
                gpu.set_pose(gpu_ids[i],b.position,rotate);
            }
        }
        cpu.step(dt);
        gpu.step(dt);
        require(producer->pcm_continuation_commands()==0u,
            "Sphere/plane PCM scheduled unnecessary GJK/EPA continuation");
        const auto integrated=producer->dy_gpu_integrated_body_count();
        const auto resident=producer->dy_resident_pose_count();
        const auto uploaded=producer->dy_host_pose_upload_bytes();
        require(integrated==gpu_ids.size() && resident<=integrated &&
                uploaded==0u,
            "GPU sphere mass poses have inconsistent source upload accounting");
        if(frame==0u)
            require(resident==0u,
                "New sphere actors reused unauthorised device mass poses");
        else if(frame==3u)
            require(resident<integrated &&
                    producer->dy_launch_stats().actor_local_upload_bytes>=
                        sizeof(physx_gpu::SourceDyActorLocalNative),
                "Authored sphere quaternion did not invalidate its device pose");
        else
            require(resident>=2u && uploaded==0u,
                "Sphere contact preparation still uploads mass poses");
        require(cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size(),
            "GPU sphere scene changed Sc PCM/Gu manager identity");
        if(frame==0u) {
            const auto pcm_bytes=producer->pcm_contact_readback_bytes();
            const auto host_contact_bytes=
                producer->pcm_host_contact_upload_bytes();
            const auto pcm_rows=producer->dy_pcm_device_contact_rows();
            const auto cpu_bytes=producer->dy_cpu_gu_geometry_bytes();
            // Sc admits one speculative plane manager per dynamic
            // sphere. The two elevated spheres have no physical plane
            // touch, but their GPU PCM records still execute this frame.
            constexpr std::size_t pcm_managers=5u;
            if(gpu.contacts().size()!=3u || cpu_bytes!=0u ||
               pcm_rows!=3u || pcm_bytes!=0u ||
               host_contact_bytes!=pcm_managers*sizeof(
                   physx_gpu::SourcePcmGpuContactStreamNative)) {
                throw std::runtime_error(
                    "Sphere GPU contact path: managers="+
                    std::to_string(gpu.contacts().size())+
                    " device_rows="+std::to_string(pcm_rows)+
                    " cpu_gu_bytes="+std::to_string(cpu_bytes)+
                    " pcm_readback_bytes="+std::to_string(pcm_bytes)+
                    " host_pcm_upload_bytes="+std::to_string(host_contact_bytes));
            }
            require(cpu.contacts().size()==3u,
                "Source sphere impact fixtures lost a contact manager");
            for(std::size_t j=0;j<cpu.contacts().size();++j) {
                const auto& source=cpu.contacts()[j];
                const auto& device=gpu.contacts()[j];
                require(source.count==1u && device.count==1u,
                    "Sphere impact must retain one Gu contact per manager");
                const double expected=cpu.solved_normal_impulses()[source.contact_offset];
                const double actual=gpu.solved_normal_impulses()[device.contact_offset];
                require(expected>0.001 && actual>0.001 &&
                        std::fabs(expected-actual)<0.05,
                    "GPU sphere impact did not produce source-equivalent normal impulse");
            }
            touched_all=true;
        }
        require(producer->dy_distance_dispatch_count()==frame+1u &&
                producer->dy_gpu_integrated_body_count()>0u,
            "Source sphere bodies did not execute GPU Dy integration");
        // Cross-check Gu's actual surface witnesses and orientations,
        // not just agreement of the integrated body positions.
        for(std::size_t j=0;j<cpu.contacts().size();++j) {
            const auto& expected=cpu.contacts()[j];
            const auto& actual=gpu.contacts()[j];
            require(expected.pair==actual.pair && expected.count==actual.count,
                "Device sphere Gu changed Sc contact-manager geometry identity");
            for(std::uint32_t k=0;k<expected.count;++k) {
                const auto& a=expected.points[k],&b=actual.points[k];
                require(std::fabs(a.position.x-b.position.x)<0.0004f &&
                        std::fabs(a.position.y-b.position.y)<0.0004f &&
                        std::fabs(a.position.z-b.position.z)<0.0004f &&
                        std::fabs(a.normal.x-b.normal.x)<0.0004f &&
                        std::fabs(a.normal.y-b.normal.y)<0.0004f &&
                        std::fabs(a.normal.z-b.normal.z)<0.0004f &&
                        std::fabs(a.separation-b.separation)<0.0004f,
                    "Device sphere Gu witness diverged from source PCM");
            }
        }
        // Spherical geometry carries its own radius in the shared shape
        // cache; Dy publishes exact spherical bounds after integration.
        saw_device_bounds|=producer->last_frame().device_cached_box_count>0u;
        for(std::size_t i=0;i<cpu_ids.size();++i) {
            const auto a=cpu.body_state(cpu_ids[i]);
            const auto b=gpu.body_state(gpu_ids[i]);
            require(std::fabs(a.position.x-b.position.x)<0.035f &&
                    std::fabs(a.position.y-b.position.y)<0.035f &&
                    std::fabs(a.velocity.x-b.velocity.x)<0.04f &&
                    std::fabs(a.velocity.y-b.velocity.y)<0.04f,
                "Source sphere GPU Dy motion diverged from native CPU");
        }
    }
    require(touched_all,
        "GPU sphere scene did not publish all source contacts");
    require(saw_device_bounds,
        "GPU sphere integration did not retain a device-authored bounds cache");

    // The embedded-center exit-face rule selects Z before X/Y on a tie.
    // A GPU contact must retain the actual box witness, not approximate
    // the sphere's surface or introduce a CPU-only special case.
    cpu.add_static_box({0.5f,0.5f,0.5f},{20.f,10.f,0.f});
    gpu.add_static_box({0.5f,0.5f,0.5f},{20.f,10.f,0.f});
    cpu.add_dynamic_sphere(0.25f,{20.f,10.f,0.f},1.f);
    gpu.add_dynamic_sphere(0.25f,{20.f,10.f,0.f},1.f);
    const auto solved_before=producer->dy_distance_dispatch_count();
    cpu.advance_collision(true);
    gpu.advance_collision(true);
    require(producer->pcm_contact_readback_bytes()==0u &&
            producer->pcm_host_contact_upload_bytes()>0u &&
            producer->dy_distance_dispatch_count()==solved_before,
        "Collision-only primitive PCM unexpectedly fetched GPU contacts");
    const auto embedded=[](const native::Scene& scene) -> native::ContactPoint {
        for(const auto& manager:scene.contacts())
            for(const auto& point:manager.points)
                if(std::fabs(point.separation+0.75f)<0.0004f)
                    return point;
        throw std::runtime_error("Embedded sphere/box PCM witness missing");
    };
    const auto expected=embedded(cpu),actual=embedded(gpu);
    require(std::fabs(expected.position.z-actual.position.z)<0.0004f &&
            std::fabs(expected.normal.z-actual.normal.z)<0.0004f &&
            std::fabs(actual.position.z-0.5f)<0.0004f,
        "GPU embedded sphere did not preserve source PCM exit-face witness");
}

void verify_host_primitive_pcm_solver(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream source(dy_shader,std::ios::binary);
    require(bool(source),"Primitive GPU Dy shader bundle unavailable");
    const std::vector<char> data(
        std::istreambuf_iterator<char>{source},std::istreambuf_iterator<char>{});
    compute::ShaderBinary dy{};
    dy.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    dy.bytes.assign(data.begin(),data.end());

    native::Scene cpu,gpu;
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),128u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),std::move(dy));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto make=[](native::Scene& scene) {
        scene.add_static_floor(0.f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.f,0.48f,0.f},1.f);
        const auto sphere=scene.add_dynamic_sphere(
            0.4f,{6.f,0.38f,0.f},1.f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        scene.set_dynamics(box,dynamics);
        scene.set_dynamics(sphere,dynamics);
        return std::array<BodyId,2>{box,sphere};
    };
    const auto actual=make(gpu);
    const auto expected=make(cpu);
    constexpr float dt=1.f/120.f;
    for(unsigned frame=0u;frame<5u;++frame) {
        if(frame==2u || frame==3u) {
            const Vec3 p=frame==2u
                ?Vec3{0.f,4.f,0.f}:Vec3{0.f,0.48f,0.f};
            cpu.set_position(expected[0],p);
            gpu.set_position(actual[0],p);
        }
        const auto before_fetch=producer->stream_submission_stats();
        cpu.step(dt);
        gpu.step(dt);
        const auto& completed=producer->stream_submission_stats();
        const auto indirect=
            producer->dy_launch_stats().device_island_indirect_dispatches;
        const auto fetch_batches=
            completed.fetch_batches-before_fetch.fetch_batches;
        const auto fetch_spans=
            completed.fetch_spans-before_fetch.fetch_spans;
        const auto fetch_bytes=
            completed.fetch_bytes-before_fetch.fetch_bytes;
        const auto numerical_bytes=producer->dy_distance_transfer_bytes().second;
        require(fetch_bytes>=numerical_bytes,
            "GPU numerical readback exceeds the recorded frame transfer");
        std::cout<<"FYZZXYZZY_METRIC primitive_pcm frame="<<frame
                 <<" readback_batches="<<fetch_batches
                 <<" readback_spans="<<fetch_spans
                 <<" total_readback_bytes="<<fetch_bytes
                 <<" dy_readback_bytes="<<numerical_bytes
                 <<" indirect_dispatches="<<indirect<<'\n';
        require(producer->pcm_contact_readback_bytes()==0u &&
                producer->pcm_host_contact_upload_bytes()==
                    2u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
                producer->dy_gpu_integrated_body_count()==2u,
            "Primitive CPU Sc/GPU Dy path duplicated numerical PCM work");
        require(cpu.contacts().size()==gpu.contacts().size(),
            "Deferred primitive PCM changed the Sc touch graph");
        if(frame==2u)
            require(cpu.contacts().size()==1u,
                "Lifted source plane-box retained its physical touch");
        else
            require(cpu.contacts().size()==2u,
                "Restored source plane-box lost its physical touch");
        for(std::size_t i=0u;i<cpu.contacts().size();++i) {
            const auto& a=cpu.contacts()[i],&b=gpu.contacts()[i];
            require(a.pair==b.pair && a.count==b.count,
                "Deferred primitive Gu manager identity changed");
            for(std::size_t j=0u;j<a.count;++j) {
                const auto& x=a.points[j],&y=b.points[j];
                require(std::fabs(x.position.x-y.position.x)<0.003f &&
                        std::fabs(x.position.y-y.position.y)<0.003f &&
                        std::fabs(x.position.z-y.position.z)<0.003f &&
                        std::fabs(x.separation-y.separation)<0.003f,
                    "Deferred primitive Gu witnesses differ");
            }
        }
    }

    // Switch CPU primitive -> mixed GPU Gu -> CPU primitive. Sc must
    // regenerate the correct side's persistent EdgeIndex witness cache.
    const auto add_general=[](native::Scene& scene) {
        scene.add_static_box({0.5f,0.5f,0.5f},{50.f,5.f,0.f});
        const auto moving=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{50.997f,5.f,0.f},1.f);
        RigidBodyDynamics no_gravity{};
        no_gravity.gravity_enabled=false;
        scene.set_dynamics(moving,no_gravity);
        return moving;
    };
    const auto cpu_general=add_general(cpu);
    const auto gpu_general=add_general(gpu);
    // The second CPU primitive run lies AFTER the general box pair in
    // Sc source order. Both host islands must retain their device slots.
    const auto add_remote_sphere=[](native::Scene& scene) {
        scene.add_static_sphere(0.5f,{100.f,5.f,0.f});
        const auto moving=scene.add_dynamic_sphere(
            0.5f,{100.7f,5.f,0.f},1.f);
        RigidBodyDynamics no_gravity{};
        no_gravity.gravity_enabled=false;
        scene.set_dynamics(moving,no_gravity);
    };
    add_remote_sphere(cpu);
    add_remote_sphere(gpu);
    cpu.step(dt);
    gpu.step(dt);
    require(producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            producer->pcm_host_contact_upload_bytes()==
                5u*sizeof(physx_gpu::SourcePcmGpuContactStreamNative) &&
            cpu.contacts().size()==gpu.contacts().size(),
        "Mixed Sc PCM changed GPU witness and CPU Gu contact publication");
    native::GpuPcmFrictionAnchorFixture::verify(cpu,gpu);
    cpu.remove(cpu_general);
    gpu.remove(gpu_general);
    cpu.step(dt);
    gpu.step(dt);
    require(producer->pcm_contact_readback_bytes()==0u &&
            producer->pcm_host_contact_upload_bytes()>0u &&
            cpu.contacts().size()==gpu.contacts().size(),
        "Primitive frame failed to resume Sc CPU PCM ownership");
    for(std::size_t i=0u;i<cpu.contacts().size();++i) {
        const auto& a=cpu.contacts()[i],&b=gpu.contacts()[i];
        require(a.pair==b.pair && a.count==b.count,
            "PCM producer transition changed Sc contact identity");
        for(std::size_t j=0u;j<a.count;++j)
            require(std::fabs(a.points[j].separation-
                              b.points[j].separation)<0.003f,
                "PCM producer transition reused stale Gu witnesses");
    }
}

void verify_compound_actor_gpu_solver(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream stream(dy_shader,std::ios::binary);
    require(bool(stream),"Compound GPU Dy shader bundle missing");
    std::vector<char> bytes(std::istreambuf_iterator<char>{stream},std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    auto gpu=create_native_gpu_world(backend,load_native_gpu_sap_shaders(backend,sap),
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader),
        {.execution=ExecutionMode::NativePortableGpu,.contact_notifications=true});
    World cpu({.execution=ExecutionMode::NativeCpu,.contact_notifications=true});
    const auto verify_stage=[&](const char* name,auto&& verify) {
        try { verify(); }
        catch(...) {
            std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case=compound_solver"
                     <<" rejected_stage="<<name<<'\n';
            throw;
        }
    };
    verify_stage("compound_mass_and_motion",[&]{test::compound_mass_and_motion(gpu,&cpu);});
    verify_stage("compound_contact_lifetimes",[&]{test::compound_contact_lifetimes(gpu,&cpu);});
    verify_stage("compound_default_contact_response",[&]{test::compound_default_contact_response(gpu,&cpu);});
    verify_stage("compound_joint_response",[&]{test::compound_joint_response(gpu,&cpu);});
    verify_stage("fixed_joint_response",[&]{test::fixed_joint_response(gpu,&cpu);});
    verify_stage("spherical_joint_response",[&]{test::spherical_joint_response(gpu,&cpu);});
    verify_stage("revolute_joint_response",[&]{test::revolute_joint_response(gpu,&cpu);});
    verify_stage("prismatic_joint_response",[&]{test::prismatic_joint_response(gpu,&cpu);});
    verify_stage("compound_plane_box_device_pcm",[&]{test::compound_plane_box_device_pcm(gpu,&cpu);});
    verify_stage("compound_plane_sphere_device_pcm",[&]{test::compound_plane_sphere_device_pcm(gpu,&cpu);});
    // Native GPU and CPU World APIs query the same authored attachment
    // geometry, including shapes excluded from simulation collision.
    const auto gpu_body=gpu.create_static_body({{250.f,0.f,0.f},{}});
    const auto cpu_body=cpu.create_static_body({{250.f,0.f,0.f},{}});
    const auto gpu_shape=gpu.attach_sphere_shape(gpu_body,0.5f);
    const auto cpu_shape=cpu.attach_sphere_shape(cpu_body,0.5f);
    gpu.set_shape_simulation_enabled(gpu_shape,false);
    cpu.set_shape_simulation_enabled(cpu_shape,false);
    const auto gpu_hit=gpu.raycast({247.f,0.f,0.f},{1.f,0.f,0.f},10.f);
    const auto cpu_hit=cpu.raycast({247.f,0.f,0.f},{1.f,0.f,0.f},10.f);
    require(gpu_hit && cpu_hit && gpu_hit->shape==gpu_shape &&
            cpu_hit->shape==cpu_shape &&
            std::abs(gpu_hit->distance-cpu_hit->distance)<1.e-5f &&
            std::abs(gpu_hit->normal.x+1.f)<1.e-5f,
        "Native GPU World did not expose authored scene query geometry");
    gpu.remove_body(gpu_body);
    cpu.remove_body(cpu_body);
}

void verify_gpu_free_integration(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled source Dy shader unavailable");
    const std::vector<char> binary{
        std::istreambuf_iterator<char>{input},std::istreambuf_iterator<char>{}};
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());

    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),1024u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));

    const auto assemble=[](native::Scene& scene,ShapeId& compound_shape) {
        std::vector<BodyId> bodies;
        scene.add_static_box({1.f,0.25f,1.f},{0.f,-0.25f,0.f});
        bodies.push_back(scene.add_dynamic_box(
            {0.25f,0.25f,0.25f},{0.f,0.8f,0.f},1.f));
        scene.set_velocity(bodies.front(),{0.f,-2.5f,0.f});
        RigidBodyDynamics free_motion{};
        free_motion.gravity_enabled=false;
        free_motion.angular_damping=0.0f;
        free_motion.lock_flags=LockLinearZ|LockAngularX;
        for(unsigned i=1;i<=260u;++i) {
            const float x=3.f*static_cast<float>(i);
            const auto body=scene.add_dynamic_box(
                {0.25f,0.4f,0.6f},{x,3.f,0.f},1.f);
            bodies.push_back(body);
            RigidBodyDynamics dynamics=free_motion;
            if(i==3u)dynamics.gyroscopic_forces_enabled=true;
            if(i==14u)dynamics.retain_accelerations=true;
            scene.set_dynamics(body,dynamics);
            scene.set_velocity(body,{0.2f,0.3f,1.f});
            scene.set_angular_velocity(body,{1.f,0.2f,-0.6f});
        }
        const auto compound=scene.create_body({{0.f,3.f,4.f},{}},true);
        compound_shape=scene.attach_box(
            compound,{0.2f,0.3f,0.4f},{{-0.5f,0.f,0.f},{}});
        scene.attach_sphere(compound,0.4f,{{0.7f,0.4f,0.f},{}});
        scene.update_mass_properties(compound,1.f);
        RigidBodyDynamics damped_motion=free_motion;
        damped_motion.angular_damping=0.05f;
        scene.set_dynamics(compound,damped_motion);
        scene.set_angular_velocity(compound,{0.f,0.5f,0.9f});
        bodies.push_back(compound);
        const auto shapeless=scene.create_body({{0.f,6.f,0.f},{}},true);
        scene.set_dynamics(shapeless,damped_motion);
        scene.set_velocity(shapeless,{0.4f,-0.3f,0.f});
        bodies.push_back(shapeless);
        return bodies;
    };
    ShapeId expected_shape{},actual_shape{};
    const auto expected=assemble(cpu,expected_shape);
    const auto actual=assemble(gpu,actual_shape);
    bool touched=false,released=false;
    for(unsigned frame=0;frame<12u;++frame) {
        if(frame==3u) {
            RigidBodyDynamics clamped{};
            clamped.gravity_enabled=false;
            clamped.angular_damping=0.0f;
            clamped.lock_flags=LockLinearZ|LockAngularX;
            clamped.max_linear_velocity=0.4f;
            clamped.max_angular_velocity=0.5f;
            cpu.set_dynamics(expected[15],clamped);
            gpu.set_dynamics(actual[15],clamped);
        }
        if(frame==5u) {
            const Vec3 impulse{0.4f,-0.15f,0.f};
            const Vec3 acceleration{0.f,-1.0f,0.f};
            const Vec3 torque{0.f,0.15f,0.05f};
            for(auto* scene:{&cpu,&gpu}) {
                const auto& ids=scene==&cpu?expected:actual;
                scene->add_spatial_velocity_mod(ids[10],impulse,ForceMode::Impulse,false);
                scene->add_spatial_velocity_mod(ids[11],acceleration,ForceMode::Acceleration,false);
                scene->add_spatial_velocity_mod(ids[12],torque,ForceMode::VelocityChange,true);
                scene->add_spatial_velocity_mod(ids[13],torque,ForceMode::Force,true);
                scene->add_spatial_velocity_mod(ids[14],acceleration,ForceMode::Acceleration,false);
            }
        }
        if(frame==7u) {
            // An authored change to a steady free body must overwrite the
            // previous GPU velocity without changing its stable actor ID.
            cpu.set_velocity(expected[2],{1.f,0.7f,1.f});
            gpu.set_velocity(actual[2],{1.f,0.7f,1.f});
        }
        if(frame==4u) {
            // Recenter the compound and change its principal mass frame.
            // Sc's authored edit must replace exactly this GPU actor's
            // resident local transform before contact preparation.
            const Transform edited{{-0.35f,0.1f,0.05f},{}};
            cpu.set_shape_local_pose(expected_shape,edited);
            gpu.set_shape_local_pose(actual_shape,edited);
            cpu.update_mass_properties(expected[expected.size()-2u],1.f);
            gpu.update_mass_properties(actual[actual.size()-2u],1.f);
        }
        if(touched && !released) {
            cpu.set_position(expected.front(),{0.f,2.f,0.f});
            gpu.set_position(actual.front(),{0.f,2.f,0.f});
            cpu.set_velocity(expected.front(),{0.f,1.f,0.f});
            gpu.set_velocity(actual.front(),{0.f,1.f,0.f});
            released=true;
        }
        cpu.step(1.f/30.f);
        gpu.step(1.f/30.f);
        require(producer->dy_distance_dispatch_count()==frame+1u &&
                producer->dy_gpu_integrated_body_count()==actual.size() &&
                producer->dy_upload_span_count()==
                    1u+producer->dy_launch_stats().device_island_indirect_dispatches,
                "GPU Dy did not integrate every active body in one upload");
        const auto resident=producer->dy_resident_pose_count();
        const auto metadata=producer->dy_launch_stats();
        require(metadata.device_preintegrated_bodies==actual.size() &&
                metadata.device_sleep_bodies==actual.size() &&
                metadata.sleep_readback_bytes==
                    actual.size()*sizeof(native::GpuDySleepState) &&
                metadata.free_velocity_readback_bytes<=
                    actual.size()*sizeof(physx_gpu::SourceDyPreparedVelocityNative) &&
                (frame<5u?metadata.force_mod_upload_bytes==0u
                         :metadata.force_mod_upload_bytes>0u),
            "GPU Dy did not prepare rigid velocities and force streams");
        constexpr auto velocity_bytes=
            sizeof(physx_gpu::SourceDyPreparedVelocityNative);
        require(metadata.prepared_velocity_updated_bodies+
                    metadata.resident_velocity_reuses==actual.size() &&
                metadata.prepared_velocity_upload_bytes>=
                    metadata.prepared_velocity_updated_bodies*sizeof(float) &&
                metadata.prepared_velocity_upload_bytes<=
                    metadata.prepared_velocity_updated_bodies*velocity_bytes &&
                (frame==0u
                    ? metadata.prepared_velocity_upload_bytes==
                          actual.size()*velocity_bytes &&
                      metadata.resident_velocity_reuses==0u
                    : metadata.resident_velocity_reuses>=
                          (frame==7u?259u:260u)) &&
                (frame==0u || metadata.prepared_velocity_updated_bodies==0u ||
                 metadata.prepared_velocity_upload_bytes<
                     metadata.prepared_velocity_updated_bodies*velocity_bytes) &&
                (frame!=7u ||
                 metadata.prepared_velocity_updated_bodies>=1u),
            "Native Dy did not retain and compact exact GPU velocities");
        const auto resident_record_bytes=
            sizeof(physx_gpu::SourceDyActorLocalNative);
        const auto mass_frame_edits=(frame==3u || frame==4u)?1u:0u;
        const auto transfer_error=
            std::string{"Source Dy actor-state refresh exceeded authored changes: frame="}+
            std::to_string(frame)+
            " bytes="+std::to_string(metadata.actor_local_upload_bytes)+
            " reuses="+std::to_string(metadata.reused_actor_local_poses)+
            " spans="+std::to_string(metadata.actor_local_upload_spans);
        require((frame==0u
                    ? metadata.actor_local_upload_bytes==
                          actual.size()*resident_record_bytes &&
                      metadata.actor_local_upload_spans==1u &&
                      metadata.reused_actor_local_poses==0u
                    : metadata.reused_actor_local_poses==
                          actual.size()-mass_frame_edits &&
                      metadata.actor_local_upload_bytes%resident_record_bytes==0u &&
                      metadata.actor_local_upload_bytes<=
                          6u*resident_record_bytes &&
                      metadata.actor_local_upload_spans<=3u &&
                      (mass_frame_edits==0u ||
                       metadata.actor_local_upload_bytes>=resident_record_bytes)),
            transfer_error.c_str());
        require(resident<=actual.size() &&
                producer->dy_host_pose_upload_bytes()==0u &&
                (frame==0u?resident==0u:resident>=260u),
                "GPU Dy did not retain authorized Sc poses across frames");
        if(frame==1u)
            require(resident==actual.size(),
                "Shape-less native rigid actor did not reuse its GPU world pose");
        require(cpu.contacts().size()==gpu.contacts().size(),
                "Device-integrated free body changed contact admission");
        const bool contacts=producer->dy_pcm_device_contact_rows()!=0u;
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_world_inertia_bodies<=actual.size() &&
                (!contacts || schedule.device_world_inertia_bodies>0u),
            "Contact-only native Dy skipped resident world-inertia preparation");
        if(contacts) {
            require(!released &&
                    schedule.dispatches==2u+
                        schedule.device_island_indirect_dispatches &&
                    schedule.device_initialization_dispatches==0u &&
                    schedule.device_initialized_bodies>0u &&
                    schedule.device_world_motion_bodies>0u &&
                    schedule.device_world_motion_bodies<=
                        schedule.device_initialized_bodies &&
                    schedule.device_body_upload_bytes==0u,
                    "Mixed Dy skipped device initialization, PGS or free-body integration");
            touched=true;
        } else {
            require(schedule.dispatches==1u &&
                    schedule.device_initialized_bodies==0u &&
                    schedule.device_world_motion_bodies==0u &&
                    producer->dy_distance_transfer_bytes().second==
                        actual.size()*
                            (sizeof(native::GpuDyIntegratedPose)+
                             sizeof(physx_gpu::SourceDyPreparedVelocityNative)+
                             sizeof(native::GpuDySleepState)),
                    "Free-only Dy staged solver rows or dispatched PGS");
        }
        if(frame>0u)
            require(producer->last_frame().device_cached_box_count>=262u,
                    "Free body or compound shape lost resident bounds");
        for(std::size_t i=0;i<actual.size();++i) {
            test::compound_compare_state(gpu.body_state(actual[i]),
                cpu.body_state(expected[i]),0.002f,
                "GPU Dy free or transitioning body");
            require(std::fabs(cpu.body_wake_counter(expected[i])-
                              gpu.body_wake_counter(actual[i]))<0.0005f &&
                    cpu.body_ready_for_sleep(expected[i])==
                        gpu.body_ready_for_sleep(actual[i]),
                "GPU Dy sleep energy or wake-counter arbitration diverged");
        }
        // Query a moving, rotating body after Dy has published its pose.
        // The ray crosses the body's center, independent of its rotation.
        const auto cpu_position=cpu.body_state(expected[1]).position;
        const auto gpu_position=gpu.body_state(actual[1]).position;
        const auto cpu_hit=cpu.raycast(
            {cpu_position.x,cpu_position.y+2.f,cpu_position.z},
            {0.f,-1.f,0.f},3.f);
        const auto gpu_hit=gpu.raycast(
            {gpu_position.x,gpu_position.y+2.f,gpu_position.z},
            {0.f,-1.f,0.f},3.f);
        require(cpu_hit && gpu_hit &&
                cpu_hit->body==expected[1] && gpu_hit->body==actual[1] &&
                std::abs(cpu_hit->distance-gpu_hit->distance)<0.005f,
            "Device-integrated rigid pose did not publish its scene query");
    }
    require(touched && released,"Free body did not enter and leave GPU PCM response");
    // A retired Sc BodyId must never supply the metadata for a newly
    // admitted actor, even when the source device pool grows this frame.
    cpu.remove(expected.back());
    gpu.remove(actual.back());
    const auto replacement_cpu=cpu.add_dynamic_box(
        {0.2f,0.3f,0.4f},{1250.f,5.f,0.f},1.f);
    const auto replacement_gpu=gpu.add_dynamic_box(
        {0.2f,0.3f,0.4f},{1250.f,5.f,0.f},1.f);
    RigidBodyDynamics replacement_dynamics{};
    replacement_dynamics.gravity_enabled=false;
    cpu.set_dynamics(replacement_cpu,replacement_dynamics);
    gpu.set_dynamics(replacement_gpu,replacement_dynamics);
    cpu.step(1.f/30.f);
    gpu.step(1.f/30.f);
    const auto after_replacement=producer->dy_launch_stats();
    require(after_replacement.prepared_velocity_updated_bodies+
                after_replacement.resident_velocity_reuses==actual.size() &&
            after_replacement.prepared_velocity_upload_bytes<=
                after_replacement.prepared_velocity_updated_bodies*
                    sizeof(physx_gpu::SourceDyPreparedVelocityNative),
        "Retired native body contaminated resident velocity publication");
    require(after_replacement.actor_local_upload_bytes==
                sizeof(physx_gpu::SourceDyActorLocalNative) &&
            after_replacement.actor_local_upload_spans==1u &&
            after_replacement.reused_actor_local_poses==actual.size()-1u,
        "GPU Dy retired actor metadata leaked into newly admitted rigid");
    test::compound_compare_state(
        gpu.body_state(replacement_gpu),cpu.body_state(replacement_cpu),
        0.002f,"Resident GPU Dy actor replacement");

    // The source wake counter must reach IslandSim readiness on device,
    // then accept an authored wake without replaying old accumulators.
    const auto idle_cpu=cpu.add_dynamic_box(
        {0.2f,0.2f,0.2f},{2200.f,5.f,0.f},1.f);
    const auto idle_gpu=gpu.add_dynamic_box(
        {0.2f,0.2f,0.2f},{2200.f,5.f,0.f},1.f);
    RigidBodyDynamics idle_dynamics{};
    idle_dynamics.gravity_enabled=false;
    idle_dynamics.linear_damping=0.0f;
    idle_dynamics.angular_damping=0.0f;
    cpu.set_dynamics(idle_cpu,idle_dynamics);
    gpu.set_dynamics(idle_gpu,idle_dynamics);
    bool reached_sleep=false;
    for(unsigned frame=0u;frame<16u;++frame) {
        cpu.step(1.f/30.f);
        gpu.step(1.f/30.f);
        require(std::fabs(cpu.body_wake_counter(idle_cpu)-
                          gpu.body_wake_counter(idle_gpu))<0.0005f &&
                cpu.body_ready_for_sleep(idle_cpu)==
                    gpu.body_ready_for_sleep(idle_gpu) &&
                cpu.body_is_sleeping(idle_cpu)==
                    gpu.body_is_sleeping(idle_gpu),
            "GPU Dy idle wake-counter or island sleep transition diverged");
        reached_sleep|=cpu.body_is_sleeping(idle_cpu);
    }
    require(reached_sleep,
        "Source idle rigid did not reach IslandSim deactivation");
    cpu.set_velocity(idle_cpu,{0.1f,0.f,0.f});
    gpu.set_velocity(idle_gpu,{0.1f,0.f,0.f});
    cpu.step(1.f/30.f);
    gpu.step(1.f/30.f);
    require(!gpu.body_is_sleeping(idle_gpu) &&
            std::fabs(cpu.body_wake_counter(idle_cpu)-
                      gpu.body_wake_counter(idle_gpu))<0.0005f,
        "GPU Dy did not restore resident sleep state on authored wake");
    test::compound_compare_state(
        gpu.body_state(idle_gpu),cpu.body_state(idle_cpu),0.002f,
        "Resident GPU Dy sleep-to-wake response");
}

void verify_gpu_wide_island_solver(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader,unsigned connected_bodies,bool diagnose=false,
    bool disable_friction=false,bool disable_strong_friction=false) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled source Dy shader unavailable");
    const std::vector<char> binary{
        std::istreambuf_iterator<char>{input},std::istreambuf_iterator<char>{}};
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv : compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());
    std::optional<compute::ShaderBinary> diagnostic_shader;
    if(diagnose)diagnostic_shader=shader;

    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        connected_bodies>768u ? 32768u : 4096u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    std::unique_ptr<native::Scene> pcm_cpu_dy;
    if(diagnose) {
        pcm_cpu_dy=std::make_unique<native::Scene>();
        pcm_cpu_dy->set_enhanced_determinism(true);
        pcm_cpu_dy->set_gpu_broadphase(
            std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
                backend,load_native_gpu_sap_shaders(backend,sap),
                connected_bodies>768u ? 32768u : 4096u,128u,
                std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
                std::move(*diagnostic_shader)));
        pcm_cpu_dy->set_gpu_dy_enabled(false);
    }

    struct Assembly {
        std::vector<BodyId> bodies;
        ConstraintId split_joint=0;
    };
    const auto assemble=[connected_bodies,disable_friction,disable_strong_friction](
        native::Scene& scene) {
        Assembly out;
        const auto chain=[&](unsigned count,float z,std::uint8_t pos,std::uint8_t vel) {
            RigidBodyDynamics dynamics;
            dynamics.position_iterations=pos;
            dynamics.velocity_iterations=vel;
            BodyId previous=0;
            for(unsigned i=0;i<count;++i) {
                const float x=1.25f*static_cast<float>(i);
                scene.add_static_box({0.55f,0.25f,0.55f},{x,-0.25f,z});
                const auto body=scene.add_dynamic_box(
                    {0.4f,0.5f,0.4f},{x,0.48f,z},1.0f);
                out.bodies.push_back(body);
                if(disable_friction || disable_strong_friction) {
                    Material material{};
                    material.disable_friction=disable_friction;
                    material.disable_strong_friction=disable_strong_friction;
                    scene.set_material(body,material);
                }
                scene.set_dynamics(body,dynamics);
                scene.set_velocity(body,{0.08f,0.0f,i%2u ? -0.06f : 0.06f});
                if(i) {
                    const auto joint=i%64u==24u
                        ? scene.create_prismatic_constraint(previous,body,
                            {{0.625f,0.12f,0.0f},{}},{{-0.625f,0.12f,0.0f},{}})
                        : i%32u==12u
                        ? scene.create_revolute_constraint(previous,body,
                            {{0.625f,0.12f,0.0f},{}},{{-0.625f,0.12f,0.0f},{}})
                        : i%16u==4u
                        ? scene.create_spherical_constraint(previous,body,
                            {{0.625f,0.12f,0.0f},{}},{{-0.625f,0.12f,0.0f},{}})
                        : i%8u==0u
                            ? scene.create_fixed_constraint(previous,body,
                                {{0.625f,0.12f,0.0f},{}},{{-0.625f,0.12f,0.0f},{}})
                        : scene.create_anchored_distance_constraint(
                            previous,body,{0.1f,0.12f,0.0f},{-0.1f,-0.08f,0.0f},1.04f);
                    if(count==640u && i==count/2u)out.split_joint=joint;
                }
                previous=body;
            }
        };
        chain(connected_bodies,0.0f,5u,2u);
        chain(640u,4.0f,3u,2u);
        chain(2u,8.0f,4u,1u);
        return out;
    };
    const auto expected=assemble(cpu);
    const auto actual=assemble(gpu);
    std::optional<Assembly> diagnostic;
    if(pcm_cpu_dy)diagnostic=assemble(*pcm_cpu_dy);
    for(unsigned frame=0;frame<4u;++frame) {
        if(frame==2u) {
            cpu.retire_constraint(expected.split_joint);
            gpu.retire_constraint(actual.split_joint);
            if(pcm_cpu_dy)
                pcm_cpu_dy->retire_constraint(diagnostic->split_joint);
        }
        constexpr float dt=1.0f/120.0f;
        cpu.step(dt);
        if(pcm_cpu_dy)pcm_cpu_dy->step(dt);
        gpu.step(dt);
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_colored_tasks>=1u &&
                schedule.device_colored_descriptors>=connected_bodies &&
                schedule.host_degree_scanned_descriptors>=connected_bodies-2u &&
                schedule.device_joint_input_rows>=connected_bodies/2u &&
                schedule.device_joint_geometry_count>=connected_bodies/2u,
            "Wide Dy task lost device joint preparation or coloring");
        // Once the assembly moves, individual supports can lose touch.
        // Joint connectivity must still supply a wide partition.
        const auto width=frame==0u?connected_bodies-2u:connected_bodies/2u;
        const auto partition_groups=(width+127u)/128u;
        require(gpu.largest_accurate_island_size()==connected_bodies &&
                schedule.wide_tasks==(frame<2u?2u:1u) &&
                schedule.max_partition_groups>=partition_groups &&
                schedule.max_partition_groups<=
                    (connected_bodies+127u)/128u &&
                schedule.tile_bytes<=schedule.device_colored_descriptors*
                    sizeof(native::GpuDyDistanceDescriptor) &&
                schedule.dispatches>7u &&
                schedule.device_initialization_dispatches==0u,
                "Connected Dy islands did not execute tiled conflict partitions");
        require(schedule.device_generated_bytes==
                    schedule.device_colored_descriptors*
                        (sizeof(native::GpuDyDistanceDescriptor)+
                         sizeof(std::uint32_t)) &&
                schedule.device_source_descriptors==
                    schedule.device_colored_descriptors &&
                schedule.source_descriptor_upload_bytes<=
                    schedule.device_colored_descriptors*
                        sizeof(std::uint32_t),
                "Source Sc did not materialize Dy descriptors on the device");
        require(schedule.device_upload_batches==1u &&
                producer->dy_upload_span_count()==1u &&
                producer->dy_gpu_integrated_body_count()==actual.bodies.size() &&
                producer->dy_pcm_device_contact_rows()>=actual.bodies.size() &&
                producer->dy_cpu_gu_geometry_bytes()==0u,
                "Wide island lost GPU contact preparation or body integration");
        if(pcm_cpu_dy) {
            const auto max_axis_error=[](Vec3 a,Vec3 b) {
                return std::max({std::abs(double(a.x)-b.x),
                                 std::abs(double(a.y)-b.y),
                                 std::abs(double(a.z)-b.z)});
            };
            const auto max_force_error=[](auto lhs,auto rhs) {
                if(lhs.size()!=rhs.size())return -1.0;
                double difference=0.0;
                for(std::size_t k=0;k<lhs.size();++k)
                    difference=std::max(difference,
                        std::abs(double(lhs[k])-double(rhs[k])));
                return difference;
            };
            struct FrictionDifference {
                double impulse=0.0,anchor_position=0.0;
                std::size_t count_mismatch=0u,broken_mismatch=0u;
            };
            const auto friction_difference=[&](auto lhs,auto rhs) {
                FrictionDifference result{};
                if(lhs.size()!=rhs.size()) {
                    result.count_mismatch=std::max(lhs.size(),rhs.size());
                    return result;
                }
                for(std::size_t k=0u;k<lhs.size();++k) {
                    const auto& a=lhs[k];
                    const auto& b=rhs[k];
                    if(a.pair!=b.pair || a.anchor_count!=b.anchor_count) {
                        ++result.count_mismatch;
                        continue;
                    }
                    if(a.broken!=b.broken)++result.broken_mismatch;
                    for(std::size_t n=0u;n<a.anchor_count;++n) {
                        result.anchor_position=std::max(
                            result.anchor_position,
                            max_axis_error(a.anchor_positions[n],
                                           b.anchor_positions[n]));
                        const auto& x=a.anchor_impulses[n];
                        const auto& y=b.anchor_impulses[n];
                        result.impulse=std::max({
                            result.impulse,std::abs(x.x-y.x),
                            std::abs(x.y-y.y),std::abs(x.z-y.z)});
                    }
                }
                return result;
            };
            struct ContactDifference {
                double position=0.0,normal=0.0,separation=0.0;
                std::size_t topology_mismatches=0u;
            };
            const auto contact_difference=[&](auto lhs,auto rhs) {
                ContactDifference result{};
                if(lhs.size()!=rhs.size()) {
                    result.topology_mismatches=std::max(lhs.size(),rhs.size());
                    return result;
                }
                for(std::size_t k=0u;k<lhs.size();++k) {
                    const auto& a=lhs[k];
                    const auto& b=rhs[k];
                    if(a.pair!=b.pair || a.count!=b.count) {
                        ++result.topology_mismatches;
                        continue;
                    }
                    for(std::size_t j=0u;j<a.count;++j) {
                        result.position=std::max(result.position,
                            max_axis_error(a.points[j].position,b.points[j].position));
                        result.normal=std::max(result.normal,
                            max_axis_error(a.points[j].normal,b.points[j].normal));
                        result.separation=std::max(result.separation,
                            std::abs(double(a.points[j].separation)-
                                     double(b.points[j].separation)));
                    }
                }
                return result;
            };
            const auto contact_geometry=contact_difference(
                cpu.contacts(),pcm_cpu_dy->contacts());
            const auto solver_geometry=contact_difference(
                pcm_cpu_dy->contacts(),gpu.contacts());
            const auto contact_friction=friction_difference(
                cpu.solved_friction_patches(),
                pcm_cpu_dy->solved_friction_patches());
            const auto solver_friction=friction_difference(
                pcm_cpu_dy->solved_friction_patches(),
                gpu.solved_friction_patches());
            double contact_linear=0.0,contact_angular=0.0;
            double dy_linear=0.0,dy_angular=0.0;
            double contact_position=0.0,dy_position=0.0;
            for(std::size_t i=0;i<actual.bodies.size();++i) {
                const auto reference=cpu.body_state(expected.bodies[i]);
                const auto contact=pcm_cpu_dy->body_state(diagnostic->bodies[i]);
                const auto solved=gpu.body_state(actual.bodies[i]);
                contact_position=std::max(contact_position,
                    max_axis_error(reference.position,contact.position));
                contact_linear=std::max(contact_linear,
                    max_axis_error(reference.velocity,contact.velocity));
                contact_angular=std::max(contact_angular,
                    max_axis_error(reference.angular_velocity,
                                   contact.angular_velocity));
                dy_position=std::max(dy_position,
                    max_axis_error(contact.position,solved.position));
                dy_linear=std::max(dy_linear,
                    max_axis_error(contact.velocity,solved.velocity));
                dy_angular=std::max(dy_angular,
                    max_axis_error(contact.angular_velocity,
                                   solved.angular_velocity));
            }
            std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm"
                     <<" case=wide_islands_diagnose frame="<<frame
                     <<" cpu_vs_pcm_position="<<contact_position
                     <<" cpu_vs_pcm_linear="<<contact_linear
                     <<" cpu_vs_pcm_angular="<<contact_angular
                     <<" pcm_vs_gpu_position="<<dy_position
                     <<" pcm_vs_gpu_linear="<<dy_linear
                     <<" pcm_vs_gpu_angular="<<dy_angular
                     <<" cpu_vs_pcm_normal_force="<<max_force_error(
                         cpu.solved_normal_impulses(),
                         pcm_cpu_dy->solved_normal_impulses())
                     <<" pcm_vs_gpu_normal_force="<<max_force_error(
                         pcm_cpu_dy->solved_normal_impulses(),
                         gpu.solved_normal_impulses())
                     <<" cpu_vs_pcm_contact_position="<<contact_geometry.position
                     <<" pcm_vs_gpu_contact_position="<<solver_geometry.position
                     <<" cpu_vs_pcm_contact_normal="<<contact_geometry.normal
                     <<" pcm_vs_gpu_contact_normal="<<solver_geometry.normal
                     <<" cpu_vs_pcm_contact_separation="<<contact_geometry.separation
                     <<" pcm_vs_gpu_contact_separation="<<solver_geometry.separation
                     <<" cpu_vs_pcm_contact_topology_mismatch="<<contact_geometry.topology_mismatches
                     <<" pcm_vs_gpu_contact_topology_mismatch="<<solver_geometry.topology_mismatches
                     <<" cpu_vs_pcm_friction_impulse="<<contact_friction.impulse
                     <<" pcm_vs_gpu_friction_impulse="<<solver_friction.impulse
                     <<" cpu_vs_pcm_anchor_position="<<contact_friction.anchor_position
                     <<" pcm_vs_gpu_anchor_position="<<solver_friction.anchor_position
                     <<" cpu_vs_pcm_friction_count_mismatch="<<contact_friction.count_mismatch
                     <<" pcm_vs_gpu_friction_count_mismatch="<<solver_friction.count_mismatch
                     <<" cpu_vs_pcm_friction_broken_mismatch="<<contact_friction.broken_mismatch
                     <<" pcm_vs_gpu_friction_broken_mismatch="<<solver_friction.broken_mismatch
                     <<" cpu_contacts="<<cpu.contacts().size()
                     <<" pcm_contacts="<<pcm_cpu_dy->contacts().size()
                     <<" gpu_contacts="<<gpu.contacts().size()<<'\n';
        }
        for(std::size_t i=0;i<actual.bodies.size();++i) {
            const auto actual_state=gpu.body_state(actual.bodies[i]);
            const auto expected_state=cpu.body_state(expected.bodies[i]);
            try {
                test::compound_compare_state(
                    actual_state,expected_state,0.002f,
                    "Tiled Dy mixed contact and joint island");
            } catch(const std::runtime_error&) {
                const auto actual_forces=gpu.solved_normal_impulses();
                const auto expected_forces=cpu.solved_normal_impulses();
                double max_force_difference=0.0;
                if(actual_forces.size()==expected_forces.size())
                    for(std::size_t k=0;k<actual_forces.size();++k)
                        max_force_difference=std::max(
                            max_force_difference,
                            std::abs(actual_forces[k]-expected_forces[k]));
                std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm case=wide_islands"
                         <<" phase=state_mismatch"
                         <<" frame="<<frame<<" body_index="<<i
                         <<" gpu_angular=("
                         <<actual_state.angular_velocity.x<<","
                         <<actual_state.angular_velocity.y<<","
                         <<actual_state.angular_velocity.z<<")"
                         <<" cpu_angular=("
                         <<expected_state.angular_velocity.x<<","
                         <<expected_state.angular_velocity.y<<","
                         <<expected_state.angular_velocity.z<<")"
                         <<" gpu_contacts="<<gpu.contacts().size()
                         <<" cpu_contacts="<<cpu.contacts().size()
                         <<" gpu_normal_rows="<<actual_forces.size()
                         <<" cpu_normal_rows="<<expected_forces.size()
                         <<" max_normal_force_error="
                         <<max_force_difference<<'\n';
                throw;
            }
        }
        if(frame==0u)
            std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm case=wide_islands"
                     <<" bodies="<<actual.bodies.size()
                     <<" wide_tasks="<<schedule.wide_tasks
                     <<" max_partition_groups="<<schedule.max_partition_groups
                     <<" dy_dispatches="<<schedule.dispatches
                     <<" tile_bytes="<<schedule.tile_bytes<<'\n';
    }
}

// A chain of zero-response contacts shares an accurate island with
// responding joint bodies, but its contact-only actors do not enter
// the Dy integration-writeback ordinal list. Every colored descriptor
// must still have a tile, including those actors' contact managers.
void verify_gpu_zero_response_wide_island(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"Compiled wide Dy shader unavailable");
    const std::vector<char> binary{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),8192u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));

    constexpr unsigned joint_bodies=640u,zero_contact_bodies=1024u;
    const auto assemble=[](native::Scene& scene) {
        std::vector<BodyId> ids;
        ids.reserve(joint_bodies+zero_contact_bodies);
        RigidBodyDynamics response{};
        response.gravity_enabled=false;
        response.position_iterations=4u;
        response.velocity_iterations=2u;
        BodyId previous=0u;
        for(unsigned i=0u;i<joint_bodies;++i) {
            const auto body=scene.add_dynamic_box(
                {0.35f,0.35f,0.35f},
                {1.4f*static_cast<float>(i),0.5f,0.f},1.f);
            ids.push_back(body);
            scene.set_dynamics(body,response);
            scene.set_velocity(body,{0.025f,0.f,0.f});
            if(i)scene.create_center_distance_constraint(
                previous,body,1.4f);
            previous=body;
        }
        RigidBodyDynamics zero=response;
        zero.max_contact_impulse=0.f;
        for(unsigned i=0u;i<zero_contact_bodies;++i) {
            const auto body=scene.add_dynamic_box(
                {0.41f,0.41f,0.41f},
                {0.75f*static_cast<float>(i),0.5f,
                 i==0u?0.7f:1.f},1.f);
            ids.push_back(body);
            scene.set_dynamics(body,zero);
            scene.set_velocity(body,{0.005f,0.f,0.f});
        }
        return ids;
    };
    const auto a=assemble(cpu),b=assemble(gpu);
    cpu.step(1.f/120.f);
    gpu.step(1.f/120.f);
    const auto schedule=producer->dy_launch_stats();
    const auto cpu_island=cpu.largest_accurate_island_size();
    const auto gpu_island=gpu.largest_accurate_island_size();
    std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm case=zero_response_wide"
             <<" bodies="<<b.size()
             <<" cpu_island="<<cpu_island
             <<" gpu_island="<<gpu_island
             <<" cpu_contacts="<<cpu.contacts().size()
             <<" gpu_contacts="<<gpu.contacts().size()
             <<" wide_tasks="<<schedule.wide_tasks
             <<" descriptors="<<schedule.device_colored_descriptors
             <<" max_partition_groups="<<schedule.max_partition_groups
             <<" dy_dispatches="<<schedule.dispatches
             <<" tile_bytes="<<schedule.tile_bytes
             <<" device_generated_bytes="<<schedule.device_generated_bytes<<'\n';
    require(cpu_island>=joint_bodies+zero_contact_bodies &&
            gpu_island>=joint_bodies+zero_contact_bodies,
        "Zero-response fixture did not form the intended accurate island");
    require(schedule.wide_tasks==1u &&
            schedule.device_colored_descriptors>=
                joint_bodies+zero_contact_bodies-2u &&
            schedule.device_generated_bytes==
                schedule.device_colored_descriptors*20u,
        "Zero-response fixture did not produce wide contact/joint descriptors");
    require(schedule.max_partition_groups>=
                (joint_bodies+zero_contact_bodies+127u)/128u,
        "Device Dy partition tiles omit zero-response contact endpoints");
    require(schedule.tile_bytes>0u &&
            schedule.tile_bytes<=schedule.device_colored_descriptors*
                sizeof(native::GpuDyDistanceDescriptor),
        "Device Dy partition tile metadata exceeds its source workload");
    require(cpu.contacts().size()==gpu.contacts().size(),
        "Device PCM changed zero-response contact publication");
    for(std::size_t i=0u;i<b.size();++i)
        test::compound_compare_state(gpu.body_state(b[i]),
            cpu.body_state(a[i]),0.003f,
            "Wide Dy zero-response contact and joint island");
}

// Hub and connected two-hub graphs exercise the source-order
// 32-color rollover. Mixed static contact and 1D rows reserve colors
// without writing one dynamic body from concurrent PGS lanes.
void verify_gpu_high_degree_coloring(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader,unsigned leaves=36u,
    bool dual_hub=false,unsigned frames_override=0u) {
    using namespace fyzz;
    std::ifstream input(dy_shader,std::ios::binary);
    require(bool(input),"High-degree GPU Dy shader unavailable");
    const std::vector<char> binary{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ? compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(binary.begin(),binary.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),512u,128u,
        std::nullopt,load_native_gpu_pcm_shaders(backend,pcm),
        std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    const auto assemble=[leaves,dual_hub](native::Scene& scene) {
        std::vector<BodyId> ids;
        ids.reserve(leaves+1u+(dual_hub?1u:0u));
        const auto fixed=scene.add_static_box(
            {0.4f,0.4f,0.4f},{-0.72f,3.f,0.f});
        const auto center=scene.add_dynamic_box(
            {0.4f,0.4f,0.4f},{0.f,3.f,0.f},1.f);
        ids.push_back(center);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.position_iterations=4u;
        dynamics.velocity_iterations=1u;
        scene.set_dynamics(center,dynamics);
        scene.set_velocity(center,{0.05f,0.f,0.07f});
        scene.create_center_distance_constraint(fixed,center,0.72f);
        BodyId second_center=0u;
        if(dual_hub) {
            second_center=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},{-3.f,3.f,0.f},1.f);
            ids.push_back(second_center);
            scene.set_dynamics(second_center,dynamics);
            scene.set_velocity(second_center,{-0.05f,0.f,0.07f});
            scene.create_center_distance_constraint(
                center,second_center,3.f);
        }
        for(unsigned i=0u;i<leaves;++i) {
            const bool second=dual_hub && i>=leaves/2u;
            const float x=second
                ?-6.f-1.5f*static_cast<float>(i-leaves/2u)
                :3.f+1.5f*static_cast<float>(i);
            const auto anchor=second?second_center:center;
            const float rest=second?(-3.f-x):x;
            const auto body=scene.add_dynamic_box(
                {0.3f,0.3f,0.3f},{x,3.f,0.f},1.f);
            ids.push_back(body);
            scene.set_dynamics(body,dynamics);
            scene.set_velocity(body,
                {-0.03f,0.f,i%2u ? 0.02f:-0.02f});
            scene.create_center_distance_constraint(anchor,body,rest);
        }
        return ids;
    };
    if(dual_hub || leaves>=1024u || frames_override!=0u)
        std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                 <<" spokes="<<leaves<<" phase=setup\n";
    const auto a=assemble(cpu);
    if(dual_hub || leaves>=1024u || frames_override!=0u)
        std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                 <<" spokes="<<leaves<<" phase=cpu_assembled\n";
    const auto b=assemble(gpu);
    if(dual_hub || leaves>=1024u || frames_override!=0u)
        std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                 <<" spokes="<<leaves<<" phase=gpu_assembled\n";
    const unsigned frames=frames_override?frames_override:
        leaves>=10000u?1u:leaves>=1024u?2u:leaves>=64u?1u:3u;
    for(unsigned frame=0u;frame<frames;++frame) {
        const auto tick=std::chrono::steady_clock::now();
        cpu.step(1.f/120.f);
        const auto cpu_done=std::chrono::steady_clock::now();
        if(dual_hub || leaves>=1024u || frames_override!=0u)
            std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                     <<" spokes="<<leaves
                     <<" frame="<<frame<<" phase=cpu"
                     <<" wall_ms="<<std::chrono::duration<double,std::milli>(
                          cpu_done-tick).count()<<'\n';
        gpu.step(1.f/120.f);
        const auto gpu_done=std::chrono::steady_clock::now();
        if(dual_hub || leaves>=1024u || frames_override!=0u)
            std::cerr<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                     <<" spokes="<<leaves
                     <<" frame="<<frame<<" phase=gpu"
                     <<" wall_ms="<<std::chrono::duration<double,std::milli>(
                          gpu_done-cpu_done).count()<<'\n';
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_colored_tasks>0u &&
                schedule.device_colored_descriptors>=leaves+1u &&
                schedule.wide_tasks==0u &&
                schedule.max_partition_groups==1u &&
                schedule.tile_bytes==0u &&
                schedule.dispatches==
                    (b.size()<=128u?1u:2u)+
                        schedule.device_island_indirect_dispatches &&
                schedule.device_initialization_dispatches==
                    (b.size()<=128u?0u:1u) &&
                producer->dy_gpu_integrated_body_count()==b.size() &&
                gpu.contacts().size()==cpu.contacts().size() &&
                !gpu.contacts().empty() &&
                (!dual_hub ||
                 (gpu.largest_accurate_island_size()>=b.size() &&
                  cpu.largest_accurate_island_size()>=a.size())),
            "High-degree Dy task emitted empty tiled partitions");
        for(std::size_t i=0u;i<a.size();++i)
            test::compound_compare_state(gpu.body_state(b[i]),
                cpu.body_state(a[i]),0.015f,
                "GPU Dy multiwave dynamic/static constraint coloring");
        if(frame==0u && leaves>=1024u)
            std::cout<<"FYZZXYZZY_METRIC suite=sc_pcm case="<<(dual_hub?"dual_hub":"star_island")
                     <<" spokes="<<leaves
                     <<" dy_dispatches="<<schedule.dispatches
                     <<" wide_tasks="<<schedule.wide_tasks
                     <<" tile_bytes="<<schedule.tile_bytes<<'\n';
    }
}



void qualify_hardware_scene(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_path,std::uint32_t body_count,bool gpu_first) {
    using namespace fyzz;
    const auto compiled=[backend](const char* path) {
        std::ifstream file(path,std::ios::binary);
        require(bool(file),"Hardware scene GPU Dy shader is missing");
        std::vector<char> bytes(std::istreambuf_iterator<char>{file},
                                std::istreambuf_iterator<char>{});
        require(!bytes.empty(),"Hardware scene GPU Dy shader is empty");
        compute::ShaderBinary binary{};
        binary.format=backend==compute::Backend::Vulkan
            ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
        binary.bytes.assign(bytes.begin(),bytes.end());
        return binary;
    };
    const auto create=[&] {
        return create_native_gpu_world(
            backend,NativeGpuPipelineShaders{
                .sap=load_native_gpu_sap_shaders(backend,sap),
                .pcm=load_native_gpu_pcm_shaders(backend,pcm),
                .dy=compiled(dy_path)},
            {.execution=ExecutionMode::NativePortableGpu});
    };
    const auto width=static_cast<std::uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(body_count))));
    const float spacing=3.0f;
    const float origin=0.5f*float(width-1u)*spacing;
    const float platform_extent=float(width)*spacing*0.5f+1.f;
    constexpr float dt=1.f/120.f;
    constexpr std::uint32_t warmup=4u;
    const std::uint32_t measured=body_count>=10000u?16u:48u;
    const auto run=[&](bool gpu_dy,std::vector<double>& step_ms) {
        const auto setup_begin=std::chrono::steady_clock::now();
        World world=create();
        const auto create_end=std::chrono::steady_clock::now();
        world.set_gpu_dy_enabled(gpu_dy);
        world.add_static_box(
            {platform_extent,0.5f,platform_extent},{0.f,-0.5f,0.f});
        std::vector<BodyId> ids;
        ids.reserve(body_count);
        for(std::uint32_t i=0u;i<body_count;++i) {
            const Vec3 position{
                float(i%width)*spacing-origin,0.48f,
                float(i/width)*spacing-origin};
            ids.push_back(world.add_dynamic_box(
                {0.5f,0.5f,0.5f},position,1.f));
        }
        const auto setup_end=std::chrono::steady_clock::now();
        std::cout<<"FYZZXYZZY_METRIC suite=hardware_setup"
                 <<" bodies="<<body_count
                 <<" gpu_dy="<<gpu_dy
                 <<" create_ms="<<std::chrono::duration<double,std::milli>(
                    create_end-setup_begin).count()
                 <<" assemble_ms="<<std::chrono::duration<double,std::milli>(
                    setup_end-create_end).count()<<'\n';
        step_ms.reserve(measured);
        for(std::uint32_t frame=0u;frame<warmup+measured;++frame) {
            const auto start=std::chrono::steady_clock::now();
            world.step(dt);
            if(frame>=warmup)
                step_ms.push_back(std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count());
        }
        for(std::size_t sample=0u;sample<step_ms.size();++sample)
            std::cout<<"FYZZXYZZY_METRIC suite=hardware_step"
                     <<" bodies="<<body_count
                     <<" gpu_dy="<<gpu_dy
                     <<" sample="<<sample
                     <<" step_ms="<<step_ms[sample]<<'\n';
        std::vector<BodyState> final_states;
        final_states.reserve(body_count);
        for(const BodyId id:ids)
            final_states.push_back(world.body_state(id));
        return final_states;
    };
    std::vector<double> cpu_ms,gpu_ms;
    std::vector<BodyState> cpu_states,gpu_states;
    if(gpu_first) {
        gpu_states=run(true,gpu_ms);
        cpu_states=run(false,cpu_ms);
    } else {
        cpu_states=run(false,cpu_ms);
        gpu_states=run(true,gpu_ms);
    }
    float max_position_error=0.f,max_velocity_error=0.f;
    for(std::uint32_t i=0u;i<body_count;++i) {
        const auto& a=cpu_states[i];
        const auto& b=gpu_states[i];
        for(const auto values:std::array<std::array<float,2>,6>{{
                {a.position.x,b.position.x},
                {a.position.y,b.position.y},
                {a.position.z,b.position.z},
                {a.velocity.x,b.velocity.x},
                {a.velocity.y,b.velocity.y},
                {a.velocity.z,b.velocity.z}}})
            require(std::isfinite(values[0]) && std::isfinite(values[1]),
                    "Hardware scene published nonfinite rigid state");
        max_position_error=std::max({max_position_error,
            std::fabs(a.position.x-b.position.x),
            std::fabs(a.position.y-b.position.y),
            std::fabs(a.position.z-b.position.z)});
        max_velocity_error=std::max({max_velocity_error,
            std::fabs(a.velocity.x-b.velocity.x),
            std::fabs(a.velocity.y-b.velocity.y),
            std::fabs(a.velocity.z-b.velocity.z)});
        require(a.position.y>0.44f && b.position.y>0.44f,
                "Hardware scene lost physical platform contact");
    }
    require(max_position_error<0.05f && max_velocity_error<0.05f,
            "Hardware scene CPU/GPU Dy trajectories diverged");
    const auto percentile=[](std::vector<double>& values,double fraction) {
        std::sort(values.begin(),values.end());
        const double rank=(values.size()-1u)*fraction;
        const auto lo=static_cast<std::size_t>(std::floor(rank));
        const auto hi=static_cast<std::size_t>(std::ceil(rank));
        return values[lo]+(rank-double(lo))*(values[hi]-values[lo]);
    };
    std::cout<<"FYZZXYZZY_METRIC suite=hardware_scene"
             <<" bodies="<<body_count
             <<" samples="<<measured
             <<" gpu_first="<<gpu_first
             <<" cpu_step_p50_ms="<<percentile(cpu_ms,0.50)
             <<" gpu_step_p50_ms="<<percentile(gpu_ms,0.50)
             <<" cpu_step_p95_ms="<<percentile(cpu_ms,0.95)
             <<" gpu_step_p95_ms="<<percentile(gpu_ms,0.95)
             <<" cpu_step_p99_ms="<<percentile(cpu_ms,0.99)
             <<" gpu_step_p99_ms="<<percentile(gpu_ms,0.99)
             <<" max_position_abs_error="<<max_position_error
             <<" max_velocity_abs_error="<<max_velocity_error<<'\n';
}

void verify_host_lane_packed_islands(
    fyzz::compute::Backend backend,const char* sap,
    const char* pcm,const char* dy_shader) {
    using namespace fyzz;
    std::ifstream stream(dy_shader,std::ios::binary);
    require(bool(stream),"GPU Dy shader bundle missing");
    std::vector<char> bytes(
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{});
    compute::ShaderBinary shader{};
    shader.format=backend==compute::Backend::Vulkan
        ?compute::ShaderFormat::Spirv:compute::ShaderFormat::Dxil;
    shader.bytes.assign(bytes.begin(),bytes.end());
    native::Scene cpu,gpu;
    cpu.set_enhanced_determinism(true);
    gpu.set_enhanced_determinism(true);
    auto owner=std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
        backend,load_native_gpu_sap_shaders(backend,sap),
        128u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));
    std::array<BodyId,2> cpu_bodies{},gpu_bodies{};
    for(std::size_t i=0;i<cpu_bodies.size();++i) {
        const Vec3 floor{static_cast<float>(i)*8.f,0.f,0.f};
        const Vec3 position{floor.x,0.86f,0.f};
        cpu.add_static_box({0.6f,0.5f,0.6f},floor);
        gpu.add_static_box({0.6f,0.5f,0.6f},floor);
        cpu_bodies[i]=cpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
        gpu_bodies[i]=gpu.add_dynamic_box({0.5f,0.5f,0.5f},position,1.f);
    }
    for(unsigned frame=0;frame<4u;++frame) {
        cpu.step(1.f/120.f);
        gpu.step(1.f/120.f);
        const auto schedule=producer->dy_launch_stats();
        require(schedule.host_lane_packed_tasks==cpu_bodies.size() &&
                schedule.device_island_indirect_dispatches==0u &&
                schedule.wide_tasks==0u,
            "Independent CPU-admitted islands did not use packed GPU lanes");
        require(cpu.contacts().size()==cpu_bodies.size() &&
                cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size(),
            "Packed CPU-admitted islands changed Sc contact publication");
        for(std::size_t i=0;i<cpu_bodies.size();++i)
            test::compound_compare_state(
                gpu.body_state(gpu_bodies[i]),cpu.body_state(cpu_bodies[i]),
                0.03f,"Packed GPU singleton diverged from CPU rigid contact");
    }
}

int main(int argc,char** argv) {
    try {
        require(argc>=4 && argc<=7,
                "Expected SAP/PCM manifests, source Dy and convex Dy shaders, optional scenario");
        const bool has_convex_shader=argc>=5 &&
            (std::string_view(argv[4]).ends_with(".spv") ||
             std::string_view(argv[4]).ends_with(".dxil"));
        const std::string_view scenario=argc>=6?argv[5]:
            (argc==5 && !has_convex_shader?argv[4]:"all");
        if((scenario=="hardware_scene_1k" || scenario=="hardware_scene_10k") &&
           argc==7)
            require(std::string_view(argv[6])=="gpu-first",
                    "Unknown hardware scenario execution-order selector");
        require(scenario=="all" || scenario=="scene" ||
                scenario=="aggregate_coupled" ||
                scenario=="sparse_contacts" || scenario=="coupled_stack" ||
                scenario=="coupled_solver" || scenario=="single_island" ||
                scenario=="legacy_gu_solver" ||
                scenario=="mixed_solver" || scenario=="dy_switch" ||
                scenario=="plane_only_pcm" ||
                scenario=="distance_solver" || scenario=="host_packed" ||
                scenario=="compound_shapes" || scenario=="compound_solver" ||
                scenario=="sphere_solver" || scenario=="convex_plane" ||
                scenario=="convex_finite" || scenario=="wide_islands" ||
                scenario=="wide_islands_10k" ||
                scenario=="zero_response_wide" ||
                scenario=="star_island" ||
                scenario=="dual_hub" ||
                scenario=="star_island_10k" ||
                scenario=="free_integration" ||
                scenario=="device_admission" ||
                scenario=="indirect_api" ||
                scenario=="hardware_scene_1k" ||
                scenario=="hardware_scene_10k" ||
                scenario=="wide_islands_diagnose" ||
                scenario=="wide_islands_diagnose_no_friction" ||
                scenario=="wide_islands_diagnose_no_strong_friction",
                "Unknown GPU scene regression scenario");
#ifdef FYZZXYZZY_TEST_D3D12
        constexpr auto backend=fyzz::compute::Backend::D3D12;
#else
        constexpr auto backend=fyzz::compute::Backend::Vulkan;
#endif
        const auto adapters=fyzz::compute::enumerate_devices(backend);
        require(!adapters.empty(),"No compute adapter is available");
        const auto& adapter=adapters.front();
        std::cout<<"FYZZXYZZY_ADAPTER backend="
                 <<(backend==fyzz::compute::Backend::D3D12?"d3d12":"vulkan")
                 <<" ordinal="<<adapter.ordinal
                 <<" vendor_id="<<adapter.vendor_id
                 <<" device_id="<<adapter.device_id
                 <<" subgroup_size="<<adapter.subgroup_size
                 <<" max_workgroup_count_x="<<adapter.max_workgroup_count[0]
                 <<" unified_memory="<<adapter.unified_memory
                 <<" name=\""<<adapter.name<<"\"\n";
        verify_rigid_shader_requirements(backend,argv[3]);
        if(scenario=="all" || scenario=="compound_shapes")
            fyzz::tests::measure_gpu_case("sc_pcm","compound_shapes",
                [&]{verify_compound_shape_gpu_pairs(backend,argv[1]);});
        if(scenario=="all" || scenario=="compound_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","compound_solver",
                [&]{verify_compound_actor_gpu_solver(backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="sphere_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","sphere_solver",
                [&]{
                    verify_sphere_gpu_solver(
                        backend,argv[1],argv[2],argv[3]);
                    verify_host_primitive_pcm_solver(
                        backend,argv[1],argv[2],argv[3]);
                });
        if(scenario=="all" || scenario=="convex_plane")
            fyzz::tests::measure_gpu_case("sc_pcm","convex_plane",
                [&]{
                    require(has_convex_shader,
                        "Cooked convex GPU case requires compiled hull Dy shader");
                    verify_convex_plane_gpu_solver(
                        backend,argv[1],argv[2],argv[3],argv[4]);
                });
        if(scenario=="all" || scenario=="convex_finite")
            fyzz::tests::measure_gpu_case("sc_pcm","convex_finite",
                [&]{
                    require(has_convex_shader,
                        "Finite convex GPU case requires compiled hull Dy shader");
                    verify_convex_finite_gpu_solver(
                        backend,argv[1],argv[2],argv[3],argv[4]);
                    verify_convex_face_dispatch_window(
                        backend,argv[1],argv[2]);
                });
        if(scenario=="indirect_api")
            fyzz::tests::measure_gpu_case("sc_pcm","indirect_api",
                [&]{verify_device_sc_dy_indirect_api(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="device_admission")
            fyzz::tests::measure_gpu_case("sc_pcm","device_admission",
                [&]{verify_device_sc_dy_task_admission(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="scene")
            fyzz::tests::measure_gpu_case("sc_pcm","scene",
                [&]{
                    verify(backend,argv[1],argv[2]);
                    verify_queued_pcm_dy(backend,argv[1],argv[2],argv[3]);
                    verify_device_sc_dy_task_admission(
                        backend,argv[1],argv[2],argv[3]);
                });
        if(scenario=="all" || scenario=="sparse_contacts")
            fyzz::tests::measure_gpu_case("sc_pcm","sparse_contacts",
                [&]{
                    verify_sparse_contact_stream(backend,argv[1],argv[2]);
                    verify_dormant_contact_stream(backend,argv[1],argv[2]);
                    verify_dormant_empty_pcm(backend,argv[1],argv[2]);
                });
        if(scenario=="all" || scenario=="coupled_stack")
            fyzz::tests::measure_gpu_case("sc_pcm","coupled_stack",
                [&]{verify_coupled_rigid_stack(backend,argv[1],argv[2]);});
        if(scenario=="all" || scenario=="single_island")
            fyzz::tests::measure_gpu_case("sc_pcm","single_island",
                [&]{verify_gpu_single_island_hybrid_solver(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="coupled_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","coupled_solver",
                [&]{
                    for(std::uint32_t spring=0u;spring<3u;++spring)
                        verify_gpu_coupled_solver(
                            backend,argv[1],argv[2],argv[3],spring);
                });
        if(scenario=="all" || scenario=="legacy_gu_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","legacy_gu_solver",
                [&]{verify_cpu_gu_gpu_dy_solver(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="aggregate_coupled") {
            require(argc==7 && has_convex_shader,
                "Integrated aggregate scene needs the aggregate manifest");
            fyzz::tests::measure_gpu_case("sc_pcm","aggregate_coupled",
                [&]{verify_integrated_aggregate_gpu_pipeline(
                    backend,argv[1],argv[2],argv[3],argv[4],argv[6]);});
        }
        if(scenario=="all" || scenario=="plane_only_pcm")
            fyzz::tests::measure_gpu_case("sc_pcm","plane_only_pcm",
                [&]{verify_plane_pcm_without_finite_pcm(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="dy_switch")
            fyzz::tests::measure_gpu_case("sc_pcm","dy_switch",
                [&]{verify_dy_execution_switch(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="mixed_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","mixed_solver",
                [&]{verify_mixed_gu_pcm_gpu_dy_solver(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="distance_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","distance_solver",
                [&]{verify_gpu_distance_solver(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="host_packed")
            fyzz::tests::measure_gpu_case("sc_pcm","host_packed",
                [&]{verify_host_lane_packed_islands(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="free_integration")
            fyzz::tests::measure_gpu_case("sc_pcm","free_integration",
                [&]{verify_gpu_free_integration(backend,argv[1],argv[2],argv[3]);});
        if(scenario=="zero_response_wide")
            fyzz::tests::measure_gpu_case("sc_pcm","zero_response_wide",
                [&]{verify_gpu_zero_response_wide_island(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="wide_islands")
            fyzz::tests::measure_gpu_case("sc_pcm","multiwave_coloring",
                [&]{verify_gpu_high_degree_coloring(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="wide_islands" ||
           scenario=="star_island")
            fyzz::tests::measure_gpu_case("sc_pcm","star_island",
                [&]{verify_gpu_high_degree_coloring(
                    backend,argv[1],argv[2],argv[3],1024u);});
        if(scenario=="dual_hub")
            fyzz::tests::measure_gpu_case("sc_pcm","dual_hub",
                [&]{verify_gpu_high_degree_coloring(
                    backend,argv[1],argv[2],argv[3],68u,true,1u);});
        if(scenario=="star_island_10k")
            fyzz::tests::measure_gpu_case("sc_pcm","star_island_10k",
                [&]{verify_gpu_high_degree_coloring(
                    backend,argv[1],argv[2],argv[3],10000u);});
        if(scenario=="all" || scenario=="wide_islands" ||
           scenario=="wide_islands_10k" ||
           scenario=="wide_islands_diagnose" ||
           scenario=="wide_islands_diagnose_no_friction" ||
           scenario=="wide_islands_diagnose_no_strong_friction") {
            const bool large=scenario=="wide_islands_10k";
            const bool diagnose=scenario.starts_with("wide_islands_diagnose");
            const bool no_friction=scenario=="wide_islands_diagnose_no_friction";
            const bool no_strong_friction=
                scenario=="wide_islands_diagnose_no_strong_friction";
            fyzz::tests::measure_gpu_case("sc_pcm",scenario=="all" ?
                "wide_islands" : scenario,
                [&]{verify_gpu_wide_island_solver(
                    backend,argv[1],argv[2],argv[3],
                    large ? 10240u : 768u,diagnose,
                    no_friction,no_strong_friction);});
        }
        if(scenario=="hardware_scene_1k" || scenario=="hardware_scene_10k")
            fyzz::tests::measure_gpu_case("hardware_scene",scenario,
                [&]{qualify_hardware_scene(
                    backend,argv[1],argv[2],argv[3],
                    scenario=="hardware_scene_1k"?1000u:10000u,
                    argc==7 && std::string_view(argv[6])=="gpu-first");});
        std::cout<<"Native Sc GPU PCM scene "<<scenario<<" PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Native Sc GPU PCM scene regression FAILED: "
                  <<error.what()<<'\n';
        return 1;
    }
}
