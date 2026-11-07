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
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace fyzz::native {
struct GpuPcmFrictionAnchorFixture {
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
            if(a.pair!=b.pair || a.contact_count!=b.contact_count ||
               a.anchor_count!=b.anchor_count ||
               !near(a.normal,b.normal) ||
               !near(a.bounds_min,b.bounds_min) ||
               !near(a.bounds_max,b.bounds_max))
                throw std::runtime_error(
                    "GPU PCM changed source correlated friction topology");
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
    const auto quiet=gpu.advance(shapes,false,false);
    require(quiet.found.empty() && quiet.lost.empty(),
        "Source GPU broadphase republished unchanged compound shape pairs");
    // ScBodySim changes one rigid actor's motion and refilters every attached
    // ElementSim, even when the shapes overlap each other.
    gpu.set_actor_motion(0u,Motion::Kinematic);
    shapes[0].motion=Motion::Kinematic;
    shapes[1].motion=Motion::Kinematic;
    const auto kinematic=gpu.advance(shapes,false,true);
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
    const auto edge=gpu.contact_edge_index({gpu_a,gpu_b});
    // The device's 84-byte stream reconstructs Sc's world-space Gu
    // contacts and the physical-touch transition on the same EdgeIndex.
    const std::array<native::GpuPcmBoxPair,1> request{{
        {gpu_a,gpu_b,edge,false}
    }};
    std::array<native::GpuPcmBoxResult,1> direct{};
    producer->generate_gpu_pcm_box_contacts(request,direct);
    require(producer->pcm_contact_readback_bytes()==
                sizeof(physx_gpu::SourcePcmGpuContactStreamNative),
            "Direct GPU PCM readback did not use the compact device stream");
    require(direct[0].edge==edge && direct[0].status==0u &&
            direct[0].cached_count<=4u &&
            direct[0].count<=direct[0].cached_count &&
            (direct[0].selected_and_regenerated&0x7fffffffu)<=4u &&
            direct[0].count>0u &&
            direct[0].touch_transition==native::GpuPcmTouchTransition::Persists &&
            std::isfinite(direct[0].normal.x) &&
            std::isfinite(direct[0].points[0].separation),
            "Direct Sc GPU PCM readback disagrees with the shader record");
    // A quiet refresh must reuse Sc's original device pair descriptor:
    // the GPU still runs PCM, but no pair-list HtoD submission is needed.
    std::array<native::GpuPcmBoxResult,1> repeated{};
    producer->generate_gpu_pcm_box_contacts(request,repeated);
    require(producer->pcm_pair_upload_bytes()==0u &&
            repeated[0].status==0u && repeated[0].edge==edge &&
            repeated[0].count>0u &&
            repeated[0].touch_transition==native::GpuPcmTouchTransition::Persists,
            "Stable Sc PCM manager unexpectedly re-uploaded source pair input");
    // Explicitly invalidate a live Sc EdgeIndex. A newly admitted or
    // rejected source frame uses the identical candidate-owned reset.
    const std::array<native::GpuPcmBoxPair,1> reset_request{{
        {gpu_a,gpu_b,edge,true,16u,false,0.f,0.f,true}
    }};
    std::array<native::GpuPcmBoxResult,1> reset{};
    producer->generate_gpu_pcm_box_contacts(reset_request,reset);
    require(producer->pcm_pair_upload_bytes()==
                sizeof(physx_gpu::SourcePcmBoxPairNative),
            "Source PCM reset did not refresh its GPU pair descriptor");
    require(reset[0].edge==edge && reset[0].status==0u &&
            reset[0].count>0u &&
            reset[0].touch_transition==native::GpuPcmTouchTransition::Persists &&
            (reset[0].selected_and_regenerated&0x80000000u)!=0u,
            "Source GPU PCM failed to regenerate a reset EdgeIndex");
    for(std::uint32_t i=0u;i<reset[0].count;++i)
        require(std::isfinite(reset[0].points[i].separation) &&
                std::fabs(reset[0].points[i].separation-
                          direct[0].points[i].separation)<0.002f,
                "Source GPU PCM reset altered the Gu witness");
    // Force the original AABB volume domain to grow while the first
    // contact manager retains GPU-local cached shape geometry.
    for(std::uint32_t i=0u;i<36u;++i) {
        const Vec3 position{20.f+3.f*float(i),1.f,0.f};
        cpu.add_static_box({0.5f,0.5f,0.5f},position);
        gpu.add_static_box({0.5f,0.5f,0.5f},position);
    }
    check(true);
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
    gpu.set_gpu_broadphase(
        std::make_unique<physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,load_native_gpu_sap_shaders(backend,sap),
            128u,128u,std::nullopt,
            load_native_gpu_pcm_shaders(backend,pcm)));
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
        require(gpu.solved_result_frame()==gpu.frame_index(),
                "GPU coupled rigid stack did not complete Dy solve");
    }
    require(coupled,
            "GPU coupled rigid stack never admitted dynamic-dynamic contacts");
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
    bool saw_device_pcm_contact_rows=false;
    bool saw_compact_prep=false;
    std::int64_t saved_prep_bytes=0;
    double max_normal_error=0.0,max_friction_error=0.0;
    float max_position_error=0.f,max_velocity_error=0.f;
    std::size_t uploaded_bytes=0,downloaded_bytes=0;
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
        cpu.step(dt);
        gpu.step(dt);
        native::GpuPcmFrictionAnchorFixture::verify(cpu,gpu);
        require(producer->source_handle_projection_count()==1u,
            "Source GPU SAP remapped unchanged Sc ElementSim handles");
        if(frame>0u)
            require(producer->last_frame().device_cached_box_count>0u,
                "Source Dy integrated Sc pose was reuploaded to GPU SAP");
        const auto integrated=producer->dy_gpu_integrated_body_count();
        const auto resident=producer->dy_resident_pose_count();
        const auto host_pose_bytes=producer->dy_host_pose_upload_bytes();
        require(integrated>=3u && resident<=integrated &&
                host_pose_bytes==(integrated-resident)*
                    sizeof(physx_gpu::SourceDyHostPoseNative),
            "GPU Dy resident contact poses have inconsistent transfer accounting");
        if(frame==0u)
            require(resident==0u,
                "First Sc contact solve reused an unauthorized device pose");
        else if(frame==6u)
            require(resident>=2u && resident<integrated &&
                    host_pose_bytes>=sizeof(physx_gpu::SourceDyHostPoseNative),
                "Authored Sc contact pose did not invalidate only its device owner");
        else
            require(resident>=2u && host_pose_bytes<=
                    sizeof(physx_gpu::SourceDyHostPoseNative),
                "Active GPU contact bodies still upload authored mass poses");
        const auto [uploaded,downloaded]=producer->dy_distance_transfer_bytes();
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_colored_tasks>0u &&
                schedule.device_colored_descriptors>0u,
            "Coupled Dy solver skipped device-owned constraint partitioning");
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
        require(producer->dy_upload_span_count()==1u &&
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
    require(saw_compact_prep,
        "Source Dy contact prep did not reuse GPU-resident body inputs");
    require(saw_contact && saw_force && saw_friction &&
            saw_correlated_multi_point,
        "Source coupled GPU test never exercised multi-point contact/friction");
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
                    producer->dy_upload_span_count()==1u,
                "Mixed source Gu and PCM rows did not enter one GPU Dy task");
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
            for(std::size_t i=0;i<cpu.solved_normal_impulses().size();++i)
                require(std::fabs(cpu.solved_normal_impulses()[i]-
                                  gpu.solved_normal_impulses()[i])<0.08,
                    "Mixed CPU Gu/GPU PCM normal impulse lost source ordinal");
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
    fyzz::compute::Backend backend,const char* sap,const char* dy_shader) {
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
            128u,128u,std::nullopt,std::nullopt,std::move(shader));
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
        if(i==0u)d.lock_flags=LockAngularX;
        cpu.set_dynamics(ids_cpu[i],d);
        gpu.set_dynamics(ids_gpu[i],d);
    }
    // Two disconnected accurate islands must advance through independent
    // source task workgroups. Off-center anchors exercise angular Jacobians.
    cpu.create_anchored_distance_constraint(ids_cpu[0],ids_cpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    gpu.create_anchored_distance_constraint(ids_gpu[0],ids_gpu[1],
        {0.0f,0.2f,0.0f},{0.0f,-0.2f,0.0f},3.7f);
    cpu.create_center_distance_constraint(ids_cpu[2],ids_cpu[3],3.5f);
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
                producer->dy_gpu_integrated_body_count()>0u,
            "Source GPU Dy 1D task never published device body motion");
        const auto [uploaded,downloaded]=
            producer->dy_distance_transfer_bytes();
        // Five dynamic solver bodies and three disjoint task reports.
        // Joint impulses stay in their device PGS rows, not the
        // completed Sc result stream.
        constexpr std::size_t expected_output=
            5u*(48u+sizeof(native::GpuDyIntegratedPose))+
            3u*sizeof(std::uint32_t);
        require(downloaded==expected_output && downloaded<uploaded/2u,
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
                    std::fabs(expected.angular_velocity.z-
                              actual.angular_velocity.z)<0.001f &&
                    std::fabs(expected.orientation.x-actual.orientation.x)<0.001f &&
                    std::fabs(expected.orientation.y-actual.orientation.y)<0.001f &&
                    std::fabs(expected.orientation.z-actual.orientation.z)<0.001f &&
                    std::fabs(expected.orientation.w-actual.orientation.w)<0.001f,
                "Source GPU Dy locked-axis integration diverged from CPU PGS");
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
        const auto integrated=producer->dy_gpu_integrated_body_count();
        const auto resident=producer->dy_resident_pose_count();
        const auto uploaded=producer->dy_host_pose_upload_bytes();
        require(integrated==gpu_ids.size() && resident<=integrated &&
                uploaded==(integrated-resident)*
                    sizeof(physx_gpu::SourceDyHostPoseNative),
            "GPU sphere mass poses have inconsistent source upload accounting");
        if(frame==0u)
            require(resident==0u,
                "New sphere actors reused unauthorised device mass poses");
        else if(frame==3u)
            require(resident<integrated && uploaded>0u,
                "Authored sphere quaternion did not invalidate its device pose");
        else
            require(resident>=2u && uploaded<=
                    sizeof(physx_gpu::SourceDyHostPoseNative),
                "Sphere contact preparation still uploads mass poses");
        require(cpu.contacts().size()==gpu.contacts().size() &&
                cpu.solved_normal_impulses().size()==
                    gpu.solved_normal_impulses().size(),
            "GPU sphere scene changed Sc PCM/Gu manager identity");
        if(frame==0u) {
            const auto pcm_bytes=producer->pcm_contact_readback_bytes();
            const auto pcm_rows=producer->dy_pcm_device_contact_rows();
            const auto cpu_bytes=producer->dy_cpu_gu_geometry_bytes();
            // Sc admits one speculative plane manager per dynamic
            // sphere. The two elevated spheres have no physical plane
            // touch, but their GPU PCM records still execute this frame.
            constexpr std::size_t pcm_managers=5u;
            if(gpu.contacts().size()!=3u || cpu_bytes!=0u ||
               pcm_rows!=3u ||
               pcm_bytes!=pcm_managers*sizeof(
                   physx_gpu::SourcePcmGpuContactStreamNative)) {
                throw std::runtime_error(
                    "Sphere GPU contact path: managers="+
                    std::to_string(gpu.contacts().size())+
                    " device_rows="+std::to_string(pcm_rows)+
                    " cpu_gu_bytes="+std::to_string(cpu_bytes)+
                    " pcm_readback_bytes="+std::to_string(pcm_bytes));
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
    cpu.advance_collision(true);
    gpu.advance_collision(true);
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
    test::compound_mass_and_motion(gpu,&cpu);
    test::compound_contact_lifetimes(gpu,&cpu);
    test::compound_default_contact_response(gpu,&cpu);
    test::compound_joint_response(gpu,&cpu);
    test::fixed_joint_response(gpu,&cpu);
    test::spherical_joint_response(gpu,&cpu);
    test::revolute_joint_response(gpu,&cpu);
    test::prismatic_joint_response(gpu,&cpu);
    test::compound_plane_box_device_pcm(gpu,&cpu);
    test::compound_plane_sphere_device_pcm(gpu,&cpu);
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
        free_motion.lock_flags=LockLinearZ|LockAngularX;
        for(unsigned i=1;i<=260u;++i) {
            const float x=3.f*static_cast<float>(i);
            const auto body=scene.add_dynamic_box(
                {0.25f,0.4f,0.6f},{x,3.f,0.f},1.f);
            bodies.push_back(body);
            scene.set_dynamics(body,free_motion);
            scene.set_velocity(body,{0.2f,0.3f,1.f});
            scene.set_angular_velocity(body,{1.f,0.2f,-0.6f});
        }
        const auto compound=scene.create_body({{0.f,3.f,4.f},{}},true);
        compound_shape=scene.attach_box(
            compound,{0.2f,0.3f,0.4f},{{-0.5f,0.f,0.f},{}});
        scene.attach_sphere(compound,0.4f,{{0.7f,0.4f,0.f},{}});
        scene.update_mass_properties(compound,1.f);
        scene.set_dynamics(compound,free_motion);
        scene.set_angular_velocity(compound,{0.f,0.5f,0.9f});
        bodies.push_back(compound);
        const auto shapeless=scene.create_body({{0.f,6.f,0.f},{}},true);
        scene.set_dynamics(shapeless,free_motion);
        scene.set_velocity(shapeless,{0.4f,-0.3f,0.f});
        bodies.push_back(shapeless);
        return bodies;
    };
    ShapeId expected_shape{},actual_shape{};
    const auto expected=assemble(cpu,expected_shape);
    const auto actual=assemble(gpu,actual_shape);
    bool touched=false,released=false;
    for(unsigned frame=0;frame<12u;++frame) {
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
                producer->dy_upload_span_count()==1u,
                "GPU Dy did not integrate every active body in one upload");
        const auto resident=producer->dy_resident_pose_count();
        const auto metadata=producer->dy_launch_stats();
        constexpr auto velocity_bytes=
            sizeof(physx_gpu::SourceDyPreparedVelocityNative);
        require(metadata.prepared_velocity_upload_bytes%velocity_bytes==0u &&
                metadata.prepared_velocity_upload_bytes/velocity_bytes+
                    metadata.resident_velocity_reuses==actual.size() &&
                (frame==0u
                    ? metadata.prepared_velocity_upload_bytes==
                          actual.size()*velocity_bytes &&
                      metadata.resident_velocity_reuses==0u
                    : metadata.resident_velocity_reuses>=
                          (frame==7u?259u:260u)) &&
                (frame!=7u ||
                 metadata.prepared_velocity_upload_bytes>=velocity_bytes),
            "Native Dy did not retain final GPU velocities across frames");
        require((frame==0u
                    ? metadata.actor_local_upload_bytes==
                          actual.size()*sizeof(physx_gpu::SourceDyActorLocalNative) &&
                      metadata.actor_local_upload_spans==1u
                    : frame==4u
                        ? metadata.actor_local_upload_bytes==
                              sizeof(physx_gpu::SourceDyActorLocalNative) &&
                          metadata.actor_local_upload_spans==1u &&
                          metadata.reused_actor_local_poses==actual.size()-1u
                        : metadata.actor_local_upload_bytes==0u &&
                          metadata.actor_local_upload_spans==0u &&
                          metadata.reused_actor_local_poses==actual.size()),
            "Source Dy reuploaded unchanged actor-local mass frames");
        require(resident<=actual.size() &&
                producer->dy_host_pose_upload_bytes()==
                    (actual.size()-resident)*28u &&
                (frame==0u?resident==0u:resident>=260u),
                "GPU Dy did not retain authorized Sc poses across frames");
        require(cpu.contacts().size()==gpu.contacts().size(),
                "Device-integrated free body changed contact admission");
        const bool contacts=producer->dy_pcm_device_contact_rows()!=0u;
        const auto schedule=producer->dy_launch_stats();
        if(contacts) {
            require(!released && schedule.dispatches==2u &&
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
                        actual.size()*sizeof(native::GpuDyIntegratedPose),
                    "Free-only Dy staged solver rows or dispatched PGS");
        }
        if(frame>0u)
            require(producer->last_frame().device_cached_box_count>=262u,
                    "Free body or compound shape lost resident bounds");
        for(std::size_t i=0;i<actual.size();++i)
            test::compound_compare_state(gpu.body_state(actual[i]),
                cpu.body_state(expected[i]),0.002f,
                "GPU Dy free or transitioning body");
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
    require(after_replacement.prepared_velocity_upload_bytes%
                sizeof(physx_gpu::SourceDyPreparedVelocityNative)==0u &&
            after_replacement.prepared_velocity_upload_bytes/
                    sizeof(physx_gpu::SourceDyPreparedVelocityNative)+
                after_replacement.resident_velocity_reuses==actual.size(),
        "Retired native body contaminated resident velocity publication");
    require(after_replacement.actor_local_upload_bytes==
                sizeof(physx_gpu::SourceDyActorLocalNative) &&
            after_replacement.actor_local_upload_spans==1u &&
            after_replacement.reused_actor_local_poses==actual.size()-1u,
        "GPU Dy retired actor metadata leaked into newly admitted rigid");
    test::compound_compare_state(
        gpu.body_state(replacement_gpu),cpu.body_state(replacement_cpu),
        0.002f,"Resident GPU Dy actor replacement");
}

void verify_gpu_wide_island_solver(
    fyzz::compute::Backend backend,const char* sap,const char* pcm,
    const char* dy_shader,unsigned connected_bodies) {
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
        backend,load_native_gpu_sap_shaders(backend,sap),
        connected_bodies>768u ? 32768u : 4096u,128u,std::nullopt,
        load_native_gpu_pcm_shaders(backend,pcm),std::move(shader));
    auto* producer=owner.get();
    gpu.set_gpu_broadphase(std::move(owner));

    struct Assembly {
        std::vector<BodyId> bodies;
        ConstraintId split_joint=0;
    };
    const auto assemble=[connected_bodies](native::Scene& scene) {
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
    for(unsigned frame=0;frame<4u;++frame) {
        if(frame==2u) {
            cpu.retire_constraint(expected.split_joint);
            gpu.retire_constraint(actual.split_joint);
        }
        constexpr float dt=1.0f/120.0f;
        cpu.step(dt);
        gpu.step(dt);
        const auto schedule=producer->dy_launch_stats();
        require(schedule.device_colored_tasks>=1u &&
                schedule.device_colored_descriptors>=connected_bodies,
            "Wide Dy task retained CPU-generated conflict partitions");
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
                         sizeof(std::uint32_t)),
                "Device Dy coloring still uploads its derived output arrays");
        require(producer->dy_upload_span_count()==1u &&
                producer->dy_gpu_integrated_body_count()==actual.bodies.size() &&
                producer->dy_pcm_device_contact_rows()>=actual.bodies.size() &&
                producer->dy_cpu_gu_geometry_bytes()==0u,
                "Wide island lost GPU contact preparation or body integration");
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
                schedule.dispatches==(b.size()<=128u?1u:2u) &&
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

int main(int argc,char** argv) {
    try {
        require(argc>=4 && argc<=6,
                "Expected SAP/PCM manifests, source Dy and convex Dy shaders, optional scenario");
        const bool has_convex_shader=argc>=5 &&
            (std::string_view(argv[4]).ends_with(".spv") ||
             std::string_view(argv[4]).ends_with(".dxil"));
        const std::string_view scenario=argc==6?argv[5]:
            (argc==5 && !has_convex_shader?argv[4]:"all");
        require(scenario=="all" || scenario=="scene" ||
                scenario=="sparse_contacts" || scenario=="coupled_stack" ||
                scenario=="coupled_solver" || scenario=="legacy_gu_solver" ||
                scenario=="mixed_solver" || scenario=="distance_solver" ||
                scenario=="compound_shapes" || scenario=="compound_solver" ||
                scenario=="sphere_solver" || scenario=="convex_plane" ||
                scenario=="convex_finite" || scenario=="wide_islands" ||
                scenario=="wide_islands_10k" ||
                scenario=="zero_response_wide" ||
                scenario=="star_island" ||
                scenario=="dual_hub" ||
                scenario=="star_island_10k" ||
                scenario=="free_integration",
                "Unknown GPU scene regression scenario");
#ifdef FYZZXYZZY_TEST_D3D12
        constexpr auto backend=fyzz::compute::Backend::D3D12;
#else
        constexpr auto backend=fyzz::compute::Backend::Vulkan;
#endif
        verify_rigid_shader_requirements(backend,argv[3]);
        if(scenario=="all" || scenario=="compound_shapes")
            fyzz::tests::measure_gpu_case("sc_pcm","compound_shapes",
                [&]{verify_compound_shape_gpu_pairs(backend,argv[1]);});
        if(scenario=="all" || scenario=="compound_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","compound_solver",
                [&]{verify_compound_actor_gpu_solver(backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="sphere_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","sphere_solver",
                [&]{verify_sphere_gpu_solver(
                    backend,argv[1],argv[2],argv[3]);});
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
                });
        if(scenario=="all" || scenario=="scene")
            fyzz::tests::measure_gpu_case("sc_pcm","scene",
                [&]{verify(backend,argv[1],argv[2]);});
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
        if(scenario=="all" || scenario=="mixed_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","mixed_solver",
                [&]{verify_mixed_gu_pcm_gpu_dy_solver(
                    backend,argv[1],argv[2],argv[3]);});
        if(scenario=="all" || scenario=="distance_solver")
            fyzz::tests::measure_gpu_case("sc_pcm","distance_solver",
                [&]{verify_gpu_distance_solver(backend,argv[1],argv[3]);});
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
        if(scenario=="all" || scenario=="wide_islands" || scenario=="wide_islands_10k") {
            const bool large=scenario=="wide_islands_10k";
            fyzz::tests::measure_gpu_case("sc_pcm",large ? "wide_islands_10k" : "wide_islands",
                [&]{verify_gpu_wide_island_solver(
                    backend,argv[1],argv[2],argv[3],large ? 10240u : 768u);});
        }
        std::cout<<"Native Sc GPU PCM scene "<<scenario<<" PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Native Sc GPU PCM scene regression FAILED: "
                  <<error.what()<<'\n';
        return 1;
    }
}
