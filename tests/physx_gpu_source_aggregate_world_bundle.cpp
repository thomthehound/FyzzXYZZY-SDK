// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool pass,const char* message) {
    if(!pass)throw std::runtime_error(message);
}
void verify_aggregate_gpu_rollback(
    fyzz::compute::Backend backend,
    const char* sap_manifest,const char* aggregate_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            128u,128u,
            fyzz::load_native_gpu_aggregate_shaders(backend,aggregate_manifest));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto aggregate=scene.create_aggregate(2u,true);
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f,aggregate);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.4f,1.0f,0.0f},1.0f,aggregate);
    const auto outside=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId a,fyzz::BodyId b) {
        const fyzz::native::Pair expected{std::min(a,b),std::max(a,b)};
        return std::find(pairs.begin(),pairs.end(),expected)!=pairs.end();
    };
    scene.advance_collision();
    require(contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside),
            "Aggregate GPU rollback fixture lacks real source pairs");

    // Simulate a failed Sc continuation after GPU fetch. The source
    // owner must rebuild its own histories, not replay CPU SAP pairs.
    producer->invalidate_after_rollback();
    scene.advance_collision();
    require(producer->last_frame().update.counts.previous_boxes==0u &&
            producer->last_frame().update.counts.created==2u &&
            contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside) &&
            scene.found_pairs().empty() && scene.lost_pairs().empty(),
            "Aggregate GPU rollback did not preserve Sc pairs after source rebuild");
    scene.set_position(a,{8.0f,1.0f,0.0f});
    scene.advance_collision();
    require(contains(scene.lost_pairs(),a,b) &&
            !contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside),
            "Aggregate GPU rebuild did not restore native lost-pair history");

    // An unchanged actor overlap is not a new Sc pair merely because
    // the source AABB owner replaces an aggregate ElementSim handle.
    // Subsequent removal from the released aggregate must retain the
    // standalone GPU history, including a real later lost crossing.
    scene.remove_actor_from_aggregate(aggregate,b);
    scene.advance_collision();
    require(contains(scene.pairs(),b,outside) &&
            std::count(scene.pairs().begin(),scene.pairs().end(),
                       fyzz::native::Pair{std::min(b,outside),
                                          std::max(b,outside)})==1,
            "Aggregate actor removal duplicated or retired a live Sc overlap");
    scene.destroy_aggregate(aggregate);
    scene.advance_collision();
    require(contains(scene.pairs(),b,outside),
            "Aggregate release lost a surviving standalone GPU overlap");
    scene.set_position(b,{16.0f,1.0f,0.0f});
    scene.advance_collision();
    require(contains(scene.lost_pairs(),b,outside) &&
            !contains(scene.pairs(),b,outside),
            "Released actor failed source GPU lost-pair retirement");
}


// A real Dy failure after a completed GPU fetch must reject the Sc
// transaction and reconstruct original AABB/SAP history on the next step.
// This uses the source velocity-modification path, not a scripted GPU
// report or a manually invoked history invalidation.
void verify_aggregate_postfetch_sc_rollback(
    fyzz::compute::Backend backend,
    const char* sap_manifest,const char* aggregate_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            128u,128u,
            fyzz::load_native_gpu_aggregate_shaders(backend,aggregate_manifest));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto root=scene.create_aggregate(2u,true);
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f,root);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.4f,1.0f,0.0f},1.0f,root);
    const auto outside=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.8f,1.0f,0.0f});
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId x,fyzz::BodyId y) {
        const fyzz::native::Pair key{std::min(x,y),std::max(x,y)};
        return std::find(pairs.begin(),pairs.end(),key)!=pairs.end();
    };

    scene.step(1.0f/120.0f);
    require(contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside),
            "Postfetch rollback fixture did not publish GPU aggregate contacts");
    const auto previous_frame=scene.frame_index();
    const auto previous_result=scene.solved_result_frame();
    const auto previous_edge=scene.contact_edge_index({a,b});
    require(previous_result==previous_frame,
            "Postfetch rollback fixture has no completed source Dy result");

    // This authored transform makes the fetched GPU frame remove a/b.
    // Two separately valid, finite angular inputs overflow only when
    // Dy::preIntegration combines the velocity-modification streams.
    scene.set_position(a,{8.0f,1.0f,0.0f});
    const float extreme=std::numeric_limits<float>::max()*0.75f;
    scene.set_angular_velocity(a,{extreme,0.0f,0.0f});
    scene.add_spatial_velocity_mod(
        a,{extreme,0.0f,0.0f},
        fyzz::ForceMode::VelocityChange,true);
    const auto before=scene.body_state(a);
    bool rejected=false;
    try {
        scene.step(1.0f/120.0f);
    }catch(const std::overflow_error&) {
        rejected=true;
    }
    require(rejected &&
            producer->last_frame().update.counts.previous_boxes==2u,
            "Dy overflow did not occur after a real source GPU fetch");
    require(scene.frame_index()==previous_frame &&
            scene.solved_result_frame()==previous_result &&
            contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside) &&
            scene.contact_edge_index({a,b})==previous_edge,
            "Failed postfetch Dy continuation published partial Sc state");
    const auto restored=scene.body_state(a);
    require(restored.position.x==before.position.x &&
            restored.angular_velocity.x==before.angular_velocity.x,
            "Postfetch Sc rollback did not preserve authored actor motion");

    scene.clear_spatial_velocity_mod(
        a,fyzz::ForceMode::VelocityChange,true);
    scene.set_angular_velocity(a,{});
    scene.step(1.0f/120.0f);
    require(producer->last_frame().update.counts.previous_boxes==0u &&
            producer->last_frame().update.counts.created==2u,
            "Failed GPU frame did not rebuild original root/SAP histories");
    require(!contains(scene.pairs(),a,b) &&
            contains(scene.lost_pairs(),a,b) &&
            contains(scene.pairs(),b,outside) &&
            scene.solved_result_frame()==scene.frame_index(),
            "Retry after GPU fetch failed to publish canonical Sc lost/contact history");
    scene.step(1.0f/120.0f);
    require(scene.lost_pairs().empty() &&
            !contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),b,outside),
            "Postfetch retry replayed a stale GPU aggregate lost report");
}


void verify_source_sap_report_overflow_recovery(
    fyzz::compute::Backend backend,const char* sap_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            8u,8u);
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.25f,1.0f,0.0f},1.0f);
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId x,fyzz::BodyId y) {
        const fyzz::native::Pair pair{std::min(x,y),std::max(x,y)};
        return std::find(pairs.begin(),pairs.end(),pair)!=pairs.end();
    };
    scene.advance_collision();
    require(scene.pairs().size()==1u && contains(scene.pairs(),a,b),
            "Source SAP overflow fixture has no completed Sc contact");
    const auto edge=scene.contact_edge_index({a,b});

    // This frame contains twenty-one new overlaps but only eight mapped
    // source report slots. The real GPU fetch must reject the entire Sc
    // publication without emitting any truncated found-pair set.
    std::vector<fyzz::BodyId> extra;
    extra.reserve(7u);
    for(std::uint32_t i=0;i<7u;++i)
        extra.push_back(scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20.0f,1.0f,0.0f},1.0f));
    bool overflow=false;
    try {
        scene.advance_collision();
    }catch(const std::overflow_error&) {
        overflow=true;
    }
    require(overflow && scene.pairs().size()==1u &&
            contains(scene.pairs(),a,b) &&
            scene.contact_edge_index({a,b})==edge,
            "GPU source overflow published a partial Sc contact history");

    for(std::size_t i=0;i<5u;++i)
        scene.remove(extra[i]);
    scene.advance_collision();
    require(producer->last_frame().update.counts.previous_boxes==0u &&
            producer->last_frame().update.counts.created==4u,
            "Source GPU overflow retry did not rebuild live SAP handles");
    require(scene.pairs().size()==2u &&
            contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),extra[5],extra[6]) &&
            scene.found_pairs().size()==1u &&
            contains(scene.found_pairs(),extra[5],extra[6]),
            "GPU overflow retry did not publish canonical Sc pairs");
    scene.advance_collision();
    require(scene.pairs().size()==2u &&
            scene.found_pairs().empty() && scene.lost_pairs().empty(),
            "Recovered source GPU SAP replayed stale overlap reports");
}


// A GPU aggregate self-report overflow occurs AFTER SAP's root report
// succeeds. Neither original aggregate counters nor Sc contacts can be
// partially consumed; retry must reconstruct the source member histories.
void verify_aggregate_report_overflow_recovery(
    fyzz::compute::Backend backend,
    const char* sap_manifest,const char* aggregate_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            8u,8u,
            fyzz::load_native_gpu_aggregate_shaders(backend,aggregate_manifest));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto root=scene.create_aggregate(5u,true);
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f,root);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.1f,1.0f,0.0f},1.0f,root);
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId x,fyzz::BodyId y) {
        const fyzz::native::Pair pair{std::min(x,y),std::max(x,y)};
        return std::find(pairs.begin(),pairs.end(),pair)!=pairs.end();
    };
    scene.advance_collision();
    require(scene.pairs().size()==1u && contains(scene.pairs(),a,b),
            "Original GPU aggregate overflow fixture has no Sc contact");
    const auto prior_edge=scene.contact_edge_index({a,b});
    std::vector<fyzz::BodyId> extra;
    for(std::uint32_t i=0;i<3u;++i)
        extra.push_back(scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.2f+0.1f*i,1.0f,0.0f},1.0f,root));
    bool rejected=false;
    try {
        scene.advance_collision();
    }catch(const std::overflow_error&) {
        rejected=true;
    }
    require(rejected && scene.pairs().size()==1u &&
            contains(scene.pairs(),a,b) &&
            scene.contact_edge_index({a,b})==prior_edge,
            "Aggregate GPU overflow published truncated Sc contact history");
    scene.remove(extra[1]);
    scene.remove(extra[2]);
    scene.advance_collision();
    require(producer->last_frame().update.counts.previous_boxes==0u &&
            producer->last_frame().update.counts.created==1u &&
            scene.pairs().size()==3u &&
            contains(scene.pairs(),a,b) &&
            contains(scene.pairs(),a,extra[0]) &&
            contains(scene.pairs(),b,extra[0]) &&
            scene.found_pairs().size()==2u,
            "Aggregate GPU report overflow retry lost source/Sc pair history");
    scene.advance_collision();
    require(scene.found_pairs().empty() && scene.lost_pairs().empty() &&
            scene.pairs().size()==3u,
            "Aggregate GPU overflow retry replayed stale reports");
}

void verify_retired_aggregate_stops_gpu_work(
    fyzz::compute::Backend backend,
    const char* sap_manifest,const char* aggregate_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            128u,128u,
            fyzz::load_native_gpu_aggregate_shaders(backend,aggregate_manifest));
    auto* producer=owner.get();
    scene.set_gpu_broadphase(std::move(owner));
    const auto root=scene.create_aggregate(1u,true);
    // An admitted root without member shapes is not a SAP endpoint.
    // After its initial descriptor reaches the GPU, repeated quiet frames
    // must not execute aggregate stages merely because the root exists.
    scene.advance_collision();
    scene.advance_collision();
    scene.advance_collision();
    require(!producer->last_frame().update.execute_gpu_stages,
            "Empty aggregate root forced a quiet source GPU update");
    const auto member=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f,root);
    const auto outside=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.25f,1.0f,0.0f});
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId x,fyzz::BodyId y) {
        const fyzz::native::Pair pair{std::min(x,y),std::max(x,y)};
        return std::find(pairs.begin(),pairs.end(),pair)!=pairs.end();
    };
    scene.advance_collision();
    require(contains(scene.pairs(),member,outside),
            "Source GPU aggregate fixture did not publish its initial pair");

    // The final root is deleted, but its source GPU descriptor and pair
    // history still require this frame's lost-report processing.
    scene.remove_actor_from_aggregate(root,member);
    scene.destroy_aggregate(root);
    scene.advance_collision();
    require(contains(scene.pairs(),member,outside),
            "Retiring the last aggregate lost a surviving standalone overlap");

    // A forced source history-convergence frame may still execute SAP.
    // Once it completes, a retired high-watermark must not keep the
    // aggregate kernels or SAP update executing forever.
    scene.advance_collision();
    scene.advance_collision();
    require(!producer->last_frame().update.execute_gpu_stages &&
            contains(scene.pairs(),member,outside),
            "Retired aggregate GPU slots forced a quiet SAP update");

    // Reusing the source aggregate ID after idle must reinitialize its
    // own projection history and publish genuine member contacts.
    const auto next=scene.create_aggregate(5u,true);
    const auto other=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.5f,1.0f,0.0f},1.0f,next);
    scene.advance_collision();
    require(contains(scene.pairs(),other,outside) &&
            contains(scene.pairs(),other,member),
            "Source aggregate ID reuse after idle lost GPU member contacts");
}

// Source actor and aggregate found/lost payloads share one mapped heap
// fetch without changing their separate original report classification.
void verify_combined_actor_aggregate_report_fetch(
    fyzz::compute::Backend backend,
    const char* sap_manifest,const char* aggregate_manifest) {
    fyzz::native::Scene scene;
    auto owner=std::make_unique<
        fyzz::physx_gpu::SourceScOwnedGpuBroadphaseNative>(
            backend,fyzz::load_native_gpu_sap_shaders(backend,sap_manifest),
            64u,64u,
            fyzz::load_native_gpu_aggregate_shaders(backend,aggregate_manifest));
    scene.set_gpu_broadphase(std::move(owner));
    const auto root=scene.create_aggregate(1u,true);
    const auto member=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f,root);
    const auto outside=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0.2f,1.0f,0.0f});
    const auto standalone_a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.0f,1.0f,0.0f},1.0f);
    const auto standalone_b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.25f,1.0f,0.0f},1.0f);
    const auto contains=[](std::span<const fyzz::native::Pair> pairs,
                           fyzz::BodyId x,fyzz::BodyId y) {
        const fyzz::native::Pair key{std::min(x,y),std::max(x,y)};
        return std::find(pairs.begin(),pairs.end(),key)!=pairs.end();
    };
    scene.advance_collision();
    require(scene.pairs().size()==2u && scene.found_pairs().size()==2u &&
            contains(scene.found_pairs(),member,outside) &&
            contains(scene.found_pairs(),standalone_a,standalone_b),
            "Combined original aggregate/actor GPU found reports lost a pair");
    scene.set_position(member,{4.0f,1.0f,0.0f});
    scene.set_position(standalone_b,{16.0f,1.0f,0.0f});
    scene.advance_collision();
    require(scene.pairs().empty() && scene.lost_pairs().size()==2u &&
            contains(scene.lost_pairs(),member,outside) &&
            contains(scene.lost_pairs(),standalone_a,standalone_b),
            "Combined original aggregate/actor GPU lost reports changed history");
    scene.advance_collision();
    require(scene.pairs().empty() && scene.found_pairs().empty() &&
            scene.lost_pairs().empty(),
            "Combined actor/aggregate report fetch replayed stale GPU reports");
}

void run(fyzz::compute::Backend backend,
         const char* sap_manifest,const char* aggregate_manifest) {
    bool rejected=false;
    try {
        (void)fyzz::load_native_gpu_aggregate_shaders(
            backend,sap_manifest);
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected,
        "Original SAP manifest was accepted as the aggregate source bundle");
    auto sap=fyzz::load_native_gpu_sap_shaders(backend,sap_manifest);
    auto aggregate=fyzz::load_native_gpu_aggregate_shaders(
        backend,aggregate_manifest);
    require(aggregate.size()==12u,
        "Original aggregate GPU bundle lost a required source stage");

    // The shader validation is performed before creating any original
    // PhysX GPU heap, source stream, or Sc scene.
    auto invalid=aggregate;
    invalid.back().bytes.clear();
    rejected=false;
    try {
        (void)fyzz::create_native_gpu_world(
            backend,sap,std::move(invalid));
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected,
        "Incomplete aggregate GPU shader set reached source Sc admission");

    // This constructs all twelve real backend pipelines in the same
    // Context as the thirty-four source SAP kernels. The ordinary
    // actor-only scene continues to use exactly the same Sc/PCM/Dy path.
    auto world=fyzz::create_native_gpu_world(
        backend,std::move(sap),std::move(aggregate),
        {fyzz::ExecutionMode::NativePortableGpu,true});
    const auto body=world.add_dynamic_box(
        {0.25f,0.25f,0.25f},{0.0f,2.0f,0.0f},1.0f);

    bool rejected_aggregate=false;
    try { (void)world.create_aggregate(0u,true); }
    catch(const std::invalid_argument&) {rejected_aggregate=true;}
    require(rejected_aggregate,"Original GPU World accepted zero-capacity aggregate");

    const auto root=world.create_aggregate(3u,true);
    const auto a=world.add_dynamic_box_to_aggregate(
        root,{0.5f,0.5f,0.5f},{5.0f,1.0f,0.0f},1.0f);
    const auto b=world.add_dynamic_box_to_aggregate(
        root,{0.5f,0.5f,0.5f},{5.4f,1.0f,0.0f},1.0f);
    // A source aggregate root requires a distinct AGGREGATE filter
    // group. Test it against an ordinary static box AND another
    // aggregate root as well as the within-root member collision.
    const auto outside=world.add_static_box(
        {0.5f,0.5f,0.5f},{6.0f,1.0f,0.0f});
    const auto root2=world.create_aggregate(1u,false);
    const auto other=world.add_static_box_to_aggregate(
        root2,{0.5f,0.5f,0.5f},{4.8f,1.0f,0.0f});
    // Only the original root IDs, not the members, enter SAP. GPU
    // aggregate member found-pairs must reach the same Sc contact path
    // for all three classes of overlap.
    world.step(1.0f/120.0f);
    const auto state=world.body_state(body);
    require(std::isfinite(state.position.y) && state.position.y<2.0f,
        "Aggregate-enabled source GPU World changed ordinary Sc body integration");
    const auto stateA=world.body_state(a);
    const auto stateB=world.body_state(b);
    require(std::isfinite(stateA.position.x) &&
            std::isfinite(stateB.position.x),
        "Original GPU aggregate member world states are non-finite");
    bool self_found=false,static_found=false,cross_aggregate_found=false;
    for(const auto& e:world.contact_events()) {
        if(e.kind!=fyzz::ContactEventKind::Found)continue;
        if(e.body0==a && e.body1==b)self_found=true;
        if(e.body0==b && e.body1==outside)static_found=true;
        if(e.body0==a && e.body1==other)cross_aggregate_found=true;
    }
    require(self_found && static_found && cross_aggregate_found,
        "Original GPU aggregate self/root-static/root-root contacts were not published to Sc");

    // Change an admitted aggregate rigid member's Bp motion group
    // across two real GPU fetches, not just within a host-only fixture.
    // The Sc actor and aggregate membership survive both transitions.
    world.set_body_kinematic(a,true);
    world.step(1.0f/120.0f);
    const auto kinematic_state=world.body_state(a);
    require(std::isfinite(kinematic_state.position.x),
        "Original aggregate member kinematic GPU publication was not finite");
    world.set_body_kinematic(a,false);
    world.step(1.0f/120.0f);
    require(std::isfinite(world.body_state(a).position.x),
        "Original aggregate member dynamic GPU reinsertion failed");

    // PhysX removeActor returns the existing actor to the scene.
    // It retains its public BodyId while its old aggregate ElementSim
    // retires and its new standalone SAP endpoint produces GPU pairs.
    world.remove_actor_from_aggregate(root,b);
    world.step(1.0f/120.0f);
    require(std::isfinite(world.body_state(b).position.x) &&
            std::isfinite(world.body_state(a).position.x),
        "Removed aggregate actor did not survive independent GPU reinsertion");

    // PxAggregate::release returns all remaining members to the scene,
    // including actors of another aggregate. No member is destroyed.
    world.destroy_aggregate(root);
    world.destroy_aggregate(root2);
    world.step(1.0f/120.0f);
    require(std::isfinite(world.body_state(a).position.x) &&
            std::isfinite(world.body_state(b).position.x) &&
            std::isfinite(world.body_state(other).position.x),
        "Source aggregate release destroyed or orphaned live Sc actors");
    world.remove_body(b);
    world.remove_body(a);
    world.remove_body(other);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)
            throw std::invalid_argument(
                "Expected source SAP and aggregate GPU shader manifests");
#if defined(FYZZXYZZY_TEST_D3D12)
        run(fyzz::compute::Backend::D3D12,argv[1],argv[2]);
#else
        run(fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
#endif
#if defined(FYZZXYZZY_TEST_D3D12)
        verify_aggregate_gpu_rollback(
            fyzz::compute::Backend::D3D12,argv[1],argv[2]);
        verify_aggregate_postfetch_sc_rollback(
            fyzz::compute::Backend::D3D12,argv[1],argv[2]);
        verify_source_sap_report_overflow_recovery(
            fyzz::compute::Backend::D3D12,argv[1]);
        verify_aggregate_report_overflow_recovery(
            fyzz::compute::Backend::D3D12,argv[1],argv[2]);
        verify_combined_actor_aggregate_report_fetch(
            fyzz::compute::Backend::D3D12,argv[1],argv[2]);
        verify_retired_aggregate_stops_gpu_work(
            fyzz::compute::Backend::D3D12,argv[1],argv[2]);
#else
        verify_aggregate_gpu_rollback(
            fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
        verify_aggregate_postfetch_sc_rollback(
            fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
        verify_source_sap_report_overflow_recovery(
            fyzz::compute::Backend::Vulkan,argv[1]);
        verify_aggregate_report_overflow_recovery(
            fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
        verify_combined_actor_aggregate_report_fetch(
            fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
        verify_retired_aggregate_stops_gpu_work(
            fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
#endif
        std::cout<<"Original source aggregate GPU World stage ownership PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original source aggregate GPU World stage ownership FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
