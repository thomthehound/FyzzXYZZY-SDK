// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/native_gpu_world.h"
#include "native/scene.h"
#include "physx_gpu/source_sc_owned_gpu_broadphase_native.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

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

    bool live_destroy_rejected=false;
    try {world.destroy_aggregate(root);}
    catch(const std::logic_error&) {live_destroy_rejected=true;}
    require(live_destroy_rejected,
        "Native GPU World destroyed a root with attached Sc members");
    world.remove_body(b);
    world.remove_body(a);
    world.destroy_aggregate(root);
    world.remove_body(other);
    world.destroy_aggregate(root2);
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
#else
        verify_aggregate_gpu_rollback(
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
