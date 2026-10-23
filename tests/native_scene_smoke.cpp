// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
void expect(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
bool has(std::span<const fyzz::native::Pair> p, fyzz::BodyId a, fyzz::BodyId b) {
    const fyzz::native::Pair match{std::min(a,b), std::max(a,b)};
    return std::find(p.begin(), p.end(), match) != p.end();
}
}
// PxRigidDynamic::setGlobalPose(autowake=true) resets sleep state even
// when the authored translation equals the current pose. Geometry-only
// caches and manager handles must remain untouched by that no-op edit.
void verify_idempotent_pose_autowake() {
    using fyzz::native::Scene;
    {
        Scene connected;
        const auto floor=connected.add_static_floor(0.0f);
        const auto left=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            connected.step(1.0f/120.0f);
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right) &&
               connected.active_accurate_nodes().empty() &&
               connected.active_speculative_nodes().empty(),
               "Same-pose autowake fixture never reached a dormant island");
        const auto before=connected.body_state(left);
        const auto original_edge=connected.contact_edge_index({left,right});
        const auto rebuilds=connected.island_full_rebuild_count();
        const auto revision=connected.island_topology_revision();
        const auto manager_count=connected.contacts().size();
        connected.set_position(left,before.position);
        expect(!connected.body_is_sleeping(left) &&
               !connected.body_ready_for_sleep(left) &&
               connected.body_wake_counter(left)>0.0f &&
               connected.body_state(left).position.x==before.position.x &&
               connected.body_state(left).position.y==before.position.y &&
               connected.body_state(left).position.z==before.position.z,
               "Same-position dynamic pose authoring skipped PhysX autowake");
        connected.advance_collision();
        expect(!connected.body_is_sleeping(left) &&
               !connected.body_is_sleeping(right) &&
               connected.active_accurate_nodes().size()==2 &&
               connected.active_speculative_nodes().size()==2 &&
               connected.island_full_rebuild_count()==rebuilds &&
               connected.island_topology_revision()==revision &&
               connected.contact_edge_index({left,right})==original_edge &&
               connected.contacts().size()==manager_count &&
               connected.found_pairs().empty() &&
               connected.lost_pairs().empty() &&
               connected.found_contacts().empty() &&
               connected.lost_contacts().empty(),
               "Same-pose autowake rebuilt unchanged collision topology");
        // A matching static pose must not wake its actual contacting
        // sleepers; the automatic wake is specific to dynamic actors.
        for(int frame=0;frame<360;++frame)
            connected.step(1.0f/120.0f);
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right),
               "Static same-pose fixture never returned to sleep");
        const auto quiet_revision=connected.island_topology_revision();
        connected.set_position(floor,connected.body_state(floor).position);
        connected.advance_collision();
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right) &&
               connected.island_topology_revision()==quiet_revision,
               "Same-position static pose changed contacting sleep state");
    }
    {
        // Source Dy::sleepCheck can mark a body ready one frame before
        // the third pass deactivates it. The same-pose write must reset
        // that readiness as well as waking bodies already asleep.
        Scene ready;
        ready.add_static_floor(0.0f);
        const auto body=ready.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool reached_ready=false;
        for(int frame=0;frame<600;++frame) {
            ready.step(1.0f/120.0f);
            if(ready.body_ready_for_sleep(body)) {
                reached_ready=true;
                break;
            }
        }
        expect(reached_ready && !ready.body_is_sleeping(body),
               "Same-pose readiness fixture did not reach the source boundary");
        const auto old=ready.body_state(body);
        ready.set_position(body,old.position);
        expect(!ready.body_ready_for_sleep(body) &&
               !ready.body_is_sleeping(body) &&
               ready.body_wake_counter(body)>0.0f,
               "Same-position authoring retained stale Dy sleep readiness");
        ready.step(1.0f/120.0f);
        expect(!ready.body_is_sleeping(body),
               "Same-position wake was lost at the next source third pass");
    }
}
// PxsIslandSim::deactivateNodeInternal emits eCONTACT_MANAGER edges
// only when the final awake endpoint sleeps. Static world connections
// count; destroyed edges never become sleep-deactivation events.
void verify_source_contact_edge_deactivations() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene;
    const auto floor=scene.add_static_floor(0.0f);
    const auto left=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1);
    const auto right=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1);
    scene.advance_collision();
    const auto lr=scene.contact_edge_index(Pair{left,right});
    const auto fl=scene.contact_edge_index(Pair{floor,left});
    const auto fr=scene.contact_edge_index(Pair{floor,right});
    expect(lr!=std::numeric_limits<std::uint32_t>::max() &&
           fl!=std::numeric_limits<std::uint32_t>::max() &&
           fr!=std::numeric_limits<std::uint32_t>::max() &&
           scene.deactivating_accurate_contact_edges().empty() &&
           scene.deactivating_speculative_contact_edges().empty(),
           "Contact edges deactivated before rigid sleep");
    bool entered_sleep=false;
    for(int frame=0;frame<600;++frame) {
        scene.step(1.0f/120.0f);
        if(scene.body_is_sleeping(left) &&
           scene.body_is_sleeping(right)) {
            entered_sleep=true;
            break;
        }
    }
    expect(entered_sleep,"Source contact-edge fixture never slept");
    for(const auto edges:{
        scene.deactivating_accurate_contact_edges(),
        scene.deactivating_speculative_contact_edges()}) {
        expect(edges.size()==3 &&
               std::count(edges.begin(),edges.end(),lr)==1 &&
               std::count(edges.begin(),edges.end(),fl)==1 &&
               std::count(edges.begin(),edges.end(),fr)==1,
               "Rigid sleep did not deactivate each contact edge once");
    }
    expect(scene.contact_edge_connected(Pair{left,right}) &&
           scene.contact_edge_connected(Pair{floor,left}) &&
           scene.contact_edge_connected(Pair{floor,right}),
           "Rigid sleep incorrectly retired persistent contact managers");
    scene.advance_collision();
    expect(scene.deactivating_accurate_contact_edges().empty() &&
           scene.deactivating_speculative_contact_edges().empty(),
           "Quiet source publication replayed edge deactivations");
    scene.set_position(left,scene.body_state(left).position);
    scene.advance_collision();
    expect(!scene.body_is_sleeping(left) &&
           !scene.body_is_sleeping(right) &&
           scene.deactivating_accurate_contact_edges().empty(),
           "Authored wake replayed stale edge deactivations");
    Scene lost;
    const auto a=lost.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,12,0},1);
    const auto b=lost.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,12,0},1);
    lost.advance_collision();
    lost.set_position(b,{5,12,0});
    lost.advance_collision();
    expect(has(lost.lost_contacts(),a,b) &&
           lost.deactivating_accurate_contact_edges().empty() &&
           lost.deactivating_speculative_contact_edges().empty(),
           "Lost contact edge incorrectly entered sleep output");
}

// NpRigidDynamic::setGlobalPose refreshes an awake body's wake counter
// without clearing DySleep's accumulated low-motion samples. Use the
// existing setLinearVelocity autowake path as a source-compatible control,
// with an unchanged velocity to preserve the same physical state.
void verify_active_pose_preserves_sleep_filter() {
    using fyzz::native::Scene;
    Scene authored, control;
    const auto populate=[](Scene& scene) {
        const auto floor=scene.add_static_floor(0.0f);
        const auto body=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material material{};
        material.disable_friction=true;
        scene.set_material(floor,material);
        scene.set_material(body,material);
        scene.set_velocity(body,{0.006f,0,0});
        return body;
    };
    const auto a=populate(authored);
    const auto b=populate(control);
    constexpr float dt=1.0f/120.0f;
    // Once the source wake counter is below 0.2 seconds, DySleep is
    // integrating small but nonzero lateral velocities into its filter.
    // Preserve those samples through a same-pose authoring operation.
    for(int frame=0;frame<40;++frame) {
        authored.step(dt);
        control.step(dt);
    }
    const auto before=authored.body_state(a);
    const auto control_before=control.body_state(b);
    expect(!before.sleeping && !control_before.sleeping &&
           authored.body_wake_counter(a)>0.0f &&
           authored.body_wake_counter(a)<0.2f &&
           authored.body_wake_counter(a)==control.body_wake_counter(b) &&
           before.velocity.x>0.005f &&
           before.velocity.x==control_before.velocity.x,
           "Active pose fixture did not accumulate low-motion sleep samples");
    authored.set_position(a,before.position);
    control.set_velocity(b,control_before.velocity);
    expect(authored.body_wake_counter(a)==control.body_wake_counter(b) &&
           !authored.body_ready_for_sleep(a),
           "Active pose write did not refresh the source wake counter");
    bool renewed=false;
    float last_control_counter=control.body_wake_counter(b);
    for(int frame=0;frame<42;++frame) {
        authored.step(dt);
        control.step(dt);
        const float current=control.body_wake_counter(b);
        renewed |= current>last_control_counter+0.02f;
        last_control_counter=current;
        expect(authored.body_wake_counter(a)==current &&
               authored.body_ready_for_sleep(a)==control.body_ready_for_sleep(b) &&
               authored.body_is_sleeping(a)==control.body_is_sleeping(b),
               "Active pose write discarded source DySleep accumulator history");
    }
    expect(renewed,
           "Active pose filter fixture did not exercise accumulated-energy renewal");
}
// Keep multi-island lifetime regressions in separate stack frames. The
// native scene smoke runner contains many Scene fixtures; combining all
// of their local allocations into one MSVC frame can exhaust the
// default Windows stack even though the individual tests are small.
void verify_simultaneous_initial_sleep_compaction() {
    using fyzz::native::Scene;
    // Multiple initially active rigid nodes may enter the third-pass
    // deactivation list together while a new rigid is being admitted.
    // markInactive relocates each initial member to the initial-prefix
    // tail, then removes it using ordinary swap-last. The traversal
    // must recheck the ORIGINAL compacted position, including any
    // other dormant initial member moved there; otherwise [A, B, Cnew]
    // can leave dormant B in mActiveNodes and misreport the suffix.
    // Exercise both numerical and collision-only publication paths.
    for(const bool query_only : {false,true}) {
        Scene multi_sleep;
        multi_sleep.add_static_floor(0.0f);
        const auto first=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto second=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
        multi_sleep.advance_collision();
        expect(multi_sleep.active_accurate_nodes().size()==2 &&
               multi_sleep.accurate_active_node_index(first)==0 &&
               multi_sleep.accurate_active_node_index(second)==1,
               "Multi-sleep source fixture did not index two active islands");
        bool both_ready=false;
        for(int frame=0;frame<600;++frame) {
            multi_sleep.step(1.0f/120.0f);
            if(multi_sleep.body_ready_for_sleep(first) &&
               multi_sleep.body_ready_for_sleep(second)) {
                both_ready=true;
                break;
            }
        }
        expect(both_ready && !multi_sleep.body_is_sleeping(first) &&
               !multi_sleep.body_is_sleeping(second),
               "Multi-sleep source fixture missed shared readiness boundary");
        const auto newcomer=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,10,0},1.0f);
        if(query_only)
            multi_sleep.advance_collision();
        else
            multi_sleep.step(1.0f/120.0f);
        const auto active=multi_sleep.active_accurate_nodes();
        const auto fresh=multi_sleep.newly_activated_accurate_nodes();
        const auto retired=multi_sleep.deactivating_accurate_nodes();
        expect(multi_sleep.body_is_sleeping(first) &&
               multi_sleep.body_is_sleeping(second) &&
               active.size()==1 && active[0]==newcomer &&
               fresh.size()==1 && fresh[0]==newcomer &&
               retired.size()==2 &&
               std::find(retired.begin(),retired.end(),first)!=retired.end() &&
               std::find(retired.begin(),retired.end(),second)!=retired.end() &&
               multi_sleep.accurate_active_node_index(first)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.accurate_active_node_index(second)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.accurate_active_node_index(newcomer)==0,
               "Source initial-prefix compaction skipped another dormant body");
        // The wake path must not replay the stale sleeper or lose the
        // surviving initial active node when it starts a new pass.
        multi_sleep.set_velocity(first,{0.25f,0,0});
        multi_sleep.advance_collision();
        const auto reactivated=multi_sleep.active_accurate_nodes();
        expect(reactivated.size()==2 &&
               std::find(reactivated.begin(),reactivated.end(),first)!=
                   reactivated.end() &&
               std::find(reactivated.begin(),reactivated.end(),newcomer)!=
                   reactivated.end() &&
               multi_sleep.accurate_active_node_index(second)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.newly_activated_accurate_nodes().size()==1 &&
               multi_sleep.newly_activated_accurate_nodes()[0]==first,
               "Multi-sleep compaction corrupted later authored wake suffix");
    }

}
// Independent speculative IslandSim active indices must follow the
// original wakeIslands/markInactive initial-prefix contract, including
// quiet authored wakes, actor admission and failed graph reconstruction.
void verify_speculative_active_node_lifecycle() {
    using fyzz::native::Scene;
    Scene scene;
    scene.add_static_floor(0.0f);
    const auto first=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto sleeper=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
    scene.advance_collision();
    expect(scene.active_speculative_nodes().size()==2 &&
           scene.newly_activated_speculative_nodes().size()==2 &&
           scene.speculative_active_node_index(first)==0 &&
           scene.speculative_active_node_index(sleeper)==1,
           "Speculative IslandSim did not index initial dynamic actors");
    for(int i=0;i<360;++i)scene.step(1.0f/120.0f);
    expect(scene.body_is_sleeping(first) && scene.body_is_sleeping(sleeper) &&
           scene.active_speculative_nodes().empty() &&
           scene.active_accurate_nodes().empty() &&
           scene.speculative_active_node_index(first)==
               std::numeric_limits<std::size_t>::max() &&
           scene.speculative_active_node_index(sleeper)==
               std::numeric_limits<std::size_t>::max(),
           "Speculative graph retained source-deactivated rigid nodes");

    scene.set_velocity(first,{0.2f,0,0});
    scene.advance_collision();
    const auto waking=scene.newly_activated_speculative_nodes();
    expect(scene.active_speculative_nodes().size()==1 &&
           waking.size()==1 && waking[0]==first &&
           scene.speculative_active_node_index(first)==0 &&
           scene.speculative_active_node_index(sleeper)==
               std::numeric_limits<std::size_t>::max(),
           "Quiet authored wake failed speculative active-index publication");
    scene.advance_collision();
    expect(scene.newly_activated_speculative_nodes().empty() &&
           scene.active_speculative_nodes().size()==1,
           "Quiet collision query replayed speculative activation suffix");

    const auto admitted=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{20,5,0},1.0f);
    scene.advance_collision();
    const auto suffix=scene.newly_activated_speculative_nodes();
    expect(scene.active_speculative_nodes().size()==2 &&
           suffix.size()==1 && suffix[0]==admitted &&
           scene.speculative_active_node_index(admitted)==1 &&
           scene.newly_activated_accurate_nodes().size()==1,
           "Speculative active prefix was lost during actor admission");

    const auto published=scene.active_speculative_nodes();
    const auto old_frame=scene.frame_index();
    const auto candidate=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{30,5,0},1.0f);
    bool failed=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {failed=true;}
    const auto restored=scene.active_speculative_nodes();
    expect(failed && scene.frame_index()==old_frame &&
           restored.data()==published.data() &&
           restored.size()==published.size() &&
           restored[0]==first && restored[1]==admitted &&
           scene.speculative_active_node_index(candidate)==
               std::numeric_limits<std::size_t>::max(),
           "Rejected graph admission replaced published speculative active span");
    scene.step(1.0f/120.0f);
    expect(scene.active_speculative_nodes().size()==3 &&
           scene.speculative_active_node_index(candidate)<
               scene.active_speculative_nodes().size(),
           "Speculative IslandSim could not admit actor after rejected frame");
    scene.remove(first);
    scene.advance_collision();
    expect(scene.active_speculative_nodes().size()==2 &&
           scene.speculative_active_node_index(first)==
               std::numeric_limits<std::size_t>::max(),
           "Speculative graph kept a retired rigid in its active list");
}

// Two orphan components can share one original indexed-manager array.
// PhysX processes their dirty nodes in ascending mDirtyMap index,
// NOT in the order of the smallest BodyId in each new component. These
// orders disagree: right G is dirty ID 3; left B is dirty ID 5, while
// left A has smaller BodyId 0 than right F's BodyId 2. Swap-last
// compaction therefore moves F-G before A-B out of the old root array.
void verify_dirty_order_multisplit_manager_remainder() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene;
    // Form all three islands separately first. The central C-D-E
    // source root must acquire size three before the two size-two
    // arms connect, otherwise mergeIslands could retain an arm root.
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-5.0f,10,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto f=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{6.2f,10,0},1);
    const auto g=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5.4f,10,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-4.2f,10,0},1);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==3 &&
           scene.contacts().size()==4 &&
           scene.accurate_island_root(c)==d,
           "Two-way split fixture did not establish its larger central root");
    scene.set_position(a,{-1.6f,10,0});
    scene.set_position(b,{-0.8f,10,0});
    scene.set_position(f,{3.2f,10,0});
    scene.set_position(g,{2.4f,10,0});
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.accurate_island_root(a)==d &&
           scene.accurate_island_root(g)==d &&
           has(scene.found_contacts(),b,c) &&
           has(scene.found_contacts(),g,e),
           "Two-way split fixture lost its central source root during merges");
    // Append root-owned static managers only *after* both arm arrays
    // have joined the central island. Their suffix is essential:
    // removing F-G before A-B swaps these later root managers into
    // a different order than removing A-B before F-G.
    const auto sc=scene.add_static_box(
        {0.1f,0.1f,0.1f},{0,10,0.54f});
    const auto sd=scene.add_static_box(
        {0.1f,0.1f,0.1f},{0.8f,10,0.54f});
    const auto se=scene.add_static_box(
        {0.1f,0.1f,0.1f},{1.6f,10,0.54f});
    scene.advance_collision();
    scene.step(1.0f/480.0f);
    const auto initial_managers=scene.indexed_contact_manager_order();
    const auto initial_gu=scene.contacts();
    expect(scene.accurate_island_count()==1 &&
           initial_gu.size()==9 && initial_managers.size()==9,
           "Dirty-ordered two-way split fixture lacked nine managers");
    std::vector<Pair> root_array;
    for(std::size_t index:initial_managers)
        root_array.push_back(initial_gu[index].pair);

    // Source lost-bridge publication occurs before processLostEdges,
    // in the deterministic Gu manager stream. Thereafter the right
    // dirty G component splits first, then left dirty B.
    const auto swap_remove=[&](Pair pair) {
        const auto found=std::find(root_array.begin(),root_array.end(),pair);
        expect(found!=root_array.end(),
               "Source split oracle missing its old indexed manager");
        *found=root_array.back();
        root_array.pop_back();
    };
    swap_remove(Pair{c,b});
    swap_remove(Pair{g,e});
    swap_remove(Pair{f,g});
    swap_remove(Pair{a,b});
    std::vector<Pair> expected;
    for(const Pair pair:root_array)
        if(pair==Pair{c,d} || pair==Pair{d,e} ||
           pair==Pair{c,sc} || pair==Pair{d,sd} ||
           pair==Pair{e,se})
            expected.push_back(pair);
    expect(expected.size()==5,
           "Dirty-order source oracle did not retain five root managers");

    scene.set_position(a,{-2.5f,10,0});
    scene.set_position(b,{-1.7f,10,0});
    scene.set_position(f,{4.1f,10,0});
    scene.set_position(g,{3.3f,10,0});
    scene.step(1.0f/480.0f);
    expect(scene.accurate_island_count()==3 &&
           has(scene.lost_contacts(),b,c) &&
           has(scene.lost_contacts(),g,e) &&
           scene.contacts().size()==7,
           "Dirty-order fixture did not detach both orphan components");
    const auto gu=scene.contacts();
    const auto managers=scene.indexed_contact_manager_order();
    std::vector<Pair> retained;
    for(std::size_t index:managers) {
        const Pair pair=gu[index].pair;
        if(pair==Pair{c,d} || pair==Pair{d,e} ||
           pair==Pair{c,sc} || pair==Pair{d,sd} ||
           pair==Pair{e,se})
            retained.push_back(pair);
    }
    expect(retained==expected,
           "Source mDirtyMap multi-split order changed retained manager sequence");
}

// Unequal original mHopCounts must change *which branch is popped*
// before the other branch. Starting at dirty B, the source EdgeInstance
// head discovers A before X, but a prior successful lost-chord repair
// gives X a finite route estimate and leaves A without one. The queue
// therefore expands X-H before A-G; FIFO would reverse that ordering.
// PhysX addNode starts hop counts at zero, then processNewEdges
// adjusts them through source mergeIslandsInternal. Here X joins the
// root island through C and has fewer hops than A, which later merges
// through B. The older B-X edge is behind newer B-A in B's adjacency;
// when C-X and B-C vanish together, the hop-priority queue expands X
// before A despite having *discovered* A first. The discovered child H
// must then precede G in the source mVisitedNodes migration list.
// FIFO would reverse their manager order.
void verify_hop_priority_split_migration() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene split;
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.8f,10,0},1);
    const auto x=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.35f,10.65f,0},1);
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-3.2f,9.6f,0},1);
    const auto g=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-2.4f,9.6f,0},1);
    const auto h=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-1.1f,11.6f,0},1);
    const auto sg=split.add_static_box(
        {0.1f,0.1f,0.1f},{-2.4f,9.6f,0.54f});
    const auto sh=split.add_static_box(
        {0.1f,0.1f,0.1f},{-1.1f,11.6f,0.54f});
    split.advance_collision();
    expect(has(split.found_contacts(),c,x) &&
           has(split.found_contacts(),b,x) &&
           has(split.found_contacts(),x,h) &&
           has(split.found_contacts(),a,g) &&
           has(split.found_contacts(),g,sg) &&
           has(split.found_contacts(),h,sh) &&
           split.accurate_island_count()==2 &&
           split.accurate_cached_route_hops(x)>0 &&
           split.accurate_cached_route_hops(a)>0,
           "Hop-priority fixture did not establish source merge estimates");

    // A-G is the smaller island: merging it at the newer B-A manager
    // gives A the original mergeIslandsInternal endpoint hop B+1.
    // X remains connected by the older B-X and C-X edges.
    split.set_position(a,{-1.6f,9.6f,0});
    split.advance_collision();
    expect(has(split.found_contacts(),a,b) &&
           split.accurate_island_count()==1 &&
           split.accurate_cached_route_hops(x)<
               split.accurate_cached_route_hops(a),
           "Hop-priority fixture did not establish unequal source merge hops");

    // Lose C-X and B-C in one publication. An earlier independent
    // lost-chord repair would raise X's estimate to B+1, obscuring the
    // valid shorter source estimate that existed when both edges broke.
    split.set_position(x,{-1.1f,10.8f,0});
    split.set_position(c,{1.5f,10,0});
    split.set_position(d,{2.3f,10,0});
    split.set_position(e,{3.1f,10,0});
    split.step(1.0f/480.0f);
    const auto gu=split.contacts();
    const auto managers=split.indexed_contact_manager_order();
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),b,c) &&
           has(split.lost_contacts(),c,x) &&
           !has(split.lost_contacts(),b,x) &&
           !has(split.lost_contacts(),x,h) &&
           gu.size()==managers.size(),
           "Hop-priority fixture did not detach its original branched island");
    std::size_t h_position=managers.size();
    std::size_t g_position=managers.size();
    for(std::size_t i=0;i<managers.size();++i) {
        const Pair pair=gu[managers[i]].pair;
        if(pair==Pair{h,sh})h_position=i;
        if(pair==Pair{g,sg})g_position=i;
    }
    expect(h_position<g_position && g_position<managers.size(),
           "Split manager migration ignored source unequal merge-hop priorities");
}

// CmPriorityQueue::pop favors the left child on equal priorities.
// With five equal-hop branches, it expands the first, second, fourth,
// fifth and third queued branches (not FIFO). Each branch owns a leaf
// with a static contact manager: their migration order makes this heap
// detail observable on the actual Dy indexed constraint input.
void verify_equal_hop_split_queue_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    using fyzz::Vec3;
    Scene split;
    const Vec3 origin{0,10,0};
    // Admit root-anchor first so the two-node original root survives
    // the star's later one-node merges. Losing root-center then forces
    // a real dirty-route split, rather than visiting an already-rooted
    // star whose dirty seed is trivially its root.
    const auto root=split.add_dynamic_box(
        {0.4f,0.4f,0.4f},{0,10,-1.1f},1);
    const auto anchor=split.add_dynamic_box(
        {0.4f,0.4f,0.4f},{0,10,-1.75f},1);
    const auto center=split.add_dynamic_box(
        {0.75f,0.75f,0.75f},origin,1);
    const std::array<Vec3,5> axes{{
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1}}};
    std::array<fyzz::BodyId,5> arms{},leaves{},markers{};
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        arms[i]=split.add_dynamic_box(
            {0.4f,0.4f,0.4f},
            {a.x*1.1f,10+a.y*1.1f,a.z*1.1f},1);
    }
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        leaves[i]=split.add_dynamic_box(
            {0.3f,0.3f,0.3f},
            {a.x*1.7f,10+a.y*1.7f,a.z*1.7f},1);
    }
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        markers[i]=split.add_static_box(
            {0.08f,0.08f,0.08f},
            {a.x*1.96f,10+a.y*1.96f,a.z*1.96f});
    }
    split.advance_collision();
    expect(split.accurate_island_count()==1 &&
           has(split.found_contacts(),root,center) &&
           has(split.found_contacts(),root,anchor) &&
           split.accurate_island_root(center)==anchor,
           "Equal-hop source fixture did not preserve the original root");
    for(std::size_t i=0;i<axes.size();++i) {
        expect(has(split.found_contacts(),center,arms[i]) &&
               has(split.found_contacts(),arms[i],leaves[i]) &&
               has(split.found_contacts(),leaves[i],markers[i]),
               "Equal-hop fixture lacked an isolated marked branch");
    }
    // All five child islands joined their parent through the same
    // hop depth. No newer redundant chords should alter the estimates.
    const auto hop=split.accurate_cached_route_hops(arms[0]);
    for(auto arm:arms)
        expect(split.accurate_cached_route_hops(arm)==hop,
               "Equal-hop priority fixture has unequal source estimates");

    split.set_position(root,{0,10,-1.4f});
    split.set_position(anchor,{0,10,-2.05f});
    split.step(1.0f/480.0f);
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),root,center) &&
           split.accurate_island_root(center)==center,
           "Equal-hop queue fixture did not separate the dirty star");
    const auto gu=split.contacts();
    const auto indices=split.indexed_contact_manager_order();
    std::vector<fyzz::BodyId> visited_leaves;
    for(auto index:indices) {
        const auto pair=gu[index].pair;
        for(std::size_t i=0;i<axes.size();++i)
            if(pair==Pair{leaves[i],markers[i]})
                visited_leaves.push_back(leaves[i]);
    }
    // Center's intrusive adjacency enumerates arms in reverse insertion
    // order: 4,3,2,1,0. Source CmPriorityQueue pops the five equal-hop
    // branches in order 4,3,1,0,2, so their grandchildren and marker
    // managers migrate in that order. FIFO/tie-by-discovery differs.
    const std::vector<fyzz::BodyId> expected{
        leaves[4],leaves[3],leaves[1],leaves[0],leaves[2]};
    expect(visited_leaves==expected,
           "Equal-hop source heap tie order changed split manager migration");
}

// A root-bridge loss and an internal orphan chord loss occur in the
// same publication. PhysX processLostEdges splits on the first failed
// dirty route, then *skips* other dirty nodes already present in that
// split's shared mVisitedState. The initial findRoute visitation supplies
// the new node/manager ordering directly; it must not traverse twice.
void verify_source_dirty_witness_split() {
    using fyzz::native::Scene;
    Scene scene;
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,16,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,16,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,16,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.4f,16,0},1);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.55f,16,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.speculative_island_count()==1 &&
           scene.accurate_island_root(c)==b &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),b,c) &&
           has(scene.found_contacts(),c,d) &&
           has(scene.found_contacts(),c,e) &&
           has(scene.found_contacts(),d,e),
           "Dirty witness split fixture lacks its pre-loss root and chord");
    const auto old_handle=scene.accurate_island_id(b);
    const auto checks=scene.island_dirty_route_check_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    const auto splits=scene.island_local_split_count();
    // Preserve the A-B root island, but disconnect C-D-E at B-C.
    // Remove C-E while preserving C-D-E's alternate route.
    scene.set_position(a,{-2,16,0});
    scene.set_position(b,{-1.2f,16,0});
    scene.set_position(e,{3.35f,16,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),b,c) &&
           has(scene.lost_contacts(),c,e) &&
           !has(scene.lost_contacts(),d,e) &&
           scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==2 &&
           scene.accurate_island_id(b)==old_handle &&
           scene.accurate_island_root(b)==b &&
           scene.accurate_island_root(e)==c &&
           scene.accurate_island_id(c)!=old_handle &&
           scene.accurate_island_id(c)==scene.accurate_island_id(e) &&
           scene.accurate_cached_route_hops(c)==0 &&
           scene.accurate_cached_route_hops(d)==1 &&
           scene.accurate_cached_route_hops(e)==2 &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()==splits+2,
           "Immediate source dirty split lost root, routes or handle");
    // Dirty bitmap B,C,E per graph. B is old root; C splits all of
    // C,D,E. E has already been witnessed and should *not* be searched.
    // A deferred split or a second island-wide traversal does not
    // implement this source third-pass contract.
    expect(scene.island_dirty_route_check_count()==checks+4,
           "Dirty orphan witnesses were re-searched after their split");
    scene.advance_collision();
    expect(scene.accurate_island_count()==2 &&
           scene.island_local_split_count()==splits+2 &&
           scene.island_full_rebuild_count()==rebuilds,
           "Quiet publication replayed the source dirty split");
}

// After two redundant chords disappear, the first dirty-node search
// reaches its original root through C while also visiting later-dirty D.
// Source unwindRoute repairs the A-C-B path, then processLostEdges
// updates D's off-path estimate via A and treats D as an established
// witness instead of searching D a second time.
void verify_successful_route_branch_witness() {
    using fyzz::native::Scene;
    Scene scene;
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,18,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.4f,18,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,18,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.4f,18,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.accurate_island_root(d)==b &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),b,d) &&
           has(scene.found_contacts(),a,d) &&
           has(scene.found_contacts(),a,c) &&
           has(scene.found_contacts(),b,c),
           "Shared-witness fixture lacks its redundant root chords");
    const auto checks=scene.island_dirty_route_check_count();
    const auto splits=scene.island_local_split_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    scene.set_position(b,{1.1f,18,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),a,b) &&
           has(scene.lost_contacts(),b,d) &&
           !has(scene.lost_contacts(),b,c) &&
           scene.accurate_island_count()==1 &&
           scene.speculative_island_count()==1 &&
           scene.accurate_island_root(d)==b &&
           scene.accurate_cached_route_hops(c)==1 &&
           scene.accurate_cached_route_hops(a)==2 &&
           scene.accurate_cached_route_hops(d)==3 &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()==splits,
           "Source successful route failed to update off-path witnesses");
    // A and B are processed in each graph; D was already reached in
    // A's heap search and is skipped by the source visited bitmap.
    expect(scene.island_dirty_route_check_count()==checks+4,
           "Successful source search repeated a witnessed dirty node");
}

// A split in one island must not abort the rest of either IslandSim's
// dirty bitmap. Replacing a temporary root chord in an unrelated island
// requires actual source findRoute repair, rather than just retaining the
// same island handle after the bounded component reconstruction.
void verify_route_repair_continues_after_split() {
    using fyzz::native::Scene;
    Scene scene;
    // The initial two-node island chooses B (the second node) as its
    // PhysX merge-tie root. Losing A-B makes the earlier dirty A fail.
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,14,0},1.0f);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,14,0},1.0f);
    // A separate chain C-D-E-F has source root D. Its eventual D-F
    // shortcut sets F's hop to one; removing D-F leaves F-E-D intact,
    // requiring its cached hop to increase back to two.
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10,14,0},1.0f);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.8f,14,0},1.0f);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{11.6f,14,0},1.0f);
    const auto f=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{12.4f,14,0},1.0f);
    scene.advance_collision();
    expect(scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==2 &&
           scene.accurate_island_root(a)==b &&
           scene.accurate_island_root(f)==d &&
           scene.accurate_cached_route_hops(f)==2 &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),c,d) &&
           has(scene.found_contacts(),d,e) &&
           has(scene.found_contacts(),e,f),
           "Multi-island dirty fixture did not establish source root hops");
    scene.set_position(f,{11.55f,14,0});
    scene.advance_collision();
    expect(has(scene.found_contacts(),d,f) &&
           scene.accurate_cached_route_hops(f)==1 &&
           scene.accurate_island_root(f)==d,
           "Added redundant D-F chord failed to shorten source hop");
    const auto checks=scene.island_dirty_route_check_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    const auto local_splits=scene.island_local_split_count();
    const auto surviving=scene.accurate_island_id(d);
    scene.set_position(b,{4,14,0});
    scene.set_position(f,{12.4f,14,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),a,b) &&
           has(scene.lost_contacts(),d,f) &&
           !has(scene.lost_contacts(),e,f) &&
           scene.accurate_island_count()==3 &&
           scene.speculative_island_count()==3 &&
           scene.accurate_island_id(d)==surviving &&
           scene.accurate_island_root(f)==d &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()>local_splits,
           "Concurrent source-local split rebuilt unrelated island graphs");
    // Four distinct dirty actors per graph: A fails, B and D are
    // roots, and F searches its surviving route via E. Both source
    // graphs must complete all four checks even when A fails.
    expect(scene.island_dirty_route_check_count()==checks+8 &&
           scene.accurate_cached_route_hops(f)==2,
           "First failed route aborted later source dirty-node repairs");
    scene.advance_collision();
    expect(scene.accurate_cached_route_hops(f)==2 &&
           scene.accurate_island_id(d)==surviving,
           "Source route repair did not survive quiet island publication");
}

// Source mDirtyMap begins a detached component's findRoute at its
// lowest dirty node, not its lowest BodyId. Source mVisitedNodes records
// discovered entries as they enter the hop-priority queue. In this
// arrangement B is the dirty endpoint but A is the smaller node ID;
// moving A's static manager ahead of B's is a real descriptor-order bug.
void verify_dirty_seed_split_manager_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene split;
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-1.6f,10,0},1);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.8f,10,0},1);
    const auto sa=split.add_static_box(
        {0.1f,0.1f,0.1f},{-1.6f,10,0.54f});
    const auto sb=split.add_static_box(
        {0.1f,0.1f,0.1f},{-0.8f,10,0.54f});
    split.advance_collision();
    expect(split.accurate_island_count()==1 &&
           split.contacts().size()==6 &&
           has(split.found_contacts(),b,c) &&
           has(split.found_contacts(),a,sa) &&
           has(split.found_contacts(),b,sb),
           "Dirty-seed source split fixture did not connect six managers");

    // Move only the retained-root C-D-E side. The orphan's three
    // contact managers must survive with unchanged contact geometry.
    split.set_position(c,{1.0f,10,0});
    split.set_position(d,{1.8f,10,0});
    split.set_position(e,{2.6f,10,0});
    split.step(1.0f/480.0f);
    const auto gu=split.contacts();
    const auto indices=split.indexed_contact_manager_order();
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),b,c) &&
           gu.size()==5 && indices.size()==5 &&
           split.accurate_island_root(a)==b &&
           split.accurate_island_root(b)==b &&
           split.accurate_island_root(c)==d &&
           split.accurate_island_root(d)==d,
           "Dirty-seed split did not preserve original root and promote B");

    std::vector<Pair> orphan_managers;
    for(std::size_t index:indices) {
        const Pair pair=gu[index].pair;
        if(pair==Pair{b,sb} || pair==Pair{a,sa} ||
           pair==Pair{a,b})
            orphan_managers.push_back(pair);
    }
    expect(orphan_managers.size()==3 &&
           orphan_managers[0]==Pair{b,sb} &&
           orphan_managers[1]==Pair{a,sa} &&
           orphan_managers[2]==Pair{a,b},
           "Split manager order ignored source first-dirty B visitation");
}

// PxsIslandSim::processLostEdges collects a separated component's
// managers by visiting its node adjacency, then splitEdges appends that
// collected sequence to its new island. Replaying globally preserved
// manager order gives a different finite-iteration PGS input here.
void verify_split_contact_manager_migration() {
    using fyzz::native::Scene;
    Scene split;
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1.0f);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1.0f);
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1.0f);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.4f,10,0},1.0f);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{3.2f,10,0},1.0f);
    split.advance_collision();
    expect(split.contacts().size()==4 &&
           split.accurate_island_count()==1 &&
           has(split.contacts().size()?split.found_contacts():
                                     split.lost_contacts(),b,c),
           "Split manager fixture did not form its five-body chain");
    // Losing B-C leaves A-B on the original root-B island. The
    // orphan C-D-E side must migrate contacts in visited C,D,E
    // adjacency order: C-D followed by D-E. The source's earlier
    // array swap-last on B-C would leave D-E before C-D if we
    // simply filtered/replayed the old array.
    split.set_position(c,{2.0f,10,0});
    split.step(1.0f/480.0f);
    const auto managers=split.indexed_contact_manager_order();
    const auto gu=split.contacts();
    expect(split.accurate_island_count()==2 &&
           gu.size()==3 && managers.size()==3 &&
           gu[0].pair==fyzz::native::Pair{a,b} &&
           gu[1].pair==fyzz::native::Pair{c,d} &&
           gu[2].pair==fyzz::native::Pair{d,e} &&
           managers[0]==0 && managers[1]==1 && managers[2]==2,
           "Split migration replayed old manager order instead of visited edges");
}

// Source contact-edge ordering and removal conformance remains in its
// own stack frame, alongside the other multi-island lifecycle fixtures.
void verify_island_contact_manager_order() {
    using fyzz::native::Scene;
    // PhysX indexed-manager order belongs to the accurate island's
    // eCONTACT_MANAGER edge array, not the pair-sorted Gu stream.
    // Build C first, then B and D, producing source [C,B,D] against
    // canonical Gu [B,C,D]. Removing C must swap-last to [D,B], not
    // perform a stable erase or rebuild from canonical pair order.
    {
        Scene island_managers;
        const auto center=island_managers.add_dynamic_box(
            {1,1,1},{0,10,0},1);
        const auto b=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{4,10,0});
        const auto c=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{-1.1f,10,0});
        const auto d=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{0,10,4});
        island_managers.advance_collision();
        expect(island_managers.contacts().size()==1 &&
               has(island_managers.found_contacts(),center,c),
               "Island manager fixture did not establish the first touch");
        island_managers.set_position(b,{1.1f,10,0});
        island_managers.set_position(d,{0,10,1.1f});
        island_managers.step(1.0f/480.0f);
        const auto initial=island_managers.indexed_contact_manager_order();
        const auto contacts=island_managers.contacts();
        expect(contacts.size()==3 && initial.size()==3 &&
               contacts[0].pair==fyzz::native::Pair{center,b} &&
               contacts[1].pair==fyzz::native::Pair{center,c} &&
               contacts[2].pair==fyzz::native::Pair{center,d} &&
               initial[0]==1 && initial[1]==0 && initial[2]==2,
               "Dy partitioning ignored island-owned manager insertion order");
        // A failed solver frame must restore the exact published manager
        // allocation, not just the same values. Admit an additional contact
        // in the *failed* frame so the transient indexed manager sequence
        // has four entries while the published sequence still has three.
        // The source material-combination overflow occurs in
        // prepare_constraints(), after partition_constraints() has rebuilt
        // this four-manager descriptor input.
        const auto published_pointer=initial.data();
        const auto published_contacts=contacts.data();
        const auto published_frame=island_managers.frame_index();
        const auto e=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{0,10,-1.1f});
        fyzz::Material overflowing{};
        overflowing.static_friction=1.0e30f;
        overflowing.dynamic_friction=1.0e30f;
        overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
        island_managers.set_material(center,overflowing);
        island_managers.set_material(b,overflowing);
        bool rejected=false;
        try { island_managers.step(1.0f/480.0f); }
        catch(const std::overflow_error& error) {
            rejected=std::string(error.what())==
                "Native Pxs material combination overflow";
        }
        const auto restored=island_managers.indexed_contact_manager_order();
        const auto restored_contacts=island_managers.contacts();
        expect(rejected &&
               island_managers.frame_index()==published_frame &&
               restored.data()==published_pointer &&
               restored.size()==3 &&
               restored[0]==1 && restored[1]==0 && restored[2]==2 &&
               restored_contacts.data()==published_contacts &&
               restored_contacts.size()==3,
               "Post-partition rejected PGS step invalidated published manager span");
        island_managers.set_material(center,fyzz::Material{});
        island_managers.set_material(b,fyzz::Material{});
        // The same pending authored fourth contact must publish on a
        // clean retry. This also proves the failed step would have rebuilt
        // a different (four-entry) indexed-manager stream before throwing.
        island_managers.step(1.0f/480.0f);
        const auto admitted=island_managers.indexed_contact_manager_order();
        const auto admitted_contacts=island_managers.contacts();
        const fyzz::native::Pair fourth_pair{std::min(center,e),
                                             std::max(center,e)};
        expect(admitted_contacts.size()==4 &&
               std::any_of(admitted_contacts.begin(),admitted_contacts.end(),
                   [&](const auto& contact) {
                       return contact.pair==fourth_pair;
                   }) &&
               admitted.size()==4 &&
               admitted[0]==1 && admitted[1]==0 &&
               admitted[2]==2 && admitted[3]==3,
               "Post-partition failure fixture did not admit its fourth manager");
        island_managers.set_position(center,{0,10,0});
        island_managers.set_position(c,{-4,10,0});
        island_managers.set_position(e,{0,10,-4});
        island_managers.step(1.0f/480.0f);
        const auto after=island_managers.indexed_contact_manager_order();
        const auto remaining=island_managers.contacts();
        expect(remaining.size()==2 && after.size()==2 &&
               remaining[0].pair==fyzz::native::Pair{center,b} &&
               remaining[1].pair==fyzz::native::Pair{center,d} &&
               after[0]==1 && after[1]==0,
               "PhysX island edge swap-last was replaced by stable order");
    }

    // PxsIslandSim::mergeEdges concatenates the retained winner's
    // manager array with the losing island's array before appending
    // the newly connected bridge. Equal-size merges preserve the
    // second island, independent of canonical Gu BodyId ordering.
    {
        Scene merge_order;
        const auto a=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1);
        const auto b=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,10,0},1);
        const auto c=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,10,0},1);
        const auto d=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3.8f,10,0},1);
        merge_order.advance_collision();
        expect(merge_order.contacts().size()==2 &&
               has(merge_order.found_contacts(),a,b) &&
               has(merge_order.found_contacts(),c,d),
               "Indexed manager merge fixture did not form separate islands");
        merge_order.set_position(c,{1.6f,10,0});
        merge_order.set_position(d,{2.4f,10,0});
        merge_order.step(1.0f/480.0f);
        const auto sequence=merge_order.indexed_contact_manager_order();
        const auto gu=merge_order.contacts();
        expect(gu.size()==3 && sequence.size()==3 &&
               gu[0].pair==fyzz::native::Pair{a,b} &&
               gu[1].pair==fyzz::native::Pair{b,c} &&
               gu[2].pair==fyzz::native::Pair{c,d} &&
               sequence[0]==2 && sequence[1]==0 && sequence[2]==1,
               "Source size-tied island merge did not concatenate manager order");
        // The tie retains C-D's D-root island. Source
        // mergeIslandsInternal computes extraPath =
        // hops[C](1) + hops[B](0) + 1 = 2, adds that to
        // the losing A-B island, then pins B to hops[C]+1.
        expect(merge_order.accurate_cached_route_hops(a)==3 &&
               merge_order.accurate_cached_route_hops(b)==2 &&
               merge_order.accurate_cached_route_hops(c)==1 &&
               merge_order.accurate_cached_route_hops(d)==0,
               "Source mergeIslandsInternal did not translate loser hop offsets");
    }

}

// Keep the long-running actor, sleep and rollback suites in a separate
// frame from the broadphase and constraint fixture groups on Windows.
// The source graph's rollback ownership is a cold-path concern. Steady
// PGS steps may update active-node publication and motion, but must not
// copy every persistent island node/handle array when no topology changes.
void verify_demand_driven_island_ownership_checkpoint() {
    using fyzz::native::Scene;
    Scene quiet;
    for(int i=0;i<24;++i)
        quiet.add_dynamic_box({0.5f,0.5f,0.5f},
                              {static_cast<float>(i*4),150,0},1.0f);
    quiet.advance_collision();
    const auto baseline=quiet.island_graph_checkpoint_count();
    const auto revision=quiet.island_topology_revision();
    const auto rebuilds=quiet.island_full_rebuild_count();
    expect(baseline>0 && quiet.accurate_island_count()==24,
           "Quiet checkpoint fixture did not admit all source islands");
    for(int frame=0;frame<80;++frame)
        quiet.step(1.0f/240.0f);
    expect(quiet.island_graph_checkpoint_count()==baseline &&
           quiet.island_full_rebuild_count()==rebuilds &&
           quiet.island_topology_revision()==revision &&
           quiet.accurate_island_count()==24,
           "Steady PGS copied persistent island topology without a mutation");
    const auto added=quiet.add_dynamic_box(
        {0.5f,0.5f,0.5f},{200,150,0},1.0f);
    quiet.advance_collision();
    expect(quiet.island_graph_checkpoint_count()==baseline+1 &&
           quiet.island_full_rebuild_count()==rebuilds+1 &&
           quiet.accurate_island_count()==25 &&
           quiet.accurate_island_root(added)==added,
           "Source node admission skipped the demand-driven ownership snapshot");
    const auto after=quiet.island_graph_checkpoint_count();
    quiet.advance_collision();
    expect(quiet.island_graph_checkpoint_count()==after,
           "Unchanged geometry republished a redundant graph checkpoint");
}

// Independent source edge-activity regressions live outside the full
// rigid integration fixture to keep their stack frames and failures isolated.
void verify_source_active_island_handoff() {
    using fyzz::native::Scene;
    Scene active;
    const auto floor=active.add_static_floor(0.0f);
    const auto a=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto b=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
    for(int frame=0;frame<360;++frame)
        active.step(1.0f/120.0f);
    expect(active.body_is_sleeping(a) && active.body_is_sleeping(b) &&
           active.active_accurate_islands().empty() &&
           active.active_speculative_islands().empty(),
           "Sleeping isolated rigid islands retained source active handles");
    // Source markIslandActive appends on the first genuine node wake.
    // Each independent graph owns its own active-index permutation.
    active.set_position(a,active.body_state(a).position);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_accurate_islands()[0]==active.accurate_island_id(a) &&
           active.active_speculative_islands().size()==1 &&
           active.active_speculative_islands()[0]==
               active.speculative_island_id(a),
           "Same-pose wake failed to append source island handles");
    const auto a_accurate=active.accurate_island_id(a);
    const auto a_speculative=active.speculative_island_id(a);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1,
           "Quiet publication duplicated already active source island");
    active.set_position(b,active.body_state(b).position);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands()[0]==a_speculative,
           "Additional awake island lost existing source activation order");
    const auto* accurate_published=active.active_accurate_islands().data();
    const auto* speculative_published=active.active_speculative_islands().data();
    const std::vector<std::uint32_t> accurate_before(
        active.active_accurate_islands().begin(),
        active.active_accurate_islands().end());
    const std::vector<std::uint32_t> speculative_before(
        active.active_speculative_islands().begin(),
        active.active_speculative_islands().end());
    bool rejected=false;
    try { active.step(std::numeric_limits<float>::max()); }
    catch(const std::overflow_error&) { rejected=true; }
    expect(rejected &&
           active.active_accurate_islands().data()==accurate_published &&
           active.active_speculative_islands().data()==speculative_published &&
           std::equal(accurate_before.begin(),accurate_before.end(),
                      active.active_accurate_islands().begin(),
                      active.active_accurate_islands().end()) &&
           std::equal(speculative_before.begin(),speculative_before.end(),
                      active.active_speculative_islands().begin(),
                      active.active_speculative_islands().end()),
           "Rejected source island handoff changed published storage/order");
    active.remove(b);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands().size()==1 &&
           active.active_speculative_islands()[0]==a_speculative,
           "Retiring rigid actor left stale mActiveIslands handles");
    const auto c=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{6,3,0},1.0f);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands()[0]==a_speculative &&
           active.accurate_island_id(c)!=a_accurate,
           "Actor admission rebuilt instead of preserving active island owner");
    const auto c_accurate=active.accurate_island_id(c);
    const auto c_speculative=active.speculative_island_id(c);
    // Retire the FIRST active island, not the last. Source
    // markIslandInactive replaces it with the final active handle.
    active.remove(a);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1 &&
           active.active_accurate_islands()[0]==c_accurate &&
           active.active_speculative_islands()[0]==c_speculative,
           "First-slot source active-island retirement broke swap-last");
    // Rejected actor admission rebuilds both IslandSims and changes their
    // visited-island scratch capacities. The old source permutation,
    // handles and scratch watermark must all survive and work on retry.
    const auto d=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{9,3,0},1.0f);
    bool failed_rebuild=false;
    try { active.step(std::numeric_limits<float>::max()); }
    catch(const std::overflow_error&) { failed_rebuild=true; }
    expect(failed_rebuild &&
           active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1 &&
           active.active_accurate_islands()[0]==c_accurate &&
           active.active_speculative_islands()[0]==c_speculative,
           "Rejected full island rebuild corrupted source active handles");
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.accurate_island_id(d)!=c_accurate &&
           active.speculative_island_id(d)!=c_speculative,
           "Retried source island rebuild reused stale active-list slots");
    (void)floor;
}

void verify_source_mixed_split_and_sleep_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene;
    // Admit the eventually awake root FIRST so a lost SAP-only bridge
    // detaches the already-sleeping second rod as a new island handle.
    // PxsIslandSim::split_lost_island appends that handle after an
    // unrelated, independently ready island's active handle.
    const auto awake=scene.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,0},1.0f);
    const auto dormant=scene.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,5},1.0f);
    for(const auto id:{awake,dormant}) {
        scene.set_velocity(id,{0,9.81f,0});
        scene.set_angular_velocity(id,{0,0.83f,0});
    }
    scene.step(1.0f);
    for(const auto id:{awake,dormant}) {
        scene.set_angular_velocity(id,{});
        scene.set_velocity(id,{});
    }
    scene.set_position(awake,{0.6f,0.1f,0.6f});
    scene.set_position(dormant,{0,0.1f,0});
    const auto floor=scene.add_static_floor(0.0f);
    scene.advance_collision();
    expect(has(scene.pairs(),awake,dormant) &&
           scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==1,
           "Mixed source sleep fixture lacks its SAP-only bridge");
    bool dormant_ready=false;
    for(int frame=0;frame<160;++frame) {
        scene.set_velocity(awake,{0,0,0.25f});
        scene.step(1.0f/120.0f);
        if(scene.body_is_sleeping(dormant) &&
           !scene.body_is_sleeping(awake) &&
           scene.speculative_island_count()==1) {
            dormant_ready=true;
            break;
        }
    }
    expect(dormant_ready,
           "Mixed source fixture could not sleep one SAP-only endpoint");
    const auto independent=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{30,0.5f,0},1.0f);
    const Pair dormant_floor{dormant,floor};
    const Pair independent_floor{std::min(independent,floor),
                                 std::max(independent,floor)};
    bool third_pass_ready=false;
    for(int frame=0;frame<150;++frame) {
        scene.set_velocity(awake,{0,0,0.25f});
        scene.step(1.0f/120.0f);
        if(scene.body_ready_for_sleep(independent) &&
           !scene.body_is_sleeping(independent)) {
            third_pass_ready=true;
            break;
        }
    }
    expect(third_pass_ready,
           "Mixed source fixture independent island never became ready");
    expect(scene.body_is_sleeping(dormant),
           "Mixed source fixture dormant rod woke before independent readiness");
    expect(!scene.body_is_sleeping(awake),
           "Mixed source fixture bridge root slept before independent readiness");
    expect(has(scene.pairs(),awake,dormant),
           "Mixed source fixture lost its SAP-only bridge before splitting");
    expect(scene.speculative_contact_edge_active(dormant_floor),
           "Mixed source fixture dormant floor edge not speculative-active");
    expect(scene.speculative_contact_edge_active(independent_floor),
           "Mixed source fixture independent floor edge not speculative-active");
    const auto dormant_edge=scene.contact_edge_index(dormant_floor);
    const auto independent_edge=scene.contact_edge_index(independent_floor);
    expect(dormant_edge!=std::numeric_limits<std::uint32_t>::max() &&
           independent_edge!=std::numeric_limits<std::uint32_t>::max(),
           "Mixed source fixture lost its floor-manager source handles");
    // This collision-only handoff must split the sleeping component
    // while Dy's previously ready independent node deactivates.
    // Source speculative reverse mActiveIslands visits the appended
    // split handle BEFORE the still-indexed independent island.
    scene.set_position(awake,{12,0.1f,12});
    scene.advance_collision();
    const auto events=scene.deactivating_speculative_contact_edges();
    const auto a=std::find(events.begin(),events.end(),dormant_edge);
    const auto b=std::find(events.begin(),events.end(),independent_edge);
    expect(a!=events.end() && b!=events.end() && a<b &&
           scene.body_is_sleeping(dormant) &&
           scene.body_is_sleeping(independent) &&
           scene.speculative_island_count()==3,
           "Split sleeper lost its original reverse speculative active index");
    scene.advance_collision();
    expect(scene.deactivating_speculative_contact_edges().empty() &&
           !scene.speculative_contact_edge_active(dormant_floor) &&
           !scene.speculative_contact_edge_active(independent_floor),
           "Mixed split/sleep replayed the one-shot speculative events");
}

void verify_source_speculative_reverse_sleep_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene;
    const auto a=scene.add_dynamic_box({2.0f,0.1f,0.2f},{0,10,0},1.0f);
    const auto b=scene.add_dynamic_box({2.0f,0.1f,0.2f},{20,10,0},1.0f);
    const auto c=scene.add_dynamic_box({2.0f,0.1f,0.2f},{0,10,5},1.0f);
    // Two long rotated boxes overlap in SAP, but the Gu boxes remain
    // separate. Source speculative merge swaps the first active handle
    // with the last while accurate singleton handles retain their order.
    for(const auto id:{a,b,c}) {
        scene.set_velocity(id,{0,9.81f,0});
        scene.set_angular_velocity(id,{0,0.83f,0});
    }
    scene.step(1.0f);
    for(const auto id:{a,b,c}) {
        scene.set_angular_velocity(id,{});
        scene.set_velocity(id,{});
    }
    scene.set_position(a,{0,0.1f,0});
    scene.set_position(b,{20,0.1f,0});
    scene.set_position(c,{0.6f,0.1f,0.6f});
    const auto floor=scene.add_static_floor(0.0f);
    scene.advance_collision();
    expect(has(scene.pairs(),a,c) &&
           !has(scene.found_contacts(),a,c) &&
           scene.accurate_island_count()==3 &&
           scene.speculative_island_count()==2 &&
           scene.active_accurate_islands().size()==3 &&
           scene.active_speculative_islands().size()==2,
           "Speculative sleep fixture did not establish SAP-only source merge");

    const std::array<fyzz::BodyId,3> bodies{a,b,c};
    const std::array<std::uint32_t,3> floor_edges{
        scene.contact_edge_index({a,floor}),
        scene.contact_edge_index({b,floor}),
        scene.contact_edge_index({c,floor})};
    for(const auto edge:floor_edges)
        expect(edge!=std::numeric_limits<std::uint32_t>::max(),
               "Speculative sleep fixture lacks an active floor manager");
    std::array<std::uint32_t,3> accurate_handles{};
    for(std::size_t i=0;i<bodies.size();++i)
        accurate_handles[i]=scene.accurate_island_id(bodies[i]);

    // Verify that the fixture would fail with candidate-order
    // publication. An accurate reverse traversal must yield a different
    // speculative order from the speculative reverse active list.
    std::vector<std::uint32_t> accurate_derived;
    for(auto it=scene.active_accurate_islands().rbegin();
        it!=scene.active_accurate_islands().rend();++it)
        for(std::size_t j=0;j<bodies.size();++j)
            if(*it==accurate_handles[j]) {
                const auto handle=scene.speculative_island_id(bodies[j]);
                if(std::find(accurate_derived.begin(),
                             accurate_derived.end(),handle)==
                   accurate_derived.end())
                    accurate_derived.push_back(handle);
            }
    const std::vector<std::uint32_t> source_reverse{
        scene.active_speculative_islands().rbegin(),
        scene.active_speculative_islands().rend()};
    expect(accurate_derived!=source_reverse &&
           source_reverse.size()==2,
           "Speculative sleep fixture does not discriminate graph ordering");

    bool retired=false;
    for(int frame=0;frame<360;++frame) {
        scene.step(1.0f/120.0f);
        const auto events=scene.deactivating_speculative_contact_edges();
        if(events.empty())continue;
        expect(!retired,"Speculative source sleep re-published retired edges");
        std::vector<std::uint32_t> observed;
        for(const auto edge:events)
            for(std::size_t i=0;i<bodies.size();++i)
                if(edge==floor_edges[i]) {
                    const auto handle=scene.speculative_island_id(bodies[i]);
                    if(std::find(observed.begin(),observed.end(),handle)==
                       observed.end())
                        observed.push_back(handle);
                }
        expect(observed==source_reverse &&
               scene.body_is_sleeping(a) &&
               scene.body_is_sleeping(b) &&
               scene.body_is_sleeping(c),
               "Source speculative deactivation ignored reverse active-island order");
        retired=true;
    }
    expect(retired && scene.active_speculative_islands().empty(),
           "Speculative source island retirement did not complete");
    scene.advance_collision();
    expect(scene.deactivating_speculative_contact_edges().empty(),
           "Quiet source publication repeated speculative retirement");
}

void verify_source_solver_task_batch_partitioning() {
    using fyzz::native::Scene;
    Scene scene;
    scene.add_static_floor(0.0f);
    // DyDynamicsContext::updatePostKinematic uses the default
    // solverBatchSize=128 and requires at least one contact before
    // ending a task group. These 130 independent rigid contact islands
    // must form two descriptor groups, not one global partition zero.
    constexpr std::size_t bodies=130;
    for(std::size_t i=0;i<bodies;++i)
        scene.add_dynamic_box({0.5f,0.5f,0.5f},
            {static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    scene.step(1.0f/120.0f);
    expect(scene.contacts().size()==bodies &&
           scene.indexed_contact_manager_order().size()==bodies &&
           scene.solver_partition_count()==2,
           "Source Dy task-group boundary failed to partition 128+2 islands");
    const auto impulses=scene.solved_normal_impulses();
    std::size_t total=0;
    for(std::size_t i=0;i<scene.contacts().size();++i) {
        const auto& manager=scene.contacts()[i];
        expect(manager.contact_offset==total &&
               total+manager.count<=impulses.size(),
               "Task-group source descriptor writeback lost its Gu offset");
        double force=0.0;
        for(std::size_t j=0;j<manager.count;++j)
            force+=impulses[total+j];
        expect(std::isfinite(force) && force>0.0,
               i>=128
                   ? "Final source task group dropped rigid solver work"
                   : "First source task group dropped rigid solver work");
        total+=manager.count;
    }
    expect(total==impulses.size(),
           "Source task groups did not write all rigid contact impulses");
    scene.step(1.0f/120.0f);
    expect(scene.solver_partition_count()==2 &&
           scene.contacts().size()==bodies,
           "Source task-group boundaries leaked partition state across steps");
}

void verify_source_reverse_active_island_sleep_retirement() {
    using fyzz::native::Scene;
    Scene scene;
    scene.add_static_floor(0.0f);
    std::array<fyzz::BodyId,4> bodies{};
    for(std::size_t i=0;i<bodies.size();++i)
        bodies[i]=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},
            {static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    scene.advance_collision();
    expect(scene.active_accurate_islands().size()==bodies.size(),
           "Sleep order fixture failed to publish independent active islands");
    bool deactivated=false;
    for(int frame=0;frame<360;++frame) {
        const std::vector<std::uint32_t> before(
            scene.active_accurate_islands().begin(),
            scene.active_accurate_islands().end());
        scene.step(1.0f/120.0f);
        const auto retired=scene.deactivating_accurate_nodes();
        if(retired.empty())continue;
        expect(!deactivated && retired.size()==bodies.size() &&
               before.size()==bodies.size(),
               "Source independent sleep islands did not retire together");
        // PxsIslandSim::processLostEdges walks mActiveIslands backwards.
        // The old forward scan of the full island-handle table must
        // fail this assertion even though it sleeps the same bodies.
        for(std::size_t i=0;i<retired.size();++i)
            expect(scene.accurate_island_id(retired[i])==
                   before[before.size()-i-1],
                   "Third-pass deactivation lost source reverse active-island order");
        expect(scene.active_accurate_islands().empty() &&
               scene.active_speculative_islands().empty(),
               "Third-pass sleep left stale published active islands");
        deactivated=true;
    }
    expect(deactivated,
           "Source reverse active-island retirement never reached sleep");
    scene.advance_collision();
    expect(scene.deactivating_accurate_nodes().empty() &&
           scene.active_accurate_islands().empty(),
           "Quiet publication replayed already retired active islands");
}

void verify_source_reverse_wake_order_and_rollback_capacity() {
    using fyzz::native::Scene;
    Scene scene;
    scene.add_static_floor(0.0f);
    std::array<fyzz::BodyId,4> ids{};
    for(std::size_t i=0;i<ids.size();++i)
        ids[i]=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    for(int frame=0;frame<360;++frame)
        scene.step(1.0f/120.0f);
    expect(scene.active_accurate_islands().empty() &&
           scene.active_speculative_islands().empty(),
           "Reverse wake fixture did not retire the dormant source islands");
    std::array<std::uint32_t,4> accurate{},speculative{};
    for(std::size_t i=0;i<ids.size();++i) {
        accurate[i]=scene.accurate_island_id(ids[i]);
        speculative[i]=scene.speculative_island_id(ids[i]);
        expect(scene.body_is_sleeping(ids[i]),
               "Reverse wake fixture did not put all bodies to sleep");
    }
    // Queue multiple distinct authored wakes before a single source
    // wakeIslands publication, explicitly against the sorted BodyId order.
    // Reject the publication first: its rollback must restore both
    // graphs' pending event sequences and reserve no-throw list capacity.
    const std::array<std::size_t,4> order{3,1,2,0};
    // Exercise all four public wake sources. The event sequence must
    // survive the failed step exactly as authored, rather than silently
    // falling back to accurate graph handle or BodyId ordering.
    scene.set_velocity(ids[3],{});
    scene.set_angular_velocity(ids[1],{});
    scene.apply_torque_impulse(ids[2],{});
    scene.set_position(ids[0],scene.body_state(ids[0]).position);
    bool rejected=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {rejected=true;}
    expect(rejected &&
           scene.active_accurate_islands().empty() &&
           scene.active_speculative_islands().empty(),
           "Rejected multi-wake step prematurely published source islands");
    scene.advance_collision();
    const auto actual=scene.active_accurate_islands();
    const auto sap=scene.active_speculative_islands();
    expect(actual.size()==order.size() && sap.size()==order.size(),
           "Wake events omitted some previously dormant island");
    for(std::size_t i=0;i<order.size();++i)
        expect(actual[i]==accurate[order[i]] &&
               sap[i]==speculative[order[i]],
               "Source mActivatingNodes order lost in graph-specific handoff");
    scene.advance_collision();
    for(std::size_t i=0;i<order.size();++i)
        expect(scene.active_accurate_islands()[i]==accurate[order[i]] &&
               scene.active_speculative_islands()[i]==speculative[order[i]],
               "Quiet handoff duplicated or sorted source active handles");
}

void verify_source_contact_interaction_consumption() {
    using fyzz::native::Scene;
    // Sc::InteractionFlag::eIS_ACTIVE follows the speculative graph,
    // not the accurate Gu contact or an actor's individual sleep flag.
    Scene scene;
    const auto floor=scene.add_static_floor(0.0f);
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{4,0.5f,0},1.0f);
    const fyzz::native::Pair floor_a{std::min(a,floor),std::max(a,floor)};
    scene.advance_collision();
    expect(scene.speculative_contact_edge_active(floor_a) &&
           scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==2,
           "Sc activation did not consume source speculative contact edges");
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==2,
           "Sc interaction was spuriously reactivated on a quiet query");
    for(int i=0;i<360;++i)scene.step(1.0f/120.0f);
    expect(scene.body_is_sleeping(a) &&
           !scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==0 &&
           !scene.speculative_contact_edge_active(floor_a),
           "Sc third-pass failed to consume sleeping speculative edges");
    // Wake only one dormant island; its other floor interaction must
    // remain inactive and cannot be woken by accurate island ordering.
    scene.set_velocity(a,{});
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Sc speculative interaction did not reactivate on authored wake");
    // Force a Dy overflow after a published interaction transition.
    scene.set_position(a,{0,3,0});
    bool rejected=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {rejected=true;}
    expect(rejected && scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Rejected Sc pipeline step corrupted interaction activity");
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           !scene.contact_edge_connected(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Gu-only touch loss incorrectly retired the speculative interaction");
    // A plane has unbounded broadphase coverage. Removing it, not
    // translating the dynamic box, actually destroys the SAP overlap.
    scene.remove(floor);
    scene.advance_collision();
    expect(!scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==0,
           "Destroyed source overlap retained stale Sc interaction activity");

    // A rejected frame can discover a NEW contact-manager EdgeIndex,
    // activate its speculative interaction, then throw in Dy. Rollback
    // must not publish either the new manager or the interaction count;
    // retry must admit the same logical contact exactly once.
    Scene admission;
    const auto wall=admission.add_static_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0});
    const auto box=admission.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
    const fyzz::native::Pair new_pair{
        std::min(wall,box),std::max(wall,box)};
    admission.advance_collision();
    expect(admission.active_contact_interaction_count()==0 &&
           !admission.contact_interaction_active(new_pair),
           "Sc new-contact rollback fixture started with active interaction");
    admission.set_position(box,{0.9f,0.5f,0});
    bool admission_failed=false;
    try {admission.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {admission_failed=true;}
    expect(admission_failed &&
           admission.active_contact_interaction_count()==0 &&
           !admission.contact_interaction_active(new_pair) &&
           admission.contact_edge_index(new_pair)==
               std::numeric_limits<std::uint32_t>::max(),
           "Failed first-touch frame published Sc interaction prematurely");
    admission.advance_collision();
    expect(admission.active_contact_interaction_count()==1 &&
           admission.contact_interaction_active(new_pair) &&
           admission.speculative_contact_edge_active(new_pair),
           "Retried first-touch failed to consume source edge activation");
}

void verify_source_active_reference_lifecycle() {
    using fyzz::native::Scene;
    Scene refs;
    const auto floor=refs.add_static_floor(0.0f);
    const auto a=refs.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,0.5f,0},1.0f);
    const auto b=refs.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1.0f);
    const fyzz::native::Pair pair{std::min(a,b),std::max(a,b)};
    refs.advance_collision();
    const auto handle=refs.contact_edge_index(pair);
    const auto valid=std::numeric_limits<std::uint32_t>::max();
    expect(handle!=valid &&
           refs.speculative_contact_edge_active(pair) &&
           refs.accurate_contact_edge_active(pair) &&
           refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1 &&
           refs.speculative_active_reference_count(floor)==0 &&
           refs.accurate_active_reference_count(floor)==0,
           "Source dynamic contact references incorrectly include static edges");
    // A persistent active edge cannot double-count its endpoints during
    // unchanged collision publications or another dynamic actor's entry.
    refs.advance_collision();
    refs.add_dynamic_box({0.5f,0.5f,0.5f},{50,50,0},1.0f);
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Source active-edge references drifted on quiet/rebuilt graph");
    refs.set_position(b,{6,0.5f,0});
    bool failed=false;
    try {refs.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {failed=true;}
    expect(failed && refs.contact_edge_index(pair)==handle &&
           refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Rejected lost-contact step corrupted source endpoint references");
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==0 &&
           refs.speculative_active_reference_count(b)==0 &&
           refs.accurate_active_reference_count(a)==0 &&
           refs.accurate_active_reference_count(b)==0,
           "Retired dynamic contact did not release both source references");
    refs.set_position(b,{0.85f,0.5f,0});
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Reconnected source manager did not reacquire endpoint references");
    refs.remove(b);
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==0 &&
           refs.accurate_active_reference_count(a)==0,
           "Actor retirement left stale source endpoint references");
}

void verify_source_speculative_only_active_references() {
    using fyzz::native::Scene;
    Scene separated;
    const auto a=separated.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,0},1.0f);
    const auto b=separated.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,5},1.0f);
    for(fyzz::BodyId id:{a,b}) {
        separated.set_velocity(id,{0,9.81f,0});
        separated.set_angular_velocity(id,{0,0.83f,0});
    }
    separated.step(1.0f);
    for(fyzz::BodyId id:{a,b}) {
        separated.set_velocity(id,{});
        separated.set_angular_velocity(id,{});
    }
    separated.set_position(a,{0,0.1f,0});
    separated.set_position(b,{0.6f,0.1f,0.6f});
    separated.advance_collision();
    const fyzz::native::Pair candidate{a,b};
    expect(has(separated.pairs(),a,b) &&
           !has(separated.found_contacts(),a,b) &&
           separated.speculative_island_count()==1 &&
           separated.accurate_island_count()==2 &&
           separated.speculative_active_reference_count(a)==1 &&
           separated.speculative_active_reference_count(b)==1 &&
           separated.accurate_active_reference_count(a)==0 &&
           separated.accurate_active_reference_count(b)==0,
           "SAP-only source contact references leaked into accurate graph");
    separated.set_position(b,{12.0f,0.1f,12.0f});
    separated.advance_collision();
    expect(has(separated.lost_pairs(),a,b) &&
           separated.speculative_active_reference_count(a)==0 &&
           separated.speculative_active_reference_count(b)==0 &&
           separated.accurate_active_reference_count(a)==0 &&
           separated.accurate_active_reference_count(b)==0,
           "SAP-only source reference retirement missed the lost edge");
}

void verify_source_retired_bridge_deactivation() {
    using fyzz::native::Scene;
    {
        // Retiring an awake actor from an otherwise dormant SAP-only
        // bridge invokes the FULL island rebuild in the first pass.
        // The retained floor manager must still publish one speculative
        // deactivation in the third pass, without requiring a new
        // accurate sleep event or a local lost-edge split.
        Scene retired_bridge;
        const auto sleeping_rod=retired_bridge.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto active_rod=retired_bridge.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(fyzz::BodyId id:{sleeping_rod,active_rod}) {
            retired_bridge.set_velocity(id,{0,9.81f,0});
            retired_bridge.set_angular_velocity(id,{0,0.83f,0});
        }
        retired_bridge.step(1.0f);
        for(fyzz::BodyId id:{sleeping_rod,active_rod}) {
            retired_bridge.set_angular_velocity(id,{});
            retired_bridge.set_velocity(id,{});
        }
        retired_bridge.set_position(sleeping_rod,{0,0.1f,0});
        retired_bridge.set_position(active_rod,{0.6f,0.1f,0.6f});
        const auto floor=retired_bridge.add_static_floor(0.0f);
        retired_bridge.advance_collision();
        bool independent_sleep=false;
        for(int frame=0;frame<160;++frame) {
            retired_bridge.set_velocity(active_rod,{0,0,0.25f});
            retired_bridge.step(1.0f/120.0f);
            if(retired_bridge.body_is_sleeping(sleeping_rod) &&
               !retired_bridge.body_is_sleeping(active_rod) &&
               retired_bridge.speculative_island_count()==1 &&
               retired_bridge.accurate_island_count()==2) {
                independent_sleep=true;
                break;
            }
        }
        const fyzz::native::Pair floor_pair{
            std::min(floor,sleeping_rod),std::max(floor,sleeping_rod)};
        const auto floor_edge=retired_bridge.contact_edge_index(floor_pair);
        const auto prior_spec_count=
            retired_bridge.active_speculative_contact_edge_count();
        expect(independent_sleep &&
               floor_edge!=std::numeric_limits<std::uint32_t>::max() &&
               retired_bridge.speculative_contact_edge_active(floor_pair) &&
               retired_bridge.contact_interaction_active(floor_pair) &&
               !retired_bridge.accurate_contact_edge_active(floor_pair),
               "Retirement fixture did not establish independent island sleep");
        retired_bridge.remove(active_rod);
        retired_bridge.advance_collision();
        const auto events=retired_bridge.deactivating_speculative_contact_edges();
        expect(retired_bridge.body_is_sleeping(sleeping_rod) &&
               !retired_bridge.speculative_contact_edge_active(floor_pair) &&
               !retired_bridge.contact_interaction_active(floor_pair) &&
               std::count(events.begin(),events.end(),floor_edge)==1 &&
               retired_bridge.deactivating_accurate_contact_edges().empty() &&
               prior_spec_count>0 &&
               retired_bridge.active_speculative_contact_edge_count()==0,
               "Full rebuild skipped speculative retirement of dormant manager");
        retired_bridge.advance_collision();
        expect(retired_bridge.deactivating_speculative_contact_edges().empty() &&
               !retired_bridge.speculative_contact_edge_active(floor_pair),
               "Retired-actor rebuild replayed speculative deactivation");
    }

}
void verify_source_activity_rollback() {
    using fyzz::native::Scene;
    {
        // A failed Gu touch-loss must restore the active contact-manager
        // bits as well as each IslandSim's counts and indexed worklists.
        // Structural EdgeUndo snapshots are taken *after* the edge's
        // activity was cleared; they cannot supersede ActivityUndo.
        Scene activity_rollback;
        const auto floor=activity_rollback.add_static_floor(0.0f);
        const auto body=activity_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const fyzz::native::Pair touch{std::min(floor,body),std::max(floor,body)};
        activity_rollback.advance_collision();
        const auto original_handle=activity_rollback.contact_edge_index(touch);
        const auto accurate_active=
            activity_rollback.active_accurate_contact_edge_count();
        const auto speculative_active=
            activity_rollback.active_speculative_contact_edge_count();
        expect(activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch),
               "Source activity rollback fixture failed to create active edge");
        activity_rollback.set_position(body,{0,3,0});
        bool failed=false;
        try {activity_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed &&
               activity_rollback.contact_edge_index(touch)==original_handle &&
               activity_rollback.contact_edge_connected(touch) &&
               activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch) &&
               activity_rollback.active_accurate_contact_edge_count()==
                   accurate_active &&
               activity_rollback.active_speculative_contact_edge_count()==
                   speculative_active,
               "Failed touch-loss step did not restore source edge activity");
        activity_rollback.advance_collision();
        expect(!activity_rollback.contact_edge_connected(touch) &&
               !activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch) &&
               activity_rollback.active_accurate_contact_edge_count()+1==
                   accurate_active &&
               activity_rollback.active_speculative_contact_edge_count()==
                   speculative_active,
               "Retried touch loss did not retire accurate activity exactly once");
    }

}

void verify_native_rigid_integration_and_island_lifecycle() {
    using fyzz::native::Scene;
    // End-to-end native CPU execution, without a reference PxScene.
    {
        Scene falling_scene;
        falling_scene.add_static_floor(0);
        const auto falling = falling_scene.add_dynamic_box(
            {0.25f,0.25f,0.25f}, {0,2,0}, 1);
        for (int frame = 0; frame < 360; ++frame)
            falling_scene.step(1.0f / 120.0f);
        const auto state = falling_scene.body_state(falling);
        if (!(std::isfinite(state.position.y) &&
              std::abs(state.position.y - 0.25f) < 0.06f &&
              std::abs(state.velocity.y) < 0.6f))
            std::fprintf(stderr,
                "floor y=%g vy=%g spin=(%g,%g,%g) quat=(%g,%g,%g,%g) contacts=%zu\n",
                state.position.y,state.velocity.y,
                state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z,
                state.orientation.x,state.orientation.y,state.orientation.z,state.orientation.w,
                falling_scene.contacts().size());
        expect(std::isfinite(state.position.y) &&
               std::abs(state.position.y - 0.25f) < 0.06f &&
               std::abs(state.velocity.y) < 0.6f,
               "Native CPU floor contact did not settle");
        expect(falling_scene.frame_index() == 360 &&
               falling_scene.contacts().size() == 1,
               "Native step must publish exactly one collision frame");
        falling_scene.remove(falling);
        falling_scene.step(1.0f / 120.0f);
        expect(falling_scene.contacts().empty() &&
               !falling_scene.lost_contacts().empty(),
               "Native CPU actor retirement failed to publish touch loss");
    }
    {
        // The provisional translation-only mass admission must not
        // silently turn a constrained rigid body into a rotation-locked
        // one when source PxReal angular inertia is not representable.
        Scene heavy;
        const auto obstacle=heavy.add_static_box({1,1,1},{0,5,0});
        const auto body = heavy.add_dynamic_box(
            {1,1,1},{1.5f,5,0},std::numeric_limits<float>::max());
        heavy.set_velocity(body,{-1,0,0});
        bool rejected_contact=false;
        try { heavy.step(1.0f/120.0f); }
        catch(const std::logic_error&) { rejected_contact=true; }
        expect(rejected_contact &&
               heavy.frame_index()==0 &&
               heavy.body_state(body).position.x==1.5f &&
               heavy.body_state(body).velocity.x==-1.0f,
               "Nonrepresentable heavy-body contact was solved or leaked state");
        bool angular_rejected = false;
        try { heavy.apply_torque_impulse(body,{1,0,0}); }
        catch(const std::logic_error&) { angular_rejected = true; }
        expect(angular_rejected,
               "Heavy body with nonrepresentable angular inertia accepted torque");
        heavy.remove(obstacle);
        heavy.step(1.0f/120.0f);
        expect(heavy.body_state(body).position.x<1.5f &&
               std::isfinite(heavy.body_state(body).velocity.x),
               "Admitted free translation failed after unsupported contact rejection");
    }

    {
        Scene stacked;
        const auto ground=stacked.add_static_box({2,0.5f,2},{0,-0.5f,0});
        const auto lower = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,2,0},1);
        const auto upper = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,3.5f,0},1);
        // Source PGS, independently reproduced on Linux and Windows:
        // at frame 86, the lower box/ground patch saturates during a
        // position pass, then recovers (no clamp) during final velocity
        // writeback. The previous "broken |= clamped" across ALL passes
        // incorrectly reports a broken patch. Its local anchors must
        // also remain eligible for reuse in the following frame.
        const fyzz::native::Pair support_pair{ground,lower};
        std::array<fyzz::Vec3,2> saved_support_anchors{};
        std::uint8_t saved_anchor_count=0;
        for (int frame=0;frame<600;++frame) {
            stacked.step(1.0f/120.0f);
            if (stacked.frame_index()==86 ||
                stacked.frame_index()==87) {
                const auto reports=stacked.solved_friction_patches();
                const auto it=std::find_if(reports.begin(),reports.end(),
                    [&](const auto& patch) {
                        return patch.pair==support_pair;
                    });
                expect(it!=reports.end() && it->anchor_count>0,
                       "Stack contact lost the frame-86 friction anchors");
                if (stacked.frame_index()==86) {
                    expect(!it->broken,
                           "Position-only clamp incorrectly persisted as final friction break");
                    saved_anchor_count=it->anchor_count;
                    saved_support_anchors=it->anchor_positions;
                } else {
                    expect(it->anchor_count==saved_anchor_count,
                           "Unbroken friction anchors were discarded next frame");
                    for (std::size_t k=0;k<saved_anchor_count;++k) {
                        const auto& a=it->anchor_positions[k];
                        const auto& b=saved_support_anchors[k];
                        expect(std::abs(a.x-b.x)<0.001f &&
                               std::abs(a.y-b.y)<0.001f &&
                               std::abs(a.z-b.z)<0.001f,
                               "Recovered patch did not reuse its cached local anchors");
                    }
                }
            }
        }
        const auto lo = stacked.body_state(lower);
        const auto hi = stacked.body_state(upper);
        if (!(std::isfinite(lo.position.y) &&
               std::isfinite(hi.position.y) &&
               lo.position.y > 0.37f && lo.position.y < 0.65f &&
               hi.position.y > 1.25f && hi.position.y < 1.70f &&
               std::abs(lo.velocity.y) < 0.8f &&
               std::abs(hi.velocity.y) < 0.8f))
            std::fprintf(stderr,
                "stack lo(y=%g vy=%g spin=%g,%g,%g) hi(y=%g vy=%g spin=%g,%g,%g) contacts=%zu\n",
                lo.position.y,lo.velocity.y,lo.angular_velocity.x,
                lo.angular_velocity.y,lo.angular_velocity.z,
                hi.position.y,hi.velocity.y,hi.angular_velocity.x,
                hi.angular_velocity.y,hi.angular_velocity.z,
                stacked.contacts().size());
        expect(std::isfinite(lo.position.y) &&
               std::isfinite(hi.position.y) &&
               lo.position.y > 0.37f && lo.position.y < 0.65f &&
               hi.position.y > 1.25f && hi.position.y < 1.70f &&
               std::abs(lo.velocity.y) < 0.8f &&
               std::abs(hi.velocity.y) < 0.8f,
               "Native two-body contact chain failed");
    }

    {
        // Tiny mass can still have a representable inverse mass while
        // its source angular tensor is not representable. Contact PGS
        // must reject this geometry instead of using a zero tensor.
        Scene very_light;
        const auto left = very_light.add_dynamic_box(
            {1,1,1},{0,5,0},5.0e-40f);
        const auto right = very_light.add_dynamic_box(
            {1,1,1},{1.5f,5,0},5.0e-40f);
        bool rejected_angular = false;
        try { very_light.apply_torque_impulse(left, {1,0,0}); }
        catch (const std::logic_error&) { rejected_angular = true; }
        expect(rejected_angular,
               "Unrepresentable angular inertia silently used a substitute");
        rejected_angular = false;
        try { very_light.set_angular_velocity(left, {1,0,0}); }
        catch (const std::logic_error&) { rejected_angular = true; }
        expect(rejected_angular,
               "Tiny-mass body admitted an unrepresentable angular state");
        bool rejected_contact=false;
        try { very_light.step(1.0f/120.0f); }
        catch(const std::logic_error&) { rejected_contact=true; }
        expect(rejected_contact &&
               very_light.frame_index()==0 &&
               very_light.body_state(left).position.x==0.0f &&
               very_light.body_state(right).position.x==1.5f,
               "Tiny-mass contact substituted a zero angular tensor");
        very_light.set_position(right,{5,5,0});
        very_light.set_velocity(left,{1,0,0});
        very_light.step(1.0f/120.0f);
        expect(std::isfinite(very_light.body_state(left).velocity.x) &&
               very_light.body_state(left).position.x>0.0f,
               "Rejected tiny-mass contact poisoned later free translation");
    }

    {
        // Inverse mass can remain representable when a source PxReal box
        // inertia component underflows, overflows or has an
        // unrepresentable inverse. Those are translation-only bodies,
        // not grounds for rejecting the existing mass admission.
        const std::array<std::pair<fyzz::Vec3, float>, 4> cases{{
            {{1.0e-13f,1.0e-13f,1.0e-13f},1.0f},
            {{1.0e-13f,1.0e-13f,1.0f},1.0f},
            {{1.0f,1.0f,1.0f},5.0e-40f},
            {{1.0e13f,1.0e13f,1.0e13f},1.0e-39f}
        }};
        for (const auto& [half_extents,density] : cases) {
            Scene body;
            const auto id = body.add_dynamic_box(
                half_extents,{0,5,0},density);
            bool rejected = false;
            try { body.set_angular_velocity(id,{1,0,0}); }
            catch(const std::logic_error&) { rejected = true; }
            expect(rejected,
                   "Unrepresentable box inertia incorrectly admitted spin");
            rejected = false;
            try { body.apply_torque_impulse(id,{1,0,0}); }
            catch(const std::logic_error&) { rejected = true; }
            expect(rejected,
                   "Unrepresentable box inertia incorrectly admitted torque");
            body.set_velocity(id,{1,0,0});
            body.step(1.0f/120.0f);
            const auto state=body.body_state(id);
            expect(std::isfinite(state.position.x) &&
                   std::isfinite(state.position.y) &&
                   std::isfinite(state.velocity.x) &&
                   std::isfinite(state.velocity.y) &&
                   state.position.x>0.0f && state.velocity.x>0.0f &&
                   state.orientation.w==1.0f,
                   "Valid translational body rejected or corrupted by inertia");
        }
    }

    {
        // Two individually finite extreme inputs must pass through the
        // exact PhysX unconstrained-velocity clamp before contact prep.
        // Preserve finite impulse state without a custom wide dot product.
        Scene extreme;
        const auto a = extreme.add_dynamic_box({50,50,50}, {0,60,0}, 1);
        const auto b = extreme.add_dynamic_box({50,50,50}, {80,60,0}, 1);
        extreme.set_velocity(a, {2.0e38f,0,0});
        extreme.set_velocity(b, {-2.0e38f,0,0});
        // Smallest normal float: portable even with flush-to-zero.
        extreme.step(std::numeric_limits<float>::min());
        const auto va = extreme.body_state(a).velocity;
        const auto vb = extreme.body_state(b).velocity;
        expect(std::isfinite(va.x) && std::isfinite(va.y) &&
               std::isfinite(va.z) && std::isfinite(vb.x) &&
               std::isfinite(vb.y) && std::isfinite(vb.z) &&
               va.x < 0.0f && vb.x > 0.0f &&
               std::abs(va.x) < 1.0e38f && std::abs(vb.x) < 1.0e38f,
               "Finite high-speed contact overflowed or corrupted velocities");
        expect(extreme.contacts().size() == 1,
               "High-speed dynamic pair lost its contact");
    }
    {
        // Regression for a second float overflow: relative velocity and
        // impulse remain finite in double, but unequal inverse masses
        // previously overflowed one body's float velocity on writeback.
        // Mirror the PhysX dynamic-body maxLinearVelocitySq=1e32 default.
        // The unconstrained speed clamp precedes the contact solver.
        // Use touching faces (zero penetration bias) so this tests that
        // clamp rather than incorrectly capping post-solver correction.
        Scene unequal_mass;
        const auto light = unequal_mass.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1.0e-6f);
        const auto heavy = unequal_mass.add_dynamic_box(
            {1,1,1}, {2.0f,5,0}, 1.0e6f);
        const float max_v = std::numeric_limits<float>::max();
        unequal_mass.set_velocity(light, {max_v,0,0});
        unequal_mass.set_velocity(heavy, {-max_v,0,0});
        unequal_mass.step(std::numeric_limits<float>::min());
        const auto vlight = unequal_mass.body_state(light).velocity;
        const auto vheavy = unequal_mass.body_state(heavy).velocity;
        expect(std::isfinite(vlight.x) && std::isfinite(vlight.y) &&
               std::isfinite(vlight.z) && std::isfinite(vheavy.x) &&
               std::isfinite(vheavy.y) && std::isfinite(vheavy.z),
               "Unequal-mass contact published infinite or NaN velocity");
        expect(std::abs(vlight.x) < 2.0e16f &&
               std::abs(vheavy.x) < 2.0e16f &&
               unequal_mass.contacts().size() == 1,
               "Source unconstrained linear-speed clamp failed zero-bias contact");
    }
    {
        // A free body uses the same source default velocity clamp, so
        // safety is independent of whether contacts are generated.
        Scene unopposed;
        const auto b = unopposed.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1);
        unopposed.set_velocity(b, {2.0e38f,2.0e38f,2.0e38f});
        unopposed.step(std::numeric_limits<float>::min());
        const auto vel = unopposed.body_state(b).velocity;
        // PhysX's PxReal squared speed overflows to infinity here.
        // Consequently sqrt(1e32f / inf) is zero, not 1e16 normalization.
        expect(vel.x == 0.0f && vel.y == 0.0f && vel.z == 0.0f,
               "Native high-speed clamp diverged from PhysX float arithmetic");
        Scene finite_limit;
        const auto limited = finite_limit.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1);
        finite_limit.set_velocity(limited, {2.0e16f,0,0});
        finite_limit.step(1.0f / 120.0f);
        const auto moderate = finite_limit.body_state(limited).velocity;
        expect(std::isfinite(moderate.x) &&
               std::abs(moderate.x - 1.0e16f) < 1.0e10f,
               "Finite squared velocity failed PhysX source clamp");
    }
    {
        // Velocity authoring has no full-manifold invalidation scan:
        // an isolated edit changes only the selected actor's velocity.
        Scene authoring;
        const auto static_id = authoring.add_static_floor(0);
        const auto dynamic_id = authoring.add_dynamic_box(
            {0.25f,0.25f,0.25f}, {0,1,0}, 1);
        authoring.set_velocity(dynamic_id, {2,0,3});
        expect(authoring.body_state(dynamic_id).velocity.x == 2,
               "Native dynamic-body velocity setter failed");
        bool rejected = false;
        try { authoring.set_velocity(static_id, {1,0,0}); }
        catch (const std::logic_error&) { rejected = true; }
        expect(rejected, "Native static body accepted linear velocity");
        bool invalid = false;
        try { authoring.set_velocity(dynamic_id, {INFINITY,0,0}); }
        catch (const std::invalid_argument&) { invalid = true; }
        expect(invalid, "Native scene accepted nonfinite velocity");
    }

    {
        // Source inertia is directional and density-scaled. The same
        // world torque impulse must produce different angular velocity
        // for a rectangular body on its distinct principal axes.
        Scene torque;
        const auto id = torque.add_dynamic_box(
            {0.5f,1.0f,2.0f}, {0,10,0}, 2.0f);
        torque.apply_torque_impulse(id, {1,1,1});
        torque.step(1.0f/120.0f);
        const auto w = torque.body_state(id).angular_velocity;
        // 8*0.5*1*2*2 = 16 kg; inertia = mass/3*(hy^2+hz^2...).
        const float angular_damp = 1.0f-0.05f/120.0f;
        const float ix=16.0f*(1.0f+4.0f)/3.0f;
        const float iy=16.0f*(0.25f+4.0f)/3.0f;
        const float iz=16.0f*(0.25f+1.0f)/3.0f;
        expect(std::abs(w.x-angular_damp/ix)<0.0001f &&
               std::abs(w.y-angular_damp/iy)<0.0001f &&
               std::abs(w.z-angular_damp/iz)<0.0001f &&
               w.x < w.y && w.y < w.z,
               "Native box impulse failed source mass/inertia axes");
        torque.step(1.0f/120.0f);
        const auto twice=torque.body_state(id).angular_velocity;
        expect(twice.z < w.z,
               "Per-step angular impulse was replayed on the next frame");
        bool invalid=false;
        try {torque.apply_torque_impulse(id,{INFINITY,0,0});}
        catch(const std::invalid_argument&) {invalid=true;}
        expect(invalid,"Native torque accepted nonfinite input");
        const auto floor_id=torque.add_static_floor(0);
        bool rejected=false;
        try {torque.apply_torque_impulse(floor_id,{1,0,0});}
        catch(const std::logic_error&) {rejected=true;}
        expect(rejected,"Static floor accepted angular impulse");
    }
    {
        // Queued angular impulses are consumed at the next scene step
        // even when static contacts are present. Normal contact rows
        // must now accept the resulting rotated body without replaying
        // the pending torque on later frames.
        Scene rotating_contact;
        const auto id = rotating_contact.add_dynamic_box(
            {0.5f,1.0f,2.0f},{0,2,0},1);
        rotating_contact.add_static_floor(0);
        rotating_contact.apply_torque_impulse(id,{1,0,0});
        const auto before = rotating_contact.body_state(id);
        rotating_contact.step(1.0f/120.0f);
        const auto after = rotating_contact.body_state(id);
        expect(after.position.y < before.position.y &&
               after.angular_velocity.x > before.angular_velocity.x &&
               std::isfinite(after.angular_velocity.x),
               "Queued angular impulse was not consumed in active scene");
        rotating_contact.step(1.0f/120.0f);
        expect(std::isfinite(rotating_contact.body_state(id).angular_velocity.x),
               "Angular motion failed to continue without replaying torque");
    }
    {
        // Source ScBodyCore creation counter and Dy::sleepCheck: motion
        // readiness precedes whole-island deactivation. Once a rigid
        // body rests on its static floor it must stop integrating gravity
        // and remain stationary, but an explicit velocity edit wakes it.
        Scene sleeping;
        sleeping.add_static_floor(0.0f);
        const auto id=sleeping.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        expect(std::abs(sleeping.body_wake_counter(id)-0.4f)<1e-6f &&
               !sleeping.body_ready_for_sleep(id) &&
               !sleeping.body_is_sleeping(id),
               "Source rigid-body creation did not initialize wake counter");
        for(int frame=0;frame<360;++frame)
            sleeping.step(1.0f/120.0f);
        const auto still=sleeping.body_state(id);
        expect(sleeping.body_is_sleeping(id) &&
               sleeping.body_ready_for_sleep(id) &&
               still.sleeping &&
               sleeping.body_wake_counter(id)==0.0f &&
               still.velocity.y==0.0f &&
               sleeping.solver_partition_count()==0,
               "PhysX island did not deactivate a settled floor body");
        for(double impulse:sleeping.solved_normal_impulses())
            expect(impulse==0.0,
                   "Sleeping contact manager emitted an active solver impulse");
        // A failed later step must restore all existing sleep state,
        // not just actor motion and topology. An independently admitted
        // awake body forces a finite-input overflow before publication.
        const auto awake=sleeping.add_dynamic_box(
            {0.25f,0.25f,0.25f},{20,10,0},1.0f);
        const auto before_failure=sleeping.body_state(awake);
        bool failed=false;
        try {sleeping.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        const auto after_failure=sleeping.body_state(awake);
        expect(failed && sleeping.body_is_sleeping(id) &&
               sleeping.body_wake_counter(id)==0.0f &&
               sleeping.body_state(id).sleeping &&
               after_failure.position.y==before_failure.position.y &&
               sleeping.body_wake_counter(awake)==0.4f,
               "Failed source step corrupted dormant island or wake counter");
        sleeping.set_velocity(id,{1,0,0});
        expect(!sleeping.body_is_sleeping(id) &&
               !sleeping.body_ready_for_sleep(id) &&
               sleeping.body_wake_counter(id)==0.4f,
               "Source velocity authoring did not wake the rigid body");
        sleeping.step(1.0f/120.0f);
        const auto moved=sleeping.body_state(id);
        expect(!moved.sleeping &&
               moved.position.x>still.position.x &&
               std::isfinite(moved.velocity.x),
               "Woken source island failed to reenter native solver");
    }

    {
        // PxsIslandSim::processLostEdges(third pass) consumes the *prior*
        // frame's Dy::sleepCheck readiness. The frame in which the body
        // first becomes ready is not allowed to deactivate it immediately.
        // The next third pass then deactivates before gravity and PGS:
        // that frame must not integrate one extra position increment.
        Scene phase_boundary;
        phase_boundary.add_static_floor(0.0f);
        const auto sleeper=phase_boundary.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool saw_ready_before_sleep=false;
        for(int frame=0;frame<600;++frame) {
            phase_boundary.step(1.0f/120.0f);
            if(phase_boundary.body_ready_for_sleep(sleeper)) {
                expect(!phase_boundary.body_is_sleeping(sleeper),
                       "New Dy sleep readiness deactivated during its own solver frame");
                saw_ready_before_sleep=true;
                const auto ready_pose=phase_boundary.body_state(sleeper);
                phase_boundary.step(1.0f/120.0f);
                const auto inactive=phase_boundary.body_state(sleeper);
                expect(inactive.sleeping &&
                       inactive.position.x==ready_pose.position.x &&
                       inactive.position.y==ready_pose.position.y &&
                       inactive.position.z==ready_pose.position.z &&
                       inactive.velocity.x==0.0f &&
                       inactive.velocity.y==0.0f &&
                       inactive.velocity.z==0.0f,
                       "Third-pass deactivation integrated an already-ready island");
                break;
            }
        }
        expect(saw_ready_before_sleep,
               "Source third-pass fixture never produced Dy sleep readiness");
    }

    // A user-authored dynamic teleport can remove a contact edge and
    // split the original island before its waking node is propagated.
    // Its dormant former contact partner must be reactivated directly
    // from the previously published Gu manager, just as for a static
    // support edit, even when the new topology no longer connects them.
    {
        Scene moved_peer;
        moved_peer.add_static_floor(0.0f);
        const auto left=moved_peer.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=moved_peer.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            moved_peer.step(1.0f/120.0f);
        expect(moved_peer.body_is_sleeping(left) &&
               moved_peer.body_is_sleeping(right) &&
               moved_peer.accurate_island_id(left)==
                   moved_peer.accurate_island_id(right),
               "Dynamic support edit fixture lacked dormant contact island");
        moved_peer.set_position(left,{-10,0.5f,0});
        expect(!moved_peer.body_is_sleeping(right) &&
               !moved_peer.body_ready_for_sleep(right) &&
               moved_peer.body_wake_counter(right)>0.0f,
               "Dynamic pose edit left old contact peer dormant");
        moved_peer.step(1.0f/120.0f);
        expect(moved_peer.accurate_island_id(left)!=
                   moved_peer.accurate_island_id(right) &&
               !moved_peer.body_is_sleeping(right),
               "Contact island split discarded the old peer wake");
    }

    // Third-pass sleep readiness belongs to the previous solver frame.
    // A support edit between the ready and inactive passes must revoke
    // the ready flag while the body is still awake. Verify both static
    // movement and retirement, with and without an intermediate
    // collision-only publication. These cases must integrate gravity on
    // their next actual step, not suspend the body in midair.
    for(bool retire_support : {false,true}) {
        for(bool query_first : {false,true}) {
            Scene unsupported;
            const auto floor=unsupported.add_static_floor(0.0f);
            const auto body=unsupported.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            bool became_ready=false;
            for(int frame=0;frame<600;++frame) {
                unsupported.step(1.0f/120.0f);
                if(unsupported.body_ready_for_sleep(body)) {
                    became_ready=true;
                    break;
                }
            }
            expect(became_ready && !unsupported.body_is_sleeping(body),
                   "Support-edit fixture missed ready-but-awake third-pass boundary");
            const auto settled=unsupported.body_state(body);
            if(retire_support)
                unsupported.remove(floor);
            else
                unsupported.set_position(floor,{0,-10,0});
            expect(!unsupported.body_ready_for_sleep(body) &&
                   !unsupported.body_is_sleeping(body) &&
                   unsupported.body_wake_counter(body)>0.0f,
                   "Support edit retained stale ready-but-awake sleep state");
            if(query_first) {
                unsupported.advance_collision();
                expect(!unsupported.body_ready_for_sleep(body) &&
                       !unsupported.body_is_sleeping(body),
                       "Collision query restored obsolete third-pass readiness");
            }
            unsupported.step(1.0f/120.0f);
            const auto falling=unsupported.body_state(body);
            expect(!falling.sleeping &&
                   !unsupported.body_ready_for_sleep(body) &&
                   falling.position.y<settled.position.y-1.0e-5f &&
                   falling.velocity.y<0.0f,
                   "Support edit deactivated ready actor before gravity");
        }
    }

    {
        // A geometry-only query also publishes changed island membership
        // and may consume an authored node wake, without a numerical
        // solver step. Its accurate active-node index must not vanish
        // after a graph rebuild or remain stale after an island split.
        Scene queried;
        queried.add_static_floor(0.0f);
        const auto sleeper=queried.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.accurate_active_node_index(sleeper)==0,
               "Geometry-only island admission did not index its active rigid");
        for(int frame=0;frame<360;++frame)
            queried.step(1.0f/120.0f);
        expect(queried.body_is_sleeping(sleeper) &&
               queried.active_accurate_nodes().empty(),
               "Geometry query fixture never deactivated its first node");
        queried.set_velocity(sleeper,{0.25f,0,0});
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.active_accurate_nodes()[0]==sleeper &&
               queried.accurate_active_node_index(sleeper)==0,
               "Quiet authored wake failed to update the accurate active map");
        const auto added=queried.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==2 &&
               queried.accurate_active_node_index(added)<
                   queried.active_accurate_nodes().size(),
               "Geometry-only rebuild lost an active rigid node");
        queried.remove(sleeper);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.active_accurate_nodes()[0]==added &&
               queried.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               queried.accurate_active_node_index(added)==0,
               "Actor retirement left a stale active-node reverse index");
    }

    {
        // Rejected steps must preserve retained source active-node
        // spans, their actual storage, and their published elements.
        // A failure after a newly ready body deactivates exercises
        // swap-last active index retirement before rollback.
        Scene retained_active;
        retained_active.add_static_floor(0.0f);
        const auto sleeper=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            retained_active.step(1.0f/120.0f);
            if(retained_active.body_ready_for_sleep(sleeper)) {
                ready=true;
                break;
            }
        }
        expect(ready && !retained_active.body_is_sleeping(sleeper),
               "Retained active span fixture did not become ready");
        const auto published=retained_active.active_accurate_nodes();
        const auto published_deactivations=
            retained_active.deactivating_accurate_nodes();
        expect(published.size()==1 && published[0]==sleeper &&
               published_deactivations.empty(),
               "Retained active span fixture was not an active rigid");
        const auto* const published_ptr=published.data();
        const auto* const deact_ptr=published_deactivations.data();
        // The ready body will deactivate at the next third pass, so a
        // second, unconnected active body is required to force the
        // subsequent integration to reject an enormous timestep.
        const auto overflow_body=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10,5,0},1.0f);
        const auto frame=retained_active.frame_index();
        for(int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {retained_active.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected && retained_active.frame_index()==frame &&
                   !retained_active.body_is_sleeping(sleeper) &&
                   retained_active.active_accurate_nodes().data()==
                       published_ptr &&
                   retained_active.deactivating_accurate_nodes().data()==
                       deact_ptr &&
                   published.size()==1 && published[0]==sleeper &&
                   retained_active.active_accurate_nodes().size()==1 &&
                   retained_active.accurate_active_node_index(sleeper)==0 &&
                   retained_active.deactivating_accurate_nodes().empty(),
                   "Failed third-pass deactivation invalidated active-node span");
        }
        retained_active.step(1.0f/120.0f);
        expect(retained_active.body_is_sleeping(sleeper) &&
               retained_active.deactivating_accurate_nodes().size()==1 &&
               retained_active.deactivating_accurate_nodes()[0]==sleeper,
               "Valid retry failed to emit accurate third-pass deactivation");
        // The next rejected frame can rebuild topology after a newly
        // admitted awake body; retained deactivation storage remains
        // readable even if the working graph was replaced.
        const auto old_output=retained_active.deactivating_accurate_nodes();
        const auto* const old_output_ptr=old_output.data();
        const auto newcomer=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        const auto next_frame=retained_active.frame_index();
        bool rejected=false;
        try {retained_active.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected && retained_active.frame_index()==next_frame &&
               old_output.data()==old_output_ptr && old_output.size()==1 &&
               old_output[0]==sleeper &&
               retained_active.deactivating_accurate_nodes().data()==
                   old_output_ptr &&
               retained_active.deactivating_accurate_nodes().size()==1 &&
               retained_active.deactivating_accurate_nodes()[0]==sleeper,
               "Rejected island rebuild invalidated deactivation report span");
        retained_active.step(1.0f/120.0f);
        const auto active_after_retry=retained_active.active_accurate_nodes();
        expect(retained_active.deactivating_accurate_nodes().empty() &&
               active_after_retry.size()==2 &&
               std::find(active_after_retry.begin(),active_after_retry.end(),
                         overflow_body)!=active_after_retry.end() &&
               std::find(active_after_retry.begin(),active_after_retry.end(),
                         newcomer)!=active_after_retry.end(),
               "Post-rollback deactivation output was replayed on retry");
    }

    {
        // PxsIslandSim::wakeIslands captures its initially active rigid
        // prefix before admitting newly activated nodes. Deactivating
        // an initial member must use source markInactive's prefix-tail
        // relocation so only genuinely new rigid nodes remain in
        // getNodesToActivate's active suffix. Rebuilding the graph for
        // actor admission must preserve the existing active permutation.
        Scene partitions;
        partitions.add_static_floor(0.0f);
        const auto sleeper=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto flyer=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10,10,0},1.0f);
        partitions.advance_collision();
        expect(partitions.newly_activated_accurate_nodes().size()==2,
               "First source wake did not admit both new rigid bodies");
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            // Keep the independent free-flight body awake while the
            // resting island reaches its prior-frame sleep boundary.
            partitions.set_velocity(flyer,{0,9.81f,0});
            partitions.step(1.0f/120.0f);
            if(partitions.body_ready_for_sleep(sleeper)) {
                ready=true;
                break;
            }
        }
        expect(ready && !partitions.body_is_sleeping(sleeper),
               "Initial-active partition fixture did not reach readiness");
        // Admit a new island in the very pass where the old floor
        // island deactivates. The original two-node initial prefix must
        // shrink to the still-active flyer, while the newcomer belongs
        // strictly to the newly activated suffix. This fails if
        // markInactive uses unconditional swap-last without the
        // source initial-partition tail permutation.
        const auto newcomer=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,10,0},1.0f);
        partitions.set_velocity(flyer,{0,9.81f,0});
        partitions.step(1.0f/120.0f);
        expect(partitions.body_is_sleeping(sleeper) &&
               partitions.deactivating_accurate_nodes().size()==1 &&
               partitions.deactivating_accurate_nodes()[0]==sleeper &&
               partitions.active_accurate_nodes().size()==2 &&
               partitions.active_accurate_nodes()[0]==flyer &&
               partitions.active_accurate_nodes()[1]==newcomer &&
               partitions.newly_activated_accurate_nodes().size()==1 &&
               partitions.newly_activated_accurate_nodes()[0]==newcomer &&
               partitions.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               partitions.accurate_active_node_index(flyer)==0 &&
               partitions.accurate_active_node_index(newcomer)==1,
               "Simultaneous source markInactive/admission broke initial-active partition");
        const auto published=partitions.active_accurate_nodes();
        const auto prior_new=partitions.newly_activated_accurate_nodes();
        const auto* const original=published.data();
        bool failed=false;
        try {partitions.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed &&
               partitions.active_accurate_nodes().data()==original &&
               published[0]==flyer && published[1]==newcomer &&
               prior_new.size()==1 && prior_new[0]==newcomer,
               "Rejected frame damaged published active-node partitions");
        partitions.set_velocity(flyer,{0,9.81f,0});
        partitions.step(1.0f/120.0f);
        expect(partitions.newly_activated_accurate_nodes().empty() &&
               partitions.active_accurate_nodes().size()==2,
               "Previous initial-active suffix replayed on next frame");
    }

    verify_idempotent_pose_autowake();
    verify_active_pose_preserves_sleep_filter();
    verify_simultaneous_initial_sleep_compaction();
    verify_speculative_active_node_lifecycle();

    {
        // Source accurate active-node indexing must stay dense and
        // bidirectional through third-pass sleep, user wake, and
        // reactivation. The separate deactivation output is published
        // for exactly the boundary in which ready nodes go inactive.
        Scene indexed;
        indexed.add_static_floor(0.0f);
        const auto body=indexed.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        indexed.step(1.0f/120.0f);
        expect(indexed.active_accurate_nodes().size()==1 &&
               indexed.active_accurate_nodes()[0]==body &&
               indexed.accurate_active_node_index(body)==0,
               "Source active-node index missed the initially awake body");
        bool deactivated=false;
        for(int frame=0;frame<600;++frame) {
            indexed.step(1.0f/120.0f);
            if(indexed.body_is_sleeping(body)) {
                const auto retired=indexed.deactivating_accurate_nodes();
                expect(retired.size()==1 && retired[0]==body &&
                       indexed.active_accurate_nodes().empty() &&
                       indexed.accurate_active_node_index(body)==
                           std::numeric_limits<std::size_t>::max(),
                       "Third pass did not publish indexed rigid deactivation");
                deactivated=true;
                break;
            }
        }
        expect(deactivated,"Source indexed active node never slept");
        // A public collision-only query consumes a new publication clock
        // without invoking another ready-island transition. Source
        // mNodesToPutToSleep is per-pass output, not sticky sleep state.
        indexed.advance_collision();
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().empty() &&
               indexed.accurate_active_node_index(body)==
                   std::numeric_limits<std::size_t>::max(),
               "Quiet collision publication replayed a deactivation event");
        indexed.advance_collision();
        expect(indexed.deactivating_accurate_nodes().empty(),
               "Repeated quiet publication resurrected an old deactivation");
        indexed.step(1.0f/120.0f);
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().empty(),
               "Dormant third pass repeated an already published deactivation");
        indexed.set_velocity(body,{1,0,0});
        indexed.step(1.0f/120.0f);
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().size()==1 &&
               indexed.active_accurate_nodes()[0]==body &&
               indexed.accurate_active_node_index(body)==0,
               "Source node activation did not rebuild the active index");
    }

    {
        // The source active-island list is a strict subset of all accurate
        // islands. Dormant bodies remain in the persistent source graph
        // but are excluded from unconstrained integration and PGS. An
        // isolated awake node must retain its own active island batch.
        Scene active_handoff;
        active_handoff.add_static_floor(0.0f);
        const auto resting=active_handoff.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto flyer=active_handoff.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        active_handoff.advance_collision();
        expect(active_handoff.accurate_island_count()==2,
               "Active-island fixture did not admit both source islands");
        // Isolate the dormant transition from the free-flight wake state:
        // first sleep the floor body, then introduce the distant flyer.
        Scene staged;
        staged.add_static_floor(0.0f);
        const auto dormant=staged.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            staged.step(1.0f/120.0f);
        expect(staged.body_is_sleeping(dormant) &&
               staged.active_accurate_island_count()==0,
               "Dormant accurate island remained on Dy active list");
        const auto dynamic=staged.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        staged.step(1.0f/120.0f);
        expect(staged.accurate_island_count()==2 &&
               staged.active_accurate_island_count()==1 &&
               staged.body_is_sleeping(dormant) &&
               staged.body_state(dynamic).position.y<5.0f,
               "Dy active-island handoff admitted sleeper or skipped free flight");
        staged.set_velocity(dormant,{1,0,0});
        staged.step(1.0f/120.0f);
        expect(staged.active_accurate_island_count()==2 &&
               !staged.body_is_sleeping(dormant),
               "Wake did not restore accurate active-island admission");
        (void)resting;
        (void)flyer;
    }

    // A quiet collision query must consume genuinely authored source
    // wake requests without scanning the island graph for every query.
    // Ensure a rejected step does not clear that pending wake indicator:
    // the next quiet query still propagates to the sleeping island peer.
    {
        Scene pending_query;
        pending_query.add_static_floor(0.0f);
        const auto left=pending_query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=pending_query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int i=0;i<360;++i)pending_query.step(1.0f/120.0f);
        pending_query.advance_collision();
        expect(pending_query.body_is_sleeping(left) &&
               pending_query.body_is_sleeping(right),
               "Quiet wake indicator fixture failed to settle");
        const auto revision=pending_query.island_topology_revision();
        for(int i=0;i<4;++i)pending_query.advance_collision();
        expect(pending_query.island_topology_revision()==revision,
               "Quiet collision queries modified source island topology");
        pending_query.set_velocity(left,{0.3f,0,0});
        bool rejected=false;
        try {pending_query.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected && pending_query.body_is_sleeping(right),
               "Rejected frame corrupted queued wake");
        pending_query.advance_collision();
        expect(!pending_query.body_is_sleeping(right) &&
               pending_query.island_topology_revision()==revision,
               "Queued authored wake was lost after quiet rollback");
    }

    {
        // A rejected Dy step rebuilds the active accurate island view
        // after source wake/deactivation. The previous published active
        // batch count must survive rollback without copying the list.
        Scene active_rollback;
        active_rollback.add_static_floor(0.0f);
        const auto sleeper=active_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int i=0;i<360;++i)active_rollback.step(1.0f/120.0f);
        expect(active_rollback.active_accurate_island_count()==0 &&
               active_rollback.body_is_sleeping(sleeper),
               "Active-batch rollback fixture did not become dormant");
        active_rollback.set_velocity(sleeper,{1,0,0});
        // Prior published active view remains empty until the next
        // simulation publication, despite the authored actor wake.
        bool rejected=false;
        try {active_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               active_rollback.active_accurate_island_count()==0 &&
               active_rollback.active_accurate_nodes().empty() &&
               active_rollback.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               active_rollback.deactivating_accurate_nodes().empty() &&
               !active_rollback.body_is_sleeping(sleeper),
               "Rejected Dy frame replaced the published active-node/index state");
        active_rollback.step(1.0f/120.0f);
        expect(active_rollback.active_accurate_island_count()==1 &&
               active_rollback.active_accurate_nodes().size()==1 &&
               active_rollback.accurate_active_node_index(sleeper)==0,
               "Active-island retry did not republish the indexed source node");
    }

    {
        // IslandSim::activateNode clears an already-awake node's ready
        // flag when a new contact wakes its island. A body that just
        // reached readiness must still resolve the newly added static
        // contact, not sleep before submitting the new Gu rows.
        Scene new_touch;
        new_touch.add_static_floor(0.0f);
        const auto candidate=new_touch.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            new_touch.step(1.0f/120.0f);
            if(new_touch.body_ready_for_sleep(candidate)) {
                ready=true;
                break;
            }
        }
        expect(ready && !new_touch.body_is_sleeping(candidate),
               "New-touch activation fixture missed the ready-before-sleep frame");
        const float previous_x=new_touch.body_state(candidate).position.x;
        new_touch.add_static_box(
            {0.5f,0.5f,0.5f},{previous_x+0.9f,0.5f,0});
        new_touch.step(1.0f/120.0f);
        const auto moved=new_touch.body_state(candidate);
        expect(!moved.sleeping &&
               moved.position.x<previous_x-0.001f,
               "New Gu touch failed to revoke an awake node's sleep readiness");
    }

    {
        // IslandSim::activateIslandInternal wakes connected rigid nodes,
        // not only the node explicitly edited by a user. A persistent
        // dynamic contact must admit both nodes together on reactivation.
        Scene coupled_sleep;
        coupled_sleep.add_static_floor(0.0f);
        const auto left=coupled_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=coupled_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            coupled_sleep.step(1.0f/120.0f);
        expect(coupled_sleep.accurate_island_count()==1 &&
               coupled_sleep.body_is_sleeping(left) &&
               coupled_sleep.body_is_sleeping(right),
               "Source dynamic contact island did not deactivate together");
        coupled_sleep.set_velocity(left,{0.5f,0,0});
        expect(!coupled_sleep.body_is_sleeping(left) &&
               coupled_sleep.body_is_sleeping(right),
               "Direct user wake bypassed source island activation stage");
        coupled_sleep.step(1.0f/120.0f);
        expect(!coupled_sleep.body_is_sleeping(left) &&
               !coupled_sleep.body_is_sleeping(right),
               "Source island wake did not reactivate contact-connected peer");
    }

    {
        // IslandSim::wakeIslands consumes an actual authored activation,
        // including when a collision-only query is otherwise quiet.
        // An already-awake neighbor is not itself a perpetual wake event.
        Scene queued_wake;
        queued_wake.add_static_floor(0.0f);
        const auto left=queued_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=queued_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            queued_wake.step(1.0f/120.0f);
        expect(queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Activation queue fixture did not reach dormant island");
        queued_wake.advance_collision();
        expect(queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Quiet query spuriously activated a dormant island");
        queued_wake.set_velocity(left,{0.3f,0,0});
        expect(!queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Authored activation did not retain per-node handoff");
        // A failed Dy frame may consume an activating node while
        // traversing the speculative island. The checkpoint must restore
        // both the dormant peer and the queued activating flag so the
        // subsequent collision-only query still wakes the peer.
        const auto queued_frame=queued_wake.frame_index();
        for(int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {queued_wake.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected &&
                   queued_wake.frame_index()==queued_frame &&
                   !queued_wake.body_is_sleeping(left) &&
                   queued_wake.body_is_sleeping(right),
                   "Failed Dy frame consumed a pending source activation");
        }
        queued_wake.advance_collision();
        expect(!queued_wake.body_is_sleeping(right),
               "Quiet collision publication discarded the queued source wake");
    }

    {
        // The speculative graph may contain a pair of overlapping AABBs
        // whose rotated Gu boxes never touch. PhysX's third pass uses
        // accurate islands to decide rigid-body deactivation; only
        // contact interactions use speculative deactivation state.
        // Keep one rod awake without translating far enough to destroy
        // the SAP overlap, while its accurate neighbor is allowed to sleep.
        Scene distinct_sleep;
        const auto rod_a=distinct_sleep.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto rod_b=distinct_sleep.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(fyzz::BodyId id:{rod_a,rod_b}) {
            distinct_sleep.set_velocity(id,{0,9.81f,0});
            distinct_sleep.set_angular_velocity(id,{0,0.83f,0});
        }
        distinct_sleep.step(1.0f);
        for(fyzz::BodyId id:{rod_a,rod_b}) {
            distinct_sleep.set_angular_velocity(id,{});
            distinct_sleep.set_velocity(id,{});
        }
        distinct_sleep.set_position(rod_a,{0,0.1f,0});
        distinct_sleep.set_position(rod_b,{0.6f,0.1f,0.6f});
        const auto source_floor=distinct_sleep.add_static_floor(0.0f);
        distinct_sleep.advance_collision();
        expect(has(distinct_sleep.pairs(),rod_a,rod_b) &&
               distinct_sleep.speculative_island_count()==1 &&
               distinct_sleep.accurate_island_count()==2,
               "Sleep fixture failed to separate speculative and accurate islands");
        bool independent_sleep=false;
        for(int frame=0;frame<160;++frame) {
            distinct_sleep.set_velocity(rod_b,{0,0,0.25f});
            distinct_sleep.step(1.0f/120.0f);
            if(distinct_sleep.body_is_sleeping(rod_a) &&
               !distinct_sleep.body_is_sleeping(rod_b) &&
               distinct_sleep.speculative_island_count()==1 &&
               distinct_sleep.accurate_island_count()==2) {
                const auto active=distinct_sleep.active_accurate_nodes();
                expect(active.size()==1 && active[0]==rod_b &&
                       distinct_sleep.accurate_active_node_index(rod_b)==0 &&
                       distinct_sleep.accurate_active_node_index(rod_a)==
                           std::numeric_limits<std::size_t>::max(),
                       "Accurate active node list included dormant speculative-only peer");
                const auto floor_edge=distinct_sleep.contact_edge_index(
                    fyzz::native::Pair{rod_a,source_floor});
                const auto accurate_events=
                    distinct_sleep.deactivating_accurate_contact_edges();
                expect(floor_edge!=std::numeric_limits<std::uint32_t>::max(),
                       "Independent accurate sleep lost floor manager handle");
                expect(std::count(accurate_events.begin(),
                                  accurate_events.end(),floor_edge)==1 &&
                       !distinct_sleep.accurate_contact_edge_active(
                           fyzz::native::Pair{rod_a,source_floor}) &&
                       distinct_sleep.speculative_contact_edge_active(
                           fyzz::native::Pair{rod_a,source_floor}) &&
                       distinct_sleep.contact_interaction_active(
                           fyzz::native::Pair{rod_a,source_floor}),
                       "Accurate sleep did not transition only its active edge");
                expect(distinct_sleep.deactivating_speculative_contact_edges().empty(),
                       "Speculative sleep edges published while its island remained awake");
                independent_sleep=true;
                break;
            }
        }
        expect(independent_sleep,
               "Accurate rigid island could not deactivate independently of SAP candidate");
        // Source processLostEdges also deactivates a speculative island
        // that becomes entirely inactive after its awake SAP-only bridge
        // is removed. No new accurate rigid node falls asleep in this
        // publication, so sourcing candidates solely from the accurate
        // deactivation output misses this edge event.
        const auto sleeping_floor=distinct_sleep.contact_edge_index(
            fyzz::native::Pair{rod_a,source_floor});
        distinct_sleep.set_position(rod_b,{12.0f,0.1f,12.0f});
        distinct_sleep.advance_collision();
        const auto speculative_after_split=
            distinct_sleep.deactivating_speculative_contact_edges();
        expect(has(distinct_sleep.lost_pairs(),rod_a,rod_b) &&
               distinct_sleep.speculative_island_count()==2 &&
               distinct_sleep.body_is_sleeping(rod_a) &&
               !distinct_sleep.body_is_sleeping(rod_b) &&
               sleeping_floor!=std::numeric_limits<std::uint32_t>::max() &&
               std::count(speculative_after_split.begin(),
                          speculative_after_split.end(),sleeping_floor)==1 &&
               distinct_sleep.deactivating_accurate_contact_edges().empty(),
               "Lost SAP-only bridge missed source speculative island sleep event");
        distinct_sleep.advance_collision();
        expect(distinct_sleep.deactivating_speculative_contact_edges().empty() &&
               !distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               !distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}),
               "Quiet publication replayed an already inactive edge");
        const auto before_wake_spec=
            distinct_sleep.active_speculative_contact_edge_count();
        const auto before_wake_acc=
            distinct_sleep.active_accurate_contact_edge_count();
        distinct_sleep.set_velocity(rod_a,{0,0,0.25f});
        distinct_sleep.advance_collision();
        const auto activated_spec=
            distinct_sleep.activated_speculative_contact_edges();
        const auto activated_acc=
            distinct_sleep.activated_accurate_contact_edges();
        expect(distinct_sleep.contact_edge_index(
                   fyzz::native::Pair{rod_a,source_floor})==sleeping_floor &&
               distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               std::count(activated_spec.begin(),activated_spec.end(),
                          sleeping_floor)==1 &&
               std::count(activated_acc.begin(),activated_acc.end(),
                          sleeping_floor)==1 &&
               distinct_sleep.active_speculative_contact_edge_count()==
                   before_wake_spec+1 &&
               distinct_sleep.active_accurate_contact_edge_count()==
                   before_wake_acc+1,
               "Retained manager did not publish source edge activation");
        distinct_sleep.advance_collision();
        expect(distinct_sleep.activated_speculative_contact_edges().empty() &&
               distinct_sleep.activated_accurate_contact_edges().empty() &&
               distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}),
               "eACTIVATING was replayed or persistent eACTIVE was lost");
        // A complete membership rebuild may change island handles but
        // must not reset Edge::eACTIVE or mActiveEdgeCount. The new
        // distant actor creates no accurate contact with the old floor.
        distinct_sleep.add_dynamic_box(
            {0.25f,0.25f,0.25f},{50.0f,50.0f,50.0f},1.0f);
        distinct_sleep.advance_collision();
        expect(distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.active_accurate_contact_edge_count()==
                   before_wake_acc+1,
               "Source edge active count was reset by actor admission");
        distinct_sleep.set_velocity(rod_a,{0,0,0.25f});
        distinct_sleep.advance_collision();
        expect(distinct_sleep.activated_accurate_contact_edges().empty() &&
               distinct_sleep.activated_speculative_contact_edges().empty(),
               "Already awake rigid body incorrectly reactivated active edges");
    }

    {
        // Newly published contacts and island activation precede
        // unconstrained velocity integration. On a later overflow,
        // failure atomicity must roll back both the fresh contact
        // manager and the previously sleeping peer's activation. The
        // identical valid retry must admit both bodies to the solver.
        Scene wake_rollback;
        wake_rollback.add_static_floor(0.0f);
        const auto sleeper=wake_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            wake_rollback.step(1.0f/120.0f);
        expect(wake_rollback.body_is_sleeping(sleeper),
               "Wake rollback source fixture did not reach sleep");
        const auto newcomer=wake_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1.0f);
        const auto prior_sleeper=wake_rollback.body_state(sleeper);
        const auto prior_newcomer=wake_rollback.body_state(newcomer);
        const auto prior_contacts=wake_rollback.contacts().size();
        bool rejected=false;
        try {wake_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               wake_rollback.body_is_sleeping(sleeper) &&
               wake_rollback.body_wake_counter(sleeper)==0.0f &&
               wake_rollback.body_state(sleeper).position.y==
                   prior_sleeper.position.y &&
               wake_rollback.body_state(newcomer).position.x==
                   prior_newcomer.position.x &&
               wake_rollback.contacts().size()==prior_contacts,
               "Failed activation frame published wake or new Gu contact");
        wake_rollback.step(1.0f/120.0f);
        expect(!wake_rollback.body_is_sleeping(sleeper) &&
               wake_rollback.accurate_island_id(sleeper)==
                   wake_rollback.accurate_island_id(newcomer),
               "Source activation rollback lost new-touch retry");
        // PxvDynamics.h initializes the unconfigured rigid body's
        // maxPenBias to -1e32f, not an implicit -3 m/s clamp.
        // The source default is effectively unbounded for this contact.
        // An incorrect finite -8 or -10 m/s clamp also must not pass.
        // The reference-enabled suite checks a stronger deep-contact
        // short-timestep case against actual PhysX legacy Gu/PGS.
        expect(wake_rollback.body_state(sleeper).position.x < -0.05f,
               "Source default penetration bias was spuriously capped");

    }

    {
        // Sc static-actor pose authoring invalidates interactions with
        // an already sleeping rigid body. Reusing the existing accurate
        // contact managers wakes the touched actor before subsequent
        // Gu/PGS progression; quiet static-world contacts do not.
        Scene moving_ground;
        const auto floor=moving_ground.add_static_floor(0.0f);
        const auto body=moving_ground.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            moving_ground.step(1.0f/120.0f);
        expect(moving_ground.body_is_sleeping(body),
               "Source static wake fixture never reached sleep");
        moving_ground.set_position(floor,{0,0.1f,0});
        expect(!moving_ground.body_is_sleeping(body),
               "Static pose change left its touching rigid body asleep");
        moving_ground.step(1.0f/120.0f);
        expect(!moving_ground.body_state(body).sleeping &&
               std::isfinite(moving_ground.body_state(body).position.y),
               "Native PGS could not advance rigid body after static wake");
    }

    {
        // A geometry-only Gu publication can create a static touch
        // before the next step. IslandSim::wakeIslands consumes it at
        // publication, not from a one-frame event stream that may be
        // cleared by a later quiet collision query.
        Scene collision_wake;
        collision_wake.add_static_floor(0);
        const auto body=collision_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1);
        for(int frame=0;frame<360;++frame)
            collision_wake.step(1.0f/120.0f);
        expect(collision_wake.body_is_sleeping(body),
               "Geometry-only wake fixture failed to reach sleep");
        collision_wake.add_static_box(
            {0.5f,0.5f,0.5f},{0.85f,0.5f,0});
        collision_wake.advance_collision();
        expect(!collision_wake.body_is_sleeping(body) &&
               collision_wake.accurate_island_static_touches(body)==2,
               "Gu-only publication lost the static contact wake");
        collision_wake.advance_collision();
        expect(!collision_wake.body_is_sleeping(body),
               "Quiet collision query lost previously published activation");
        collision_wake.step(1.0f/120.0f);
        expect(!collision_wake.body_state(body).sleeping &&
               std::isfinite(collision_wake.body_state(body).velocity.y),
               "Geometry-published source activation failed solver admission");
    }

    {
        // Sc::Scene destroys contact-manager links when a supporting
        // static actor is retired. A sleeping box must wake and resume
        // falling rather than remain indefinitely suspended in space.
        Scene removed_support;
        const auto floor=removed_support.add_static_floor(0);
        const auto box=removed_support.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            removed_support.step(1.0f/120.0f);
        expect(removed_support.body_is_sleeping(box),
               "Retired-support fixture failed to reach source sleep");
        const float old_y=removed_support.body_state(box).position.y;
        removed_support.remove(floor);
        expect(!removed_support.body_is_sleeping(box),
               "Retiring static support failed to reactivate sleeping box");
        for(int frame=0;frame<8;++frame)
            removed_support.step(1.0f/120.0f);
        expect(removed_support.body_state(box).position.y<old_y-0.005f &&
               removed_support.body_state(box).velocity.y<0.0f,
               "Source rigid body did not fall after losing static support");
    }

}

int run_native_scene_smoke() {
    using fyzz::native::Scene;
    Scene scene, independent;
    auto wall = scene.add_static_box({1,1,1}, {0,0,0});
    auto static2 = scene.add_static_box({1,1,1}, {1,0,0});
    auto moving = scene.add_dynamic_box({0.5f,0.5f,0.5f}, {5,0,0}, 1);
    auto second = independent.add_dynamic_box({0.5f,0.5f,0.5f},{1,1,1},1);
    scene.advance_collision();
    expect(scene.pairs().empty(), "static/static or distant pair admitted");
    scene.set_position(moving, {1.5f,0,0});
    scene.advance_collision();
    expect(has(scene.found_pairs(), wall, moving) &&
           has(scene.found_pairs(), static2, moving) &&
           scene.pairs().size() == 2 && scene.lost_pairs().empty(),
           "source-ordered found pairs absent");
    scene.advance_collision();
    expect(scene.found_pairs().empty() && scene.lost_pairs().empty(),
           "quiet frame changed incremental overlap state");
    scene.advance_collision(true);
    expect(scene.found_pairs().empty() && scene.lost_pairs().empty(),
           "forced unchanged update emitted pairs");
    scene.set_position(moving,{10,0,0});
    scene.advance_collision();
    expect(scene.pairs().empty() && scene.lost_pairs().size() == 2,
           "movement did not retire previous pair state");
    scene.set_position(moving,{1,0,0});
    scene.advance_collision();
    expect(scene.found_pairs().size() == 2, "overlap re-entry absent");
    scene.remove(wall);
    scene.advance_collision();
    expect(scene.lost_pairs().size() == 1 &&
           has(scene.lost_pairs(), wall, moving), "retired actor leaked pair");
    bool stale = false;
    try { (void)scene.body_state(wall); }
    catch (const std::out_of_range&) { stale = true; }
    expect(stale, "retired body ID still accessible");
    expect(scene.add_static_box({1,1,1},{100,0,0}) > moving,
           "retired body identifier reused");
    scene.advance_collision();
    expect(scene.found_pairs().empty(), "distant actor created false pair");
    expect(independent.frame_index() == 0 &&
           independent.body_state(second).position.x == 1,
           "separate native scene was modified");
    auto floor = independent.add_static_floor(-100);
    independent.set_position(floor, {0, 0.75f, 0});
    expect(independent.body_state(floor).position.y == 0.75f,
           "Native static plane failed to relocate");
    independent.advance_collision();
    expect(has(independent.found_pairs(), floor, second),
           "infinite floor broadphase candidate missing");
    independent.remove(second);
    independent.advance_collision();
    expect(has(independent.lost_pairs(), floor, second),
           "infinite floor lost pair missing");
    bool invalid = false;
    try { (void)scene.add_dynamic_box({1,1,1},{INFINITY,0,0},1); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "invalid native geometry admitted");
    expect(scene.frame_index() == 8, "scene frame counter drift");


    // An unforced quiet geometry publication has no throwing work and
    // must not checkpoint the complete edge/island state. It still
    // advances the publication frame, clears transient previous-step
    // results/events, and retains the original contact manager identity.
    {
        Scene quiet;
        const auto floor=quiet.add_static_floor(0.0f);
        const auto box=quiet.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        quiet.step(1.0f/120.0f);
        const fyzz::native::Pair pair{floor,box};
        const auto handle=quiet.contact_edge_index(pair);
        const auto revision=quiet.island_topology_revision();
        const auto rebuilds=quiet.island_full_rebuild_count();
        const auto start=quiet.frame_index();
        expect(!quiet.solved_normal_impulses().empty() &&
               handle!=std::numeric_limits<std::uint32_t>::max(),
               "Quiet collision fixture has no prior solved contact");
        for(int i=0;i<4;++i) {
            quiet.advance_collision();
            expect(quiet.frame_index()==start+static_cast<std::uint64_t>(i+1) &&
                   quiet.found_pairs().empty() && quiet.lost_pairs().empty() &&
                   quiet.found_contacts().empty() && quiet.lost_contacts().empty() &&
                   quiet.solved_normal_impulses().empty() &&
                   quiet.solved_friction_patches().empty() &&
                   quiet.force_threshold_changes().empty() &&
                   quiet.contact_edge_index(pair)==handle &&
                   quiet.contact_edge_connected(pair) &&
                   quiet.island_topology_revision()==revision &&
                   quiet.island_full_rebuild_count()==rebuilds,
                   "Quiet collision publication changed contact or island state");
        }
        quiet.advance_collision(true);
        expect(quiet.frame_index()==start+5 &&
               quiet.contact_edge_index(pair)==handle &&
               quiet.island_topology_revision()==revision,
               "Forced collision refresh lost persistent contact edge identity");
    }

    // SimpleIslandManager::addContactManager obtains an EdgeIndex once
    // per SAP candidate; setEdgeDisconnected does not retire it on Gu
    // touch loss. removeConnection defers release until the joined
    // post-third-pass retirement, including overlapping found/lost pairs.
    {
        Scene handles;
        const auto floor=handles.add_static_floor(0.0f);
        const auto first=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const fyzz::native::Pair first_pair{floor,first};
        handles.advance_collision();
        const auto first_edge=handles.contact_edge_index(first_pair);
        const auto invalid=std::numeric_limits<std::uint32_t>::max();
        expect(first_edge!=invalid &&
               handles.contact_edge_connected(first_pair) &&
               handles.active_contact_edge_count()==1,
               "PhysX contact manager did not acquire a persistent edge");
        handles.set_position(first,{0,8,0});
        handles.advance_collision();
        expect(handles.contact_edge_index(first_pair)==first_edge &&
               !handles.contact_edge_connected(first_pair) &&
               handles.active_contact_edge_count()==1,
               "Physical touch loss discarded the speculative EdgeIndex");
        handles.set_position(first,{0,0.5f,0});
        handles.advance_collision();
        expect(handles.contact_edge_index(first_pair)==first_edge &&
               handles.contact_edge_connected(first_pair),
               "Touch reentry allocated a different contact-manager edge");
        // Retiring one overlap and discovering another in the same
        // publication cannot reuse the old source handle prematurely.
        handles.remove(first);
        const auto second=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
        handles.advance_collision();
        const fyzz::native::Pair second_pair{floor,second};
        const auto second_edge=handles.contact_edge_index(second_pair);
        expect(second_edge!=invalid && second_edge!=first_edge &&
               handles.contact_edge_index(first_pair)==invalid &&
               handles.active_contact_edge_count()==1,
               "Removed contact edge was reused before third-pass retirement");
        handles.remove(second);
        const auto third=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{6,0.5f,0},1.0f);
        handles.advance_collision();
        const fyzz::native::Pair third_pair{floor,third};
        expect(handles.contact_edge_index(third_pair)==first_edge &&
               handles.contact_edge_connected(third_pair) &&
               handles.active_contact_edge_count()==1,
               "Retired contact edge handle was not returned to source pool");
    }

    // PxsSimpleIslandManager's two distinct IslandSim graphs, for the
    // currently admitted all-awake rigid-body subset. Broadphase
    // candidates build speculative connectivity, while only resolved
    // touching dynamic pairs join accurate solver islands. The static
    // world is never a dynamic connectivity bridge.
    {
        Scene connected;
        const auto a=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,5.0f,0.0f},1.0f);
        const auto b=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,5.0f,0.0f},1.0f);
        const auto c=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,5.0f,0.0f},1.0f);
        const auto isolated=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10.0f,5.0f,0.0f},1.0f);
        connected.advance_collision();
        expect(connected.contacts().size()==2 &&
               connected.speculative_island_count()==2 &&
               connected.accurate_island_count()==2,
               "Source first/second island graphs failed to merge the touching chain");
        const auto revision=connected.island_topology_revision();
        const auto initial_rebuilds=connected.island_full_rebuild_count();
        const auto initial_local_splits=connected.island_local_split_count();
        expect(initial_rebuilds==1,
               "First active island topology was not built once");
        connected.advance_collision(true);
        expect(connected.island_topology_revision()==revision,
               "Unchanged island connectivity rebuilt expensive graph state");
        // Removing the center's two dynamic contact edges must split the
        // component into three independent accurate islands, and the
        // still-alive isolated node contributes its own singleton.
        connected.set_position(b,{5.0f,5.0f,0.0f});
        connected.advance_collision();
        expect(connected.speculative_island_count()==4 &&
               connected.accurate_island_count()==4 &&
               connected.island_topology_revision()==revision+1 &&
               connected.island_full_rebuild_count()==initial_rebuilds &&
               connected.island_local_split_count()>initial_local_splits,
               "Lost source contact edges rebuilt unrelated island graphs");
        connected.set_position(b,{0.85f,5.0f,0.0f});
        connected.advance_collision();
        expect(connected.speculative_island_count()==2 &&
               connected.accurate_island_count()==2 &&
               connected.island_full_rebuild_count()==initial_rebuilds,
               "Restored source contact edges did not incrementally rejoin islands");
        connected.remove(b);
        connected.advance_collision();
        expect(connected.speculative_island_count()==3 &&
               connected.accurate_island_count()==3 &&
               connected.island_full_rebuild_count()==initial_rebuilds+1,
               "Destroyed source island node remained active in the graph");
        connected.step(1.0f/120.0f);
        expect(std::isfinite(connected.body_state(a).position.y) &&
               std::isfinite(connected.body_state(c).position.y) &&
               std::isfinite(connected.body_state(isolated).position.y) &&
               connected.body_state(isolated).position.y<5.0f,
               "Native island-node admission omitted free-flight integration");

        // Source task continuations can overlap the numerical solve.
        // The admitted synchronous schedule processes completed lost
        // contact events before solving, and the split result must be
        // published for the new frame through the same native scene.
        Scene phased;
        phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto middle=phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,5,0},1.0f);
        phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,5,0},1.0f);
        phased.advance_collision();
        expect(phased.accurate_island_count()==1,
               "Third-pass fixture must begin with one accurate island");
        phased.set_position(middle,{5.0f,5,0});
        phased.step(1.0f/120.0f);
        expect(phased.accurate_island_count()==3 &&
               phased.largest_accurate_island_size()==1 &&
               phased.contacts().empty(),
               "Source lost-contact retirement did not split accurate islands");
        phased.step(1.0f/120.0f);
        expect(phased.accurate_island_count()==3 &&
               phased.contacts().empty(),
               "Next-frame island admission reused retired contact edges");

        // IslandSim::findRoute: deleting a non-bridge edge must not
        // split a three-node cycle. After that, deleting the surviving
        // two-edge route really must split the dynamic components.
        // Both speculative SAP and accurate Gu graphs must agree on
        // the independently published resulting connectivity.
        Scene cycle;
        const auto ca=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto cb=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,8,0},1.0f);
        const auto cc=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        cycle.advance_collision();
        expect(cycle.pairs().size()==3 &&
               cycle.contacts().size()==3 &&
               cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1,
               "Lost-edge route fixture did not begin as a connected cycle");
        const auto cycle_rebuilds=cycle.island_full_rebuild_count();
        cycle.set_position(cc,{1.3f,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().size()==2 &&
               cycle.contacts().size()==2 &&
               cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.largest_accurate_island_size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Source lost-edge route was discarded despite an alternate path");
        // A-C is the only removed chord. Source processNewEdges merges
        // A-B with a size tie, retaining B's island root; both A and C
        // are non-root dirty endpoints. Reintroducing A-C must retain
        // both cached routes through B in each graph (four hits total).
        const auto cycle_fast_hits=cycle.island_fast_route_hit_count();
        cycle.set_position(cc,{0.8f,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().size()==3 &&
               cycle.contacts().size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Redundant cycle chord insertion rebuilt the island");
        cycle.set_position(cc,{1.3f,8,0});
        cycle.advance_collision();
        expect(cycle.lost_pairs().size()==1 &&
               cycle.lost_contacts().size()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.speculative_island_count()==1 &&
               cycle.island_fast_route_hit_count()==cycle_fast_hits+4 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Redundant cycle chord invalidated its alternate fast route");
        // A-C is no longer connected. Removing B's remaining two
        // contacts must trigger actual split repair, not the route skip.
        cycle.set_position(cb,{5,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().empty() && cycle.contacts().empty() &&
               cycle.speculative_island_count()==3 &&
               cycle.accurate_island_count()==3 &&
               cycle.largest_accurate_island_size()==1 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds &&
               cycle.island_local_split_count()>0,
               "Source island split lost its final surviving route");
        cycle.set_position(cb,{0.4f,8,0});
        cycle.advance_collision();
        expect(cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.largest_accurate_island_size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Source lost-edge repair did not admit incremental rejoin");

        // Multiple dynamic losses in one collision publication: each
        // removed edge has a surviving replacement path in the final
        // graph. The proof must not stop at the first still-connected
        // pair, nor treat multiple redundant removals as a split.
        Scene redundant_losses;
        redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,12,0},1.0f);
        redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,12,0},1.0f);
        const auto rc=redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,12,0},1.0f);
        const auto rd=redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.2f,12,0},1.0f);
        redundant_losses.advance_collision();
        expect(redundant_losses.pairs().size()==5 &&
               redundant_losses.contacts().size()==5 &&
               redundant_losses.accurate_island_count()==1,
               "Multi-edge source route fixture did not begin connected");
        const auto multi_rebuilds=
            redundant_losses.island_full_rebuild_count();
        redundant_losses.set_position(rc,{1.2f,12,0});
        redundant_losses.set_position(rd,{1.5f,12,0});
        redundant_losses.advance_collision();
        expect(redundant_losses.lost_pairs().size()==2 &&
               redundant_losses.lost_contacts().size()==2 &&
               redundant_losses.pairs().size()==3 &&
               redundant_losses.contacts().size()==3 &&
               redundant_losses.speculative_island_count()==1 &&
               redundant_losses.accurate_island_count()==1 &&
               redundant_losses.largest_accurate_island_size()==4 &&
               redundant_losses.island_full_rebuild_count()==multi_rebuilds,
               "Multiple redundant edge losses incorrectly rebuilt an intact island");

        // Source EdgeInstance adjacency must survive handle retirement
        // and reuse without retaining stale next/prev endpoint links.
        // Retire the A-C edge of a three-node cycle, recycle its handle
        // for A-D, then delete A-D while B still provides another route.
        // Finally remove B and verify real component split repair.
        Scene relink;
        const auto aa=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,15,0},1.0f);
        const auto bb=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,15,0},1.0f);
        const auto relink_c=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,15,0},1.0f);
        relink.advance_collision();
        const fyzz::native::Pair ac{aa,relink_c};
        const auto recycled=relink.contact_edge_index(ac);
        expect(relink.pairs().size()==3 &&
               relink.contacts().size()==3 &&
               recycled!=std::numeric_limits<std::uint32_t>::max(),
               "Intrusive adjacency fixture did not establish a three-edge cycle");
        relink.set_position(relink_c,{1.3f,15,0});
        relink.advance_collision();
        expect(relink.contact_edge_index(ac)==
                   std::numeric_limits<std::uint32_t>::max() &&
               relink.accurate_island_count()==1,
               "Retired source graph edge leaked into surviving component");
        const auto dd=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,15,0},1.0f);
        relink.advance_collision();
        const fyzz::native::Pair ad{aa,dd};
        expect(relink.contact_edge_index(ad)==recycled &&
               relink.accurate_island_count()==1 &&
               relink.largest_accurate_island_size()==4,
               "Recycled edge handle failed to link its new island endpoints");
        // Adding a dynamic node necessarily rebuilds the admitted island
        // roster; compare only the subsequent redundant edge retirement.
        const auto before_recycle=relink.island_full_rebuild_count();
        relink.set_position(dd,{1.3f,15,0});
        relink.advance_collision();
        expect(relink.contact_edge_index(ad)==
                   std::numeric_limits<std::uint32_t>::max() &&
               relink.accurate_island_count()==1 &&
               relink.island_full_rebuild_count()==before_recycle,
               "Recycled edge unlink corrupted an alternate island route");
        relink.set_position(bb,{5,15,0});
        relink.advance_collision();
        // Removing B strands both A and B as singleton islands,
        // while the overlapping C-D pair remains a third island.
        expect(relink.speculative_island_count()==3 &&
               relink.accurate_island_count()==3 &&
               relink.largest_accurate_island_size()==2 &&
               relink.island_full_rebuild_count()==before_recycle &&
               relink.island_local_split_count()>0,
               "Recycled intrusive graph failed to split on lost bridge");

        // Source mDirtyMap deduplicates endpoints of overlapping lost
        // edges: A-tip and B-tip remove three distinct dynamic nodes
        // in each graph, not four entries with a repeated tip.
        // Re-adding redundant chords must retain the cached tip->C->root
        // route for the next removal; only a real union changes the root.
        Scene dirty_routes;
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,22,0},1.0f);
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.2f,22,0},1.0f);
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,22,0},1.0f);
        const auto route_tip=dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.6f,22,0},1.0f);
        dirty_routes.advance_collision();
        expect(dirty_routes.pairs().size()==6 &&
               dirty_routes.contacts().size()==6 &&
               dirty_routes.accurate_island_count()==1,
               "Dirty-route fixture did not begin fully connected");
        const auto before_rebuild=dirty_routes.island_full_rebuild_count();
        const auto before_fast=dirty_routes.island_fast_route_hit_count();
        const auto before_checks=dirty_routes.island_dirty_route_check_count();
        dirty_routes.set_position(route_tip,{1.3f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.lost_pairs().size()==2 &&
               dirty_routes.lost_contacts().size()==2 &&
               dirty_routes.accurate_island_count()==1 &&
               dirty_routes.speculative_island_count()==1 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_dirty_route_check_count()==before_checks+6 &&
               dirty_routes.island_fast_route_hit_count()==before_fast,
               "Source mDirtyMap repeated an endpoint during lost-edge repair");

        // Reintroducing these redundant chords cannot change island
        // membership. Source processNewEdges *can*, however, shorten a
        // root-directed hop estimate through one of the new edges.
        // Removing a shortened chord invalidates its own fast route.
        // Source-style shortening retains one alternate root fast path
        // per IslandSim graph, so the next loss yields two hits, not
        // the four hits from the older non-shortening implementation.
        dirty_routes.set_position(route_tip,{0.6f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.pairs().size()==6 &&
               dirty_routes.contacts().size()==6 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild,
               "Redundant chord readdition rebuilt an existing island");
        const auto fast_before_retry=dirty_routes.island_fast_route_hit_count();
        const auto checks_before_retry=dirty_routes.island_dirty_route_check_count();
        dirty_routes.set_position(route_tip,{1.3f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.lost_pairs().size()==2 &&
               dirty_routes.lost_contacts().size()==2 &&
               dirty_routes.accurate_island_count()==1 &&
               dirty_routes.speculative_island_count()==1 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_dirty_route_check_count()==checks_before_retry+6 &&
               dirty_routes.island_fast_route_hit_count()==fast_before_retry+2,
               "Redundant chord removal changed source fast-route reuse");

        // Retiring the last surviving tip edge is a real split.
        dirty_routes.set_position(route_tip,{3.0f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.accurate_island_count()==2 &&
               dirty_routes.speculative_island_count()==2 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_local_split_count()>0,
               "Dirty fast route falsely preserved a disconnected island");

        // IslandSim::addNode initializes only its own route state.
        // Rebuilding membership for an unrelated actor must not evict
        // the alternate paths learned in an established dynamic island.
        // Two root-directed fast hits (one per graph) must survive.
        {
            Scene admitted_routes;
            const auto a=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.0f,26,0},1.0f);
            admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.2f,26,0},1.0f);
            admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.4f,26,0},1.0f);
            const auto tip=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.6f,26,0},1.0f);
            admitted_routes.advance_collision();
            admitted_routes.set_position(tip,{1.3f,26,0});
            admitted_routes.advance_collision();
            expect(admitted_routes.lost_pairs().size()==2 &&
                   admitted_routes.lost_contacts().size()==2,
                   "Actor-admission route fixture did not prime lost chords");
            admitted_routes.set_position(tip,{0.6f,26,0});
            admitted_routes.advance_collision();
            const auto root_accurate=admitted_routes.accurate_island_id(a);
            const auto root_speculative=admitted_routes.speculative_island_id(a);
            const auto before_rebuild=
                admitted_routes.island_full_rebuild_count();
            const auto before_hits=admitted_routes.island_fast_route_hit_count();
            const auto unrelated=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,26,0},1.0f);
            admitted_routes.advance_collision();
            expect(admitted_routes.island_full_rebuild_count()==
                       before_rebuild+1 &&
                   admitted_routes.accurate_island_count()==2 &&
                   admitted_routes.speculative_island_count()==2 &&
                   admitted_routes.accurate_island_id(a)==root_accurate &&
                   admitted_routes.speculative_island_id(a)==root_speculative &&
                   admitted_routes.accurate_island_id(unrelated)!=root_accurate,
                   "Unrelated node admission changed source island ownership");
            admitted_routes.set_position(tip,{1.3f,26,0});
            admitted_routes.advance_collision();
            expect(admitted_routes.lost_pairs().size()==2 &&
                   admitted_routes.lost_contacts().size()==2 &&
                   admitted_routes.island_fast_route_hit_count()==before_hits+2 &&
                   admitted_routes.island_full_rebuild_count()==before_rebuild+1,
                   "Unrelated node admission evicted valid root routes");
        }

        // PxsIslandSim owns stable island handles, not transient
        // union/find roots. A split preserves the root's handle, an
        // unaffected island retains its handle, and a merge keeps the
        // larger island handle. Node admission/retirement must not
        // renumber unrelated source islands in either graph.
        {
            Scene lifetime;
            const auto a=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,30,0},1.0f);
            const auto b=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.4f,30,0},1.0f);
            const auto c=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.8f,30,0},1.0f);
            const auto isolated=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{8,30,0},1.0f);
            lifetime.advance_collision();
            const auto island=lifetime.accurate_island_id(a);
            const auto speculative=lifetime.speculative_island_id(a);
            const auto quiet=lifetime.accurate_island_id(isolated);
            const auto quiet_spec=lifetime.speculative_island_id(isolated);
            expect(lifetime.accurate_island_count()==2 &&
                   lifetime.speculative_island_count()==2 &&
                   island!=quiet && speculative!=quiet_spec &&
                   island==lifetime.accurate_island_id(b) &&
                   island==lifetime.accurate_island_id(c) &&
                   speculative==lifetime.speculative_island_id(c),
                   "Source IslandSim did not publish persistent node handles");
            lifetime.set_position(c,{3,30,0});
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.speculative_island_count()==3 &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.speculative_island_id(a)==speculative &&
                   lifetime.accurate_island_id(b)==island &&
                   lifetime.accurate_island_id(c)!=island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(isolated)==quiet_spec,
                   "Source island split renumbered old-root or unrelated island");
            lifetime.set_position(c,{0.8f,30,0});
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==2 &&
                   lifetime.speculative_island_count()==2 &&
                   lifetime.accurate_island_id(c)==island &&
                   lifetime.speculative_island_id(c)==speculative &&
                   lifetime.accurate_island_id(isolated)==quiet,
                   "Source island merge failed to retain larger island identity");
            const auto newcomer=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{12,30,0},1.0f);
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(a)==speculative &&
                   lifetime.speculative_island_id(isolated)==quiet_spec &&
                   lifetime.accurate_island_id(newcomer)!=island &&
                   lifetime.accurate_island_id(newcomer)!=quiet,
                   "Source node admission renumbered existing island handles");
            lifetime.remove(b);
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.speculative_island_count()==3 &&
                   lifetime.accurate_island_id(b)==
                       std::numeric_limits<std::uint32_t>::max() &&
                   lifetime.speculative_island_id(b)==
                       std::numeric_limits<std::uint32_t>::max() &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.accurate_island_id(c)==island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(isolated)==quiet_spec,
                   "Source node retirement invalidated a live island handle");
        }

        // Actor admission may coincide with a merge of two existing
        // islands. Source mergeIslands retains the *larger* island handle
        // even if the smaller island owns the lower BodyId and appears
        // first in the full component rebuild's traversal.
        {
            Scene concurrent;
            const auto singleton=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,36,0},1.0f);
            const auto large_root=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,36,0},1.0f);
            const auto large_member=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4.4f,36,0},1.0f);
            concurrent.advance_collision();
            const auto accurate_large=concurrent.accurate_island_id(large_root);
            const auto speculative_large=
                concurrent.speculative_island_id(large_root);
            expect(concurrent.accurate_island_count()==2 &&
                   concurrent.speculative_island_count()==2 &&
                   concurrent.accurate_island_id(singleton)!=accurate_large &&
                   concurrent.accurate_island_id(large_member)==accurate_large,
                   "Rebuild merge fixture did not begin with unequal islands");
            const auto unrelated=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,36,0},1.0f);
            concurrent.set_position(singleton,{3.4f,36,0});
            concurrent.advance_collision();
            expect(concurrent.accurate_island_count()==2 &&
                   concurrent.speculative_island_count()==2 &&
                   concurrent.accurate_island_id(singleton)==accurate_large &&
                   concurrent.accurate_island_id(large_root)==accurate_large &&
                   concurrent.accurate_island_id(large_member)==accurate_large &&
                   concurrent.speculative_island_id(singleton)==speculative_large &&
                   concurrent.speculative_island_id(large_member)==speculative_large &&
                   concurrent.accurate_island_id(unrelated)!=accurate_large,
                   "Rebuild-time merge discarded the larger source island handle");
        }

        // IslandSim::processNewEdges grows island sizes after each
        // found edge. Two old 2-node islands merge before a 3-node
        // island joins: the newly 4-node handle wins the next merge.
        // Unrelated actor admission must not change that sequence or
        // either graph's retained source island identity.
        {
            auto populate=[](Scene& scene) {
                return std::array<fyzz::BodyId,7>{
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{0,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{0.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{4,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{4.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8.6f,42,0},1)};
            };
            Scene incremental, admitted;
            const auto ci=populate(incremental);
            const auto ai=populate(admitted);
            incremental.advance_collision();
            admitted.advance_collision();
            const auto survivor=incremental.accurate_island_id(ci[2]);
            const auto speculative_survivor=
                incremental.speculative_island_id(ci[2]);
            const auto largest_original=incremental.accurate_island_id(ci[4]);
            expect(incremental.accurate_island_count()==3 &&
                   incremental.speculative_island_count()==3 &&
                   survivor!=largest_original &&
                   survivor==incremental.accurate_island_id(ci[3]) &&
                   incremental.accurate_island_id(ci[0])!=survivor,
                   "Accumulated source merge fixture lacks three old islands");
            const auto newcomer=admitted.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,42,0},1);
            const std::array<float,7> joined_x{
                0.0f,0.3f,1.0f,1.3f,2.0f,2.3f,2.6f};
            for(std::size_t i=0;i<ci.size();++i) {
                incremental.set_position(ci[i],{joined_x[i],42,0});
                admitted.set_position(ai[i],{joined_x[i],42,0});
            }
            incremental.advance_collision();
            admitted.advance_collision();
            expect(incremental.accurate_island_count()==1 &&
                   incremental.speculative_island_count()==1 &&
                   admitted.accurate_island_count()==2 &&
                   admitted.speculative_island_count()==2 &&
                   incremental.accurate_island_id(ci[0])==survivor &&
                   incremental.speculative_island_id(ci[0])==
                       speculative_survivor &&
                   admitted.accurate_island_id(newcomer)!=survivor,
                   "Incremental source did not retain accumulated merge winner");
            for(std::size_t i=0;i<ci.size();++i)
                expect(admitted.accurate_island_id(ai[i])==
                           incremental.accurate_island_id(ci[i]) &&
                       admitted.speculative_island_id(ai[i])==
                           incremental.speculative_island_id(ci[i]),
                       "Actor admission changed source accumulated merge identity");
        }

        // Found edges are processed before the third-pass lost-edge
        // split, including when independent node admission forces the
        // full source island-membership rebuild. A-B first forms the
        // two-node island rooted at B. In one publication A-B is lost,
        // B-C is found, and an unrelated node is admitted in one scene.
        // Both execution paths must preserve B's island handle for B-C.
        {
            auto populate=[](Scene& scene) {
                const auto a=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,48,0},1.0f);
                const auto b=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.4f,48,0},1.0f);
                const auto c=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{3,48,0},1.0f);
                scene.advance_collision();
                return std::array{a,b,c};
            };
            Scene normal, admitted;
            const auto n=populate(normal);
            const auto a=populate(admitted);
            const auto root_accurate=normal.accurate_island_id(n[1]);
            const auto root_speculative=normal.speculative_island_id(n[1]);
            expect(normal.accurate_island_id(n[0])==root_accurate &&
                   normal.speculative_island_id(n[0])==root_speculative &&
                   normal.accurate_island_id(n[2])!=root_accurate,
                   "Source mixed-loss fixture lacks the initial A-B island");
            const auto unrelated=admitted.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,48,0},1.0f);
            for(auto* scene:{&normal,&admitted}) {
                const auto& ids=scene==&normal?n:a;
                scene->set_position(ids[0],{-3,48,0});
                scene->set_position(ids[2],{0.8f,48,0});
                scene->advance_collision();
                expect(scene->lost_pairs().size()==1 &&
                       scene->found_pairs().size()==1 &&
                       scene->lost_contacts().size()==1 &&
                       scene->found_contacts().size()==1,
                       "Mixed first-pass lost/found fixture did not swap contacts");
            }
            expect(normal.accurate_island_count()==2 &&
                   normal.speculative_island_count()==2 &&
                   admitted.accurate_island_count()==3 &&
                   admitted.speculative_island_count()==3 &&
                   normal.accurate_island_id(n[1])==root_accurate &&
                   normal.accurate_island_id(n[2])==root_accurate &&
                   normal.speculative_island_id(n[1])==root_speculative &&
                   normal.speculative_island_id(n[2])==root_speculative &&
                   admitted.accurate_island_id(a[1])==root_accurate &&
                   admitted.accurate_island_id(a[2])==root_accurate &&
                   admitted.speculative_island_id(a[1])==root_speculative &&
                   admitted.speculative_island_id(a[2])==root_speculative &&
                   admitted.accurate_island_id(a[0])!=root_accurate &&
                   admitted.speculative_island_id(a[0])!=root_speculative &&
                   admitted.accurate_island_id(unrelated)!=root_accurate,
                   "Actor admission changed first-pass merge and lost-edge split identity");
        }

        // A failed step must restore pre-existing source route hints,
        // not merely topology and diagnostics. Prime both IslandSim
        // caches with a redundant chord removal, reinsert the chords,
        // then trigger repeated solver failures on the next removal.
        // Redundant chords can shorten source fast routes upon
        // reintroduction; the retried scene must match the clean
        // reference's hints, counters, and motion instead of an
        // obsolete absolute fast-hit count.
        {
            auto populate=[](Scene& scene) {
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.0f,22,0},1.0f);
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.2f,22,0},1.0f);
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.4f,22,0},1.0f);
                const auto tip=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.6f,22,0},1.0f);
                const auto floor=scene.add_static_floor(0);
                const auto load=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                scene.advance_collision();
                return std::tuple{tip,floor,load};
            };
            Scene rejected, clean;
            const auto [rt,rf,rl]=populate(rejected);
            const auto [ct,cf,cl]=populate(clean);
            for(auto* scene:{&rejected,&clean}) {
                const auto tip=scene==&rejected?rt:ct;
                scene->set_position(tip,{1.3f,22,0});
                scene->advance_collision();
                expect(scene->island_dirty_route_check_count()==6 &&
                       scene->island_fast_route_hit_count()==0 &&
                       scene->accurate_island_count()==2 &&
                       scene->speculative_island_count()==2,
                       "Fast-route rollback fixture did not prime both graphs");
                scene->set_position(tip,{0.6f,22,0});
                scene->advance_collision();
                // Six box/box candidates plus the source plane's
                // five dynamic-body candidates; only the load touches
                // that plane, giving seven physical contact managers.
                expect(scene->pairs().size()==11 &&
                       scene->contacts().size()==7 &&
                       scene->accurate_island_count()==2 &&
                       scene->speculative_island_count()==2,
                       "Fast-route rollback fixture did not restore redundant chords");
                scene->set_position(tip,{1.3f,22,0});
            }
            const auto checks=rejected.island_dirty_route_check_count();
            const auto hits=rejected.island_fast_route_hit_count();
            const auto revision=rejected.island_topology_revision();
            const auto frame=rejected.frame_index();
            expect(checks==clean.island_dirty_route_check_count() &&
                   hits==clean.island_fast_route_hit_count(),
                   "Primed route histories disagree before rollback test");
            auto overflowing=fyzz::Material{};
            overflowing.static_friction=1.0e30f;
            overflowing.dynamic_friction=1.0e30f;
            overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
            rejected.set_material(rf,overflowing);
            rejected.set_material(rl,overflowing);
            for(int attempt=0;attempt<2;++attempt) {
                bool threw=false;
                try {rejected.step(1.0f/120.0f);}
                catch(const std::overflow_error&) {threw=true;}
                expect(threw &&
                       rejected.frame_index()==frame &&
                       rejected.island_topology_revision()==revision &&
                       rejected.island_dirty_route_check_count()==checks &&
                       rejected.island_fast_route_hit_count()==hits,
                       "Failed PGS publication committed island route diagnostics");
            }
            rejected.set_material(rf,fyzz::Material{});
            rejected.set_material(rl,fyzz::Material{});
            rejected.step(1.0f/120.0f);
            clean.step(1.0f/120.0f);
            expect(rejected.island_dirty_route_check_count()==checks+6 &&
                   rejected.island_dirty_route_check_count()==
                       clean.island_dirty_route_check_count() &&
                   rejected.island_fast_route_hit_count()==hits+2 &&
                   rejected.island_fast_route_hit_count()==
                       clean.island_fast_route_hit_count() &&
                   rejected.accurate_island_count()==clean.accurate_island_count() &&
                   rejected.speculative_island_count()==clean.speculative_island_count() &&
                   rejected.frame_index()==clean.frame_index() &&
                   rejected.body_state(rt).position.x==
                       clean.body_state(ct).position.x &&
                   rejected.body_state(rl).velocity.y==
                       clean.body_state(cl).velocity.y,
                   "Failed island route repair changed subsequent source retry");
        }

        // A failed solver frame after genuine *local* island splitting
        // must roll back handle/node-chain migration, indexed managers,
        // route hints, split diagnostics and publication. Retrying must
        // then perform exactly the same split as a clean production scene.
        {
            auto populate=[](Scene& scene) {
                const auto a=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,20,0},1);
                const auto b=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.85f,20,0},1);
                const auto c=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{1.7f,20,0},1);
                const auto floor=scene.add_static_floor(0);
                const auto load=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.49f,0},1);
                scene.advance_collision();
                expect(scene.accurate_island_count()==2 &&
                       scene.speculative_island_count()==2 &&
                       has(scene.found_contacts(),a,b) &&
                       has(scene.found_contacts(),b,c),
                       "Local split rollback fixture did not form the source chain");
                scene.set_position(b,{5,20,0});
                return std::tuple{a,b,c,floor,load};
            };
            Scene rejected,clean;
            const auto [ra,rb,rc,rf,rl]=populate(rejected);
            const auto [ca,cb,cc,cf,cl]=populate(clean);
            const auto before_frame=rejected.frame_index();
            const auto before_revision=rejected.island_topology_revision();
            const auto before_local=rejected.island_local_split_count();
            const auto before_global=rejected.island_full_rebuild_count();
            const auto before_graph_checkpoint=
                rejected.island_graph_checkpoint_count();
            const auto original_root=rejected.accurate_island_id(rb);
            auto overflowing=fyzz::Material{};
            overflowing.static_friction=1.0e30f;
            overflowing.dynamic_friction=1.0e30f;
            overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
            rejected.set_material(rf,overflowing);
            rejected.set_material(rl,overflowing);
            for(int attempt=0;attempt<2;++attempt) {
                bool threw=false;
                try {rejected.step(1.0f/120.0f);}
                catch(const std::overflow_error&) {threw=true;}
                expect(threw &&
                       rejected.frame_index()==before_frame &&
                       rejected.island_topology_revision()==before_revision &&
                       rejected.island_local_split_count()==before_local &&
                       rejected.island_full_rebuild_count()==before_global &&
                       rejected.island_graph_checkpoint_count()==
                           before_graph_checkpoint &&
                       rejected.accurate_island_id(rb)==original_root &&
                       rejected.accurate_island_count()==2 &&
                       rejected.contacts().size()==clean.contacts().size(),
                       "Failed local split published partial island migration");
            }
            rejected.set_material(rf,fyzz::Material{});
            rejected.set_material(rl,fyzz::Material{});
            rejected.step(1.0f/120.0f);
            clean.step(1.0f/120.0f);
            expect(rejected.island_full_rebuild_count()==before_global &&
                   rejected.island_graph_checkpoint_count()>
                       before_graph_checkpoint &&
                   rejected.island_graph_checkpoint_count()==
                       clean.island_graph_checkpoint_count() &&
                   rejected.island_local_split_count()>before_local &&
                   rejected.island_local_split_count()==
                       clean.island_local_split_count() &&
                   rejected.accurate_island_count()==4 &&
                   rejected.speculative_island_count()==4 &&
                   rejected.accurate_island_id(rb)==original_root &&
                   rejected.accurate_island_id(ra)!=original_root &&
                   rejected.accurate_island_id(rc)!=original_root &&
                   rejected.accurate_island_id(ra)==
                       clean.accurate_island_id(ca) &&
                   rejected.accurate_island_id(rc)==
                       clean.accurate_island_id(cc) &&
                   rejected.frame_index()==clean.frame_index() &&
                   rejected.body_state(rl).velocity.y==
                       clean.body_state(cl).velocity.y,
                   "Local PhysX split rollback did not reproduce clean migration");
        }

        // Two nearly parallel diagonal thin boxes may overlap in their
        // rotated SAP AABBs without touching in Gu SAT. This discriminates
        // the speculative source graph from the accurate touch graph;
        // replacing both with Gu manager connectivity fails here.
        Scene candidate_only;
        const auto rod_a=candidate_only.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,20,0},1.0f);
        const auto rod_b=candidate_only.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,25,0},1.0f);
        candidate_only.set_velocity(rod_a,{0,9.81f,0});
        candidate_only.set_velocity(rod_b,{0,9.81f,0});
        candidate_only.set_angular_velocity(rod_a,{0,0,0.83f});
        candidate_only.set_angular_velocity(rod_b,{0,0,0.83f});
        candidate_only.step(1.0f);
        candidate_only.set_angular_velocity(rod_a,{0,0,0});
        candidate_only.set_angular_velocity(rod_b,{0,0,0});
        const auto candidate_rebuilds=candidate_only.island_full_rebuild_count();
        candidate_only.set_position(rod_a,{0,20,0});
        candidate_only.set_position(rod_b,{0,22,0});
        candidate_only.advance_collision();
        expect(has(candidate_only.pairs(),rod_a,rod_b) &&
               candidate_only.contacts().empty() &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==2 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds,
               "Speculative island insertion rebuilt or conflated accurate topology");
        candidate_only.set_position(rod_b,{0,20.1f,0});
        candidate_only.advance_collision();
        expect(candidate_only.contacts().size()==1 &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==1 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds,
               "Accurate island insertion rebuilt or missed a touching edge");
        candidate_only.set_position(rod_b,{0,22,0});
        candidate_only.advance_collision();
        expect(candidate_only.contacts().empty() &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==2 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds &&
               candidate_only.island_local_split_count()>0,
               "Accurate island removal did not repair separated components");

        // Three coupled dynamic bodies generate two competing contact
        // managers, not a single order-insensitive pair. Check both the
        // contiguous Dy island-node batch and the finite-iteration PGS
        // impulses against a scene which began with the same topology.
        Scene joined, initially_joined;
        const auto j0=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto j1=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,5,0},1.0f);
        const auto j2=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{8,5,0},1.0f);
        const auto b0=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto b1=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,5,0},1.0f);
        const auto b2=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.6f,5,0},1.0f);
        joined.advance_collision();
        expect(joined.accurate_island_count()==3 &&
               joined.largest_accurate_island_size()==1,
               "Disjoint source nodes were not initially isolated");
        const auto before_join=joined.island_full_rebuild_count();
        joined.set_position(j1,{0.8f,5,0});
        joined.set_position(j2,{1.6f,5,0});
        joined.step(1.0f/120.0f);
        initially_joined.step(1.0f/120.0f);
        expect(joined.accurate_island_count()==1 &&
               joined.largest_accurate_island_size()==3 &&
               joined.island_full_rebuild_count()==before_join &&
               joined.contacts().size()==2 &&
               joined.solver_partition_count()>=2 &&
               joined.solved_normal_impulses().size()==
                    initially_joined.solved_normal_impulses().size(),
               "Incremental merge did not form a coupled accurate PGS island");
        // Both managers must contribute positive forces. Merely solving
        // the first contact would leave the outer body unconstrained.
        for (const auto& manager:joined.contacts()) {
            double impulse=0.0;
            for (std::size_t i=0;i<manager.count;++i)
                impulse+=joined.solved_normal_impulses()[
                    manager.contact_offset+i];
            expect(impulse>0.0,
                   "Coupled incremental island omitted a source contact solve");
        }
        const auto p=joined.solved_normal_impulses();
        const auto q=initially_joined.solved_normal_impulses();
        for(std::size_t i=0;i<p.size();++i)
            expect(std::abs(p[i]-q[i])<0.0001,
                   "Incremental island changed coupled PGS force ordering");
        for (const auto [incremental,initial] :
             {std::pair{j0,b0},std::pair{j1,b1},std::pair{j2,b2}}) {
            const auto lhs=joined.body_state(incremental);
            const auto rhs=initially_joined.body_state(initial);
            expect(std::abs(lhs.velocity.x-rhs.velocity.x)<0.0001f &&
                   std::abs(lhs.velocity.y-rhs.velocity.y)<0.0001f &&
                   std::abs(lhs.position.x-rhs.position.x)<0.0001f &&
                   std::abs(lhs.position.y-rhs.position.y)<0.0001f,
                   "Incremental accurate-island merge changed coupled PGS motion");
        }

        // IslandSim::Node::mStaticTouchCount and
        // mIslandStaticTouchCount: source SAP candidates against an
        // infinite plane remain speculative edges even when Gu reports
        // no touch. Dynamic-island merges accumulate static contacts;
        // lost dynamic edges split them back into individual counters.
        // None of these static-only changes should fabricate an island
        // topology revision.
        {
            Scene static_counts;
            const auto ground=static_counts.add_static_floor(0);
            const auto a=static_counts.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1);
            const auto b=static_counts.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,0.5f,0},1);
            static_counts.advance_collision();
            expect(static_counts.speculative_island_count()==2 &&
                   static_counts.accurate_island_count()==2 &&
                   static_counts.speculative_island_static_touches(a)==1 &&
                   static_counts.speculative_island_static_touches(b)==1 &&
                   static_counts.accurate_island_static_touches(a)==1 &&
                   static_counts.accurate_island_static_touches(b)==1,
                   "PhysX static touches not initialized in both island graphs");
            const auto stable_revision=static_counts.island_topology_revision();
            static_counts.set_position(b,{4,3,0});
            static_counts.advance_collision();
            expect(static_counts.island_topology_revision()==stable_revision &&
                   static_counts.speculative_island_static_touches(b)==1 &&
                   static_counts.accurate_island_static_touches(b)==0 &&
                   static_counts.accurate_island_static_touches(a)==1,
                   "Gu touch loss changed SAP static counts or island identity");
            static_counts.set_position(b,{4,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.island_topology_revision()==stable_revision &&
                   static_counts.accurate_island_static_touches(b)==1,
                   "Gu touch reentry did not restore static-touch ownership");
            static_counts.set_position(b,{0.6f,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.speculative_island_count()==1 &&
                   static_counts.accurate_island_count()==1 &&
                   static_counts.speculative_island_static_touches(a)==2 &&
                   static_counts.accurate_island_static_touches(a)==2 &&
                   static_counts.accurate_island_static_touches(b)==2,
                   "Dynamic island merge lost source static-touch counts");
            static_counts.set_position(b,{4,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.accurate_island_count()==2 &&
                   static_counts.speculative_island_count()==2 &&
                   static_counts.accurate_island_static_touches(a)==1 &&
                   static_counts.accurate_island_static_touches(b)==1 &&
                   static_counts.speculative_island_static_touches(a)==1,
                   "Lost-edge split duplicated or lost static-touch counts");
            static_counts.remove(ground);
            static_counts.advance_collision();
            expect(static_counts.accurate_island_static_touches(a)==0 &&
                   static_counts.accurate_island_static_touches(b)==0 &&
                   static_counts.speculative_island_static_touches(a)==0 &&
                   static_counts.speculative_island_static_touches(b)==0,
                   "Retired static actor retained ghost island touches");
        }

        Scene static_world;
        static_world.add_static_floor(0.0f);
        const auto ground_box=static_world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.5f,0.0f},1.0f);
        static_world.add_dynamic_box({0.5f,0.5f,0.5f},
                                     {4.0f,0.5f,0.0f},1.0f);
        static_world.advance_collision();
        expect(static_world.contacts().size()==2 &&
               static_world.speculative_island_count()==2 &&
               static_world.accurate_island_count()==2,
               "Shared static world incorrectly merged independent dynamic islands");
        const auto static_rev=static_world.island_topology_revision();
        const auto static_rebuilds=static_world.island_full_rebuild_count();
        const auto fixture=static_world.add_static_box(
            {1.0f,1.0f,1.0f},{20.0f,20.0f,20.0f});
        static_world.advance_collision();
        expect(static_world.island_topology_revision()==static_rev &&
               static_world.accurate_island_count()==2,
               "Static actor creation rebuilt dynamic island graph");
        static_world.remove(fixture);
        static_world.advance_collision();
        expect(static_world.island_topology_revision()==static_rev &&
               static_world.accurate_island_count()==2,
               "Static actor retirement rebuilt dynamic island graph");
        static_world.set_position(ground_box,{0,5,0});
        static_world.advance_collision();
        expect(static_world.contacts().size()==1 &&
               static_world.island_topology_revision()==static_rev,
               "Static-only contact loss unnecessarily rebuilt dynamic islands");
        static_world.set_position(ground_box,{0,0.5f,0});
        static_world.advance_collision();
        expect(static_world.contacts().size()==2 &&
               static_world.island_topology_revision()==static_rev &&
               static_world.island_full_rebuild_count()==static_rebuilds,
               "Static-only contact reentry unnecessarily rebuilt dynamic islands");
        static_world.step(1.0f/120.0f);
        expect(static_world.accurate_island_count()==2 &&
               static_world.solver_partition_count()==1,
               "Static world island split changed conflict-free solver partitioning");
    }

    // The same scene must distinguish SAP candidate overlaps from actual
    // shape contacts, and preserve touch found/lost across quiet frames.
    {
        Scene contact_scene;
        const auto distant_floor = contact_scene.add_static_floor(0.0f);
        const auto falling = contact_scene.add_dynamic_box(
            {1.0f,1.0f,1.0f}, {0,20,0}, 1.0f);
        contact_scene.advance_collision();
        expect(has(contact_scene.found_pairs(), distant_floor, falling),
               "Unbounded plane did not enter broadphase");
        expect(contact_scene.contacts().empty() &&
               contact_scene.found_contacts().empty(),
               "Far plane candidate was misreported as a contact");
        contact_scene.set_position(falling, {0,1.02f,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 4 &&
               has(contact_scene.found_contacts(), distant_floor, falling) &&
               contact_scene.lost_contacts().empty(),
               "PhysX plane/box near-contact vertex manifold absent");
        const auto& above = contact_scene.contacts()[0];
        expect(std::abs(above.points[0].separation - 0.02f) < 0.0001f &&
               above.points[0].normal.y == -1.0f,
               "Plane contact separation or normal incorrect");
        // GuContactPlaneBox enumerates +x,-y,+z before the other
        // three near-plane corners. Preserve source contact ordering
        // because contact force buffers are indexed per point.
        expect(std::abs(above.points[0].position.x - 1.0f) < 0.0001f &&
               std::abs(above.points[0].position.y - 0.02f) < 0.0001f &&
               std::abs(above.points[0].position.z - 1.0f) < 0.0001f,
               "GuContactPlaneBox source corner order or position diverged");
        contact_scene.advance_collision(true);
        expect(contact_scene.found_contacts().empty() &&
               contact_scene.lost_contacts().empty(),
               "Forced unchanged narrowphase published duplicate touch");
        contact_scene.set_position(falling, {0,0.8f,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 4 &&
               contact_scene.contacts()[0].points[0].separation < -0.19f &&
               contact_scene.found_contacts().empty(),
               "Plane penetration did not refresh contact manifold");
        contact_scene.set_position(falling, {0,20,0});
        contact_scene.advance_collision();
        expect(has(contact_scene.lost_contacts(), distant_floor, falling) &&
               contact_scene.lost_pairs().empty(),
               "Physical touch should be lost while plane BP candidate survives");
        contact_scene.advance_collision();
        expect(contact_scene.lost_contacts().empty(),
               "Quiet noncontact frame emitted repeated lost touch");
        contact_scene.set_position(falling, {0,-3,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 8,
               "Plane penetration must retain all eight source vertices");
        contact_scene.remove(falling);
        contact_scene.advance_collision();
        expect(has(contact_scene.lost_contacts(), distant_floor, falling),
               "Actor retirement failed to report lost physical touch");
    }
    {
        // The ordered pair must not determine which way gravity would
        // be opposed: creating the box before the floor reverses IDs
        // but not the physical plane orientation.
        Scene reversed;
        const auto box = reversed.add_dynamic_box(
            {1,1,1}, {0,1.02f,0}, 1);
        const auto floor = reversed.add_static_floor(0);
        reversed.advance_collision();
        expect(has(reversed.found_pairs(), box, floor) &&
               has(reversed.found_contacts(), box, floor) &&
               reversed.contacts().size() == 1 &&
               reversed.contacts()[0].count == 4,
               "Box-before-floor contact stream absent");
        for (unsigned i = 0; i < reversed.contacts()[0].count; ++i) {
            const auto& point = reversed.contacts()[0].points[i];
            expect(point.normal.x == 0.0f && point.normal.y == 1.0f &&
                   point.normal.z == 0.0f &&
                   std::abs(point.separation - 0.02f) < 0.0001f,
                   "Box-before-floor contact normal or separation incorrect");
        }
        reversed.set_position(box, {0,20,0});
        reversed.advance_collision();
        expect(has(reversed.lost_contacts(), box, floor) &&
               reversed.lost_pairs().empty(),
               "Box-before-floor touch loss emitted wrong events");
    }
    {
        Scene box_scene;
        const auto fixed = box_scene.add_static_box(
            {1,1,1},{0,0,0});
        const auto moveable = box_scene.add_dynamic_box(
            {1,1,1},{1.5f,0,0},1);
        box_scene.advance_collision();
        expect(box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].count == 4 &&
               has(box_scene.found_contacts(), fixed, moveable),
               "Axis-aligned box contact manifold absent");
        const auto& manifold = box_scene.contacts()[0];
        expect(manifold.points[0].normal.x == -1.0f &&
               std::abs(manifold.points[0].separation + 0.5f) < 0.0001f,
               "Box contact normal or signed penetration incorrect");
        box_scene.set_position(moveable, {2.03f,0,0});
        box_scene.advance_collision();
        expect(box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].points[0].separation > 0.029f,
               "Box proximity contacts must preserve positive separation");
        box_scene.set_position(moveable, {2.10f,0,0});
        box_scene.advance_collision();
        expect(has(box_scene.lost_contacts(), fixed, moveable) &&
               box_scene.contacts().empty(),
               "Box separation did not retire touch history");
        box_scene.set_position(moveable, {1.5f,0,0});
        box_scene.advance_collision();
        expect(has(box_scene.found_contacts(), fixed, moveable),
               "Box re-entry did not recreate narrowphase touch");

        // The original GuContactBoxBox face expansion retains near-corner
        // contacts inside per-axis contactDistance, even for diagonal gaps.
        // Preserve its generated point stream instead of the retired
        // face-only implementation's rejection of those candidates.
        box_scene.set_position(moveable, {2.03f,2.03f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count>0 &&
               box_scene.lost_contacts().empty(),
               "Source near-corner contact generation was not retained");
        const auto old_count=box_scene.contacts()[0].count;
        box_scene.advance_collision(true);
        expect(box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count==old_count &&
               box_scene.found_contacts().empty() &&
               box_scene.lost_contacts().empty(),
               "Forced near-corner update changed source touch history");
        box_scene.set_position(moveable, {2.01f,2.01f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count>0 &&
               box_scene.found_contacts().empty(),
               "Original Gu near-edge contact generation was lost");

        // A single-axis near contact with positive separation is still
        // supported when the two transverse face intervals overlap.
        box_scene.set_position(moveable, {1.5f,2.03f,0});
        box_scene.advance_collision();
        expect(box_scene.found_contacts().empty() &&
               box_scene.lost_contacts().empty() &&
               box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].count > 0 &&
               box_scene.contacts()[0].points[0].normal.y == -1.0f &&
               box_scene.contacts()[0].points[0].separation > 0.029f,
               "Source single-axis proximity or persistent-touch semantics drifted");
    }



    {
        // Direct production-path translation of the original PhysX
        // DyBodyCoreIntegrator closed-form quaternion step, limited to
        // an isolated free body until source angular contact constraints
        // and the original PGS execution are integrated.
        Scene angular;
        const auto body = angular.add_dynamic_box(
            {0.25f,0.5f,0.75f},{0,10,0},1);
        angular.set_angular_velocity(body,{0,2,0});
        for (int i=0; i<120; ++i) angular.step(1.0f/120.0f);
        const auto state = angular.body_state(body);
        const float quat_norm = std::sqrt(
            state.orientation.x*state.orientation.x +
            state.orientation.y*state.orientation.y +
            state.orientation.z*state.orientation.z +
            state.orientation.w*state.orientation.w);
        expect(std::isfinite(quat_norm) &&
               std::abs(quat_norm-1.0f)<1.0e-4f &&
               std::abs(state.orientation.y)>0.5f &&
               state.angular_velocity.y>1.8f &&
               state.angular_velocity.y<2.0f,
               "Source quaternion integration/damping did not advance");
        bool invalid=false;
        try {angular.set_angular_velocity(body,{0,INFINITY,0});}
        catch(const std::invalid_argument&) {invalid=true;}
        expect(invalid,"Nonfinite native angular velocity was admitted");
        auto floor=angular.add_static_floor(0);
        bool static_rejected=false;
        try {angular.set_angular_velocity(floor,{0,1,0});}
        catch(const std::logic_error&) {static_rejected=true;}
        expect(static_rejected,"Static actor accepted angular velocity");
        // The source normal-contact rows include rotational Jacobians,
        // so a rotated actor may advance alongside static geometry.
        angular.step(1.0f/120.0f);
        angular.set_angular_velocity(body,{0,0,0});
        angular.step(1.0f/120.0f);
        // Collision-only Gu narrowphase remains independently usable.
        angular.advance_collision(true);
        expect(has(angular.pairs(),body,floor) &&
               angular.contacts().empty(),
               "Rotated distant floor candidate must not invent a contact");
    }

    // Per-point Dy rigid-body normals must exert angular response at an
    // offset contact; the retired one-impulse-per-manifold solver could
    // not produce contact torque at any rigid-body geometry.
    {
        Scene angular_contact;
        angular_contact.add_static_box({1,1,1},{0,0,0});
        const auto body=angular_contact.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.3f,1.25f,0},1.0f);
        angular_contact.set_velocity(body,{-2,0,0});
        angular_contact.step(1.0f/120.0f);
        const auto result=angular_contact.body_state(body);
        expect(!angular_contact.contacts().empty() &&
               std::isfinite(result.angular_velocity.z) &&
               std::abs(result.angular_velocity.z)>1.0e-5f,
               "Per-point source contact angular Jacobian produced no torque");
    }

    // Original Gu box-box SAT/contact path must accept an oblique
    // body directly in the owning scene, not only the free-flight path.
    {
        Scene oblique;
        auto moving=oblique.add_dynamic_box({0.5f,0.5f,0.5f},
                                             {0,5,0},1.0f);
        oblique.set_angular_velocity(moving,{0,0,2.0f});
        oblique.step(0.4f);
        oblique.set_angular_velocity(moving,{0,0,0});
        oblique.set_position(moving,{1.35f,0,0});
        auto fixed=oblique.add_static_box({1,1,1},{0,0,0});
        oblique.advance_collision();
        expect(has(oblique.found_pairs(),moving,fixed) &&
               has(oblique.found_contacts(),moving,fixed) &&
               oblique.contacts().size()==1 &&
               oblique.contacts()[0].count>0 &&
               oblique.contacts()[0].collision_axis>0,
               "Rotated Gu box contact generation or pair axis missing");
        for(const auto& pt:oblique.contacts()[0].points) {
            const float length=std::sqrt(pt.normal.x*pt.normal.x+
                 pt.normal.y*pt.normal.y+pt.normal.z*pt.normal.z);
            expect(std::isfinite(pt.separation) && std::abs(length-1.0f)<0.002f,
                   "Rotated box contact normal/separation invalid");
        }
        const auto axis=oblique.contacts()[0].collision_axis;
        oblique.advance_collision(true);
        expect(oblique.contacts().size()==1 &&
               oblique.contacts()[0].collision_axis==axis &&
               oblique.found_contacts().empty() &&
               oblique.lost_contacts().empty(),
               "Gu box-box collision axis cache was not retained");
        // Per-point angular contact constraints must now admit the
        // oblique geometry without the retired translation-only guard.
        oblique.step(1.0f/120.0f);
        expect(std::isfinite(oblique.body_state(moving).angular_velocity.z),
               "Rotated Dy normal-contact solver produced invalid angular motion");
    }
    // A pitched box has more than an AABB corner representation. The
    // source plane contact stream must carry its real rotated vertices.
    {
        Scene pitched;
        const auto moving=pitched.add_dynamic_box(
            {0.4f,0.7f,0.3f},{0,5,0},1);
        pitched.set_angular_velocity(moving,{0,0,2.0f});
        pitched.step(0.4f);
        pitched.set_angular_velocity(moving,{0,0,0});
        pitched.set_position(moving,{0,0.7f,0});
        const auto floor=pitched.add_static_floor(0);
        pitched.advance_collision();
        expect(has(pitched.found_pairs(),moving,floor) &&
               pitched.contacts().size()==1 &&
               pitched.contacts()[0].count>0,
               "Rotated plane-box source contact stream absent");
        bool displaced=false;
        for(const auto& pt:pitched.contacts()[0].points) {
            if(std::abs(std::abs(pt.position.x)-0.4f)>0.05f)
                displaced=true;
            expect(pt.normal.y==1.0f && std::isfinite(pt.separation),
                   "Rotated plane-box normal or separation invalid");
        }
        expect(displaced,"Rotated plane-box contacts remained axis-aligned");
    }

    // The scene-owned source solver-body pool prepares only true contact
    // participants. An unrelated free-flight body must keep its original
    // unconstrained motion through repeated solver rounds and retirement.
    {
        Scene sparse;
        const auto floor = sparse.add_static_floor(0.0f);
        const auto contact = sparse.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto free = sparse.add_dynamic_box(
            {0.25f,0.25f,0.25f},{8,12,0},1.0f);
        sparse.set_velocity(free,{3.0f,0.0f,0.0f});
        for (unsigned i=0;i<100;++i)
            sparse.step(1.0f/120.0f);
        const auto body = sparse.body_state(free);
        expect(std::abs(body.velocity.x-3.0f)<0.0001f &&
               std::abs(body.position.x-10.5f)<0.001f,
               "Contact solver body state modified unrelated free flight");
        for (const auto& manifold : sparse.contacts())
            expect(manifold.pair.a!=free && manifold.pair.b!=free,
                   "Distant free-flight body admitted to source contact solve");
        sparse.remove(contact);
        sparse.step(1.0f/120.0f);
        expect(std::abs(sparse.body_state(free).velocity.x-3.0f)<0.0001f &&
               sparse.contacts().empty() &&
               sparse.frame_index()==101,
               "Retiring a contact participant leaked stale solver-body state");
        (void)floor;
    }

    // Two static contact managers must reference the same zero-mass world
    // solver body, without duplicating or accumulating static velocity.
    {
        Scene supports;
        const auto left = supports.add_static_box(
            {0.5f,0.5f,1.0f},{-0.5f,-0.5f,0});
        const auto right = supports.add_static_box(
            {0.5f,0.5f,1.0f},{0.5f,-0.5f,0});
        const auto load = supports.add_dynamic_box(
            {0.75f,0.5f,0.75f},{0,0.5f,0},1.0f);
        supports.set_velocity(load,{0,-1.0f,0});
        supports.step(1.0f/120.0f);
        const auto v = supports.body_state(load).velocity;
        expect(supports.contacts().size()==2 &&
               std::isfinite(v.y) && v.y>-1.1f &&
               supports.body_state(left).velocity.y==0.0f &&
               supports.body_state(right).velocity.y==0.0f,
               "Shared world solver body failed multi-static contact");
    }

    // DyFrictionCorrelation retains the floor anchors while contact
    // moves, and the two source tangent constraints must dissipate
    // sliding motion with the reference material (0.5/0.5 friction).
    // A normal-only solver preserves almost all horizontal velocity.
    {
        Scene sliding;
        sliding.add_static_floor(0);
        const auto box=sliding.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        sliding.set_velocity(box,{2.0f,0,0});
        for (unsigned frame=0;frame<180;++frame)
            sliding.step(1.0f/120.0f);
        const auto state=sliding.body_state(box);
        expect(std::isfinite(state.velocity.x) &&
               std::abs(state.velocity.x)<0.9f &&
               std::isfinite(state.position.y) &&
               state.position.y>0.34f && state.position.y<0.68f &&
               sliding.contacts().size()==1,
               "Correlated PhysX friction anchors did not resist floor sliding");
    }

    // DyCpuGpuBiasCoefficient PGS rigidContact=0.8 is shared by
    // penetration bias and friction-row relaxation. A shallow symmetric
    // overlap must receive source-scale positional correction, rather
    // than the earlier unrelated 0.2 normal-bias approximation.
    {
        Scene correction;
        correction.add_static_floor(0.0f);
        const auto body=correction.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.48f,0},1.0f);
        correction.step(1.0f/120.0f);
        const auto state=correction.body_state(body);
        expect(std::isfinite(state.position.y) &&
               state.position.y>0.488f &&
               state.position.y<0.55f,
               "Source 0.8 PGS rigid-contact penetration bias was not applied");
    }
    // When the Coulomb limit is not reached, source friction relaxation
    // removes tangential slip more strongly than the old 0.2 multiplier.
    // Use a low sliding speed so the limit does not dominate this check.
    {
        Scene relaxation;
        relaxation.add_static_floor(0.0f);
        const auto body=relaxation.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        relaxation.set_velocity(body,{0.02f,0,0});
        relaxation.step(1.0f/120.0f);
        const auto state=relaxation.body_state(body);
        expect(std::isfinite(state.velocity.x) &&
               std::abs(state.velocity.x)<0.012f,
               "Source friction relaxation left excessive low-speed slip");
    }

    // DyConstraintPartition's rigid-body 32-bit partition mask must
    // roll into a second wave when one dynamic body has >32 independent
    // dynamic/dynamic contacts. None of the leaf bodies touch each other;
    // the center is the one genuine per-partition conflict.
    {
        Scene crowded;
        const auto center = crowded.add_dynamic_box(
            {2.0f,2.0f,2.0f},{0,5,0},1.0f);
        std::vector<fyzz::BodyId> leaves;
        leaves.reserve(35);
        for (unsigned y=0;y<5;++y) {
            for (unsigned z=0;z<7;++z) {
                leaves.push_back(crowded.add_dynamic_box(
                    {0.1f,0.1f,0.1f},
                    {2.08f,3.6f+0.7f*y,-1.5f+0.5f*z},1.0f));
            }
        }
        crowded.advance_collision();
        expect(crowded.contacts().size()==35,
               "Dense source contact-manager star did not create 35 dynamic edges");
        crowded.step(1.0f/120.0f);
        expect(crowded.solver_partition_count()==35,
               "Source 32-bit partition rollover collapsed conflicting dynamic contacts");
        const auto center_state = crowded.body_state(center);
        expect(std::isfinite(center_state.position.x) &&
               std::isfinite(center_state.velocity.x) &&
               crowded.contacts().size()==35,
               "Source 32-partition overflow did not execute all contact managers");
        // Dy writeback indexes the original Gu manifold stream even
        // though the solver traverses reordered constraint partitions.
        // All 35 manager reports must have a positive final impulse,
        // including three managers assigned to the overflow wave.
        const auto reported=crowded.solved_normal_impulses();
        std::size_t expected_offset=0;
        for (std::size_t i=0;i<crowded.contacts().size();++i) {
            const auto& manager=crowded.contacts()[i];
            expect(manager.contact_offset==expected_offset &&
                   manager.contact_offset+manager.count<=reported.size(),
                   "Partitioned Dy force writeback lost Gu contact offsets");
            double normal_sum=0.0;
            for (std::size_t k=0;k<manager.count;++k)
                normal_sum+=reported[manager.contact_offset+k];
            expect(std::isfinite(normal_sum) && normal_sum>0.0,
                   i>=32
                       ? "Overflow-wave manager has no solved contact force writeback"
                       : "First-wave manager has no solved contact force writeback");
            expected_offset+=manager.count;
        }
        expect(expected_offset==reported.size(),
               "Dy force writeback omitted original Gu contact points");
        // Constructing 35 partitions is insufficient proof of execution:
        // these boxes start with no X velocity and gravity has no X
        // component. Every active center/leaf contact must therefore
        // push its leaf outward along +X on the *first* solve. In
        // particular leaves 32-34 are assigned to the second 32-bit
        // overflow wave; dropping their ordered descriptors must fail.
        for (std::size_t index=0;index<leaves.size();++index) {
            const auto state=crowded.body_state(leaves[index]);
            expect(std::isfinite(state.position.x) &&
                   std::isfinite(state.velocity.x) &&
                   state.velocity.x>0.01f && state.position.x>2.08f,
                   index>=32
                       ? "Source second-wave contact produced no outward impulse"
                       : "Source first-wave contact produced no outward impulse");
        }
        // A new frame must recompute partition masks without retaining
        // previous wave assignments or introducing another body owner.
        crowded.step(1.0f/120.0f);
        expect(crowded.solver_partition_count()==35 &&
               std::isfinite(crowded.body_state(center).position.x),
               "Source constraint partition did not reset on next frame");
    }

    verify_island_contact_manager_order();
    verify_split_contact_manager_migration();
    verify_dirty_seed_split_manager_order();
    verify_hop_priority_split_migration();
    verify_equal_hop_split_queue_order();
    verify_source_dirty_witness_split();
    verify_successful_route_branch_witness();
    verify_route_repair_continues_after_split();
    verify_dirty_order_multisplit_manager_remainder();

    // The source zero-mass world solver body does not create a dynamic
    // conflict between distinct static-floor contacts; all may share
    // partition zero, even though they reference one world solver row.
    {
        Scene detached;
        detached.add_static_floor(0.0f);
        for (unsigned i=0;i<12;++i)
            detached.add_dynamic_box({0.25f,0.25f,0.25f},
                                     {float(i)*2.0f,0.25f,0},1.0f);
        detached.step(1.0f/120.0f);
        expect(detached.contacts().size()==12 &&
               detached.solver_partition_count()==1,
               "Static world solver row incorrectly serialized unrelated contacts");
    }

    // DySolverConstraints.cpp::writeBackContact must preserve Gu contact
    // offsets even when DyConstraintPartition changes constraint order.
    // Two disjoint floor contacts with different masses independently
    // verify final per-point normal impulse sums against velocity change.
    {
        constexpr float dt=1.0f/120.0f;
        Scene reported;
        reported.add_static_floor(0.0f);
        const auto light=reported.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.5f,0.0f},1.0f);
        const auto heavy=reported.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3.0f,0.5f,0.0f},2.0f);
        reported.advance_collision(true);
        expect(reported.contacts().size()==2 &&
               reported.solved_normal_impulses().empty(),
               "Collision-only Gu query exposed stale Dy force writeback");
        reported.step(dt);
        const auto manifolds=reported.contacts();
        const auto impulses=reported.solved_normal_impulses();
        expect(manifolds.size()==2 && impulses.size()==8,
               "Dy writeback did not preserve the flat Gu contact layout");
        double sums[2]{};
        for (std::size_t m=0;m<manifolds.size();++m) {
            const auto& manifold=manifolds[m];
            expect(manifold.count==4 &&
                   manifold.contact_offset+manifold.count<=impulses.size(),
                   "Dy contact output does not match original Gu offsets");
            for (std::size_t i=0;i<manifold.count;++i) {
                const double impulse=impulses[manifold.contact_offset+i];
                expect(std::isfinite(impulse) && impulse>=0.0,
                       "Dy reported an invalid final contact normal impulse");
                sums[m]+=impulse;
            }
        }
        const auto l=reported.body_state(light);
        const auto h=reported.body_state(heavy);
        // For unit-half boxes, mass is density (volume = 1).
        // Source solver normal impulse equals mass times vertical
        // velocity delta from pre-solver gravity, not a force in newtons.
        const double expected_light=l.velocity.y+9.81*dt;
        const double expected_heavy=2.0*(h.velocity.y+9.81*dt);
        expect(sums[0]>0.02 && sums[1]>sums[0]*1.5 &&
               std::abs(sums[0]-expected_light)<0.001 &&
               std::abs(sums[1]-expected_heavy)<0.001,
               "Source contact writeback does not reconcile solved body velocity");
        reported.advance_collision(true);
        expect(reported.solved_normal_impulses().empty(),
               "Geometry-only advance retained obsolete solver force results");
        reported.step(dt);
        expect(reported.solved_normal_impulses().size()==
                   reported.contacts()[0].count+
                   reported.contacts()[1].count,
               "Following Dy step did not republish fresh contact impulses");
        reported.set_position(light,{0.0f,5.0f,0.0f});
        reported.set_position(heavy,{3.0f,5.0f,0.0f});
        reported.advance_collision(true);
        expect(reported.contacts().empty() &&
               reported.solved_normal_impulses().empty(),
               "Lost contact retained previous normal force writeback");
    }

    // Genuine DyConstraintPartition permutation:
    // Gu order is (A,B), (A,C), (D,E), while two conflicting A edges
    // occupy partitions 0,1 and independent D/E falls back to 0.
    // Dy execution is therefore (A,B), (D,E), (A,C). Detect any
    // writeback accidentally appended in solver order by checking each
    // unique leaf's momentum against its original Gu manager offset.
    {
        Scene reordered;
        const auto a=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {0,5,0},1.0f);
        const auto b=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {0.80f,5,0},1.0f);
        const auto c=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {-0.78f,5,0},2.0f);
        const auto d=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {4,5,0},3.0f);
        const auto e=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {4.92f,5,0},4.0f);
        reordered.step(1.0f/120.0f);
        const auto contacts=reordered.contacts();
        const auto impulses=reordered.solved_normal_impulses();
        expect(contacts.size()==3 &&
               reordered.solver_partition_count()==2 &&
               contacts[0].pair==fyzz::native::Pair{a,b} &&
               contacts[1].pair==fyzz::native::Pair{a,c} &&
               contacts[2].pair==fyzz::native::Pair{d,e},
               "Native permutation regression did not construct [0,1,0] partitions");
        auto sum_for=[&](std::size_t m) {
            double sum=0;
            const auto& manager=contacts[m];
            expect(manager.contact_offset+manager.count<=impulses.size(),
                   "Reordered Dy force buffer does not match Gu offsets");
            for(std::size_t k=0;k<manager.count;++k)
                sum+=impulses[manager.contact_offset+k];
            return sum;
        };
        const double ib=sum_for(0),ic=sum_for(1),ie=sum_for(2);
        const double vb=std::abs(reordered.body_state(b).velocity.x);
        const double vc=std::abs(reordered.body_state(c).velocity.x);
        const double ve=std::abs(reordered.body_state(e).velocity.x);
        expect(ib>0 && ic>0 && ie>0 &&
               std::abs(ib-vb)<0.003 &&
               std::abs(ic-2.0*vc)<0.003 &&
               std::abs(ie-4.0*ve)<0.003 &&
               std::abs(ic-ie)>0.003,
               "Dy writeback followed partition order instead of Gu contact order");
    }

    // DyContactPrep.cpp::writeBackContactFriction reconstructs each
    // retained anchor impulse as tangent0*lambda0 + tangent1*lambda1.
    // Compare the aggregate world X impulse with each independent body's
    // tangential velocity change, and verify per-manager patch offsets.
    {
        Scene friction_report;
        friction_report.add_static_floor(0.0f);
        const auto left=friction_report.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-5.0f,0.5f,0.0f},1.0f);
        const auto right=friction_report.add_dynamic_box(
            {0.5f,0.5f,0.5f},{5.0f,0.5f,0.0f},2.0f);
        friction_report.set_velocity(left,{0.015f,0,0});
        friction_report.set_velocity(right,{-0.025f,0,0});
        friction_report.step(1.0f/120.0f);
        const auto contacts=friction_report.contacts();
        const auto patches=friction_report.solved_friction_patches();
        expect(contacts.size()==2 && patches.size()==2 &&
               contacts[0].friction_patch_offset==0 &&
               contacts[1].friction_patch_offset==1 &&
               contacts[0].friction_patch_count==1 &&
               contacts[1].friction_patch_count==1,
               "Source friction writeback did not retain Gu patch order");
        const auto left_after=friction_report.body_state(left);
        const auto right_after=friction_report.body_state(right);
        for (std::size_t m=0;m<2;++m) {
            const auto& patch=patches[m];
            expect(patch.pair==contacts[m].pair &&
                   patch.anchor_count>0 && patch.anchor_count<=2,
                   "Source friction writeback lost per-patch anchors");
            double total_x=0.0;
            for(std::size_t k=0;k<patch.anchor_count;++k) {
                const auto& impulse=patch.anchor_impulses[k];
                const auto& position=patch.anchor_positions[k];
                expect(std::isfinite(impulse.x) &&
                       std::isfinite(impulse.y) &&
                       std::isfinite(impulse.z) &&
                       std::isfinite(position.x) &&
                       std::isfinite(position.y) &&
                       std::abs(impulse.y)<0.001,
                       "Source friction two-axis impulse was not finite/tangential");
                total_x+=impulse.x;
            }
            const double velocity_before=m==0 ? 0.015 : -0.025;
            const double velocity_after=m==0
                ? left_after.velocity.x : right_after.velocity.x;
            const double mass=m==0 ? 1.0 : 2.0;
            expect(std::abs(total_x+
                            mass*(velocity_after-velocity_before))<0.002 &&
                   (m==0 ? total_x>0.0 : total_x<0.0),
                   "Source friction anchor writeback disagrees with body impulse");
        }
        friction_report.advance_collision(true);
        expect(friction_report.solved_normal_impulses().empty() &&
               friction_report.solved_friction_patches().empty(),
               "Geometry-only collision advance left stale friction impulses");
        friction_report.step(1.0f/120.0f);
        expect(friction_report.solved_friction_patches().size()==2,
               "Source friction writeback did not republish after collision query");
    }

    // A failed *geometry-only* advance must preserve the exact backing
    // allocations of the previously published Gu pair/contact and
    // solved force streams. A finite static box can overflow its AABB
    // during source broadphase preparation without any solver step.
    {
        Scene geometry;
        const auto floor=geometry.add_static_floor(0.0f);
        const auto box=geometry.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        geometry.step(1.0f/120.0f);
        const auto existing=fyzz::native::Pair{floor,box};
        const auto old_edge=geometry.contact_edge_index(existing);
        const auto* const old_pairs=geometry.pairs().data();
        const auto* const old_contacts=geometry.contacts().data();
        const auto* const old_points=geometry.contacts()[0].points.data();
        const auto* const old_forces=geometry.solved_normal_impulses().data();
        const auto old_frame=geometry.frame_index();
        const auto old_impulse=geometry.solved_normal_impulses()[0];
        const auto giant=geometry.add_static_box(
            {std::numeric_limits<float>::max()*0.5f,1,1},
            {std::numeric_limits<float>::max()*0.75f,0,0});
        bool rejected=false;
        try {geometry.advance_collision();}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               geometry.frame_index()==old_frame &&
               geometry.pairs().data()==old_pairs &&
               geometry.contacts().data()==old_contacts &&
               geometry.contacts()[0].points.data()==old_points &&
               geometry.solved_normal_impulses().data()==old_forces &&
               geometry.solved_normal_impulses()[0]==old_impulse &&
               geometry.contact_edge_index(existing)==old_edge &&
               geometry.contact_edge_connected(existing),
               "Failed geometry publication invalidated previous source spans");
        geometry.remove(giant);
        geometry.advance_collision();
        expect(geometry.contacts().size()==1 &&
               geometry.contact_edge_index(existing)==old_edge,
               "Rejected geometry advance prevented subsequent publication");
    }

    // Quiet failed solver preparation must also restore Gu manifold
    // descriptor metadata, not merely the contact-point span and pair
    // event streams. Put the overflowing contact manager second so
    // preparation has already advanced the solver offset of its row.
    {
        Scene rejected, clean;
        auto construct=[](Scene& s) {
            const auto floor=s.add_static_floor(0.0f);
            const auto first=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            const auto second=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5,0.49f,0},1.0f);
            s.advance_collision();
            return std::tuple{floor,first,second};
        };
        const auto [floor_bad,first_bad,second_bad]=construct(rejected);
        const auto [floor_good,first_good,second_good]=construct(clean);
        expect(rejected.contacts().size()==2 &&
               clean.contacts().size()==2,
               "Quiet manifold rollback fixture requires two Gu managers");
        const auto* const published_ptr=rejected.contacts().data();
        const auto* const points0=rejected.contacts()[0].points.data();
        const auto* const points1=rejected.contacts()[1].points.data();
        const auto original0=rejected.contacts()[0];
        const auto original1=rejected.contacts()[1];
        const auto frame=rejected.frame_index();
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(floor_bad,exploding);
        rejected.set_material(second_bad,exploding);
        bool failed=false;
        try {rejected.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {failed=true;}
        const auto after=rejected.contacts();
        expect(failed && rejected.frame_index()==frame &&
               after.size()==2 && after.data()==published_ptr &&
               after[0].points.data()==points0 &&
               after[1].points.data()==points1 &&
               after[0].solver_offset==original0.solver_offset &&
               after[1].solver_offset==original1.solver_offset &&
               after[0].friction_patch_offset==original0.friction_patch_offset &&
               after[1].friction_patch_offset==original1.friction_patch_offset &&
               after[0].friction_patch_count==original0.friction_patch_count &&
               after[1].friction_patch_count==original1.friction_patch_count,
               "Rejected quiet preparation leaked partial Gu/Dy descriptors");
        rejected.set_material(floor_bad,fyzz::Material{});
        rejected.set_material(second_bad,fyzz::Material{});
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        const auto bad=rejected.body_state(second_bad);
        const auto good=clean.body_state(second_good);
        expect(bad.position.y==good.position.y &&
               bad.velocity.y==good.velocity.y &&
               rejected.contacts().size()==clean.contacts().size(),
               "Quiet manifold rollback changed valid retry dynamics");
        (void)first_bad;
        (void)first_good;
    }

    // Rejected solver scratch must not be admitted again when the
    // checkpoint rolls back the frame clock and the author changes the
    // geometry before retry. Both scenes start with the same Gu contact;
    // only one attempts an overflowing material preparation. On retry,
    // their boxes are teleported out of contact and must follow the
    // identical unconstrained free-flight path.
    {
        constexpr float dt=1.0f/120.0f;
        Scene rejected, clean;
        auto create=[](Scene& s) {
            const auto floor=s.add_static_floor(0.0f);
            const auto box=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            s.advance_collision();
            return std::pair{floor,box};
        };
        const auto [floor_bad,body_bad]=create(rejected);
        const auto [floor_good,body_good]=create(clean);
        auto material=fyzz::Material{};
        material.static_friction=1.0e30f;
        material.dynamic_friction=1.0e30f;
        material.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(floor_bad,material);
        rejected.set_material(body_bad,material);
        bool failed=false;
        const auto old_frame=rejected.frame_index();
        try {rejected.step(dt);}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed && rejected.frame_index()==old_frame,
               "Solver generation fixture failed after source row admission");
        rejected.set_material(floor_bad,fyzz::Material{});
        rejected.set_material(body_bad,fyzz::Material{});
        rejected.set_position(body_bad,{0,5,0});
        clean.set_position(body_good,{0,5,0});
        rejected.step(dt);
        clean.step(dt);
        const auto a=rejected.body_state(body_bad);
        const auto b=clean.body_state(body_good);
        expect(rejected.contacts().empty() && clean.contacts().empty() &&
               a.position.x==b.position.x &&
               a.position.y==b.position.y &&
               a.position.z==b.position.z &&
               a.velocity.x==b.velocity.x &&
               a.velocity.y==b.velocity.y &&
               a.velocity.z==b.velocity.z &&
               rejected.solved_normal_impulses().empty() &&
               rejected.frame_index()==clean.frame_index(),
               "Rejected solver row leaked into changed-contact retry");
    }

    // A quiet failed step must not erase Found/Lost events from a
    // preceding successful geometry-only publication. Only event streams
    // rotate on that path: contact managers, their point spans, and pair
    // buffers retain their original allocations and publication epochs.
    // Exercise both found and lost events, and two failed retries.
    for(const bool expect_lost:{false,true}) {
        Scene rejected, clean;
        auto setup=[&](Scene& s) {
            // A bounded static box allows both the SAP overlap and the
            // Gu contact to be lost. An infinite floor remains a SAP
            // candidate after the dynamic body moves away vertically.
            const auto ground=s.add_static_box(
                {2.0f,0.5f,2.0f},{0,-0.5f,0});
            const auto box=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            s.advance_collision();
            if(expect_lost) {
                s.set_position(box,{0,10.0f,0});
                s.advance_collision();
            }
            return std::pair{ground,box};
        };
        const auto [floor_a,box_a]=setup(rejected);
        const auto [floor_b,box_b]=setup(clean);
        const auto found_pairs=rejected.found_pairs();
        const auto lost_pairs=rejected.lost_pairs();
        const auto found_contacts=rejected.found_contacts();
        const auto lost_contacts=rejected.lost_contacts();
        expect((expect_lost
                    ? has(lost_pairs,floor_a,box_a) &&
                      has(lost_contacts,floor_a,box_a)
                    : has(found_pairs,floor_a,box_a) &&
                      has(found_contacts,floor_a,box_a)) &&
               rejected.body_state(box_a).position.y==
                   clean.body_state(box_b).position.y,
               "Quiet-event rollback fixture failed to publish events");
        const std::vector<fyzz::native::Pair> prior_found(
            found_pairs.begin(),found_pairs.end());
        const std::vector<fyzz::native::Pair> prior_lost(
            lost_pairs.begin(),lost_pairs.end());
        const std::vector<fyzz::native::Pair> prior_found_contacts(
            found_contacts.begin(),found_contacts.end());
        const std::vector<fyzz::native::Pair> prior_lost_contacts(
            lost_contacts.begin(),lost_contacts.end());
        const auto* const found_ptr=found_pairs.data();
        const auto* const lost_ptr=lost_pairs.data();
        const auto* const touch_ptr=found_contacts.data();
        const auto* const lost_touch_ptr=lost_contacts.data();
        const auto* const pairs_ptr=rejected.pairs().data();
        const auto* const manifolds_ptr=rejected.contacts().data();
        const auto frame=rejected.frame_index();
        const auto topology=rejected.island_topology_revision();
        const auto original=rejected.body_state(box_a);
        for(int attempt=0;attempt<2;++attempt) {
            bool threw=false;
            try {rejected.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {threw=true;}
            const auto body=rejected.body_state(box_a);
            expect(threw &&
                   rejected.frame_index()==frame &&
                   rejected.island_topology_revision()==topology &&
                   rejected.pairs().data()==pairs_ptr &&
                   rejected.contacts().data()==manifolds_ptr &&
                   rejected.found_pairs().data()==found_ptr &&
                   rejected.lost_pairs().data()==lost_ptr &&
                   rejected.found_contacts().data()==touch_ptr &&
                   rejected.lost_contacts().data()==lost_touch_ptr &&
                   body.position.x==original.position.x &&
                   body.position.y==original.position.y &&
                   body.velocity.y==original.velocity.y,
                   "Rejected quiet step altered previous scene publication");
            const auto equal=[](auto span,const auto& expected) {
                return span.size()==expected.size() &&
                    std::equal(span.begin(),span.end(),expected.begin());
            };
            expect(equal(rejected.found_pairs(),prior_found) &&
                   equal(rejected.lost_pairs(),prior_lost) &&
                   equal(rejected.found_contacts(),prior_found_contacts) &&
                   equal(rejected.lost_contacts(),prior_lost_contacts),
                   "Rejected quiet step erased prior Found/Lost events");
        }
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        const auto recovered=rejected.body_state(box_a);
        const auto baseline=clean.body_state(box_b);
        expect(rejected.found_pairs().empty() &&
               rejected.lost_pairs().empty() &&
               rejected.found_contacts().empty() &&
               rejected.lost_contacts().empty() &&
               recovered.position.y==baseline.position.y &&
               recovered.velocity.y==baseline.velocity.y &&
               rejected.frame_index()==clean.frame_index(),
               "Quiet event rollback changed successful retry state");
        (void)floor_b;
    }

    // A failing Dy step must not publish its unconstrained velocities,
    // consume pending torque, advance topology/frame history, or replace
    // contact/force/threshold reports. Retrying with a valid material must
    // reproduce an identical scene which never attempted the failed step.
    // Material coefficients are individually finite at authoring, while
    // the source MULTIPLY combination can still overflow at contact prep.
    {
        constexpr float dt=1.0f/120.0f;
        Scene failed, clean;
        auto construct=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto box=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            scene.set_contact_report_threshold(box,0.1f);
            scene.step(1.0f/120.0f);
            scene.apply_torque_impulse(box,{0,0,0.1f});
            return std::pair{floor,box};
        };
        const auto [bad_floor,bad_box]=construct(failed);
        const auto [good_floor,good_box]=construct(clean);
        const auto prior=failed.body_state(bad_box);
        const auto frame=failed.frame_index();
        const auto topology=failed.island_topology_revision();
        const auto count=failed.accurate_island_count();
        const auto partitions=failed.solver_partition_count();
        const std::vector<double> forces(
            failed.solved_normal_impulses().begin(),
            failed.solved_normal_impulses().end());
        const auto prior_manifolds=failed.contacts();
        const auto* const manifold_memory=prior_manifolds.data();
        const auto* const pair_memory=failed.pairs().data();
        const auto* const found_memory=failed.found_pairs().data();
        const auto* const touch_memory=failed.found_contacts().data();
        const auto* const force_memory=failed.solved_normal_impulses().data();
        const auto* const threshold_memory=failed.force_threshold_changes().data();
        expect(prior_manifolds.size()==1 &&
               prior_manifolds[0].count==4,
               "Step failure fixture has no authoritative prior contacts");
        const auto* const point_memory=prior_manifolds[0].points.data();
        const auto first_contact=prior_manifolds[0].points[0];
        const std::vector<fyzz::native::ForceThresholdReport> threshold(
            failed.force_threshold_changes().begin(),
            failed.force_threshold_changes().end());
        const auto original_pair=prior_manifolds[0].pair;
        const auto original_edge=failed.contact_edge_index(original_pair);
        expect(original_edge!=std::numeric_limits<std::uint32_t>::max() &&
               failed.contact_edge_connected(original_pair),
               "Step rollback fixture did not establish an accurate edge");
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        failed.set_material(bad_floor,exploding);
        failed.set_material(bad_box,exploding);
        for (int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {failed.step(dt);}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected,
                   "Combined material overflow did not fail source preparation");
            const auto after=failed.body_state(bad_box);
            const auto same_vec=[](fyzz::Vec3 a,fyzz::Vec3 b) {
                return a.x==b.x && a.y==b.y && a.z==b.z;
            };
            const auto same_quat=[](fyzz::Quat a,fyzz::Quat b) {
                return a.x==b.x && a.y==b.y &&
                       a.z==b.z && a.w==b.w;
            };
            expect(same_vec(after.position,prior.position) &&
                   same_vec(after.velocity,prior.velocity) &&
                   same_quat(after.orientation,prior.orientation) &&
                   same_vec(after.angular_velocity,prior.angular_velocity) &&
                   failed.frame_index()==frame &&
                   failed.island_topology_revision()==topology &&
                   failed.accurate_island_count()==count &&
                   failed.solver_partition_count()==partitions &&
                   failed.contact_edge_index(original_pair)==original_edge &&
                   failed.contact_edge_connected(original_pair) &&
                   failed.active_contact_edge_count()==1,
                   "Rejected Dy step published partial body or edge state");
            const auto current=failed.contacts();
            expect(current.size()==1 &&
                   current.data()==manifold_memory &&
                   current[0].points.data()==point_memory &&
                   failed.pairs().data()==pair_memory &&
                   failed.found_pairs().data()==found_memory &&
                   failed.found_contacts().data()==touch_memory &&
                   failed.solved_normal_impulses().data()==force_memory &&
                   failed.force_threshold_changes().data()==threshold_memory &&
                   current[0].pair==original_pair &&
                   current[0].count==4 &&
                   same_vec(current[0].points[0].position,
                            first_contact.position) &&
                   same_vec(current[0].points[0].normal,
                            first_contact.normal) &&
                   current[0].points[0].separation==first_contact.separation,
                   "Rejected step invalidated previous authoritative Gu contacts");
            const auto reported=failed.solved_normal_impulses();
            expect(reported.size()==forces.size() &&
                   std::equal(reported.begin(),reported.end(),forces.begin()),
                   "Rejected step changed previous published force writeback");
            const auto changes=failed.force_threshold_changes();
            expect(changes.size()==threshold.size(),
                   "Rejected step changed previous force-threshold events");
            for(std::size_t i=0;i<changes.size();++i)
                expect(changes[i].key==threshold[i].key &&
                       changes[i].normal_impulse==threshold[i].normal_impulse &&
                       changes[i].change==threshold[i].change,
                       "Rejected step modified previous threshold payload");
        }
        const fyzz::Material ordinary{};
        failed.set_material(bad_floor,ordinary);
        failed.set_material(bad_box,ordinary);
        clean.set_material(good_floor,ordinary);
        clean.set_material(good_box,ordinary);
        failed.step(dt);
        clean.step(dt);
        const auto a=failed.body_state(bad_box);
        const auto b=clean.body_state(good_box);
        const auto near=[](float x,float y) {
            return std::abs(x-y)<2.0e-5f;
        };
        expect(near(a.position.x,b.position.x) &&
               near(a.position.y,b.position.y) &&
               near(a.position.z,b.position.z) &&
               near(a.velocity.x,b.velocity.x) &&
               near(a.velocity.y,b.velocity.y) &&
               near(a.velocity.z,b.velocity.z) &&
               near(a.orientation.x,b.orientation.x) &&
               near(a.orientation.y,b.orientation.y) &&
               near(a.orientation.z,b.orientation.z) &&
               near(a.orientation.w,b.orientation.w) &&
               near(a.angular_velocity.z,b.angular_velocity.z) &&
               std::abs(a.angular_velocity.z)>1.0e-4f &&
               failed.frame_index()==clean.frame_index() &&
               failed.contact_edge_index(original_pair)==original_edge,
               "Failed step consumed an impulse or changed edge identity");
    }

    // The source EdgeInstance checkpoint is sparse. Stable contact
    // managers must not copy the historical handle watermark on every
    // simulation step merely to guarantee exception rollback.
    {
        Scene stable;
        stable.add_static_floor(0.0f);
        for(int i=0;i<16;++i)
            stable.add_dynamic_box({0.5f,0.5f,0.5f},
                {static_cast<float>(i)*3.0f,0.49f,0},1.0f);
        stable.step(1.0f/120.0f);
        expect(stable.active_contact_edge_count()==16,
               "Sparse edge checkpoint fixture lacks persistent managers");
        stable.step(1.0f/120.0f);
        expect(stable.active_contact_edge_count()==16 &&
               stable.edge_rollback_journal_size()==0,
               "Unchanged source contact edges were copied during PGS");
    }

    // An exception *after* an existing accurate edge is disconnected
    // must restore both endpoint list links and the connected bit. A
    // second independent contact with invalid combined material forces
    // the failure in contact preparation, after Gu island publication.
    {
        auto initialize=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto first=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            const auto second=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{3,0.49f,0},1.0f);
            scene.step(1.0f/120.0f);
            return std::tuple{floor,first,second};
        };
        Scene rejected, clean;
        const auto [rf,ra,rb]=initialize(rejected);
        const auto [cf,ca,cb]=initialize(clean);
        const fyzz::native::Pair ra_pair{rf,ra}, rb_pair{rf,rb};
        const auto id=rejected.contact_edge_index(ra_pair);
        const auto* const old_contacts=rejected.contacts().data();
        const auto* const old_points=rejected.contacts()[0].points.data();
        const auto old_frame=rejected.frame_index();
        rejected.set_position(ra,{0,8,0});
        clean.set_position(ca,{0,8,0});
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(rf,exploding);
        rejected.set_material(rb,exploding);
        bool threw=false;
        try {rejected.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {threw=true;}
        expect(threw &&
               rejected.edge_rollback_journal_size()>0 &&
               rejected.frame_index()==old_frame &&
               rejected.contacts().data()==old_contacts &&
               rejected.contacts()[0].points.data()==old_points &&
               rejected.contact_edge_index(ra_pair)==id &&
               rejected.contact_edge_connected(ra_pair) &&
               rejected.contact_edge_connected(rb_pair),
               "Failed PGS step did not restore intrusive contact links");
        rejected.set_material(rf,fyzz::Material{});
        rejected.set_material(rb,fyzz::Material{});
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        expect(rejected.contact_edge_index(ra_pair)==id &&
               !rejected.contact_edge_connected(ra_pair) &&
               rejected.contact_edge_connected(rb_pair) &&
               rejected.contact_edge_connected(ra_pair)==
                   clean.contact_edge_connected({cf,ca}) &&
               rejected.contact_edge_connected(rb_pair)==
                   clean.contact_edge_connected({cf,cb}) &&
               rejected.accurate_island_count()==clean.accurate_island_count(),
               "Sparse intrusive rollback contaminated successful retry");
    }

    // A failed solver step can happen after SAP creates a new contact
    // edge. Rollback must not consume a handle or leave its accurate
    // connection bit in the scene. Retrying must produce the same edge
    // identity as a scene that never attempted the unsuccessful step.
    {
        auto make=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto a=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            scene.step(1.0f/120.0f);
            const auto b=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,0.49f,0},1.0f);
            return std::tuple{floor,a,b};
        };
        Scene rollback_scene, reference_scene;
        const auto [rf,ra,rb]=make(rollback_scene);
        const auto [cf,ca,cb]=make(reference_scene);
        const fyzz::native::Pair old_pair{rf,ra}, new_pair{rf,rb};
        const auto old_handle=rollback_scene.contact_edge_index(old_pair);
        const auto old_count=rollback_scene.active_contact_edge_count();
        const auto invalid=std::numeric_limits<std::uint32_t>::max();
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        for(const auto id:{rf,ra,rb})
            rollback_scene.set_material(id,exploding);
        bool rejected=false;
        try {rollback_scene.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               rollback_scene.active_contact_edge_count()==old_count &&
               rollback_scene.contact_edge_index(old_pair)==old_handle &&
               rollback_scene.contact_edge_index(new_pair)==invalid &&
               rollback_scene.contact_edge_connected(old_pair),
               "Failed contact preparation leaked a newly allocated EdgeIndex");
        for(const auto id:{rf,ra,rb})
            rollback_scene.set_material(id,fyzz::Material{});
        rollback_scene.step(1.0f/120.0f);
        reference_scene.step(1.0f/120.0f);
        expect(rollback_scene.contact_edge_index(new_pair)==
                   reference_scene.contact_edge_index({cf,cb}) &&
               rollback_scene.contact_edge_connected(new_pair) &&
               rollback_scene.contact_edge_index(old_pair)==
                   reference_scene.contact_edge_index({cf,ca}),
               "Rejected simulation changed subsequent contact-edge identity");
    }

    // Late integration failures also must not expose earlier body updates.
    // The second body's near-FLT_MAX origin plus the finite final motion
    // yields an unrepresentable position after the first body integrated.
    {
        Scene rollback_pose;
        const auto first=rollback_pose.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        const auto second=rollback_pose.add_dynamic_box(
            {0.5f,0.5f,0.5f},
            {std::numeric_limits<float>::max(),10,0},1.0f);
        rollback_pose.set_velocity(second,{1.0e18f,0,0});
        const auto before_a=rollback_pose.body_state(first);
        const auto before_b=rollback_pose.body_state(second);
        const auto frame=rollback_pose.frame_index();
        bool threw=false;
        try {rollback_pose.step(1.0e17f);}
        catch(const std::overflow_error&) {threw=true;}
        const auto after_a=rollback_pose.body_state(first);
        const auto after_b=rollback_pose.body_state(second);
        expect(threw &&
               before_a.position.x==after_a.position.x &&
               before_a.position.y==after_a.position.y &&
               before_a.velocity.y==after_a.velocity.y &&
               before_b.position.x==after_b.position.x &&
               before_b.velocity.x==after_b.velocity.x &&
               rollback_pose.frame_index()==frame,
               "Late pose integration error published earlier body changes");
        rollback_pose.set_velocity(second,{0,0,0});
        rollback_pose.step(1.0f/120.0f);
        expect(rollback_pose.frame_index()==frame+1 &&
               rollback_pose.body_state(first).position.y<10,
               "Failed integration prevented subsequent valid simulation");
    }

    // Authored pose edits invalidate the affected Dy friction cache.
    // A tangent-only teleport leaves old anchor separation projected
    // along the normal at zero, so Dy pointsAreClose alone cannot reject
    // it. Compare the next actual solve against an uncached Scene seeded
    // with the same physical state, for both collision participants.
    {
        constexpr float dt=1.0f/120.0f;
        for (const bool relocate_floor : {false,true}) {
            Scene cached;
            const auto floor=cached.add_static_floor(0.0f);
            const auto box=cached.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            cached.step(dt);
            cached.step(dt);
            const auto patches=cached.solved_friction_patches();
            expect(patches.size()==1 &&
                   patches[0].anchor_count>0 && !patches[0].broken,
                   "Teleport test did not establish reusable friction anchors");
            const auto before=cached.body_state(box);
            expect(std::abs(before.orientation.x)<1.0e-5f &&
                   std::abs(before.orientation.z)<1.0e-5f,
                   "Teleport baseline has unexpected angular motion");
            const fyzz::Vec3 shifted_body{
                before.position.x+(relocate_floor?0.0f:0.6f),
                before.position.y,before.position.z};
            if (relocate_floor)
                cached.set_position(floor,{0.6f,0.0f,0.0f});
            else
                cached.set_position(box,shifted_body);

            Scene fresh;
            const auto new_floor=fresh.add_static_floor(0.0f);
            const auto new_box=fresh.add_dynamic_box(
                {0.5f,0.5f,0.5f},shifted_body,1.0f);
            if (relocate_floor)
                fresh.set_position(new_floor,{0.6f,0.0f,0.0f});
            fresh.set_velocity(new_box,before.velocity);
            fresh.set_angular_velocity(new_box,before.angular_velocity);

            cached.step(dt);
            fresh.step(dt);
            const auto a=cached.body_state(box);
            const auto b=fresh.body_state(new_box);
            // The settled prior scene has a small nonidentity orientation
            // not expressible by the box-only fresh-scene API. This bound
            // tolerates its sub-millimeter-per-second contact differences
            // while rejecting teleport-induced friction kicks.
            constexpr float tolerance=2.0e-3f;
            expect(std::abs(a.velocity.x-b.velocity.x)<tolerance &&
                   std::abs(a.velocity.y-b.velocity.y)<tolerance &&
                   std::abs(a.velocity.z-b.velocity.z)<tolerance &&
                   std::abs(a.angular_velocity.x-b.angular_velocity.x)<tolerance &&
                   std::abs(a.angular_velocity.y-b.angular_velocity.y)<tolerance &&
                   std::abs(a.angular_velocity.z-b.angular_velocity.z)<tolerance,
                   "Explicit tangent teleport reused invalid friction anchors");
        }
    }

    // DySolverConstraints.cpp::solveContact(_BStatic) resets its
    // per-call friction break condition and only stores it during the
    // final velocity/writeback solve. With equal default static/dynamic
    // coefficients (0.5), a row that actually clamps on the FINAL pass
    // ends at the exact per-anchor Coulomb limit. Thus a broken patch
    // must have at least one reported anchor at that limit; saturation
    // during an earlier position iteration cannot justify "broken".
    // Exercise persistent anchors through a range of slip magnitudes so
    // false historical breaks also poison the next frame's cache.
    {
        bool observed_break=false, observed_unbroken=false;
        for (const float sliding_speed :
             {0.0f,0.01f,0.04f,0.08f,0.2f,0.6f,2.0f,4.0f}) {
            Scene final_pass;
            final_pass.add_static_floor(0.0f);
            const auto box=final_pass.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            final_pass.set_velocity(box,{sliding_speed,0,0});
            for (unsigned frame=0;frame<32;++frame) {
                final_pass.step(1.0f/120.0f);
                if (final_pass.contacts().size()!=1 ||
                    final_pass.solved_friction_patches().size()!=1)
                    continue;
                const auto& patch=final_pass.solved_friction_patches()[0];
                if (!patch.anchor_count) continue;
                double normal_sum=0;
                for(const double impulse: final_pass.solved_normal_impulses())
                    normal_sum+=impulse;
                if (normal_sum<=1.0e-6) continue;
                const double anchor_limit=0.5*normal_sum/
                    static_cast<double>(patch.anchor_count);
                double largest_anchor_impulse=0;
                for(std::size_t k=0;k<patch.anchor_count;++k) {
                    const auto& f=patch.anchor_impulses[k];
                    const double magnitude=std::sqrt(
                        f.x*f.x+f.y*f.y+f.z*f.z);
                    expect(std::isfinite(magnitude),
                           "Friction writeback produced nonfinite anchor impulse");
                    largest_anchor_impulse=std::max(
                        largest_anchor_impulse,magnitude);
                }
                if (patch.broken) {
                    observed_break=true;
                    expect(largest_anchor_impulse>=anchor_limit*0.995,
                           "Position-pass friction clamp leaked into final writeback");
                } else {
                    observed_unbroken=true;
                }
            }
        }
        expect(observed_break && observed_unbroken,
               "Friction final-writeback test did not exercise both break outcomes");
    }

    // DySolverConstraints::writeBackContact -> DyThresholdTable ->
    // PxsForceThresholdTask: a per-body threshold uses normal IMPULSE,
    // compared strictly against force*dt, and retains the preceding
    // exceeded-force stream across unsolved collision-only queries.
    // Static contacts share the source world node, while dynamic/dynamic
    // contacts retain their canonical BodyId pair identity.
    {
        using Change=fyzz::native::ForceThresholdChange;
        constexpr float dt=1.0f/120.0f;
        Scene thresholds;
        const auto ground=thresholds.add_static_floor(0);
        const auto monitored=thresholds.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto unmonitored=thresholds.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
        bool rejected=false;
        try { thresholds.set_contact_report_threshold(ground,1.0f); }
        catch(const std::logic_error&) {rejected=true;}
        expect(rejected,
               "Static actor admitted dynamic source contact threshold");
        rejected=false;
        try { thresholds.set_contact_report_threshold(monitored,-1.0f); }
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"Negative source contact threshold admitted");
        // A 1 kg supported body produces about 0.082 N*s per frame.
        // Threshold 5 N is exceeded only under PhysX's impulse > 5*dt
        // rule, NOT the erroneous impulse > 5 comparison.
        thresholds.set_contact_report_threshold(monitored,5.0f);
        thresholds.set_contact_report_threshold(unmonitored,30.0f);
        thresholds.step(dt);
        auto changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].change==Change::Found &&
               changes[0].normal_impulse>5.0*dt &&
               changes[0].normal_impulse<5.0 &&
               changes[0].threshold==5.0f,
               "Source force-threshold found stream missing/incorrect");
        const double first_impulse=changes[0].normal_impulse;
        thresholds.advance_collision(true);
        expect(thresholds.force_threshold_changes().empty(),
               "Unsolved Gu collision exposed stale threshold change stream");
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Persists &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].normal_impulse==first_impulse,
               "Source threshold persistence lost across collision-only query");
        thresholds.set_contact_report_threshold(monitored,
            std::numeric_limits<float>::max());
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].normal_impulse==0.0,
               "Disabling source threshold omitted lost-force report");
        thresholds.step(dt);
        expect(thresholds.force_threshold_changes().empty(),
               "Lost threshold was repeated after source state retirement");
        thresholds.set_contact_report_threshold(monitored,0.0f);
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Found,
               "Re-enabling threshold did not generate a new found report");
        thresholds.set_position(monitored,{0,5,0});
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost,
               "Loss of contact failed to retire exceeded-force pair");
    }

    // DySolverConstraints::writeBackContact uses the MINIMUM of the
    // two dynamic bodies' thresholds for a rigid-body contact manager.
    // Raising the lower threshold beyond the solved pair force must
    // produce LOST without retiring the underlying Gu contact.
    {
        using Change=fyzz::native::ForceThresholdChange;
        Scene pair_threshold;
        const auto first=pair_threshold.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto second=pair_threshold.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,5,0},1.0f);
        pair_threshold.set_velocity(second,{-2,0,0});
        pair_threshold.set_contact_report_threshold(first,1000.0f);
        pair_threshold.set_contact_report_threshold(second,0.0f);
        pair_threshold.step(1.0f/120.0f);
        auto changes=pair_threshold.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==fyzz::native::ForceThresholdKey{first,second,false} &&
               changes[0].change==Change::Found &&
               changes[0].threshold==0.0f,
               "Dy threshold stream did not use minimum dynamic-body threshold");
        pair_threshold.set_contact_report_threshold(second,1000.0f);
        pair_threshold.step(1.0f/120.0f);
        changes=pair_threshold.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==fyzz::native::ForceThresholdKey{first,second,false} &&
               changes[0].normal_impulse==0.0,
               "Dy exceeded stream failed to retire when pair threshold rose");
    }

    // PhysX PxRigidDynamic::getContactReportThreshold and
    // DyThresholdTable group all static contacts through the ONE static
    // world solver node. In this fixture neither support alone exceeds
    // a threshold strictly between the larger support impulse and their
    // SUM; aggregation by individual Gu shape Pair incorrectly emits
    // no Found event (or duplicates if each is above the threshold).
    {
        using Change=fyzz::native::ForceThresholdChange;
        constexpr float dt=1.0f/120.0f;
        auto populate=[](Scene& scene) {
            const auto left=scene.add_static_box(
                {0.5f,0.5f,1.0f},{-0.5f,-0.5f,0});
            const auto right=scene.add_static_box(
                {0.5f,0.5f,1.0f},{0.5f,-0.5f,0});
            const auto load=scene.add_dynamic_box(
                {0.75f,0.5f,0.75f},{0,0.5f,0},1.0f);
            scene.set_velocity(load,{0,-1.0f,0});
            return std::array<fyzz::BodyId,3>{left,right,load};
        };
        Scene baseline;
        const auto ids=populate(baseline);
        baseline.step(dt);
        const auto managers=baseline.contacts();
        const auto impulses=baseline.solved_normal_impulses();
        expect(managers.size()==2 &&
               managers[0].pair==fyzz::native::Pair{ids[0],ids[2]} &&
               managers[1].pair==fyzz::native::Pair{ids[1],ids[2]},
               "Static-world threshold fixture failed to create two supports");
        std::array<double,2> sums{};
        for (std::size_t j=0;j<2;++j) {
            const auto& manager=managers[j];
            expect(manager.contact_offset+manager.count<=impulses.size(),
                   "Static support impulse offsets were invalid");
            for (std::size_t k=0;k<manager.count;++k)
                sums[j]+=impulses[manager.contact_offset+k];
        }
        const double total=sums[0]+sums[1];
        const double largest=std::max(sums[0],sums[1]);
        expect(sums[0]>0.0 && sums[1]>0.0 &&
               total>largest+0.005,
               "Both static supports must contribute normal impulse");
        const float threshold=static_cast<float>(
            (total+largest)/(2.0*dt));
        expect(largest < static_cast<double>(threshold)*dt &&
               static_cast<double>(threshold)*dt < total,
               "Threshold fixture did not separate individual and summed force");
        Scene combined;
        const auto both=populate(combined);
        combined.set_contact_report_threshold(both[2],threshold);
        combined.step(dt);
        auto changes=combined.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{both[2],0,true} &&
               changes[0].change==Change::Found &&
               std::abs(changes[0].normal_impulse-total)<0.001,
               "Static world contacts were not summed before threshold comparison");
        // The shape-partner identity must not survive in the stream:
        // removing both supports causes exactly one Lost event, followed
        // by no further exceeded-world threshold reports.
        combined.remove(both[0]);
        combined.remove(both[1]);
        combined.step(dt);
        changes=combined.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{both[2],0,true} &&
               changes[0].change==Change::Lost &&
               changes[0].normal_impulse==0.0,
               "Shared static-world exceeded pair was not retired once");
        combined.step(dt);
        expect(combined.force_threshold_changes().empty(),
               "Lost static-world threshold repeated after retirement");
        // With zero threshold, both separately qualifying static
        // contacts must still generate ONE aggregated Found report.
        Scene duplicates;
        const auto duplicate_ids=populate(duplicates);
        duplicates.set_contact_report_threshold(duplicate_ids[2],0.0f);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true} &&
               changes[0].change==Change::Found &&
               std::abs(changes[0].normal_impulse-total)<0.001,
               "Both qualifying static supports generated duplicate reports");
        const double prior_impulse=changes[0].normal_impulse;
        // Changing the static shape partner must preserve the ONE world
        // node identity. A per-shape key would incorrectly report Lost
        // for one static pair even while the other support persists.
        duplicates.remove(duplicate_ids[0]);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true} &&
               changes[0].change==Change::Persists &&
               changes[0].normal_impulse==prior_impulse,
               "Changing static support changed source world-node identity");
        duplicates.remove(duplicate_ids[1]);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true},
               "Final static support removal omitted shared world-node Lost");
    }

    // PxsMaterialCombiner.h: a material belongs to each authored native
    // shape, not to a global/parallel material registry. The source uses
    // the higher-precedence combine mode: MAX beats MIN/AVERAGE. A disabled
    // friction flag on EITHER side wins and suppresses all tangent rows.
    {
        using fyzz::Material;
        using Mode=fyzz::MaterialCombineMode;
        auto friction_run=[](Mode mode, bool disable) {
            Scene sample;
            const auto ground=sample.add_static_floor(0.0f);
            const auto box=sample.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            Material floor_material{};
            floor_material.static_friction=0.0f;
            floor_material.dynamic_friction=0.0f;
            floor_material.friction_combine=Mode::Average;
            Material box_material{};
            box_material.static_friction=1.0f;
            box_material.dynamic_friction=1.0f;
            box_material.friction_combine=mode;
            box_material.disable_friction=disable;
            sample.set_material(ground,floor_material);
            sample.set_material(box,box_material);
            sample.set_velocity(box,{0.2f,0,0});
            sample.step(1.0f/120.0f);
            double total_tangential_impulse=0.0;
            for(const auto& patch:sample.solved_friction_patches()) {
                for(std::size_t i=0;i<patch.anchor_count;++i)
                    total_tangential_impulse+=
                        std::abs(patch.anchor_impulses[i].x);
            }
            return std::pair{sample.body_state(box).velocity.x,
                             total_tangential_impulse};
        };
        const auto minimum=friction_run(Mode::Minimum,false);
        const auto average=friction_run(Mode::Average,false);
        const auto multiply=friction_run(Mode::Multiply,false);
        const auto maximum=friction_run(Mode::Maximum,false);
        const auto disabled=friction_run(Mode::Maximum,true);
        expect(std::abs(minimum.first-0.2f)<0.002f &&
               minimum.second<1.0e-8 &&
               multiply.second<1.0e-8 &&
               average.second>0.001 &&
               maximum.second>average.second &&
               maximum.first<minimum.first-0.001f,
               "Source contact friction mode precedence was not applied");
        expect(std::abs(disabled.first-0.2f)<0.002f &&
               disabled.second==0.0,
               "PxsMaterial flag eDISABLE_FRICTION did not suppress tangent rows");
        Scene invalid;
        const auto box=invalid.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,2,0},1.0f);
        Material compliant{};
        compliant.restitution=-0.1f;
        bool refused=false;
        try {invalid.set_material(box,compliant);}
        catch(const std::invalid_argument&) {refused=true;}
        expect(refused,
               "Unimplemented source compliant-spring material was silently admitted");
    }
    // PxsCombineMaterials' restitution mode uses the higher-precedence
    // mode independently of the friction mode. Keep both rigid inputs
    // nonnegative: compliant springs are a separate Dy solver branch.
    {
        using fyzz::Material;
        using Mode=fyzz::MaterialCombineMode;
        auto rebound=[](Mode mode, float initial_height=0.49f) {
            Scene bounce;
            const auto ground=bounce.add_static_floor(0.0f);
            const auto box=bounce.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,initial_height,0},1.0f);
            Material floor_material{};
            floor_material.restitution=0.0f;
            floor_material.restitution_combine=Mode::Average;
            Material box_material{};
            box_material.restitution=1.0f;
            box_material.restitution_combine=mode;
            bounce.set_material(ground,floor_material);
            bounce.set_material(box,box_material);
            bounce.set_velocity(box,{0,-4,0});
            bounce.step(1.0f/120.0f);
            const auto state=bounce.body_state(box);
            return std::pair{state.velocity.y,state.position.y};
        };
        const auto low=rebound(Mode::Minimum);
        const auto multiply=rebound(Mode::Multiply);
        const auto average=rebound(Mode::Average);
        const auto high=rebound(Mode::Maximum);
        const auto touching=rebound(Mode::Maximum,0.5f);
        expect(std::isfinite(low.first) && std::isfinite(high.first) &&
               std::isfinite(multiply.first) && std::isfinite(average.first) &&
               std::abs(low.first-multiply.first)<0.02f &&
               average.first>low.first+0.4f &&
               high.first>average.first+0.4f,
               "Source material restitution combine modes did not order rebound");
        // DyContactPrepShared clears scaledBias for a qualifying bounce
        // BEFORE the position-solver motion snapshot is captured. A
        // slightly penetrating body must not acquire an additional upward
        // motion correction compared with the otherwise identical touching
        // contact. Final unbiased velocity alone cannot detect this bug.
        expect(std::isfinite(high.second) && std::isfinite(touching.second) &&
               std::abs((high.second-touching.second)+0.01f)<0.002f,
               "Restitution contact retained penetration bias during position solve");
    }

    // Material authoring must flow through the real public World
    // NativeCpu backend, not just through native::Scene test fixtures.
    {
        fyzz::World world{
            fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu}};
        const auto ground=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material surface{};
        surface.static_friction=0.0f;
        surface.dynamic_friction=0.0f;
        surface.friction_combine=fyzz::MaterialCombineMode::Minimum;
        world.set_body_material(ground,surface);
        world.set_body_material(box,surface);
        world.set_body_velocity(box,{0.3f,0,0});
        world.step(1.0f/120.0f);
        expect(std::abs(world.body_state(box).velocity.x-0.3f)<0.002f,
               "Public World noncompliant material authoring missed native solver");
        surface.restitution=-0.2f;
        bool invalid=false;
        try{world.set_body_material(box,surface);}
        catch(const std::invalid_argument&){invalid=true;}
        expect(invalid,
               "Public World accepted unsupported compliant-source material");
    }

#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    // The reference adapter must bind a real PxMaterial to its existing
    // PxShape, preserving the very same public World authoring contract.
    {
        fyzz::World world{
            fyzz::WorldOptions{fyzz::ExecutionMode::ReferencePhysXCpu}};
        const auto ground=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material frictionless{};
        frictionless.static_friction=0.0f;
        frictionless.dynamic_friction=0.0f;
        world.set_body_material(ground,frictionless);
        world.set_body_material(box,frictionless);
        world.set_body_velocity(box,{0.3f,0,0});
        world.step(1.0f/120.0f);
        expect(std::abs(world.body_state(box).velocity.x-0.3f)<0.005f,
               "World reference adapter failed to author PhysX shape material");
    }
#endif

    verify_demand_driven_island_ownership_checkpoint();
    verify_source_contact_edge_deactivations();
    verify_source_retired_bridge_deactivation();
    verify_source_active_island_handoff();
    verify_source_speculative_reverse_sleep_order();
    verify_source_mixed_split_and_sleep_order();
    verify_source_solver_task_batch_partitioning();
    verify_source_reverse_active_island_sleep_retirement();
    verify_source_reverse_wake_order_and_rollback_capacity();
    verify_source_active_reference_lifecycle();
    verify_source_contact_interaction_consumption();
    verify_source_speculative_only_active_references();
    verify_source_activity_rollback();
    verify_native_rigid_integration_and_island_lifecycle();

    // Exercise the actual World selection, not only the owner fixture.
    fyzz::World native_world(fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu});
    auto owned = native_world.add_dynamic_box({0.5f,0.5f,0.5f}, {0,1,0}, 1);
    expect(native_world.body_state(owned).position.y == 1,
           "SDK World did not own native actor state");
    native_world.set_body_position(owned, {0,3,0});
    expect(native_world.body_state(owned).position.y == 3,
           "World did not relocate its native actor");
    native_world.set_body_velocity(owned, {1,0,0});
    native_world.step(1.0f/60.0f);
    expect(native_world.body_state(owned).position.x > 0.0f &&
           native_world.body_state(owned).position.y < 3.0f,
           "Native World step did not advance gravity and actor state");
    native_world.remove_body(owned);
    bool retired = false;
    try { (void)native_world.body_state(owned); }
    catch (const std::out_of_range&) { retired = true; }
    expect(retired, "World did not retire the native actor");
    return 0;
}

int main() {
    try {
        return run_native_scene_smoke();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native scene smoke: %s\n", e.what());
        return 1;
    }
}
