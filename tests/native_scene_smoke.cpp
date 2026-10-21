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
               connected.island_full_rebuild_count()==initial_rebuilds+1,
               "Lost source contact edges did not split both island graphs");
        connected.set_position(b,{0.85f,5.0f,0.0f});
        connected.advance_collision();
        expect(connected.speculative_island_count()==2 &&
               connected.accurate_island_count()==2 &&
               connected.island_full_rebuild_count()==initial_rebuilds+1,
               "Restored source contact edges did not incrementally rejoin islands");
        connected.remove(b);
        connected.advance_collision();
        expect(connected.speculative_island_count()==3 &&
               connected.accurate_island_count()==3 &&
               connected.island_full_rebuild_count()==initial_rebuilds+2,
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
               cycle.island_full_rebuild_count()==cycle_rebuilds+1,
               "Source island split lost its final surviving route");
        cycle.set_position(cb,{0.4f,8,0});
        cycle.advance_collision();
        expect(cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.largest_accurate_island_size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds+1,
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
               relink.island_full_rebuild_count()==before_recycle+1,
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

        // The chord reintroduction does not merge distinct roots, so
        // root-directed fast-route hints remain valid. Removing those
        // same redundant chords must now reuse the route discovered above.
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
               dirty_routes.island_fast_route_hit_count()==fast_before_retry+4,
               "Redundant chord readdition discarded valid source fast routes");

        // Retiring the last surviving tip edge is a real split.
        dirty_routes.set_position(route_tip,{3.0f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.accurate_island_count()==2 &&
               dirty_routes.speculative_island_count()==2 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild+1,
               "Dirty fast route falsely preserved a disconnected island");

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
        // The successful retry must reuse all four cached non-root
        // routes just as the never-failed scene does.
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
                   rejected.island_fast_route_hit_count()==hits+4 &&
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
               candidate_only.island_full_rebuild_count()==candidate_rebuilds+1,
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
        // Source Dy contact preparation can require an impulse above the
        // representable PxReal range for an SDK-admitted, extremely heavy
        // translational body. Narrowing its biased/unbiased error to float
        // must not reject the actor or lose a finite solver velocity delta.
        Scene heavy;
        heavy.add_static_box({1,1,1},{0,5,0});
        const auto body = heavy.add_dynamic_box(
            {1,1,1},{1.5f,5,0},std::numeric_limits<float>::max());
        heavy.set_velocity(body,{-1,0,0});
        heavy.step(1.0f/120.0f);
        const auto state = heavy.body_state(body);
        expect(std::isfinite(state.position.x) &&
               std::isfinite(state.velocity.x) &&
               std::isfinite(state.velocity.y) &&
               state.position.x > 1.5f &&
               state.velocity.x > -1.0f &&
               heavy.contacts().size() == 1,
               "Valid heavy translation-only contact was lost to float impulse narrowing");
        bool angular_rejected = false;
        try { heavy.apply_torque_impulse(body,{1,0,0}); }
        catch(const std::logic_error&) { angular_rejected = true; }
        expect(angular_rejected,
               "Heavy body with nonrepresentable angular inertia accepted torque");
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
        // Finite per-body inverse masses can sum beyond float response.
        // The source position-iteration motion must separate the overlap;
        // unbiased velocity iterations can return zero final velocities.
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
        very_light.step(1.0f / 120.0f);
        const auto lv = very_light.body_state(left);
        const auto rv = very_light.body_state(right);
        expect(std::isfinite(lv.velocity.x) &&
               std::isfinite(rv.velocity.x) &&
               std::isfinite(lv.position.x) &&
               std::isfinite(rv.position.x) &&
               lv.position.x < 0.0f && rv.position.x > 1.5f &&
               very_light.contacts().size() == 1,
               "Finite tiny-mass position correction was silently discarded");
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
        Scene unequal_mass;
        const auto light = unequal_mass.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1.0e-6f);
        const auto heavy = unequal_mass.add_dynamic_box(
            {1,1,1}, {1.5f,5,0}, 1.0e6f);
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
               "Source dynamic linear-speed limit was not enforced");
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
               still.velocity.y==0.0f,
               "PhysX island did not deactivate a settled floor body");
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
