// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
void expect(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
bool has(std::span<const fyzz::native::Pair> p, fyzz::BodyId a, fyzz::BodyId b) {
    const fyzz::native::Pair match{std::min(a,b), std::max(a,b)};
    return std::find(p.begin(), p.end(), match) != p.end();
}
}
int main() {
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

        // Per-axis AABBs still overlap inside the pair contact offset,
        // but diagonal gaps have no intersecting incident/reference faces.
        // This face-only implementation must not invent a midpoint contact.
        box_scene.set_position(moveable, {2.03f,2.03f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().empty() &&
               has(box_scene.lost_contacts(), fixed, moveable) &&
               box_scene.lost_pairs().empty(),
               "Diagonal gap fabricated a face manifold");
        box_scene.advance_collision(true);
        expect(box_scene.contacts().empty() &&
               box_scene.found_contacts().empty() &&
               box_scene.lost_contacts().empty(),
               "Forced diagonal-gap update published spurious touch");
        // The true edge distance here is within the offset, but accurate
        // edge contacts are not yet ported: reject rather than fabricate.
        box_scene.set_position(moveable, {2.01f,2.01f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().empty() &&
               box_scene.found_contacts().empty(),
               "Unimplemented edge contact was misreported as a face");

        // A single-axis near contact with positive separation is still
        // supported when the two transverse face intervals overlap.
        box_scene.set_position(moveable, {1.5f,2.03f,0});
        box_scene.advance_collision();
        expect(has(box_scene.found_contacts(), fixed, moveable) &&
               box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].count == 4 &&
               box_scene.contacts()[0].points[0].normal.y == -1.0f &&
               box_scene.contacts()[0].points[0].separation > 0.029f,
               "Valid single-axis proximity contact was lost");
    }


    // End-to-end native CPU execution, without a reference PxScene.
    {
        Scene falling_scene;
        falling_scene.add_static_floor(0);
        const auto falling = falling_scene.add_dynamic_box(
            {0.25f,0.25f,0.25f}, {0,2,0}, 1);
        for (int frame = 0; frame < 360; ++frame)
            falling_scene.step(1.0f / 120.0f);
        const auto state = falling_scene.body_state(falling);
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
        Scene stacked;
        stacked.add_static_box({2,0.5f,2},{0,-0.5f,0});
        const auto lower = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,2,0},1);
        const auto upper = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,3.5f,0},1);
        for (int frame = 0; frame < 600; ++frame)
            stacked.step(1.0f / 120.0f);
        const auto lo = stacked.body_state(lower);
        const auto hi = stacked.body_state(upper);
        expect(std::isfinite(lo.position.y) &&
               std::isfinite(hi.position.y) &&
               lo.position.y > 0.37f && lo.position.y < 0.65f &&
               hi.position.y > 1.25f && hi.position.y < 1.70f &&
               std::abs(lo.velocity.y) < 0.8f &&
               std::abs(hi.velocity.y) < 0.8f,
               "Native two-body contact chain failed");
    }

    // Exercise the actual World selection, not only the owner fixture.
    fyzz::World native_world(fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu});
    auto owned = native_world.add_dynamic_box({0.5f,0.5f,0.5f}, {0,1,0}, 1);
    expect(native_world.body_state(owned).position.y == 1,
           "SDK World did not own native actor state");
    native_world.set_body_position(owned, {0,3,0});
    expect(native_world.body_state(owned).position.y == 3,
           "World did not relocate its native actor");
    native_world.step(1.0f/60.0f);
    expect(native_world.body_state(owned).position.y < 3.0f,
           "Native World step did not advance gravity and actor state");
    native_world.remove_body(owned);
    bool retired = false;
    try { (void)native_world.body_state(owned); }
    catch (const std::out_of_range&) { retired = true; }
    expect(retired, "World did not retire the native actor");
    return 0;
}
