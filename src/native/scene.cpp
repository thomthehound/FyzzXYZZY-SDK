// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace fyzz::native {
namespace {

// Source transcription: PhysX BpBroadPhaseIntegerAABB.h::encodeFloat,
// IntegerAABB::encodeFloatMin/Max, with its four-bit conservative expansion.
std::uint32_t encode_float(float v) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
std::uint32_t encode_min(float v) {
    return ((encode_float(v) >> 4) - 1u) << 4;
}
std::uint32_t encode_max(float v) {
    return (((encode_float(v) >> 4) + 1u) << 4) | 4u;
}
bool finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
void validate_box(Vec3 h, Vec3 p) {
    if (!finite(h) || !finite(p) ||
        h.x <= 0.0f || h.y <= 0.0f || h.z <= 0.0f)
        throw std::invalid_argument("Native scene requires positive finite box geometry");
}
Pair ordered_pair(BodyId a, BodyId b) {
    return {std::min(a, b), std::max(a, b)};
}
} // namespace

BodyId Scene::add(Actor actor) {
    if (actors_.size() >= std::numeric_limits<BodyId>::max())
        throw std::overflow_error("Native scene body identifier exhausted");
    const auto id = static_cast<BodyId>(actors_.size());
    actor.alive = true;
    actors_.push_back(actor);
    dirty_ = true;
    return id;
}
BodyId Scene::add_static_box(Vec3 h, Vec3 p) {
    validate_box(h, p);
    Actor a;
    a.kind = Kind::StaticBox;
    a.half_extents = h;
    a.position = p;
    return add(a);
}
BodyId Scene::add_static_floor(float height) {
    if (!std::isfinite(height))
        throw std::invalid_argument("Native scene floor must be finite");
    Actor a;
    a.kind = Kind::StaticPlane;
    a.position = {0.0f, height, 0.0f};
    return add(a);
}
BodyId Scene::add_dynamic_box(Vec3 h, Vec3 p, float density) {
    validate_box(h, p);
    if (!std::isfinite(density) || density <= 0.0f)
        throw std::invalid_argument("Native scene density must be finite and positive");
    Actor a;
    a.kind = Kind::DynamicBox;
    a.half_extents = h;
    a.position = p;
    a.density = density;
    // Cache source-derived inverse mass once, with a checked double
    // intermediate so extreme but finite dimensions cannot silently
    // turn a dynamic body into an infinite-mass actor.
    const double mass = 8.0 * static_cast<double>(h.x) * h.y * h.z * density;
    const double inverse = 1.0 / mass;
    if (!std::isfinite(inverse) || inverse <= 0.0 ||
        inverse > std::numeric_limits<float>::max() ||
        static_cast<float>(inverse) == 0.0f)
        throw std::invalid_argument("Native dynamic-box mass outside supported range");
    a.inverse_mass = static_cast<float>(inverse);
    return add(a);
}
Scene::Actor& Scene::require(BodyId id) {
    if (id >= actors_.size() || !actors_[id].alive)
        throw std::out_of_range("Unknown or retired native body ID");
    return actors_[id];
}
const Scene::Actor& Scene::require(BodyId id) const {
    if (id >= actors_.size() || !actors_[id].alive)
        throw std::out_of_range("Unknown or retired native body ID");
    return actors_[id];
}
void Scene::set_position(BodyId id, Vec3 position) {
    if (!finite(position))
        throw std::invalid_argument("Native scene position must be finite");
    auto& actor = require(id);
    // Infinite planes have a fixed world +Y normal in the current native
    // shape model. Translating one changes its height, not its orientation.
    actor.position = position;
    dirty_ = true;
    invalidate_contact_cache(id);
}
void Scene::set_velocity(BodyId id, Vec3 velocity) {
    if (!finite(velocity))
        throw std::invalid_argument("Native scene velocity must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors have no linear velocity");
    actor.velocity = velocity;
    invalidate_contact_cache(id);
}
void Scene::invalidate_contact_cache(BodyId id) {
    // Teleports and external velocity edits invalidate solver warm starts;
    // they do not manufacture pair/touch transitions.
    for (auto& contact : manifolds_) {
        if (contact.pair.a == id || contact.pair.b == id) {
            contact.normal_impulse = 0.0;
            contact.tangent_impulses = {};
        }
    }
}
void Scene::remove(BodyId id) {
    require(id).alive = false;
    dirty_ = true;
}
BodyState Scene::body_state(BodyId id) const {
    const auto& actor = require(id);
    return {actor.position, actor.velocity};
}
void Scene::update_bounds() {
    // Current reference SDK uses PhysX's ordinary 0.02m shape contact offset.
    // That value will become scene material/shape state in the source import.
    constexpr float contact_offset = 0.02f;
    endpoints_.clear();
    for (std::size_t index = 0; index < actors_.size(); ++index) {
        auto& actor = actors_[index];
        if (!actor.alive || actor.kind == Kind::StaticPlane) continue;
        const Vec3 p = actor.position, h = actor.half_extents;
        const std::array<float, 6> e{
            p.x-h.x-contact_offset, p.y-h.y-contact_offset,
            p.z-h.z-contact_offset, p.x+h.x+contact_offset,
            p.y+h.y+contact_offset, p.z+h.z+contact_offset};
        for (float v : e)
            if (!std::isfinite(v))
                throw std::overflow_error("Native scene AABB is not finite");
        for (unsigned axis = 0; axis < 3; ++axis) {
            actor.encoded_bounds[axis] = encode_min(e[axis]);
            actor.encoded_bounds[axis+3] = encode_max(e[axis+3]);
        }
        endpoints_.push_back({actor.encoded_bounds[0], static_cast<BodyId>(index)});
    }
    std::sort(endpoints_.begin(), endpoints_.end(),
        [](Endpoint a, Endpoint b) {
            return a.min_x < b.min_x || (a.min_x == b.min_x && a.id < b.id);
        });
}
void Scene::update_pairs() {
    pending_pairs_.clear();
    active_.clear();
    // Source-corresponding SAP: x-axis sweep and integer y/z overlap.
    // Retain ordered pair history across frames; exclude static/static.
    for (Endpoint ep : endpoints_) {
        const auto& actor = actors_[ep.id];
        const auto& a = actor.encoded_bounds;
        active_.erase(std::remove_if(active_.begin(), active_.end(),
            [&](BodyId id) { return actors_[id].encoded_bounds[3] < ep.min_x; }),
            active_.end());
        for (BodyId other_id : active_) {
            const auto& other = actors_[other_id];
            if (actor.kind != Kind::DynamicBox && other.kind != Kind::DynamicBox)
                continue;
            const auto& b = other.encoded_bounds;
            if (a[1] > b[4] || b[1] > a[4] ||
                a[2] > b[5] || b[2] > a[5]) continue;
            pending_pairs_.push_back(ordered_pair(ep.id, other_id));
        }
        active_.push_back(ep.id);
    }
    // The infinite plane is only a *candidate* broadphase shape, not a
    // resolved contact. Native narrowphase must reject noncontact candidates.
    for (std::size_t i = 0; i < actors_.size(); ++i) {
        if (!actors_[i].alive || actors_[i].kind != Kind::StaticPlane) continue;
        for (std::size_t j = 0; j < actors_.size(); ++j) {
            if (actors_[j].alive && actors_[j].kind == Kind::DynamicBox)
                pending_pairs_.push_back(ordered_pair(
                    static_cast<BodyId>(i), static_cast<BodyId>(j)));
        }
    }
    std::sort(pending_pairs_.begin(), pending_pairs_.end());
    pending_pairs_.erase(std::unique(pending_pairs_.begin(), pending_pairs_.end()),
                         pending_pairs_.end());
}
void Scene::publish_pairs() {
    found_.clear();
    lost_.clear();
    std::set_difference(pending_pairs_.begin(), pending_pairs_.end(),
                        previous_pairs_.begin(), previous_pairs_.end(),
                        std::back_inserter(found_));
    std::set_difference(previous_pairs_.begin(), previous_pairs_.end(),
                        pending_pairs_.begin(), pending_pairs_.end(),
                        std::back_inserter(lost_));
    previous_pairs_.swap(pending_pairs_);
}
void Scene::update_contacts() {
    // The source's default contact offsets are currently 0.02m per shape.
    // The near-contact generation distance is their *sum* (0.04m).
    // Once N1 imports per-shape state, this is computed per candidate.
    constexpr float contact_distance = 0.04f;
    next_manifolds_.clear();
    for (const Pair pair : previous_pairs_) {
        const Actor& a = actors_[pair.a];
        const Actor& b = actors_[pair.b];
        ContactManifold manifold{};
        manifold.pair = pair;

        if (a.kind == Kind::StaticPlane || b.kind == Kind::StaticPlane) {
            // Adapted from GuContactPlaneBox.cpp::contactPlaneBox:
            // test the eight box vertices against the plane and preserve
            // signed separation; a far broadphase candidate has no contact.
            const Actor& plane = a.kind == Kind::StaticPlane ? a : b;
            const Actor& box = a.kind == Kind::StaticPlane ? b : a;
            if (box.kind != Kind::DynamicBox)
                continue;
            // The canonical pair is ordered by body ID, not by geometry.
            // Source contact normals always point from shape1 (b) to
            // shape0 (a), regardless of which actor was created first.
            const Vec3 normal = a.kind == Kind::StaticPlane
                ? Vec3{0.0f, -1.0f, 0.0f}
                : Vec3{0.0f, 1.0f, 0.0f};
            for (unsigned corner = 0; corner < 8; ++corner) {
                const float sx = (corner & 1u) ? 1.0f : -1.0f;
                const float sy = (corner & 2u) ? 1.0f : -1.0f;
                const float sz = (corner & 4u) ? 1.0f : -1.0f;
                const Vec3 point{
                    box.position.x + sx * box.half_extents.x,
                    box.position.y + sy * box.half_extents.y,
                    box.position.z + sz * box.half_extents.z};
                const float separation = point.y - plane.position.y;
                if (separation < contact_distance) {
                    manifold.points[manifold.count++] =
                        {point, normal, separation};
                }
            }
        } else {
            // GuContactBoxBox.cpp Class I/II SAT restricted to the shapes
            // this native scene actually represents (unrotated boxes).
            // With identity orientations the three unique face axes are
            // sufficient; cross-product axes vanish. Clip the incident face
            // rectangle to the reference box's corresponding face extents.
            const std::array<float, 3> center_a{
                a.position.x, a.position.y, a.position.z};
            const std::array<float, 3> center_b{
                b.position.x, b.position.y, b.position.z};
            const std::array<float, 3> half_a{
                a.half_extents.x, a.half_extents.y, a.half_extents.z};
            const std::array<float, 3> half_b{
                b.half_extents.x, b.half_extents.y, b.half_extents.z};
            std::array<float, 3> signed_gaps{};
            bool separated = false;
            unsigned axis = 0;
            for (unsigned k = 0; k < 3; ++k) {
                signed_gaps[k] = std::abs(center_b[k] - center_a[k]) -
                                 (half_a[k] + half_b[k]);
                if (signed_gaps[k] > contact_distance) {
                    separated = true;
                    break;
                }
                if (k == 0u || signed_gaps[k] > signed_gaps[axis])
                    axis = k;
            }
            if (separated)
                continue;
            const float sign = center_b[axis] >= center_a[axis]
                ? -1.0f : 1.0f;
            std::array<float, 3> face = center_b;
            face[axis] += sign * half_b[axis];
            const unsigned u = (axis + 1) % 3;
            const unsigned v = (axis + 2) % 3;
            const float low_u = std::max(
                center_a[u] - half_a[u], center_b[u] - half_b[u]);
            const float high_u = std::min(
                center_a[u] + half_a[u], center_b[u] + half_b[u]);
            const float low_v = std::max(
                center_a[v] - half_a[v], center_b[v] - half_b[v]);
            const float high_v = std::min(
                center_a[v] + half_a[v], center_b[v] + half_b[v]);
            // A face manifold requires overlapping transverse intervals.
            // A diagonal edge/corner gap is NOT a face contact even when
            // the gaps are individually inside the contact distance.
            // Reserve these cases for source-derived edge/corner generation;
            // do not manufacture an intersection at a gap midpoint.
            if (low_u > high_u || low_v > high_v)
                continue;
            const float span_u[2] = {low_u, high_u};
            const float span_v[2] = {low_v, high_v};
            const unsigned count_u = low_u == high_u ? 1u : 2u;
            const unsigned count_v = low_v == high_v ? 1u : 2u;
            std::array<float, 3> normal{0.0f, 0.0f, 0.0f};
            normal[axis] = sign; // from shape b to a
            for (unsigned iv = 0; iv < count_v; ++iv) {
                for (unsigned iu = 0; iu < count_u; ++iu) {
                    auto point = face;
                    point[u] = span_u[iu];
                    point[v] = span_v[iv];
                    manifold.points[manifold.count++] = {
                        Vec3{point[0], point[1], point[2]},
                        Vec3{normal[0], normal[1], normal[2]},
                        signed_gaps[axis]};
                }
            }
        }
        if (manifold.count > 0)
            next_manifolds_.push_back(manifold);
    }

    // BP pair found/lost and narrowphase touch found/lost are distinct
    // source events. Report changed touch pairs exactly once per frame.
    found_contacts_.clear();
    lost_contacts_.clear();
    const auto pair_less = [](const ContactManifold& lhs,
                              const ContactManifold& rhs) {
        return lhs.pair < rhs.pair;
    };
    // Sorted broadphase candidate iteration preserves manifold order.
    auto next = next_manifolds_.begin();
    auto old = manifolds_.begin();
    while (next != next_manifolds_.end() && old != manifolds_.end()) {
        if (pair_less(*next, *old)) {
            found_contacts_.push_back(next->pair);
            ++next;
        } else if (pair_less(*old, *next)) {
            lost_contacts_.push_back(old->pair);
            ++old;
        } else {
            // Source-style persistent contact patch state follows the pair
            // only while its Jacobian direction remains compatible.
            const Vec3 n = next->points[0].normal;
            const Vec3 o = old->points[0].normal;
            if (n.x == o.x && n.y == o.y && n.z == o.z) {
                next->normal_impulse = old->normal_impulse;
                next->tangent_impulses = old->tangent_impulses;
            }
            ++next; ++old;
        }
    }
    for (; next != next_manifolds_.end(); ++next)
        found_contacts_.push_back(next->pair);
    for (; old != manifolds_.end(); ++old)
        lost_contacts_.push_back(old->pair);
    manifolds_.swap(next_manifolds_);
}
// Source correspondence:
//   DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
//     unconstrained linear velocity += gravity * dt
//   DyContactPrep.cpp: normal contact response from inverse masses,
//     separation and target velocity
//   DySolverConstraints.cpp::solveContact: unilateral accumulated PGS impulse
// Angular terms are constrained to zero in the present nonrotating-box API;
// this is not a general PhysX rigid-body solver, TGS or articulation port.
//
// Predict positions first, then generate contacts at published positions.
// This explicit scheduling departure avoids a second broadphase rebuild
// solely to display final contacts; full ScPipeline continuation parity
// remains a separate N1 acceptance requirement.
void Scene::prepare_constraints(float seconds) {
    constexpr float positional_bias = 0.2f;
    constexpr float max_depenetration_speed = 3.0f;
    constexpr float bounce_threshold = 2.0f;
    constexpr float restitution = 0.2f;
    for (auto& manifold : manifolds_) {
        const Actor& a = actors_[manifold.pair.a];
        const Actor& b = actors_[manifold.pair.b];
        const Vec3 n = manifold.points[0].normal;
        const float relative_velocity =
            (a.velocity.x - b.velocity.x) * n.x +
            (a.velocity.y - b.velocity.y) * n.y +
            (a.velocity.z - b.velocity.z) * n.z;
        float separation = manifold.points[0].separation;
        for (unsigned k = 1; k < manifold.count; ++k)
            separation = std::min(separation, manifold.points[k].separation);
        // Positive-separation (speculative) constraints never pull bodies
        // together; negative separation requests bounded correction.
        manifold.velocity_target = separation >= 0.0f
            ? -separation / seconds
            : std::min(max_depenetration_speed,
                       -positional_bias * separation / seconds);
        if (separation <= 0.0f && relative_velocity < -bounce_threshold)
            manifold.velocity_target = std::max(
                manifold.velocity_target, -restitution * relative_velocity);
        // Preserve the previous frame's contact impulses as a warm start,
        // but rescale for a changed timestep and discard extreme changes.
        if (previous_step_seconds_ > 0.0f) {
            const double scale = static_cast<double>(seconds) /
                                 previous_step_seconds_;
            if (scale >= 0.5 && scale <= 2.0) {
                manifold.normal_impulse *= scale;
                for (double& impulse : manifold.tangent_impulses)
                    impulse *= scale;
            } else {
                manifold.normal_impulse = 0.0;
                manifold.tangent_impulses = {};
            }
        }
    }
}
void Scene::solve_constraints() {
    // Source: DySolverConstraints.cpp normal/friction accumulated PGS;
    // DyContactPrep.cpp::frictionPatch correlates normal contact patches.
    // In the restricted zero-angular-response engine, all points on one
    // manifold share a translational Jacobian. Solve one normal + two
    // tangents per patch rather than duplicating full-body impulses.
    //
    // Reference PhysX default material is static=0.5, dynamic=0.5.
    // Coulomb disk projection here is a stated bounded specialization,
    // not a port of PhysX's angular friction-anchor correlation.
    constexpr double static_friction = 0.5;
    constexpr double dynamic_friction = 0.5;
    constexpr unsigned position_iterations = 4;
    constexpr unsigned velocity_iterations = 2;

    const auto tangents = [](Vec3 n) -> std::array<Vec3, 2> {
        if (n.x != 0.0f) return {Vec3{0,1,0}, Vec3{0,0,1}};
        if (n.y != 0.0f) return {Vec3{1,0,0}, Vec3{0,0,1}};
        return {Vec3{1,0,0}, Vec3{0,1,0}};
    };
    const auto dot = [](Vec3 v, Vec3 axis) -> double {
        return static_cast<double>(v.x) * axis.x +
               static_cast<double>(v.y) * axis.y +
               static_cast<double>(v.z) * axis.z;
    };
    const auto apply = [](Actor& a, Actor& b, Vec3 axis, double impulse) {
        const double da = impulse * a.inverse_mass;
        const double db = impulse * b.inverse_mass;
        a.velocity.x = static_cast<float>(static_cast<double>(a.velocity.x) + axis.x * da);
        a.velocity.y = static_cast<float>(static_cast<double>(a.velocity.y) + axis.y * da);
        a.velocity.z = static_cast<float>(static_cast<double>(a.velocity.z) + axis.z * da);
        b.velocity.x = static_cast<float>(static_cast<double>(b.velocity.x) - axis.x * db);
        b.velocity.y = static_cast<float>(static_cast<double>(b.velocity.y) - axis.y * db);
        b.velocity.z = static_cast<float>(static_cast<double>(b.velocity.z) - axis.z * db);
    };

    // Warm-start cached normal and tangential constraints before iteration.
    // Same-geometry impulses are retained by update_contacts' sorted merge;
    // external edits, lost contacts and normal changes invalidate them.
    for (auto& contact : manifolds_) {
        Actor& a = actors_[contact.pair.a];
        Actor& b = actors_[contact.pair.b];
        if (a.inverse_mass + static_cast<double>(b.inverse_mass) <= 0.0)
            continue;
        const Vec3 n = contact.points[0].normal;
        const auto t = tangents(n);
        contact.normal_impulse = std::max(0.0, contact.normal_impulse);
        const double limit = static_friction * contact.normal_impulse;
        const double length = std::hypot(
            contact.tangent_impulses[0], contact.tangent_impulses[1]);
        if (length > limit) {
            const double factor = length > 0.0 ? limit / length : 0.0;
            for (double& impulse : contact.tangent_impulses)
                impulse *= factor;
        }
        apply(a, b, n, contact.normal_impulse);
        apply(a, b, t[0], contact.tangent_impulses[0]);
        apply(a, b, t[1], contact.tangent_impulses[1]);
    }

    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        for (auto& contact : manifolds_) {
            Actor& a = actors_[contact.pair.a];
            Actor& b = actors_[contact.pair.b];
            // Double arithmetic prevents finite extreme inverse masses
            // summing to an infinite float response.
            const double response =
                static_cast<double>(a.inverse_mass) + b.inverse_mass;
            if (response <= 0.0) continue;
            const Vec3 n = contact.points[0].normal;
            const auto t = tangents(n);
            const Vec3 relative{
                a.velocity.x - b.velocity.x,
                a.velocity.y - b.velocity.y,
                a.velocity.z - b.velocity.z};
            const double candidate = contact.normal_impulse +
                (static_cast<double>(contact.velocity_target) -
                 dot(relative, n)) / response;
            const double next = std::max(0.0, candidate);
            const double normal_delta = next - contact.normal_impulse;
            contact.normal_impulse = next;
            apply(a, b, n, normal_delta);

            // Friction is bounded by the current normal force. Project
            // both tangent impulses together (Coulomb disk), avoiding
            // artificial axis-dependent friction at diagonal velocities.
            const Vec3 post_normal{
                a.velocity.x - b.velocity.x,
                a.velocity.y - b.velocity.y,
                a.velocity.z - b.velocity.z};
            std::array<double, 2> desired{
                contact.tangent_impulses[0] - dot(post_normal, t[0]) / response,
                contact.tangent_impulses[1] - dot(post_normal, t[1]) / response};
            const double length = std::hypot(desired[0], desired[1]);
            const double max_static = static_friction * next;
            if (length > max_static && length > 0.0) {
                const double scale = dynamic_friction * next / length;
                desired[0] *= scale;
                desired[1] *= scale;
            }
            for (unsigned k = 0; k < 2; ++k) {
                const double delta = desired[k] - contact.tangent_impulses[k];
                contact.tangent_impulses[k] = desired[k];
                apply(a, b, t[k], delta);
            }
        }
    }
}
void Scene::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");
    constexpr double gravity = -9.81;
    for (auto& actor : actors_) {
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        const double vy = static_cast<double>(actor.velocity.y) + gravity * seconds;
        const Vec3 p{
            static_cast<float>(static_cast<double>(actor.position.x) +
                               actor.velocity.x * static_cast<double>(seconds)),
            static_cast<float>(static_cast<double>(actor.position.y) +
                               vy * static_cast<double>(seconds)),
            static_cast<float>(static_cast<double>(actor.position.z) +
                               actor.velocity.z * static_cast<double>(seconds))};
        if (!finite(p) || !std::isfinite(vy) ||
            !std::isfinite(static_cast<float>(vy)))
            throw std::overflow_error("Native CPU velocity or position overflow");
        actor.velocity.y = static_cast<float>(vy);
        actor.position = p;
        dirty_ = true;
    }
    // One owned frame progression; no second task graph, host PxScene,
    // per-contact heap allocation or GPU synchronization.
    advance_collision();
    prepare_constraints(seconds);
    solve_constraints();
    previous_step_seconds_ = seconds;
}

void Scene::advance_collision(bool force_update) {
    // ScPipeline.cpp CPU ordering:
    // BP first pass -> BP update -> post-BP pair publication ->
    // Gu narrowphase -> physical touch publication.
    // No parallel timestep or dynamics scheduler is introduced.
    if (dirty_ || force_update) {
        update_bounds();
        update_pairs();
        publish_pairs();
        update_contacts();
    } else {
        found_.clear();
        lost_.clear();
        found_contacts_.clear();
        lost_contacts_.clear();
    }
    dirty_ = false;
    ++frame_index_;
}
} // namespace fyzz::native

namespace fyzz {
namespace {
class NativeCpuWorld final : public WorldBackend {
public:
    BodyId add_static_box(Vec3 h, Vec3 p) override {
        return scene_.add_static_box(h, p);
    }
    BodyId add_static_floor(float height) override {
        return scene_.add_static_floor(height);
    }
    BodyId add_dynamic_box(Vec3 h, Vec3 p, float density) override {
        return scene_.add_dynamic_box(h, p, density);
    }
    void step(float seconds) override { scene_.step(seconds); }
    BodyState body_state(BodyId id) const override { return scene_.body_state(id); }
    void set_body_position(BodyId id, Vec3 p) override {
        scene_.set_position(id, p);
    }
    void remove_body(BodyId id) override { scene_.remove(id); }
    void set_body_velocity(BodyId id, Vec3 v) override {
        scene_.set_velocity(id, v);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend() {
    return std::make_unique<NativeCpuWorld>();
}
} // namespace fyzz
