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
// Compute relative velocity in double BEFORE subtraction. Two finite
// components may have an unrepresentable difference in float, and that
// difference must not poison a contact Jacobian or its impulse.
double normal_relative_velocity(Vec3 a, Vec3 b, Vec3 n) {
    return (static_cast<double>(a.x) - b.x) * n.x +
           (static_cast<double>(a.y) - b.y) * n.y +
           (static_cast<double>(a.z) - b.z) * n.z;
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
}
void Scene::set_velocity(BodyId id, Vec3 velocity) {
    if (!finite(velocity))
        throw std::invalid_argument("Native scene velocity must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors have no linear velocity");
    actor.velocity = velocity;
    // This base solver has no inter-frame constraint cache to invalidate.
    // No contact scan or broadphase rebuild is needed for velocity edits.
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
        const double relative_velocity =
            normal_relative_velocity(a.velocity, b.velocity, n);
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
            manifold.velocity_target = static_cast<float>(std::max(
                static_cast<double>(manifold.velocity_target),
                -restitution * relative_velocity));
        manifold.normal_impulse = 0.0f;
    }
}
void Scene::solve_constraints() {
    // DySolverConstraints::solveContact restricted to zero angular
    // response. All vertices in a face patch share a translational normal
    // Jacobian: do not apply the same rigid impulse four/eight times.
    constexpr unsigned position_iterations = 4;
    constexpr unsigned velocity_iterations = 2;
    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        for (auto& manifold : manifolds_) {
            Actor& a = actors_[manifold.pair.a];
            Actor& b = actors_[manifold.pair.b];
            // Two individually finite inverse masses can overflow when
            // added in float (tiny but otherwise admitted dynamic boxes).
            // Accumulate the scalar response/impulse in double; retain
            // float storage and SIMD-friendly vector state for bodies.
            const double response =
                static_cast<double>(a.inverse_mass) + b.inverse_mass;
            if (response <= 0.0) continue;
            const Vec3 n = manifold.points[0].normal;
            const double normal_velocity =
                normal_relative_velocity(a.velocity, b.velocity, n);
            const double candidate = manifold.normal_impulse +
                (static_cast<double>(manifold.velocity_target) -
                 normal_velocity) / response;
            const double next = std::max(0.0, candidate);
            const double impulse = next - manifold.normal_impulse;
            manifold.normal_impulse = next;
            const double da = impulse * a.inverse_mass;
            const double db = impulse * b.inverse_mass;
            // An admissible pair can have a finite wide impulse but an
            // unrepresentable float velocity update. Validate all six
            // prospective components BEFORE committing either actor,
            // instead of publishing infinities or NaNs. Keep this in the
            // existing solver, with no extra buffer or per-pair allocation.
            const std::array<double, 6> after{
                static_cast<double>(a.velocity.x) + n.x * da,
                static_cast<double>(a.velocity.y) + n.y * da,
                static_cast<double>(a.velocity.z) + n.z * da,
                static_cast<double>(b.velocity.x) - n.x * db,
                static_cast<double>(b.velocity.y) - n.y * db,
                static_cast<double>(b.velocity.z) - n.z * db};
            constexpr double max_float =
                static_cast<double>(std::numeric_limits<float>::max());
            for (double v : after) {
                if (!std::isfinite(v) || std::abs(v) > max_float)
                    throw std::overflow_error(
                        "Native CPU contact impulse exceeds float velocity range");
            }
            a.velocity = {static_cast<float>(after[0]),
                          static_cast<float>(after[1]),
                          static_cast<float>(after[2])};
            b.velocity = {static_cast<float>(after[3]),
                          static_cast<float>(after[4]),
                          static_cast<float>(after[5])};
        }
    }
}
void Scene::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");
    constexpr double gravity = -9.81;
    // PhysX source: ScBodyCore.cpp dynamic default maxLinearVelocitySq
    // is 1e32f; DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity
    // applies gravity/damping then clamps unconstrained linear speed.
    // The present scene has zero linear damping, just like PhysX dynamic
    // bodies by default. Widen the norm calculation so finite float inputs
    // cannot overflow their squared length before the source clamp.
    constexpr double max_linear_velocity_squared = 1.0e32;
    for (auto& actor : actors_) {
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        double vx = actor.velocity.x;
        double vy = static_cast<double>(actor.velocity.y) + gravity * seconds;
        double vz = actor.velocity.z;
        const double speed_squared = vx * vx + vy * vy + vz * vz;
        if (speed_squared > max_linear_velocity_squared) {
            const double factor =
                std::sqrt(max_linear_velocity_squared / speed_squared);
            vx *= factor;
            vy *= factor;
            vz *= factor;
        }
        const Vec3 velocity{
            static_cast<float>(vx), static_cast<float>(vy),
            static_cast<float>(vz)};
        const Vec3 p{
            static_cast<float>(static_cast<double>(actor.position.x) +
                               vx * seconds),
            static_cast<float>(static_cast<double>(actor.position.y) +
                               vy * seconds),
            static_cast<float>(static_cast<double>(actor.position.z) +
                               vz * seconds)};
        if (!finite(velocity) || !finite(p))
            throw std::overflow_error("Native CPU velocity or position overflow");
        actor.velocity = velocity;
        actor.position = p;
        dirty_ = true;
    }
    // One owned frame progression; no second task graph, host PxScene,
    // per-contact heap allocation or GPU synchronization.
    advance_collision();
    prepare_constraints(seconds);
    solve_constraints();
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
    void set_body_velocity(BodyId id, Vec3 velocity) override {
        scene_.set_velocity(id, velocity);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend() {
    return std::make_unique<NativeCpuWorld>();
}
} // namespace fyzz
