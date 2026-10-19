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
bool finite(Quat q) {
    return std::isfinite(q.x) && std::isfinite(q.y) &&
           std::isfinite(q.z) && std::isfinite(q.w);
}
// Equivalent to the original PxQuat::rotate and rotateInv using an
// orthonormal body pose; inertia is transformed into world space by
// R * diag(inverseInertia) * R^T, as in Cm::transformInertiaTensor.
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x};
}
Vec3 rotate(Quat q, Vec3 v) {
    const Vec3 u{q.x,q.y,q.z};
    const Vec3 c = cross(u,v);
    const Vec3 t{c.x*2.0f,c.y*2.0f,c.z*2.0f};
    const Vec3 d = cross(u,t);
    return {v.x + q.w*t.x + d.x,
            v.y + q.w*t.y + d.y,
            v.z + q.w*t.z + d.z};
}
Vec3 rotate_inverse(Quat q, Vec3 v) {
    return rotate({-q.x,-q.y,-q.z,q.w},v);
}
Quat multiply(Quat a, Quat b) {
    // Source PxQuat::operator*: (x,y,z,w) Hamilton product.
    return {
        a.w*b.x + b.w*a.x + a.y*b.z - a.z*b.y,
        a.w*b.y + b.w*a.y + a.z*b.x - a.x*b.z,
        a.w*b.z + b.w*a.z + a.x*b.y - a.y*b.x,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    };
}
Quat integrate_quaternion(Quat pose, Vec3 angular, float dt) {
    // Direct translation of DyBodyCoreIntegrator.h::integrateCore's
    // closed-form post-solver quaternion step for zero solver angular
    // delta. Rotational contact response remains unavailable to the provisional PGS.
    float w = angular.x*angular.x + angular.y*angular.y +
              angular.z*angular.z;
    if (w != 0.0f) {
        w = std::sqrt(w);
        constexpr float max_w = 1.0e7f;
        if (w > max_w) {
            const float scale = max_w / w;
            angular = {angular.x*scale,angular.y*scale,angular.z*scale};
            w = max_w;
        }
        const float v = dt * w * 0.5f;
        const float s = std::sin(v) / w;
        const float cosine = std::cos(v);
        const Quat dq = multiply({
            angular.x*s,angular.y*s,angular.z*s,0.0f},pose);
        const Quat next{
            dq.x + pose.x*cosine, dq.y + pose.y*cosine,
            dq.z + pose.z*cosine, dq.w + pose.w*cosine};
        const float len = std::sqrt(next.x*next.x + next.y*next.y +
                                    next.z*next.z + next.w*next.w);
        if (!(len > 0.0f) || !std::isfinite(len))
            throw std::overflow_error("Native orientation normalization failed");
        pose = {next.x/len,next.y/len,next.z/len,next.w/len};
    }
    return pose;
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
    // Original PhysX PxMassProperties(PxBoxGeometry) computes a
    // density-one mass of 8*hx*hy*hz and a principal-axis inertia
    // Ixx = mass*(hy^2+hz^2)/3 (and cyclic permutations). The rigid
    // body extension then scales both mass and inertia by density.
    // Keep the source's float products, rather than replacing
    // inertial properties with a made-up size-independent response.
    const float unit_mass = (h.x * h.y * h.z) * 8.0f;
    const float tensor_scale = (unit_mass * (1.0f/3.0f)) * density;
    const Vec3 inertia{
        tensor_scale * (h.y*h.y + h.z*h.z),
        tensor_scale * (h.x*h.x + h.z*h.z),
        tensor_scale * (h.x*h.x + h.y*h.y)};
    if (!finite(inertia) || inertia.x <= 0.0f ||
        inertia.y <= 0.0f || inertia.z <= 0.0f)
        throw std::invalid_argument("Native box has nonrepresentable PhysX inertia");
    const Vec3 inverse_inertia{
        1.0f/inertia.x, 1.0f/inertia.y, 1.0f/inertia.z};
    a.angular_inertia_available =
        finite(inverse_inertia) &&
        inverse_inertia.x > 0.0f && inverse_inertia.y > 0.0f &&
        inverse_inertia.z > 0.0f;
    if (a.angular_inertia_available)
        a.inverse_inertia_local = inverse_inertia;
    // Keep the pre-existing tiny-mass TRANSLATION stress case admitted.
    // It is not an angular-body admission and must not silently run
    // with zero/infinite substitute inertia in the actual source solver.
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
void Scene::set_angular_velocity(BodyId id, Vec3 velocity) {
    if (!finite(velocity))
        throw std::invalid_argument("Native angular velocity must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors have no angular velocity");
    if (!actor.angular_inertia_available &&
        (velocity.x != 0.0f || velocity.y != 0.0f || velocity.z != 0.0f))
        throw std::logic_error(
            "Angular inertia is not representable in PhysX float body state");
    actor.angular_velocity = velocity;
}
void Scene::apply_torque_impulse(BodyId id, Vec3 impulse) {
    if (!finite(impulse))
        throw std::invalid_argument("Native torque impulse must be finite");
    auto& actor = require(id);
    if (actor.kind != Kind::DynamicBox)
        throw std::logic_error("Static actors cannot receive torque impulses");
    if (!actor.angular_inertia_available)
        throw std::logic_error(
            "Angular impulse requires representable PhysX inverse inertia");

    // NpRigidBodyTemplate::setSpatialForce(eIMPULSE):
    // world angular delta = global inverse inertia tensor * torque.
    // The equivalent local principal-axes transform avoids an extra
    // matrix allocation: R * diag(invI) * R^T * worldImpulse.
    const Vec3 local = rotate_inverse(actor.orientation, impulse);
    const Vec3 delta = rotate(actor.orientation, {
        local.x * actor.inverse_inertia_local.x,
        local.y * actor.inverse_inertia_local.y,
        local.z * actor.inverse_inertia_local.z});
    const Vec3 pending = {
        actor.pending_angular_velocity_delta.x + delta.x,
        actor.pending_angular_velocity_delta.y + delta.y,
        actor.pending_angular_velocity_delta.z + delta.z};
    if (!finite(delta) || !finite(pending))
        throw std::overflow_error("Native world-space torque impulse overflow");
    actor.pending_angular_velocity_delta = pending;
}
void Scene::remove(BodyId id) {
    require(id).alive = false;
    dirty_ = true;
}
BodyState Scene::body_state(BodyId id) const {
    const auto& actor = require(id);
    return {actor.position, actor.velocity,
            actor.orientation, actor.angular_velocity};
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
        // Source: GuBounds.cpp::computeBounds(eBOX), basisExtentV.
        // The world AABB of a rotated box is the sum of absolute
        // world-space basis columns weighted by the half extents,
        // plus its shape contact offset. The identity fast path
        // avoids three quaternion rotations for common unrotated boxes.
        Vec3 extent = h;
        const Quat q = actor.orientation;
        if (q.x != 0.0f || q.y != 0.0f ||
            q.z != 0.0f || q.w != 1.0f) {
            const Vec3 x = rotate(q, {1.0f,0.0f,0.0f});
            const Vec3 y = rotate(q, {0.0f,1.0f,0.0f});
            const Vec3 z = rotate(q, {0.0f,0.0f,1.0f});
            extent = {
                std::abs(x.x)*h.x + std::abs(y.x)*h.y + std::abs(z.x)*h.z,
                std::abs(x.y)*h.x + std::abs(y.y)*h.y + std::abs(z.y)*h.z,
                std::abs(x.z)*h.x + std::abs(y.z)*h.y + std::abs(z.z)*h.z
            };
        }
        extent.x += contact_offset;
        extent.y += contact_offset;
        extent.z += contact_offset;
        const std::array<float, 6> e{
            p.x-extent.x, p.y-extent.y, p.z-extent.z,
            p.x+extent.x, p.y+extent.y, p.z+extent.z};
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
    next_contact_points_.clear();
    std::size_t old_cursor = 0;
    for (const Pair pair : previous_pairs_) {
        // The sorted source contact-manager stream retains its collision
        // axis across successive physical-touch frames.
        while (old_cursor < manifolds_.size() &&
               manifolds_[old_cursor].pair < pair)
            ++old_cursor;
        const Actor& a = actors_[pair.a];
        const Actor& b = actors_[pair.b];
        ContactManifold manifold{};
        manifold.pair = pair;
        manifold.contact_offset = next_contact_points_.size();
        if (old_cursor < manifolds_.size() &&
            manifolds_[old_cursor].pair == pair)
            manifold.collision_axis = manifolds_[old_cursor].collision_axis;

        if (a.kind == Kind::StaticPlane || b.kind == Kind::StaticPlane) {
            // Direct source order/geometry from
            // GuContactPlaneBox.cpp::contactPlaneBox. The SDK plane's
            // logical normal is +Y, corresponding to PhysX plane-local
            // +X. This computes boxToPlane.m.column[k].x times each
            // half-extent, constructs the eight source depth tests, and
            // publishes world-space rotated vertices. Do not replace
            // this with identity-only AABB corners.
            const Actor& plane = a.kind == Kind::StaticPlane ? a : b;
            const Actor& box = a.kind == Kind::StaticPlane ? b : a;
            if (box.kind != Kind::DynamicBox)
                continue;
            const Vec3 normal = a.kind == Kind::StaticPlane
                ? Vec3{0.0f, -1.0f, 0.0f}
                : Vec3{0.0f, 1.0f, 0.0f};
            // Identity boxes dominate the admitted floor-contact path.
            // Preserve the source-derived rotated basis only when needed;
            // this avoids three quaternion rotations per plane/box pair.
            Vec3 x{box.half_extents.x, 0.0f, 0.0f};
            Vec3 y{0.0f, box.half_extents.y, 0.0f};
            Vec3 z{0.0f, 0.0f, box.half_extents.z};
            const Quat q = box.orientation;
            if (q.x != 0.0f || q.y != 0.0f ||
                q.z != 0.0f || q.w != 1.0f) {
                const Vec3 basis_x = rotate(q, {1.0f,0.0f,0.0f});
                const Vec3 basis_y = rotate(q, {0.0f,1.0f,0.0f});
                const Vec3 basis_z = rotate(q, {0.0f,0.0f,1.0f});
                x = {basis_x.x * box.half_extents.x,
                     basis_x.y * box.half_extents.x,
                     basis_x.z * box.half_extents.x};
                y = {basis_y.x * box.half_extents.y,
                     basis_y.y * box.half_extents.y,
                     basis_y.z * box.half_extents.y};
                z = {basis_z.x * box.half_extents.z,
                     basis_z.y * box.half_extents.z,
                     basis_z.z * box.half_extents.z};
            }
            const float limit = contact_distance -
                (box.position.y - plane.position.y);
            const float depths[8]{
                x.y + y.y + z.y - limit,
                x.y + y.y - z.y - limit,
                x.y - y.y + z.y - limit,
                x.y - y.y - z.y - limit,
               -x.y + y.y + z.y - limit,
               -x.y + y.y - z.y - limit,
               -x.y - y.y + z.y - limit,
               -x.y - y.y - z.y - limit
            };
            for (unsigned corner = 0; corner < 8; ++corner) {
                if (depths[corner] >= 0.0f) continue;
                const float sx = (corner & 4u) ? -1.0f : 1.0f;
                const float sy = (corner & 2u) ? -1.0f : 1.0f;
                const float sz = (corner & 1u) ? -1.0f : 1.0f;
                const Vec3 point{
                    box.position.x + sx*x.x + sy*y.x + sz*z.x,
                    box.position.y + sx*x.y + sy*y.y + sz*z.y,
                    box.position.z + sx*x.z + sy*y.z + sz*z.z};
                next_contact_points_.push_back({
                    point, normal, depths[corner] + contact_distance});
                ++manifold.count;
            }
        } else {
            // Original GuContactBoxBox SAT, face/edge candidate generation,
            // six-axis separation cache, and source contact ordering.
            // Its contact buffer owns up to the original 255 contacts.
            manifold.count = source_box_box_contacts(
                a.half_extents, a.position, a.orientation,
                b.half_extents, b.position, b.orientation,
                manifold.collision_axis, box_contact_scratch_);
            next_contact_points_.insert(next_contact_points_.end(),
                box_contact_scratch_.begin(),
                box_contact_scratch_.begin() + manifold.count);
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
    // Assign all pointers only after the arena has stopped growing.
    // Both old and new contact arenas swap together with their manifolds.
    for (auto& manifold : next_manifolds_)
        manifold.points = std::span<const ContactPoint>(
            next_contact_points_.data() + manifold.contact_offset,
            manifold.count);
    manifolds_.swap(next_manifolds_);
    contact_points_.swap(next_contact_points_);
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
void Scene::prepare_solver_bodies() {
    // DyDynamics::preIntegrationParallel -> Dy::copyToSolverBodyData.
    // Reserve once across frames; align the transient row with its actor ID.
    // This is the sole scene-owned solver-body state, not a second body owner.
    solver_rows_.resize(actors_.size());
    for (std::size_t i = 0; i < actors_.size(); ++i) {
        auto& row = solver_rows_[i];
        row = {};
        const Actor& actor = actors_[i];
        if (!actor.alive) continue;
        row.data.position = actor.position;
        row.data.orientation = actor.orientation;
        row.data.linear_velocity = actor.velocity;
        row.data.angular_velocity = actor.angular_velocity;
        row.data.inverse_mass = actor.inverse_mass;
        if (actor.kind != Kind::DynamicBox ||
            !actor.angular_inertia_available)
            continue;

        // CmUtils.h::transformInertiaTensor(sqrt(invInertia), pose.q).
        // The source computes a symmetric world-space R*diag(sqrt(invI))*R^T;
        // retain its six unique terms rather than reconstructing a tensor
        // independently for every contact row.
        const Vec3 local{
            std::sqrt(actor.inverse_inertia_local.x),
            std::sqrt(actor.inverse_inertia_local.y),
            std::sqrt(actor.inverse_inertia_local.z)};
        const Quat q = actor.orientation;
        if (q.x == 0.0f && q.y == 0.0f &&
            q.z == 0.0f && q.w == 1.0f) {
            row.data.sqrt_inverse_inertia = {
                Vec3{local.x, 0.0f, 0.0f},
                Vec3{0.0f, local.y, 0.0f},
                Vec3{0.0f, 0.0f, local.z}};
            continue;
        }
        const Vec3 rx = rotate(q, {1.0f, 0.0f, 0.0f});
        const Vec3 ry = rotate(q, {0.0f, 1.0f, 0.0f});
        const Vec3 rz = rotate(q, {0.0f, 0.0f, 1.0f});
        const float xx = local.x*rx.x*rx.x + local.y*ry.x*ry.x +
                         local.z*rz.x*rz.x;
        const float yy = local.x*rx.y*rx.y + local.y*ry.y*ry.y +
                         local.z*rz.y*rz.y;
        const float zz = local.x*rx.z*rx.z + local.y*ry.z*ry.z +
                         local.z*rz.z*rz.z;
        const float xy = local.x*rx.x*rx.y + local.y*ry.x*ry.y +
                         local.z*rz.x*rz.y;
        const float xz = local.x*rx.x*rx.z + local.y*ry.x*ry.z +
                         local.z*rz.x*rz.z;
        const float yz = local.x*rx.y*rx.z + local.y*ry.y*ry.z +
                         local.z*rz.y*rz.z;
        row.data.sqrt_inverse_inertia = {
            Vec3{xx,xy,xz}, Vec3{xy,yy,yz}, Vec3{xz,yz,zz}};
    }
}

void Scene::prepare_constraints(float seconds) {
    constexpr float positional_bias = 0.2f;
    constexpr float max_depenetration_speed = 3.0f;
    constexpr float bounce_threshold = 2.0f;
    constexpr float restitution = 0.2f;
    for (auto& manifold : manifolds_) {
        const Actor& a = actors_[manifold.pair.a];
        const Actor& b = actors_[manifold.pair.b];
        const Vec3 n = manifold.points[0].normal;
        // PhysX::DyContactPrep consumes the already clamped unconstrained
        // linear velocities. The source default speed limit keeps both
        // magnitudes <= 1e16; float differences cannot overflow here.
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
        manifold.normal_impulse = 0.0f;
    }
}
void Scene::solve_constraints() {
    // Transitional single-normal solver. The source pre-solver velocity is
    // immutable across iterations; update only solver-body deltas, then
    // publish them once. This removes direct actor mutation from PGS, but
    // does NOT yet implement the original per-contact Dy constraints.
    constexpr unsigned position_iterations = 4;
    constexpr unsigned velocity_iterations = 2;
    constexpr double max_float =
        static_cast<double>(std::numeric_limits<float>::max());
    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        for (auto& manifold : manifolds_) {
            auto& a = solver_rows_[manifold.pair.a];
            auto& b = solver_rows_[manifold.pair.b];
            const double response =
                static_cast<double>(a.data.inverse_mass) +
                b.data.inverse_mass;
            if (response <= 0.0) continue;
            const Vec3 n = manifold.points[0].normal;
            const Vec3 va{
                a.data.linear_velocity.x + a.solver.linear_velocity_delta.x,
                a.data.linear_velocity.y + a.solver.linear_velocity_delta.y,
                a.data.linear_velocity.z + a.solver.linear_velocity_delta.z};
            const Vec3 vb{
                b.data.linear_velocity.x + b.solver.linear_velocity_delta.x,
                b.data.linear_velocity.y + b.solver.linear_velocity_delta.y,
                b.data.linear_velocity.z + b.solver.linear_velocity_delta.z};
            const float normal_velocity =
                (va.x - vb.x)*n.x + (va.y - vb.y)*n.y +
                (va.z - vb.z)*n.z;
            const double candidate = manifold.normal_impulse +
                (static_cast<double>(manifold.velocity_target) -
                 normal_velocity) / response;
            const double next = std::max(0.0, candidate);
            const double impulse = next - manifold.normal_impulse;
            const double da = impulse * a.data.inverse_mass;
            const double db = impulse * b.data.inverse_mass;
            const std::array<double,6> after{
                static_cast<double>(va.x) + n.x*da,
                static_cast<double>(va.y) + n.y*da,
                static_cast<double>(va.z) + n.z*da,
                static_cast<double>(vb.x) - n.x*db,
                static_cast<double>(vb.y) - n.y*db,
                static_cast<double>(vb.z) - n.z*db};
            std::array<float,6> deltas{};
            for (unsigned k = 0; k < after.size(); ++k) {
                if (!std::isfinite(after[k]) || std::abs(after[k]) > max_float)
                    throw std::overflow_error(
                        "Native CPU contact impulse exceeds float velocity range");
                const Vec3 original = k < 3 ? a.data.linear_velocity
                                            : b.data.linear_velocity;
                const float baseline = k%3 == 0 ? original.x :
                                       k%3 == 1 ? original.y : original.z;
                const double delta = static_cast<double>(
                    static_cast<float>(after[k])) - baseline;
                if (!std::isfinite(delta) || std::abs(delta) > max_float)
                    throw std::overflow_error(
                        "Native solver velocity delta exceeds PhysX float range");
                deltas[k] = static_cast<float>(delta);
            }
            manifold.normal_impulse = next;
            a.solver.linear_velocity_delta = {
                deltas[0],deltas[1],deltas[2]};
            b.solver.linear_velocity_delta = {
                deltas[3],deltas[4],deltas[5]};
        }
    }
}
void Scene::publish_solver_bodies() {
    // The source retains pre-solver velocity independently and publishes
    // solved motion velocity after the iteration/conclude/writeback sequence.
    // Angular solver deltas remain zero until original Dy contact rows land.
    for (std::size_t i = 0; i < actors_.size(); ++i) {
        auto& actor = actors_[i];
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        const auto& row = solver_rows_[i];
        const Vec3& d = row.solver.linear_velocity_delta;
        const Vec3 linear{
            row.data.linear_velocity.x+d.x,
            row.data.linear_velocity.y+d.y,
            row.data.linear_velocity.z+d.z};
        if (!finite(linear))
            throw std::overflow_error("Native solver publication is not finite");
        actor.velocity = linear;
    }
}
void Scene::reject_unsupported_rotated_contacts() const {
    // Rotated Gu narrowphase is now source-ported, but the provisional
    // scalar Dy PGS cannot solve angular contact Jacobians.
    // A rotation persists when angular velocity is subsequently set to
    // zero, so checking spin alone would silently admit false contacts.
    unsigned live_colliders = 0;
    bool needs_rotated_contacts = false;
    for (const auto& actor : actors_) {
        if (!actor.alive) continue;
        ++live_colliders;
        if (actor.kind != Kind::DynamicBox) continue;
        const Quat q = actor.orientation;
        if (actor.angular_velocity.x != 0.0f ||
            actor.angular_velocity.y != 0.0f ||
            actor.angular_velocity.z != 0.0f ||
            actor.pending_angular_velocity_delta.x != 0.0f ||
            actor.pending_angular_velocity_delta.y != 0.0f ||
            actor.pending_angular_velocity_delta.z != 0.0f ||
            q.x != 0.0f || q.y != 0.0f ||
            q.z != 0.0f || q.w != 1.0f)
            needs_rotated_contacts = true;
    }
    if (needs_rotated_contacts && live_colliders > 1)
        throw std::logic_error(
            "Rotated multi-collider dynamics require the source angular PGS solver");
}
void Scene::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");

    reject_unsupported_rotated_contacts();

    // Source: ScBodyCore.cpp::BodyCore dynamic defaults, and
    // DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
    // gravity, linear/angular damping, then PxReal speed clamps.
    constexpr float gravity = -9.81f;
    constexpr float linear_damping = 0.0f;
    constexpr float angular_damping = 0.05f;
    constexpr float max_linear_velocity_squared = 1.0e32f;
    constexpr float max_angular_velocity_squared = 1.0e4f;
    for (auto& actor : actors_) {
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        float vx = actor.velocity.x;
        float vy = actor.velocity.y + gravity * seconds;
        float vz = actor.velocity.z;
        const float lin_mult = std::max(
            1.0f - linear_damping*seconds,0.0f);
        vx *= lin_mult; vy *= lin_mult; vz *= lin_mult;
        // Per-step angular velocity modification is accumulated on the
        // actor and consumed exactly once, as in ScBodySim::VelocityMod.
        Vec3 angular{
            actor.angular_velocity.x + actor.pending_angular_velocity_delta.x,
            actor.angular_velocity.y + actor.pending_angular_velocity_delta.y,
            actor.angular_velocity.z + actor.pending_angular_velocity_delta.z};
        if (!finite(angular))
            throw std::overflow_error("Native angular impulse accumulation overflow");
        const float ang_mult = std::max(
            1.0f - angular_damping*seconds,0.0f);
        angular.x *= ang_mult;
        angular.y *= ang_mult;
        angular.z *= ang_mult;
        const float speed_squared = vx*vx + vy*vy + vz*vz;
        if (speed_squared > max_linear_velocity_squared) {
            const float factor =
                std::sqrt(max_linear_velocity_squared / speed_squared);
            vx *= factor; vy *= factor; vz *= factor;
        }
        const float ang_speed_squared = angular.x*angular.x +
            angular.y*angular.y + angular.z*angular.z;
        if (ang_speed_squared > max_angular_velocity_squared) {
            const float factor =
                std::sqrt(max_angular_velocity_squared / ang_speed_squared);
            angular.x *= factor;
            angular.y *= factor;
            angular.z *= factor;
        }
        const Vec3 velocity{vx, vy, vz};
        const Vec3 p{
            static_cast<float>(static_cast<double>(actor.position.x) +
                               static_cast<double>(vx) * seconds),
            static_cast<float>(static_cast<double>(actor.position.y) +
                               static_cast<double>(vy) * seconds),
            static_cast<float>(static_cast<double>(actor.position.z) +
                               static_cast<double>(vz) * seconds)};
        if (!finite(velocity) || !finite(angular) || !finite(p))
            throw std::overflow_error("Native CPU velocity or position overflow");
        actor.velocity = velocity;
        actor.angular_velocity = angular;
        actor.pending_angular_velocity_delta = {};
        actor.position = p;
        dirty_ = true;
    }
    // The legacy source-limited translational contact path remains until
    // the integrated original Gu/Dy pipeline replaces it. This draft is
    // NOT an accepted PhysX PGS port or an N1 completion.
    // The scene was admitted before the step. Only the provisional
    // dynamics solver requires this guard; collision alone supports rotated
    // geometry. No duplicate scan is needed.
    advance_collision_internal(false);
    prepare_solver_bodies();
    prepare_constraints(seconds);
    solve_constraints();
    publish_solver_bodies();
    // Source: DyBodyCoreIntegrator.h::integrateCore angular quaternion
    // integration occurs after solver motion-velocity publication.
    for (auto& actor : actors_) {
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        const Quat prior = actor.orientation;
        actor.orientation = integrate_quaternion(
            actor.orientation, actor.angular_velocity, seconds);
        if (!finite(actor.orientation))
            throw std::overflow_error("Native CPU orientation overflow");
        if (actor.orientation.x != prior.x ||
            actor.orientation.y != prior.y ||
            actor.orientation.z != prior.z ||
            actor.orientation.w != prior.w)
            dirty_ = true;
    }
}

void Scene::advance_collision(bool force_update) {
    // Original Gu narrowphase now admits rotated collision-only scenes.
    // Simulation still guards unsupported angular solver constraints.
    advance_collision_internal(force_update);
}
void Scene::advance_collision_internal(bool force_update) {
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
    void set_body_angular_velocity(BodyId id, Vec3 velocity) override {
        scene_.set_angular_velocity(id, velocity);
    }
    void apply_body_torque_impulse(BodyId id, Vec3 impulse) override {
        scene_.apply_torque_impulse(id, impulse);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend() {
    return std::make_unique<NativeCpuWorld>();
}
} // namespace fyzz
