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
    const bool dynamic_node=actor.kind==Kind::DynamicBox;
    actors_.push_back(actor);
    dirty_ = true;
    // Static nodes belong to the shared world endpoint, not an island.
    // Their insertion changes SAP/Gu edges but not dynamic connectivity.
    island_topology_dirty_ |= dynamic_node;
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
    // Preserve the source's evaluation sequence: PxMassProperties
    // first builds a unit-density box tensor from squared extents and
    // (unit mass * 1) / 3, then scales each tensor component by density.
    // Scaling the common factor by density *before* multiplying the
    // diagonal can change float underflow/overflow and rounding.
    const float unit_mass = (h.x * h.y * h.z) * 8.0f;
    const Vec3 extent_squared{h.x*h.x, h.y*h.y, h.z*h.z};
    const float unit_scale = (unit_mass * 1.0f) / 3.0f;
    const Vec3 unit_inertia{
        (extent_squared.y + extent_squared.z) * unit_scale,
        (extent_squared.x + extent_squared.z) * unit_scale,
        (extent_squared.x + extent_squared.y) * unit_scale};
    const Vec3 inertia{
        unit_inertia.x * density,
        unit_inertia.y * density,
        unit_inertia.z * density};
    // The translational admission boundary is the representability of
    // inverse mass above, not of the inertia tensor. Valid boxes can have
    // finite inverse mass while their source PxReal inertia underflows,
    // overflows, or yields an unrepresentable inverse inertia. Retain
    // translation-only actors without inventing an angular tensor.
    if (finite(inertia) && inertia.x > 0.0f &&
        inertia.y > 0.0f && inertia.z > 0.0f) {
        const Vec3 inverse_inertia{
            1.0f/inertia.x, 1.0f/inertia.y, 1.0f/inertia.z};
        a.angular_inertia_available =
            finite(inverse_inertia) &&
            inverse_inertia.x > 0.0f && inverse_inertia.y > 0.0f &&
            inverse_inertia.z > 0.0f;
        if (a.angular_inertia_available)
            a.inverse_inertia_local = inverse_inertia;
    }
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
    if (actor.position.x==position.x &&
        actor.position.y==position.y &&
        actor.position.z==position.z)
        return;
    // Explicit pose authoring is a discontinuity, unlike integration.
    // DyFrictionCorrelation's retained local anchors must not bridge a
    // teleport, even when projected separation along the contact normal
    // remains zero. Retire only the affected manager patches; this rare
    // edit-time scan avoids additional hot-path checks or cache registries.
    std::erase_if(friction_patches_, [id](const FrictionPatch& patch) {
        return patch.pair.a==id || patch.pair.b==id;
    });
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
void Scene::set_material(BodyId id, Material material) {
    // PxMaterial / PxsMaterialData admission for ordinary rigid contacts.
    // The source compliant branch uses negative restitution and damping;
    // refuse those inputs until its spring constraint rows are integrated.
    if (!std::isfinite(material.static_friction) ||
        !std::isfinite(material.dynamic_friction) ||
        !std::isfinite(material.restitution) ||
        material.static_friction<0.0f ||
        material.dynamic_friction<0.0f ||
        material.restitution<0.0f || material.restitution>1.0f ||
        static_cast<unsigned>(material.friction_combine)>3 ||
        static_cast<unsigned>(material.restitution_combine)>3)
        throw std::invalid_argument(
            "Native noncompliant material coefficients/modes invalid");
    require(id).material=material;
}
void Scene::set_contact_report_threshold(BodyId id, float force) {
    // ScBodyCore::contactReportThreshold and
    // DySolverConstraints::writeBackContact's finite threshold gate.
    // PX_MAX_REAL is the source disabled default; negative, NaN and
    // infinity are not valid report-threshold values.
    if (!std::isfinite(force) || force < 0.0f)
        throw std::invalid_argument(
            "Native contact report threshold must be nonnegative and finite");
    auto& actor=require(id);
    if (actor.kind!=Kind::DynamicBox)
        throw std::logic_error(
            "Source contact report threshold belongs to dynamic rigid bodies");
    actor.contact_report_threshold=force;
}
void Scene::remove(BodyId id) {
    auto& actor=require(id);
    const bool dynamic_node=actor.kind==Kind::DynamicBox;
    actor.alive = false;
    dirty_ = true;
    island_topology_dirty_ |= dynamic_node;
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
    next_found_.clear();
    next_lost_.clear();
    std::set_difference(pending_pairs_.begin(), pending_pairs_.end(),
                        previous_pairs_.begin(), previous_pairs_.end(),
                        std::back_inserter(next_found_));
    std::set_difference(previous_pairs_.begin(), previous_pairs_.end(),
                        pending_pairs_.begin(), pending_pairs_.end(),
                        std::back_inserter(next_lost_));
    found_.swap(next_found_);
    lost_.swap(next_lost_);
    previous_pairs_.swap(pending_pairs_);
    ++pair_epoch_;
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
    next_found_contacts_.clear();
    next_lost_contacts_.clear();
    const auto pair_less = [](const ContactManifold& lhs,
                              const ContactManifold& rhs) {
        return lhs.pair < rhs.pair;
    };
    // Sorted broadphase candidate iteration preserves manifold order.
    auto next = next_manifolds_.begin();
    auto old = manifolds_.begin();
    while (next != next_manifolds_.end() && old != manifolds_.end()) {
        if (pair_less(*next, *old)) {
            next_found_contacts_.push_back(next->pair);
            ++next;
        } else if (pair_less(*old, *next)) {
            next_lost_contacts_.push_back(old->pair);
            ++old;
        } else {
            ++next; ++old;
        }
    }
    for (; next != next_manifolds_.end(); ++next)
        next_found_contacts_.push_back(next->pair);
    for (; old != manifolds_.end(); ++old)
        next_lost_contacts_.push_back(old->pair);
    // Assign all pointers only after the arena has stopped growing.
    // Both old and new contact arenas swap together with their manifolds.
    for (auto& manifold : next_manifolds_)
        manifold.points = std::span<const ContactPoint>(
            next_contact_points_.data() + manifold.contact_offset,
            manifold.count);
    manifolds_.swap(next_manifolds_);
    contact_points_.swap(next_contact_points_);
    found_contacts_.swap(next_found_contacts_);
    lost_contacts_.swap(next_lost_contacts_);
    ++contact_epoch_;
}
// IG::SimpleIslandManager::firstPassIslandGen/secondPassIslandGen:
// speculative edges come from the scene's published SAP candidates,
// whereas accurate edges come only from the resolved physical touches.
// In the current all-awake rigid-only admission, IslandSim's connected
// components may be maintained by a reusable union/find lowering rather
// than its incremental wake/split queues. Topology rebuild is skipped for
// unchanged candidate AND touch connectivity. Source sleeping, persistent
// island handles and the three-pass lost-edge/deactivation lifecycle are
// not represented by this lowering and are still N1 requirements.
// IslandSim::processLostEdges calls findRoute on dirty endpoints before
// splitting an island. For the admitted all-awake dynamic contact subset,
// a lost edge can only split its component if its endpoints are no longer
// connected in the *surviving* graph. Prove that property directly using
// the existing published candidate/touch pairs. Source IslandSim's
// persistent edge instances, fast routes, hop estimates and deferred
// deactivation are not yet imported; this temporary adjacency is rebuilt
// only when dynamic connections disappear, using scene-owned capacity.
bool Scene::lost_island_edges_retain_routes(bool accurate) {
    const auto& lost=accurate ? lost_contacts_ : lost_;
    const auto dynamic_edge=[this](Pair pair) {
        return actors_[pair.a].alive && actors_[pair.b].alive &&
               actors_[pair.a].kind==Kind::DynamicBox &&
               actors_[pair.b].kind==Kind::DynamicBox;
    };
    bool needs_route=false;
    for(Pair p:lost)
        if(dynamic_edge(p)) {needs_route=true;break;}
    if(!needs_route)return true;

    constexpr auto invalid=std::numeric_limits<std::size_t>::max();
    island_route_head_.assign(actors_.size(),invalid);
    island_route_next_.clear();
    island_route_other_.clear();
    const auto add_edge=[&](Pair p) {
        if(!dynamic_edge(p))return;
        for(const auto [from,to] :
            {std::pair{p.a,p.b},std::pair{p.b,p.a}}) {
            island_route_next_.push_back(island_route_head_[from]);
            island_route_other_.push_back(to);
            island_route_head_[from]=island_route_other_.size()-1;
        }
    };
    if(accurate) {
        for(const auto& manifold:manifolds_)
            add_edge(manifold.pair);
    } else {
        for(Pair p:previous_pairs_)
            add_edge(p);
    }
    island_route_seen_.assign(actors_.size(),0);
    std::uint32_t stamp=0;
    for(Pair p:lost) {
        if(!dynamic_edge(p))continue;
        if(++stamp==0) {
            std::fill(island_route_seen_.begin(),
                      island_route_seen_.end(),0);
            stamp=1;
        }
        island_route_queue_.clear();
        island_route_queue_.push_back(p.a);
        island_route_seen_[p.a]=stamp;
        bool connected=false;
        for(std::size_t cursor=0;
            cursor<island_route_queue_.size() && !connected;++cursor) {
            const BodyId from=island_route_queue_[cursor];
            for(std::size_t edge=island_route_head_[from];
                edge!=invalid;edge=island_route_next_[edge]) {
                const BodyId to=island_route_other_[edge];
                if(to==p.b) {connected=true;break;}
                if(island_route_seen_[to]==stamp)continue;
                island_route_seen_[to]=stamp;
                island_route_queue_.push_back(to);
            }
        }
        if(!connected)return false;
    }
    return true;
}

void Scene::update_island_connectivity(bool repair_lost_edges) {
    // PxsSimpleIslandManager::firstPassIslandGen and
    // secondPassIslandGenPart1 insert newly discovered edges through
    // IslandSim::processNewEdges, merging existing islands rather than
    // rebuilding their connectivity. The currently admitted all-awake
    // dynamic/contact subset can make the same incremental merges in
    // its scene-owned component arrays. Lost edges still require
    // IslandSim::processLostEdges' path/dirty-node repair, so rebuild
    // components from the surviving authoritative edges on removal.
    const auto dynamic_edge=[this](Pair pair) {
        return actors_[pair.a].kind==Kind::DynamicBox &&
               actors_[pair.b].kind==Kind::DynamicBox;
    };
    bool rebuild=island_topology_dirty_;
    bool added_speculative=false;
    bool added_accurate=false;
    // Source first/second passes admit new edges; third pass does the
    // dirty-route split repair. A lost contact must not prematurely split
    // the solver's active island before the current solve has completed.
    // Actor-node creation/retirement still uses the bounded full-rebuild
    // path until its source lifecycle has been integrated.
    if(!rebuild && repair_lost_edges)
        rebuild=!lost_island_edges_retain_routes(false) ||
                 !lost_island_edges_retain_routes(true);
    if(!rebuild && repair_lost_edges)
        return;
    if (!rebuild) {
        for (Pair pair:found_)
            if (dynamic_edge(pair)) {added_speculative=true;break;}
        for (Pair pair:found_contacts_)
            if (dynamic_edge(pair)) {added_accurate=true;break;}
        if (!added_speculative && !added_accurate)
            return;
    }

    const auto find_root=[](std::vector<BodyId>& parent,BodyId node) {
        BodyId root=node;
        while(parent[root]!=root)root=parent[root];
        while(parent[node]!=node) {
            const BodyId next=parent[node];
            parent[node]=root;
            node=next;
        }
        return root;
    };
    const auto join=[&](std::vector<BodyId>& parent,
                        std::vector<std::uint8_t>& rank,Pair pair) {
        // Neither static shape nor shared world solver body is an island
        // node: static contacts do not connect independent rigid bodies.
        if (!dynamic_edge(pair))return false;
        BodyId a=find_root(parent,pair.a);
        BodyId b=find_root(parent,pair.b);
        if(a==b)return false;
        if(rank[a]<rank[b])std::swap(a,b);
        parent[b]=a;
        if(rank[a]==rank[b])++rank[a];
        return true;
    };

    bool order_changed=rebuild;
    if (rebuild) {
        ++island_full_rebuild_count_;
        const BodyId invalid=std::numeric_limits<BodyId>::max();
        const std::size_t count=actors_.size();
        speculative_island_parent_.resize(count);
        accurate_island_parent_.resize(count);
        speculative_island_rank_.resize(count);
        accurate_island_rank_.resize(count);
        speculative_island_count_=0;
        accurate_island_count_=0;
        accurate_island_nodes_.clear();
        for(std::size_t i=0;i<count;++i) {
            const bool dynamic=actors_[i].alive &&
                actors_[i].kind==Kind::DynamicBox;
            const BodyId root=dynamic?static_cast<BodyId>(i):invalid;
            speculative_island_parent_[i]=root;
            accurate_island_parent_[i]=root;
            speculative_island_rank_[i]=0;
            accurate_island_rank_[i]=0;
            if(dynamic) {
                ++speculative_island_count_;
                ++accurate_island_count_;
                accurate_island_nodes_.push_back(root);
            }
        }
        // First pass: speculative SAP candidates. Second pass: physically
        // connected Gu contact managers. The streams are already ordered
        // and published by their sole owning scene.
        for(Pair pair:previous_pairs_)
            if(join(speculative_island_parent_,
                    speculative_island_rank_,pair))
                --speculative_island_count_;
        for(const auto& manifold:manifolds_)
            if(join(accurate_island_parent_,
                    accurate_island_rank_,manifold.pair))
                --accurate_island_count_;
    } else {
        // Source first/second pass insertion can merge islands without
        // touching unrelated nodes. Deletions and deactivation still
        // follow the rebuild path above until source lost-edge repair
        // and third-pass sleeping have been imported.
        for(Pair pair:found_)
            if(join(speculative_island_parent_,
                    speculative_island_rank_,pair))
                --speculative_island_count_;
        for(Pair pair:found_contacts_)
            if(join(accurate_island_parent_,
                    accurate_island_rank_,pair)) {
                --accurate_island_count_;
                order_changed=true;
            }
    }

    if (order_changed) {
        // Dy::DynamicsContext::iterateIslandsNodes needs contiguous nodes
        // belonging to each accurate island. Only source accurate merges
        // or complete lost-edge rebuilds can change this iteration order.
        for(BodyId id:accurate_island_nodes_)
            find_root(accurate_island_parent_,id);
        std::sort(accurate_island_nodes_.begin(),
                  accurate_island_nodes_.end(),
            [&](BodyId a,BodyId b) {
                const BodyId ra=accurate_island_parent_[a];
                const BodyId rb=accurate_island_parent_[b];
                return ra<rb || (ra==rb && a<b);
            });
        accurate_island_offsets_.clear();
        accurate_island_offsets_.push_back(0);
        for(std::size_t i=1;i<accurate_island_nodes_.size();++i)
            if(accurate_island_parent_[accurate_island_nodes_[i-1]]!=
               accurate_island_parent_[accurate_island_nodes_[i]])
                accurate_island_offsets_.push_back(i);
        if(!accurate_island_nodes_.empty())
            accurate_island_offsets_.push_back(accurate_island_nodes_.size());
    }
    island_topology_dirty_=false;
    ++island_topology_revision_;
}
void Scene::third_pass_island_gen() {
    // Both original IslandSim::processLostEdges third-pass tasks depend
    // on lost-contact management and join into postThirdPassIslandGen,
    // potentially overlapping the numerical solver. This synchronous
    // source-subset execution is scheduled once those collision products
    // are ready, *not* as a mandatory post-solver serial stage.
    // Persistent edge handles, sleep/deactivation and their continuations
    // remain N1 admission work.
    update_island_connectivity(true);
}
// Active native CPU stages remain within the one Scene owner:
//   DyBodyCoreIntegrator: unconstrained velocity/force preparation
//   DyConstraintPartition: ordered rigid-body contact manager batches
//   DyContactPrep: point-normal and correlated friction-anchor constraints
//   DySolverConstraints/DySolverControl: four position + one velocity pass
//   DyBodyCoreIntegrator: pose integration from saved position-phase motion
// The all-awake rigid island-node order is active here. Incremental island
// retirement, sleep/wake, task joins, joints, CCD, and exact packed ABI
// remain separate N1 requirements.
void Scene::prepare_solver_bodies() {
    // DyDynamics::preIntegrationParallel -> Dy::copyToSolverBodyData.
    // Retain capacity across frames. Prepare each contact participant once;
    // free-flight actors stay on the existing source body integrator.
    // This is the sole scene-owned transient solver state.
    solver_rows_.resize(actors_.size());
    solver_active_ids_.clear();
    world_solver_row_ = {};
    // Mark the actual contact participants using the one Gu manager stream.
    // The source's accurate island node sequence then determines solver
    // body admission, not the incidental encounter order of manifolds.
    for (const auto& manifold:manifolds_) {
        const BodyId ids[2]{manifold.pair.a,manifold.pair.b};
        for(BodyId id:ids)
            if(actors_[id].kind==Kind::DynamicBox)
                solver_rows_[id].prepared_frame=frame_index_;
    }
    // Dy::DynamicsContext::iterateIslandsNodes hands contiguous accurate
    // island nodes to preIntegrationParallel. Free-flight bodies retain the
    // source no-constraint integration path without preparing unused rows.
    for(std::size_t island=0;island+1<accurate_island_offsets_.size();
        ++island) {
        for(std::size_t index=accurate_island_offsets_[island];
            index<accurate_island_offsets_[island+1];++index) {
            const BodyId id=accurate_island_nodes_[index];
            auto& row=solver_rows_[id];
            if(row.prepared_frame!=frame_index_)continue;
            row={};
            row.prepared_frame=frame_index_;
            solver_active_ids_.push_back(id);
            const Actor& actor=actors_[id];
            row.data.position=actor.position;
            row.data.orientation=actor.orientation;
            row.data.linear_velocity=actor.velocity;
            row.data.angular_velocity=actor.angular_velocity;
            row.data.inverse_mass=actor.inverse_mass;
            if(!actor.angular_inertia_available)continue;

            // CmUtils.h::transformInertiaTensor(sqrt(invInertia), pose.q).
            // Reuse the six unique terms of the symmetric world tensor.
            const Vec3 local{
                std::sqrt(actor.inverse_inertia_local.x),
                std::sqrt(actor.inverse_inertia_local.y),
                std::sqrt(actor.inverse_inertia_local.z)};
            const Quat q=actor.orientation;
            if(q.x==0.0f && q.y==0.0f && q.z==0.0f && q.w==1.0f) {
                row.data.sqrt_inverse_inertia={
                    Vec3{local.x,0.0f,0.0f},Vec3{0.0f,local.y,0.0f},
                    Vec3{0.0f,0.0f,local.z}};
                continue;
            }
            const Vec3 rx=rotate(q,{1.0f,0.0f,0.0f});
            const Vec3 ry=rotate(q,{0.0f,1.0f,0.0f});
            const Vec3 rz=rotate(q,{0.0f,0.0f,1.0f});
            const float xx=local.x*rx.x*rx.x + local.y*ry.x*ry.x +
                           local.z*rz.x*rz.x;
            const float yy=local.x*rx.y*rx.y + local.y*ry.y*ry.y +
                           local.z*rz.y*rz.y;
            const float zz=local.x*rx.z*rx.z + local.y*ry.z*ry.z +
                           local.z*rz.z*rz.z;
            const float xy=local.x*rx.x*rx.y + local.y*ry.x*ry.y +
                           local.z*rz.x*rz.y;
            const float xz=local.x*rx.x*rx.z + local.y*ry.x*ry.z +
                           local.z*rz.x*rz.z;
            const float yz=local.x*rx.y*rx.z + local.y*ry.y*ry.z +
                           local.z*rz.y*rz.z;
            row.data.sqrt_inverse_inertia={
                Vec3{xx,xy,xz},Vec3{xy,yy,yz},Vec3{xz,yz,zz}};
        }
    }
}

void Scene::partition_constraints() {
    // DyConstraintPartition.cpp::partitionContactConstraints, rigid-body
    // classification, and the source's static-contact reservation rule.
    // Contact managers are indivisible descriptors; their per-point normal
    // and friction rows execute together in the original owner. This
    // serial dispatch does not invent parallel task execution or islands.
    //
    // The source assigns conflict-free dynamic/dynamic constraints from
    // successive 32-bit partition masks; overflow starts a new wave.
    // A dynamic/static contact is then placed after that dynamic body's
    // last dynamic/dynamic partition, one static edge per partition.
    constexpr std::size_t partition_bits = 32;
    constexpr std::size_t unassigned = SIZE_MAX;
    constraint_partitions_.assign(manifolds_.size(), unassigned);
    ordered_constraints_.clear();
    partition_ends_.clear();
    partition_cursors_.clear();
    // DyConstraintPartition.cpp::classifyConstraintDesc queues only the
    // unassigned dynamic descriptors for each overflow wave; compact the
    // same reusable descriptor buffer in place to avoid an O(waves*all
    // manifolds) rescan for high-degree constraint graphs.
    pending_dynamic_constraints_.clear();
    for (std::size_t i=0;i<manifolds_.size();++i) {
        const Pair pair=manifolds_[i].pair;
        if (actors_[pair.a].kind == Kind::DynamicBox &&
            actors_[pair.b].kind == Kind::DynamicBox)
            pending_dynamic_constraints_.push_back(i);
    }

    for (std::size_t wave=0;!pending_dynamic_constraints_.empty();
         wave+=partition_bits) {
        // Source 32-partition rollover; reset only the transient progress
        // fields on already-prepared dynamic solver bodies.
        for (BodyId id : solver_active_ids_)
            solver_rows_[id].solver.partition_mask=0;
        std::size_t overflow_count=0;
        for (std::size_t k=0;k<pending_dynamic_constraints_.size();++k) {
            const std::size_t i=pending_dynamic_constraints_[k];
            const Pair pair=manifolds_[i].pair;
            auto& a=solver_rows_[pair.a].solver;
            auto& b=solver_rows_[pair.b].solver;
            const std::uint32_t available=~(a.partition_mask |
                                             b.partition_mask);
            if (!available) {
                pending_dynamic_constraints_[overflow_count++]=i;
                continue;
            }
            const unsigned bit=std::countr_zero(available);
            const std::uint32_t mask=std::uint32_t{1}<<bit;
            a.partition_mask |= mask;
            b.partition_mask |= mask;
            const std::size_t partition=wave+bit;
            a.max_dynamic_partition=std::max(
                a.max_dynamic_partition,partition+1);
            b.max_dynamic_partition=std::max(
                b.max_dynamic_partition,partition+1);
            constraint_partitions_[i]=partition;
        }
        if (overflow_count==pending_dynamic_constraints_.size())
            throw std::logic_error(
                "Native source constraint partition made no progress");
        pending_dynamic_constraints_.resize(overflow_count);
        if (overflow_count &&
            wave>std::numeric_limits<std::size_t>::max()-partition_bits)
            throw std::overflow_error(
                "Native source constraint partition index overflow");
    }

    for (std::size_t i=0;i<manifolds_.size();++i) {
        if (constraint_partitions_[i]!=unassigned) continue;
        const Pair pair=manifolds_[i].pair;
        const bool dynamic_a=actors_[pair.a].kind==Kind::DynamicBox;
        const bool dynamic_b=actors_[pair.b].kind==Kind::DynamicBox;
        // Source static/static contact managers are not solver edges.
        if (!dynamic_a && !dynamic_b) continue;
        auto& active=solver_rows_[dynamic_a ? pair.a : pair.b].solver;
        const std::size_t partition=active.max_dynamic_partition+
                                      active.static_constraint_count++;
        if (partition<active.max_dynamic_partition)
            throw std::overflow_error(
                "Native source static constraint partition overflow");
        constraint_partitions_[i]=partition;
    }

    // The source accumulates counts into ordered descriptor offsets, then
    // stable-writes each descriptor by partition. This preserves the
    // input contact-manager order *within* each non-conflicting batch.
    std::size_t count=0;
    for (std::size_t p : constraint_partitions_) {
        if (p==unassigned) continue;
        if (p>=partition_ends_.size()) partition_ends_.resize(p+1,0);
        ++partition_ends_[p];
        ++count;
    }
    std::size_t cumulative=0;
    for (auto& end : partition_ends_) {
        cumulative+=end;
        end=cumulative;
    }
    partition_cursors_.resize(partition_ends_.size());
    for (std::size_t p=0;p<partition_cursors_.size();++p)
        partition_cursors_[p]=p ? partition_ends_[p-1] : 0;
    ordered_constraints_.resize(count);
    for (std::size_t i=0;i<constraint_partitions_.size();++i) {
        const std::size_t partition=constraint_partitions_[i];
        if (partition!=unassigned)
            ordered_constraints_[partition_cursors_[partition]++]=i;
    }
}

void Scene::prepare_constraints(float seconds) {
    // DyContactPrepShared.h::constructContactConstraint, rigid-body
    // noncompliant branch. Every Gu contact is a distinct constraint row.
    // Contact/friction patch correlation is not yet integrated; this
    // stream retains the original contact ordering and no shadow solver.
    // DyCpuGpuBiasCoefficient.h::BiasCoefficientCollection::set(false,
    // false, ...): rigidContact = 0.8f for the CPU PGS solver. The same
    // source coefficient is used by DyContactPrep for penetration bias
    // and for friction-row impulse relaxation (not a 0.2 substitute).
    constexpr float rigid_contact_bias = 0.8f;
    constexpr float max_penetration_bias = -3.0f;
    constexpr float bounce_threshold = -2.0f;
    // PxsMaterialCombiner.h::PxsCombineMaterials: the source takes the
    // higher precedence combine mode (Average, Min, Multiply, Max) of
    // both shapes and promotes static friction to at least dynamic.
    // All currently admitted materials are noncompliant (restitution >=0).
    const auto combine=[](float a,float b,MaterialCombineMode mode) {
        switch(mode) {
        case MaterialCombineMode::Average: return 0.5f*(a+b);
        case MaterialCombineMode::Minimum: return std::min(a,b);
        case MaterialCombineMode::Multiply: return a*b;
        case MaterialCombineMode::Maximum: return std::max(a,b);
        }
        throw std::logic_error("Unrecognized source material combine mode");
    };
    const auto higher=[](MaterialCombineMode a,MaterialCombineMode b) {
        return static_cast<unsigned>(a)>static_cast<unsigned>(b)?a:b;
    };
    const float inv_dt = 1.0f / seconds;
    const auto dot = [](Vec3 u, Vec3 v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    };
    const auto sub = [](Vec3 u, Vec3 v) {
        return Vec3{u.x-v.x,u.y-v.y,u.z-v.z};
    };
    const auto transform = [](const SolverBodyData& d, Vec3 v) {
        const auto& c = d.sqrt_inverse_inertia;
        return Vec3{c[0].x*v.x+c[1].x*v.y+c[2].x*v.z,
                    c[0].y*v.x+c[1].y*v.y+c[2].y*v.z,
                    c[0].z*v.x+c[1].z*v.y+c[2].z*v.z};
    };
    solver_contact_points_.clear();
    solver_friction_rows_.clear();
    next_friction_patches_.clear();
    // PXC_SAME_NORMAL and DyCorrelationBuffer::MAX_FRICTION_PATCHES.
    constexpr float same_normal = 0.999f;
    constexpr float friction_offset_threshold = 0.04f;
    constexpr float correlation_distance = 0.025f;
    const auto local_point = [](const Actor& actor, Vec3 world) {
        const Quat& q = actor.orientation;
        return rotate(Quat{-q.x,-q.y,-q.z,q.w},
            {world.x-actor.position.x,
             world.y-actor.position.y,
             world.z-actor.position.z});
    };
    const auto local_normal = [](const Actor& actor, Vec3 world_normal) {
        const Quat& q = actor.orientation;
        return rotate(Quat{-q.x,-q.y,-q.z,q.w}, world_normal);
    };
    const auto world_point = [](const Actor& actor, Vec3 local) {
        Vec3 r = rotate(actor.orientation,local);
        return Vec3{r.x+actor.position.x,
                    r.y+actor.position.y,
                    r.z+actor.position.z};
    };
    const auto squared_length = [](Vec3 v) {
        return v.x*v.x + v.y*v.y + v.z*v.z;
    };
    std::size_t old_patch_cursor=0;
    for (auto& manifold : manifolds_) {
        // Patches follow sorted contact managers. One monotonic cursor
        // prevents a full previous-frame scan for every contact manager.
        while (old_patch_cursor<friction_patches_.size() &&
               friction_patches_[old_patch_cursor].pair < manifold.pair)
            ++old_patch_cursor;
        const auto& a = actors_[manifold.pair.a].kind == Kind::DynamicBox
            ? solver_rows_[manifold.pair.a].data : world_solver_row_.data;
        const auto& b = actors_[manifold.pair.b].kind == Kind::DynamicBox
            ? solver_rows_[manifold.pair.b].data : world_solver_row_.data;
        manifold.solver_offset = solver_contact_points_.size();
        manifold.friction_patch_offset = next_friction_patches_.size();
        manifold.friction_patch_count = 0;
        const Actor& actor0 = actors_[manifold.pair.a];
        const Actor& actor1 = actors_[manifold.pair.b];
        const Material& m0=actor0.material;
        const Material& m1=actor1.material;
        const auto friction_mode=higher(m0.friction_combine,
                                        m1.friction_combine);
        const auto restitution_mode=higher(m0.restitution_combine,
                                           m1.restitution_combine);
        const float restitution=combine(m0.restitution,m1.restitution,
                                        restitution_mode);
        const bool disable_friction=m0.disable_friction || m1.disable_friction;
        const bool disable_strong_friction=disable_friction ||
            m0.disable_strong_friction || m1.disable_strong_friction;
        const float dynamic_friction=disable_friction ? 0.0f :
            std::max(0.0f,combine(m0.dynamic_friction,m1.dynamic_friction,
                                  friction_mode));
        const float static_friction=disable_friction ? 0.0f :
            std::max(dynamic_friction,
                     combine(m0.static_friction,m1.static_friction,
                             friction_mode));
        if (!std::isfinite(restitution) ||
            !std::isfinite(static_friction) ||
            !std::isfinite(dynamic_friction))
            throw std::overflow_error("Native Pxs material combination overflow");
        for (const ContactPoint& contact : manifold.points) {
            // DyFrictionCorrelation::createContactPatches/correlatePatches:
            // all currently admitted contacts use the same material, so
            // identical-normal contiguous groups correlate into one
            // friction patch. Source cap remains 32 per contact manager.
            std::size_t patch_id = next_friction_patches_.size();
            for (std::size_t j=manifold.friction_patch_offset;
                 j<next_friction_patches_.size(); ++j) {
                if (dot(next_friction_patches_[j].normal,
                        contact.normal) >= same_normal) {
                    patch_id = j;
                    break;
                }
            }
            if (patch_id == next_friction_patches_.size()) {
                if (manifold.friction_patch_count == 32)
                    throw std::overflow_error(
                        "Native source friction correlation exceeds 32 patches");
                FrictionPatch patch{};
                patch.pair = manifold.pair;
                patch.normal = contact.normal;
                patch.static_friction=static_friction;
                patch.dynamic_friction=dynamic_friction;
                patch.restitution=restitution;
                patch.disable_friction=disable_friction;
                patch.disable_strong_friction=disable_strong_friction;
                // DyFrictionCorrelation::initFrictionPatch saves the normal
                // in both actor frames. These local normals are only reused
                // when the bodies still agree on the contact orientation.
                patch.body0_normal = local_normal(actor0,contact.normal);
                patch.body1_normal = local_normal(actor1,contact.normal);
                patch.bounds_min = patch.bounds_max = contact.position;
                patch.contact_ids.fill(0xffff);
                // Source getFrictionPatches keeps unbroken local anchors
                // only if their normal and separation still correlate.
                for (std::size_t previous=old_patch_cursor;
                     previous<friction_patches_.size() &&
                     friction_patches_[previous].pair == manifold.pair;
                     ++previous) {
                    const auto& old=friction_patches_[previous];
                    // DyContactPrepShared::getFrictionPatches:
                    // validate the cached normal under the *current*
                    // relative actor orientation, then correlate it with
                    // the current generated world's contact normal.
                    const Vec3 old_normal_1_world =
                        rotate(actor1.orientation,old.body1_normal);
                    const Vec3 old_normal_1_in_0 =
                        local_normal(actor0,old_normal_1_world);
                    const Vec3 old_normal_0_world =
                        rotate(actor0.orientation,old.body0_normal);
                    // DyFrictionCorrelation groups by combined material
                    // and only reuses cached anchors when strong friction
                    // is enabled and those properties remain identical.
                    if (disable_strong_friction || old.disable_strong_friction ||
                        old.static_friction!=static_friction ||
                        old.dynamic_friction!=dynamic_friction ||
                        old.restitution!=restitution ||
                        old.broken || !old.anchor_count ||
                        dot(old.body0_normal,old_normal_1_in_0) <=
                            same_normal ||
                        dot(old_normal_0_world,patch.normal) <
                            same_normal)
                        continue;
                    bool separated = false;
                    for (std::size_t k=0;k<old.anchor_count;++k) {
                        // DyContactPrepShared::pointsAreClose tests the
                        // absolute normal separation in body-0 local space.
                        // Deep penetration ALSO invalidates the anchor,
                        // contrary to the recent proposed signed shortcut.
                        const Vec3 p1_in0=local_point(actor0,
                            world_point(actor1,old.body1_anchors[k]));
                        const Vec3 between=sub(old.body0_anchors[k],p1_in0);
                        if (std::abs(dot(between,old.body0_normal)) >=
                            correlation_distance) {
                            separated=true;
                            break;
                        }
                    }
                    if (!separated) {
                        // Preserve the original local normal with its
                        // anchors. Replacing it with this frame's contact
                        // normal would forgive incremental drift and
                        // eventually extend a cache past source admission.
                        patch.body0_normal = old.body0_normal;
                        patch.body1_normal = old.body1_normal;
                        patch.body0_anchors = old.body0_anchors;
                        patch.body1_anchors = old.body1_anchors;
                        patch.anchor_count = old.anchor_count;
                        break;
                    }
                }
                next_friction_patches_.push_back(patch);
                ++manifold.friction_patch_count;
            }
            auto& patch = next_friction_patches_[patch_id];
            ++patch.contact_count;
            patch.bounds_min = {
                std::min(patch.bounds_min.x,contact.position.x),
                std::min(patch.bounds_min.y,contact.position.y),
                std::min(patch.bounds_min.z,contact.position.z)};
            patch.bounds_max = {
                std::max(patch.bounds_max.x,contact.position.x),
                std::max(patch.bounds_max.y,contact.position.y),
                std::max(patch.bounds_max.z,contact.position.z)};
            const Vec3 normal = contact.normal;
            const Vec3 ra_x_n = cross(sub(contact.position,a.position),normal);
            const Vec3 rb_x_n = cross(sub(contact.position,b.position),normal);
            const Vec3 ra = transform(a,ra_x_n);
            const Vec3 rb = transform(b,rb_x_n);
            const float linear_vrel = dot(normal,sub(a.linear_velocity,b.linear_velocity));
            const float angular_vrel =
                dot(ra_x_n,a.angular_velocity)-dot(rb_x_n,b.angular_velocity);
            const float vrel = linear_vrel+angular_vrel;
            const double response = static_cast<double>(a.inverse_mass) +
                static_cast<double>(b.inverse_mass) +
                static_cast<double>(dot(ra,ra)) + dot(rb,rb);
            const double vel_multiplier = response > 0.0
                ? 1.0 / response : 0.0;
            const float penetration = contact.separation;
            const float penetration_inv_dt = penetration*inv_dt;
            const bool separated = penetration >= 0.0f;
            const bool closing = -vrel > penetration_inv_dt;
            const bool bouncing = restitution > 0.0f &&
                bounce_threshold > vrel && closing;
            const float target_velocity = bouncing
                ? -vrel*restitution : 0.0f;
            // Source noncompliant (restitution >= 0) branch. Negative
            // penetration is bias-limited, positive speculative separation
            // retains full separation speed; restitution clears positive
            // velocity-phase bias before conclude.
            const float penetration_inv_dt_scaled = separated
                ? penetration_inv_dt
                : penetration*(inv_dt*rigid_contact_bias);
            // DyContactPrepShared::constructContactConstraint clears
            // scaledBias on a qualifying restitution bounce. The current
            // rigid subset does not admit CCD, so its CCD separation gate
            // does not exclude these contacts.
            const double scaled_bias = bouncing ? 0.0 :
                vel_multiplier *
                    std::max(max_penetration_bias,penetration_inv_dt_scaled);
            const double target = target_velocity-vrel;
            SolverContactPoint row{};
            row.normal = normal;
            row.ra_x_n_sqrt_inertia = ra;
            row.rb_x_n_sqrt_inertia = rb;
            row.friction_patch = patch_id;
            row.velocity_multiplier = vel_multiplier;
            row.biased_error = target*vel_multiplier-scaled_bias;
            row.unbiased_error = target*vel_multiplier -
                (bouncing ? 0.0 : std::max(scaled_bias,0.0));
            if (!std::isfinite(row.biased_error) ||
                !std::isfinite(row.unbiased_error) ||
                !std::isfinite(row.velocity_multiplier))
                throw std::overflow_error("Native Dy contact preparation overflow");
            solver_contact_points_.push_back(row);
        }
        // DyFrictionCorrelation::growPatches: retain up to two anchors
        // spread across each friction patch's contact bounds; the anchors
        // persist in the body-local frames, not as world-space contacts.
        for (std::size_t patch_id=manifold.friction_patch_offset;
             patch_id<next_friction_patches_.size(); ++patch_id) {
            auto& patch = next_friction_patches_[patch_id];
            // PxsCombineMaterials::eDISABLE_FRICTION produces zero
            // tangent constraints. Do not grow anchors or compute
            // local-to-world error/Jacobians for disabled patches.
            if (patch.disable_friction) {
                patch.anchor_count=0;
                patch.row_offset=solver_friction_rows_.size();
                continue;
            }
            const Vec3 extent=sub(patch.bounds_max,patch.bounds_min);
            const float diagonal_sq=squared_length(extent);
            if (patch.anchor_count == 2) {
                const Vec3 diff=sub(patch.body0_anchors[0],
                                    patch.body0_anchors[1]);
                if (squared_length(diff)*4.0f < diagonal_sq)
                    patch.anchor_count = 0;
            }
            std::array<Vec3,2> anchors{};
            std::uint8_t count = patch.anchor_count;
            if (count == 1)
                anchors[0]=world_point(actor0,patch.body0_anchors[0]);
            float point_dist_sq=0.0f;
            for (std::size_t j=0;j<manifold.count;++j) {
                const std::size_t idx=manifold.solver_offset+j;
                if (solver_contact_points_[idx].friction_patch != patch_id)
                    continue;
                const auto& point=manifold.points[j];
                if (point.separation >= friction_offset_threshold)
                    continue;
                const Vec3 w=point.position;
                if (count == 0) {
                    patch.contact_ids[0]=static_cast<std::uint16_t>(j);
                    anchors[0]=w;
                    ++count;
                } else if (count == 1) {
                    const float d=squared_length(sub(w,anchors[0]));
                    if (d>1.0e-8f) {
                        patch.contact_ids[1]=static_cast<std::uint16_t>(j);
                        anchors[1]=w;
                        point_dist_sq=d;
                        ++count;
                    }
                } else {
                    const float d0=squared_length(sub(w,anchors[0]));
                    const float d1=squared_length(sub(w,anchors[1]));
                    if (d0>d1 && d0>point_dist_sq) {
                        patch.contact_ids[1]=static_cast<std::uint16_t>(j);
                        anchors[1]=w;
                        point_dist_sq=d0;
                    } else if (d1>=d0 && d1>point_dist_sq) {
                        patch.contact_ids[0]=static_cast<std::uint16_t>(j);
                        anchors[0]=w;
                        point_dist_sq=d1;
                    }
                }
            }
            for (std::size_t k=patch.anchor_count;k<count;++k) {
                patch.body0_anchors[k]=local_point(actor0,anchors[k]);
                patch.body1_anchors[k]=local_point(actor1,anchors[k]);
            }
            patch.anchor_count=count;
            patch.row_offset=solver_friction_rows_.size();
            if (count==0) continue;
            // DyContactPrep.cpp constructs tangents from relative linear
            // velocity, with the same 0.7071 normal-axis fallback and
            // 1e-4 squared-tangent threshold as the source.
            const Vec3 n=patch.normal;
            const Vec3 rel=sub(a.linear_velocity,b.linear_velocity);
            Vec3 t0=sub(rel,{n.x*dot(rel,n),
                              n.y*dot(rel,n),n.z*dot(rel,n)});
            if (squared_length(t0)<=0.0001f) {
                t0=std::abs(n.x)<0.70710678f
                    ? Vec3{0.0f,-n.z,n.y}
                    : Vec3{-n.y,n.x,0.0f};
            }
            const float inv_l=1.0f/std::sqrt(squared_length(t0));
            t0={t0.x*inv_l,t0.y*inv_l,t0.z*inv_l};
            const Vec3 t1=cross(n,t0);
            for (std::size_t k=0;k<count;++k) {
                const Vec3 ra=rotate(actor0.orientation,patch.body0_anchors[k]);
                const Vec3 rb=rotate(actor1.orientation,patch.body1_anchors[k]);
                const Vec3 error=sub(
                    {ra.x+actor0.position.x,ra.y+actor0.position.y,
                     ra.z+actor0.position.z},
                    {rb.x+actor1.position.x,rb.y+actor1.position.y,
                     rb.z+actor1.position.z});
                for (const Vec3 tangent : {t0,t1}) {
                    const Vec3 ax0=cross(ra,tangent);
                    const Vec3 ax1=cross(rb,tangent);
                    const Vec3 j0=transform(a,ax0);
                    const Vec3 j1=transform(b,ax1);
                    const double response=static_cast<double>(a.inverse_mass)
                        +b.inverse_mass+dot(j0,j0)+dot(j1,j1);
                    SolverContactFriction fr{};
                    fr.tangent=tangent;
                    fr.ra_x_t_sqrt_inertia=j0;
                    fr.rb_x_t_sqrt_inertia=j1;
                    fr.velocity_multiplier=response>0.0
                        ? rigid_contact_bias/response : 0.0;
                    fr.bias=dot(tangent,error)*inv_dt;
                    fr.target_velocity=-(dot(tangent,rel)
                        +dot(ax0,a.angular_velocity)
                        -dot(ax1,b.angular_velocity));
                    if (!std::isfinite(fr.velocity_multiplier) ||
                        !std::isfinite(fr.bias) ||
                        !std::isfinite(fr.target_velocity))
                        throw std::overflow_error(
                            "Native source friction row preparation overflow");
                    solver_friction_rows_.push_back(fr);
                }
            }
        }
    }
    // Swap only after the new rows and patch anchors are fully prepared.
    // Old shape-local anchors remain available throughout correlation.
    friction_patches_.swap(next_friction_patches_);
}
void Scene::solve_constraints() {
    // DySolverConstraintsShared.h::solveDynamicContacts /
    // solveStaticContacts and DySolverConstraints.cpp friction patch solve,
    // with the PhysX default of four position
    // iterations and one final velocity/writeback iteration. The final
    // writeback is itself a solve pass, not a second velocity sweep.
    // Wide prepared errors and impulses remain for the independently
    // admitted extreme translational mass range. Ordinary source PhysX
    // PxReal row precision is a separate strict-parity acceptance gate.
    constexpr unsigned position_iterations = 4;
    constexpr unsigned velocity_iterations = 1;
    constexpr double max_float =
        static_cast<double>(std::numeric_limits<float>::max());
    const auto dot = [](Vec3 u, Vec3 v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    };
    for (unsigned iter = 0;
         iter < position_iterations + velocity_iterations; ++iter) {
        // DyConstraintPartition's ordered descriptors, rather than
        // raw broadphase pair order, govern every position/velocity pass.
        // All contacts within a descriptor retain original Gu ordering.
        for (const std::size_t index : ordered_constraints_) {
            const auto& manifold = manifolds_[index];
            auto& a = actors_[manifold.pair.a].kind == Kind::DynamicBox
                ? solver_rows_[manifold.pair.a] : world_solver_row_;
            auto& b = actors_[manifold.pair.b].kind == Kind::DynamicBox
                ? solver_rows_[manifold.pair.b] : world_solver_row_;
            for (std::size_t i=0; i<manifold.count; ++i) {
                auto& row = solver_contact_points_[manifold.solver_offset+i];
                const Vec3 n = row.normal;
                const Vec3 va{
                    a.solver.linear_velocity_delta.x,
                    a.solver.linear_velocity_delta.y,
                    a.solver.linear_velocity_delta.z};
                const Vec3 vb{
                    b.solver.linear_velocity_delta.x,
                    b.solver.linear_velocity_delta.y,
                    b.solver.linear_velocity_delta.z};
                // PxSolverBody holds only velocity deltas; pre-solver
                // velocity is already included in row.biased_error.
                const float normal_velocity = dot(n,va)-dot(n,vb) +
                    dot(a.solver.angular_state,row.ra_x_n_sqrt_inertia) -
                    dot(b.solver.angular_state,row.rb_x_n_sqrt_inertia);
                const double error = iter < position_iterations
                    ? row.biased_error : row.unbiased_error;
                const double delta_candidate = std::max(
                    -static_cast<double>(normal_velocity) *
                        row.velocity_multiplier + error,
                    -row.applied_force);
                const double next_impulse = std::max(0.0,
                    row.applied_force + delta_candidate);
                const double delta = next_impulse-row.applied_force;
                const double da = delta*a.data.inverse_mass;
                const double db = delta*b.data.inverse_mass;
                const std::array<double,12> results{
                    static_cast<double>(va.x)+n.x*da,
                    static_cast<double>(va.y)+n.y*da,
                    static_cast<double>(va.z)+n.z*da,
                    static_cast<double>(vb.x)-n.x*db,
                    static_cast<double>(vb.y)-n.y*db,
                    static_cast<double>(vb.z)-n.z*db,
                    static_cast<double>(a.solver.angular_state.x)+row.ra_x_n_sqrt_inertia.x*delta,
                    static_cast<double>(a.solver.angular_state.y)+row.ra_x_n_sqrt_inertia.y*delta,
                    static_cast<double>(a.solver.angular_state.z)+row.ra_x_n_sqrt_inertia.z*delta,
                    static_cast<double>(b.solver.angular_state.x)-row.rb_x_n_sqrt_inertia.x*delta,
                    static_cast<double>(b.solver.angular_state.y)-row.rb_x_n_sqrt_inertia.y*delta,
                    static_cast<double>(b.solver.angular_state.z)-row.rb_x_n_sqrt_inertia.z*delta};
                std::array<float,12> commit{};
                for (std::size_t k=0;k<results.size();++k) {
                    if (!std::isfinite(results[k]) ||
                        std::abs(results[k]) > max_float)
                        throw std::overflow_error(
                            "Native Dy contact velocity delta exceeds float range");
                    commit[k] = static_cast<float>(results[k]);
                }
                a.solver.linear_velocity_delta = {commit[0],commit[1],commit[2]};
                b.solver.linear_velocity_delta = {commit[3],commit[4],commit[5]};
                a.solver.angular_state = {commit[6],commit[7],commit[8]};
                b.solver.angular_state = {commit[9],commit[10],commit[11]};
                row.applied_force = next_impulse;
            }
            // DySolverConstraints.cpp::solveContact/solveContact_BStatic:
            // each correlated friction patch uses the sum of its normal
            // impulses to clamp two tangent axes per retained anchor.
            // DySolverControl skips friction on the first of four
            // position iterations; later position and final velocity
            // writeback passes all solve these same patch constraints.
            if (iter != 0) {
                for (std::size_t p=manifold.friction_patch_offset;
                     p<manifold.friction_patch_offset+
                       manifold.friction_patch_count; ++p) {
                    auto& patch=friction_patches_[p];
                    if (!patch.anchor_count) continue;
                    double normal_impulse=0.0;
                    for (std::size_t j=0;j<manifold.count;++j) {
                        const auto& row=solver_contact_points_[
                            manifold.solver_offset+j];
                        if (row.friction_patch==p)
                            normal_impulse+=row.applied_force;
                    }
                    const double coefficient=patch.anchor_count==2
                        ? 0.5 : 1.0;
                    const double max_static=
                        patch.static_friction*coefficient*normal_impulse;
                    const double max_dynamic=
                        patch.dynamic_friction*coefficient*normal_impulse;
                    // DySolverConstraints.cpp::solveContact and
                    // solveContact_BStatic create a fresh local 'broken'
                    // value on EVERY solve call. Only the final
                    // writeBackIteration publishes it to the friction
                    // patch's cached broken byte. Position-pass
                    // saturation must not invalidate an anchor that has
                    // converged during the unbiased velocity pass.
                    bool broken_this_iteration=false;
                    for (std::size_t k=patch.row_offset;
                         k<patch.row_offset+patch.anchor_count*2; ++k) {
                        auto& fr=solver_friction_rows_[k];
                        const Vec3 n=fr.tangent;
                        const Vec3 va=a.solver.linear_velocity_delta;
                        const Vec3 vb=b.solver.linear_velocity_delta;
                        const float tangent_delta=
                            dot(n,va)-dot(n,vb)+
                            dot(a.solver.angular_state,fr.ra_x_t_sqrt_inertia)-
                            dot(b.solver.angular_state,fr.rb_x_t_sqrt_inertia);
                        // concludeContact zeroes friction bias on the
                        // final position pass, before motion-velocity
                        // publication and the final velocity iteration.
                        const double bias=iter < position_iterations
                            ? fr.bias : 0.0;
                        const double candidate=fr.applied_force+
                            (fr.target_velocity-bias-tangent_delta)*
                                fr.velocity_multiplier;
                        const bool clamped=std::abs(candidate)>max_static;
                        const double next=clamped
                            ? std::clamp(candidate,-max_dynamic,max_dynamic)
                            : candidate;
                        const double delta=next-fr.applied_force;
                        const double da=delta*a.data.inverse_mass;
                        const double db=delta*b.data.inverse_mass;
                        const auto& j0=fr.ra_x_t_sqrt_inertia;
                        const auto& j1=fr.rb_x_t_sqrt_inertia;
                        const std::array<double,12> results{
                            static_cast<double>(va.x)+n.x*da,
                            static_cast<double>(va.y)+n.y*da,
                            static_cast<double>(va.z)+n.z*da,
                            static_cast<double>(vb.x)-n.x*db,
                            static_cast<double>(vb.y)-n.y*db,
                            static_cast<double>(vb.z)-n.z*db,
                            static_cast<double>(a.solver.angular_state.x)+j0.x*delta,
                            static_cast<double>(a.solver.angular_state.y)+j0.y*delta,
                            static_cast<double>(a.solver.angular_state.z)+j0.z*delta,
                            static_cast<double>(b.solver.angular_state.x)-j1.x*delta,
                            static_cast<double>(b.solver.angular_state.y)-j1.y*delta,
                            static_cast<double>(b.solver.angular_state.z)-j1.z*delta};
                        std::array<float,12> commit{};
                        for (std::size_t v=0;v<results.size();++v) {
                            if (!std::isfinite(results[v]) ||
                                std::abs(results[v]) > max_float)
                                throw std::overflow_error(
                                    "Native Dy friction velocity delta exceeds float range");
                            commit[v]=static_cast<float>(results[v]);
                        }
                        a.solver.linear_velocity_delta={
                            commit[0],commit[1],commit[2]};
                        b.solver.linear_velocity_delta={
                            commit[3],commit[4],commit[5]};
                        a.solver.angular_state={
                            commit[6],commit[7],commit[8]};
                        b.solver.angular_state={
                            commit[9],commit[10],commit[11]};
                        fr.applied_force=next;
                        broken_this_iteration |= clamped;
                    }
                    if (iter + 1 ==
                        position_iterations + velocity_iterations)
                        patch.broken=broken_this_iteration;
                }
            }
        }
        if (iter + 1 == position_iterations) {
            // DySolverControl.cpp::saveMotionVelocities occurs immediately
            // after the last biased/conclude position iteration, BEFORE
            // unbiased velocity iterations alter the solver deltas.
            for (BodyId id : solver_active_ids_) {
                auto& row = solver_rows_[id];
                row.motion_linear_delta = row.solver.linear_velocity_delta;
                row.motion_angular_state = row.solver.angular_state;
            }
        }
    }
}
void Scene::write_back_contacts() {
    // DySolverConstraints.cpp::writeBackContact copies final forceBuffer
    // impulses after the last velocity/writeback solve. Gu contact order,
    // not the reordered Dy partition list, owns output offsets. Here the
    // source force buffer is represented by SolverContactPoint::applied_force
    // and the output retains SDK-wide impulses for admitted extreme masses.
    normal_impulse_writeback_.resize(contact_points_.size());
    // DyContactPrep::updateFrictionAnchorCountAndPosition pairs each
    // friction patch with its output's anchorPositions/anchorImpulses.
    // Preserve the Gu manager's patch offsets even when solver partitions
    // execute in a different order from the contact-manager stream.
    friction_impulse_writeback_.resize(friction_patches_.size());
    for (const auto& manifold : manifolds_) {
        if (manifold.contact_offset > contact_points_.size() ||
            manifold.count > contact_points_.size()-manifold.contact_offset ||
            manifold.solver_offset > solver_contact_points_.size() ||
            manifold.count > solver_contact_points_.size()-manifold.solver_offset)
            throw std::logic_error(
                "Native Dy contact writeback offset exceeds source streams");
        for (std::size_t i=0;i<manifold.count;++i) {
            const double impulse=solver_contact_points_[
                manifold.solver_offset+i].applied_force;
            if (!std::isfinite(impulse) || impulse < 0.0)
                throw std::overflow_error(
                    "Native Dy contact writeback has invalid normal impulse");
            normal_impulse_writeback_[manifold.contact_offset+i]=impulse;
        }
        if (manifold.friction_patch_offset>friction_patches_.size() ||
            manifold.friction_patch_count>
                friction_patches_.size()-manifold.friction_patch_offset)
            throw std::logic_error(
                "Native Dy friction writeback exceeds correlated patches");
        for (std::size_t p=manifold.friction_patch_offset;
             p<manifold.friction_patch_offset+
                 manifold.friction_patch_count; ++p) {
            const auto& source=friction_patches_[p];
            if (!(source.pair==manifold.pair) || source.anchor_count>2 ||
                source.row_offset>solver_friction_rows_.size() ||
                source.anchor_count*2>
                    solver_friction_rows_.size()-source.row_offset)
                throw std::logic_error(
                    "Native Dy friction writeback patch ownership invalid");
            FrictionPatchWriteback& result=friction_impulse_writeback_[p];
            result={};
            result.pair=source.pair;
            result.anchor_count=source.anchor_count;
            result.broken=source.broken;
            const auto& actor=actors_[source.pair.a];
            for (std::size_t k=0;k<source.anchor_count;++k) {
                const Vec3 local=source.body0_anchors[k];
                const Vec3 rotated=rotate(actor.orientation,local);
                result.anchor_positions[k]={
                    actor.position.x+rotated.x,
                    actor.position.y+rotated.y,
                    actor.position.z+rotated.z};
                // DyContactPrep.cpp::writeBackContactFriction reconstructs
                // world impulse by summing both tangent vectors scaled
                // by each source SolverContactFriction::appliedForce.
                const auto& x=solver_friction_rows_[source.row_offset+2*k];
                const auto& y=solver_friction_rows_[source.row_offset+2*k+1];
                result.anchor_impulses[k]={
                    static_cast<double>(x.tangent.x)*x.applied_force+
                        static_cast<double>(y.tangent.x)*y.applied_force,
                    static_cast<double>(x.tangent.y)*x.applied_force+
                        static_cast<double>(y.tangent.y)*y.applied_force,
                    static_cast<double>(x.tangent.z)*x.applied_force+
                        static_cast<double>(y.tangent.z)*y.applied_force};
                const auto& v=result.anchor_impulses[k];
                if (!finite(result.anchor_positions[k]) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) ||
                    !std::isfinite(v.z))
                    throw std::overflow_error(
                        "Native Dy friction impulse writeback is not finite");
            }
        }
    }
}
void Scene::update_threshold_stream(float seconds) {
    // DySolverConstraints.cpp::writeBackContact emits a threshold element
    // only for a supported rigid-body manager with a nonzero final
    // normal impulse and an enabled reportThreshold on either body.
    // DyThresholdTable groups all static contacts against a dynamic body
    // using the SAME source world solver node. Do not key by the Gu
    // contact pair: distinct static shapes must sum before comparison.
    // A dynamic/dynamic contact retains its canonical body-pair key.
    //
    // DyThresholdTable::build aggregates impulses by rigid-body pair and
    // PxsForceThresholdTask::createForceChangeThresholdStream compares
    // aggregate impulse against threshold*dt. Never divide impulse by
    // dt and narrow to PxReal: this SDK also admits extreme mass cases.
    threshold_current_.clear();
    threshold_changes_.clear();
    for (const auto& manifold:manifolds_) {
        const Pair pair=manifold.pair;
        const bool dynamic_a=actors_[pair.a].kind==Kind::DynamicBox;
        const bool dynamic_b=actors_[pair.b].kind==Kind::DynamicBox;
        if (!dynamic_a && !dynamic_b)
            continue;
        const bool static_world=!(dynamic_a && dynamic_b);
        const ForceThresholdKey key=static_world
            ? ForceThresholdKey{dynamic_a ? pair.a : pair.b,0,true}
            : ForceThresholdKey{pair.a,pair.b,false};
        const float threshold=static_world
            ? actors_[key.actor0].contact_report_threshold
            : std::min(actors_[pair.a].contact_report_threshold,
                       actors_[pair.b].contact_report_threshold);
        if (threshold==std::numeric_limits<float>::max())
            continue;
        if (manifold.contact_offset>normal_impulse_writeback_.size() ||
            manifold.count>normal_impulse_writeback_.size()-
                manifold.contact_offset)
            throw std::logic_error(
                "Native Dy threshold stream exceeds source force writeback");
        double sum=0.0;
        for (std::size_t i=0;i<manifold.count;++i)
            sum+=normal_impulse_writeback_[manifold.contact_offset+i];
        if (!std::isfinite(sum))
            throw std::overflow_error(
                "Native Dy threshold normal impulse accumulation overflow");
        if (sum==0.0)
            continue;
        threshold_current_.push_back({key,sum,threshold});
    }
    // Map Gu manager pairs onto the shared static-world identity BEFORE
    // DyThresholdTable aggregation. The mapped sequence is no longer
    // necessarily ordered (different static actors interleave), so sort
    // only these active threshold candidates, in the reusable arena.
    // Do not allocate a parallel hash table or reorder Gu contacts.
    std::sort(threshold_current_.begin(),threshold_current_.end(),
        [](const ThresholdPair& a, const ThresholdPair& b) {
            return a.key<b.key;
        });
    std::size_t write=0;
    for (const auto& row:threshold_current_) {
        if (write && threshold_current_[write-1].key==row.key) {
            threshold_current_[write-1].impulse+=row.impulse;
            threshold_current_[write-1].threshold=std::min(
                threshold_current_[write-1].threshold,row.threshold);
        } else {
            threshold_current_[write++]=row;
        }
    }
    threshold_current_.resize(write);
    // Source tests accumulated per-pair impulse *strictly greater*
    // than reportThreshold * dt. This runs after final force writeback.
    write=0;
    for (const auto& row:threshold_current_) {
        if (row.impulse>static_cast<double>(row.threshold)*seconds)
            threshold_current_[write++]=row;
    }
    threshold_current_.resize(write);
    // Previous/current exceeded-force stream merge. The source also
    // includes persistent exceeded pairs in its force-change stream;
    // LOST events have zero accumulated force, while PERSISTS uses the
    // preceding element's force (as in PxsForceThresholdTask).
    std::size_t old_index=0,new_index=0;
    while (old_index<threshold_previous_.size() ||
           new_index<threshold_current_.size()) {
        const bool has_old=old_index<threshold_previous_.size();
        const bool has_new=new_index<threshold_current_.size();
        if (has_old &&
            (!has_new || threshold_previous_[old_index].key <
                         threshold_current_[new_index].key)) {
            const auto& old=threshold_previous_[old_index++];
            threshold_changes_.push_back({
                old.key,0.0,old.threshold,ForceThresholdChange::Lost});
        } else if (has_new &&
                   (!has_old || threshold_current_[new_index].key <
                                threshold_previous_[old_index].key)) {
            const auto& now=threshold_current_[new_index++];
            threshold_changes_.push_back({
                now.key,now.impulse,now.threshold,
                ForceThresholdChange::Found});
        } else {
            const auto& old=threshold_previous_[old_index++];
            ++new_index;
            threshold_changes_.push_back({
                old.key,old.impulse,old.threshold,
                ForceThresholdChange::Persists});
        }
    }
    threshold_previous_.swap(threshold_current_);
}
void Scene::publish_solver_bodies() {
    // The source retains pre-solver velocity independently and publishes
    // solved motion velocity after the iteration/conclude/writeback sequence.
    // The source's rigid-body normal-contact rows now update both linear
    // and angular velocity state. Friction/correlation/writeback parity
    // remains a separate N1 acceptance requirement.
    for (BodyId id : solver_active_ids_) {
        auto& actor = actors_[id];
        if (!actor.alive || actor.kind != Kind::DynamicBox) continue;
        const auto& row = solver_rows_[id];
        const Vec3& d = row.solver.linear_velocity_delta;
        const Vec3 linear{
            row.data.linear_velocity.x+d.x,
            row.data.linear_velocity.y+d.y,
            row.data.linear_velocity.z+d.z};
        // DyBodyCoreIntegrator::integrateCore publishes angular solver
        // state through the *world* square-root inverse inertia tensor,
        // rather than treating PxSolverBody::angularState as angular velocity.
        const Vec3& angular_state = row.solver.angular_state;
        const auto& i = row.data.sqrt_inverse_inertia;
        const Vec3 angular{
            row.data.angular_velocity.x + i[0].x*angular_state.x +
                i[1].x*angular_state.y + i[2].x*angular_state.z,
            row.data.angular_velocity.y + i[0].y*angular_state.x +
                i[1].y*angular_state.y + i[2].y*angular_state.z,
            row.data.angular_velocity.z + i[0].z*angular_state.x +
                i[1].z*angular_state.y + i[2].z*angular_state.z};
        if (!finite(linear) || !finite(angular))
            throw std::overflow_error("Native solver publication is not finite");
        actor.velocity = linear;
        actor.angular_velocity = angular;
    }
}
void Scene::StepCheckpoint::capture(Scene& scene) {
    // Reuse capacities across frames. The checkpoint contains only state
    // that persists beyond a step or is exposed by Scene's read-only APIs.
    // It intentionally excludes rebuilt solver constraints and scratch.
    motions.clear();
    for(std::size_t id=0;id<scene.actors_.size();++id) {
        const auto& a=scene.actors_[id];
        if(!a.alive || a.kind!=Kind::DynamicBox)continue;
        motions.push_back({static_cast<BodyId>(id),
            a.position,a.velocity,a.angular_velocity,a.orientation,
            a.pending_angular_velocity_delta});
    }
    friction_patches.assign(scene.friction_patches_.begin(),scene.friction_patches_.end());
     threshold_previous.assign(scene.threshold_previous_.begin(),
                              scene.threshold_previous_.end());
     partition_ends.assign(scene.partition_ends_.begin(),scene.partition_ends_.end());
    speculative_parent.assign(scene.speculative_island_parent_.begin(),
                              scene.speculative_island_parent_.end());
    accurate_parent.assign(scene.accurate_island_parent_.begin(),
                           scene.accurate_island_parent_.end());
    speculative_rank.assign(scene.speculative_island_rank_.begin(),
                            scene.speculative_island_rank_.end());
    accurate_rank.assign(scene.accurate_island_rank_.begin(),
                         scene.accurate_island_rank_.end());
    accurate_nodes.assign(scene.accurate_island_nodes_.begin(),
                          scene.accurate_island_nodes_.end());
    accurate_offsets.assign(scene.accurate_island_offsets_.begin(),
                            scene.accurate_island_offsets_.end());
    speculative_count=scene.speculative_island_count_;
    accurate_count=scene.accurate_island_count_;
    topology_revision=scene.island_topology_revision_;
    full_rebuild_count=scene.island_full_rebuild_count_;
    frame_index=scene.frame_index_;
    pair_epoch=scene.pair_epoch_;
    contact_epoch=scene.contact_epoch_;
    topology_dirty=scene.island_topology_dirty_;
    dirty=scene.dirty_;
    // The prior published buffers are moved into the checkpoint only
    // after all potentially throwing copies have completed. Successful
    // solves write directly into the reusable alternate allocations.
    // Rollback reverses these swaps without copying or invalidating
    // previously published report spans.
    normal_impulses.clear();
    friction_impulses.clear();
    threshold_changes.clear();
    scene.normal_impulse_writeback_.swap(normal_impulses);
    scene.friction_impulse_writeback_.swap(friction_impulses);
    scene.threshold_changes_.swap(threshold_changes);
}

void Scene::StepCheckpoint::rollback(Scene& scene) noexcept {
    // Swaps cannot allocate or fail with the standard vector allocator.
    // In particular, pending angular impulses and the authoritative
    // previous-frame friction/threshold streams survive failed solves.
    for(const Motion& motion:motions) {
        auto& a=scene.actors_[motion.id];
        a.position=motion.position;
        a.velocity=motion.velocity;
        a.angular_velocity=motion.angular_velocity;
        a.orientation=motion.orientation;
        a.pending_angular_velocity_delta=motion.pending_angular_velocity_delta;
    }
    // Swap back the exact original arena allocations. This preserves the
    // lifetime of contact spans returned before the unsuccessful step.
    if(scene.pair_epoch_!=pair_epoch) {
        scene.previous_pairs_.swap(scene.pending_pairs_);
        scene.found_.swap(scene.next_found_);
        scene.lost_.swap(scene.next_lost_);
    }
    scene.pair_epoch_=pair_epoch;
    if(scene.contact_epoch_!=contact_epoch) {
        scene.manifolds_.swap(scene.next_manifolds_);
        scene.contact_points_.swap(scene.next_contact_points_);
        scene.found_contacts_.swap(scene.next_found_contacts_);
        scene.lost_contacts_.swap(scene.next_lost_contacts_);
    }
    scene.contact_epoch_=contact_epoch;
    scene.friction_patches_.swap(friction_patches);
    scene.normal_impulse_writeback_.swap(normal_impulses);
    scene.friction_impulse_writeback_.swap(friction_impulses);
    scene.threshold_previous_.swap(threshold_previous);
    scene.threshold_changes_.swap(threshold_changes);
    scene.partition_ends_.swap(partition_ends);
    scene.speculative_island_parent_.swap(speculative_parent);
    scene.accurate_island_parent_.swap(accurate_parent);
    scene.speculative_island_rank_.swap(speculative_rank);
    scene.accurate_island_rank_.swap(accurate_rank);
    scene.accurate_island_nodes_.swap(accurate_nodes);
    scene.accurate_island_offsets_.swap(accurate_offsets);
    scene.speculative_island_count_=speculative_count;
    scene.accurate_island_count_=accurate_count;
    scene.island_topology_revision_=topology_revision;
    scene.island_full_rebuild_count_=full_rebuild_count;
    scene.frame_index_=frame_index;
    scene.island_topology_dirty_=topology_dirty;
    scene.dirty_=dirty;
 }

void Scene::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f)
        throw std::invalid_argument("Native CPU timestep must be finite and positive");

    // Source: ScBodyCore.cpp::BodyCore dynamic defaults, and
    // DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity:
    // gravity, linear/angular damping, then PxReal speed clamps.
    constexpr float gravity = -9.81f;
    constexpr float linear_damping = 0.0f;
    constexpr float angular_damping = 0.05f;
    constexpr float max_linear_velocity_squared = 1.0e32f;
    constexpr float max_angular_velocity_squared = 1.0e4f;
    // A failed frame must not consume impulses or publish partial
    // contacts, island nodes, body motion or result streams. Snapshot only
    // authoritative state; all per-step Dy work buffers remain reusable.
    step_checkpoint_.capture(*this);
    try {
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
        if (!finite(velocity) || !finite(angular))
            throw std::overflow_error("Native CPU unconstrained velocity overflow");
        actor.velocity = velocity;
        actor.angular_velocity = angular;
        actor.pending_angular_velocity_delta = {};
        // Source ScPipeline/DySolverControl: do not predict and publish
        // the position before Gu contacts and the biased PGS passes.
        // Its integration requires the saved position-solver motion
        // velocity, not the final unbiased next-frame velocity.
        dirty_ = true;
    }
    // Source-derived per-point normals and correlated friction rows
    // share the original actor/contact owner and ordered contact batches.
    // The complete island task graph, packed constraint ABI, heterogeneous
    // material/contact callbacks and CCD remain explicit N1 admission gates.
    advance_collision_internal(false);
    // The source third-pass island tasks may overlap the numerical solve.
    // For this all-awake, single-threaded subset, the lost-contact
    // producer is already complete, so this is a valid dependency-ready
    // execution order; no ordering of third pass *after* solve is implied.
    // Actual source task continuations and deactivation are still N1 gates.
    third_pass_island_gen();
    prepare_solver_bodies();
    partition_constraints();
    prepare_constraints(seconds);
    solve_constraints();
    write_back_contacts();
    update_threshold_stream(seconds);
    publish_solver_bodies();
    // DyBodyCoreIntegrator.h::integrateCore uses the saved *position*
    // iteration motion velocities to move/rotate bodies. The velocity
    // iterations above publish their different, unbiased final velocities
    // for subsequent frames; they do not undo this frame's correction.
    for(std::size_t island=0;island+1<accurate_island_offsets_.size();
        ++island) {
        for(std::size_t index=accurate_island_offsets_[island];
            index<accurate_island_offsets_[island+1];++index) {
            const BodyId id=accurate_island_nodes_[index];
            auto& actor=actors_[id];
            Vec3 motion_linear = actor.velocity;
            Vec3 motion_angular = actor.angular_velocity;
            // Scene clock increments in advance_collision_internal; only
            // this frame's actual contact participants own motion snapshots.
            const bool solved_contact = id < solver_rows_.size() &&
                solver_rows_[id].prepared_frame == frame_index_;
            if (solved_contact) {
                const auto& row = solver_rows_[id];
                const Vec3& delta = row.motion_linear_delta;
                motion_linear = {
                    row.data.linear_velocity.x + delta.x,
                    row.data.linear_velocity.y + delta.y,
                    row.data.linear_velocity.z + delta.z};
                const auto& i = row.data.sqrt_inverse_inertia;
                const Vec3& angular_delta = row.motion_angular_state;
                motion_angular = {
                    row.data.angular_velocity.x + i[0].x*angular_delta.x +
                        i[1].x*angular_delta.y + i[2].x*angular_delta.z,
                    row.data.angular_velocity.y + i[0].y*angular_delta.x +
                        i[1].y*angular_delta.y + i[2].y*angular_delta.z,
                    row.data.angular_velocity.z + i[0].z*angular_delta.x +
                        i[1].z*angular_delta.y + i[2].z*angular_delta.z};
            }
            const Vec3 p{
                static_cast<float>(static_cast<double>(actor.position.x) +
                                   static_cast<double>(motion_linear.x)*seconds),
                static_cast<float>(static_cast<double>(actor.position.y) +
                                   static_cast<double>(motion_linear.y)*seconds),
                static_cast<float>(static_cast<double>(actor.position.z) +
                                   static_cast<double>(motion_linear.z)*seconds)};
            if (!finite(p) || !finite(motion_linear) || !finite(motion_angular))
                throw std::overflow_error("Native position-iteration motion overflow");
            const Quat q = integrate_quaternion(
                actor.orientation, motion_angular, seconds);
            if (!finite(q))
                throw std::overflow_error("Native CPU orientation overflow");
            actor.position = p;
            actor.orientation = q;
            dirty_ = true;
        }
    }
    // Solver and lost-edge producers have both completed before this
    // successful step returns its published body and contact results.
    } catch (...) {
        step_checkpoint_.rollback(*this);
        throw;
    }
}

void Scene::advance_collision(bool force_update) {
    // Geometry-only queries are synchronous completed publications: run
    // the same post-collision island retirement barrier as a full step.
    advance_collision_internal(force_update);
    third_pass_island_gen();
}
void Scene::advance_collision_internal(bool force_update) {
    // This also serves geometry-only queries. Never publish last frame's
    // solved force-buffer values against an unsolved contact stream.
    normal_impulse_writeback_.clear();
    friction_impulse_writeback_.clear();
    // This clears only the exposed reports; prior exceeded-force state
    // must survive collision queries until the next simulation step.
    threshold_changes_.clear();
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
    // Source first speculative and second accurate pass: edge insertion
    // and active body admission, but no lost-edge island splitting yet.
    update_island_connectivity(false);
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
    void set_body_material(BodyId id, Material material) override {
        scene_.set_material(id,material);
    }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend() {
    return std::make_unique<NativeCpuWorld>();
}
} // namespace fyzz
