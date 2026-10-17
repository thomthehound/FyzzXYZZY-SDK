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
                    // Fixed global +Y plane: source normal is its negative,
                    // from the box toward the plane.
                    manifold.points[manifold.count++] =
                        {point, Vec3{0.0f, -1.0f, 0.0f}, separation};
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
            // Within contactDistance, transverse faces may have a tiny gap.
            // Its midpoint is a valid proximity contact point.
            const float span_u[2] = {
                low_u <= high_u ? low_u : (low_u + high_u) * 0.5f,
                low_u <= high_u ? high_u : (low_u + high_u) * 0.5f};
            const float span_v[2] = {
                low_v <= high_v ? low_v : (low_v + high_v) * 0.5f,
                low_v <= high_v ? high_v : (low_v + high_v) * 0.5f};
            std::array<float, 3> normal{0.0f, 0.0f, 0.0f};
            normal[axis] = sign; // from shape b to a
            for (unsigned i = 0; i < 4; ++i) {
                auto point = face;
                point[u] = span_u[i & 1u];
                point[v] = span_v[(i >> 1u) & 1u];
                manifold.points[manifold.count++] = {
                    Vec3{point[0], point[1], point[2]},
                    Vec3{normal[0], normal[1], normal[2]},
                    signed_gaps[axis]};
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
    void step(float) override {
        // The native scene owns actors and broadphase; dynamics is not ported.
        throw std::logic_error("Native CPU narrowphase and solver are not yet integrated");
    }
    BodyState body_state(BodyId id) const override { return scene_.body_state(id); }
    void set_body_position(BodyId id, Vec3 p) override {
        scene_.set_position(id, p);
    }
    void remove_body(BodyId id) override { scene_.remove(id); }
private:
    native::Scene scene_;
};
} // namespace
std::unique_ptr<WorldBackend> create_native_cpu_backend() {
    return std::make_unique<NativeCpuWorld>();
}
} // namespace fyzz
