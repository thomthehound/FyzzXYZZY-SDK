// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/world.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace fyzz::native {

// Source correspondence: BpBroadPhaseSap / BpBroadPhaseIntegerAABB.
// The scene exclusively owns actors, encoded bounds, overlap history and reports.
// An overlap candidate is not a narrowphase contact or a solved constraint.
struct Pair {
    BodyId a = 0;
    BodyId b = 0;
    friend bool operator==(const Pair&, const Pair&) = default;
    friend bool operator<(const Pair& lhs, const Pair& rhs) {
        return lhs.a < rhs.a || (lhs.a == rhs.a && lhs.b < rhs.b);
    }
};

// Matches Gu::PxContactPoint semantics for the presently supported shapes:
// normal points from shape1 toward shape0, signed separation is negative
// for penetration. Contacts are geometry results, not solved impulses.
struct ContactPoint {
    Vec3 position{};
    Vec3 normal{};
    float separation = 0.0f;
};
struct ContactManifold {
    Pair pair{};
    std::array<ContactPoint, 8> points{};
    std::uint8_t count = 0;
    // Per-frame scalar PGS contact constraint: one physical patch, not
    // duplicate per-vertex full-body impulses.
    float velocity_target = 0.0f;
    float normal_impulse = 0.0f;
};

class Scene final {
public:
    BodyId add_static_box(Vec3 half_extents, Vec3 position);
    BodyId add_static_floor(float height);
    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density);
    void set_position(BodyId id, Vec3 position);
    void remove(BodyId id);
    BodyState body_state(BodyId id) const;

    // One scene-owned collision progression: update bounds -> SAP ->
    // overlap publication -> narrowphase -> contact publication.
    // Dynamics/constraints/solver/fetch are intentionally not admitted.
    // Returned spans remain valid until the next successful advance.
    void advance_collision(bool force_update = false);
    // Native CPU translation-constrained rigid bodies: single scene clock,
    // collision stream, unilateral normal constraint solver and publication.
    // Angular DOFs/friction/joints/CCD remain unsupported.
    void step(float seconds);
    std::span<const Pair> pairs() const noexcept { return previous_pairs_; }
    std::span<const Pair> found_pairs() const noexcept { return found_; }
    std::span<const Pair> lost_pairs() const noexcept { return lost_; }
    std::span<const ContactManifold> contacts() const noexcept {
        return manifolds_;
    }
    std::span<const Pair> found_contacts() const noexcept { return found_contacts_; }
    std::span<const Pair> lost_contacts() const noexcept { return lost_contacts_; }
    std::uint64_t frame_index() const noexcept { return frame_index_; }

private:
    enum class Kind : std::uint8_t { StaticBox, DynamicBox, StaticPlane };
    struct Actor {
        Kind kind = Kind::StaticBox;
        bool alive = false;
        Vec3 half_extents{};
        Vec3 position{};
        Vec3 velocity{};
        float density = 0.0f;
        float inverse_mass = 0.0f;  // immutable cached derived body property
        std::array<std::uint32_t, 6> encoded_bounds{};
    };
    struct Endpoint {
        std::uint32_t min_x = 0;
        BodyId id = 0;
    };
    BodyId add(Actor actor);
    Actor& require(BodyId id);
    const Actor& require(BodyId id) const;
    void update_bounds();
    void update_pairs();
    void publish_pairs();
    void update_contacts();
    void prepare_constraints(float seconds);
    void solve_constraints();

    std::vector<Actor> actors_;
    std::vector<Endpoint> endpoints_;
    std::vector<BodyId> active_;
    std::vector<Pair> previous_pairs_;
    std::vector<Pair> pending_pairs_;
    std::vector<Pair> found_;
    std::vector<Pair> lost_;

    // Source contact streams (GuContactPlaneBox/GuContactBoxBox) are owned
    // here, not by a second narrowphase scene or external reference PxScene.
    // Reuse both vectors' allocations across frames, swap only at publish.
    std::vector<ContactManifold> manifolds_;
    std::vector<ContactManifold> next_manifolds_;
    std::vector<Pair> found_contacts_;
    std::vector<Pair> lost_contacts_;
    std::uint64_t frame_index_ = 0;
    bool dirty_ = true;
};

} // namespace fyzz::native
