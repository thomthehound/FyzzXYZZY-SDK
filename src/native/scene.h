// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/world.h"

#include <array>
#include <cstddef>
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
    // Source contact streams use a scene-owned contiguous arena, not an
    // eight-vertex limit or an allocating vector for each manager.
    std::span<const ContactPoint> points{};
    std::size_t contact_offset = 0;
    std::uint8_t count = 0;
    // GuContactBoxBox collision-axis cache (mPairData), retained by pair.
    std::uint8_t collision_axis = 0;
    // Transient source Dy contact constraints are held in the scene's
    // contiguous row arena, indexed by this manager's offset and count.
    std::size_t solver_offset = 0;
    std::size_t friction_patch_offset = 0;
    std::uint8_t friction_patch_count = 0;
};

// Source translation: GuContactBoxBox's SAT and contact generation on
// caller-owned contact storage, preserving per-pair axis coherence.
std::uint8_t source_box_box_contacts(
    Vec3 half0, Vec3 position0, Quat rotation0,
    Vec3 half1, Vec3 position1, Quat rotation1,
    std::uint8_t& previous_axis,
    std::span<ContactPoint> output);

class Scene final {
public:
    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = default;
    Scene& operator=(Scene&&) = default;

    BodyId add_static_box(Vec3 half_extents, Vec3 position);
    BodyId add_static_floor(float height);
    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density);
    void set_position(BodyId id, Vec3 position);
    void set_velocity(BodyId id, Vec3 velocity);
    void set_angular_velocity(BodyId id, Vec3 velocity);
    void apply_torque_impulse(BodyId id, Vec3 impulse);
    void remove(BodyId id);
    BodyState body_state(BodyId id) const;

    // One scene-owned collision progression: update bounds -> SAP ->
    // overlap publication -> source Gu narrowphase -> contact publication.
    // Collision-only queries return Gu geometry independent of a solver step.
    // Returned spans remain valid until the next successful advance.
    void advance_collision(bool force_update = false);
    // Native CPU scene owns free-body angular motion and source-derived
    // per-point normal/angular Dy contact rows in its original clock.
    // Friction correlation, joints, CCD and complete island/PGS scheduling
    // are not integrated; N1 general rigid-body parity is NOT yet accepted.
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
        Quat orientation{};
        Vec3 angular_velocity{};
        float density = 0.0f;
        float inverse_mass = 0.0f;  // immutable cached derived body property
        // PhysX PxMassProperties + ScBodyCore inverse inertia for a box
        // about its mass center, in actor-local principal axes.
        Vec3 inverse_inertia_local{};
        // A finite extremely small mass can be admitted by the legacy
        // translational solver even when PxReal angular inverse inertia
        // cannot be represented. Do not invent an angular tensor: reject
        // angular operations on those bodies until full source admission.
        bool angular_inertia_available = false;
        // PhysX ScBodySim::VelocityMod's per-step angular delta; consumed
        // by the existing scene clock, not a second force owner.
        Vec3 pending_angular_velocity_delta{};
        std::array<std::uint32_t, 6> encoded_bounds{};
    };
    // DyRigidBodyToSolverBody.cpp::copyToSolverBodyData and
    // PxSolverDefs.h::PxSolverBody/PxSolverBodyData. These are
    // transient source solver state owned by this scene, not a second world.
    struct SolverBody {
        Vec3 linear_velocity_delta{};
        Vec3 angular_state{};
    };
    struct SolverBodyData {
        Vec3 linear_velocity{};
        Vec3 angular_velocity{};
        float inverse_mass = 0.0f;
        std::array<Vec3, 3> sqrt_inverse_inertia{};
        Vec3 position{};
        Quat orientation{};
    };
    struct SolverBodyRow {
        SolverBody solver{};
        SolverBodyData data{};
        // DySolverControl::saveMotionVelocities saves the solver deltas
        // after position iterations, before conclude/velocity iterations.
        // DyBodyCoreIntegrator uses this snapshot to advance the pose,
        // while the final solver deltas write back next-frame velocity.
        Vec3 motion_linear_delta{};
        Vec3 motion_angular_state{};
        std::uint64_t prepared_frame = 0;
    };
    // Source: DySolverContact.h::SolverContactPoint and
    // DySolverConstraintsShared.h::solveDynamicContacts. Each generated
    // contact owns its individual effective mass, angular Jacobians,
    // bias and impulse; no manifold-level normal impulse approximation.
    struct SolverContactPoint {
        Vec3 normal{};
        Vec3 ra_x_n_sqrt_inertia{};
        Vec3 rb_x_n_sqrt_inertia{};
        double velocity_multiplier = 0.0;
        // Extended only for SDK-admitted inverse masses below the range
        // of a representable PhysX float impulse. No float narrowing
        // between constraint preparation and the actual impulse update.
        double biased_error = 0.0;
        double unbiased_error = 0.0;
        double applied_force = 0.0;
        std::size_t friction_patch = 0;
    };
    // DyFrictionPatch.h, DyCorrelationBuffer.h and DySolverContact.h:
    // persistent shape-local anchors, manager-correlated patches, and the
    // two source tangential constraint axes per anchor.
    struct FrictionPatch {
        Pair pair{};
        // Source DyFrictionPatch stores both previous-frame normals in
        // body-local coordinates; the active row normal remains world-space.
        Vec3 normal{};
        Vec3 body0_normal{};
        Vec3 body1_normal{};
        std::array<Vec3,2> body0_anchors{};
        std::array<Vec3,2> body1_anchors{};
        std::array<std::uint16_t,2> contact_ids{};
        Vec3 bounds_min{}, bounds_max{};
        std::size_t row_offset = 0;
        std::uint16_t contact_count = 0;
        std::uint8_t anchor_count = 0;
        bool broken = false;
    };
    struct SolverContactFriction {
        Vec3 tangent{};
        Vec3 ra_x_t_sqrt_inertia{};
        Vec3 rb_x_t_sqrt_inertia{};
        double velocity_multiplier = 0.0;
        double bias = 0.0;
        double target_velocity = 0.0;
        double applied_force = 0.0;
    };
    struct Endpoint {
        std::uint32_t min_x = 0;
        BodyId id = 0;
    };
    BodyId add(Actor actor);
    Actor& require(BodyId id);
    const Actor& require(BodyId id) const;
    // Shared Gu collision progression, independent of Dy solver execution.
    void advance_collision_internal(bool force_update);
    void update_bounds();
    void update_pairs();
    void publish_pairs();
    void update_contacts();
    void prepare_solver_bodies();
    void prepare_constraints(float seconds);
    void solve_constraints();
    void publish_solver_bodies();

    std::vector<Actor> actors_;
    // Capacity persists across frames; matching actor indices avoid a shadow
    // body-ID registry. Only solver deltas are published back to actors.
    std::vector<SolverBodyRow> solver_rows_;
    // Reusable flat Dy normal-constraint stream; patch ownership and all
    // point offsets remain with this scene, not with allocating manifolds.
    std::vector<SolverContactPoint> solver_contact_points_;
    // Correlation and friction constraint streams retain capacity across
    // frames, while patches carry actor-local anchors across contact frames.
    std::vector<FrictionPatch> friction_patches_;
    std::vector<FrictionPatch> next_friction_patches_;
    std::vector<SolverContactFriction> solver_friction_rows_;
    // DyDynamics::mWorldSolverBody / mWorldSolverBodyData: one zero-mass
    // solver body shared by all static collision participants.
    SolverBodyRow world_solver_row_{};
    // Prepared contact bodies for the current frame; avoids re-preparing
    // free-flight actors before the original island owner is integrated.
    std::vector<BodyId> solver_active_ids_;
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
    // Double-buffer the source contact-point stream with manifold ownership.
    // Spans are assigned after fill, when growth can no longer invalidate them.
    std::vector<ContactPoint> contact_points_;
    std::vector<ContactPoint> next_contact_points_;
    // PxContactBuffer::MAX_CONTACTS, reused for all box-box candidates.
    std::array<ContactPoint, 255> box_contact_scratch_{};
    std::vector<Pair> found_contacts_;
    std::vector<Pair> lost_contacts_;
    std::uint64_t frame_index_ = 0;
    bool dirty_ = true;
};

} // namespace fyzz::native
