// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/world.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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
// DyContactPrep.cpp::updateFrictionAnchorCountAndPosition and
// writeBackContactFriction: solver anchor impulses are vectors in world
// space, one per retained friction anchor. The native representation uses
// double components only to preserve the SDK's separately admitted
// extreme inverse-mass range; original PhysX reports PxVec3.
struct FrictionImpulse {
    double x=0.0, y=0.0, z=0.0;
};
struct FrictionPatchWriteback {
    Pair pair{};
    std::uint8_t anchor_count=0;
    std::array<Vec3,2> anchor_positions{};
    std::array<FrictionImpulse,2> anchor_impulses{};
    bool broken=false;
};
// DyThresholdTable / DyDynamics::PxsForceThresholdTask. One authored
// shape per actor in the current native scene, so the shape interaction
// identity is the canonical BodyId pair. The source's island node index
// and Sc contact callback ownership are still future N1 integrations.
enum class ForceThresholdChange : std::uint8_t { Found, Persists, Lost };
// DyThresholdTable keys by island node indices, not by Gu shape pairs.
// One dynamic body shares the same static-world node across ALL static
// collision partners. For that case actor0 is the dynamic BodyId,
// actor1 is unused and static_world is true. Dynamic/dynamic keys store
// both canonical BodyIds with static_world false.
struct ForceThresholdKey {
    BodyId actor0=0;
    BodyId actor1=0;
    bool static_world=false;
    friend bool operator==(const ForceThresholdKey&,
                           const ForceThresholdKey&) = default;
    friend bool operator<(const ForceThresholdKey& a,
                          const ForceThresholdKey& b) {
        if (a.actor0!=b.actor0) return a.actor0<b.actor0;
        if (a.static_world!=b.static_world)
            return a.static_world<b.static_world;
        return a.actor1<b.actor1;
    }
};
struct ForceThresholdReport {
    ForceThresholdKey key{};
    double normal_impulse=0.0;
    float threshold=0.0f;
    ForceThresholdChange change=ForceThresholdChange::Found;
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
    // A finite contact-report threshold opts a dynamic body into the
    // source's low-level force-threshold stream. Default PX_MAX_REAL
    // disables it. Units are force; evaluation uses impulse > force*dt.
    void set_contact_report_threshold(BodyId id, float force);
    // One noncompliant PxsMaterialData per currently admitted shape.
    // A material change is consumed by next native contact preparation.
    void set_material(BodyId id, Material material);
    void remove(BodyId id);
    BodyState body_state(BodyId id) const;

    // One scene-owned collision progression: update bounds -> SAP ->
    // overlap publication -> source Gu narrowphase -> contact publication.
    // Collision-only queries return Gu geometry independent of a solver step.
    // Returned spans remain valid until the next successful advance.
    void advance_collision(bool force_update = false);
    // Native CPU scene owns free-body angular motion and source-derived
    // per-point normal/angular Dy contact rows in its original clock.
    // Per-point normal and friction anchors, contact partitions, final
    // normal/friction writeback and low-level threshold streams are active.
    // Joints, CCD, heterogeneous materials, shape-interaction callbacks
    // and complete island/task execution remain N1 acceptance gates.
    void step(float seconds);
    std::span<const Pair> pairs() const noexcept { return previous_pairs_; }
    std::span<const Pair> found_pairs() const noexcept { return found_; }
    std::span<const Pair> lost_pairs() const noexcept { return lost_; }
    std::span<const ContactManifold> contacts() const noexcept {
        return manifolds_;
    }
    // DySolverConstraints.cpp::writeBackContact writes solved normal
    // impulses in Gu contact order; each manifold's contact_offset/count
    // indexes this flat scene-owned result. An advance_collision() call
    // invalidates writeback (geometry-only queries are not solver results).
    // Retain double for SDK-admitted extreme inverse-mass cases: ordinary
    // PhysX uses a PxReal output buffer. The corresponding friction-anchor
    // and low-level threshold streams are exposed separately below.
    std::span<const double> solved_normal_impulses() const noexcept {
        return normal_impulse_writeback_;
    }
    // DyContactPrep/DySolverConstraints friction writeback mirrors current
    // manifold friction_patch_offset/count and carries the two source
    // tangent-row impulses per retained anchor. As with normal writeback,
    // values are invalidated by collision-only advancement.
    std::span<const FrictionPatchWriteback> solved_friction_patches() const noexcept {
        return friction_impulse_writeback_;
    }
    // The current source Dy threshold-change stream after a solver step.
    // Like contact-force writeback, geometry-only progression invalidates
    // the exposed reports; previous-frame exceeded-pair history persists
    // until the next actual solver step.
    std::span<const ForceThresholdReport> force_threshold_changes() const noexcept {
        return threshold_changes_;
    }
    std::span<const Pair> found_contacts() const noexcept { return found_contacts_; }
    std::span<const Pair> lost_contacts() const noexcept { return lost_contacts_; }
    std::uint64_t frame_index() const noexcept { return frame_index_; }
    // Read-only solver instrumentation for source constraint partitioning.
    // Counts last-step nonempty conflict-free partition batches; the
    // shared static world row is not a dynamic partition participant.
    std::size_t solver_partition_count() const noexcept {
        return partition_ends_.size();
    }
    // IG::SimpleIslandManager's speculative (BP candidate) and accurate
    // (physical touch) connectivity, limited to the admitted awake rigid
    // bodies. Static world contacts do not merge dynamic islands.
    std::size_t speculative_island_count() const noexcept {
        return speculative_island_count_;
    }
    std::size_t accurate_island_count() const noexcept {
        return accurate_island_count_;
    }
    // Island membership admitted at Dy solver preparation, before the
    // lost-edge repair completion barrier.
    std::size_t pre_third_pass_island_count() const noexcept {
        return pre_third_pass_island_count_;
    }
    std::uint64_t island_topology_revision() const noexcept {
        return island_topology_revision_;
    }
    // Internal source-island instrumentation: unlike the topology
    // revision, this advances only for complete component rebuilds.
    std::uint64_t island_full_rebuild_count() const noexcept {
        return island_full_rebuild_count_;
    }
    // Read the existing accurate-node batch boundaries without exposing
    // transient solver arrays or allocating diagnostic copies.
    std::size_t largest_accurate_island_size() const noexcept {
        std::size_t largest=0;
        for(std::size_t i=1;i<accurate_island_offsets_.size();++i)
            largest=std::max(largest,
                accurate_island_offsets_[i]-accurate_island_offsets_[i-1]);
        return largest;
    }

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
        float contact_report_threshold=std::numeric_limits<float>::max();
        Material material{};
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
        // DyConstraintPartition.cpp reuses PxSolverBody::solverProgress
        // (partition bit mask), maxSolverNormalProgress (last dynamic
        // partition + 1), and maxSolverFrictionProgress (static edges).
        // These are transient per-frame scheduler fields on this SAME
        // source solver body; static world rows are never partition owners.
        std::uint32_t partition_mask = 0;
        std::size_t max_dynamic_partition = 0;
        std::size_t static_constraint_count = 0;
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
        float static_friction=0.5f;
        float dynamic_friction=0.5f;
        float restitution=0.2f;
        bool disable_friction=false;
        bool disable_strong_friction=false;
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
    void update_island_connectivity(bool repair_lost_edges);
    // Synchronous lowering of the source third-pass island continuation.
    // Only lost-edge component repair is admitted here; deactivation and
    // sleeping await source-owned node/edge lifecycle integration.
    void third_pass_island_gen();
    bool lost_island_edges_retain_routes(bool accurate);
    void prepare_solver_bodies();
    void partition_constraints();
    void prepare_constraints(float seconds);
    void solve_constraints();
    void write_back_contacts();
    void update_threshold_stream(float seconds);
    void publish_solver_bodies();

    std::vector<Actor> actors_;
    // The admitted all-awake rigid subset of PxsSimpleIslandManager's
    // speculative and accurate IslandSim connectivity. Pair and contact
    // streams remain the unique authoritative graph edges. These reusable
    // union/find work arrays are a topology-change lowering of the source
    // graph's connected components; they do NOT implement source node
    // activation, incremental lost-edge repair, or sleeping.
    std::vector<BodyId> speculative_island_parent_;
    std::vector<BodyId> accurate_island_parent_;
    std::vector<std::uint8_t> speculative_island_rank_;
    std::vector<std::uint8_t> accurate_island_rank_;
    // Dy::DynamicsContext::iterateIslandsNodes consumes the accurate
    // active-island node sequence. These are existing actor identifiers,
    // not a duplicated actor or solver-body registry.
    std::vector<BodyId> accurate_island_nodes_;
    std::vector<std::size_t> accurate_island_offsets_;
    // Reused route-search scratch for IslandSim::findRoute on lost
    // dynamic edges. Only authoritative published pairs/contacts supply
    // edges; this does not maintain a competing graph or actor registry.
    std::vector<std::size_t> island_route_head_, island_route_next_;
    std::vector<BodyId> island_route_other_, island_route_queue_;
    std::vector<std::uint32_t> island_route_seen_;
    std::size_t speculative_island_count_ = 0;
    std::size_t accurate_island_count_ = 0;
    std::size_t pre_third_pass_island_count_ = 0;
    std::uint64_t island_topology_revision_ = 0;
    std::uint64_t island_full_rebuild_count_ = 0;
    bool island_topology_dirty_ = true;
    // Capacity persists across frames; matching actor indices avoid a shadow
    // body-ID registry. Only solver deltas are published back to actors.
    std::vector<SolverBodyRow> solver_rows_;
    // Reusable flat Dy normal-constraint stream; patch ownership and all
    // point offsets remain with this scene, not with allocating manifolds.
    std::vector<SolverContactPoint> solver_contact_points_;
    // DySolverConstraints.cpp::writeBackContact output is indexed by the
    // same authoritative Gu contact offset, irrespective of Dy ordering.
    // Capacity persists, while output is explicitly invalidated on a
    // collision-only advancement.
    std::vector<double> normal_impulse_writeback_;
    std::vector<FrictionPatchWriteback> friction_impulse_writeback_;
    // DyThresholdTable::build and PxsForceThresholdTask's previous/current
    // exceeded-force streams and mapped change stream. Reused scene-owned
    // storage; BodyIds are stable within this scene's lifetime.
    struct ThresholdPair {
        ForceThresholdKey key{};
        double impulse=0.0;
        float threshold=0.0f;
    };
    // Reused authoritative-state checkpoint for the strong step failure
    // boundary. Transient solver rows/descriptors and scratch are rebuilt
    // on retry; only observable state and persistent contact history are
    // saved. This is a journal, not an independent scene or solver.
    struct StepCheckpoint {
        struct Motion {
            BodyId id=0;
            Vec3 position{}, velocity{}, angular_velocity{};
            Quat orientation{};
            Vec3 pending_angular_velocity_delta{};
        };
        // Only dynamic motion is mutated by a step. Static geometry,
        // material ownership and immutable mass properties stay in actors_.
        std::vector<Motion> motions;
          std::vector<FrictionPatch> friction_patches;
        std::vector<double> normal_impulses;
        std::vector<FrictionPatchWriteback> friction_impulses;
        std::vector<ThresholdPair> threshold_previous;
        std::vector<ForceThresholdReport> threshold_changes;
        std::vector<std::size_t> partition_ends;
        std::vector<BodyId> speculative_parent, accurate_parent;
        std::vector<std::uint8_t> speculative_rank, accurate_rank;
        std::vector<BodyId> accurate_nodes;
        std::vector<std::size_t> accurate_offsets;
        std::size_t speculative_count=0, accurate_count=0;
        std::size_t pre_third_pass_count=0;
        std::uint64_t topology_revision=0, frame_index=0;
        std::uint64_t full_rebuild_count=0;
        std::uint64_t pair_epoch=0, contact_epoch=0;
        bool topology_dirty=true, dirty=true;
        void capture(Scene& scene);
        void rollback(Scene& scene) noexcept;
    };
    StepCheckpoint step_checkpoint_;
    std::vector<ThresholdPair> threshold_current_;
    std::vector<ThresholdPair> threshold_previous_;
    std::vector<ForceThresholdReport> threshold_changes_;
    // DyConstraintPartition::partitionContactConstraints writes ordered
    // contact-manager descriptors grouped by body-conflict-free partition.
    // Indices reference authoritative manifolds, not copied constraints.
    // Source counts/prefix offsets and scratch remain allocation-reused.
    std::vector<std::size_t> constraint_partitions_;
    // Source overflow descriptor array, compacted in place each 32-bit
    // partition wave; previously assigned managers are never rescanned.
    std::vector<std::size_t> pending_dynamic_constraints_;
    std::vector<std::size_t> ordered_constraints_;
    std::vector<std::size_t> partition_ends_;
    std::vector<std::size_t> partition_cursors_;
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
    // Publish swaps are reversible without copying the source-owned
    // contact/pair arenas. Rollback keeps externally held Gu spans valid.
    std::uint64_t pair_epoch_=0;
    std::uint64_t contact_epoch_=0;
    std::vector<Pair> found_;
    std::vector<Pair> lost_;
    std::vector<Pair> next_found_, next_lost_;

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
    std::vector<Pair> next_found_contacts_, next_lost_contacts_;
    std::uint64_t frame_index_ = 0;
    bool dirty_ = true;
};

} // namespace fyzz::native
