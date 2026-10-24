// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/world.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace fyzz::native {

// The exact rigid friction saturation expression from
// DySolverConstraints.cpp::{solveContact,solveContact_BStatic}.
// The original FMin(maxDyn,FMax(-maxDyn,candidate)) is defined even
// when a compliant spring makes maxDyn negative. This belongs to the
// production solver's source math, not a second test-only solver.
inline double source_contact_friction_limit(
    double candidate,double max_dynamic,bool saturated) noexcept {
    return saturated
        ? std::min(max_dynamic,std::max(-max_dynamic,candidate))
        : candidate;
}

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
    // PhysX contact manager owns this stable island EdgeIndex; the Gu
    // manifold is simply its current generation of contact geometry.
    std::uint32_t edge_handle=std::numeric_limits<std::uint32_t>::max();
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
    // Dy descriptor offsets apply only to the preparation epoch in
    // which this Gu manager was admitted through an active island.
    std::uint64_t prepared_friction_epoch = 0;
};

// Source translation: GuContactBoxBox's SAT and contact generation on
// caller-owned contact storage, preserving per-pair axis coherence.
std::uint8_t source_box_box_contacts(
    Vec3 half0, Vec3 position0, Quat rotation0,
    Vec3 half1, Vec3 position1, Quat rotation1,
    float contact_distance, std::uint8_t& previous_axis,
    std::span<ContactPoint> output);

class Scene final {
public:
    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = default;
    Scene& operator=(Scene&&) = default;
    // Configured only at World construction, before scene admission.
    void set_contact_notifications(bool enabled) noexcept {
        contact_notifications_=enabled;
    }
    // Configured at scene creation, before any native actors are admitted.
    // PhysX DyDynamicsContext::updatePostKinematic keeps each active
    // accurate island in its own solver task when this flag is enabled.
    void set_enhanced_determinism(bool enabled) noexcept {
        enhanced_determinism_=enabled;
    }
    std::span<const ContactEvent> contact_events() const noexcept {
        return contact_events_;
    }
    std::span<const ContactEventPoint> contact_event_points() const noexcept {
        return contact_event_points_;
    }

    BodyId add_static_box(Vec3 half_extents, Vec3 position);
    BodyId add_static_floor(float height);
    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density);
    void set_position(BodyId id, Vec3 position);
    void set_pose(BodyId id, Vec3 position, Quat orientation);
    void set_velocity(BodyId id, Vec3 velocity);
    void set_angular_velocity(BodyId id, Vec3 velocity);
    void apply_torque_impulse(BodyId id, Vec3 impulse);
    void add_spatial_velocity_mod(BodyId id, Vec3 value,
                                  ForceMode mode, bool angular);
    void clear_spatial_velocity_mod(BodyId id, ForceMode mode, bool angular);
    // A finite contact-report threshold opts a dynamic body into the
    // source's low-level force-threshold stream. Default PX_MAX_REAL
    // disables it. Units are force; evaluation uses impulse > force*dt.
    void set_contact_report_threshold(BodyId id, float force);
    // One noncompliant PxsMaterialData per currently admitted shape.
    // A material change is consumed by next native contact preparation.
    void set_material(BodyId id, Material material);
    void set_shape_offsets(BodyId id, float contact_offset, float rest_offset);
    // Source ScBodyCore per-rigid body unconstrained Dy inputs; no
    // separate solver owner or scene-global parameter shadow.
    void set_dynamics(BodyId id, RigidBodyDynamics dynamics);
    void remove(BodyId id);
    BodyState body_state(BodyId id) const;
    // ScBodyCore/PxsBodyCore's authoritative counter and the
    // Dy::sleepCheck readiness output. These are not yet island sleep
    // states: only the subsequent island deactivation stage may exclude
    // a node from simulation.
    float body_wake_counter(BodyId id) const {
        return require(id).wake_counter;
    }
    bool body_ready_for_sleep(BodyId id) const {
        return require(id).ready_for_sleep;
    }
    bool body_is_sleeping(BodyId id) const {
        return require(id).sleeping;
    }

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
    // PxsSimpleIslandManager's persistent contact-manager edge identity.
    // One source edge survives physical touch loss while SAP overlap lives.
    // The invalid sentinel is returned once the overlap is retired.
    std::uint32_t contact_edge_index(Pair pair) const noexcept;
    bool contact_edge_connected(Pair pair) const noexcept;
    bool speculative_contact_edge_active(Pair pair) const noexcept;
    bool accurate_contact_edge_active(Pair pair) const noexcept;
    // Sc::InteractionFlag::eIS_ACTIVE for the admitted rigid contact
    // manager, distinct from both IslandSim edge-state bitfields.
    bool contact_interaction_active(Pair pair) const noexcept;
    std::size_t active_contact_interaction_count() const noexcept {
        return active_contact_interaction_count_;
    }
    std::size_t active_speculative_contact_edge_count() const noexcept {
        return speculative_islands_.active_edge_count;
    }
    std::size_t active_accurate_contact_edge_count() const noexcept {
        return accurate_islands_.active_edge_count;
    }
    // PxsIslandSim::mActiveIslands: independently owned, dense active
    // handle lists, not another connectivity graph or actor collection.
    std::span<const std::uint32_t> active_speculative_islands() const noexcept {
        return speculative_islands_.active_islands;
    }
    std::span<const std::uint32_t> active_accurate_islands() const noexcept {
        return accurate_islands_.active_islands;
    }
    // IslandSim::Node::mActiveRefCount, only for pairs with two
    // dynamic endpoints. Each IslandSim owns its own reference history.
    std::uint32_t speculative_active_reference_count(BodyId id) const noexcept {
        return id<speculative_islands_.active_reference_counts.size()
            ? speculative_islands_.active_reference_counts[id] : 0;
    }
    std::uint32_t accurate_active_reference_count(BodyId id) const noexcept {
        return id<accurate_islands_.active_reference_counts.size()
            ? accurate_islands_.active_reference_counts[id] : 0;
    }
    std::size_t active_contact_edge_count() const noexcept {
        return active_edge_handles_.size();
    }
    std::span<const Pair> found_contacts() const noexcept { return found_contacts_; }
    std::span<const Pair> lost_contacts() const noexcept { return lost_contacts_; }
    std::uint64_t frame_index() const noexcept { return frame_index_; }
    // The last successfully published solver-result generation. A Gu
    // collision-only publication invalidates solved reports and resets
    // this marker; failed steps restore its original value atomically.
    // Zero means this Scene exposes no completed Dy result at present.
    std::uint64_t solved_result_frame() const noexcept {
        return solved_result_frame_;
    }
    // Read-only solver instrumentation for source constraint partitioning.
    // Counts last-step nonempty conflict-free partition batches; the
    // shared static world row is not a dynamic partition participant.
    std::size_t solver_partition_count() const noexcept {
        return partition_ends_.size();
    }
    std::span<const std::size_t> indexed_contact_manager_order() const noexcept {
        return indexed_managers_;
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
    // Source IslandSim persistent handle, independently owned by each
    // graph. Static/retired actors have no island.
    std::uint32_t speculative_island_id(BodyId id) const noexcept {
        return id<speculative_islands_.node_island.size()
            ? speculative_islands_.node_island[id] : IslandGraph::invalid;
    }
    std::uint32_t accurate_island_id(BodyId id) const noexcept {
        return id<accurate_islands_.node_island.size()
            ? accurate_islands_.node_island[id] : IslandGraph::invalid;
    }
    // Source IslandSim::Island::mRootNode: the surviving prior root
    // remains authoritative, while a split gets its failed-route
    // mVisitedNodes[0] root. Invalid IDs have no source island root.
    BodyId accurate_island_root(BodyId id) const noexcept {
        if(id>=accurate_islands_.node_island.size())
            return std::numeric_limits<BodyId>::max();
        const auto owner=accurate_islands_.node_island[id];
        return owner<accurate_islands_.islands.size() &&
               accurate_islands_.islands[owner].live
            ? accurate_islands_.islands[owner].root
            : std::numeric_limits<BodyId>::max();
    }
    // Source island static-contact counters used by the Dy::sleepCheck
    // stage. These differ between SAP candidates and Gu physical touches.
    std::size_t speculative_island_static_touches(BodyId id) const noexcept {
        if(id>=speculative_islands_.node_island.size())return 0;
        const auto h=speculative_islands_.node_island[id];
        return h<speculative_islands_.islands.size()
            ? speculative_islands_.islands[h].static_touches : 0;
    }
    std::size_t accurate_island_static_touches(BodyId id) const noexcept {
        if(id>=accurate_islands_.node_island.size())return 0;
        const auto h=accurate_islands_.node_island[id];
        return h<accurate_islands_.islands.size()
            ? accurate_islands_.islands[h].static_touches : 0;
    }
    std::uint64_t island_topology_revision() const noexcept {
        return island_topology_revision_;
    }
    // Internal source-island instrumentation: unlike the topology
    // revision, this advances only for complete component rebuilds.
    std::uint64_t island_full_rebuild_count() const noexcept {
        return island_full_rebuild_count_;
    }
    std::uint64_t island_local_split_count() const noexcept {
        return island_local_split_count_;
    }
    // Number of ownership snapshots, excluding source active-node
    // publication snapshots. A quiet step must leave this unchanged.
    std::uint64_t island_graph_checkpoint_count() const noexcept {
        return island_graph_checkpoint_count_;
    }
    std::size_t island_dirty_route_check_count() const noexcept {
        return island_dirty_route_checks_;
    }
    std::size_t island_fast_route_hit_count() const noexcept {
        return island_fast_route_hits_;
    }
    // Advisory source mHopCounts used by IslandSim::findRoute's
    // priority queue. UINT_MAX denotes an unestablished route.
    std::uint32_t accurate_cached_route_hops(BodyId id) const noexcept {
        return id<accurate_hops_.size() ? accurate_hops_[id] :
            std::numeric_limits<std::uint32_t>::max();
    }
    // Number of edited source edge instances saved for step rollback,
    // distinct from the number of historical contact-manager handles.
    std::size_t edge_rollback_journal_size() const noexcept {
        return step_checkpoint_.edge_undo.size();
    }
    // Number of actor motions recorded by the previous step checkpoint.
    std::size_t motion_rollback_journal_size() const noexcept {
        return step_checkpoint_.motions.size();
    }
    // Read the existing accurate-node batch boundaries without exposing
    // transient solver arrays or allocating diagnostic copies.
    // Dy::DynamicsContext::iterateIslandsNodes receives only currently
    // active accurate island batches. The indices reference the existing
    // source island-node sequence; there is no second actor registry.
    std::size_t active_accurate_island_count() const noexcept {
        return active_accurate_island_batches_.size();
    }
    // The independent speculative IslandSim owns its own rigid active
    // list, reverse indices and wakeIslands initial-active boundary.
    // These views are graph-owned; they do not duplicate actor state.
    std::span<const BodyId> active_speculative_nodes() const noexcept {
        return speculative_islands_.active_nodes;
    }
    std::span<const BodyId> newly_activated_speculative_nodes() const noexcept {
        return std::span<const BodyId>(speculative_islands_.active_nodes)
            .subspan(speculative_islands_.initial_active_node_count);
    }
    std::size_t speculative_active_node_index(BodyId id) const noexcept {
        return id<speculative_islands_.active_node_index.size()
            ? speculative_islands_.active_node_index[id] : invalid_instance;
    }
    // PxsIslandSim::mActiveNodes[eRIGID_BODY_TYPE] and
    // mActiveNodeIndex in the separate accurate graph. The invalid
    // index denotes an inactive node.
    std::span<const BodyId> active_accurate_nodes() const noexcept {
        return accurate_islands_.active_nodes;
    }
    // Source IslandSim::getNodesToActivate(eRIGID_BODY_TYPE) exposes
    // the suffix appended after wakeIslands captured its initial-active
    // count. This is an indexed view into the existing active nodes.
    std::span<const BodyId> newly_activated_accurate_nodes() const noexcept {
        return std::span<const BodyId>(accurate_islands_.active_nodes)
            .subspan(accurate_islands_.initial_active_node_count);
    }
    std::size_t accurate_active_node_index(BodyId id) const noexcept {
        return id<accurate_islands_.active_node_index.size()
            ? accurate_islands_.active_node_index[id] : invalid_instance;
    }
    // Source mActivatedEdges[eCONTACT_MANAGER] is the per-pass transition
    // list, not the set of all persistently active contact managers.
    std::span<const std::uint32_t>
    activated_speculative_contact_edges() const noexcept {
        return speculative_islands_.activated_edges;
    }
    std::span<const std::uint32_t>
    activated_accurate_contact_edges() const noexcept {
        return accurate_islands_.activated_edges;
    }
    // Third-pass IslandSim::mNodesToPutToSleep for admitted rigid nodes.
    // Distinct from Dy's prior-frame "ready" flags and from the
    // speculative graph's contact-interaction deactivation.
    std::span<const BodyId> deactivating_accurate_nodes() const noexcept {
        return accurate_islands_.nodes_to_deactivate;
    }
    // Source IslandSim::getDeactivatingEdges(eCONTACT_MANAGER).
    // Each graph owns its separate edge-lifetime publication; indices
    // reference the one scene-owned contact manager slot arena.
    std::span<const std::uint32_t>
    deactivating_speculative_contact_edges() const noexcept {
        return speculative_islands_.deactivating_edges;
    }
    std::span<const std::uint32_t>
    deactivating_accurate_contact_edges() const noexcept {
        return accurate_islands_.deactivating_edges;
    }
    std::size_t largest_accurate_island_size() const noexcept {
        std::size_t largest=0;
        for(std::size_t i=1;i<accurate_island_offsets_.size();++i)
            largest=std::max(largest,
                accurate_island_offsets_[i]-accurate_island_offsets_[i-1]);
        return largest;
    }

private:
    // Native conformance probes source Gu contact runs against the
    // actual Dy preparation state without introducing test-only public
    // simulation operations or a second friction-correlation path.
    friend struct ContactCorrelationSourceFixture;
    friend struct SolverIterationSourceFixture;
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
        // PxShapeCore owns both distances for the sole admitted shape.
        float contact_offset=0.02f;
        float rest_offset=0.0f;
        RigidBodyDynamics dynamics{};
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
        Vec3 pending_linear_velocity_delta{};
        Vec3 pending_linear_acceleration{};
        Vec3 pending_angular_acceleration{};
        // ScBodyCore::BodyCore initializes the PxsBodyCore wake counter
        // to 20 * 0.02 seconds; Dy::sleepCheck accumulates unscaled
        // motion energies until this source counter expires.
        float wake_counter=0.4f;
        Vec3 sleep_linear_acc{}, sleep_angular_acc{};
        bool ready_for_sleep=false;
        bool sleeping=false;
        // IslandSim::Node::eACTIVATING, set by a genuine authored
        // sleep-to-wake transition; consumed by wakeIslands after Gu
        // publication. Active nodes do not reawaken peers every frame.
        bool activating=false;
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
        std::uint64_t prepared_generation = 0;
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
        // DyContactPrepShared's compliant spring recurrence, applied on
        // every PGS normal iteration; always 1.0 for noncompliant rows.
        double impulse_multiplier = 1.0;
        // Extended only for SDK-admitted inverse masses below the range
        // of a representable PhysX float impulse. No float narrowing
        // between constraint preparation and the actual impulse update.
        double biased_error = 0.0;
        double unbiased_error = 0.0;
        double applied_force = 0.0;
        // DyContactPrepShared::extractContacts supplies the minimum
        // PxsBodyCore contact-impulse limit (overridden per-point only
        // by the source contact-modification branch, still unadmitted).
        double max_impulse = std::numeric_limits<float>::max();
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
        float damping=0.0f;
        bool acceleration_spring=false;
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
    void advance_collision_internal(bool force_update, bool checkpointed);
    void update_bounds();
    void update_pairs();
    void publish_pairs();
    void update_contacts();
    // Original contact-manager addEdge/setEdgeConnected/removeConnection
    // lifecycle, within the existing single scene-owned island pipeline.
    std::uint32_t allocate_contact_edge(Pair pair);
    void retire_destroyed_contact_edges();
    // IslandSim::addConnectionToGraph/removeConnectionInternal,
    // two source edge instances per contact manager, per graph.
    void link_contact_edge(std::uint32_t id, bool accurate);
    void unlink_contact_edge(std::uint32_t id, bool accurate);
    // PxsIslandSim's accurate island eCONTACT_MANAGER edge array:
    // append, concatenate on merge, and swap-last on retirement.
    void append_accurate_island_edge(std::uint32_t id, std::uint32_t island);
    void remove_accurate_island_edge(std::uint32_t id);
    void update_island_connectivity(bool repair_lost_edges);
    // A dependency-ready, synchronous lowering of third-pass lost-edge
    // repair. Source activation and whole-island deactivation consume
    // already published readiness before this frame's solver begins.
    // Concurrent continuations and non-rigid callback consumers
    // remain open. Contact-manager deactivation lists are active.
    void third_pass_island_gen();
    void deactivate_ready_islands() noexcept;
    // Serial ScPipeline::activateEdgesInternal/postThirdPassIslandGen
    // consumption for rigid eCONTACT_MANAGER, using source edge outputs.
    void consume_contact_interaction_activity() noexcept;
    void clear_activated_edges(bool checkpointed) noexcept;
    void activate_contact_edge(std::uint32_t id, bool accurate) noexcept;
    void deactivate_contact_edge(std::uint32_t id, bool accurate,
                                 bool publish) noexcept;
    void activate_node_contact_edges(BodyId node) noexcept;
    void synchronize_active_island_nodes(
        bool capture_speculative_order=false) noexcept;
    void prepare_active_accurate_island_batches();
    // IslandSim::wakeIslands lowering, shared by collision-only and
    // full-frame source publications. No duplicated activation owner.
    void activate_published_islands() noexcept;
    void queue_node_activation(BodyId id);
    bool lost_island_edges_retain_routes(bool accurate);
    void split_lost_island(bool accurate,BodyId seed);
    void prepare_solver_bodies();
    void partition_constraints();
    void prepare_constraints(float seconds);
    void erase_shape_friction_history(BodyId id);
    void solve_constraints();
    void write_back_contacts();
    void update_threshold_stream(float seconds);
    void publish_contact_events();
    void publish_solver_bodies();

    std::vector<Actor> actors_;
    using EdgeIndex=std::uint32_t;
    static constexpr EdgeIndex invalid_edge=std::numeric_limits<EdgeIndex>::max();
    static constexpr std::size_t invalid_instance=
        std::numeric_limits<std::size_t>::max();
    struct EdgeInstance {
        std::size_t next=invalid_instance;
        std::size_t prev=invalid_instance;
        bool linked=false;
    };
    struct ContactEdge {
        Pair pair{};                // PhysX mEdgeNodeIndices[2*EdgeIndex]
        bool live=false;            // allocated by addContactManager
        bool connected=false;       // source accurate mConnectedMap bit
        bool retiring=false;        // removeConnection, pending post-third-pass
        bool interaction_active=false; // Sc::InteractionFlag::eIS_ACTIVE
        // IG::Edge::eACTIVE/eACTIVATING belong to independent IslandSims.
        std::uint8_t speculative_state=0, accurate_state=0;
        // PhysX mEdgeInstances[2*EdgeIndex + endpoint] for the separate
        // speculative and accurate IslandSim graphs.
        std::array<EdgeInstance,2> speculative{};
        std::array<EdgeInstance,2> accurate{};
        // Ordered accurate-island manager membership. These are links in
        // the existing source EdgeIndex slot, not another contact owner.
        EdgeIndex island_prev=invalid_edge, island_next=invalid_edge;
    };
    // Source HandleManager-style reusable slots. Active handles align
    // with the one authoritative sorted SAP pair stream; the slots own
    // lifecycle and accurate touch state, not another collision graph.
    std::vector<ContactEdge> contact_edges_;
    // One aggregate count, derived only from transitions on the same
    // authoritative EdgeIndex contact-manager slots.
    std::size_t active_contact_interaction_count_=0;
    std::vector<EdgeIndex> active_edge_handles_, next_edge_handles_;
    std::vector<EdgeIndex> retired_edge_handles_, free_edge_handles_;
    // Per-node intrusive adjacency heads for the two original IslandSim
    // graphs. Static world endpoints never become dynamic island nodes.
    std::vector<std::size_t> speculative_node_first_, accurate_node_first_;
    // PxsIslandSim::IslandSim owns persistent island handles, a per-node
    // island ID and intrusive island-node membership. This *replaces*
    // the provisional union/find roots/ranks; the contact-manager graph
    // above remains the sole authoritative source for connectivity.
    // Every dynamic rigid node belongs to exactly one live island in
    // each speculative/accurate graph, including isolated nodes.
    struct IslandGraph {
        using IslandId=std::uint32_t;
        static constexpr IslandId invalid=std::numeric_limits<IslandId>::max();
        struct Island {
            BodyId root=std::numeric_limits<BodyId>::max();
            BodyId tail=std::numeric_limits<BodyId>::max();
            std::size_t count=0;
            bool live=false;
            // IslandSim::mIslandStaticTouchCount, updated on static-edge
            // found/lost and transferred on dynamic-island merges.
            std::size_t static_touches=0;
            // Per-island ordered eCONTACT_MANAGER edges. The indexed
            // descriptor stream reads these slots, not Gu pair order.
            EdgeIndex first_contact=invalid_edge, last_contact=invalid_edge;
            std::size_t contact_count=0;
            // mActiveIndex, invalid outside the source mActiveIslands list.
            std::size_t active_index=invalid_instance;
        };
        std::vector<Island> islands;
        std::vector<IslandId> active_islands;
        // Source mActivatingNodes order, independently published by
        // each IslandSim. This is a pending event stream, not a second
        // actor owner or a global sorted waking-body registry.
        std::vector<BodyId> activating_nodes;
        // Per-publication duplicate suppression for the source
        // mActivatingNodes stream. Stamped scratch, not another owner.
        std::vector<std::uint32_t> activating_seen;
        std::uint32_t activating_generation=0;
        // Reusable island-handle visit stamps for reconciling source
        // active-node transitions without walking dormant islands.
        std::vector<std::uint32_t> active_island_seen;
        std::uint32_t active_island_visit_stamp=0;
        std::vector<IslandId> node_island;
        std::vector<BodyId> node_next, node_prev;
        // IslandSim::Node::mStaticTouchCount for live dynamic nodes.
        std::vector<std::uint32_t> node_static_touches;
        std::vector<IslandId> free_handles;
        // IslandSim::mActiveNodes / mActiveNodeIndex (rigid type).
        // markInactive uses swap-last, keeping the map O(1) to update.
        // The node IDs are references into actors_, not copied bodies.
        std::vector<BodyId> active_nodes;
        std::vector<std::size_t> active_node_index;
        // IslandSim::mInitialActiveNodeCount[eRIGID_BODY_TYPE] splits
        // the original active prefix from the wakeIslands additions.
        // markInactive may shrink this prefix by relocating its tail.
        std::size_t initial_active_node_count=0;
        // Cleared at the next third pass; authoritative prior-frame
        // ready flags determine the members to deactivate.
        std::vector<BodyId> nodes_to_deactivate;
        // PxsIslandSim::deactivateNodeInternal emits each active
        // contact-manager edge when its last awake endpoint sleeps.
        // A lost/retired edge is not a sleep-deactivation event.
        std::vector<EdgeIndex> deactivating_edges;
        std::vector<EdgeIndex> activated_edges;
        std::size_t active_edge_count=0; // mActiveEdgeCount[eCONTACT_MANAGER]
        // IslandSim::Node::mActiveRefCount is incremented on
        // markEdgeActive and decremented on removeEdgeFromActivatingList
        // only if BOTH edge endpoints are non-static. These counts are
        // prerequisites for correct kinematic activation admission.
        std::vector<std::uint32_t> active_reference_counts;
    };
    IslandGraph speculative_islands_, accurate_islands_;
    // Bound the noexcept motion journal by living rigid actors rather
    // than the historical BodyId watermark that includes retired actors.
    std::size_t live_dynamic_count_=0;
    // Dy::DynamicsContext::iterateIslandsNodes consumes the accurate
    // active-island node sequence. These are existing actor identifiers,
    // not a duplicated actor or solver-body registry.
    std::vector<BodyId> accurate_island_nodes_;
    std::vector<std::size_t> accurate_island_offsets_;
    // Source IslandSim::mActiveIslands equivalent for the serial Dy
    // handoff: indexes into the authoritative accurate island offsets.
    // Reconstructed after the third-pass sleep decision, not a second
    // registry of actors, island handles, or solver bodies.
    std::vector<std::size_t> active_accurate_island_batches_;
    // IslandSim::findRoute traversal reuses its search queue/visited
    // marks, but now traverses the persistent EdgeInstance adjacency.
    std::vector<BodyId> island_route_queue_, island_route_predecessor_;
    // The same source EdgeIndex used to discover each BFS predecessor.
    std::vector<EdgeIndex> island_route_predecessor_edge_;
    std::vector<std::uint32_t> island_route_seen_;
    std::uint32_t island_route_visit_stamp_=0;
    // Per-third-pass processed mVisitedState witnesses. Visit generations
    // avoid clearing the entire actor vector for each dirty publication.
    std::vector<std::uint32_t> island_route_confirmed_;
    std::uint32_t island_route_confirmed_stamp_=0;
    // Rare rigid sleep publication: process each speculative island
    // exactly once when its final member sleeps. Indexed by root BodyId
    // to avoid an O(all islands) scan or post-commit allocations.
    std::vector<std::uint32_t> speculative_sleep_seen_;
    std::uint32_t speculative_sleep_visit_stamp_=0;
    // Transient source third-pass candidate order. Accurate node sleep
    // and speculative edge loss nominate islands, but only speculative
    // mActiveIndex determines their deactivation publication order.
    // Reserve in the throwing checkpoint before noexcept island work.
    struct SpeculativeSleepCandidate {
        IslandGraph::IslandId handle;
        std::size_t discovery;
    };
    std::vector<SpeculativeSleepCandidate> speculative_sleep_order_;
    // Saved at the wake/third-pass boundary, BEFORE the active-node
    // reconciliation can swap-last and retire speculative split handles.
    // This is generation-stamped scratch, not another island owner.
    struct SpeculativePreRetirementIndex {
        std::size_t index=invalid_instance;
        std::uint32_t generation=0;
    };
    std::vector<SpeculativePreRetirementIndex>
        speculative_pre_retirement_index_;
    std::uint32_t speculative_pre_retirement_generation_=0;
    // Source processLostEdges can make a formerly awake speculative
    // island entirely inactive merely by disconnecting its last awake
    // member. Retain both resulting roots as third-pass candidates;
    // no duplicate actor registry or speculative-wide sleep scan.
    std::vector<BodyId> speculative_split_sleep_candidates_;
    // IslandSim::mDirtyMap, mFastRoute and mHopCounts lowerings for the
    // admitted all-awake rigid-node graph. Fast routes are advisory and
    // validated against current intrusive adjacency on every use.
    std::vector<BodyId> speculative_dirty_nodes_, accurate_dirty_nodes_;
    // Source processLostEdges' failed dirty-route seeds, consumed by
    // local split migration in their original ascending dirty order.
    // Reused source mVisitedNodes and mIslandSplitEdges working sets.
    std::vector<BodyId> island_split_members_;
    std::vector<EdgeIndex> island_split_edges_;
    // IslandSim::mDirtyMap lowers to reusable per-graph membership bits.
    // Reset only touched nodes after a pass or a failed publication.
    std::vector<std::uint8_t> speculative_dirty_map_, accurate_dirty_map_;
    std::vector<BodyId> speculative_fast_route_, accurate_fast_route_;
    // Each cached hop refers directly to its source contact-manager edge.
    // Fast-path validation checks the slot and both linked instances,
    // never rescanning a high-degree node's intrusive adjacency.
    std::vector<EdgeIndex> speculative_fast_edge_, accurate_fast_edge_;
    std::vector<std::uint32_t> speculative_hops_, accurate_hops_;
    std::vector<std::uint64_t> speculative_hint_stamps_, accurate_hint_stamps_;
    std::uint64_t speculative_hint_epoch_=1, accurate_hint_epoch_=1;
    std::size_t island_dirty_route_checks_=0, island_fast_route_hits_=0;
    std::size_t speculative_island_count_ = 0;
    std::size_t accurate_island_count_ = 0;
    std::uint64_t island_topology_revision_ = 0;
    std::uint64_t island_full_rebuild_count_ = 0;
    std::uint64_t island_local_split_count_ = 0;
    std::uint64_t island_graph_checkpoint_count_ = 0;
    bool island_topology_dirty_ = true;
    // Capacity persists across frames; matching actor indices avoid a shadow
    // body-ID registry. Only solver deltas are published back to actors.
    std::vector<SolverBodyRow> solver_rows_;
    // A failed frame restores the published frame index but deliberately
    // retains scratch rows. Unique preparation generations prevent those
    // rejected rows from being mistaken for current contact participants.
    // This is transient Dy state, not a second scene publication clock.
    std::uint64_t solver_preparation_generation_=0;
    // Reusable flat Dy normal-constraint stream; patch ownership and all
    // point offsets remain with this scene, not with allocating manifolds.
    std::vector<SolverContactPoint> solver_contact_points_;
    // DyFrictionCorrelation::ContactPatchData is constructed from
    // CONTIGUOUS runs of compatible Gu contacts before friction patch
    // correlation; one reused scratch arena, not per-manager objects.
    struct ContactPatchData {
        std::size_t start=0, count=0, next=invalid_instance;
        Vec3 normal{}, bounds_min{}, bounds_max{};
    };
    std::vector<ContactPatchData> contact_patch_scratch_;
    // The source correlates each contact run with a friction patch via
    // a reverse-linked list. Its ordered solver contact stream retains
    // global Gu row IDs, preserving original Gu-indexed writeback.
    std::vector<std::size_t> solver_contact_order_;
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
            Vec3 pending_linear_velocity_delta{};
            Vec3 pending_linear_acceleration{};
            Vec3 pending_angular_acceleration{};
            float wake_counter=0.4f;
            Vec3 sleep_linear_acc{}, sleep_angular_acc{};
            bool ready_for_sleep=false;
            bool sleeping=false;
            bool activating=false;
        };
        // Only dynamic motion is mutated by a step. Static geometry,
        // material ownership and immutable mass properties stay in actors_.
        std::vector<Motion> motions;
        // Sparse undo records for intrusive source EdgeInstances and
        // per-node list heads. Replayed in reverse order on failure;
        // unchanged or historically allocated slots incur no copies.
        struct EdgeUndo { EdgeIndex id; ContactEdge previous; };
        struct HintUndo {
            BodyId node;
            BodyId next;
            EdgeIndex edge;
            std::uint32_t hops;
            std::uint64_t stamp;
            bool accurate;
        };
        struct HeadUndo {
            BodyId node;
            bool accurate;
            std::size_t previous;
        };
        std::vector<EdgeUndo> edge_undo;
        struct ActivityUndo {
            EdgeIndex id;
            std::uint8_t speculative, accurate;
            bool interaction_active;
        };
        std::vector<ActivityUndo> activity_undo;
        // Per-node sparse rollback for source active references. The
        // generation journal saves a dynamic endpoint at most once per
        // frame, avoiding two O(actor_count) copies on quiet PGS steps.
        struct ReferenceUndo {
            BodyId node;
            std::uint32_t speculative, accurate;
        };
        std::vector<ReferenceUndo> reference_undo;
        std::size_t initial_speculative_references=0;
        std::size_t initial_accurate_references=0;
        std::vector<HintUndo> hint_undo;
        std::vector<HeadUndo> head_undo;
        std::size_t initial_edges=0;
        std::size_t initial_active_interactions=0;
        std::size_t initial_speculative_hints=0, initial_accurate_hints=0;
        std::uint64_t speculative_hint_epoch=1, accurate_hint_epoch=1;
        std::size_t initial_speculative_heads=0;
        std::size_t initial_accurate_heads=0;
        std::vector<EdgeIndex> active_handles;
        std::vector<EdgeIndex> retired_handles, free_handles;
        bool handles_saved=false;
        // These prior caches remain in scene-owned scratch on rotation.
        std::uint64_t friction_cache_epoch=0, threshold_cache_epoch=0;
        // A quiet step reuses published Gu manifold storage. Save only
        // descriptor slots actually overwritten by Dy preparation or
        // writeback; dirty Gu republication is undone by arena swap.
        struct ManifoldSolverMetadata {
            std::size_t index=0;
            std::size_t solver_offset=0;
            std::size_t friction_patch_offset=0;
            std::uint8_t friction_patch_count=0;
            std::uint64_t prepared_friction_epoch=0;
        };
        std::vector<ManifoldSolverMetadata> quiet_manifold_metadata;
        bool quiet_manifold_undo_enabled=false;
        std::vector<double> normal_impulses;
        std::vector<FrictionPatchWriteback> friction_impulses;
        std::vector<ForceThresholdReport> threshold_changes;
        std::vector<std::size_t> partition_ends;
        // Source graph membership is snapshotted only before its first
        // mutation, never on a quiet step. Active-node publication
        // remains a separate per-frame rollback concern.
        IslandGraph speculative_islands, accurate_islands;
        bool graph_ownership_saved=false;
        std::vector<BodyId> accurate_nodes;
        std::vector<std::size_t> accurate_offsets;
        // Swap-only checkpoint for the transient active Dy batch view:
        // no O(active islands) per-frame copying or undo allocation.
        std::vector<std::size_t> active_accurate_batches;
        std::vector<std::size_t> indexed_managers;
        std::size_t speculative_count=0, accurate_count=0;
        std::uint64_t topology_revision=0, frame_index=0;
        std::uint64_t solved_result_frame=0;
        std::uint64_t full_rebuild_count=0;
        std::uint64_t local_split_count=0;
        std::uint64_t graph_checkpoint_count=0;
        std::size_t dirty_route_checks=0, fast_route_hits=0;
        std::uint64_t pair_epoch=0, contact_epoch=0;
        std::uint64_t manifold_lookup_generation=0;
        std::uint64_t quiet_event_epoch=0;
        bool topology_dirty=true, dirty=true;
        bool authored_activation_pending=false;
        bool includes_solver_state=true;
        void capture(Scene& scene, bool capture_solver_state=true);
        void save_motion(Scene& scene, BodyId id) noexcept;
        void save_manifold_solver_metadata(Scene& scene,
                                           std::size_t index);
        void save_edge(const Scene& scene, EdgeIndex id);
        void save_activity(Scene& scene, EdgeIndex id) noexcept;
        void save_reference(Scene& scene, BodyId node) noexcept;
        void save_hint(const Scene& scene, BodyId node, bool accurate);
        void save_head(const Scene& scene, BodyId node, bool accurate);
        void save_handle_lists(const Scene& scene);
        void save_graph_ownership(Scene& scene);
        void rollback(Scene& scene) noexcept;
    };
    StepCheckpoint step_checkpoint_;
    std::vector<std::uint64_t> motion_undo_generations_;
    std::uint64_t motion_undo_generation_=0;
    std::vector<std::uint64_t> manifold_undo_generations_;
    std::uint64_t manifold_undo_generation_=0;
    // Only changed EdgeIndex states are journaled for frame rollback.
    std::vector<std::uint64_t> activity_undo_generations_;
    std::uint64_t activity_undo_generation_=0;
    std::vector<std::uint64_t> reference_undo_generations_;
    std::uint64_t reference_undo_generation_=0;
    // Only an in-progress failure-atomic publication may journal activity.
    // Quiet collision queries mutate no rollback state and remain noexcept.
    bool activity_checkpoint_active_=false;
    std::vector<ThresholdPair> threshold_current_;
    std::vector<ThresholdPair> threshold_previous_;
    std::uint64_t threshold_cache_epoch_=0;
    std::vector<ForceThresholdReport> threshold_changes_;
    // DyConstraintPartition::partitionContactConstraints writes ordered
    // contact-manager descriptors grouped by body-conflict-free partition.
    // Indices reference authoritative manifolds, not copied constraints.
    // Source counts/prefix offsets and scratch remain allocation-reused.
    std::vector<std::size_t> constraint_partitions_;
    // Partition membership is valid for precisely one source task
    // dispatch. Stamp only active Gu manager slots so dormant retained
    // manifolds do not cost an O(all contacts) sentinel reset each step.
    std::vector<std::uint64_t> constraint_partition_stamps_;
    std::uint64_t constraint_partition_generation_=0;
    // Source overflow descriptor array, compacted in place each 32-bit
    // partition wave; previously assigned managers are never rescanned.
    std::vector<std::size_t> pending_dynamic_constraints_;
    // DyDynamicsBase::iterateIslandsContactEdges source-indexed manager
    // sequence, expressed as indices into the authoritative Gu manifolds.
    // Both the constraint partitioner and its stable output consume it.
    std::vector<std::size_t> indexed_managers_;
    // DyDynamicsContext::updatePostKinematic's contiguous source task
    // groups: cumulative offsets into indexed_managers_. The admitted
    // rigid-only default batches 128 bodies or until one contact exists.
    // Scratch boundaries only; source islands and Gu managers own data.
    // Serial lowering of DyDynamicsContext task chains. Iteration counts
    // come from the MAXIMUM of each chain's active PxsBodyCore counts,
    // and are never promoted to a scene-global maximum.
    struct SolverTask {
        std::size_t first_batch=0, batch_end=0;
        std::size_t manager_end=0, partition_end=0;
        std::size_t constraint_end=0, solver_body_end=0;
        std::uint8_t position_iterations=0, velocity_iterations=0;
    };
    std::vector<SolverTask> solver_tasks_;
    struct IndexedManifold {
        std::size_t index=invalid_instance;
        std::uint64_t generation=0;
    };
    // The live Gu manager-to-island-edge index is rebuilt ONLY when
    // narrowphase publishes a new contact arena. An unchanged geometry
    // epoch reuses it across all PGS frames, including dormant scenes.
    // Staging has its own buffer so a rejected contact publication
    // restores both the Gu arena and this index with reversible swaps.
    std::vector<IndexedManifold> manifold_by_edge_;
    std::vector<IndexedManifold> next_manifold_by_edge_;
    std::uint64_t indexed_manifold_generation_=0;
    std::uint64_t manifold_lookup_generation_=0;
    std::vector<std::size_t> ordered_constraints_;
    std::vector<std::size_t> partition_ends_;
    std::vector<std::size_t> partition_cursors_;
    // Correlation and friction constraint streams retain capacity across
    // frames, while patches carry actor-local anchors across contact frames.
    std::vector<FrictionPatch> friction_patches_;
    std::vector<FrictionPatch> next_friction_patches_;
    // The original PxcNp manager is the friction-history owner. Rotate
    // this bounded span index together with the source friction arena.
    struct FrictionManagerSpan {
        Pair pair{};
        std::size_t offset=0;
        std::uint8_t count=0;
        std::uint64_t epoch=0;
    };
    std::vector<FrictionManagerSpan> friction_by_edge_;
    std::vector<FrictionManagerSpan> next_friction_by_edge_;
    std::uint64_t friction_cache_epoch_=0;
    std::vector<SolverContactFriction> solver_friction_rows_;
    // DyDynamics::mWorldSolverBody / mWorldSolverBodyData: one zero-mass
    // solver body shared by all static collision participants.
    SolverBodyRow world_solver_row_{};
    // Prepared contact bodies for the current frame; avoids re-preparing
    // free-flight actors before the original island owner is integrated.
    std::vector<BodyId> solver_active_ids_;
    // Sc::BodyCore::numCountedInteractions, derived once from the
    // authoritative contact manager stream for Dy::sleepCheck.
    std::vector<std::uint32_t> sleep_interaction_counts_;
    std::vector<Endpoint> endpoints_;
    std::vector<BodyId> active_;
    std::vector<Pair> previous_pairs_;
    std::vector<Pair> pending_pairs_;
    // Publish swaps are reversible without copying the source-owned
    // contact/pair arenas. Rollback keeps externally held Gu spans valid.
    std::uint64_t pair_epoch_=0;
    std::uint64_t contact_epoch_=0;
    // Quiet publications rotate only Found/Lost event buffers, not the
    // authoritative pair/Gu arenas. Keep an independent rollback marker.
    std::uint64_t quiet_event_epoch_=0;
    std::vector<Pair> found_;
    std::vector<Pair> lost_;
    std::vector<Pair> next_found_, next_lost_;

    // Source ScNPhaseCore notification events and their current Gu contact
    // point windows are doubly buffered: a rejected Scene::step never
    // overwrites the previous externally observed event storage.
    // Event descriptors carry point offsets, not pointers to scratch.
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
    // Source ScNPhaseCore touch events are emitted only if contact
    // notifications were requested when the World was constructed.
    bool contact_notifications_=false;
    bool enhanced_determinism_=false;
    std::vector<ContactEvent> contact_events_;
    std::vector<ContactEvent> next_contact_events_;
    // Non-owning Gu-manifold indices for this frame's actual report
    // producers (active Dy managers and newly found physical touches).
    // Sorted by original Gu index before Sc report publication.
    std::vector<std::size_t> contact_report_managers_;
    std::vector<ContactEventPoint> contact_event_points_;
    std::vector<ContactEventPoint> next_contact_event_points_;
    std::uint64_t frame_index_ = 0;
    // Scene-owned result visibility marker, separate from Gu collision
    // progression and written only at the successful solver/fetch join.
    std::uint64_t solved_result_frame_=0;
    // One source wake-work indicator for authored transitions. A quiet
    // geometry query with no found edges avoids scanning all island
    // slots; active node flags remain on the authoritative actors.
    bool authored_activation_pending_=false;
    bool dirty_ = true;
};

} // namespace fyzz::native
