// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace fyzz {
namespace {

physx::PxVec3 to_physx(Vec3 value) {
    return physx::PxVec3(value.x, value.y, value.z);
}

Vec3 from_physx(const physx::PxVec3& value) {
    return {value.x, value.y, value.z};
}
Quat from_physx(const physx::PxQuat& value) {
    return {value.x, value.y, value.z, value.w};
}

// PhysX Foundation and Physics are process-level services, not individual
// World resources. Hold them as long as any CPU World needs them.
std::mutex& runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct PhysXRuntime {
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback errors;
    physx::PxFoundation* foundation = nullptr;
    physx::PxPhysics* physics = nullptr;

    PhysXRuntime() {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        if (!foundation) {
            throw std::runtime_error("Failed to create PhysX foundation");
        }
        try {
            physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
                                     physx::PxTolerancesScale(), false, nullptr);
            if (!physics) {
                throw std::runtime_error("Failed to create PhysX physics");
            }
        } catch (...) {
            foundation->release();
            foundation = nullptr;
            throw;
        }
    }

    ~PhysXRuntime() {
        // Prevent another CPU world from recreating PhysX while its previous
        // singleton is in the middle of releasing.
        std::lock_guard<std::mutex> guard(runtime_mutex());
        if (physics) physics->release();
        if (foundation) foundation->release();
    }
};

std::shared_ptr<PhysXRuntime> acquire_runtime() {
    static std::weak_ptr<PhysXRuntime> cached;
    std::lock_guard<std::mutex> guard(runtime_mutex());
    if (auto existing = cached.lock()) {
        return existing;
    }
    auto created = std::make_shared<PhysXRuntime>();
    cached = created;
    return created;
}

} // namespace

// Preserve PxDefaultSimulationFilterShader's group policy and only
// request original Sc contact notifications for surviving rigid pairs.
physx::PxFilterFlags source_touch_reporting_filter(
    physx::PxFilterObjectAttributes attributes0,
    physx::PxFilterData data0,
    physx::PxFilterObjectAttributes attributes1,
    physx::PxFilterData data1,
    physx::PxPairFlags& pair_flags,
    const void* constant_block, physx::PxU32 constant_block_size) {
    const auto filter=physx::PxDefaultSimulationFilterShader(
        attributes0,data0,attributes1,data1,pair_flags,
        constant_block,constant_block_size);
    if(!(filter & physx::PxFilterFlag::eSUPPRESS) &&
       !physx::PxFilterObjectIsTrigger(attributes0) &&
       !physx::PxFilterObjectIsTrigger(attributes1)) {
        pair_flags |= physx::PxPairFlag::eNOTIFY_TOUCH_FOUND |
                      physx::PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                      physx::PxPairFlag::eNOTIFY_TOUCH_LOST |
                      physx::PxPairFlag::eNOTIFY_CONTACT_POINTS;
    }
    return filter;
}

struct PhysXCpuWorld final : WorldBackend, physx::PxSimulationEventCallback {
    std::shared_ptr<PhysXRuntime> runtime;
    physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
    physx::PxScene* scene = nullptr;
    physx::PxMaterial* material = nullptr;
    std::vector<physx::PxRigidActor*> actors;
    // Deleted PhysX callback pointers are address tokens ONLY:
    // a removed actor must never be dereferenced after PxActor::release.
    struct RetiredActorToken {
        const physx::PxActor* address=nullptr;
        std::uintptr_t token=0;
    };
    std::vector<RetiredActorToken> retired_actor_tokens;
    bool contact_notifications=false;
    bool enhanced_determinism=false;
    std::vector<ContactEvent> published_contact_events;
    std::vector<ContactEvent> pending_contact_events;
    std::vector<ContactEventPoint> published_contact_points;
    std::vector<ContactEventPoint> pending_contact_points;
    // A C++ allocation/validation failure must never unwind through
    // PhysX's C++ task/callback frames. Surface it after fetchResults.
    std::exception_ptr callback_error;

    // Sc/NPhaseCore reports callbacks at PxScene::fetchResults on the
    // user's thread. The authoring-time userData token maps directly
    // back to the already-existing World actor ID without another
    // reverse-lookup registry or per-contact O(actor count) search.
    void onContact(const physx::PxContactPairHeader& header,
                   const physx::PxContactPair* pairs,
                   physx::PxU32 count) override {
        if(callback_error)return;
        try {
        if(!header.actors[0] || !header.actors[1])return;
        // PxContactPairHeaderFlag::eREMOVED_ACTOR_* forbids actor
        // dereference. Match the captured pointer address to the ID
        // token recorded before release. Live actors take the direct
        // zero-search userData path.
        const auto token_for=[&](unsigned index) -> std::uintptr_t {
            const bool removed=header.flags.isSet(index==0
                ? physx::PxContactPairHeaderFlag::eREMOVED_ACTOR_0
                : physx::PxContactPairHeaderFlag::eREMOVED_ACTOR_1);
            if(!removed)
                return reinterpret_cast<std::uintptr_t>(
                    header.actors[index]->userData);
            for(const auto& retired:retired_actor_tokens)
                if(retired.address==header.actors[index])
                    return retired.token;
            return 0;
        };
        const auto raw0=token_for(0);
        const auto raw1=token_for(1);
        if(!raw0 || !raw1)return;
        const BodyId a=static_cast<BodyId>(std::min(raw0,raw1)-1);
        const BodyId b=static_cast<BodyId>(std::max(raw0,raw1)-1);
        // PxContactPairPoint::normal is directed from shape1 toward
        // shape0. When the original callback actor pair order differs
        // from our canonical BodyId order, invert the normal together
        // with the actor ordering, not the normal force magnitude.
        const float normal_sign=raw0<=raw1 ? 1.0f : -1.0f;
        for(physx::PxU32 p=0;p<count;++p) {
            const auto& pair=pairs[p];
            const bool found=pair.events.isSet(
                physx::PxPairFlag::eNOTIFY_TOUCH_FOUND);
            const bool lost=pair.events.isSet(
                physx::PxPairFlag::eNOTIFY_TOUCH_LOST);
            const bool persists=pair.events.isSet(
                physx::PxPairFlag::eNOTIFY_TOUCH_PERSISTS);
            if(!found && !lost && !persists)continue;
            double impulse=0.0;
            const std::uint8_t n=lost ? 0 : pair.contactCount;
            const auto offset=pending_contact_points.size();
            if(n>0) {
                // Original PxContactPair::extractContacts returns
                // source PxContactPairPoint position/normal/separation.
                // Contact force-buffer slots stay Gu-indexed; the
                // summed scalar below is not a fictitious manifold cap.
                std::array<physx::PxContactPairPoint,255> extracted{};
                const auto count=pair.extractContacts(extracted.data(),
                    static_cast<physx::PxU32>(extracted.size()));
                if(count!=n)
                    throw std::logic_error(
                        "Original PxContactPair contact count exceeds admitted Gu buffer");
                for(std::size_t i=0;i<count;++i) {
                    const double normal_force=pair.contactImpulses
                        ? static_cast<double>(pair.contactImpulses[i]):0.0;
                    impulse+=normal_force;
                    const Vec3 normal=from_physx(extracted[i].normal);
                    pending_contact_points.push_back({
                        from_physx(extracted[i].position),
                        {normal.x*normal_sign,normal.y*normal_sign,
                         normal.z*normal_sign},
                        extracted[i].separation,normal_force});
                }
            }
            pending_contact_events.push_back({
                a,b,found?ContactEventKind::Found:
                    lost?ContactEventKind::Lost:ContactEventKind::Persists,
                n,offset,impulse});
        }
        } catch (...) {
            callback_error=std::current_exception();
        }
    }
    void onConstraintBreak(physx::PxConstraintInfo*,physx::PxU32) override {}
    void onWake(physx::PxActor**,physx::PxU32) override {}
    void onSleep(physx::PxActor**,physx::PxU32) override {}
    void onTrigger(physx::PxTriggerPair*,physx::PxU32) override {}
    void onAdvance(const physx::PxRigidBody*const*,
                   const physx::PxTransform*,physx::PxU32) override {}

    std::span<const ContactEvent> contact_events() const override {
        return published_contact_events;
    }
    std::span<const ContactEventPoint> contact_event_points() const override {
        return published_contact_points;
    }

    void initialize(bool legacy_gu) {
        runtime = acquire_runtime();
        physx::PxSceneDesc descriptor(runtime->physics->getTolerancesScale());
        descriptor.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        dispatcher = physx::PxDefaultCpuDispatcherCreate(2);
        if (!dispatcher) {
            throw std::runtime_error("Failed to create PhysX CPU dispatcher");
        }
        descriptor.cpuDispatcher = dispatcher;
        descriptor.filterShader = contact_notifications
            ? source_touch_reporting_filter
            : physx::PxDefaultSimulationFilterShader;
        descriptor.simulationEventCallback = contact_notifications ? this : nullptr;
        // Keep stock PhysX PCM in ReferencePhysXCpu. The explicit Gu
        // conformance mode selects original source contactBoxBox/PlaneBox
        // rather than silently changing the default reference behavior.
        if (legacy_gu)
            descriptor.flags &= ~physx::PxSceneFlag::eENABLE_PCM;
        if (enhanced_determinism)
            descriptor.flags |=
                physx::PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
        // The CPU execution mode is explicit. Do not ever request eGPU
        // here: pinned ScScene.cpp converts eGPU to CPU eABP when the
        // CUDA GPU factory is unavailable, making the request look valid.
        descriptor.broadPhaseType = physx::PxBroadPhaseType::ePABP;
        scene = runtime->physics->createScene(descriptor);
        if (!scene) {
            throw std::runtime_error("Failed to create PhysX scene");
        }
        if (scene->getBroadPhaseType() != descriptor.broadPhaseType) {
            throw std::runtime_error(
                "PhysX scene did not admit the requested CPU broadphase");
        }
        // The two reference oracles must stay observably distinct.
        // Verify the admitted PxScene flags, not only the descriptor:
        // silently routing all reference worlds through legacy Gu would
        // make default PCM conformance tests nondiscriminating.
        if (scene->getFlags().isSet(physx::PxSceneFlag::eENABLE_PCM)
            == legacy_gu) {
            throw std::runtime_error(
                "PhysX scene admitted the wrong PCM contact mode");
        }
        if (scene->getFlags().isSet(
                physx::PxSceneFlag::eENABLE_ENHANCED_DETERMINISM)
            != enhanced_determinism)
            throw std::runtime_error(
                "PhysX scene admitted the wrong enhanced-determinism mode");

        material = runtime->physics->createMaterial(0.5f, 0.5f, 0.2f);
        if (!material) {
            throw std::runtime_error("Failed to create PhysX material");
        }
    }

    void cleanup() noexcept {
        for (physx::PxRigidActor* actor : actors)
            if (actor) actor->release();
        actors.clear();
        retired_actor_tokens.clear();
        if (material) {
            material->release();
            material = nullptr;
        }
        if (scene) {
            scene->release();
            scene = nullptr;
        }
        if (dispatcher) {
            dispatcher->release();
            dispatcher = nullptr;
        }
        // The runtime shared_ptr releases after per-world resources are gone.
    }

    ~PhysXCpuWorld() override { cleanup(); }

    explicit PhysXCpuWorld(bool legacy_gu,bool notify,
                          bool enhanced)
        :contact_notifications(notify),enhanced_determinism(enhanced) {
        try {
            initialize(legacy_gu);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    BodyId add_static_box(Vec3 half_extents, Vec3 position) override {
        return add(physx::PxCreateStatic(
            *runtime->physics, physx::PxTransform(to_physx(position)),
            physx::PxBoxGeometry(to_physx(half_extents)), *material));
    }

    BodyId add_static_floor(float height) override {
        // Plane equation y - height = 0, facing +Y.
        return add(physx::PxCreatePlane(
            *runtime->physics, physx::PxPlane(0.0f, 1.0f, 0.0f, -height), *material));
    }

    BodyId add_dynamic_box(Vec3 half_extents, Vec3 position, float density) override {
        return add(physx::PxCreateDynamic(
            *runtime->physics, physx::PxTransform(to_physx(position)),
            physx::PxBoxGeometry(to_physx(half_extents)), *material, density));
    }

    void step(float seconds) override {
        if(contact_notifications) {
            callback_error=nullptr;
            pending_contact_events.clear();
            pending_contact_points.clear();
        }
        scene->simulate(seconds);
        if (!scene->fetchResults(true)) {
            throw std::runtime_error("PhysX simulation failed");
        }
        if(callback_error)
            std::rethrow_exception(callback_error);
        if(contact_notifications) {
            std::sort(pending_contact_events.begin(),
                      pending_contact_events.end(),
                      [](const ContactEvent& a,const ContactEvent& b) {
                          if(a.body0!=b.body0)return a.body0<b.body0;
                          if(a.body1!=b.body1)return a.body1<b.body1;
                          return static_cast<unsigned>(a.kind)<
                              static_cast<unsigned>(b.kind);
                      });
            // A callback can report several shape pairs. Each event's
            // offset was recorded during onContact, so sort only event
            // descriptors; point ranges remain in the same shared arena.
            published_contact_points.swap(pending_contact_points);
            published_contact_events.swap(pending_contact_events);
            // All callbacks referencing these retired actors were
            // consumed by this completed fetchResults boundary.
            retired_actor_tokens.clear();
        }
    }

    void set_body_position(BodyId id, Vec3 position) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        // Preserve PhysX's internal plane alignment and any box spin.
        auto pose = actors[id]->getGlobalPose();
        pose.p = to_physx(position);
        actors[id]->setGlobalPose(pose);
    }

    void set_body_pose(BodyId id, Vec3 position, Quat orientation) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        physx::PxShape* shape=nullptr;
        if(actors[id]->getShapes(&shape,1)!=1 || !shape)
            throw std::logic_error("Reference actor lacks single shape");
        // The SDK horizontal floor is a fixed +Y-normal primitive.
        // Preserve its private PxCreatePlane rotation on a logical
        // identity edit rather than rotating its physical normal.
        if(shape->getGeometry().getType()==physx::PxGeometryType::ePLANE) {
            // World validates PxQuat::isSane; PhysX normalizes a sane
            // quaternion when setting the actor pose. Apply that same
            // rule to the SDK's logical floor orientation before deciding
            // whether the fixed horizontal plane would actually rotate.
            const auto logical=physx::PxQuat(
                orientation.x,orientation.y,orientation.z,orientation.w)
                .getNormalized();
            if(logical.x!=0.0f || logical.y!=0.0f ||
               logical.z!=0.0f || std::abs(logical.w)!=1.0f)
                throw std::logic_error("Native horizontal floor cannot rotate");
            auto pose=actors[id]->getGlobalPose();
            pose.p=to_physx(position);
            actors[id]->setGlobalPose(pose);
            return;
        }
        actors[id]->setGlobalPose(physx::PxTransform(
            to_physx(position),physx::PxQuat(
                orientation.x,orientation.y,orientation.z,orientation.w)));
    }

    void remove_body(BodyId id) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* const actor=actors[id];
        if(contact_notifications) {
            // Push before release: allocation failure leaves actor
            // ownership intact. The stale pointer is compared only,
            // NEVER dereferenced by the contact callback.
            retired_actor_tokens.push_back({
                actor,static_cast<std::uintptr_t>(id)+1});
        }
        actor->release(); // PhysX removes it from the scene.
        actors[id] = nullptr;
    }

    void set_body_shape_offsets(BodyId id, float contact_offset,
                                float rest_offset) override {
        if(id>=actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        physx::PxShape* shape=nullptr;
        if(actors[id]->getShapes(&shape,1)!=1 || !shape)
            throw std::logic_error("Reference actor lacks single shape");
        // PxShape validates each setter against the current opposite
        // distance. Apply the valid final pair in an order that preserves
        // the intermediate source invariant for either edit direction.
        if(contact_offset>shape->getRestOffset()) {
            shape->setContactOffset(contact_offset);
            shape->setRestOffset(rest_offset);
        } else {
            shape->setRestOffset(rest_offset);
            shape->setContactOffset(contact_offset);
        }
    }

    void set_body_material(BodyId id, Material source) override {
        if (id>=actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        physx::PxShape* shape=nullptr;
        if (actors[id]->getShapes(&shape,1)!=1 || !shape)
            throw std::logic_error("Reference rigid actor lacks single material shape");
        // One source PxMaterial is authored on the EXISTING PhysX shape.
        // Shape::setMaterials retains its reference. Release the caller's
        // temporary reference after binding: no second material registry.
        physx::PxMaterial* authored=runtime->physics->createMaterial(
            source.static_friction,source.dynamic_friction,
            source.restitution);
        if (!authored)
            throw std::runtime_error("PhysX material creation failed");
        authored->setDamping(source.damping);
        authored->setDampingCombineMode(
            static_cast<physx::PxCombineMode::Enum>(source.damping_combine));
        authored->setFlag(physx::PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING,
                          source.compliant_acceleration_spring);
        authored->setFrictionCombineMode(
            static_cast<physx::PxCombineMode::Enum>(source.friction_combine));
        authored->setRestitutionCombineMode(
            static_cast<physx::PxCombineMode::Enum>(source.restitution_combine));
        authored->setFlag(physx::PxMaterialFlag::eDISABLE_FRICTION,
                          source.disable_friction);
        authored->setFlag(physx::PxMaterialFlag::eDISABLE_STRONG_FRICTION,
                          source.disable_strong_friction);
        shape->setMaterials(&authored,1);
        authored->release();
    }

    void set_body_dynamics(BodyId id, RigidBodyDynamics d) override {
        if(id>=actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* actor=actors[id]->is<physx::PxRigidDynamic>();
        if(!actor)
            throw std::logic_error("Static actors have no dynamic solver state");
        actor->setLinearDamping(d.linear_damping);
        actor->setAngularDamping(d.angular_damping);
        actor->setMaxLinearVelocity(d.max_linear_velocity);
        actor->setMaxAngularVelocity(d.max_angular_velocity);
        actor->setMaxDepenetrationVelocity(d.max_depenetration_velocity);
        actor->setMaxContactImpulse(d.max_contact_impulse);
        actor->setContactSlopCoefficient(d.contact_slop_coefficient);
        actor->setActorFlag(physx::PxActorFlag::eDISABLE_GRAVITY,
                            !d.gravity_enabled);
        actor->setSolverIterationCounts(d.position_iterations,
                                         d.velocity_iterations);
        actor->setRigidDynamicLockFlags(physx::PxRigidDynamicLockFlags(
            static_cast<physx::PxU8>(d.lock_flags)));
        actor->setRigidBodyFlag(
            physx::PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES,
            d.gyroscopic_forces_enabled);
        actor->setRigidBodyFlag(
            physx::PxRigidBodyFlag::eRETAIN_ACCELERATIONS,
            d.retain_accelerations);
    }

    void set_body_velocity(BodyId id, Vec3 velocity) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* dynamic = actors[id]->is<physx::PxRigidDynamic>();
        if (!dynamic)
            throw std::logic_error("Static actors have no linear velocity");
        dynamic->setLinearVelocity(to_physx(velocity));
    }

    void set_body_angular_velocity(BodyId id, Vec3 velocity) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* dynamic = actors[id]->is<physx::PxRigidDynamic>();
        if (!dynamic)
            throw std::logic_error("Static actors have no angular velocity");
        dynamic->setAngularVelocity(to_physx(velocity));
    }

    void apply_body_torque_impulse(BodyId id, Vec3 impulse) override {
        add_body_torque(id,impulse,ForceMode::Impulse);
    }
    static physx::PxForceMode::Enum physx_force_mode(ForceMode mode) {
        switch(mode) {
        case ForceMode::Force:return physx::PxForceMode::eFORCE;
        case ForceMode::Impulse:return physx::PxForceMode::eIMPULSE;
        case ForceMode::Acceleration:return physx::PxForceMode::eACCELERATION;
        case ForceMode::VelocityChange:return physx::PxForceMode::eVELOCITY_CHANGE;
        }
        throw std::invalid_argument("Unknown PhysX force mode");
    }
    physx::PxRigidDynamic* dynamic_actor(BodyId id) const {
        if(id>=actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* dynamic=actors[id]->is<physx::PxRigidDynamic>();
        if(!dynamic)
            throw std::logic_error("Static actors cannot receive spatial force changes");
        return dynamic;
    }
    void add_body_force(BodyId id, Vec3 value, ForceMode mode) override {
        dynamic_actor(id)->addForce(to_physx(value),physx_force_mode(mode));
    }
    void add_body_torque(BodyId id, Vec3 value, ForceMode mode) override {
        dynamic_actor(id)->addTorque(to_physx(value),physx_force_mode(mode));
    }
    void clear_body_force(BodyId id, ForceMode mode) override {
        dynamic_actor(id)->clearForce(physx_force_mode(mode));
    }
    void clear_body_torque(BodyId id, ForceMode mode) override {
        dynamic_actor(id)->clearTorque(physx_force_mode(mode));
    }

    BodyState body_state(BodyId id) const override {
        if (id >= actors.size() || !actors[id]) {
            throw std::out_of_range("Unknown FyzzXYZZY body ID");
        }
        const auto* actor = actors[id];
        const auto pose = actor->getGlobalPose();
        const auto* dynamic = actor->is<physx::PxRigidDynamic>();
        // PxCreatePlane orients PhysX's local +X-normal plane to the SDK
        // floor's logical +Y normal. That private geometry conversion is
        // not a user-authored floor rotation. Preserve it internally for
        // collisions and pose translation, but never expose it as the
        // provider-neutral BodyState orientation.
        //
        // Inspect the existing PhysX shape rather than maintaining an
        // extra actor-ID/shape-kind mirror alongside "actors".
        Quat logical_orientation = from_physx(pose.q);
        if (!dynamic && actor->getNbShapes() == 1) {
            physx::PxShape* shape = nullptr;
            if (actor->getShapes(&shape, 1) == 1 &&
                shape->getGeometry().getType() == physx::PxGeometryType::ePLANE)
                logical_orientation = {}; // SDK's horizontal +Y floor
        }
        return {from_physx(pose.p),
                dynamic ? from_physx(dynamic->getLinearVelocity())
                        : Vec3{0.0f, 0.0f, 0.0f},
                logical_orientation,
                dynamic ? from_physx(dynamic->getAngularVelocity())
                        : Vec3{0.0f, 0.0f, 0.0f},
                dynamic && dynamic->isSleeping()};
    }

    BodyId add(physx::PxRigidActor* actor) {
        if (!actor) {
            throw std::runtime_error("PhysX failed to create a rigid actor");
        }
        const auto id = static_cast<BodyId>(actors.size());
        try {
            actors.push_back(actor);
        } catch (...) {
            actor->release();
            throw;
        }
        if(contact_notifications)
            actor->userData=reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(id)+1);
        scene->addActor(*actor);
        return id;
    }
};

// The reference CPU engine is isolated from FyzzXYZZY's native World
// implementation. This factory must not be used by the native GPU path.
std::unique_ptr<WorldBackend> create_reference_physx_cpu_backend(
    bool legacy_gu,bool notify,bool enhanced_determinism) {
    return std::make_unique<PhysXCpuWorld>(
        legacy_gu,notify,enhanced_determinism);
}

} // namespace fyzz
