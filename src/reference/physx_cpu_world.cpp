// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include <PxPhysicsAPI.h>

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

struct PhysXCpuWorld final : WorldBackend {
    std::shared_ptr<PhysXRuntime> runtime;
    physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
    physx::PxScene* scene = nullptr;
    physx::PxMaterial* material = nullptr;
    std::vector<physx::PxRigidActor*> actors;

    void initialize(bool legacy_gu) {
        runtime = acquire_runtime();
        physx::PxSceneDesc descriptor(runtime->physics->getTolerancesScale());
        descriptor.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        dispatcher = physx::PxDefaultCpuDispatcherCreate(2);
        if (!dispatcher) {
            throw std::runtime_error("Failed to create PhysX CPU dispatcher");
        }
        descriptor.cpuDispatcher = dispatcher;
        descriptor.filterShader = physx::PxDefaultSimulationFilterShader;
        // Keep stock PhysX PCM in ReferencePhysXCpu. The explicit Gu
        // conformance mode selects original source contactBoxBox/PlaneBox
        // rather than silently changing the default reference behavior.
        if (legacy_gu)
            descriptor.flags &= ~physx::PxSceneFlag::eENABLE_PCM;
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

        material = runtime->physics->createMaterial(0.5f, 0.5f, 0.2f);
        if (!material) {
            throw std::runtime_error("Failed to create PhysX material");
        }
    }

    void cleanup() noexcept {
        for (physx::PxRigidActor* actor : actors)
            if (actor) actor->release();
        actors.clear();
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

    explicit PhysXCpuWorld(bool legacy_gu) {
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
        scene->simulate(seconds);
        if (!scene->fetchResults(true)) {
            throw std::runtime_error("PhysX simulation failed");
        }
    }

    void set_body_position(BodyId id, Vec3 position) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        // PxCreatePlane uses a non-identity pose quaternion to align its
        // local +X plane normal with world +Y. A position-only transform
        // would reset that rotation (and any dynamic body's rotation).
        // Preserve the existing orientation, changing translation only.
        auto pose = actors[id]->getGlobalPose();
        pose.p = to_physx(position);
        actors[id]->setGlobalPose(pose);
    }

    void remove_body(BodyId id) override {
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        actors[id]->release();  // PhysX removes it from the scene.
        actors[id] = nullptr;
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
        if (id >= actors.size() || !actors[id])
            throw std::out_of_range("Unknown or retired FyzzXYZZY body ID");
        auto* dynamic = actors[id]->is<physx::PxRigidDynamic>();
        if (!dynamic)
            throw std::logic_error("Static actors cannot receive torque impulses");
        dynamic->addTorque(to_physx(impulse), physx::PxForceMode::eIMPULSE);
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
        scene->addActor(*actor);
        return id;
    }
};

// The reference CPU engine is isolated from FyzzXYZZY's native World
// implementation. This factory must not be used by the native GPU path.
std::unique_ptr<WorldBackend> create_reference_physx_cpu_backend(bool legacy_gu) {
    return std::make_unique<PhysXCpuWorld>(legacy_gu);
}

} // namespace fyzz
