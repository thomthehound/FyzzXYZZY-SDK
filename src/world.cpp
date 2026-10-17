// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"

#include <PxPhysicsAPI.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fyzz {
namespace {

physx::PxVec3 to_physx(Vec3 value) {
    return physx::PxVec3(value.x, value.y, value.z);
}

Vec3 from_physx(const physx::PxVec3& value) {
    return {value.x, value.y, value.z};
}

void validate_half_extents(Vec3 half_extents) {
    if (!std::isfinite(half_extents.x) || !std::isfinite(half_extents.y) ||
        !std::isfinite(half_extents.z) || half_extents.x <= 0.0f ||
        half_extents.y <= 0.0f || half_extents.z <= 0.0f) {
        throw std::invalid_argument("Box half-extents must be finite and positive");
    }
}

void validate_position(Vec3 position) {
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
        throw std::invalid_argument("Position must be finite");
    }
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

    void initialize() {
        runtime = acquire_runtime();
        physx::PxSceneDesc descriptor(runtime->physics->getTolerancesScale());
        descriptor.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        dispatcher = physx::PxDefaultCpuDispatcherCreate(2);
        if (!dispatcher) {
            throw std::runtime_error("Failed to create PhysX CPU dispatcher");
        }
        descriptor.cpuDispatcher = dispatcher;
        descriptor.filterShader = physx::PxDefaultSimulationFilterShader;
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

        material = runtime->physics->createMaterial(0.5f, 0.5f, 0.2f);
        if (!material) {
            throw std::runtime_error("Failed to create PhysX material");
        }
    }

    void cleanup() noexcept {
        for (physx::PxRigidActor* actor : actors) actor->release();
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

    PhysXCpuWorld() {
        try {
            initialize();
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

    BodyState body_state(BodyId id) const override {
        if (id >= actors.size()) {
            throw std::out_of_range("Unknown FyzzXYZZY body ID");
        }
        const auto* actor = actors[id];
        const physx::PxVec3 position = actor->getGlobalPose().p;
        const auto* dynamic = actor->is<physx::PxRigidDynamic>();
        return {from_physx(position),
                dynamic ? from_physx(dynamic->getLinearVelocity())
                        : Vec3{0.0f, 0.0f, 0.0f}};
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

// The production scene factory is the ONLY selection point, not a
// test-local substitute for PhysX's scene/AABB-manager ownership.
// Portable GPU admission must be implemented by the actual source
// Sc::Scene/factory/task path before it may return a World.
std::unique_ptr<WorldBackend> create_physx_scene_backend(WorldOptions options) {
    switch (options.broadphase) {
    case BroadphaseMode::PhysXCpu:
        return std::make_unique<PhysXCpuWorld>();
    case BroadphaseMode::PhysXPortableGpu:
        throw std::logic_error(
            "Portable PhysX GPU broadphase is not scene-integrated: "
            "Sc::Scene would silently fall back to CPU ABP; "
            "source SAP/AABB manager GPU factory admission remains pending");
    }
    throw std::invalid_argument("Unknown PhysX broadphase mode");
}

World::World() : World(WorldOptions{}) {}
World::World(WorldOptions options)
    : World(create_physx_scene_backend(options)) {}

World::World(std::unique_ptr<WorldBackend> backend) : impl_(std::move(backend)) {
    if (!impl_) {
        throw std::invalid_argument("World requires a simulation backend");
    }
}

World::~World() = default;
World::World(World&&) noexcept = default;
World& World::operator=(World&&) noexcept = default;

BodyId World::add_static_box(Vec3 half_extents, Vec3 position) {
    validate_half_extents(half_extents);
    validate_position(position);
    return impl_->add_static_box(half_extents, position);
}

BodyId World::add_static_floor(float height) {
    if (!std::isfinite(height)) {
        throw std::invalid_argument("Floor height must be finite");
    }
    return impl_->add_static_floor(height);
}

BodyId World::add_dynamic_box(Vec3 half_extents, Vec3 position, float density) {
    validate_half_extents(half_extents);
    validate_position(position);
    if (!std::isfinite(density) || density <= 0.0f) {
        throw std::invalid_argument("Density must be finite and positive");
    }
    return impl_->add_dynamic_box(half_extents, position, density);
}

void World::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f) {
        throw std::invalid_argument("Simulation step must be finite and positive");
    }
    impl_->step(seconds);
}

BodyState World::body_state(BodyId id) const {
    return impl_->body_state(id);
}

} // namespace fyzz
