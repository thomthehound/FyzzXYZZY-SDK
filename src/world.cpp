// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fizzx/world.h"

#include <PxPhysicsAPI.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fizzx {
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

} // namespace

struct World::Impl {
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback errors;
    physx::PxFoundation* foundation = nullptr;
    physx::PxPhysics* physics = nullptr;
    physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
    physx::PxScene* scene = nullptr;
    physx::PxMaterial* material = nullptr;
    std::vector<physx::PxRigidActor*> actors;

    void initialize() {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        if (!foundation) {
            throw std::runtime_error("Failed to create PhysX foundation");
        }

        physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
                                 physx::PxTolerancesScale(), false, nullptr);
        if (!physics) {
            throw std::runtime_error("Failed to create PhysX physics");
        }

        physx::PxSceneDesc descriptor(physics->getTolerancesScale());
        descriptor.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        dispatcher = physx::PxDefaultCpuDispatcherCreate(2);
        if (!dispatcher) {
            throw std::runtime_error("Failed to create PhysX CPU dispatcher");
        }
        descriptor.cpuDispatcher = dispatcher;
        descriptor.filterShader = physx::PxDefaultSimulationFilterShader;
        scene = physics->createScene(descriptor);
        if (!scene) {
            throw std::runtime_error("Failed to create PhysX scene");
        }

        material = physics->createMaterial(0.5f, 0.5f, 0.2f);
        if (!material) {
            throw std::runtime_error("Failed to create PhysX material");
        }
    }

    ~Impl() {
        for (physx::PxRigidActor* actor : actors) {
            actor->release();
        }
        if (material) {
            material->release();
        }
        if (scene) {
            scene->release();
        }
        if (dispatcher) {
            dispatcher->release();
        }
        if (physics) {
            physics->release();
        }
        if (foundation) {
            foundation->release();
        }
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

World::World() : impl_(std::make_unique<Impl>()) {
    impl_->initialize();
}

World::~World() = default;
World::World(World&&) noexcept = default;
World& World::operator=(World&&) noexcept = default;

BodyId World::add_static_box(Vec3 half_extents, Vec3 position) {
    validate_half_extents(half_extents);
    validate_position(position);
    return impl_->add(physx::PxCreateStatic(
        *impl_->physics, physx::PxTransform(to_physx(position)),
        physx::PxBoxGeometry(to_physx(half_extents)), *impl_->material));
}

BodyId World::add_dynamic_box(Vec3 half_extents, Vec3 position, float density) {
    validate_half_extents(half_extents);
    validate_position(position);
    if (!std::isfinite(density) || density <= 0.0f) {
        throw std::invalid_argument("Density must be finite and positive");
    }
    return impl_->add(physx::PxCreateDynamic(
        *impl_->physics, physx::PxTransform(to_physx(position)),
        physx::PxBoxGeometry(to_physx(half_extents)), *impl_->material, density));
}

void World::step(float seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0f) {
        throw std::invalid_argument("Simulation step must be finite and positive");
    }
    impl_->scene->simulate(seconds);
    if (!impl_->scene->fetchResults(true)) {
        throw std::runtime_error("PhysX simulation failed");
    }
}

BodyState World::body_state(BodyId id) const {
    if (id >= impl_->actors.size()) {
        throw std::out_of_range("Unknown FizzX body ID");
    }
    const auto* actor = impl_->actors[id];
    const physx::PxVec3 position = actor->getGlobalPose().p;
    const auto* dynamic = actor->is<physx::PxRigidDynamic>();
    return {from_physx(position),
            dynamic ? from_physx(dynamic->getLinearVelocity()) : Vec3{0.0f, 0.0f, 0.0f}};
}

} // namespace fizzx
