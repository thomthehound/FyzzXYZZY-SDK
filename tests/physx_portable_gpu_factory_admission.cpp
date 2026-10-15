// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include <PxPhysicsAPI.h>
#include <portable/PxPortableGpuBroadPhaseProvider.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
struct Probe {
    std::uint32_t calls = 0;
    std::uint32_t factories = 0;
};
std::uint32_t reject(void* p, const void* scene) {
    if (!scene) throw std::runtime_error("No scene description");
    ++static_cast<Probe*>(p)->calls;
    return 0u;
}
void* unexpected_bp(void* p, const void*, std::uint64_t) {
    ++static_cast<Probe*>(p)->factories;
    return nullptr;
}
void* unexpected_aabb(void* p, const void*, void*, void*, void*, void*,
                      std::uint64_t) {
    ++static_cast<Probe*>(p)->factories;
    return nullptr;
}
void check(bool valid, const char* why) {
    if (!valid) throw std::runtime_error(why);
}
} // namespace

int main() {
    static_assert(std::is_standard_layout<PxPortableGpuBroadPhaseProviderV1>::value,
                  "Provider ABI must be a C-layout record");
    static_assert(offsetof(PxPortableGpuBroadPhaseProviderV1, abi_version) == 0,
                  "Provider ABI version cannot move");
    try {
        physx::PxDefaultAllocator allocator;
        physx::PxDefaultErrorCallback errors;
        physx::PxFoundation* foundation =
            PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        check(foundation != nullptr, "PhysX foundation construction failed");
        auto* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
                                       physx::PxTolerancesScale(), false, nullptr);
        check(physics != nullptr, "PhysX physics construction failed");
        auto* dispatcher = physx::PxDefaultCpuDispatcherCreate(2u);
        check(dispatcher != nullptr, "PhysX dispatcher construction failed");

        physx::PxSceneDesc cpu(physics->getTolerancesScale());
        cpu.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
        cpu.cpuDispatcher = dispatcher;
        cpu.filterShader = physx::PxDefaultSimulationFilterShader;
        check(cpu.isValid(), "Baseline CPU scene invalid");
        auto* scene = physics->createScene(cpu);
        check(scene != nullptr, "CPU scene regressed after provider addition");
        scene->simulate(1.0f / 60.0f);
        check(scene->fetchResults(true), "CPU scene simulation failed");
        scene->release();

        Probe probe{};
        PxPortableGpuBroadPhaseProviderV1 provider{};
        provider.abi_version = PX_PORTABLE_GPU_BP_PROVIDER_ABI_V1;
        provider.struct_size = sizeof(provider);
        provider.user_data = &probe;
        provider.supports_scene = reject;
        provider.create_broadphase = unexpected_bp;
        provider.create_aabb_manager = unexpected_aabb;

        physx::PxSceneDesc requested = cpu;
        requested.broadPhaseType = physx::PxBroadPhaseType::eGPU;
        requested.portableGpuBroadPhaseProvider = &provider;
        check(requested.isValid(), "Valid portable provider rejected structurally");
        check(physics->createScene(requested) == nullptr,
              "Unsupported portable GPU request silently fell back to CPU");
        check(probe.calls == 1u && probe.factories == 0u,
              "Portable admission did not reject before factory construction");

        auto invalid = requested;
        provider.abi_version = 2u;
        check(!invalid.isValid(), "Unsupported provider ABI accepted");
        provider.abi_version = PX_PORTABLE_GPU_BP_PROVIDER_ABI_V1;
        provider.struct_size = sizeof(provider) - 1u;
        check(!invalid.isValid(), "Truncated provider record accepted");
        provider.struct_size = sizeof(provider);
        provider.create_broadphase = nullptr;
        check(!invalid.isValid(), "Missing broadphase factory accepted");
        provider.create_broadphase = unexpected_bp;
        invalid.broadPhaseType = physx::PxBroadPhaseType::ePABP;
        check(!invalid.isValid(), "Portable provider accepted with CPU mode");
        invalid = requested;
        invalid.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
        check(!invalid.isValid(), "Portable broadphase accepted CUDA GPU dynamics");
        invalid = requested;
        invalid.flags |= physx::PxSceneFlag::eENABLE_DIRECT_GPU_API;
        check(!invalid.isValid(), "Portable broadphase accepted direct CUDA API");
        check(probe.factories == 0u,
              "Structural validation attempted to instantiate GPU components");

        dispatcher->release();
        physics->release();
        foundation->release();
        std::cout << "PhysX portable GPU host factory admission PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX portable GPU host factory admission FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
