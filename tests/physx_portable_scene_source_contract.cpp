// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include <PxPhysicsAPI.h>
#include "ScPortableGpuBroadPhaseFactory.h"

#include <iostream>
#include <stdexcept>

namespace {
// This fixture NEVER constructs a broadphase. It tests the actual
// PhysX PxSceneDesc source ABI and illegal selection combinations,
// while the production SDK still refuses GPU admission without a
// concrete source-derived SAP + AABB manager provider.
class UnavailableTestFactory final : public physx::Sc::PortableGpuBroadPhaseFactory {
public:
    physx::Bp::BroadPhase& createBroadPhase(
        const physx::PxSceneDesc&, physx::PxU64) override {
        throw std::logic_error("Fixture must not instantiate a GPU broadphase");
    }
    physx::Bp::AABBManagerBase& createAABBManager(
        const physx::PxSceneDesc&, physx::Bp::BroadPhase&,
        physx::Bp::BoundsArray&, physx::Cm::PinnableArray<physx::PxReal>&,
        physx::Cm::VirtualAllocatorCallback&, physx::PxU64) override {
        throw std::logic_error("Fixture must not instantiate a GPU AABB manager");
    }
};
} // namespace

int main() {
    try {
        physx::PxSceneDesc desc{physx::PxTolerancesScale()};
        desc.filterShader=physx::PxDefaultSimulationFilterShader;
        auto* dispatcher=physx::PxDefaultCpuDispatcherCreate(1);
        if(!dispatcher) throw std::runtime_error("Cannot construct CPU task dispatcher");
        try {
            desc.cpuDispatcher=dispatcher;
            if(!desc.isValid())
                throw std::runtime_error("Ordinary CPU source descriptor became invalid");
            UnavailableTestFactory factory;
            desc.portableGpuBroadPhaseFactory=&factory;
            if(desc.isValid())
                throw std::runtime_error("Portable GPU factory accepted CPU broadphase");
            desc.broadPhaseType=physx::PxBroadPhaseType::eGPU;
            if(!desc.isValid())
                throw std::runtime_error("Portable GPU source factory with CPU dynamics rejected");
            desc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
            if(desc.isValid())
                throw std::runtime_error("Portable GPU SAP accepted unsupported CUDA GPU dynamics");
            desc.flags &= ~physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
            desc.flags |= physx::PxSceneFlag::eENABLE_DIRECT_GPU_API;
            if(desc.isValid())
                throw std::runtime_error("Portable GPU SAP accepted CUDA direct GPU API");
            desc.flags &= ~physx::PxSceneFlag::eENABLE_DIRECT_GPU_API;
            desc.portableGpuBroadPhaseFactory=nullptr;
            desc.broadPhaseType=physx::PxBroadPhaseType::ePABP;
            if(!desc.isValid())
                throw std::runtime_error("Removing portable GPU factory did not restore CPU source descriptor");
        }catch(...){dispatcher->release();throw;}
        dispatcher->release();
        std::cout<<"PhysX source scene portable GPU factory admission contract PASS\n";
    }catch(const std::exception& e){
        std::cerr<<"PhysX source scene portable GPU admission FAILED: "<<e.what()<<"\n";
        return 1;
    }
}
