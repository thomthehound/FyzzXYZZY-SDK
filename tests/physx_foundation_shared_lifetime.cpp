// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include "physx_foundation_lifetime.h"

#include <stdexcept>

int main() {
    using namespace fyzz;
    auto gpu_lease=physx_runtime::acquire_foundation();
    auto* initial=&gpu_lease->get();
    {
        WorldOptions options{ExecutionMode::ReferencePhysXCpu};
        World reference(options);
        const auto box=reference.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,2.0f,0.0f},1.0f);
        reference.step(1.0f/120.0f);
        if(!reference.body_state(box).position.y)
            throw std::runtime_error("Reference scene failed with GPU Foundation lease");

        auto second=physx_runtime::acquire_foundation();
        if(&second->get()!=initial)
            throw std::runtime_error("Reference/GPU Foundation ownership split");
    }
    // A surviving GPU lease must allow reference Physics to be recreated.
    {
        World reference(WorldOptions{ExecutionMode::ReferencePhysXCpuLegacyGu});
        if(&physx_runtime::acquire_foundation()->get()!=initial)
            throw std::runtime_error("Reference runtime recreated an active Foundation");
        reference.step(1.0f/120.0f);
    }
    gpu_lease.reset();

    // After the final CPU and GPU users leave, the original singleton must
    // permit an entirely new reference World and another GPU lease.
    {
        World reference(WorldOptions{ExecutionMode::ReferencePhysXCpu});
        auto next=physx_runtime::acquire_foundation();
        if(PxIsFoundationValid()!=&next->get())
            throw std::runtime_error("Foundation was not recreated");
        reference.step(1.0f/120.0f);
    }
    return 0;
}
