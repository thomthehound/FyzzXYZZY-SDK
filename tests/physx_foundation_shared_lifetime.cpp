// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/world.h"
#include "physx_foundation_lifetime.h"

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

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
    // Last-release/acquire races must never create a second Foundation
    // while the first singleton is still being destroyed. Exercise
    // overlapping leases and complete release/recreation on several threads.
    std::atomic<bool> invalid=false;
    const auto acquire_cycle=[&] {
        for(unsigned i=0;i<512u;++i) {
            try {
                auto lease=physx_runtime::acquire_foundation();
                if(PxIsFoundationValid()!=&lease->get())
                    invalid.store(true,std::memory_order_relaxed);
            } catch (...) {
                invalid.store(true,std::memory_order_relaxed);
            }
        }
    };
    std::vector<std::thread> workers;
    for(unsigned i=0;i<4u;++i)
        workers.emplace_back(acquire_cycle);
    for(auto& worker:workers)
        worker.join();
    if(invalid.load(std::memory_order_relaxed) ||
       PxIsFoundationValid()!=nullptr)
        throw std::runtime_error(
            "Concurrent native/reference Foundation acquire and final release raced");
    return 0;
}
