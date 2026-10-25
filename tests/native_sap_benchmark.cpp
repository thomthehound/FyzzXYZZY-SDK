// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Native CPU SAP profiling. No timing pass/fail thresholds: CI runners
// are not stable hardware benchmarks. Compare identical shapes, motion
// patterns and build configurations when assessing regressions.
#include "native/scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
enum class Motion { Stationary, Local, Teleport };
void profile(const char* name,Motion motion,unsigned retired) {
    fyzz::native::Scene scene;
    for(unsigned i=0;i<retired;++i) {
        const auto body=scene.add_static_box(
            {0.45f,0.45f,0.45f},{float(i)*4.0f,0,0});
        scene.remove(body);
    }
    constexpr unsigned boxes=256;
    for(unsigned i=0;i<boxes;++i)
        scene.add_static_box({0.45f,0.45f,0.45f},
                             {float(i)*4.0f,0,0});
    const auto mover=scene.add_dynamic_box(
        {0.45f,0.45f,0.45f},{512,1000,0},1.0f);
    scene.advance_collision();
    const auto sorts_before=scene.broadphase_endpoint_full_sort_count();
    const auto shifts_before=scene.broadphase_endpoint_local_shift_count();
    constexpr unsigned trials=5,steps=64;
    std::vector<double> times;
    times.reserve(trials);
    unsigned frame=0;
    for(unsigned trial=0;trial<trials;++trial) {
        const auto start=Clock::now();
        for(unsigned i=0;i<steps;++i,++frame) {
            if(motion==Motion::Local) {
                // Repeatedly cross adjacent source endpoint keys.
                const float dx=float(int(frame%32)-16)*0.5f;
                scene.set_position(mover,{512.0f+dx,1000,0});
            } else if(motion==Motion::Teleport) {
                scene.set_position(mover,
                    {frame%2 ? -100.0f : 1200.0f,1000,0});
            }
            // Force only a geometry query; never use a fake elapsed
            // solver timing to describe the source SAP itself.
            scene.advance_collision(true);
        }
        const auto elapsed=std::chrono::duration<double,std::micro>(
            Clock::now()-start).count();
        times.push_back(elapsed/steps);
    }
    std::sort(times.begin(),times.end());
    std::printf(
        "native_sap mode=%s boxes=%u retired=%u median_us_per_query=%.3f"
        " full_sorts=%zu local_shifts=%zu pairs=%zu\n",
        name,boxes,retired,times[trials/2],
        scene.broadphase_endpoint_full_sort_count()-sorts_before,
        scene.broadphase_endpoint_local_shift_count()-shifts_before,
        scene.pairs().size());
}
} // namespace

int main() {
    try {
        profile("stationary",Motion::Stationary,0);
        profile("local",Motion::Local,0);
        profile("teleport",Motion::Teleport,0);
        profile("retired-history",Motion::Stationary,2048);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"native SAP benchmark: %s\n",error.what());
        return 1;
    }
}
