// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Release-build serial/concurrent source third-pass timing. CI hosts
// are noisy: there is deliberately no performance pass/fail threshold.
#include "native/scene.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace fyzz::native {
struct NativeThirdPassTimingFixture {
    struct Result {
        double median_us=0, checksum=0;
        std::size_t asleep=0, spec_islands=0, acc_islands=0;
        std::size_t starts=0, dispatched=0;
    };
    static Result profile(bool split_churn,bool concurrent) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.parallel_third_pass_enabled_=concurrent;
        std::vector<BodyId> ids,bridges;
        const std::size_t count=split_churn?16:96;
        for(std::size_t i=0;i<count;++i) {
            const float x=float(i)*(split_churn?20.0f:3.0f);
            if(split_churn) {
                ids.push_back(scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{x,20,0},1.0f));
                const auto middle=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{x+0.85f,20,0},1.0f);
                bridges.push_back(middle);
                ids.push_back(middle);
                ids.push_back(scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{x+1.70f,20,0},1.0f));
            } else ids.push_back(scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},{x,20,0},1.0f));
        }
        for(auto id:ids)scene.actors_[id].dynamics.gravity_enabled=false;
        scene.advance_collision();
        constexpr std::size_t warmup=24, trials=5, frames=72;
        std::array<double,trials> times{};
        for(std::size_t step=0;step<warmup+trials*frames;++step) {
            if(split_churn) {
                const bool linked=(step&1)!=0;
                for(std::size_t i=0;i<bridges.size();++i)
                    scene.set_position(bridges[i],
                        {float(i)*20.0f+(linked?0.85f:5.0f),20,0});
            } else {
                // Identical injected prior-Dy ready streams test
                // scheduling overhead; native smoke tests independently
                // prove the real Dy sleepCheck handoff.
                for(std::size_t k=0;k<8;++k) {
                    const auto id=ids[(step*8+k)%ids.size()];
                    scene.set_velocity(id,{});
                    scene.actors_[id].ready_for_sleep=true;
                    scene.speculative_ready_nodes_.push_back(id);
                }
            }
            const auto t0=std::chrono::steady_clock::now();
            scene.step(1.0f/120.0f);
            const double us=std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-t0).count();
            if(step>=warmup)times[(step-warmup)/frames]+=us;
        }
        for(auto& v:times)v/=frames;
        std::sort(times.begin(),times.end());
        Result result;
        result.median_us=times[trials/2];
        result.spec_islands=scene.speculative_island_count();
        result.acc_islands=scene.accurate_island_count();
        result.starts=scene.third_pass_worker_start_count_;
        result.dispatched=scene.concurrent_third_pass_dispatches_;
        for(auto id:ids) {
            auto body=scene.body_state(id);
            result.checksum+=body.position.x*0.25+body.position.y+
                             body.velocity.y*0.125;
            result.asleep+=scene.body_is_sleeping(id);
        }
        if(concurrent && (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Parallel third-pass did not reuse its worker");
        if(!concurrent && (result.starts || result.dispatched))
            throw std::runtime_error("Serial third-pass unexpectedly launched work");
        return result;
    }
    static void compare(bool split_churn) {
        const auto serial=profile(split_churn,false);
        const auto parallel=profile(split_churn,true);
        if(serial.asleep!=parallel.asleep ||
           serial.acc_islands!=parallel.acc_islands ||
           serial.spec_islands!=parallel.spec_islands ||
           std::abs(serial.checksum-parallel.checksum)>1e-4)
            throw std::runtime_error("Serial and concurrent island results diverged");
        std::printf(
            "third_pass workload=%s serial_median_us=%.3f"
            " concurrent_median_us=%.3f speedup=%.3fx"
            " worker_starts=%zu dispatches=%zu asleep=%zu"
            " accurate_islands=%zu speculative_islands=%zu\n",
            split_churn?"lost-route-churn":"readiness-events",
            serial.median_us,parallel.median_us,
            serial.median_us/parallel.median_us,
            parallel.starts,parallel.dispatched,parallel.asleep,
            parallel.acc_islands,parallel.spec_islands);
    }
};
}
int main() {
    try {
        fyzz::native::NativeThirdPassTimingFixture::compare(false);
        fyzz::native::NativeThirdPassTimingFixture::compare(true);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"native third-pass benchmark: %s\n",e.what());
        return 1;
    }
    return 0;
}
