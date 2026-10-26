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
    enum class Mode { Serial, ForcedParallel, Adaptive };
    struct Result {
        std::vector<BodyState> bodies;
        std::vector<std::uint32_t> speculative_owners,accurate_owners;
        double median_us=0, checksum=0;
        std::size_t asleep=0, spec_islands=0, acc_islands=0;
        std::size_t starts=0, dispatched=0;
    };
    static Result profile(bool split_churn,Mode mode,std::size_t chains) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.parallel_third_pass_enabled_=mode!=Mode::Serial;
        if(mode==Mode::ForcedParallel)
            scene.parallel_third_pass_min_dirty_nodes_=0;
        std::vector<BodyId> ids,bridges;
        const std::size_t count=split_churn?chains:96;
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
        result.bodies.reserve(ids.size());
        result.speculative_owners.reserve(ids.size());
        result.accurate_owners.reserve(ids.size());
        for(auto id:ids) {
            const auto body=scene.body_state(id);
            result.bodies.push_back(body);
            result.speculative_owners.push_back(scene.speculative_island_id(id));
            result.accurate_owners.push_back(scene.accurate_island_id(id));
            result.checksum+=body.position.x*0.25+body.position.y+
                             body.velocity.y*0.125;
            result.asleep+=body.sleeping;
        }
        if(mode==Mode::ForcedParallel &&
           (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Forced parallel worker was not reused");
        if(mode==Mode::Serial && (result.starts || result.dispatched))
            throw std::runtime_error("Serial third-pass launched a worker");
        if(mode==Mode::Adaptive && split_churn && chains<=64 &&
           (result.starts || result.dispatched))
            throw std::runtime_error("Short lost-route work launched a worker");
        if(mode==Mode::Adaptive && split_churn && chains>=256 &&
           (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Large lost-route work was not parallelized");
        return result;
    }
    static void verify_same_state(const Result& a,const Result& b) {
        if(a.asleep!=b.asleep || a.acc_islands!=b.acc_islands ||
           a.spec_islands!=b.spec_islands ||
           a.bodies.size()!=b.bodies.size() ||
           a.speculative_owners!=b.speculative_owners ||
           a.accurate_owners!=b.accurate_owners)
            throw std::runtime_error(
                "Native serial and concurrent third-pass ownership differs");
        const auto same_vec=[](Vec3 x,Vec3 y) {
            return x.x==y.x && x.y==y.y && x.z==y.z;
        };
        for(std::size_t i=0;i<a.bodies.size();++i) {
            const auto& x=a.bodies[i];
            const auto& y=b.bodies[i];
            if(!same_vec(x.position,y.position) ||
               !same_vec(x.velocity,y.velocity) ||
               !same_vec(x.angular_velocity,y.angular_velocity) ||
               x.orientation.x!=y.orientation.x ||
               x.orientation.y!=y.orientation.y ||
               x.orientation.z!=y.orientation.z ||
               x.orientation.w!=y.orientation.w ||
               x.sleeping!=y.sleeping)
                throw std::runtime_error(
                    "Native serial/concurrent third-pass body state differs");
        }
    }
    static void compare(bool split_churn,std::size_t chains=16) {
        const auto serial=profile(split_churn,Mode::Serial,chains);
        const auto forced=profile(split_churn,Mode::ForcedParallel,chains);
        const auto adaptive=profile(split_churn,Mode::Adaptive,chains);
        verify_same_state(serial,forced);
        verify_same_state(serial,adaptive);
        std::printf(
            "third_pass workload=%s chains=%zu serial_median_us=%.3f"
            " forced_parallel_median_us=%.3f adaptive_median_us=%.3f"
            " forced_speedup=%.3fx adaptive_speedup=%.3fx"
            " forced_starts=%zu forced_dispatches=%zu"
            " adaptive_starts=%zu adaptive_dispatches=%zu"
            " asleep=%zu accurate_islands=%zu speculative_islands=%zu\\n",
            split_churn?"lost-route-churn":"readiness-drain",chains,
            serial.median_us,forced.median_us,adaptive.median_us,
            serial.median_us/forced.median_us,
            serial.median_us/adaptive.median_us,
            forced.starts,forced.dispatched,
            adaptive.starts,adaptive.dispatched,
            adaptive.asleep,adaptive.acc_islands,adaptive.spec_islands);
    }
};
}
int main() {
    try {
        fyzz::native::NativeThirdPassTimingFixture::compare(false);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,16);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,64);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,256);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,1024);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"native third-pass benchmark: %s\n",e.what());
        return 1;
    }
    return 0;
}
