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
        std::vector<double> impulses, joint_impulses;
        double median_us=0, checksum=0;
        std::size_t asleep=0, spec_islands=0, acc_islands=0;
        std::size_t starts=0, dispatched=0;
    };
    static Result profile(bool split_churn,Mode mode,std::size_t chains,
                          bool deep_chain=false) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.parallel_third_pass_enabled_=mode!=Mode::Serial;
        // This workload measures source third-pass scheduling alone.
        // Dy PGS worker admission is independently benchmarked below.
        scene.parallel_solver_enabled_=false;
        if(mode==Mode::ForcedParallel)
            scene.parallel_third_pass_min_dirty_nodes_=0;
        std::vector<BodyId> ids,bridges;
        // A long source island exercises expensive findRoute searches
        // with very few dirty endpoints. Independent three-body chains
        // instead exercise many cheap, unrelated lost-edge repairs.
        if(deep_chain) {
            for(std::size_t i=0;i<chains;++i)
                ids.push_back(scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{float(i)*0.85f,20,0},1.0f));
            bridges.push_back(ids[chains/2]);
        }
        const std::size_t count=deep_chain?0:(split_churn?chains:96);
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
        Result result;
        // Compare exact actor and island ownership at each measured trial
        // boundary, not only after a split/remerge sequence reconverges.
        result.bodies.reserve(ids.size()*trials);
        result.speculative_owners.reserve(ids.size()*trials);
        result.accurate_owners.reserve(ids.size()*trials);
        for(std::size_t step=0;step<warmup+trials*frames;++step) {
            if(split_churn) {
                const bool linked=(step&1)!=0;
                if(deep_chain) {
                    scene.set_position(bridges.front(),
                        {float(chains/2)*0.85f,linked?20.0f:25.0f,0});
                } else for(std::size_t i=0;i<bridges.size();++i)
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
            if(deep_chain)scene.advance_collision();
            else scene.step(1.0f/120.0f);
            const double us=std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-t0).count();
            if(step>=warmup)times[(step-warmup)/frames]+=us;
            if(step>=warmup && (step-warmup+1)%frames==0) {
                // Capture outside the timed region. Each mode starts from
                // identical authored state and must agree across trials.
                for(auto id:ids) {
                    result.bodies.push_back(scene.body_state(id));
                    result.speculative_owners.push_back(
                        scene.speculative_island_id(id));
                    result.accurate_owners.push_back(
                        scene.accurate_island_id(id));
                }
            }
        }
        for(auto& v:times)v/=frames;
        std::sort(times.begin(),times.end());
        result.median_us=times[trials/2];
        result.spec_islands=scene.speculative_island_count();
        result.acc_islands=scene.accurate_island_count();
        result.starts=scene.third_pass_worker_start_count_;
        result.dispatched=deep_chain
            ? scene.concurrent_lost_route_dispatches_
            : scene.concurrent_third_pass_dispatches_;
        for(auto id:ids) {
            const auto body=scene.body_state(id);
            result.checksum+=body.position.x*0.25+body.position.y+
                             body.velocity.y*0.125;
            result.asleep+=body.sleeping;
        }
        if(mode==Mode::ForcedParallel &&
           (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Forced parallel worker was not reused");
        if(mode==Mode::Serial && (result.starts || result.dispatched))
            throw std::runtime_error("Serial third-pass launched a worker");
        if(mode==Mode::Adaptive && split_churn && !deep_chain && chains<=64 &&
           (result.starts || result.dispatched))
            throw std::runtime_error("Short lost-route work launched a worker");
        if(mode==Mode::Adaptive && split_churn && !deep_chain && chains>=256 &&
           (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Large lost-route work was not parallelized");
        return result;
    }
    static void verify_same_state(const Result& a,const Result& b) {
        if(a.asleep!=b.asleep || a.acc_islands!=b.acc_islands ||
           a.spec_islands!=b.spec_islands ||
           a.bodies.size()!=b.bodies.size() ||
           a.speculative_owners!=b.speculative_owners ||
           a.accurate_owners!=b.accurate_owners ||
           a.impulses!=b.impulses ||
           a.joint_impulses!=b.joint_impulses)
            throw std::runtime_error(
                "Native serial and concurrent third-pass trial ownership differs");
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
                    "Native serial/concurrent third-pass trial state differs");
        }
    }

    static Result profile_solver(Mode mode,std::size_t body_count,
                                 bool joint_pairs=false) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.parallel_third_pass_enabled_=false;
        scene.parallel_solver_enabled_=mode!=Mode::Serial;
        if(joint_pairs)scene.set_enhanced_determinism(true);
        if(mode==Mode::ForcedParallel)
            scene.parallel_solver_min_descriptors_=0;
        scene.add_static_floor(0.0f);
        std::vector<BodyId> ids;
        ids.reserve(body_count);
        for(std::size_t i=0;i<body_count;++i) {
            const float x=joint_pairs
                ? float(i/2)*6.0f+float(i%2)*1.4f
                : float(i)*3.0f;
            const auto id=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},
                {x,0.48f,0.0f},1.0f);
            scene.actors_[id].dynamics.gravity_enabled=false;
            ids.push_back(id);
            if(joint_pairs && (i&1u))
                scene.add_distance_constraint(ids[i-1],id,1.4f);
        }
        constexpr std::size_t warmup=16,trials=5,frames=32;
        std::array<double,trials> times{};
        Result result;
        result.bodies.reserve(trials*body_count);
        result.speculative_owners.reserve(trials*body_count);
        result.accurate_owners.reserve(trials*body_count);
        for(std::size_t step=0;step<warmup+trials*frames;++step) {
            // Identical incoming contact geometry and downward motion
            // keep all island groups active through the measured window.
            // Authored inputs remain outside the timed native step.
            for(std::size_t i=0;i<ids.size();++i) {
                const float x=joint_pairs
                    ? float(i/2)*6.0f+float(i%2)*1.4f
                    : float(i)*3.0f;
                scene.set_position(ids[i],{x,0.48f,0.0f});
                scene.set_velocity(ids[i],{0.0f,-1.0f,0.0f});
            }
            const auto start=std::chrono::steady_clock::now();
            scene.step(1.0f/120.0f);
            const double us=std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-start).count();
            if(step>=warmup)times[(step-warmup)/frames]+=us;
            if(step>=warmup && (step-warmup+1)%frames==0) {
                for(BodyId id:ids) {
                    result.bodies.push_back(scene.body_state(id));
                    result.speculative_owners.push_back(
                        scene.speculative_island_id(id));
                    result.accurate_owners.push_back(
                        scene.accurate_island_id(id));
                }
                const auto impulses=scene.solved_normal_impulses();
                result.impulses.insert(result.impulses.end(),
                                       impulses.begin(),impulses.end());
                for(const auto& row:scene.distance_rows_)
                    result.joint_impulses.push_back(row.applied_force);
            }
        }
        for(double& v:times)v/=frames;
        std::sort(times.begin(),times.end());
        result.median_us=times[trials/2];
        result.starts=scene.third_pass_worker_start_count_;
        result.dispatched=scene.concurrent_solver_dispatches_;
        result.spec_islands=scene.speculative_island_count();
        result.acc_islands=scene.accurate_island_count();
        for(BodyId id:ids) {
            const auto body=scene.body_state(id);
            result.asleep+=body.sleeping;
            result.checksum+=body.position.y+body.velocity.y;
        }
        if(mode==Mode::Serial && (result.starts || result.dispatched))
            throw std::runtime_error("Serial Dy PGS unexpectedly launched work");
        if(mode!=Mode::Serial && (result.starts!=1 || !result.dispatched))
            throw std::runtime_error("Independent Dy PGS worker did not run");
        return result;
    }
    static void compare_solver(std::size_t body_count,bool joint_pairs=false) {
        const auto serial=profile_solver(Mode::Serial,body_count,joint_pairs);
        const auto forced=profile_solver(Mode::ForcedParallel,body_count,joint_pairs);
        const auto adaptive=profile_solver(Mode::Adaptive,body_count,joint_pairs);
        verify_same_state(serial,forced);
        verify_same_state(serial,adaptive);
        std::printf(
            "dy_pgs workload=%s bodies=%zu serial_median_us=%.3f"
            " forced_parallel_median_us=%.3f adaptive_median_us=%.3f"
            " forced_speedup=%.3fx adaptive_speedup=%.3fx"
            " forced_starts=%zu forced_dispatches=%zu"
            " adaptive_starts=%zu adaptive_dispatches=%zu\n",
            joint_pairs?"contact-1d":"contacts",body_count,
            serial.median_us,forced.median_us,
            adaptive.median_us,serial.median_us/forced.median_us,
            serial.median_us/adaptive.median_us,
            forced.starts,forced.dispatched,
            adaptive.starts,adaptive.dispatched);
    }
    static void compare(bool split_churn,std::size_t chains=16,
                        bool deep_chain=false) {
        const auto serial=profile(split_churn,Mode::Serial,chains,deep_chain);
        const auto forced=profile(split_churn,Mode::ForcedParallel,chains,deep_chain);
        const auto adaptive=profile(split_churn,Mode::Adaptive,chains,deep_chain);
        verify_same_state(serial,forced);
        verify_same_state(serial,adaptive);
        std::printf(
            "third_pass workload=%s chains=%zu serial_median_us=%.3f"
            " forced_parallel_median_us=%.3f adaptive_median_us=%.3f"
            " forced_speedup=%.3fx adaptive_speedup=%.3fx"
            " forced_starts=%zu forced_dispatches=%zu"
            " adaptive_starts=%zu adaptive_dispatches=%zu"
            " asleep=%zu accurate_islands=%zu speculative_islands=%zu\n",
            deep_chain?"deep-route-only":
                (split_churn?"lost-route-churn":"readiness-drain"),chains,
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
        fyzz::native::NativeThirdPassTimingFixture::compare(true,128);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,256);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,512);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,1024);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,64,true);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,256,true);
        fyzz::native::NativeThirdPassTimingFixture::compare(true,512,true);
        fyzz::native::NativeThirdPassTimingFixture::compare_solver(256);
        fyzz::native::NativeThirdPassTimingFixture::compare_solver(512);
        fyzz::native::NativeThirdPassTimingFixture::compare_solver(256,true);
        fyzz::native::NativeThirdPassTimingFixture::compare_solver(512,true);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"native third-pass benchmark: %s\n",e.what());
        return 1;
    }
    return 0;
}
