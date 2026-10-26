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

namespace fyzz::native {
struct NativeSapTimingFixture {
    static void profile(bool crossings) {
        // Measure Bp endpoint maintenance alone: deliberately do not
        // invoke Gu, Sc pair publication or island reconciliation.
        Scene scene;
        constexpr unsigned statics=192,movers=64;
        for(unsigned i=0;i<statics;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,0});
        std::vector<BodyId> ids;
        ids.reserve(movers);
        for(unsigned i=0;i<movers;++i)
            ids.push_back(scene.add_dynamic_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,1000},1.0f));
        scene.advance_collision();
        const auto pocket_before=scene.broadphase_activity_pocket_count();
        const auto touched_before=
            scene.broadphase_activity_pocket_endpoint_count();
        const auto comparisons_before=
            scene.broadphase_activity_pocket_comparison_count();
        const auto relocations_before=
            scene.broadphase_activity_pocket_relocation_count();
        constexpr unsigned trials=7,steps=96;
        std::vector<double> measurements;
        measurements.reserve(trials);
        unsigned frame=0;
        for(unsigned trial=0;trial<trials;++trial) {
            double total=0.0;
            for(unsigned step=0;step<steps;++step,++frame) {
                for(unsigned i=0;i<movers;++i) {
                    const float dx=(frame&1u)
                        ? (crossings?3.25f:0.25f)
                        : (crossings?-3.25f:-0.25f);
                    scene.set_position(ids[i],
                        {float(i%32)*3.0f+dx,
                         float(i/32)*3.0f,1000});
                }
                const auto t0=std::chrono::steady_clock::now();
                scene.update_bounds();
                total+=std::chrono::duration<double,std::micro>(
                    std::chrono::steady_clock::now()-t0).count();
            }
            measurements.push_back(total/steps);
        }
        std::sort(measurements.begin(),measurements.end());
        std::printf(
            "native_sap mode=%s boxes=%u updated=%u"
            " median_us_per_query=%.3f pockets=%zu pocket_endpoints=%zu"
            " linked_compares=%zu linked_relocations=%zu\n",
            crossings?"endpoint-only-crossings":"endpoint-only-local",
            statics+movers,movers,measurements[trials/2],
            scene.broadphase_activity_pocket_count()-pocket_before,
            scene.broadphase_activity_pocket_endpoint_count()-touched_before,
            scene.broadphase_activity_pocket_comparison_count()-comparisons_before,
            scene.broadphase_activity_pocket_relocation_count()-relocations_before);
    }
};
} // namespace fyzz::native

namespace {
using Clock=std::chrono::steady_clock;
enum class Motion { Stationary, Local, LocalContacts, Teleport };
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
    const float mover_y=motion==Motion::LocalContacts?0.0f:1000.0f;
    const auto mover=scene.add_dynamic_box(
        {0.45f,0.45f,0.45f},{512,mover_y,0},1.0f);
    scene.advance_collision();
    const auto sorts_before=scene.broadphase_endpoint_full_sort_count();
    const auto shifts_before=scene.broadphase_endpoint_local_shift_count();
    const auto pair_updates_before=scene.broadphase_incremental_pair_update_count();
    const auto crossings_before=scene.broadphase_endpoint_crossing_count();
    std::size_t contact_transitions=0;
    constexpr unsigned trials=5,steps=64;
    std::vector<double> times;
    times.reserve(trials);
    unsigned frame=0;
    for(unsigned trial=0;trial<trials;++trial) {
        const auto start=Clock::now();
        for(unsigned i=0;i<steps;++i,++frame) {
            if(motion==Motion::Local ||
               motion==Motion::LocalContacts) {
                // Repeatedly cross adjacent source endpoint keys,
                // optionally retaining genuine Gu contact managers.
                const float dx=float(int(frame%32)-16)*0.5f;
                scene.set_position(mover,{512.0f+dx,mover_y,0});
            } else if(motion==Motion::Teleport) {
                scene.set_position(mover,
                    {frame%2 ? -100.0f : 1200.0f,1000,0});
            }
            // Force only a geometry query; never use a fake elapsed
            // solver timing to describe the source SAP itself.
            scene.advance_collision(true);
            contact_transitions+=scene.found_pairs().size()+
                                 scene.lost_pairs().size();
        }
        const auto elapsed=std::chrono::duration<double,std::micro>(
            Clock::now()-start).count();
        times.push_back(elapsed/steps);
    }
    std::sort(times.begin(),times.end());
    std::printf(
        "native_sap mode=%s boxes=%u retired=%u median_us_per_query=%.3f"
        " full_sorts=%zu local_shifts=%zu incremental_pair_updates=%zu"
        " minmax_crossings=%zu contact_transitions=%zu pairs=%zu\n",
        name,boxes,retired,times[trials/2],
        scene.broadphase_endpoint_full_sort_count()-sorts_before,
        scene.broadphase_endpoint_local_shift_count()-shifts_before,
        scene.broadphase_incremental_pair_update_count()-pair_updates_before,
        scene.broadphase_endpoint_crossing_count()-crossings_before,
        contact_transitions,scene.pairs().size());
}
// Unlike the isolated-mover benchmarks, this workload carries many
// unchanged live contact managers across one sparse crossing. It makes
// candidate reconciliation costs visible without conflating them with
// the cost of reconstructing the entire broadphase membership.
void profile_persistent_pairs() {
    fyzz::native::Scene scene;
    constexpr unsigned pairs=128;
    for(unsigned i=0;i<pairs/2;++i)
        scene.add_static_box({0.5f,0.5f,0.5f},{0,0,0});
    const auto target=scene.add_static_box(
        {0.5f,0.5f,0.5f},{4,0,0});
    for(unsigned i=pairs/2;i<pairs;++i)
        scene.add_static_box({0.5f,0.5f,0.5f},{0,0,0});
    scene.add_dynamic_box({0.5f,0.5f,0.5f},{0,0,0},1.0f);
    const auto mover=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{8,0,0},1.0f);
    scene.advance_collision();
    const auto shifts_before=scene.broadphase_endpoint_local_shift_count();
    const auto crosses_before=scene.broadphase_endpoint_crossing_count();
    const auto sorts_before=scene.broadphase_endpoint_full_sort_count();
    constexpr unsigned trials=5,steps=64;
    std::vector<double> times;
    times.reserve(trials);
    std::size_t transitions=0;
    unsigned frame=0;
    for(unsigned trial=0;trial<trials;++trial) {
        const auto start=Clock::now();
        for(unsigned step=0;step<steps;++step,++frame) {
            scene.set_position(mover,frame%2
                ? fyzz::Vec3{8,0,0}:fyzz::Vec3{4,0,0});
            scene.advance_collision(true);
            transitions+=scene.found_pairs().size()+
                         scene.lost_pairs().size();
        }
        const auto elapsed=std::chrono::duration<double,std::micro>(
            Clock::now()-start).count();
        times.push_back(elapsed/steps);
    }
    std::sort(times.begin(),times.end());
    std::printf(
        "native_sap mode=persistent-pairs pairs=%u"
        " median_us_per_query=%.3f full_sorts=%zu"
        " local_shifts=%zu minmax_crossings=%zu transitions=%zu"
        " final_pairs=%zu target=%u\n",
        pairs,times[trials/2],
        scene.broadphase_endpoint_full_sort_count()-sorts_before,
        scene.broadphase_endpoint_local_shift_count()-shifts_before,
        scene.broadphase_endpoint_crossing_count()-crosses_before,
        transitions,scene.pairs().size(),target);
}
// Full source SAP batch updates must not spend quadratic work on
// overlapping static-only volumes when their membership changes.
void profile_static_membership() {
    fyzz::native::Scene scene;
    constexpr unsigned statics=256;
    for(unsigned i=0;i<statics;++i)
        scene.add_static_box({0.5f,0.5f,0.5f},{0,0,0});
    const auto toggle=scene.add_static_box(
        {0.5f,0.5f,0.5f},{0,0,0});
    scene.advance_collision();
    const auto before=scene.broadphase_endpoint_full_sort_count();
    constexpr unsigned trials=5,steps=64;
    std::vector<double> times;
    times.reserve(trials);
    unsigned frame=0;
    for(unsigned trial=0;trial<trials;++trial) {
        const auto start=Clock::now();
        for(unsigned step=0;step<steps;++step,++frame) {
            scene.set_shape_simulation_enabled(toggle,frame%2!=0);
            scene.advance_collision(true);
            if(!scene.pairs().empty())
                throw std::runtime_error(
                    "Static-only SAP admitted a filtered pair");
        }
        const auto elapsed=std::chrono::duration<double,std::micro>(
            Clock::now()-start).count();
        times.push_back(elapsed/steps);
    }
    std::sort(times.begin(),times.end());
    std::printf(
        "native_sap mode=static-membership boxes=%u"
        " median_us_per_query=%.3f full_sorts=%zu pairs=%zu\n",
        statics,times[trials/2],
        scene.broadphase_endpoint_full_sort_count()-before,
        scene.pairs().size());
}
// Multiple updated volumes in disconnected neighborhoods use activity
// pockets, avoiding whole-axis sorting when the large-update threshold
// is crossed.
void profile_activity_pockets(bool separated) {
    fyzz::native::Scene scene;
    constexpr unsigned statics=192,movers=64;
    const float z=separated?1000.0f:0.0f;
    for(unsigned i=0;i<statics;++i)
        scene.add_static_box({0.4f,0.4f,0.4f},
                             {float(i%32)*3.0f,float(i/32)*3.0f,0});
    std::vector<fyzz::BodyId> ids;
    ids.reserve(movers);
    for(unsigned i=0;i<movers;++i)
        ids.push_back(scene.add_dynamic_box(
            {0.4f,0.4f,0.4f},
            {float(i%32)*3.0f,float(i/32)*3.0f,z},1.0f));
    scene.advance_collision();
    const auto sorts_before=scene.broadphase_endpoint_full_sort_count();
    const auto pockets_before=scene.broadphase_activity_pocket_count();
    const auto endpoints_before=
        scene.broadphase_activity_pocket_endpoint_count();
    constexpr unsigned trials=5,steps=64;
    std::vector<double> times;
    times.reserve(trials);
    unsigned frame=0;
    for(unsigned trial=0;trial<trials;++trial) {
        const auto start=Clock::now();
        for(unsigned step=0;step<steps;++step,++frame) {
            for(unsigned i=0;i<movers;++i) {
                const float dx=(frame&1u)?0.25f:-0.25f;
                scene.set_position(ids[i],
                    {float(i%32)*3.0f+dx,float(i/32)*3.0f,z});
            }
            scene.advance_collision(true);
        }
        times.push_back(std::chrono::duration<double,std::micro>(
            Clock::now()-start).count()/steps);
    }
    std::sort(times.begin(),times.end());
    std::printf(
        "native_sap mode=%s boxes=%u updated=%u"
        " median_us_per_query=%.3f full_sorts=%zu pockets=%zu"
        " pocket_endpoints=%zu pairs=%zu\n",
        separated?"activity-pockets-separated":
                  "activity-pockets-contacting",
        statics+movers,movers,times[trials/2],
        scene.broadphase_endpoint_full_sort_count()-sorts_before,
        scene.broadphase_activity_pocket_count()-pockets_before,
        scene.broadphase_activity_pocket_endpoint_count()-endpoints_before,
        scene.pairs().size());
}
} // namespace

int main() {
    try {
        profile("stationary",Motion::Stationary,0);
        profile("local",Motion::Local,0);
        profile("contact-local",Motion::LocalContacts,0);
        profile("teleport",Motion::Teleport,0);
        profile("retired-history",Motion::Stationary,2048);
        profile_persistent_pairs();
        profile_static_membership();
        profile_activity_pockets(false);
        profile_activity_pockets(true);
        fyzz::native::NativeSapTimingFixture::profile(false);
        fyzz::native::NativeSapTimingFixture::profile(true);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"native SAP benchmark: %s\n",error.what());
        return 1;
    }
}
