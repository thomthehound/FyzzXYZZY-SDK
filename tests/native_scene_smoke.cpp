// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "native/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
#include "PxPhysicsAPI.h"
#include "extensions/PxDistanceJoint.h"
#include "extensions/PxSimpleFactory.h"
#endif

// Exercise the actual production Dy preparation after installing a Gu
// contact stream with noncontiguous, nearly aligned normal runs.
// Ordinary admitted box-box/plane-box Gu managers usually supply one
// exact normal, so a purely geometric fixture would miss the source
// header-normal rule when correlation joins distinct runs.
namespace fyzz::native {
// Fixture probes ONLY the production GPU-report/Sc/Dy admission contract.
// It does not implement or claim the GPU SAP kernel, and no native CPU
// pair-production path is invoked by an attached GPU source owner.
struct GpuSapSceneAdmissionFixture {
    struct FetchedReports final : GpuBroadphaseProducer {
        std::vector<Pair> found,lost;
        std::size_t fetches=0,rollbacks=0;
        bool overflow=false;
        GpuBroadphaseReports advance(
            std::span<const GpuBroadphaseBox> boxes,
            bool,bool) override {
            ++fetches;
            for(const auto& b:boxes)
                if(b.id==std::numeric_limits<BodyId>::max())
                    throw std::runtime_error("Invalid GPU shape input");
            return {found,lost,overflow};
        }
        void invalidate_after_rollback() noexcept override {
            ++rollbacks;
        }
        void set(std::vector<Pair> next_found,std::vector<Pair> next_lost) {
            found=std::move(next_found);
            lost=std::move(next_lost);
        }
    };
    static void verify() {
        Scene scene;
        auto producer=std::make_unique<FetchedReports>();
        auto* gpu=producer.get();
        scene.set_gpu_broadphase(std::move(producer));
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
        const auto b=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0.4f,1.0f,0.0f});
        const Pair pair={std::min(a,b),std::max(a,b)};
        gpu->set({pair},{});
        scene.advance_collision();
        if(scene.found_pairs().size()!=1 || scene.found_pairs()[0]!=pair ||
           scene.pairs().size()!=1 || scene.contacts().empty())
            throw std::runtime_error(
                "GPU SAP found report did not enter native Sc/Gu");
        if(scene.broadphase_endpoint_full_sort_count()!=0 ||
           !scene.sap_volumes_.empty())
            throw std::runtime_error(
                "GPU SAP admission silently executed native CPU broadphase");
        gpu->set({},{});
        scene.advance_collision();
        if(!scene.found_pairs().empty() ||
           !scene.lost_pairs().empty() || scene.pairs().size()!=1)
            throw std::runtime_error(
                "GPU SAP quiet frame lost Sc contact-manager identity");
        // Invalid fetched reports must fail atomically and invalidate
        // source GPU histories before the next real publication.
        gpu->set({pair},{});
        bool rejected=false;
        try { scene.advance_collision(); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected || gpu->rollbacks!=1 ||
           scene.pairs().size()!=1 || scene.pairs()[0]!=pair)
            throw std::runtime_error(
                "Rejected GPU SAP report escaped Sc rollback");

        // Source PxgCudaBroadPhaseSap::fetchBroadPhaseResults must reject
        // over-capacity reports before Sc touches a single manager edge.
        gpu->set({},{});
        gpu->overflow=true;
        rejected=false;
        try { scene.advance_collision(); }
        catch(const std::overflow_error&) { rejected=true; }
        gpu->overflow=false;
        if(!rejected || gpu->rollbacks!=2 ||
           scene.pairs().size()!=1 || scene.pairs()[0]!=pair)
            throw std::runtime_error(
                "GPU SAP overflow published a partial Sc frame");
        scene.set_position(b,{8.0f,1.0f,0.0f});
        gpu->set({},{pair});
        scene.advance_collision();
        if(!scene.pairs().empty() ||
           scene.lost_pairs().size()!=1 || scene.lost_pairs()[0]!=pair ||
           !scene.contacts().empty() || gpu->fetches!=5)
            throw std::runtime_error(
                "GPU SAP lost report failed Sc contact retirement");
        if(scene.broadphase_endpoint_full_sort_count()!=0 ||
           !scene.sap_volumes_.empty())
            throw std::runtime_error(
                "GPU SAP retirement called native CPU SAP");

        // A GPU-produced found report must reach the actual N1 Dy step,
        // not just the collision-only Gu query path. The source manager
        // and native solver are still those owned by this same Scene.
        scene.set_position(b,{0.4f,1.0f,0.0f});
        gpu->set({pair},{});
        scene.step(1.0f/60.0f);
        if(scene.pairs().size()!=1 || scene.pairs()[0]!=pair ||
           scene.contacts().empty() || gpu->fetches!=6 ||
           scene.broadphase_endpoint_full_sort_count()!=0 ||
           !scene.sap_volumes_.empty())
            throw std::runtime_error(
                "GPU SAP found pair failed native N1 solver publication");

        // A shape can move while the same pair remains overlapping.
        // The GPU owner must fetch that frame and PCM must refresh, but
        // the published Sc pair list and contact edge are unchanged.
        const auto edge=scene.contacts().front().edge_handle;
        const auto pending_size=scene.pending_pairs_.size();
        gpu->set({},{});
        scene.set_position(b,{0.5f,1.0f,0.0f});
        scene.advance_collision();
        if(scene.pairs().size()!=1 || scene.pairs()[0]!=pair ||
           scene.contacts().empty() ||
           scene.contacts().front().edge_handle!=edge ||
           !scene.found_pairs().empty() || !scene.lost_pairs().empty() ||
           scene.pending_pairs_.size()!=pending_size ||
           gpu->fetches!=7 ||
           scene.broadphase_endpoint_full_sort_count()!=0 ||
           !scene.sap_volumes_.empty())
            throw std::runtime_error(
                "Quiet GPU SAP membership rebuilt Sc pairs or skipped moving PCM contacts");
    }

    static void verify_world_installation() {
        // The public World takes an already constructed source GPU
        // owner at its Sc backend boundary. No implicit CPU SAP is
        // selected, and malformed GPU reports reach the real rollback.
        bool rejected=false;
        try {
            (void)create_native_gpu_broadphase_backend(
                {},false,false,true,true);
        }catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)
            throw std::runtime_error(
                "Native GPU world admitted a missing source broadphase owner");

        auto producer=std::make_unique<FetchedReports>();
        auto* source=producer.get();
        World world(create_native_gpu_broadphase_backend(
            std::move(producer),false,false,true,true));
        const BodyId a=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,1.0f,0.0f},1.0f);
        const BodyId b=world.add_static_box(
            {0.5f,0.5f,0.5f},{0.4f,1.0f,0.0f});
        const Pair pair{std::min(a,b),std::max(a,b)};
        source->set({pair},{});
        world.step(1.0f/60.0f);
        if(source->fetches!=1u)
            throw std::runtime_error(
                "World did not advance the explicitly installed GPU SAP owner");

        source->set({pair},{});
        rejected=false;
        try {world.step(1.0f/60.0f);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected || source->rollbacks!=1u)
            throw std::runtime_error(
                "World failed to invalidate source GPU history after Sc rejection");

        source->set({},{});
        world.step(1.0f/60.0f);
        world.set_body_position(b,{8.0f,1.0f,0.0f});
        source->set({},{pair});
        world.step(1.0f/60.0f);
        if(source->fetches!=4u)
            throw std::runtime_error(
                "World skipped a quiet or lost source GPU SAP frame");
    }
};
struct SourceSapEndpointFixture {
    static void check(Scene& scene) {
        const auto& links_next=scene.sap_list_next_;
        const auto& links_prev=scene.sap_list_prev_;
        const std::size_t n=scene.sap_axis_endpoints_[0].size();
        if(!links_next.empty() || !links_prev.empty()) {
            if(links_next.size()!=n+1 || links_prev.size()!=n+1)
                throw std::runtime_error(
                    "Persistent SAP linked scratch has stale endpoint extent");
            for(std::size_t i=0;i<n;++i)
                if(links_prev[i]!=(i?i-1:n) ||
                   links_next[i]!=(i+1<n?i+1:n))
                    throw std::runtime_error(
                        "SAP activity pocket failed to restore linked indices");
            if(n && (links_next[n]!=0 || links_prev[n]!=n-1))
                throw std::runtime_error(
                    "SAP activity pocket failed to restore its sentinel");
        }
        const auto& actors=scene.actors_;
        std::vector<std::array<unsigned,2>> counts(actors.size());
        std::size_t admitted=0;
        for(const auto& actor:actors)
            if(actor.alive && actor.simulation_shape_enabled &&
               actor.kind!=Scene::Kind::StaticPlane)
                ++admitted;
        if(scene.sap_volumes_.size()!=admitted)
            throw std::runtime_error(
                "Source SAP previous-bounds registry diverged from live boxes");
        for(const auto& volume:scene.sap_volumes_)
            if(!actors[volume.id].alive ||
               !actors[volume.id].simulation_shape_enabled ||
               volume.published_bounds!=actors[volume.id].encoded_bounds)
                throw std::runtime_error(
                    "Source SAP AABB witness not owned by live box");
        for(unsigned axis=0;axis<3;++axis) {
            const auto& endpoints=scene.sap_axis_endpoints_[axis];
            if(endpoints.size()!=admitted*2)
                throw std::runtime_error(
                    "Source SAP axis missing min/max endpoints");
            for(std::size_t index=0;index<endpoints.size();++index) {
                const auto& ep=endpoints[index];
                if(ep.id>=actors.size() ||
                   !actors[ep.id].alive ||
                   !actors[ep.id].simulation_shape_enabled ||
                   actors[ep.id].kind==Scene::Kind::StaticPlane ||
                   ep.value!=actors[ep.id].encoded_bounds[
                       axis+(ep.is_max?3:0)])
                    throw std::runtime_error(
                        "Source SAP axis endpoint differs from encoded AABB");
                if(ep.volume_slot>=scene.sap_volumes_.size() ||
                   scene.sap_volumes_[ep.volume_slot].id!=ep.id ||
                   scene.sap_volumes_[ep.volume_slot].
                       endpoint_positions[axis][ep.is_max?1u:0u]!=index)
                    throw std::runtime_error(
                        "PhysX SapBox1D min/max index mapping is stale");
                if(index>0) {
                    const auto& prev=endpoints[index-1];
                    const bool wrong_key=prev.value>ep.value;
                    const bool wrong_type=prev.value==ep.value &&
                        prev.is_max && !ep.is_max;
                    const bool wrong_id=prev.value==ep.value &&
                        prev.is_max==ep.is_max && prev.id>=ep.id;
                    if(wrong_key || wrong_type || wrong_id)
                        throw std::runtime_error(
                            "Source SAP min/max endpoint order broken");
                }
                ++counts[ep.id][ep.is_max?1:0];
            }
            for(std::size_t id=0;id<actors.size();++id) {
                const auto& actor=actors[id];
                const unsigned expected=
                    actor.alive && actor.simulation_shape_enabled &&
                    actor.kind!=Scene::Kind::StaticPlane ? 1u:0u;
                if(counts[id][0]!=expected* (axis+1) ||
                   counts[id][1]!=expected* (axis+1))
                    throw std::runtime_error(
                        "Source SAP min/max owner multiplicity diverged");
            }
        }
    }
    static void verify_kinematic_sap_refilter() {
        // Refitting the same PxRigidDynamic as kinematic changes the BP
        // group filter, not its Sc contact manager identity. Existing
        // dynamic/kinematic contact edges must retain their handles.
        Scene scene;
        const BodyId a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0,0},1.0f);
        const BodyId b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0,0},1.0f);
        const BodyId s=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,0,0});
        const Pair ab{a,b},as{a,s},bs{b,s};
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=3)
            throw std::runtime_error(
                "SAP rigid refilter fixture failed initial admission");
        const auto persistent=scene.contact_edge_index(ab);
        scene.set_kinematic(b,true);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=2 ||
           !std::binary_search(scene.pairs().begin(),
                               scene.pairs().end(),ab) ||
           !std::binary_search(scene.pairs().begin(),
                               scene.pairs().end(),as) ||
           scene.contact_edge_index(ab)!=persistent ||
           scene.lost_pairs().size()!=1 ||
           !(scene.lost_pairs()[0]==bs))
            throw std::runtime_error(
                "SAP kinematic refilter replaced a persistent contact manager");
        scene.set_position(a,{7,0,0});
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(!scene.pairs().empty() || scene.lost_pairs().size()!=2)
            throw std::runtime_error(
                "SAP moving dynamics failed to retire kinematic pairs");
        scene.set_position(a,{0,0,0});
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=2 || scene.found_pairs().size()!=2)
            throw std::runtime_error(
                "SAP kinematic contact re-admission failed");
        scene.set_kinematic(b,false);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=3 ||
           scene.found_pairs().size()!=1 ||
           !(scene.found_pairs()[0]==bs))
            throw std::runtime_error(
                "SAP dynamic refilter missed the static pair");
        scene.set_shape_simulation_enabled(b,false);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=1 ||
           !(scene.pairs()[0]==as) ||
           scene.lost_pairs().size()!=2)
            throw std::runtime_error(
                "SAP disabled kinematic-capable shape retained overlap");
        scene.set_shape_simulation_enabled(b,true);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=3 ||
           scene.found_pairs().size()!=2)
            throw std::runtime_error(
                "SAP re-enabled rigid shape lost pair admission");
    }
    static void verify_pair_manager_states() {
        // Exercise source PAIR_NEW/PAIR_REMOVED cancellation and the
        // packed hash-chain swap-last removal directly. This catches
        // state-machine regressions that final AABB equality cannot.
        Scene::SapPairManager manager;
        std::vector<Pair> collisions;
        for(BodyId id=1;id<20000 && collisions.size()<40;++id) {
            const Pair pair{0,id};
            if((Scene::SapPairManager::hash(pair)&63u)==0u)
                collisions.push_back(pair);
        }
        if(collisions.size()!=40)
            throw std::runtime_error(
                "SAP hash-chain fixture did not find enough collisions");
        std::vector<Pair> created,deleted;
        for(Pair pair:collisions)manager.add(pair);
        manager.finish(created,deleted);
        if(created.size()!=collisions.size() || !deleted.empty())
            throw std::runtime_error(
                "SAP new-pair publication did not preserve collisions");
        // Remove then re-add in one update leaves an old pair alive.
        for(std::size_t i=0;i<collisions.size();i+=2) {
            manager.remove(collisions[i]);
            manager.add(collisions[i]);
        }
        // Created and removed before publication never enter Sc.
        const Pair transient{1,19999};
        manager.add(transient);
        manager.remove(transient);
        manager.finish(created,deleted);
        if(!created.empty() || !deleted.empty() ||
           manager.find(transient)!=Scene::SapPairManager::invalid)
            throw std::runtime_error(
                "SAP same-frame pair cancellation leaked an event");
        // Force removal from middle of a long collision chain. Its last
        // packed pair must relink at the moved index, not the old slot.
        for(std::size_t i=0;i<collisions.size();i+=2)
            manager.remove(collisions[i]);
        manager.finish(created,deleted);
        if(!created.empty() || deleted.size()!=collisions.size()/2)
            throw std::runtime_error(
                "SAP collision-chain retirement lost deletion reports");
        for(std::size_t i=0;i<collisions.size();++i) {
            const auto index=manager.find(collisions[i]);
            if((index!=Scene::SapPairManager::invalid)!=(i%2==1))
                throw std::runtime_error(
                    "SAP swap-last retirement corrupted the hash chain");
        }
        // A full update removes unmatched old pairs and retains only
        // specifically reaffirmed members with their persistent keys.
        manager.begin_full();
        const Pair novel{2,19998};
        manager.add(collisions[1]);
        manager.add(novel);
        manager.finish(created,deleted);
        if(created.size()!=1 || created[0]!=novel ||
           deleted.size()!=collisions.size()/2-1 ||
           manager.active_pairs.size()!=2 ||
           manager.find(collisions[1])==Scene::SapPairManager::invalid ||
           manager.find(novel)==Scene::SapPairManager::invalid)
            throw std::runtime_error(
                "SAP full-update pair-state reconciliation diverged");

        // Full-width actor IDs must not collapse into the same hash
        // simply because they share their low 16 bits.
        constexpr std::array<Pair,4> high_ids{{
            {1,2},{1,65538},{1,131074},{65537,131074}}};
        Scene::SapPairManager wide;
        for(std::size_t i=0;i<high_ids.size();++i) {
            for(std::size_t j=0;j<i;++j)
                if(Scene::SapPairManager::hash(high_ids[i])==
                   Scene::SapPairManager::hash(high_ids[j]))
                    throw std::runtime_error(
                        "SAP full-width actor IDs collapsed into one hash");
            wide.add(high_ids[i]);
        }
        wide.finish(created,deleted);
        if(created.size()!=high_ids.size() || !deleted.empty())
            throw std::runtime_error(
                "SAP full-width pair creation lost publication");
        wide.remove(high_ids[1]);
        wide.remove(high_ids[3]);
        wide.finish(created,deleted);
        if(!created.empty() || deleted.size()!=2 ||
           wide.find(high_ids[0])==Scene::SapPairManager::invalid ||
           wide.find(high_ids[2])==Scene::SapPairManager::invalid ||
           wide.find(high_ids[1])!=Scene::SapPairManager::invalid ||
           wide.find(high_ids[3])!=Scene::SapPairManager::invalid)
            throw std::runtime_error(
                "SAP full-width pair retirement corrupted a hash chain");
    }
    static void check_pair_registry(const Scene& scene) {
        if(scene.sap_pair_registry_dirty_)
            throw std::runtime_error(
                "SAP hash not synchronized with completed publication");
        std::vector<Pair> expected;
        for(const auto pair:scene.pairs())
            if(scene.actors_[pair.a].kind!=Scene::Kind::StaticPlane &&
               scene.actors_[pair.b].kind!=Scene::Kind::StaticPlane)
                expected.push_back(pair);
        auto actual=scene.sap_pairs_.active_pairs;
        std::sort(actual.begin(),actual.end());
        if(actual!=expected)
            throw std::runtime_error(
                "Hashed SAP pair ownership differs from Sc edge publication");
        for(std::size_t i=0;i<expected.size();++i) {
            const auto index=scene.sap_pairs_.find(expected[i]);
            if(index==Scene::SapPairManager::invalid ||
               scene.sap_pairs_.states[index]!=0)
                throw std::runtime_error(
                    "SAP persistent pair missing or retains transient state");
        }
        if(!scene.sap_pairs_.touched.empty())
            throw std::runtime_error(
                "SAP pair manager retained an uncommitted activity entry");
        for(std::size_t i=0;i<scene.sap_pairs_.active_pairs.size();++i)
            if(scene.sap_pairs_.find(scene.sap_pairs_.active_pairs[i])!=i)
                throw std::runtime_error(
                    "SAP compacted pair hash chain has stale slot index");
    }
    static void verify_crossing_stress() {
        // Independent final-state oracle derived directly from source
        // BpIntegerAABB encoded bounds, rather than invoking the SAP
        // candidate generator under test. Rotate and move several
        // source volumes at once, forcing sparse crossing events,
        // multi-axis changes, and full-update fallback in one test.
        Scene scene;
        std::vector<BodyId> statics,dynamics;
        for(int i=0;i<12;++i)
            statics.push_back(scene.add_static_box(
                {0.35f+0.08f*float(i%3),0.4f,0.35f},
                {float((i%4)-2)*2.0f,float(i/4-1)*2.0f,0}));
        for(int i=0;i<9;++i)
            dynamics.push_back(scene.add_dynamic_box(
                {0.32f+0.06f*float(i%4),0.4f,0.35f},
                {float((i%3)-1)*1.6f,float(i/3-1)*1.6f,0},1.0f));
        const BodyId floor=scene.add_static_floor(-5.0f);
        scene.advance_collision();
        std::vector<Pair> previous;
        const auto verify_pairs=[&]() {
            std::vector<Pair> expected;
            for(std::size_t i=0;i<scene.actors_.size();++i) {
                const auto& a=scene.actors_[i];
                if(!a.alive || !a.simulation_shape_enabled)continue;
                for(std::size_t j=i+1;j<scene.actors_.size();++j) {
                    const auto& c=scene.actors_[j];
                    if(!c.alive || !c.simulation_shape_enabled ||
                       (a.kind!=Scene::Kind::DynamicBox &&
                        c.kind!=Scene::Kind::DynamicBox))continue;
                    if(a.kind!=Scene::Kind::StaticPlane &&
                       c.kind!=Scene::Kind::StaticPlane) {
                        const auto& aa=a.encoded_bounds;
                        const auto& bb=c.encoded_bounds;
                        if(aa[0]>bb[3] || bb[0]>aa[3] ||
                           aa[1]>bb[4] || bb[1]>aa[4] ||
                           aa[2]>bb[5] || bb[2]>aa[5])continue;
                    }
                    expected.push_back({
                        static_cast<BodyId>(i),static_cast<BodyId>(j)});
                }
            }
            std::sort(expected.begin(),expected.end());
            const auto actual=scene.pairs();
            check_pair_registry(scene);
            if(!std::equal(actual.begin(),actual.end(),
                           expected.begin(),expected.end()))
                throw std::runtime_error(
                    "Three-axis source crossing pairs differ from encoded AABB oracle");
            std::vector<Pair> found,lost;
            std::set_difference(expected.begin(),expected.end(),
                                previous.begin(),previous.end(),
                                std::back_inserter(found));
            std::set_difference(previous.begin(),previous.end(),
                                expected.begin(),expected.end(),
                                std::back_inserter(lost));
            const auto created=scene.found_pairs();
            const auto removed=scene.lost_pairs();
            if(!std::equal(created.begin(),created.end(),
                           found.begin(),found.end()) ||
               !std::equal(removed.begin(),removed.end(),
                           lost.begin(),lost.end()))
                throw std::runtime_error(
                    "Three-axis source crossing Found/Lost differs from encoded AABB oracle");
            previous=std::move(expected);
        };
        verify_pairs();
        const auto initial_crosses=
            scene.broadphase_endpoint_crossing_count();
        for(int frame=0;frame<104;++frame) {
            for(int i=0;i<9;++i) {
                if(frame%13!=0 && (frame+3*i)%9!=0)continue;
                const float x=float((frame*7+i*11)%21-10)*0.57f;
                const float y=float((frame*5+i*13)%17-8)*0.55f;
                const float z=float((frame*3+i*7)%9-4)*0.34f;
                const float a=float((frame*3+i*5)%17-8)*0.1f;
                const fyzz::Quat q{0,0,std::sin(a),std::cos(a)};
                scene.set_pose(dynamics[static_cast<std::size_t>(i)],
                               {x,y,z},q);
            }
            if(frame%19==0) {
                const auto id=statics[3];
                scene.set_position(id,{float(frame%5-2)*0.65f,0,0});
            }
            if(frame==29 || frame==37)
                scene.set_shape_simulation_enabled(
                    dynamics[5],frame==37);
            if(frame==48 || frame==53)
                scene.set_shape_simulation_enabled(
                    floor,frame==53);
            if(frame==71)scene.remove(statics[10]);
            if(frame==72)scene.add_static_box(
                {0.5f,0.5f,0.5f},{0,0,0});
            scene.advance_collision();
            check(scene);
            verify_pairs();
        }
        if(scene.broadphase_endpoint_crossing_count()==initial_crosses)
            throw std::runtime_error(
                "Three-axis SAP stress oracle lacked source endpoint crossings");
    }
    static void verify_adversarial_activity_pockets() {
        // Reorder many endpoints on all three axes in every frame.
        // Reverse direction, share encoded keys, and teleport across
        // multiple pocket ranges. The encoded AABB set is independent
        // of the linked reorder under test.
        Scene scene;
        for(int i=0;i<40;++i)
            scene.add_static_box({0.48f,0.43f,0.47f},
                {float(i%8-4)*1.2f,float(i/8-2)*1.1f,0});
        std::array<BodyId,32> moving{};
        for(int i=0;i<32;++i)
            moving[i]=scene.add_dynamic_box(
                {0.36f,0.39f,0.35f},
                {float(i%8-4)*1.2f,float(i/8-2)*1.1f,0},1.0f);
        scene.advance_collision();
        check(scene);
        std::vector<Pair> previous(scene.pairs().begin(),scene.pairs().end());
        const auto original_sorts=scene.broadphase_endpoint_full_sort_count();
        const auto original_pockets=scene.broadphase_activity_pocket_count();
        for(int frame=0;frame<80;++frame) {
            for(int i=0;i<32;++i) {
                const int phase=(frame&1)?1:-1;
                const float x=float(((frame*7+i*13)%27)-13)*0.42f*phase;
                const float y=float(((frame*11+i*5)%23)-11)*0.34f;
                const float z=float(((frame*3+i*9)%15)-7)*0.23f;
                const float angle=float((frame+i)%9-4)*0.13f;
                const fyzz::Quat q{0,0,std::sin(angle),std::cos(angle)};
                scene.set_pose(moving[static_cast<std::size_t>(i)],
                               {x,y,z},q);
            }
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            std::vector<Pair> expected;
            for(BodyId i=0;i<scene.actors_.size();++i)
                for(BodyId j=i+1;j<scene.actors_.size();++j) {
                    const auto& a=scene.actors_[i];
                    const auto& b=scene.actors_[j];
                    if(a.kind!=Scene::Kind::DynamicBox &&
                       b.kind!=Scene::Kind::DynamicBox)continue;
                    const auto& aa=a.encoded_bounds;
                    const auto& bb=b.encoded_bounds;
                    if(aa[0]>bb[3] || bb[0]>aa[3] ||
                       aa[1]>bb[4] || bb[1]>aa[4] ||
                       aa[2]>bb[5] || bb[2]>aa[5])continue;
                    expected.push_back({i,j});
                }
            const auto published=scene.pairs();
            if(!std::equal(expected.begin(),expected.end(),
                           published.begin(),published.end()))
                throw std::runtime_error(
                    "Linked pocket reorder changed encoded AABB membership");
            std::vector<Pair> found,lost;
            std::set_difference(expected.begin(),expected.end(),
                                previous.begin(),previous.end(),
                                std::back_inserter(found));
            std::set_difference(previous.begin(),previous.end(),
                                expected.begin(),expected.end(),
                                std::back_inserter(lost));
            const auto born=scene.found_pairs(),dead=scene.lost_pairs();
            if(!std::equal(found.begin(),found.end(),
                           born.begin(),born.end()) ||
               !std::equal(lost.begin(),lost.end(),
                           dead.begin(),dead.end()))
                throw std::runtime_error(
                    "Linked pocket reorder changed SAP Found/Lost publication");
            previous=std::move(expected);
        }
        if(scene.broadphase_activity_pocket_count()==original_pockets ||
           scene.broadphase_endpoint_full_sort_count()!=original_sorts)
            throw std::runtime_error(
                "Adversarial multi-volume changes bypassed activity pockets");
    }
    static void verify_orthogonal_bulk_crossing_filter() {
        // A dense batch crosses many x-axis endpoints, but the
        // orthogonal shape bounds reject all new finite pairs.
        // A previously published pair must still report Lost when
        // its own x and z intervals separate simultaneously.
        // Dynamics occupy distinct y lanes, ensuring transient
        // sequential endpoint shifts cannot create mover/mover
        // candidates that would obscure orthogonal rejection.
        Scene scene;
        constexpr unsigned fixed=192,moving=64;
        for(unsigned i=0;i<fixed;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,0});
        std::array<BodyId,moving> dynamic{};
        for(unsigned i=0;i<moving;++i)
            dynamic[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i)*3.0f,
                 i==0?0.0f:1000.0f},1.0f);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=1 ||
           !(scene.pairs()[0]==Pair{0,dynamic[0]}))
            throw std::runtime_error(
                "Orthogonal SAP filter fixture failed initial admission");
        const Pair prior=scene.pairs()[0];
        for(unsigned frame=0;frame<20;++frame) {
            const float dx=(frame&1u)?-3.25f:3.25f;
            for(unsigned i=0;i<moving;++i)
                scene.set_position(dynamic[i],
                    {float(i%32)*3.0f+dx,
                     float(i)*3.0f,1000.0f});
            const auto crossings=scene.broadphase_endpoint_crossing_count();
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            if(!scene.pairs().empty() || !scene.found_pairs().empty())
                throw std::runtime_error(
                    "Orthogonal SAP bulk crossings created false 3D pairs");
            if(frame==0) {
                if(scene.lost_pairs().size()!=1 ||
                   !(scene.lost_pairs()[0]==prior))
                    throw std::runtime_error(
                        "Orthogonal SAP filter suppressed required Lost pair");
            } else if(!scene.lost_pairs().empty() ||
                      scene.broadphase_endpoint_crossing_count()!=crossings)
                throw std::runtime_error(
                    "Orthogonally separated SAP batch retained crossing candidates");
        }
    }
    static void verify_comoving_transient_crossings() {
        // Sequential indexed insertion may pass a not-yet-updated
        // neighbor, despite both bodies retaining their relative
        // order in final source bounds. Such intermediate crossings
        // cannot create net Sc contact managers.
        Scene scene;
        constexpr unsigned fixed=192,moving=64;
        for(unsigned i=0;i<fixed;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,0});
        std::array<BodyId,moving> ids{};
        for(unsigned i=0;i<moving;++i)
            ids[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,1000},1.0f);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        const auto initial_crossings=
            scene.broadphase_endpoint_crossing_count();
        const auto merges_before=scene.broadphase_batch_merge_count();
        for(unsigned frame=0;frame<12;++frame) {
            const float dx=frame&1u?-3.25f:3.25f;
            for(unsigned i=0;i<moving;++i)
                scene.set_position(ids[i],
                    {float(i%32)*3.0f+dx,
                     float(i/32)*3.0f,1000});
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            if(!scene.pairs().empty() ||
               !scene.found_pairs().empty() ||
               !scene.lost_pairs().empty() ||
               scene.broadphase_endpoint_crossing_count()!=initial_crossings)
                throw std::runtime_error(
                    "Co-moving SAP batch leaked temporary crossing candidates");
        }
        if(scene.broadphase_batch_merge_count()==merges_before)
            throw std::runtime_error(
                "Co-moving SAP batch did not exercise bounded endpoint merge");
    }
    static void verify_bounded_merge_pair_transitions() {
        // Dense updates on a predominantly unchanged endpoint stream
        // must produce exactly the final overlap deltas, including Lost
        // pairs, without depending on intermediate endpoint crossings.
        Scene scene;
        constexpr unsigned fixed=192,moving=64;
        for(unsigned i=0;i<fixed;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,0});
        std::array<BodyId,moving> dynamic{};
        for(unsigned i=0;i<moving;++i)
            dynamic[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,1000},1.0f);
        scene.advance_collision();
        std::vector<Pair> previous;
        const auto before=scene.broadphase_batch_merge_count();
        for(unsigned frame=0;frame<16;++frame) {
            const float dx=(frame&1u)?-3.25f:3.25f;
            const float z=(frame%4==2)?1000.0f:0.0f;
            for(unsigned i=0;i<moving;++i)
                scene.set_position(dynamic[i],
                    {float(i%32)*3.0f+dx,
                     float(i/32)*3.0f,z});
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            std::vector<Pair> expected;
            for(BodyId i=0;i<fixed+moving;++i)
                for(BodyId j=i+1;j<fixed+moving;++j) {
                    const auto& a=scene.actors_[i];
                    const auto& b=scene.actors_[j];
                    if(a.kind!=Scene::Kind::DynamicBox &&
                       b.kind!=Scene::Kind::DynamicBox)continue;
                    const auto& x=a.encoded_bounds;
                    const auto& y=b.encoded_bounds;
                    if(x[0]>y[3] || y[0]>x[3] ||
                       x[1]>y[4] || y[1]>x[4] ||
                       x[2]>y[5] || y[2]>x[5])continue;
                    expected.push_back({i,j});
                }
            const auto actual=scene.pairs();
            if(!std::equal(expected.begin(),expected.end(),
                           actual.begin(),actual.end()))
                throw std::runtime_error(
                    "Bounded SAP merge diverged from three-axis overlap oracle");
            std::vector<Pair> found,lost;
            std::set_difference(expected.begin(),expected.end(),
                                previous.begin(),previous.end(),
                                std::back_inserter(found));
            std::set_difference(previous.begin(),previous.end(),
                                expected.begin(),expected.end(),
                                std::back_inserter(lost));
            const auto born=scene.found_pairs(),dead=scene.lost_pairs();
            if(!std::equal(found.begin(),found.end(),
                           born.begin(),born.end()) ||
               !std::equal(lost.begin(),lost.end(),
                           dead.begin(),dead.end()))
                throw std::runtime_error(
                    "Bounded SAP merge changed Found/Lost transitions");
            previous=std::move(expected);
        }
        if(scene.broadphase_batch_merge_count()==before)
            throw std::runtime_error(
                "Bounded SAP merge transition regression missed merge route");
    }
    static void verify_stable_multi_update_pair_lifetime() {
        // A large mUpdated batch can move without crossing any indexed
        // endpoints. Its finite SAP overlaps, source EdgeIndices and
        // Found/Lost streams must remain unchanged; no linked pocket
        // reconstruction or full pair sweep is needed.
        Scene scene;
        constexpr unsigned fixed=192,moving=64;
        for(unsigned i=0;i<fixed;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i%32)*3.0f,float(i/32)*3.0f,0});
        std::array<BodyId,moving> ids{};
        for(unsigned i=0;i<moving;++i)
            ids[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},
                {float(i%32)*3.0f+(i<32?0.25f:0.45f),
                 float(i/32)*3.0f,0},1.0f);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        if(scene.pairs().size()!=moving)
            throw std::runtime_error(
                "Stable SAP batch fixture has wrong initial overlap set");
        const std::vector<Pair> initial(scene.pairs().begin(),
                                        scene.pairs().end());
        std::vector<Scene::EdgeIndex> handles;
        for(Pair pair:initial)handles.push_back(scene.contact_edge_index(pair));
        const auto first=scene.broadphase_batch_stable_order_count();
        const auto pockets=scene.broadphase_activity_pocket_count();
        for(unsigned frame=0;frame<32;++frame) {
            const float dx=(frame&1u)?0.32f:0.20f;
            for(unsigned i=0;i<moving;++i)
                scene.set_position(ids[i],
                    {float(i%32)*3.0f+dx+(i<32?0.0f:0.2f),
                     float(i/32)*3.0f,0});
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            if(!std::equal(initial.begin(),initial.end(),
                           scene.pairs().begin(),scene.pairs().end()) ||
               !scene.found_pairs().empty() || !scene.lost_pairs().empty())
                throw std::runtime_error(
                    "Coherent SAP batch published spurious overlap changes");
            for(std::size_t i=0;i<initial.size();++i)
                if(handles[i]!=scene.contact_edge_index(initial[i]))
                    throw std::runtime_error(
                        "Coherent SAP batch replaced an unchanged Sc EdgeIndex");
        }
        if(scene.broadphase_batch_stable_order_count()!=first+32 ||
           scene.broadphase_activity_pocket_count()!=pockets)
            throw std::runtime_error(
                "Coherent SAP batch rebuilt unnecessary activity pockets");
    }
    static void verify_bounded_bulk_shift_publication() {
        // More than a fifth of the SAP volumes move, but each moved
        // endpoint crosses only a bounded group of rank neighbors.
        // This exercises the source-equivalent indexed shift route
        // independently from both the stable and linked-pocket routes.
        Scene scene;
        constexpr unsigned static_count=18, moving_count=6;
        for(unsigned i=0;i<static_count;++i)
            scene.add_static_box({0.4f,0.4f,0.4f},
                {float(i)*3.0f,0,0});
        std::array<BodyId,moving_count> dynamic{};
        for(unsigned i=0;i<moving_count;++i)
            dynamic[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},{float(i)*3.0f,0,0},1.0f);
        scene.advance_collision();
        check(scene);
        check_pair_registry(scene);
        const auto old_pairs=std::vector<Pair>(
            scene.pairs().begin(),scene.pairs().end());
        std::vector<Scene::EdgeIndex> handles;
        for(Pair pair:old_pairs)
            handles.push_back(scene.contact_edge_index(pair));
        const auto old_direct=scene.broadphase_batch_direct_update_count();
        const auto old_pockets=scene.broadphase_activity_pocket_count();
        const auto old_sorts=scene.broadphase_endpoint_full_sort_count();
        for(unsigned frame=0;frame<32;++frame) {
            const float dx=(frame&1u)?0.25f:-0.25f;
            for(unsigned i=0;i<moving_count;++i)
                scene.set_position(dynamic[i],{float(i)*3.0f+dx,0,0});
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            if(!std::equal(old_pairs.begin(),old_pairs.end(),
                           scene.pairs().begin(),scene.pairs().end()) ||
               !scene.found_pairs().empty() ||
               !scene.lost_pairs().empty())
                throw std::runtime_error(
                    "Bounded SAP batch altered persistent overlap publication");
            for(std::size_t i=0;i<old_pairs.size();++i)
                if(handles[i]!=scene.contact_edge_index(old_pairs[i]))
                    throw std::runtime_error(
                        "Bounded SAP batch replaced persistent Sc EdgeIndex");
        }
        if(scene.broadphase_batch_direct_update_count()==old_direct ||
           scene.broadphase_activity_pocket_count()!=old_pockets ||
           scene.broadphase_endpoint_full_sort_count()!=old_sorts)
            throw std::runtime_error(
                "Bounded SAP motion did not retain indexed shift ownership");
    }
    static void verify_source_activity_pockets() {
        Scene scene;
        std::array<BodyId,72> statics{};
        std::array<BodyId,28> moving{};
        for(std::size_t i=0;i<statics.size();++i)
            statics[i]=scene.add_static_box(
                {0.45f,0.45f,0.45f},
                {float(i%12)*2.0f,float(i/12)*2.0f,0});
        for(std::size_t i=0;i<moving.size();++i)
            moving[i]=scene.add_dynamic_box(
                {0.4f,0.4f,0.4f},
                {float(i%12)*2.0f,float(i/12)*2.0f,0},1.0f);
        scene.advance_collision();
        const auto prior_pockets=scene.broadphase_activity_pocket_count();
        const auto prior_shifts=scene.broadphase_endpoint_local_shift_count();
        const auto prior_full=scene.broadphase_endpoint_full_sort_count();
        std::vector<Pair> previous(scene.pairs().begin(),scene.pairs().end());
        for(unsigned frame=0;frame<48;++frame) {
            for(std::size_t i=0;i<moving.size();++i) {
                const float x=float(int((frame*3+i*11)%13)-6)*0.23f;
                const float y=float(int((frame*7+i*3)%17)-8)*0.19f;
                scene.set_position(moving[i],
                    {float(i%12)*2.0f+x,
                     float(i/12)*2.0f+y,0});
            }
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            std::vector<Pair> expected;
            for(BodyId i=0;i<scene.actors_.size();++i)
                for(BodyId j=i+1;j<scene.actors_.size();++j) {
                    const auto& a=scene.actors_[i];
                    const auto& b=scene.actors_[j];
                    if(a.kind!=Scene::Kind::DynamicBox &&
                       b.kind!=Scene::Kind::DynamicBox)continue;
                    const auto& aa=a.encoded_bounds;
                    const auto& bb=b.encoded_bounds;
                    if(aa[0]>bb[3] || bb[0]>aa[3] ||
                       aa[1]>bb[4] || bb[1]>aa[4] ||
                       aa[2]>bb[5] || bb[2]>aa[5])continue;
                    expected.push_back({i,j});
                }
            const auto actual=scene.pairs();
            if(!std::equal(expected.begin(),expected.end(),
                           actual.begin(),actual.end()))
                throw std::runtime_error(
                    "Source activity pockets changed encoded-AABB pair membership");
            std::vector<Pair> found,lost;
            std::set_difference(expected.begin(),expected.end(),
                                previous.begin(),previous.end(),
                                std::back_inserter(found));
            std::set_difference(previous.begin(),previous.end(),
                                expected.begin(),expected.end(),
                                std::back_inserter(lost));
            const auto new_pairs=scene.found_pairs();
            const auto retired_pairs=scene.lost_pairs();
            if(!std::equal(found.begin(),found.end(),
                           new_pairs.begin(),new_pairs.end()) ||
               !std::equal(lost.begin(),lost.end(),
                           retired_pairs.begin(),retired_pairs.end()))
                throw std::runtime_error(
                    "Source activity pocket pair reports diverged");
            previous=std::move(expected);
        }
        if((scene.broadphase_activity_pocket_count()==prior_pockets &&
            scene.broadphase_endpoint_local_shift_count()==prior_shifts) ||
           scene.broadphase_endpoint_full_sort_count()!=prior_full)
            throw std::runtime_error(
                "Batch SAP updates neither reordered activity pockets "
                "nor maintained direct indexed endpoints");
    }
    static void verify_static_only_full_sweep() {
        // Source group filtering must reject static/static pairs even
        // when every encoded bound overlaps. Membership changes force
        // complete endpoint sweeps, not the sparse crossing path.
        Scene scene;
        for(unsigned i=0;i<96;++i)
            scene.add_static_box({0.5f,0.5f,0.5f},{0,0,0});
        const BodyId toggle=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,0,0});
        scene.advance_collision();
        const auto initial_sorts=
            scene.broadphase_endpoint_full_sort_count();
        for(unsigned frame=0;frame<16;++frame) {
            scene.set_shape_simulation_enabled(toggle,frame%2!=0);
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            if(!scene.pairs().empty() ||
               !scene.found_pairs().empty() ||
               !scene.lost_pairs().empty())
                throw std::runtime_error(
                    "Static-only SAP full sweep published filtered pairs");
        }
        if(scene.broadphase_endpoint_full_sort_count()!=initial_sorts+16)
            throw std::runtime_error(
                "Static membership changes skipped required SAP rebuilds");
    }
    static void verify_persistent_overlap_crossings() {
        // A large published contact-manager set with one sparse mover
        // exercises interleaved old/new pair keys. Endpoint crossings
        // must not retire any unaffected pair or change its EdgeIndex.
        Scene scene;
        constexpr std::size_t count=96;
        std::array<BodyId,count> statics{};
        for(std::size_t i=0;i<count;++i)
            statics[i]=scene.add_static_box(
                {0.5f,0.5f,0.5f},{0,0,0});
        const BodyId stable=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0,0},1.0f);
        const BodyId mover=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{7,7,7},1.0f);
        scene.advance_collision();
        if(scene.pairs().size()!=count ||
           scene.found_pairs().size()!=count)
            throw std::runtime_error(
                "SAP persistent-pair fixture failed initial admission");
        std::array<std::uint32_t,count> handles{};
        for(std::size_t i=0;i<count;++i) {
            handles[i]=scene.contact_edge_index({statics[i],stable});
            if(handles[i]==Scene::invalid_edge)
                throw std::runtime_error(
                    "SAP persistent manager missing its initial EdgeIndex");
        }
        const auto initial_sorts=
            scene.broadphase_endpoint_full_sort_count();
        for(unsigned frame=0;frame<24;++frame) {
            const bool overlapping=(frame%2)==0;
            scene.set_position(mover,overlapping
                ? Vec3{0,0,0}:Vec3{7,7,7});
            scene.advance_collision();
            check(scene);
            check_pair_registry(scene);
            const auto published=scene.pairs();
            if(published.size()!=count+(overlapping?count+1:0) ||
               !std::is_sorted(published.begin(),published.end()))
                throw std::runtime_error(
                    "Sparse SAP pair merge lost sorted persistent managers");
            if(scene.found_pairs().size()!=(overlapping?count+1:0) ||
               scene.lost_pairs().size()!=(overlapping?0:count+1))
                throw std::runtime_error(
                    "Sparse SAP crossing published incorrect pair deltas");
            for(std::size_t i=0;i<count;++i) {
                if(scene.contact_edge_index({statics[i],stable})!=handles[i])
                    throw std::runtime_error(
                        "Sparse SAP replaced an unaffected contact manager");
                const Pair new_pair{statics[i],mover};
                if(std::binary_search(published.begin(),published.end(),
                                      new_pair)!=overlapping)
                    throw std::runtime_error(
                        "Sparse SAP missed an interleaved crossing pair");
            }
            if(std::binary_search(published.begin(),published.end(),
                                  Pair{stable,mover})!=overlapping)
                throw std::runtime_error(
                    "Sparse SAP missed the dynamic/dynamic crossing pair");
        }
        if(scene.broadphase_endpoint_full_sort_count()!=initial_sorts)
            throw std::runtime_error(
                "Sparse SAP reconstructed endpoints for one mover");
    }
    static void verify() {
        Scene scene;
        const auto a=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,0,0});
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0,0},1.0f);
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,0,0},1.0f);
        const auto plane=scene.add_static_floor(0.0f);
        scene.advance_collision();
        check(scene); // equal encoded min/max keys across two boxes
        const auto initial_sorts=scene.broadphase_endpoint_full_sort_count();
        scene.set_position(c,{4,0.7f,-0.7f});
        scene.advance_collision();
        check(scene);
        if(scene.broadphase_endpoint_full_sort_count()!=initial_sorts)
            throw std::runtime_error(
                "Small Y/Z AABB edits unnecessarily rebuilt full source SAP");
        scene.set_position(c,{0,0,0});
        scene.advance_collision();
        check(scene);
        scene.set_shape_simulation_enabled(b,false);
        scene.advance_collision();
        check(scene);
        scene.set_shape_simulation_enabled(b,true);
        scene.advance_collision();
        check(scene);
        scene.remove(a);
        scene.advance_collision();
        check(scene);
        scene.remove(plane);
        scene.advance_collision();
        check(scene);
    }
};
// Gu and PGS conformance fixtures select diagnostic legacy contact
// generation explicitly; the default Scene constructor uses PCM.
struct ConstraintIslandSourceFixture {
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    static void verify_original_distance_joint_1d() {
        // Independent PxDistanceJoint with center anchors and equal
        // minimum/maximum distance; same gravity, rigid dimensions,
        // density and PGS timestep as the native source row fixture.
        physx::PxDefaultAllocator allocator;
        physx::PxDefaultErrorCallback errors;
        auto* foundation=PxCreateFoundation(
            PX_PHYSICS_VERSION,allocator,errors);
        if(!foundation)
            throw std::runtime_error("Cannot initialize source foundation");
        auto* physics=PxCreatePhysics(
            PX_PHYSICS_VERSION,*foundation,physx::PxTolerancesScale(),false,nullptr);
        if(!physics)throw std::runtime_error("Cannot initialize source physics");
        physx::PxSceneDesc desc(physics->getTolerancesScale());
        desc.gravity=physx::PxVec3(0.0f);
        desc.solverType=physx::PxSolverType::ePGS;
        auto* dispatcher=physx::PxDefaultCpuDispatcherCreate(1);
        desc.cpuDispatcher=dispatcher;
        desc.filterShader=physx::PxDefaultSimulationFilterShader;
        auto* scene=physics->createScene(desc);
        auto* material=physics->createMaterial(0.5f,0.5f,0.2f);
        auto* left=physx::PxCreateDynamic(*physics,
            physx::PxTransform(physx::PxVec3(0,7,0)),
            physx::PxBoxGeometry(0.5f,0.5f,0.5f),*material,1.0f);
        auto* right=physx::PxCreateDynamic(*physics,
            physx::PxTransform(physx::PxVec3(2,7,0)),
            physx::PxBoxGeometry(0.5f,0.5f,0.5f),*material,1.0f);
        if(!dispatcher || !scene || !material || !left || !right)
            throw std::runtime_error("Cannot admit source distance fixture");
        left->setLinearDamping(0);
        right->setLinearDamping(0);
        left->setAngularDamping(0);
        right->setAngularDamping(0);
        scene->addActor(*left);
        scene->addActor(*right);
        auto* joint=physx::PxDistanceJointCreate(
            *physics,left,physx::PxTransform(physx::PxIdentity),
            right,physx::PxTransform(physx::PxIdentity));
        if(!joint)
            throw std::runtime_error("Original 1D distance joint creation failed");
        joint->setMinDistance(1.0f);
        joint->setMaxDistance(1.0f);
        joint->setTolerance(0.0f);
        joint->setDistanceJointFlag(
            physx::PxDistanceJointFlag::eMIN_DISTANCE_ENABLED,true);
        joint->setDistanceJointFlag(
            physx::PxDistanceJointFlag::eMAX_DISTANCE_ENABLED,true);
        constexpr float dt=1.0f/120.0f;
        scene->simulate(dt);
        if(!scene->fetchResults(true))
            throw std::runtime_error("Original distance joint solve failed");
        const auto source_a=left->getGlobalPose().p;
        const auto source_b=right->getGlobalPose().p;
        const auto velocity_a=left->getLinearVelocity();
        const auto velocity_b=right->getLinearVelocity();
        Scene native{};
        const auto a=native.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=native.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,7,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0;
        d.angular_damping=0;
        native.set_dynamics(a,d);
        native.set_dynamics(b,d);
        native.add_distance_constraint(a,b,1.0f);
        native.step(dt);
        const auto actual_a=native.body_state(a);
        const auto actual_b=native.body_state(b);
        // A failure here is a source conformance failure; do not
        // relax the joint equation or substitute a projection pass.
        if(std::abs(actual_a.position.x-source_a.x)>1.0e-3f ||
           std::abs(actual_b.position.x-source_b.x)>1.0e-3f ||
           std::abs(actual_a.velocity.x-velocity_a.x)>1.0e-3f ||
           std::abs(actual_b.velocity.x-velocity_b.x)>1.0e-3f)
            throw std::runtime_error(
                "Native center-distance 1D row diverged from source PGS");
        joint->release();
        left->release();
        right->release();
        material->release();
        scene->release();
        dispatcher->release();
        physics->release();
        foundation->release();
    }
#endif
    static void verify_source_distance_1d_execution() {
        // Equal-limit distance joint with center anchors, no spring:
        // execute through Dy's existing PGS delta/snapshot pipeline,
        // rather than a second geometric position-projection solver.
        Scene scene{};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,7,0},1.0f);
        RigidBodyDynamics free{};
        free.gravity_enabled=false;
        free.linear_damping=0;
        free.angular_damping=0;
        scene.set_dynamics(a,free);
        scene.set_dynamics(b,free);
        const auto edge=scene.add_distance_constraint(a,b,1.0f);
        const auto sc_id=scene.island_edges_[edge].constraint_id;
        if(sc_id==Scene::invalid_edge ||
           !scene.constraint_records_[sc_id].live ||
           scene.constraint_records_[sc_id].edge!=edge)
            throw std::runtime_error(
                "Source ScConstraintSim lacks separate 1D ownership");
        scene.step(1.0f/120.0f);
        const auto first=scene.body_state(a);
        const auto second=scene.body_state(b);
        const float separation=second.position.x-first.position.x;
        if(scene.island_edges_[edge].type!=Scene::IslandEdgeType::Constraint ||
           scene.accurate_island_count()!=1 ||
           scene.indexed_distance_constraints_.size()!=1 ||
           scene.distance_rows_.size()!=1 ||
           scene.solved_result_frame()!=scene.frame_index() ||
           !std::isfinite(separation) || !(separation<2.0f) ||
           !(first.position.x>0.0f) || !(second.position.x<2.0f) ||
           std::abs(first.velocity.x)>1.0e-3f ||
           std::abs(second.velocity.x)>1.0e-3f)
            throw std::runtime_error(
                "Source 1D biased motion/unbiased velocity publication diverged");
        scene.remove_constraint_edge(edge);
        // The solved 1.0-unit center distance makes 1.0-unit boxes
        // touch, so separate their geometry before asserting that the
        // retired constraint no longer bridges two free islands.
        scene.set_position(b,{6,7,0});
        scene.step(1.0f/120.0f);
        if(!scene.indexed_distance_constraints_.empty() ||
           !scene.distance_rows_.empty() ||
           scene.constraint_records_[sc_id].live ||
           scene.accurate_island_count()!=2)
            throw std::runtime_error(
                "Native 1D distance row survived source edge retirement");
    }
    static void verify_mixed_joint_contact_partitions() {
        // The two floor managers and the 1D edge share one active island.
        // The dynamic/dynamic joint must conflict with both floor rows.
        for(bool reversed:{false,true}) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            scene.add_static_floor(0.0f);
            const Vec3 left{-1.0f,0.49f,0.0f};
            const Vec3 right{1.0f,0.49f,0.0f};
            const auto a=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},reversed?right:left,1.0f);
            const auto b=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},reversed?left:right,1.0f);
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            d.linear_damping=0.0f;
            d.angular_damping=0.0f;
            scene.set_dynamics(a,d);
            scene.set_dynamics(b,d);
            scene.add_distance_constraint(a,b,1.5f);
            scene.step(1.0f/120.0f);
            if(scene.indexed_managers_.size()!=2 ||
               scene.indexed_distance_constraints_.size()!=1 ||
               scene.constraint_descriptors_.size()!=3 ||
               scene.ordered_constraints_.size()!=3 ||
               scene.solver_partition_count()!=2 ||
               scene.solver_tasks_.size()!=1 ||
               scene.solver_tasks_[0].constraint_end!=3 ||
               !scene.ordered_constraints_[0].distance ||
               scene.distance_rows_.size()!=1)
                throw std::runtime_error(
                    "Source 1D and contact rows did not share conflict partitions");
            std::size_t joints=0,contacts=0;
            for(const auto& descriptor:scene.ordered_constraints_)
                if(descriptor.distance)++joints;
                else ++contacts;
            if(joints!=1 || contacts!=2)
                throw std::runtime_error(
                    "Mixed source partition lost a live descriptor");
            for(std::size_t i=0;i<scene.constraint_descriptors_.size();++i) {
                const auto& one=scene.constraint_descriptors_[i];
                const Pair p=one.distance
                    ? scene.island_edges_[scene.indexed_distance_constraints_[one.index]].pair
                    : scene.manifolds_[one.index].pair;
                for(std::size_t j=i+1;
                    j<scene.constraint_descriptors_.size();++j) {
                    if(scene.constraint_partitions_[i]!=
                       scene.constraint_partitions_[j])continue;
                    const auto& two=scene.constraint_descriptors_[j];
                    const Pair q=two.distance
                        ? scene.island_edges_[scene.indexed_distance_constraints_[two.index]].pair
                        : scene.manifolds_[two.index].pair;
                    for(BodyId id:{a,b})
                        if((p.a==id || p.b==id) &&
                           (q.a==id || q.b==id))
                            throw std::runtime_error(
                                "Contact and 1D edge share dynamic body in same partition");
                }
            }
            if(!(std::abs(scene.body_state(a).position.x-
                          scene.body_state(b).position.x)<2.0f))
                throw std::runtime_error(
                    "Joint did not execute with admitted floor contacts");
        }
    }
    static void verify_mixed_partition_overflow_waves() {
        // Thirty-five dynamic 1D edges share one solver body, crossing
        // the 32-bit partition-mask boundary. The floor manager must be
        // reserved AFTER the final dynamic wave, not in partition zero.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        const auto root=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.49f,0.0f},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        scene.set_dynamics(root,d);
        constexpr unsigned spokes=35;
        for(unsigned i=0;i<spokes;++i) {
            const float x=4.0f+3.0f*static_cast<float>(i);
            const auto other=scene.add_dynamic_box(
                {0.2f,0.2f,0.2f},{x,4.0f,0.0f},1.0f);
            scene.set_dynamics(other,d);
            const float dy=4.0f-0.49f;
            scene.add_distance_constraint(root,other,
                std::sqrt(x*x+dy*dy));
        }
        scene.step(1.0f/120.0f);
        if(scene.indexed_distance_constraints_.size()!=spokes ||
           scene.indexed_managers_.size()!=1 ||
           scene.distance_rows_.size()!=spokes ||
           scene.constraint_descriptors_.size()!=spokes+1 ||
           scene.ordered_constraints_.size()!=spokes+1 ||
           scene.solver_partition_count()!=spokes+1 ||
           scene.solver_tasks_.size()!=1 ||
           scene.solver_tasks_[0].constraint_end!=spokes+1 ||
           scene.solver_rows_[root].solver.max_dynamic_partition!=spokes ||
           scene.solver_rows_[root].solver.static_constraint_count!=1)
            throw std::runtime_error(
                "Mixed source partition failed 32-mask overflow or static reservation");
        for(std::size_t i=0;i<spokes;++i)
            if(scene.constraint_partitions_[i]!=i ||
               !scene.ordered_constraints_[i].distance)
                throw std::runtime_error(
                    "Dynamic 1D descriptor overflow lost source partition order");
        if(scene.constraint_partitions_[spokes]!=spokes ||
           scene.ordered_constraints_[spokes].distance)
            throw std::runtime_error(
                "Floor contact was scheduled before the final 1D partition");
    }
    static void verify_mixed_task_local_writeback() {
        // A 128-body contact-only task followed by a two-body
        // contact+1D task must execute, write back and integrate from
        // disjoint source descriptor ranges. Gu force windows remain
        // independent of Dy's task-local partition permutation.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const BodyId floor=scene.add_static_floor(0.0f);
        constexpr BodyId independent=128;
        for(BodyId i=0;i<independent;++i)
            scene.add_dynamic_box({0.5f,0.5f,0.5f},
                {3.0f*static_cast<float>(i),0.49f,0.0f},1.0f);
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{384.0f,0.49f,0.0f},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{387.0f,0.49f,0.0f},1.0f);
        scene.add_distance_constraint(a,b,2.0f);
        for(unsigned frame=0;frame<2;++frame) {
            scene.step(1.0f/120.0f);
            if(scene.solver_tasks_.size()!=2 ||
               scene.solver_tasks_[0].manager_end!=independent ||
               scene.solver_tasks_[1].manager_end!=independent+2 ||
               scene.solver_tasks_[0].distance_end!=0 ||
               scene.solver_tasks_[1].distance_end!=1 ||
               scene.solver_tasks_[0].solver_body_end!=independent ||
               scene.solver_tasks_[1].solver_body_end!=independent+2 ||
               scene.solver_tasks_[0].constraint_end!=independent ||
               scene.solver_tasks_[1].constraint_end!=independent+3 ||
               scene.solver_partition_count()!=3 ||
               scene.distance_rows_.size()!=1 ||
               scene.solved_result_frame()!=scene.frame_index())
                throw std::runtime_error(
                    "Mixed Dy task chains lost their original ownership boundaries");
            const auto forces=scene.solved_normal_impulses();
            if(forces.size()!=scene.contact_points_.size())
                throw std::runtime_error(
                    "Mixed task-local Gu writeback changed force arena size");
            std::size_t first_count=0,last_count=0;
            double first_force=0.0,last_force=0.0;
            for(const auto& manager:scene.manifolds_) {
                if(manager.pair.a!=floor)continue;
                double sum=0.0;
                for(std::size_t i=0;i<manager.count;++i)
                    sum+=forces[manager.contact_offset+i];
                if(manager.pair.b<a) {
                    ++first_count;
                    first_force+=sum;
                } else if(manager.pair.b==a || manager.pair.b==b) {
                    ++last_count;
                    last_force+=sum;
                }
            }
            if(first_count!=independent || last_count!=2 ||
               !std::isfinite(first_force) || !std::isfinite(last_force) ||
               first_force<=0.0 || last_force<=0.0)
                throw std::runtime_error(
                    "Task-local Gu force writeback dropped a separate island chain");
        }
        const auto left=scene.body_state(a);
        const auto right=scene.body_state(b);
        if(!(right.position.x-left.position.x<3.0f))
            throw std::runtime_error(
                "Mixed task-local Dy distance row failed to change rigid poses");
    }
    static void verify_prepublication_constraint_retirement() {
        // An admitted ScConstraintSim has connected EdgeInstances but
        // no accurate IslandSim list member until first/second pass.
        // Releasing it (or its actor) before any publication is legal.
        for(bool remove_actor:{false,true}) {
            Scene scene{};
            const auto a=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,5,0},1.0f);
            const auto b=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5,5,0},1.0f);
            const auto handle=scene.add_distance_constraint(a,b,1.0f);
            if(!scene.island_edges_[handle].connected ||
               scene.island_edges_[handle].indexed_accurate)
                throw std::runtime_error(
                    "Source prepublication constraint edge state incorrect");
            if(remove_actor)scene.remove(b);
            else scene.remove_constraint_edge(handle);
            scene.advance_collision(true);
            if(!scene.active_constraint_handles_.empty() ||
               scene.accurate_island_count()!=(remove_actor?1:2) ||
               scene.speculative_island_count()!=(remove_actor?1:2))
                throw std::runtime_error(
                    "Prepublication constraint retirement leaked island ownership");
        }
    }
    static void verify_post_third_pass_handle_retirement_rollback() {
        // PostThirdPassTask may recycle a removed source EdgeIndex
        // before the Dy continuation completes. A rejected later Dy
        // integration must restore the retiring edge and allocator,
        // including its prior published simulation result.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,10,0},1.0f);
        const float dt=1.0f/120.0f;
        scene.step(dt);
        const auto edge=scene.add_distance_constraint(a,b,3.0f);
        scene.advance_collision();
        scene.remove_constraint_edge(edge);
        const auto frame=scene.frame_index();
        const auto state=scene.body_state(a);
        if(scene.retired_edge_handles_.empty() ||
           !scene.island_edges_[edge].retiring)
            throw std::runtime_error(
                "PostThirdPass retirement fixture lacks a pending edge");
        RigidBodyDynamics extreme{};
        extreme.gravity_enabled=false;
        // PxsBodyCore stores finite squared velocity limits, so the
        // authored cap itself must remain representable in PxReal.
        extreme.max_linear_velocity=1.0e18f;
        extreme.max_angular_velocity=1.0e18f;
        scene.set_dynamics(a,extreme);
        scene.set_velocity(a,{1.0e17f,0,0});
        bool failed=false;
        try {scene.step(1.0e23f);}
        catch(const std::overflow_error&) {failed=true;}
        if(!failed || scene.frame_index()!=frame ||
           !scene.island_edges_[edge].live ||
           !scene.island_edges_[edge].retiring ||
           std::find(scene.retired_edge_handles_.begin(),
                     scene.retired_edge_handles_.end(),edge)==
               scene.retired_edge_handles_.end() ||
           std::find(scene.free_edge_handles_.begin(),
                     scene.free_edge_handles_.end(),edge)!=
               scene.free_edge_handles_.end() ||
           scene.body_state(a).position.x!=state.position.x ||
           scene.body_state(a).position.y!=state.position.y ||
           scene.body_state(a).position.z!=state.position.z)
            throw std::runtime_error(
                "Failed postThirdPass/Dy frame leaked a recycled edge handle");
        // A retry consumes the SAME pending retiring source edge once,
        // with no extra allocator entry or new owner.
        scene.set_velocity(a,{});
        scene.step(dt);
        if(scene.solved_result_frame()!=scene.frame_index() ||
           scene.island_edges_[edge].live ||
           !scene.retired_edge_handles_.empty() ||
           std::count(scene.free_edge_handles_.begin(),
                      scene.free_edge_handles_.end(),edge)!=1)
            throw std::runtime_error(
                "PostThirdPass retry did not commit exactly one edge retirement");
        const auto replacement=scene.add_distance_constraint(a,b,3.0f);
        if(replacement!=edge || !scene.island_edges_[edge].live ||
           scene.island_edges_[edge].retiring)
            throw std::runtime_error(
                "PostThirdPass recycled edge identity before successful commit");
    }
    static void verify_constraint_sleep_deactivation_generations() {
        // Real ScConstraintSim and Dy PGS work through sleepCheck into
        // the following IslandSim third pass. Its typed eCONSTRAINT
        // deactivation output must be exactly one publication, never a
        // persistent list of historic sleep events.
        Scene scene{};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,6,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,6,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        scene.set_dynamics(a,d);
        scene.set_dynamics(b,d);
        const auto edge=scene.add_distance_constraint(a,b,3.0f);
        constexpr float dt=1.0f/120.0f;
        bool saw_retirement=false;
        for(unsigned frame=0;frame<128;++frame) {
            scene.step(dt);
            const auto& accurate=scene.accurate_islands_;
            const auto& speculative=scene.speculative_islands_;
            if(!accurate.deactivating_constraint_edges.empty() ||
               !speculative.deactivating_constraint_edges.empty()) {
                if(saw_retirement ||
                   accurate.deactivating_constraint_edges.size()!=1 ||
                   speculative.deactivating_constraint_edges.size()!=1 ||
                   accurate.deactivating_constraint_edges[0]!=edge ||
                   speculative.deactivating_constraint_edges[0]!=edge ||
                   accurate.active_constraint_edge_count!=0 ||
                   speculative.active_constraint_edge_count!=0 ||
                   !scene.body_is_sleeping(a) || !scene.body_is_sleeping(b))
                    throw std::runtime_error(
                        "Source typed constraint third-pass deactivation duplicated");
                saw_retirement=true;
            }
        }
        if(!saw_retirement ||
           !scene.accurate_islands_.deactivating_constraint_edges.empty() ||
           !scene.speculative_islands_.deactivating_constraint_edges.empty() ||
           scene.solved_result_frame()!=scene.frame_index())
            throw std::runtime_error(
                "Source constraint deactivation history outlived its third pass");
        // NpRigidDynamic::setGlobalPose autowakes even at an identical
        // pose; IslandSim must reactivate both bodies connected by the
        // still-live source 1D edge, without replaying old retirements.
        const auto pose=scene.body_state(a);
        scene.set_pose(a,pose.position,pose.orientation);
        scene.step(dt);
        if(scene.body_is_sleeping(a) || scene.body_is_sleeping(b) ||
           scene.accurate_islands_.active_constraint_edge_count!=1 ||
           scene.speculative_islands_.active_constraint_edge_count!=1 ||
           !scene.accurate_islands_.deactivating_constraint_edges.empty() ||
           !scene.speculative_islands_.deactivating_constraint_edges.empty() ||
           scene.solved_result_frame()!=scene.frame_index())
            throw std::runtime_error(
                "Source waking constraint island retained stale deactivation");
    }
    static void verify_island_node_type_counts() {
        // Island ownership retains per-type counts independently from
        // the aggregate size used for source merge/route decisions.
        // No non-rigid type is faked with a rigid Actor or solver row.
        Scene scene{};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,7,0},1.0f);
        const auto k=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{8,7,0},1.0f);
        scene.set_kinematic(k,true);
        scene.advance_collision();
        const auto check=[&](std::size_t expected) {
            for(const auto* graph:
                {&scene.speculative_islands_,&scene.accurate_islands_}) {
                const auto owner=graph->node_island[a];
                if(owner==Scene::IslandGraph::invalid ||
                   owner>=graph->islands.size())
                    throw std::runtime_error("Rigid node has no source island");
                const auto& island=graph->islands[owner];
                if(!island.live || island.count!=expected ||
                   island.node_type_counts[Scene::rigid_node_type_index]!=
                       expected ||
                   graph->node_type[a]!=Scene::IslandNodeType::RigidBody ||
                   graph->node_type[b]!=Scene::IslandNodeType::RigidBody ||
                   graph->node_type[k]!=Scene::IslandNodeType::RigidBody ||
                   graph->node_island[k]!=Scene::IslandGraph::invalid)
                    throw std::runtime_error(
                        "Typed island aggregate/rigid ownership diverged");
                for(std::size_t type=1;
                    type<Scene::island_node_type_count;++type)
                    if(island.node_type_counts[type]!=0)
                        throw std::runtime_error(
                            "Rigid island fabricated non-rigid node ownership");
            }
        };
        check(1);
        const auto edge=scene.add_distance_constraint(a,b,4.0f);
        scene.advance_collision();
        check(2);
        scene.remove_constraint_edge(edge);
        scene.advance_collision();
        check(1);
        scene.remove(b);
        scene.advance_collision();
        check(1);
    }
    static void verify_typed_constraint_edge_lifetime() {
        // eCONSTRAINT and eCONTACT_MANAGER use the same EdgeIndex pool,
        // but each accurate island maintains a separate type list.
        // This fixture covers island ownership only, not Dy 1D solving.
        Scene scene;
        const BodyId a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const BodyId b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{5,5,0},1.0f);
        scene.advance_collision();
        if(scene.accurate_island_count()!=2 ||
           !scene.contacts().empty())
            throw std::runtime_error("Constraint fixture started connected");

        const auto edge=scene.add_constraint_edge(a,b);
        const auto parallel=scene.add_constraint_edge(a,b);
        scene.advance_collision();
        if(scene.accurate_island_count()!=1 ||
           scene.speculative_island_count()!=1 ||
           scene.accurate_island_id(a)!=scene.accurate_island_id(b) ||
           scene.island_edges_[edge].type!=
               Scene::IslandEdgeType::Constraint ||
           scene.contact_edge_index({a,b})!=Scene::invalid_edge ||
           !scene.contacts().empty())
            throw std::runtime_error(
                "Typed source constraint edge failed island admission");
        const auto owner=scene.accurate_island_id(a);
        const auto& island=scene.accurate_islands_.islands[owner];
        if(scene.speculative_islands_.active_constraint_edge_count!=2 ||
           scene.accurate_islands_.active_constraint_edge_count!=2 ||
           scene.speculative_active_reference_count(a)!=2 ||
           scene.accurate_active_reference_count(b)!=2 ||
           island.contact_count!=0 || island.constraint_count!=2 ||
           island.first_constraint!=edge ||
           island.last_constraint!=parallel ||
           scene.island_edges_[parallel].island_prev!=edge)
            throw std::runtime_error(
                "Source eCONSTRAINT contaminated contact-manager array");
        bool rejected_unsolved=false;
        try {scene.step(1.0f/120.0f);}
        catch(const std::logic_error&) {rejected_unsolved=true;}
        if(!rejected_unsolved)
            throw std::runtime_error(
                "Unadmitted Dy 1D constraint silently simulated");

        scene.remove_constraint_edge(edge);
        scene.advance_collision();
        if(scene.accurate_island_count()!=1 ||
           scene.speculative_island_count()!=1 ||
           scene.accurate_islands_.active_constraint_edge_count!=1 ||
           scene.accurate_active_reference_count(a)!=1 ||
           scene.active_constraint_handles_.size()!=1 ||
           !scene.island_edges_[parallel].live)
            throw std::runtime_error(
                "Parallel source constraint removal severed surviving route");
        scene.remove_constraint_edge(parallel);
        scene.advance_collision();
        if(scene.accurate_island_count()!=2 ||
           scene.speculative_island_count()!=2 ||
           scene.accurate_islands_.active_constraint_edge_count!=0 ||
           scene.accurate_active_reference_count(a)!=0 ||
           scene.accurate_island_id(a)==scene.accurate_island_id(b) ||
           !scene.active_constraint_handles_.empty() ||
           scene.island_edges_[edge].live ||
           scene.island_edges_[parallel].live)
            throw std::runtime_error(
                "Constraint edge retirement retained a stale island bridge");
        scene.set_position(b,{0,6,0});
        scene.advance_collision();
        const auto contact=scene.contact_edge_index({a,b});
        if((contact!=edge && contact!=parallel) ||
           contact>=scene.island_edges_.size() ||
           scene.island_edges_[contact].type!=
               Scene::IslandEdgeType::ContactManager ||
           scene.contacts().empty())
            throw std::runtime_error(
                "Shared EdgeIndex reuse confused constraint and contact owners");
    }
    static void verify_actor_retirement() {
        // Actor retirement must withdraw every incident eCONSTRAINT
        // connection before its node index can be reused by island repair.
        {
            Scene owned;
            const auto x=owned.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,5,0},1.0f);
            const auto y=owned.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5,5,0},1.0f);
            owned.advance_collision();
            owned.add_constraint_edge(x,y);
            owned.add_constraint_edge(x,y);
            owned.advance_collision();
            owned.remove(y);
            owned.advance_collision();
            if(!owned.active_constraint_handles_.empty() ||
               owned.accurate_island_count()!=1 ||
               owned.speculative_island_count()!=1)
                throw std::runtime_error(
                    "Constraint actor retirement retained an orphan island edge");
        }
    }
    static void verify_counted_interaction_lifetime() {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,6,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,6,0},1.0f);
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,9,0},1.0f);
        const auto first=scene.create_center_distance_constraint(a,b,4.0f);
        const auto second=scene.create_center_distance_constraint(a,b,4.0f);
        if(scene.actors_[a].num_counted_interactions!=2 ||
           scene.actors_[b].num_counted_interactions!=2 ||
           scene.actors_[c].num_counted_interactions!=0)
            throw std::runtime_error(
                "Parallel Sc joints were not counted on both attached bodies");
        scene.advance_collision(true);
        scene.set_position(c,{0,6.95f,0});
        scene.advance_collision(true);
        if(scene.contact_edge_index({a,c})==Scene::invalid_edge ||
           scene.actors_[a].num_counted_interactions!=3 ||
           scene.actors_[b].num_counted_interactions!=2 ||
           scene.actors_[c].num_counted_interactions!=1)
            throw std::runtime_error(
                "Gu contact manager and Sc joint counts did not coexist");
        scene.retire_center_distance_constraint(first);
        if(scene.actors_[a].num_counted_interactions!=2 ||
           scene.actors_[b].num_counted_interactions!=1)
            throw std::runtime_error(
                "Retiring one Sc joint changed surviving interaction counts");
        scene.set_position(c,{0,9,0});
        scene.advance_collision(true);
        scene.retire_center_distance_constraint(second);
        scene.advance_collision(true);
        if(scene.actors_[a].num_counted_interactions!=0 ||
           scene.actors_[b].num_counted_interactions!=0 ||
           scene.actors_[c].num_counted_interactions!=0)
            throw std::runtime_error(
                "Retired Gu and Sc edges retained body interaction counts");

        // Fail after Gu overlap admission using actual kinematic target
        // velocity overflow, then verify the exact original body counts.
        const auto k=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,12,0},1.0f);
        scene.set_kinematic(k,true);
        scene.advance_collision(true);
        scene.set_position(k,{0,6.9f,0});
        scene.set_kinematic_target(k,{1.0e35f,6.9f,0},{});
        bool rejected=false;
        try {scene.step(1.0e-10f);}
        catch(const std::overflow_error&) {rejected=true;}
        if(!rejected ||
           scene.contact_edge_index({a,k})!=Scene::invalid_edge ||
           scene.actors_[a].num_counted_interactions!=0 ||
           scene.actors_[k].num_counted_interactions!=0)
            throw std::runtime_error(
                "Failed Dy publication leaked source counted interactions");
        scene.set_kinematic_target(k,{0,6.9f,0},{});
        scene.step(1.0f/120.0f);
        if(scene.contact_edge_index({a,k})==Scene::invalid_edge ||
           scene.actors_[a].num_counted_interactions!=1 ||
           scene.actors_[k].num_counted_interactions!=1)
            throw std::runtime_error(
                "Dy retry failed to register its Gu counted interactions");
    }
    static void verify_independent_lost_route_producers() {
        // Both IslandSims begin with one three-node contact chain. Remove
        // both links but run one graph's third-pass route repair at a time.
        // Neither source producer may mutate the other graph's handles,
        // dirty-node stream, or Dy's accurate solver-node offsets.
        for(bool accurate_first:{false,true}) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto a=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,15,0},1.0f);
            const auto b=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.85f,15,0},1.0f);
            const auto c=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{1.7f,15,0},1.0f);
            scene.advance_collision();
            if(scene.accurate_island_count()!=1 ||
               scene.speculative_island_count()!=1)
                throw std::runtime_error(
                    "Independent third-pass fixture lacks merged source islands");
            const auto original_accurate=scene.accurate_island_id(a);
            const auto original_speculative=scene.speculative_island_id(a);
            const auto revision=scene.island_topology_revision();
            const auto node_offsets=scene.accurate_island_offsets_;
            scene.set_position(b,{5,15,0});
            scene.step_checkpoint_.capture(scene,false);
            scene.advance_collision_internal(false,true);
            if(scene.speculative_dirty_nodes_.empty() ||
               scene.accurate_dirty_nodes_.empty() ||
               scene.accurate_island_count()!=1 ||
               scene.speculative_island_count()!=1)
                throw std::runtime_error(
                    "First-pass contact removal repaired a third-pass route early");
            const bool first=scene.repair_lost_island_graph(accurate_first);
            if(!first)throw std::runtime_error(
                "First IslandSim did not migrate its disconnected source component");
            if(accurate_first) {
                if(scene.accurate_island_count()!=3 ||
                   scene.speculative_island_count()!=1 ||
                   scene.speculative_island_id(a)!=original_speculative ||
                   scene.speculative_dirty_nodes_.empty() ||
                   scene.accurate_island_offsets_==node_offsets)
                    throw std::runtime_error(
                        "Accurate third pass changed speculative graph ownership");
            } else if(scene.speculative_island_count()!=3 ||
                      scene.accurate_island_count()!=1 ||
                      scene.accurate_island_id(a)!=original_accurate ||
                      scene.accurate_dirty_nodes_.empty() ||
                      scene.accurate_island_offsets_!=node_offsets)
                throw std::runtime_error(
                    "Speculative third pass changed Dy accurate batch offsets");
            const bool second=scene.repair_lost_island_graph(!accurate_first);
            if(!second || scene.accurate_island_count()!=3 ||
               scene.speculative_island_count()!=3 ||
               scene.accurate_island_offsets_==node_offsets ||
               scene.repair_lost_island_graph(true) ||
               scene.repair_lost_island_graph(false) ||
               scene.island_topology_revision()!=revision)
                throw std::runtime_error(
                    "Independent source island repair replayed consumed lost routes");
            // Once both original tasks have completed, their shared
            // publication advances the scene's topology epoch only once.
            ++scene.island_topology_revision_;
            if(scene.island_topology_revision()!=revision+1)
                throw std::runtime_error(
                    "Third-pass producer order changed shared topology generation");
            scene.step_checkpoint_.rollback(scene);
            if(scene.accurate_island_count()!=1 ||
               scene.speculative_island_count()!=1 ||
               scene.accurate_island_id(a)!=original_accurate ||
               scene.speculative_island_id(a)!=original_speculative ||
               scene.island_topology_revision()!=revision)
                throw std::runtime_error(
                    "Failed third-pass publication retained split island ownership");
            // A successful ordinary publication must still complete
            // both graph repairs after the same rollback checkpoint.
            scene.advance_collision();
            if(scene.accurate_island_count()!=3 ||
               scene.speculative_island_count()!=3 ||
               scene.accurate_island_id(a)==scene.accurate_island_id(c))
                throw std::runtime_error(
                    "Rolled-back graph split did not repair on subsequent publication");
        }
    }
    static void verify_single_dispatch_for_split_and_sleep() {
        // One production third-pass worker handles both lost routes and
        // speculative sleeping; two independent worker dispatches would
        // impose an avoidable task boundary and permit cross-phase races.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,15,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,15,0},1.0f);
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,15,0},1.0f);
        for(BodyId id:{a,b,c}) {
            scene.actors_[id].dynamics.gravity_enabled=false;
            scene.actors_[id].ready_for_sleep=false;
        }
        scene.advance_collision();
        scene.set_position(b,{5,15,0});
        // Force real two-graph lost routes and prior-Dy sleep readiness
        // in the same production frame. Only the individual Islands
        // decide whether the detached components can sleep.
        scene.actors_[a].ready_for_sleep=true;
        scene.speculative_ready_nodes_.push_back(a);
        const auto prior_workers=scene.concurrent_third_pass_dispatches_;
        const auto prior_routes=scene.concurrent_lost_route_dispatches_;
        const auto prior_dy=scene.overlapped_dy_preintegration_dispatches_;
        scene.step(1.0f/120.0f);
        if(scene.concurrent_third_pass_dispatches_!=prior_workers+1 ||
           scene.concurrent_lost_route_dispatches_!=prior_routes+1 ||
           scene.overlapped_dy_preintegration_dispatches_!=prior_dy+1 ||
           scene.accurate_island_count()!=3 ||
           scene.speculative_island_count()!=3 ||
           scene.accurate_island_id(a)==scene.accurate_island_id(c) ||
           scene.speculative_island_id(a)==scene.speculative_island_id(c) ||
           scene.actors_[a].ready_for_sleep==false)
            throw std::runtime_error(
                "Combined third-pass task lost route/sleep ownership");
    }
    static void verify_short_route_repairs_stay_serial() {
        // Both source owners must repair their true lost edges without
        // paying a worker roundtrip for a three-node chain. No sleep
        // event was produced by Dy on this geometry-only admission.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,15,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,15,0},1.0f);
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,15,0},1.0f);
        for(BodyId id:{a,b,c})
            scene.actors_[id].dynamics.gravity_enabled=false;
        scene.advance_collision();
        scene.set_position(b,{5,15,0});
        scene.step(1.0f/120.0f);
        if(scene.third_pass_worker_start_count_!=0 ||
           scene.concurrent_third_pass_dispatches_!=0 ||
           scene.accurate_island_count()!=3 ||
           scene.speculative_island_count()!=3 ||
           scene.accurate_island_id(a)==scene.accurate_island_id(c) ||
           scene.speculative_island_id(a)==scene.speculative_island_id(c))
            throw std::runtime_error(
                "Short serial lost-route task failed source island parity");
    }
    static void verify_concurrent_lost_route_repair_rollback() {
        // Test the genuinely concurrent path even on a three-body
        // diagnostic scene; production scheduling avoids small jobs.
        // Both graph-owned findRoute tasks split the same contact chain
        // concurrently, then rejection must restore each original owner.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.parallel_third_pass_min_dirty_nodes_=0;
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,15,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,15,0},1.0f);
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,15,0},1.0f);
        scene.advance_collision();
        const auto speculative=scene.speculative_island_id(a);
        const auto accurate=scene.accurate_island_id(a);
        const auto offsets=scene.accurate_island_offsets_;
        scene.set_position(b,{5,15,0});
        scene.step_checkpoint_.capture(scene,false);
        scene.advance_collision_internal(false,true);
        if(scene.speculative_dirty_nodes_.empty() ||
           scene.accurate_dirty_nodes_.empty())
            throw std::runtime_error(
                "Parallel source fixture lacks both dirty worklists");
        const auto revision=scene.island_topology_revision();
        if(!scene.repair_lost_island_graphs_concurrently() ||
           scene.concurrent_lost_route_dispatches_!=1 ||
           scene.speculative_island_count()!=3 ||
           scene.accurate_island_count()!=3 ||
           scene.accurate_island_offsets_==offsets ||
           scene.step_checkpoint_.graph_ownership_mask.load()!=3u)
            throw std::runtime_error(
                "Concurrent lost-route repair changed producer ownership");
        ++scene.island_topology_revision_;
        scene.step_checkpoint_.rollback(scene);
        if(scene.speculative_island_count()!=1 ||
           scene.accurate_island_count()!=1 ||
           scene.speculative_island_id(a)!=speculative ||
           scene.accurate_island_id(a)!=accurate ||
           scene.island_topology_revision()!=revision)
            throw std::runtime_error(
                "Concurrent lost-route rollback retained split membership");
        scene.advance_collision();
        if(scene.speculative_island_count()!=3 ||
           scene.accurate_island_count()!=3 ||
           scene.speculative_island_id(a)==scene.speculative_island_id(c) ||
           scene.accurate_island_id(a)==scene.accurate_island_id(c))
            throw std::runtime_error(
                "Concurrent lost-route retry failed after graph rollback");
    }
    static void verify_after_integration_deactivation_reconciliation() {
        // ScPipeline::afterIntegration restores lastCCDTransform when Dy
        // integrated a body concurrently with the accurate third pass.
        // Use the production capture/reconcile operations; do not
        // duplicate the restoration rules in the test.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto sleeper=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto awake=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{6,8,0},1.0f);
        scene.advance_collision();
        const auto original=scene.body_state(sleeper);
        const auto control=scene.body_state(awake);
        const auto old_bounds=scene.actors_[sleeper].encoded_bounds;
        scene.step_checkpoint_.capture(scene);
        ++scene.rigid_integration_generation_;
        scene.remember_pre_integration_pose(sleeper);
        scene.remember_pre_integration_pose(awake);
        // The real solver always journals the authoritative state before
        // integrated motion so a rejected frame restores both actors.
        scene.step_checkpoint_.save_motion(scene,sleeper);
        scene.step_checkpoint_.save_motion(scene,awake);
        // Stand in for task-local Dy integration, not a second solver:
        // the original pose was saved by the exact production hook.
        scene.actors_[sleeper].position={2,9,1};
        scene.actors_[sleeper].orientation={0,0.38268343f,0,0.92387953f};
        scene.actors_[sleeper].velocity={3,4,5};
        scene.actors_[sleeper].angular_velocity={0,2,0};
        scene.actors_[sleeper].wake_counter=0.5f;
        scene.actors_[awake].position={6.25f,8,0};
        scene.actors_[sleeper].ready_for_sleep=true;
        scene.actors_[awake].ready_for_sleep=false;
        scene.deactivate_ready_accurate_islands();
        if(!scene.body_is_sleeping(sleeper) ||
           scene.body_is_sleeping(awake) ||
           scene.accurate_islands_.nodes_to_deactivate.size()!=1)
            throw std::runtime_error(
                "Concurrent afterIntegration fixture has no accurate sleep");
        // Dy sleepCheck can finish AFTER the island pass deactivates
        // the body and overwrite its counter/velocity. Source
        // afterIntegration grants the accurate island decision priority.
        scene.actors_[sleeper].velocity={4,5,6};
        scene.actors_[sleeper].angular_velocity={0,3,0};
        scene.actors_[sleeper].wake_counter=0.9f;
        scene.actors_[sleeper].ready_for_sleep=false;
        scene.reconcile_integrated_deactivations();
        const auto restored=scene.body_state(sleeper);
        const auto unaffected=scene.body_state(awake);
        if(restored.position.x!=original.position.x ||
           restored.position.y!=original.position.y ||
           restored.position.z!=original.position.z ||
           restored.orientation.x!=original.orientation.x ||
           restored.orientation.y!=original.orientation.y ||
           restored.orientation.z!=original.orientation.z ||
           restored.orientation.w!=original.orientation.w ||
           restored.velocity.x!=0 || restored.velocity.y!=0 ||
           restored.angular_velocity.y!=0 ||
           !scene.actors_[sleeper].cached_bounds_current ||
           scene.actors_[sleeper].encoded_bounds!=old_bounds ||
           scene.actors_[sleeper].wake_counter!=0 ||
           !scene.actors_[sleeper].ready_for_sleep ||
           unaffected.position.x!=6.25f ||
           unaffected.orientation.w!=control.orientation.w)
            throw std::runtime_error(
                "AfterIntegration failed to restore only island-deactivated rigid pose");
        // Idempotence inside one frame, and a stale lastCCDTransform
        // cannot rewind a different generation's already sleeping body.
        scene.reconcile_integrated_deactivations();
        ++scene.rigid_integration_generation_;
        scene.actors_[sleeper].position={0.1f,8,0};
        scene.reconcile_integrated_deactivations();
        if(scene.actors_[sleeper].position.x!=0.1f)
            throw std::runtime_error(
                "AfterIntegration replayed stale concurrent pose restoration");
        scene.step_checkpoint_.rollback(scene);
        const auto rolled=scene.body_state(sleeper);
        if(rolled.position.x!=original.position.x ||
           rolled.position.y!=original.position.y ||
           rolled.velocity.x!=original.velocity.x ||
           scene.body_is_sleeping(sleeper))
            throw std::runtime_error(
                "Rejected concurrent afterIntegration left integrated body state");
        // Step a normal serial frame after the failed transaction.
        // Its third pass deactivates before Dy, and no stale concurrent
        // mLastCCDTransform may change the resulting motion.
        const auto prior_awake=scene.body_state(awake);
        scene.step(1.0f/120.0f);
        if(scene.body_is_sleeping(awake) ||
           !std::isfinite(scene.body_state(sleeper).position.y) ||
           scene.actors_[awake].last_ccd_generation!=
               scene.rigid_integration_generation_ ||
           scene.actors_[awake].last_ccd_position.x!=prior_awake.position.x ||
           scene.actors_[awake].last_ccd_position.y!=prior_awake.position.y ||
           scene.actors_[awake].last_ccd_orientation.w!=prior_awake.orientation.w)
            throw std::runtime_error(
                "Production Dy pre-integration pose witness failed after rollback");
    }
    static void verify_dy_ready_event_reaches_both_islands() {
        // Unlike directly setting Node::isReadyForSleeping, this runs
        // the real Dy sleepCheck and checks its following-frame handoff.
        // A zero-gravity free body reaches readiness without a Gu touch
        // or constraint; both IslandSim producers must consume that one
        // transition, even with no speculative contact edge.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto id=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,12,0},1.0f);
        scene.actors_[id].dynamics.gravity_enabled=false;
        scene.actors_[id].wake_counter=0.001f;
        scene.actors_[id].ready_for_sleep=false;
        scene.step(0.02f);
        if(scene.body_is_sleeping(id) ||
           !scene.actors_[id].ready_for_sleep ||
           scene.speculative_ready_nodes_.size()!=1 ||
           scene.speculative_ready_nodes_.front()!=id ||
           scene.accurate_islands_.nodes_to_deactivate.size()!=0)
            throw std::runtime_error(
                "Dy sleepCheck did not queue next-frame speculative readiness");
        scene.step(0.02f);
        if(!scene.body_is_sleeping(id) ||
           scene.concurrent_third_pass_dispatches_!=1 ||
           scene.overlapped_dy_preintegration_dispatches_!=1 ||
           !scene.speculative_ready_nodes_.empty() ||
           scene.accurate_islands_.nodes_to_deactivate.size()!=1 ||
           scene.accurate_islands_.nodes_to_deactivate.front()!=id)
            throw std::runtime_error(
                "Independent IslandSims did not consume prior Dy readiness once");
        scene.step(0.02f);
        if(!scene.body_is_sleeping(id) ||
           scene.concurrent_third_pass_dispatches_!=1 ||
           scene.overlapped_dy_preintegration_dispatches_!=1 ||
           !scene.speculative_ready_nodes_.empty() ||
           !scene.speculative_islands_.nodes_to_deactivate.empty() ||
           !scene.accurate_islands_.nodes_to_deactivate.empty())
            throw std::runtime_error(
                "Quiet Dy frame replayed the consumed readiness transition");
    }
    static void verify_reused_worker_shutdown() {
        // A single scene has multiple back-to-back ThirdPassTasks. Its
        // worker must be created once, rearmed without lost wakeups, and
        // cleanly joined when the Scene leaves scope.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        std::array<BodyId,4> ids{};
        for(unsigned i=0;i<ids.size();++i) {
            ids[i]=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{float(i)*5.0f,12,0},1.0f);
            scene.actors_[ids[i]].dynamics.gravity_enabled=false;
        }
        scene.advance_collision();
        for(std::size_t i=0;i<ids.size();++i) {
            const auto id=ids[i];
            scene.actors_[id].ready_for_sleep=true;
            scene.speculative_ready_nodes_.push_back(id);
            scene.step(1.0f/120.0f);
            if(scene.third_pass_worker_start_count_!=1 ||
               scene.concurrent_third_pass_dispatches_!=i+1 ||
               !scene.third_pass_worker_state_ ||
               scene.third_pass_worker_state_->outstanding ||
               !scene.body_is_sleeping(id))
                throw std::runtime_error(
                    "Reusable source ThirdPassTask worker lost a handoff");
        }
        scene.step(1.0f/120.0f);
        if(scene.third_pass_worker_start_count_!=1 ||
           scene.concurrent_third_pass_dispatches_!=ids.size())
            throw std::runtime_error(
                "Quiet frame needlessly relaunched source island worker");
    }
    static void verify_parallel_third_pass_failure_rollback() {
        // Exercise actual production worker dispatch, then reject in Dy.
        // Both IslandSim outputs and the unread prior-Dy readiness
        // event must roll back even though the speculative worker ran.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto world=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,6,0});
        const auto sleeper=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,6.9f,0},1.0f);
        const auto mover=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{8,7,0},1.0f);
        scene.actors_[sleeper].dynamics.gravity_enabled=false;
        scene.actors_[mover].dynamics.gravity_enabled=false;
        scene.set_velocity(mover,{10,0,0});
        scene.advance_collision(true);
        const Pair sleeping_pair{world,sleeper};
        const auto edge=scene.contact_edge_index(sleeping_pair);
        if(edge==Scene::invalid_edge ||
           !(scene.island_edges_[edge].speculative_state&1u) ||
           !(scene.island_edges_[edge].accurate_state&1u) ||
           !scene.contact_interaction_active(sleeping_pair))
            throw std::runtime_error(
                "Parallel rollback fixture lacks a shared live manager");
        const auto baseline=scene.body_state(sleeper);
        const auto mover_baseline=scene.body_state(mover);
        const auto original_speculative=scene.speculative_islands_.active_nodes;
        const auto original_accurate=scene.accurate_islands_.active_nodes;
        scene.actors_[sleeper].ready_for_sleep=true;
        scene.speculative_ready_nodes_.push_back(sleeper);
        bool rejected=false;
        try { scene.step(1.0e38f); }
        catch(const std::overflow_error&) { rejected=true; }
        if(!rejected || scene.concurrent_third_pass_dispatches_==0 ||
           scene.body_is_sleeping(sleeper) ||
           !scene.actors_[sleeper].ready_for_sleep ||
           scene.speculative_ready_nodes_.size()!=1 ||
           scene.speculative_ready_nodes_.front()!=sleeper ||
           scene.speculative_islands_.active_nodes!=original_speculative ||
           scene.accurate_islands_.active_nodes!=original_accurate ||
           !(scene.island_edges_[edge].speculative_state&1u) ||
           !(scene.island_edges_[edge].accurate_state&1u) ||
           !scene.contact_interaction_active(sleeping_pair) ||
           scene.body_state(sleeper).position.x!=baseline.position.x ||
           scene.body_state(mover).position.x!=mover_baseline.position.x)
            throw std::runtime_error(
                "Concurrent third pass failed to restore pre-step state");
        // A valid retry consumes exactly that original event; the
        // failed task must not leave a stale sleeping-node publication.
        scene.step(1.0f/120.0f);
        if(!scene.body_is_sleeping(sleeper) ||
           !scene.speculative_ready_nodes_.empty() ||
           scene.accurate_islands_.nodes_to_deactivate.size()!=1 ||
           scene.speculative_islands_.deactivating_edges.size()!=1 ||
           scene.speculative_islands_.deactivating_edges[0]!=edge ||
           scene.accurate_islands_.deactivating_edges.size()!=1 ||
           scene.accurate_islands_.deactivating_edges[0]!=edge ||
           scene.island_edges_[edge].speculative_state&1u ||
           scene.island_edges_[edge].accurate_state&1u ||
           scene.contact_interaction_active(sleeping_pair) ||
           scene.concurrent_third_pass_dispatches_<2)
            throw std::runtime_error(
                "Concurrent third-pass failure prevented deterministic retry");
    }
    static void verify_speculative_active_island_ownership() {
        // Source accurate sleep does not forcibly deactivate a larger
        // speculative AABB-only island while another rigid stays awake.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        // Rotated narrow boxes have intersecting SAP bounds but no
        // actual OBB contact. Both remain ordinary dynamic nodes.
        const auto a=scene.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(BodyId id:{a,b}) {
            scene.set_velocity(id,{0,9.81f,0});
            scene.set_angular_velocity(id,{0,0.83f,0});
        }
        scene.step(1.0f);
        for(BodyId id:{a,b}) {
            scene.set_angular_velocity(id,{});
            scene.set_velocity(id,{});
        }
        scene.set_position(a,{0,0.1f,0});
        scene.set_position(b,{0.6f,0.1f,0.6f});
        scene.advance_collision(true);
        const Pair pair{a,b};
        const auto edge=scene.contact_edge_index(pair);
        if(edge==Scene::invalid_edge || scene.contact_edge_connected(pair) ||
           scene.speculative_island_count()!=1 ||
           scene.accurate_island_count()!=2)
            throw std::runtime_error(
                "Speculative activity fixture lacks SAP-only rigid bridge");
        scene.actors_[a].ready_for_sleep=true;
        scene.actors_[b].ready_for_sleep=false;
        scene.speculative_ready_nodes_.push_back(a);
        scene.step_checkpoint_.capture(scene);
        scene.deactivate_ready_speculative_islands();
        scene.deactivate_ready_accurate_islands();
        scene.synchronize_active_island_nodes();
        if(!scene.body_is_sleeping(a) || scene.body_is_sleeping(b) ||
           scene.speculative_islands_.active_node_index[a]==Scene::invalid_instance ||
           scene.speculative_islands_.active_node_index[b]==Scene::invalid_instance ||
           scene.speculative_islands_.active_islands.size()!=1 ||
           !(scene.island_edges_[edge].speculative_state&1u))
            throw std::runtime_error(
                "Accurate sleep retired a still-awake speculative bridge");
        scene.actors_[b].ready_for_sleep=true;
        scene.speculative_ready_nodes_.push_back(b);
        scene.deactivate_ready_speculative_islands();
        if(scene.speculative_islands_.active_node_index[a]!=Scene::invalid_instance ||
           scene.speculative_islands_.active_node_index[b]!=Scene::invalid_instance ||
           !scene.speculative_islands_.active_islands.empty() ||
           scene.speculative_islands_.nodes_to_deactivate.size()!=2 ||
           scene.island_edges_[edge].speculative_state&1u)
            throw std::runtime_error(
                "Speculative IslandSim failed to retire its own node and island activity");
        scene.step_checkpoint_.rollback(scene);
        if(scene.speculative_islands_.active_nodes.size()!=2 ||
           scene.speculative_islands_.active_islands.size()!=1 ||
           scene.speculative_islands_.active_node_index[a]==Scene::invalid_instance ||
           scene.speculative_islands_.active_node_index[b]==Scene::invalid_instance ||
           !(scene.island_edges_[edge].speculative_state&1u))
            throw std::runtime_error(
                "Failed speculative deactivation failed to restore graph activity");
    }
    static void verify_speculative_ready_is_independent() {
        // The two PxsIslandSim::ThirdPassTasks have no dependency on
        // one another. Dy publishes readiness in the preceding step;
        // speculative completion can therefore precede accurate sleep.
        for(bool accurate_first:{false,true}) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto world=scene.add_static_box(
                {0.5f,0.5f,0.5f},{0,7,0});
            const auto sleeper=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,7.9f,0},1.0f);
            const Pair pair{world,sleeper};
            scene.advance_collision();
            const auto edge=scene.contact_edge_index(pair);
            if(edge==Scene::invalid_edge ||
               !(scene.island_edges_[edge].speculative_state&1u) ||
               !(scene.island_edges_[edge].accurate_state&1u))
                throw std::runtime_error(
                    "Speculative independence fixture lacks source manager");
            scene.actors_[sleeper].ready_for_sleep=true;
            scene.speculative_ready_nodes_.push_back(sleeper);
            scene.step_checkpoint_.capture(scene);
            if(accurate_first) {
                scene.deactivate_ready_accurate_islands();
                scene.deactivate_ready_speculative_islands();
            } else {
                scene.deactivate_ready_speculative_islands();
                if(scene.body_is_sleeping(sleeper) ||
                   !scene.accurate_islands_.nodes_to_deactivate.empty() ||
                   !(scene.island_edges_[edge].accurate_state&1u))
                    throw std::runtime_error(
                        "Speculative task depended on accurate rigid deactivation");
                scene.deactivate_ready_accurate_islands();
            }
            if(!scene.body_is_sleeping(sleeper) ||
               (scene.island_edges_[edge].speculative_state&1u) ||
               (scene.island_edges_[edge].accurate_state&1u) ||
               scene.speculative_islands_.deactivating_edges.size()!=1 ||
               scene.speculative_islands_.deactivating_edges[0]!=edge ||
               scene.accurate_islands_.nodes_to_deactivate.size()!=1 ||
               !scene.speculative_ready_nodes_.empty())
                throw std::runtime_error(
                    "Independent IslandSim order changed rigid sleep or edge output");
            scene.step_checkpoint_.rollback(scene);
            if(!scene.actors_[sleeper].ready_for_sleep ||
               scene.body_is_sleeping(sleeper) ||
               scene.speculative_ready_nodes_.size()!=1 ||
               scene.speculative_ready_nodes_[0]!=sleeper ||
               !(scene.island_edges_[edge].speculative_state&1u) ||
               !(scene.island_edges_[edge].accurate_state&1u))
                throw std::runtime_error(
                    "Rejected independent third pass consumed prior-Dy readiness");
            // A replay of the very same producer input must still
            // release its speculative edge on the next accepted pass.
            scene.deactivate_ready_speculative_islands();
            if(scene.island_edges_[edge].speculative_state&1u)
                throw std::runtime_error(
                    "Speculative task did not replay restored readiness");
        }
    }
    static void verify_sc_activity_third_pass_publication() {
        // Activation and third-pass deactivation have distinct Sc
        // consumers. Neither an accurate sleep result nor an inactive
        // speculative EdgeInstance is itself a published eIS_ACTIVE
        // transition before postThirdPassIslandGen consumes that output.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto world=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,5,0});
        const auto dynamic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5.9f,0},1.0f);
        const Pair pair{world,dynamic};
        scene.advance_collision();
        const auto edge=scene.contact_edge_index(pair);
        if(edge==Scene::invalid_edge ||
           !scene.island_edges_[edge].interaction_active ||
           scene.active_contact_interaction_count()!=1)
            throw std::runtime_error(
                "Sc first/second-pass contact activity was not published");
        scene.actors_[dynamic].ready_for_sleep=true;
        // The prior Dy sleepCheck would publish this transition to
        // the speculative third-pass producer independently.
        scene.speculative_ready_nodes_.push_back(dynamic);
        scene.deactivate_ready_accurate_islands();
        if(!scene.body_is_sleeping(dynamic) ||
           !scene.island_edges_[edge].interaction_active ||
           !(scene.island_edges_[edge].speculative_state&1u) ||
           !scene.speculative_islands_.deactivating_edges.empty())
            throw std::runtime_error(
                "Accurate rigid sleep published speculative Sc retirement");
        scene.deactivate_ready_speculative_islands();
        if((scene.island_edges_[edge].speculative_state&1u) ||
           !scene.island_edges_[edge].interaction_active ||
           scene.speculative_islands_.deactivating_edges.size()!=1 ||
           scene.speculative_islands_.deactivating_edges[0]!=edge ||
           scene.active_contact_interaction_count()!=1)
            throw std::runtime_error(
                "Speculative island completion bypassed Sc post-third-pass join");
        scene.consume_deactivated_contact_interactions();
        if(scene.island_edges_[edge].interaction_active ||
           scene.active_contact_interaction_count()!=0)
            throw std::runtime_error(
                "Sc post-third-pass failed to retire speculative interaction");
        scene.consume_deactivated_contact_interactions();
        if(scene.active_contact_interaction_count()!=0)
            throw std::runtime_error(
                "Sc third-pass deactivation was consumed twice");
    }
    static void verify_active_kinematic_guards_rigid_island_sleep() {
        // PhysX marks islands touched by an active kinematic before
        // testing already-published Dy sleep readiness. A kinematic
        // constraint has no island ID; its connected dynamic member
        // identifies an entire rigid island, including an indirect peer.
        Scene scene{};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,7,0},1.0f);
        const auto k=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{6,7,0},1.0f);
        scene.set_kinematic(k,true);
        scene.add_distance_constraint(a,b,3.0f);
        scene.add_distance_constraint(b,k,3.0f);
        scene.advance_collision(true);
        if(scene.accurate_island_count()!=1 ||
           scene.accurate_island_id(k)!=Scene::IslandGraph::invalid ||
           scene.accurate_island_id(a)!=scene.accurate_island_id(b) ||
           scene.active_accurate_kinematics().size()!=1)
            throw std::runtime_error(
                "Source kinematic sleep fixture has no connected rigid island");
        // Model a previously published sleepCheck readiness result.
        // Active kinematic authoring is a separate Sc stage; the third
        // pass must not retire the island merely because those flags
        // still report readiness.
        scene.set_kinematic_target(k,{6.1f,7,0},{});
        scene.actors_[a].ready_for_sleep=true;
        scene.actors_[b].ready_for_sleep=true;
        scene.deactivate_ready_islands();
        if(scene.body_is_sleeping(a) || scene.body_is_sleeping(b) ||
           !scene.accurate_islands_.nodes_to_deactivate.empty() ||
           scene.accurate_islands_.active_constraint_edge_count!=2 ||
           !scene.accurate_islands_.deactivating_constraint_edges.empty())
            throw std::runtime_error(
                "Active kinematic failed to preserve connected rigid island");
        // Once its authored target is consumed, the kinematic is
        // settling, not actively moving. Source readiness may now
        // deactivate the same island without losing the constraint.
        scene.actors_[k].kinematic_target_pending=false;
        scene.actors_[k].ready_for_sleep=true;
        scene.deactivate_ready_accurate_islands();
        if(!scene.body_is_sleeping(a) || !scene.body_is_sleeping(b) ||
           !scene.speculative_islands_.deactivating_constraint_edges.empty())
            throw std::runtime_error(
                "Accurate IslandSim completion failed to isolate rigid deactivation");
        scene.deactivate_ready_speculative_islands();
        if(!scene.body_is_sleeping(a) || !scene.body_is_sleeping(b) ||
           scene.accurate_islands_.active_constraint_edge_count!=0 ||
           scene.accurate_islands_.deactivating_constraint_edges.size()!=2)
            throw std::runtime_error(
                "Settled kinematic retained an unwarranted island wake veto");
    }
    static void verify_kinematic_speculative_sleep_veto() {
        // Two slender rotated boxes have overlapping SAP bounds, but
        // no Gu contact. A moving kinematic must keep that speculative
        // interaction alive even when its independent accurate dynamic
        // island can sleep. Settling releases the speculative edge once.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto k=scene.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(BodyId id:{a,k}) {
            scene.set_velocity(id,{0,9.81f,0});
            scene.set_angular_velocity(id,{0,0.83f,0});
        }
        scene.step(1.0f);
        scene.set_angular_velocity(a,{});
        scene.set_velocity(a,{});
        scene.set_angular_velocity(k,{});
        scene.set_velocity(k,{});
        scene.set_kinematic(k,true);
        scene.set_position(a,{0,0.1f,0});
        scene.set_position(k,{0.6f,0.1f,0.6f});
        scene.advance_collision(true);
        const auto edge=scene.contact_edge_index({a,k});
        if(edge==Scene::invalid_edge || scene.contacts().size()!=0 ||
           scene.actors_[a].num_counted_interactions!=1 ||
           scene.actors_[k].num_counted_interactions!=1 ||
           scene.speculative_island_count()!=1 ||
           scene.accurate_island_count()!=1 ||
           scene.speculative_islands_.active_kinematic_nodes.size()!=1)
            throw std::runtime_error(
                "Source kinematic speculative fixture lacks SAP-only edge");
        const auto old=scene.body_state(k);
        scene.set_kinematic_target(k,{0.61f,0.1f,0.6f},old.orientation);
        scene.actors_[a].ready_for_sleep=true;
        scene.deactivate_ready_islands();
        if(!scene.body_is_sleeping(a) ||
           !(scene.island_edges_[edge].speculative_state&1u) ||
           scene.speculative_islands_.active_node_index[a]==
               Scene::invalid_instance ||
           !scene.speculative_islands_.deactivating_edges.empty())
            throw std::runtime_error(
                "Speculative active kinematic failed to protect SAP edge");
        scene.actors_[k].kinematic_target_pending=false;
        scene.actors_[k].kinematic_settling=true;
        scene.actors_[k].ready_for_sleep=true;
        scene.deactivate_ready_accurate_islands();
        // Completing the accurate producer must not consume the
        // speculative SAP-only interaction or publish its retirement.
        if(!(scene.island_edges_[edge].speculative_state&1u) ||
           !scene.speculative_islands_.deactivating_edges.empty())
            throw std::runtime_error(
                "Accurate third pass retired speculative activity early");
        scene.deactivate_ready_speculative_islands();
        if(scene.island_edges_[edge].speculative_state&1u ||
           scene.speculative_islands_.active_node_index[a]!=Scene::invalid_instance ||
           scene.speculative_islands_.nodes_to_deactivate.size()!=1 ||
           scene.speculative_islands_.nodes_to_deactivate[0]!=a ||
           scene.speculative_islands_.deactivating_edges.size()!=1 ||
           scene.speculative_islands_.deactivating_edges[0]!=edge)
            throw std::runtime_error(
                "Settled kinematic did not release speculative-only edge");
        scene.deactivate_ready_islands();
        if(!scene.speculative_islands_.deactivating_edges.empty() ||
           scene.actors_[a].num_counted_interactions!=1 ||
           scene.actors_[k].num_counted_interactions!=1)
            throw std::runtime_error(
                "Sleep-deactivated SAP edge lost its attached body count");
        scene.set_position(k,{0.6f,0.1f,6.0f});
        scene.advance_collision(true);
        if(scene.actors_[a].num_counted_interactions!=0 ||
           scene.actors_[k].num_counted_interactions!=0)
            throw std::runtime_error(
                "Retired SAP-only edge retained counted interactions");
    }
    static void verify_inactive_kinematic_link_sleep_arbitration() {
        // Source processLostEdges walks attached kinematic edges even
        // when Edge::eACTIVE is clear. A target vetoes sleep, while its
        // later settling must recheck the same inactive linked edge.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto rigid=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto kinematic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,7,0},1.0f);
        scene.set_kinematic(kinematic,true);
        const auto edge=scene.add_distance_constraint(
            rigid,kinematic,2.0f);
        scene.advance_collision(true);
        if(scene.accurate_island_count()!=1 ||
           scene.speculative_island_count()!=1 ||
           !(scene.island_edges_[edge].accurate_state&1u) ||
           !(scene.island_edges_[edge].speculative_state&1u))
            throw std::runtime_error(
                "Inactive kinematic link fixture lacks attached source edge");
        const auto pose=scene.body_state(kinematic);
        scene.set_kinematic_target(
            kinematic,{3.1f,7,0},pose.orientation);
        scene.actors_[rigid].ready_for_sleep=true;
        // The EdgeInstance remains in both intrusive lists while its
        // active solver/interaction state is independently cleared.
        scene.deactivate_contact_edge(edge,false,false);
        scene.deactivate_contact_edge(edge,true,false);
        if(scene.island_edges_[edge].accurate_state!=0 ||
           scene.island_edges_[edge].speculative_state!=0 ||
           !scene.island_edges_[edge].accurate[0].linked ||
           !scene.island_edges_[edge].speculative[0].linked ||
           scene.accurate_islands_.active_kinematic_nodes.empty() ||
           scene.speculative_islands_.active_kinematic_nodes.empty())
            throw std::runtime_error(
                "Inactive kinematic edge was unlinked or lost active target");
        // Previous Dy sleepCheck has nominated this rigid. Both source
        // producers must still honor the moving kinematic veto.
        scene.speculative_ready_nodes_.push_back(rigid);
        scene.deactivate_ready_speculative_islands();
        scene.deactivate_ready_accurate_islands();
        if(scene.body_is_sleeping(rigid) ||
           scene.speculative_islands_.active_node_index[rigid]==
               Scene::invalid_instance ||
           !scene.speculative_islands_.nodes_to_deactivate.empty() ||
           !scene.accurate_islands_.nodes_to_deactivate.empty() ||
           !scene.speculative_ready_nodes_.empty())
            throw std::runtime_error(
                "Moving kinematic failed to veto sleep through inactive link");
        // Readiness remains true, so no second Dy readiness transition
        // exists. The speculative producer must discover the island
        // through the settling kinematic's linked adjacency instead.
        scene.actors_[kinematic].kinematic_target_pending=false;
        scene.actors_[kinematic].kinematic_settling=true;
        scene.actors_[kinematic].ready_for_sleep=true;
        scene.deactivate_ready_speculative_islands();
        if(scene.speculative_islands_.nodes_to_deactivate.size()!=1 ||
           scene.speculative_islands_.nodes_to_deactivate[0]!=rigid ||
           scene.speculative_islands_.active_node_index[rigid]!=
               Scene::invalid_instance)
            throw std::runtime_error(
                "Settled kinematic failed to revisit inactive speculative link");
        scene.deactivate_ready_accurate_islands();
        if(!scene.body_is_sleeping(rigid) ||
           scene.accurate_islands_.nodes_to_deactivate.size()!=1 ||
           scene.accurate_islands_.nodes_to_deactivate[0]!=rigid)
            throw std::runtime_error(
                "Accurate sleep failed after linked kinematic settled");
    }
    static void verify_joint_counted_sleep_energy() {
        // DySleep uses the source lifetime interaction count, even
        // when a joint has no contact manager. For the same sub-threshold
        // motion, one counted joint raises the energy threshold while
        // an unconstrained body resets its wake counter.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto anchor=scene.add_static_box(
            {0.5f,0.5f,0.5f},{4,6,0});
        const auto constrained=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,6,0},1.0f);
        const auto free=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{8,6,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.linear_damping=0.0f;
        dynamics.angular_damping=0.0f;
        scene.set_dynamics(constrained,dynamics);
        scene.set_dynamics(free,dynamics);
        scene.create_center_distance_constraint(constrained,anchor,4.0f);
        for(BodyId id:{constrained,free}) {
            scene.set_velocity(id,{0,0.12f,0});
            scene.actors_[id].wake_counter=0.01f;
        }
        if(scene.actors_[constrained].num_counted_interactions!=1 ||
           scene.actors_[free].num_counted_interactions!=0)
            throw std::runtime_error(
                "Sc joint-only sleep fixture lacks source interaction counts");
        scene.step(1.0f/120.0f);
        if(scene.actors_[constrained].wake_counter>=0.05f ||
           scene.actors_[free].wake_counter<0.25f)
            throw std::runtime_error(
                "Dy sleep energy ignored Sc constraint lifetime interaction");
    }
    static void verify_graph_owned_kinematic_active_references() {
        Scene scene{};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto k=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,5,0},1.0f);
        scene.set_kinematic(k,true);
        if(!scene.active_accurate_kinematics().empty())
            throw std::runtime_error(
                "Unreferenced stationary kinematic started active");
        const auto edge=scene.add_distance_constraint(a,k,2.0f);
        scene.advance_collision(true);
        if(scene.active_accurate_kinematics().size()!=1 ||
           scene.active_speculative_kinematics().size()!=1 ||
           scene.active_accurate_kinematics()[0]!=k ||
           scene.accurate_islands_.active_kinematic_index[k]!=0 ||
           scene.accurate_island_id(k)!=Scene::IslandGraph::invalid)
            throw std::runtime_error(
                "Constraint omitted active kinematic island reference");
        scene.remove_constraint_edge(edge);
        scene.advance_collision(true);
        if(!scene.active_accurate_kinematics().empty() ||
           !scene.active_speculative_kinematics().empty())
            throw std::runtime_error(
                "Retired constraint retained kinematic reference");
        scene.set_kinematic_target(k,{3.25f,5,0},{});
        scene.step(1.0f/120.0f);
        if(scene.active_accurate_kinematics().size()!=1)
            throw std::runtime_error(
                "Target-driven kinematic omitted from source active list");
        scene.step(1.0f/120.0f);
        if(!scene.active_accurate_kinematics().empty())
            throw std::runtime_error(
                "Settled kinematic retained source active membership");
    }
    static void verify_kinematic_endpoint_not_static_world() {
        Scene scene;
        const auto dynamic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto kinematic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5.9f,0},1.0f);
        scene.set_kinematic(kinematic,true);
        scene.advance_collision(true);
        if(scene.contacts().empty() ||
           scene.accurate_island_count()!=1 ||
           scene.accurate_island_id(kinematic)!=
               Scene::IslandGraph::invalid ||
           scene.accurate_island_static_touches(dynamic)!=0 ||
           scene.speculative_island_static_touches(dynamic)!=0)
            throw std::runtime_error(
                "Kinematic contact was counted as a static-world touch");
    }
};

struct ContactCorrelationSourceFixture {
    static void verify_kinematic_contact_reporting_source_lifetime() {
        // ScNPhaseCore's touch notifications are keyed by a single live
        // ShapeInteraction, including a target-driven kinematic actor.
        // The target pose is integrated after Dy, so LOST belongs to
        // the NEXT Gu frame, not the target-authoring frame.
        constexpr float dt=1.0f/120.0f;
        const auto run=[](ExecutionMode mode,bool legacy,
                          bool kinematic_first) {
            WorldOptions opts{mode};
            opts.contact_notifications=true;
            opts.pcm_box_box=!legacy;
            World world(opts);
            BodyId dynamic=0,kinematic=0;
            const auto add_dynamic=[&] {
                dynamic=world.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,9,0},1.0f);
            };
            const auto add_kinematic=[&] {
                kinematic=world.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.95f,9,0},1.0f);
            };
            if(kinematic_first) {add_kinematic();add_dynamic();}
            else {add_dynamic();add_kinematic();}
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            d.linear_damping=0.0f;
            d.angular_damping=0.0f;
            d.max_contact_impulse=0.0f;
            world.set_body_dynamics(dynamic,d);
            world.set_body_kinematic(kinematic,true);
            std::array<int,6> timeline{};
            for(std::size_t frame=0;frame<timeline.size();++frame) {
                if(frame==1)
                    world.set_body_kinematic_target(
                        kinematic,{0.94f,9,0},{});
                if(frame==3)
                    world.set_body_kinematic_target(
                        kinematic,{3,9,0},{});
                world.step(dt);
                const auto events=world.contact_events();
                if(events.size()>1 ||
                   (!events.empty() &&
                    (events[0].body0!=std::min(dynamic,kinematic) ||
                     events[0].body1!=std::max(dynamic,kinematic))))
                    throw std::runtime_error(
                        "Kinematic Sc contact reporter emitted unrelated pair");
                timeline[frame]=events.empty() ? -1 :
                    static_cast<int>(events.front().kind);
                if(!events.empty() &&
                   (events.front().kind==ContactEventKind::Lost
                        ? events.front().contact_count!=0
                        : events.front().contact_count==0))
                    throw std::runtime_error(
                        "Kinematic Sc touch callback has invalid Gu point range");
            }
            return timeline;
        };
        const std::array<int,6> expected{
            static_cast<int>(ContactEventKind::Found),
            static_cast<int>(ContactEventKind::Persists),
            static_cast<int>(ContactEventKind::Persists),
            static_cast<int>(ContactEventKind::Persists),
            static_cast<int>(ContactEventKind::Lost),-1};
        for(const bool legacy:{false,true})
            for(const bool kinematic_first:{false,true}) {
                const auto native=run(ExecutionMode::NativeCpu,
                                      legacy,kinematic_first);
                if(native!=expected)
                    throw std::runtime_error(
                        "Native target-driven kinematic touch sequence diverged");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
                const auto reference=run(legacy
                    ? ExecutionMode::ReferencePhysXCpuLegacyGu
                    : ExecutionMode::ReferencePhysXCpu,
                    legacy,kinematic_first);
                if(native!=reference)
                    throw std::runtime_error(
                        "Kinematic Sc touch transitions differ from PhysX CPU");
#endif
            }
    }
    static void verify_kinematic_threshold_source_pairs() {
        // Dy::writeBackContact uses both PxsSolverBodyData::nodeIndex
        // values. A kinematic endpoint is eKINEMATIC, not eWORLD, and
        // contributes its own reportThreshold to the source MIN rule.
        // The same dynamic resting between two DIFFERENT kinematics
        // must publish two exceeded pairs, not one aggregated world pair.
        constexpr float dt=1.0f/120.0f;
        constexpr Vec3 half{0.5f,0.5f,0.5f};
        for(const bool legacy:{false,true}) {
            for(const bool kinematics_first:{false,true}) {
                Scene scene{legacy
                    ? Scene::ContactGenerator::DiagnosticLegacyGu
                    : Scene::ContactGenerator::DefaultPcm};
                BodyId dynamic=0,bottom=0,side=0;
                const auto add_dynamic=[&] {
                    dynamic=scene.add_dynamic_box(
                        half,{0,7,0},1.0f);
                };
                const auto add_kinematics=[&] {
                    bottom=scene.add_dynamic_box(
                        half,{0,6.08f,0},1.0f);
                    side=scene.add_dynamic_box(
                        half,{0.92f,7,0},1.0f);
                    scene.set_kinematic(bottom,true);
                    scene.set_kinematic(side,true);
                };
                if(kinematics_first) {add_kinematics();add_dynamic();}
                else {add_dynamic();add_kinematics();}
                RigidBodyDynamics d{};
                d.gravity_enabled=false;
                d.linear_damping=0.0f;
                d.angular_damping=0.0f;
                scene.set_dynamics(dynamic,d);
                scene.set_velocity(dynamic,{1,-1,0});
                // PxRigidDynamic retains the threshold API even after
                // eKINEMATIC is set. The ordinary dynamic has MAX_REAL,
                // so source writeback must use each kinematic's zero.
                scene.set_contact_report_threshold(bottom,0.0f);
                scene.set_contact_report_threshold(side,0.0f);
                scene.step(dt);
                if(scene.contacts().size()!=2)
                    throw std::runtime_error(
                        "Dy kinematic threshold fixture lacks two Gu managers");
                const auto pairs=scene.force_threshold_changes();
                const auto matches=[&](ForceThresholdKey key,
                                       ForceThresholdChange change) {
                    return std::count_if(pairs.begin(),pairs.end(),
                        [&](const ForceThresholdReport& report) {
                            return report.key==key &&
                                   report.change==change &&
                                   report.threshold==0.0f &&
                                   report.normal_impulse>0.0;
                        });
                };
                const auto first=ForceThresholdKey{
                    std::min(dynamic,bottom),std::max(dynamic,bottom),false};
                const auto second=ForceThresholdKey{
                    std::min(dynamic,side),std::max(dynamic,side),false};
                if(pairs.size()!=2 ||
                   matches(first,ForceThresholdChange::Found)!=1 ||
                   matches(second,ForceThresholdChange::Found)!=1)
                    throw std::runtime_error(
                        "Dy conflated distinct kinematic threshold pair identities");
                // Raising just one endpoint's threshold must retire its
                // exceeded pair while the other remains independently
                // monitored. Neither transition can use eWORLD.
                scene.set_contact_report_threshold(bottom,
                    std::numeric_limits<float>::max());
                scene.step(dt);
                const auto next=scene.force_threshold_changes();
                const auto has=[&](ForceThresholdKey key,
                                  ForceThresholdChange change) {
                    return std::count_if(next.begin(),next.end(),
                        [&](const ForceThresholdReport& report) {
                            return report.key==key &&
                                   report.change==change;
                        })==1;
                };
                if(next.size()!=2 ||
                   !has(first,ForceThresholdChange::Lost) ||
                   !has(second,ForceThresholdChange::Persists))
                    throw std::runtime_error(
                        "Dy kinematic force thresholds did not retire independently");
            }
        }
    }
    static void verify_kinematic_contact_manager_source_order() {
        // The kinematic flag changes PxRigidDynamic's contact-manager
        // orientation: a dynamic/kinematic pair keeps the ordinary
        // dynamic first regardless of actor admission order.
        constexpr Vec3 half{0.5f,0.5f,0.5f};
        constexpr Vec3 dynamic_position{0,8,0};
        constexpr Vec3 kinematic_position{0.92f,8,0.06f};
        for(const bool legacy:{false,true}) {
            for(const bool kinematic_first:{false,true}) {
                Scene scene{legacy
                    ? Scene::ContactGenerator::DiagnosticLegacyGu
                    : Scene::ContactGenerator::DefaultPcm};
                BodyId dynamic=0,kinematic=0;
                const auto add_dynamic=[&] {
                    dynamic=scene.add_dynamic_box(
                        half,dynamic_position,1.0f);
                };
                const auto add_kinematic=[&] {
                    kinematic=scene.add_dynamic_box(
                        half,kinematic_position,1.0f);
                    scene.set_kinematic(kinematic,true);
                };
                if(kinematic_first) {add_kinematic();add_dynamic();}
                else {add_dynamic();add_kinematic();}
                scene.advance_collision(true);
                const auto edge=scene.contact_edge_index(
                    {std::min(dynamic,kinematic),
                     std::max(dynamic,kinematic)});
                if(edge==Scene::invalid_edge ||
                   scene.contacts().empty() ||
                   scene.island_edges_[edge].type!=
                       Scene::IslandEdgeType::ContactManager ||
                   scene.island_edges_[edge].gu_order_reversed!=kinematic_first)
                    throw std::runtime_error(
                        "Sc contact-manager placed kinematic before dynamic");
                scene.step(1.0f/120.0f);
                const auto state=scene.body_state(dynamic);
                if(!std::isfinite(state.velocity.x) ||
                   !std::isfinite(state.angular_velocity.z))
                    throw std::runtime_error(
                        "Kinematic contact source ordering produced invalid Dy state");
            }
        }
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
        // Verify the actual solved rigid response, not just the
        // persistent Gu manager's swap bit. Both generators preserve
        // their independent original-CPU reference configuration.
        const auto run=[&](ExecutionMode mode,bool legacy,
                           bool kinematic_first) {
            WorldOptions opts{mode};
            opts.pcm_box_box=!legacy;
            World world(opts);
            BodyId dynamic=0,kinematic=0;
            const auto add_dynamic=[&] {
                dynamic=world.add_dynamic_box(
                    half,dynamic_position,1.0f);
            };
            const auto add_kinematic=[&] {
                kinematic=world.add_dynamic_box(
                    half,kinematic_position,1.0f);
            };
            if(kinematic_first) {add_kinematic();add_dynamic();}
            else {add_dynamic();add_kinematic();}
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            d.linear_damping=0.0f;
            d.angular_damping=0.0f;
            world.set_body_dynamics(dynamic,d);
            world.set_body_dynamics(kinematic,d);
            world.set_body_kinematic(kinematic,true);
            world.set_body_velocity(dynamic,{0.2f,0,0});
            for(int frame=0;frame<6;++frame)
                world.step(1.0f/120.0f);
            return world.body_state(dynamic);
        };
        const auto error=[](Vec3 a,Vec3 b) {
            return std::max({std::abs(a.x-b.x),std::abs(a.y-b.y),
                             std::abs(a.z-b.z)});
        };
        for(const bool legacy:{false,true})
            for(const bool kinematic_first:{false,true}) {
                const auto native=run(ExecutionMode::NativeCpu,
                                      legacy,kinematic_first);
                const auto original=run(legacy
                    ? ExecutionMode::ReferencePhysXCpuLegacyGu
                    : ExecutionMode::ReferencePhysXCpu,
                    legacy,kinematic_first);
                if(!std::isfinite(error(native.position,original.position)) ||
                   !std::isfinite(error(native.velocity,original.velocity)) ||
                   !std::isfinite(error(native.angular_velocity,
                                         original.angular_velocity)) ||
                   error(native.position,original.position)>0.02f ||
                   error(native.velocity,original.velocity)>0.06f ||
                   error(native.angular_velocity,
                         original.angular_velocity)>0.1f)
                    throw std::runtime_error(
                        "Native/PhysX kinematic contact ordering diverged");
            }
#endif
    }
    static void verify_kinematic_target_source_reference_parity() {
        // ScKinematics.cpp calculates target velocity at dt and moves
        // the body only in postSolver; compare the public production
        // native actor to an actual PxRigidDynamic with eKINEMATIC.
        const auto run=[](ExecutionMode mode) {
            World world(WorldOptions{mode});
            const auto id=world.add_dynamic_box(
                {0.25f,0.5f,0.4f},{0,6,0},1.0f);
            world.set_body_kinematic(id,true);
            const Quat q{0,0,std::sin(0.1f),std::cos(0.1f)};
            world.set_body_kinematic_target(id,{0.4f,6.1f,0},q);
            world.step(1.0f/120.0f);
            const BodyState first=world.body_state(id);
            world.step(1.0f/120.0f);
            const BodyState settled=world.body_state(id);
            return std::pair{first,settled};
        };
        const auto [native_first,native_settled]=run(ExecutionMode::NativeCpu);
        if(std::abs(native_first.position.x-0.4f)>1e-6f ||
           std::abs(native_first.velocity.x-48.0f)>1e-4f ||
           std::abs(native_first.angular_velocity.z-24.0f)>1e-3f ||
           native_settled.velocity.x!=0.0f ||
           native_settled.angular_velocity.z!=0.0f)
            throw std::runtime_error(
                "Native Sc kinematic target velocity/pose diverged from source");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
        const auto [source_first,source_settled]=run(
            ExecutionMode::ReferencePhysXCpu);
        if(std::abs(native_first.position.x-source_first.position.x)>1e-5f ||
           std::abs(native_first.position.y-source_first.position.y)>1e-5f ||
           std::abs(native_first.orientation.z-source_first.orientation.z)>1e-5f ||
           std::abs(native_first.velocity.x-source_first.velocity.x)>1e-3f ||
           std::abs(native_first.angular_velocity.z-
                    source_first.angular_velocity.z)>1e-3f ||
           std::abs(native_settled.velocity.x-
                    source_settled.velocity.x)>1e-4f)
            throw std::runtime_error(
                "Sc kinematic target differs from independent PhysX CPU");
#endif
    }
    static void verify_settled_kinematic_contact_solver_input() {
        constexpr float dt=1.0f/120.0f;
        Scene scene{};
        const auto dynamic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto kinematic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.95f,8,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.linear_damping=0.0f;
        dynamics.angular_damping=0.0f;
        scene.set_dynamics(dynamic,dynamics);
        scene.set_dynamics(kinematic,dynamics);
        scene.set_kinematic(kinematic,true);
        scene.set_kinematic_target(kinematic,{0.95f,8,0},{});
        scene.step(dt);
        scene.step(dt);
        if(!scene.active_kinematic_targets_.empty() ||
           scene.accurate_island_count()!=1)
            throw std::runtime_error(
                "Settled kinematic contact fixture lacked active island");
        // A previous solver epoch's row is untrusted scratch, whether
        // it came from a target or a frame later rejected by the journal.
        auto& stale=scene.solver_rows_[kinematic];
        stale.solver.linear_velocity_delta={80,12,0};
        stale.solver.angular_state={4,3,2};
        stale.data.inverse_mass=9;
        scene.step(dt);
        const auto& row=scene.solver_rows_[kinematic];
        if(row.prepared_generation!=scene.solver_preparation_generation_ ||
           row.data.inverse_mass!=0.0f ||
           row.solver.linear_velocity_delta.x!=0.0f ||
           row.solver.angular_state.x!=0.0f ||
           !std::isfinite(scene.body_state(dynamic).velocity.x))
            throw std::runtime_error(
                "Settled kinematic contact reused old solver-body state");
    }
    static void verify_moving_kinematic_target_through_source_solver() {
        constexpr float dt=1.0f/120.0f;
        Scene scene{};
        const auto moving=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto kine=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.98f,8,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        scene.set_dynamics(moving,d);
        scene.set_dynamics(kine,d);
        scene.set_kinematic(kine,true);
        const Vec3 target{0.84f,8,0};
        scene.set_kinematic_target(kine,target,{});
        scene.step(dt);
        const auto moved=scene.body_state(kine);
        if(std::abs(moved.position.x-target.x)>1.0e-6f ||
           std::abs(moved.velocity.x+0.14f/dt)>1.0e-3f ||
           scene.accurate_islands_.node_island[kine]!=
               Scene::IslandGraph::invalid ||
           !std::isfinite(scene.body_state(moving).velocity.x))
            throw std::runtime_error(
                "Sc target velocity or postSolver kinematic pose incorrect");
        // The subsequent Gu publication consumes the target pose from
        // afterIntegration, not an early pre-PGS transform mutation.
        scene.step(dt);
        if(scene.body_state(kine).position.x!=target.x ||
           scene.body_state(kine).velocity.x!=0.0f ||
           scene.body_state(kine).angular_velocity.x!=0.0f)
            throw std::runtime_error(
                "Sc kinematic settling retained previous target velocity");
        scene.step(dt);
        if(scene.actors_[kine].kinematic_tracked ||
           !scene.active_kinematic_targets_.empty())
            throw std::runtime_error(
                "Sc kinematic target list retained settled body");
    }
    static void verify_kinematic_node_is_not_a_solver_island() {
        // The same PxRigidDynamic survives its eKINEMATIC transition,
        // but PxsIslandSim invalidates its rigid island membership.
        // A contacting ordinary dynamic still enters Dy contact solving.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        scene.set_dynamics(a,d);
        scene.set_dynamics(b,d);
        scene.step(1.0f/120.0f);
        if(scene.accurate_island_count()!=1)
            throw std::runtime_error(
                "Kinematic transition fixture lacks connected rigid island");
        scene.set_kinematic(b,true);
        const auto kin_pose=scene.body_state(b).position;
        scene.step(1.0f/120.0f);
        if(scene.actors_[b].kind!=Scene::Kind::KinematicBox ||
           scene.accurate_islands_.node_island[b]!=
               Scene::IslandGraph::invalid ||
           scene.accurate_island_count()!=1 ||
           scene.contacts().empty() ||
           scene.solver_tasks_.empty() ||
           b>=scene.speculative_islands_.active_reference_counts.size() ||
           b>=scene.accurate_islands_.active_reference_counts.size() ||
           scene.speculative_islands_.active_reference_counts[b]==0 ||
           scene.accurate_islands_.active_reference_counts[b]==0 ||
           scene.body_state(b).position.x!=kin_pose.x ||
           scene.body_state(b).position.y!=kin_pose.y ||
           scene.body_state(b).position.z!=kin_pose.z)
            throw std::runtime_error(
                "Kinematic rigid node leaked into Dy island/integration");

        scene.set_kinematic(b,false);
        scene.step(1.0f/120.0f);
        if(scene.actors_[b].kind!=Scene::Kind::DynamicBox ||
           scene.accurate_islands_.node_island[b]==
               Scene::IslandGraph::invalid ||
           scene.accurate_island_count()!=1)
            throw std::runtime_error(
                "Dynamic transition failed to readmit prior rigid node");
    }
    static void verify_wake_before_lost_edge_split() {
        // A queued wake propagates through the pre-split island during
        // source first/second-pass wakeIslands; third-pass route repair
        // must not isolate a still-sleeping former contact partner first.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto lower=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        const auto upper=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,11,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        scene.set_dynamics(lower,dynamics);
        scene.set_dynamics(upper,dynamics);
        bool slept=false;
        for(int frame=0;frame<240;++frame) {
            scene.step(1.0f/120.0f);
            if(scene.body_is_sleeping(lower) &&
               scene.body_is_sleeping(upper)) {
                slept=true;
                break;
            }
        }
        const Pair link{lower,upper};
        if(!slept ||
           scene.contact_edge_index(link)==
               std::numeric_limits<std::uint32_t>::max() ||
           scene.accurate_island_count()!=1)
            throw std::runtime_error(
                "Source wake/split test lacks a dormant linked island");
        scene.set_velocity(upper,{0,0.25f,0});
        if(!scene.body_is_sleeping(lower) ||
           scene.body_is_sleeping(upper))
            throw std::runtime_error(
                "Source wake/split test lacks a queued island activation");
        // Simulate an already-integrated transform entering the next
        // Gu update. set_pose would separately wake cached Gu partners
        // during authoring, obscuring this island ordering requirement.
        auto& moved=scene.actors_[upper];
        moved.position.x=4.0f;
        moved.cached_bounds_current=false;
        scene.dirty_=true;
        scene.advance_collision(true);
        if(scene.body_is_sleeping(lower) ||
           scene.body_is_sleeping(upper) ||
           scene.accurate_island_count()!=2 ||
           std::find(scene.lost_contacts().begin(),
                     scene.lost_contacts().end(),link)==
               scene.lost_contacts().end())
            throw std::runtime_error(
                "Source lost-edge repair split before waking the old island");
    }
    static void verify_mixed_pcm_legacy_cache_ownership() {
        // Source PxcNpCacheStreamPair indexes only legacy cached
        // contacts; Gu's published contact arena also includes PCM.
        // A preceding PCM plane manager must not displace the cached
        // box/box points belonging to the subsequent source manager.
        Scene mixed{Scene::ContactGenerator::DiagnosticLegacyGu};
        mixed.set_pcm_plane_box(true);
        mixed.add_static_floor(0.0f);
        const auto lower=mixed.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto upper=mixed.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,1.47f,0},1.0f);
        Scene legacy{Scene::ContactGenerator::DiagnosticLegacyGu};
        legacy.add_static_floor(0.0f);
        legacy.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto source_upper=legacy.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,1.47f,0},1.0f);
        const Pair box_pair{lower,upper};
        const auto box_manager=[](const Scene& scene,Pair pair)
            ->const ContactManifold& {
            for(const auto& manager:scene.contacts())
                if(manager.pair==pair)return manager;
            throw std::runtime_error(
                "Mixed source PCM scene lost its legacy box-box manager");
        };
        const auto compare=[&] {
            const auto& m=box_manager(mixed,box_pair);
            const auto& l=box_manager(legacy,box_pair);
            if(m.count!=l.count || m.count==0)
                throw std::runtime_error(
                    "Mixed PCM scene altered legacy box-box contact count");
            for(std::size_t i=0;i<m.count;++i) {
                const auto& a=m.points[i],&b=l.points[i];
                if(std::abs(a.position.x-b.position.x)>2e-5f ||
                   std::abs(a.position.y-b.position.y)>2e-5f ||
                   std::abs(a.position.z-b.position.z)>2e-5f ||
                   std::abs(a.normal.x-b.normal.x)>2e-5f ||
                   std::abs(a.normal.y-b.normal.y)>2e-5f ||
                   std::abs(a.normal.z-b.normal.z)>2e-5f ||
                   std::abs(a.separation-b.separation)>2e-5f)
                    throw std::runtime_error(
                        "PCM plane points contaminated source legacy box-box contact cache");
            }
            if(mixed.gu_cache_records_.size()!=1 ||
               mixed.gu_cache_records_[0].pair!=box_pair ||
               mixed.gu_cache_records_[0].offset!=0 ||
               mixed.gu_cache_records_[0].count!=m.count ||
               mixed.gu_cached_points_.size()!=m.count ||
               m.contact_offset==mixed.gu_cache_records_[0].offset)
                throw std::runtime_error(
                    "PCM and legacy Gu offsets were incorrectly conflated");
            // PxcCacheLocalContacts retains RAW source contacts on a
            // hit. Published contacts may move by half the relative
            // displacement, while the cached payload must remain in
            // its original generating frame. Compare both independent
            // arenas with the unmodified legacy-only source run.
            // The infinite plane also owns a separate legacy Gu
            // cache for the upper box even when they are too distant
            // to touch. Locate the box/box manager by its Pair key,
            // never by an assumed position in the cache stream.
            const Scene::GuCacheRecord* source_cache=nullptr;
            for(const auto& entry:legacy.gu_cache_records_)
                if(entry.pair==box_pair) {
                    source_cache=&entry;
                    break;
                }
            if(!source_cache ||
               source_cache->count!=m.count ||
               source_cache->offset+m.count>legacy.gu_cached_points_.size())
                throw std::runtime_error(
                    "Legacy Gu source cache control has invalid ownership");
            for(std::size_t i=0;i<m.count;++i) {
                const auto& raw=mixed.gu_cached_points_[i];
                const auto& expected=legacy.gu_cached_points_[
                    source_cache->offset+i];
                if(std::abs(raw.position.x-expected.position.x)>2e-5f ||
                   std::abs(raw.position.y-expected.position.y)>2e-5f ||
                   std::abs(raw.position.z-expected.position.z)>2e-5f ||
                   std::abs(raw.normal.x-expected.normal.x)>2e-5f ||
                   std::abs(raw.normal.y-expected.normal.y)>2e-5f ||
                   std::abs(raw.normal.z-expected.normal.z)>2e-5f ||
                   std::abs(raw.separation-expected.separation)>2e-5f)
                    throw std::runtime_error(
                        "PCM contact stream displaced or mutated a legacy Pxc cache span");
            }
        };
        mixed.advance_collision(true);
        legacy.advance_collision(true);
        if(mixed.contacts().size()<2 ||
           mixed.contacts()[0].pair!=Pair{0,lower})
            throw std::runtime_error(
                "PCM floor manager did not precede legacy box-box in Gu stream");
        compare();
        // A source Pxc cache HIT must retain the same old points
        // through a small authored translation of the upper box.
        mixed.set_position(upper,{0,1.474f,0});
        legacy.set_position(source_upper,{0,1.474f,0});
        mixed.advance_collision(true);
        legacy.advance_collision(true);
        compare();
        // Mixed PCM and legacy Gu managers must enter the same Dy
        // constraint solver and successfully publish the result.
        mixed.step(1.0f/120.0f);
        if(mixed.solved_result_frame()!=mixed.frame_index())
            throw std::runtime_error(
                "Mixed PCM/legacy Gu managers failed native PGS completion");
    }
    static void verify_source_pcm_batch_reduction() {
        // GuPersistentContactManifold::reduceBatchContacts retains the
        // deepest, furthest, positive transverse extreme, and negative
        // transverse extreme. The tolerance-depth pass substitutes a
        // shallow chosen point with the deepest eligible leftover.
        constexpr Vec3 normal{0,1,0};
        constexpr std::array<PcmContactCandidate,6> candidates{{
            {{0,0,0},{0,0,0},normal,-0.20f},
            {{1,0,0},{1,0,0},normal,-0.10f},
            {{-1,0,0},{-1,0,0},normal,-0.08f},
            {{0,0,1},{0,0,1},normal,-0.05f},
            {{0,0,-1},{0,0,-1},normal,0.10f},
            {{0.5f,0,-0.5f},{0.5f,0,-0.5f},normal,-0.01f}
        }};
        Scene::PcmManifoldCache manifold{};
        Scene::reduce_pcm_box_batch(
            manifold,std::span<const PcmContactCandidate>(candidates),1.0f);
        if(manifold.count!=4 ||
           manifold.points[0].local_b.x!=0.0f ||
           manifold.points[0].separation!=-0.20f ||
           manifold.points[1].local_b.x!=1.0f ||
           manifold.points[2].local_b.z!=1.0f ||
           manifold.points[3].local_b.x!=-1.0f ||
           manifold.points[3].separation!=-0.08f)
            throw std::runtime_error(
                "PCM source batch reducer changed candidate order or depth replacement");
    }
    static void verify_default_native_pcm_scene() {
        // A newly constructed source-native scene must use PCM for the
        // admitted primitives without relying on World configuration
        // or any opt-in test switch. Legacy Gu remains diagnostic only.
        {
            Scene scene;
            scene.add_static_floor(0.0f);
            const auto moving=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               scene.contacts()[0].count!=4 ||
               !scene.gu_cache_records_.empty() ||
               !scene.gu_cached_points_.empty() ||
               !scene.island_edges_[scene.contacts()[0].edge_handle].
                   pcm_cache.initialized)
                throw std::runtime_error(
                    "Default native scene did not admit plane-box PCM");
            const auto manager=scene.contacts()[0].edge_handle;
            scene.step(1.0f/120.0f);
            if(scene.solved_result_frame()!=scene.frame_index())
                throw std::runtime_error(
                    "Default plane-box PCM did not reach native Dy");
            // Friction-anchor correlation, sleeping island retirement and
            // explicit wake must preserve this very PCM-owned manager.
            // They may not switch silently to the Gu cache once the
            // first set of contacts has reached Dy.
            bool correlated=false;
            bool slept=false;
            for(int frame=0;frame<480;++frame) {
                scene.step(1.0f/120.0f);
                for(const auto& patch:scene.solved_friction_patches())
                    if(patch.anchor_count>0)correlated=true;
                if(scene.body_is_sleeping(moving)) {
                    slept=true;
                    break;
                }
            }
            if(!correlated || !slept ||
               !scene.island_edges_[manager].pcm_cache.initialized ||
               !scene.gu_cache_records_.empty())
                throw std::runtime_error(
                    "Default PCM contact lost friction anchors or sleep lifecycle");
            scene.set_velocity(moving,{0.12f,0,0});
            if(scene.body_is_sleeping(moving))
                throw std::runtime_error(
                    "Default PCM scene failed authored velocity wake");
            scene.step(1.0f/120.0f);
            if(scene.body_is_sleeping(moving) ||
               scene.solved_result_frame()!=scene.frame_index() ||
               !scene.island_edges_[manager].pcm_cache.initialized ||
               !scene.gu_cache_records_.empty())
                throw std::runtime_error(
                    "Default PCM wake did not reactivate the same Sc/Dy manager");
        }
        {
            Scene scene;
            scene.add_static_box({0.5f,0.5f,0.5f},{0,0.5f,0});
            scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,1.46f,0},1.0f);
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               scene.contacts()[0].count!=4 ||
               !scene.gu_cache_records_.empty() ||
               !scene.gu_cached_points_.empty() ||
               !scene.island_edges_[scene.contacts()[0].edge_handle].
                   pcm_cache.initialized)
                throw std::runtime_error(
                    "Default native scene did not admit box-box PCM");
            scene.step(1.0f/120.0f);
            if(scene.solved_result_frame()!=scene.frame_index())
                throw std::runtime_error(
                    "Default box-box PCM did not reach native Dy");
        }
    }
    static void verify_default_pcm_failure_atomic_cache() {
        // A rejected Dy frame must restore the same manager-owned PCM
        // manifold it observed before Gu contact regeneration. Exercise
        // both primitive specializations with integrated (not authored)
        // motion so an existing cache is mutated inside the transaction.
        for(bool plane_pair:{false,true}) {
            Scene scene;
            if(plane_pair)
                scene.add_static_floor(0.0f);
            else
                scene.add_static_box(
                    {0.5f,0.5f,0.5f},{0,0.5f,0});
            const auto moving=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},
                {0,plane_pair?0.5f:1.46f,0},1.0f);
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               scene.contacts()[0].count!=4)
                throw std::runtime_error(
                    "PCM rollback fixture failed to establish cache");
            const auto edge=scene.contacts()[0].edge_handle;
            const auto before=scene.island_edges_[edge].pcm_cache;
            if(!before.initialized || before.count!=4)
                throw std::runtime_error(
                    "PCM rollback fixture lacks retained witnesses");
            const auto frame=scene.frame_index();
            auto& body=scene.actors_[moving];
            body.position.x+=0.125f;
            body.cached_bounds_current=false;
            scene.dirty_=true;
            bool rejected=false;
            try {
                scene.step(std::numeric_limits<float>::max());
            } catch(const std::overflow_error&) {
                rejected=true;
            }
            const auto& restored=scene.island_edges_[edge].pcm_cache;
            if(!rejected || scene.frame_index()!=frame ||
               restored.initialized!=before.initialized ||
               restored.count!=before.count ||
               restored.relative_position.x!=before.relative_position.x ||
               restored.relative_position.y!=before.relative_position.y ||
               restored.relative_position.z!=before.relative_position.z ||
               restored.relative_rotation.x!=before.relative_rotation.x ||
               restored.relative_rotation.y!=before.relative_rotation.y ||
               restored.relative_rotation.z!=before.relative_rotation.z ||
               restored.relative_rotation.w!=before.relative_rotation.w ||
               restored.warmstart.count!=before.warmstart.count)
                throw std::runtime_error(
                    "Rejected source Dy frame leaked PCM cache update");
            for(std::size_t i=0;i<before.count;++i) {
                const auto& a=before.points[i];
                const auto& b=restored.points[i];
                if(a.local_a.x!=b.local_a.x ||
                   a.local_a.y!=b.local_a.y ||
                   a.local_a.z!=b.local_a.z ||
                   a.local_b.x!=b.local_b.x ||
                   a.local_b.y!=b.local_b.y ||
                   a.local_b.z!=b.local_b.z ||
                   a.normal_b.x!=b.normal_b.x ||
                   a.normal_b.y!=b.normal_b.y ||
                   a.normal_b.z!=b.normal_b.z ||
                   a.separation!=b.separation)
                    throw std::runtime_error(
                        "Rejected PCM step changed a retained witness");
            }
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               !scene.island_edges_[edge].pcm_cache.initialized ||
               std::abs(scene.island_edges_[edge].
                        pcm_cache.relative_position.x-
                        before.relative_position.x)<0.10f)
                throw std::runtime_error(
                    "PCM rollback did not permit contact regeneration");
            scene.step(1.0f/120.0f);
            if(scene.solved_result_frame()!=scene.frame_index())
                throw std::runtime_error(
                    "PCM rollback retry did not complete native PGS");
        }
    }
    static void verify_failed_pcm_regeneration_clears_retained_points() {
        // An empty GJK/EPA result is a real touch loss. Its previous
        // refreshed witnesses may still fit the projection-breaking
        // threshold, but must not be replayed by a later quiet update.
        // This checks the actual Sc manager/cache and Gu publication,
        // including both original source actor orders.
        constexpr float scale=0.01f;
        const Vec3 half{0.5f*scale,0.5f*scale,0.5f*scale};
        const Vec3 rotated_at{0.892f*scale,1.091f*scale,0};
        const Quat yaw{0,0,std::sin(0.1f),std::cos(0.1f)};
        for(bool reversed:{false,true}) {
            Scene scene{};
            BodyId moving{},fixed{};
            if(reversed) {
                moving=scene.add_dynamic_box(half,rotated_at,1.0f);
                fixed=scene.add_static_box(half,{0,0,0});
            } else {
                fixed=scene.add_static_box(half,{0,0,0});
                moving=scene.add_dynamic_box(half,rotated_at,1.0f);
            }
            scene.set_shape_offsets(fixed,0.02f*scale,0);
            scene.set_shape_offsets(moving,0.02f*scale,0);
            scene.set_pose(moving,rotated_at,yaw);
            scene.advance_collision(true);
            const Pair pair{std::min(fixed,moving),std::max(fixed,moving)};
            const auto edge=scene.contact_edge_index(pair);
            if(edge==std::numeric_limits<std::uint32_t>::max() ||
               !scene.contacts().empty())
                throw std::runtime_error(
                    "Failed PCM regeneration test lacks source no-contact manager");

            // Seed a refreshed contact that remains tangentially
            // admissible at the current pose but has an invalid cached
            // relative transform. The next real Gu pass must regenerate
            // and reject it via the original GJK/EPA result.
            auto& cache=scene.island_edges_[edge].pcm_cache;
            const Vec3 actual_relative=cache.relative_position;
            cache.initialized=true;
            cache.count=1;
            cache.points[0]={{},actual_relative,{1,0,0},0};
            cache.relative_position={9,9,9};
            scene.advance_collision(true);
            if(!scene.contacts().empty() || cache.count!=0)
                throw std::runtime_error(
                    "Failed PCM regeneration retained phantom witnesses");
            scene.advance_collision(true);
            if(!scene.contacts().empty() ||
               !scene.found_contacts().empty() ||
               !scene.lost_contacts().empty())
                throw std::runtime_error(
                    "Failed PCM regeneration replayed stale contact on quiet frame");

            // A subsequent actual pose change must still regenerate a
            // valid face manifold on this same EdgeIndex.
            scene.set_pose(moving,{0,0.9f*scale,0},{});
            scene.advance_collision(true);
            if(scene.contact_edge_index(pair)!=edge ||
               scene.contacts().size()!=1 ||
               scene.contacts()[0].count==0 ||
               scene.found_contacts().size()!=1)
                throw std::runtime_error(
                    "Empty PCM cache prevented legitimate touch reentry");
        }
    }
    static void verify_source_pcm_box_box_lifecycle() {
        verify_source_pcm_batch_reduction();
        // Independently exercise GJK and EPA for box support points in
        // B-local coordinates. Near separation produces GJK_CONTACT;
        // a penetrating pair requires EPA_CONTACT and signed depth.
        {
            PcmContactCandidate gjk{},epa{};
            const bool near=source_pcm_box_box_gjk_epa(
                {0.5f,0.5f,0.5f},{0,1.52f,0},{},
                {0.5f,0.5f,0.5f},{0,0.5f,0},{},
                0.04f,gjk);
            const bool deep=source_pcm_box_box_gjk_epa(
                {0.5f,0.5f,0.5f},{0,1.46f,0},{},
                {0.5f,0.5f,0.5f},{0,0.5f,0},{},
                0.04f,epa);
            if(!near || !deep ||
               std::abs(gjk.separation-0.02f)>2e-3f ||
               std::abs(epa.separation+0.04f)>2e-3f ||
               gjk.normal_b.y<0.999f || epa.normal_b.y<0.999f) {
                std::cerr<<"PCM primitive GJK/EPA near="<<near
                         <<" sep="<<gjk.separation
                         <<" normal=("<<gjk.normal_b.x<<","
                         <<gjk.normal_b.y<<","<<gjk.normal_b.z<<")"
                         <<" deep="<<deep
                         <<" sep="<<epa.separation
                         <<" normal=("<<epa.normal_b.x<<","
                         <<epa.normal_b.y<<","<<epa.normal_b.z<<")\n";
                throw std::runtime_error(
                    "Native PCM GJK/EPA sign, witness or depth differs from source");
            }
        }
        // Exact speculative boundary: SAT accepts a box face at
        // contactDistance, while direct GuPCMContactBoxBox clipping
        // requires separation strictly below that distance. GJK
        // must publish the resulting contact at the inclusive limit.
        {
            constexpr Vec3 half{0.5f,0.5f,0.5f};
            constexpr float contact_distance=0.03125f;
            constexpr Vec3 at{0,1.03125f,0};
            std::array<PcmContactCandidate,64> direct{};
            std::size_t count=0;
            if(!source_pcm_box_box_candidates(
                    half,at,{},half,{0,0,0},{},
                    contact_distance,direct,count) || count!=0)
                throw std::runtime_error(
                    "PCM speculative boundary did not require GJK continuation");
            PcmContactCandidate fallback{};
            PcmGjkWarmstart simplex{};
            if(!source_pcm_box_box_gjk_epa(
                    half,at,{},half,{0,0,0},{},
                    contact_distance,fallback,&simplex) ||
               std::abs(fallback.separation-contact_distance)>1.0e-6f ||
               fallback.normal_b.y<0.999f || simplex.count==0)
                throw std::runtime_error(
                    "Native PCM GJK speculative boundary or warm-start differs");
            // This primitive's zero-face regeneration clears the old
            // simplex before GJK, even when a manager has saved indices.
            PcmGjkWarmstart stale{};
            stale.count=4;
            stale.a={7,0,6,1};
            stale.b={0,7,1,6};
            PcmContactCandidate restarted{};
            if(!source_pcm_box_box_gjk_epa(
                    half,at,{},half,{0,0,0},{},
                    contact_distance,restarted,&stale) ||
               std::abs(restarted.separation-fallback.separation)>1.0e-6f ||
               restarted.normal_b.y<0.999f ||
               stale.count==0 || stale.count>4)
                throw std::runtime_error(
                    "PCM box fallback did not discard stale GJK simplex");
            Scene actual{Scene::ContactGenerator::DiagnosticLegacyGu};
            actual.set_pcm_box_box(true);
            const auto fixed=actual.add_static_box(half,{0,0,0});
            const auto moving=actual.add_dynamic_box(half,at,1.0f);
            actual.set_shape_offsets(fixed,0.015625f,0.0f);
            actual.set_shape_offsets(moving,0.015625f,0.0f);
            actual.advance_collision(true);
            if(actual.contacts().size()!=1 ||
               actual.contacts()[0].count!=1 ||
               std::abs(actual.contacts()[0].points[0].separation-
                        contact_distance)>1.0e-6f)
                throw std::runtime_error(
                    "Native PCM boundary GJK witness not published");
            const auto edge=actual.contacts()[0].edge_handle;
            if(actual.island_edges_[edge].pcm_cache.warmstart.count==0)
                throw std::runtime_error(
                    "GJK simplex ownership lost on native contact manager");
            actual.step(1.0f/120.0f);
            if(actual.solved_result_frame()!=actual.frame_index())
                throw std::runtime_error(
                    "GJK boundary witness did not complete native PGS");
        }
        // A positive gap can be smaller than 10% of BoxV's 5%
        // minimum box-extent margin: 0.00390625 < 0.004375 for
        // half-extents 0.875. The reference/incident plane witnesses
        // must be judged by GJK geometry, not mistaken for overlap.
        // Rotating about Y keeps the exact vertical gap while exercising
        // rotated support vertices and nontrivial closest-point paths.
        {
            constexpr Vec3 half{0.875f,0.875f,0.875f};
            constexpr float gap=0.00390625f;
            constexpr Vec3 moving_at{0,1.75390625f,0};
            const Quat yaw{0,std::sin(0.14f),0,std::cos(0.14f)};
            std::array<PcmContactCandidate,64> direct{};
            std::size_t count=0;
            if(!source_pcm_box_box_candidates(
                    half,moving_at,yaw,half,{0,0,0},{},
                    gap,direct,count) || count!=0)
                throw std::runtime_error(
                    "Rotated sub-margin speculative contact did not enter GJK");
            PcmContactCandidate separated{};
            const bool admitted=source_pcm_box_box_gjk_epa(
                half,moving_at,yaw,half,{0,0,0},{},
                gap,separated);
            const float normal_sq=
                separated.normal_b.x*separated.normal_b.x+
                separated.normal_b.y*separated.normal_b.y+
                separated.normal_b.z*separated.normal_b.z;
            if(!admitted || !std::isfinite(separated.separation) ||
               std::abs(separated.separation)>gap+1e-3f ||
               !(normal_sq>0.99f && normal_sq<1.01f)) {
                std::cerr<<"sub-margin GJK admitted="<<admitted
                         <<" separation="<<separated.separation
                         <<" target="<<gap
                         <<" normal=("<<separated.normal_b.x<<","
                         <<separated.normal_b.y<<","
                         <<separated.normal_b.z<<")\n";
                throw std::runtime_error(
                    "Sub-margin positive GJK gap was lost to EPA");
            }
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            scene.set_pcm_box_box(true);
            const auto fixed=scene.add_static_box(half,{0,0,0});
            const auto moving=scene.add_dynamic_box(half,moving_at,1.0f);
            scene.set_shape_offsets(fixed,gap*0.5f,0.0f);
            scene.set_shape_offsets(moving,gap*0.5f,0.0f);
            scene.set_pose(moving,moving_at,yaw);
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               scene.contacts()[0].count!=1 ||
               std::abs(scene.contacts()[0].points[0].separation-
                        separated.separation)>1e-5f)
                throw std::runtime_error(
                    "Scene PCM manager lost sub-margin EPA contact");
        }
        // Two unit boxes diagonally separated by an exact binary
        // fraction share no projected reference/incident face region.
        // The closest witnesses lie on parallel Z edges, separated by
        // sqrt(2)*0.0009765625: below BoxV's 0.0025 EPA threshold,
        // yet strictly greater than zero. SAT admits speculation; the
        // native manager must retain one GJK edge/edge witness.
        {
            constexpr Vec3 half{0.5f,0.5f,0.5f};
            constexpr float axis_gap=0.0009765625f;
            constexpr Vec3 moving_at{
                1.0009765625f,1.0009765625f,0};
            constexpr float contact_distance=0.04f;
            std::array<PcmContactCandidate,64> faces{};
            std::size_t count=0;
            if(!source_pcm_box_box_candidates(
                    half,moving_at,{},half,{0,0,0},{},
                    contact_distance,faces,count) || count!=0)
                throw std::runtime_error(
                    "PCM edge/edge fixture did not bypass incident faces");
            PcmContactCandidate witness{};
            if(!source_pcm_box_box_gjk_epa(
                    half,moving_at,{},half,{0,0,0},{},
                    contact_distance,witness) ||
               !std::isfinite(witness.separation) ||
               std::abs(witness.separation)>contact_distance ||
               witness.normal_b.x<0.5f)
                throw std::runtime_error(
                    "PCM source EPA edge/edge witness failed");
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            scene.set_pcm_box_box(true);
            scene.add_static_box(half,{0,0,0});
            scene.add_dynamic_box(half,moving_at,1.0f);
            scene.advance_collision(true);
            if(scene.contacts().size()!=1 ||
               scene.contacts()[0].count!=1 ||
               std::abs(scene.contacts()[0].points[0].separation-
                   witness.separation)>2e-4f)
                throw std::runtime_error(
                    "Production Sc PCM edge/edge contact disappeared");
        }
        // Scale the same rotated edge/corner penetration across two
        // orders of magnitude. All three shapes of the test (the
        // support points, authored contact offsets and first pose)
        // scale together; a fixed absolute EPA facet threshold must
        // not silently discard the small contact manager.
        {
            struct Result {
                ContactPoint witness{};
                std::uint32_t edge=0;
            };
            const auto probe=[](float scale,bool reversed)->Result {
                const Vec3 half{
                    0.5f*scale,0.5f*scale,0.5f*scale};
                const Vec3 position{
                    0.892f*scale,1.091f*scale,0};
                const Quat rotation{
                    0,0,std::sin(0.1f),std::cos(0.1f)};
                const float offset=0.02f*scale;
                std::array<PcmContactCandidate,64> direct{};
                std::size_t count=0;
                if(!source_pcm_box_box_candidates(
                       half,position,rotation,half,{0,0,0},{},
                       2.0f*offset,direct,count) || count!=0)
                    throw std::runtime_error(
                        "Scaled rotated PCM probe did not reach GJK/EPA");
                Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
                scene.set_pcm_box_box(true);
                BodyId moving{},fixed{};
                if(reversed) {
                    moving=scene.add_dynamic_box(half,position,1.0f);
                    fixed=scene.add_static_box(half,{0,0,0});
                } else {
                    fixed=scene.add_static_box(half,{0,0,0});
                    moving=scene.add_dynamic_box(half,position,1.0f);
                }
                scene.set_shape_offsets(fixed,offset,0.0f);
                scene.set_shape_offsets(moving,offset,0.0f);
                scene.set_pose(moving,position,rotation);
                scene.advance_collision(true);
                if(scene.contacts().size()!=1 ||
                   scene.contacts()[0].count!=1 ||
                   !scene.island_edges_[scene.contacts()[0].edge_handle].
                        pcm_cache.initialized)
                    throw std::runtime_error(
                        "Scaled rotated GJK/EPA contact not admitted");
                const auto point=scene.contacts()[0].points[0];
                if(!std::isfinite(point.separation) ||
                   !(point.separation<0.0f) ||
                   !std::isfinite(point.normal.x) ||
                   !std::isfinite(point.normal.y))
                    throw std::runtime_error(
                        "Scaled EPA contact lost penetrating witness");
                const auto manager=scene.contacts()[0].edge_handle;
                scene.step(1.0f/120.0f);
                if(scene.solved_result_frame()!=scene.frame_index())
                    throw std::runtime_error(
                        "Scaled EPA contact failed native Dy completion");
                return {point,manager};
            };
            for(bool reversed:{false,true}) {
                const auto baseline=probe(1.0f,reversed);
                const auto small=probe(0.1f,reversed);
                const auto& a=baseline.witness;
                const auto& b=small.witness;
                const float alignment=a.normal.x*b.normal.x+
                    a.normal.y*b.normal.y+a.normal.z*b.normal.z;
                const float dx=a.position.x-b.position.x*10.0f;
                const float dy=a.position.y-b.position.y*10.0f;
                const float dz=a.position.z-b.position.z*10.0f;
                if(alignment<0.995f ||
                   std::abs(a.separation-b.separation*10.0f)>5e-4f ||
                   dx*dx+dy*dy+dz*dz>1e-4f) {
                    std::cerr<<"Scaled PCM witness reversed="<<reversed
                             <<" large_sep="<<a.separation
                             <<" small_sep="<<b.separation
                             <<" normal_dot="<<alignment
                             <<" dist2="<<(dx*dx+dy*dy+dz*dz)<<'\n';
                    throw std::runtime_error(
                        "Scaled PCM GJK/EPA witness lost geometric invariance");
                }
            }
            // GuEPA::Facet::isValid2 uses a fixed FEps() squared-area
            // threshold. The same contact configuration at 1/100
            // scale has no source EPA facet, despite finite geometry.
            for(bool reversed:{false,true}) {
                constexpr float scale=0.01f;
                const Vec3 half{0.5f*scale,0.5f*scale,0.5f*scale};
                const Vec3 position{0.892f*scale,1.091f*scale,0};
                const Quat rotation{
                    0,0,std::sin(0.1f),std::cos(0.1f)};
                Scene tiny{Scene::ContactGenerator::DiagnosticLegacyGu};
                tiny.set_pcm_box_box(true);
                BodyId moving{},fixed{};
                if(reversed) {
                    moving=tiny.add_dynamic_box(half,position,1);
                    fixed=tiny.add_static_box(half,{0,0,0});
                } else {
                    fixed=tiny.add_static_box(half,{0,0,0});
                    moving=tiny.add_dynamic_box(half,position,1);
                }
                tiny.set_shape_offsets(fixed,0.02f*scale,0.0f);
                tiny.set_shape_offsets(moving,0.02f*scale,0.0f);
                tiny.set_pose(moving,position,rotation);
                tiny.advance_collision(true);
                if(!tiny.contacts().empty())
                    throw std::runtime_error(
                        "Native PCM bypassed original EPA facet admission");
                tiny.step(1.0f/120.0f);
                if(tiny.solved_result_frame()!=tiny.frame_index())
                    throw std::runtime_error(
                        "Source EPA no-contact case did not complete Dy frame");
            }
        }
        // PCM box-box SAT uses B-local contacts and four persistent
        // witnesses, unlike GuContactBoxBox's unreduced raw contact
        // stream and PxcCacheLocalContacts pose snapshots.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_pcm_box_box(true);
        const auto fixed=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0});
        const auto dynamic=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,1.46f,0},1.0f);
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 || scene.contacts()[0].count!=4)
            throw std::runtime_error(
                "Native source PCM face/face must publish four witnesses");
        const auto edge=scene.contacts()[0].edge_handle;
        if(scene.pcm_candidate_scratch_.size()!=64)
            throw std::runtime_error(
                "PCM box pair did not allocate the reusable source candidate arena");
        const auto* const reused_pcm_buffer=
            scene.pcm_candidate_scratch_.data();
        const auto& cache=scene.island_edges_[edge].pcm_cache;
        if(!cache.initialized || cache.count!=4 ||
           !scene.gu_cache_records_.empty() ||
           !scene.gu_cached_points_.empty())
            throw std::runtime_error(
                "PCM box-box cache leaked into legacy Pxc ownership");
        const float original_local_a_x=cache.points[0].local_a.x;
        for(const auto& point:scene.contacts()[0].points) {
            if(std::abs(point.position.y-1.0f)>1e-5f ||
               std::abs(point.separation+0.04f)>1e-5f ||
               point.normal.y<0.999f)
                throw std::runtime_error(
                    "PCM box-box point lacks B-local incident witness geometry");
        }
        // Internal Dy integration refreshes retained witnesses, but
        // does not invoke authored-pose cache invalidation.
        const auto saved=cache.relative_position;
        scene.actors_[dynamic].position.x=0.003f;
        scene.actors_[dynamic].cached_bounds_current=false;
        scene.dirty_=true;
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].edge_handle!=edge ||
           scene.island_edges_[edge].pcm_cache.count!=4 ||
           scene.island_edges_[edge].pcm_cache.relative_position.x!=saved.x)
            throw std::runtime_error(
                "PCM box-box steady cache was regenerated instead of refreshed");
        // An authored pose must reset the same manager's PCM geometry
        // cache; new B witnesses follow the new physical shape pose.
        scene.set_position(dynamic,{0.02f,1.46f,0});
        if(scene.island_edges_[edge].pcm_cache.initialized)
            throw std::runtime_error(
                "Authored PCM box-box pose retained obsolete witnesses");
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].count!=4 ||
           !scene.island_edges_[edge].pcm_cache.initialized)
            throw std::runtime_error(
                "PCM box-box authored-pose reconstruction failed");
        if(scene.pcm_candidate_scratch_.data()!=reused_pcm_buffer)
            throw std::runtime_error(
                "PCM regeneration reallocated scene-owned candidate scratch");
        if(std::abs(scene.island_edges_[edge].pcm_cache.
                        relative_position.x-0.02f)>1.0e-6f ||
           std::abs(scene.island_edges_[edge].pcm_cache.
                        points[0].local_a.x-original_local_a_x)<0.01f)
            throw std::runtime_error(
                "PCM authored pose failed to rebuild first-shape local witnesses");
        // Native Dy must consume PCM points, not silently run Gu in
        // a disconnected test-only narrowphase.
        scene.step(1.0f/120.0f);
        if(scene.solved_result_frame()!=scene.frame_index())
            throw std::runtime_error(
                "PCM box-box source contacts did not complete native Dy PGS");
        scene.remove(dynamic);
        scene.advance_collision(true);
        if(!scene.contacts().empty())
            throw std::runtime_error(
                "Retired native PCM box-box manager still publishes contacts");
        const auto replacement=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,1.46f,0},1.0f);
        scene.advance_collision(true);
        if(replacement==dynamic || scene.contacts().size()!=1 ||
           scene.contacts()[0].count!=4)
            throw std::runtime_error(
                "New box contact manager retained retired PCM witnesses");
        // GuPersistentContactManifold::getWorldNormal averages cached
        // B-local normals on a quiet refresh; it does not reuse the
        // first contact normal after a GJK-updated mixed-normal patch.
        // Prime a real scene-owned manifold, change one retained normal,
        // then require the unmodified Sc/PGS contact stream to use the
        // source aggregate rather than the leading witness.
        {
            Scene mixed{Scene::ContactGenerator::DiagnosticLegacyGu};
            mixed.set_pcm_box_box(true);
            mixed.add_static_box({0.5f,0.5f,0.5f},{0,0.5f,0});
            mixed.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,1.46f,0},1.0f);
            mixed.advance_collision(true);
            if(mixed.contacts().size()!=1 ||
               mixed.contacts()[0].count!=4)
                throw std::runtime_error(
                    "PCM aggregate-normal fixture did not form a four-point patch");
            const auto manager=mixed.contacts()[0].edge_handle;
            auto& manifold=mixed.island_edges_[manager].pcm_cache;
            manifold.points[1].normal_b={0.6f,0.8f,0};
            mixed.advance_collision(true);
            if(mixed.contacts().size()!=1 ||
               mixed.contacts()[0].edge_handle!=manager ||
               mixed.island_edges_[manager].pcm_cache.count!=4)
                throw std::runtime_error(
                    "PCM mixed-normal cache did not survive quiet refresh");
            const float expected_x=0.6f/std::sqrt(0.6f*0.6f+3.8f*3.8f);
            const float expected_y=3.8f/std::sqrt(0.6f*0.6f+3.8f*3.8f);
            for(const auto& point:mixed.contacts()[0].points)
                if(std::abs(point.normal.x-expected_x)>1e-5f ||
                   std::abs(point.normal.y-expected_y)>1e-5f)
                    throw std::runtime_error(
                        "PCM cached patch emitted the first normal instead of the source aggregate");
        }
        // Independent direct-contact generator must distinguish face
        // contact from separated SAT configurations without Gu.
        std::array<PcmContactCandidate,64> candidates{};
        std::size_t count=0;
        if(source_pcm_box_box_candidates(
                {0.5f,0.5f,0.5f},{0,0.5f,0},{},
                {0.5f,0.5f,0.5f},{0,1.46f,0},{},
                0.04f,candidates,count)==false || count!=4)
            throw std::runtime_error("PCM direct SAT failed face witnesses");
        if(source_pcm_box_box_candidates(
                {0.5f,0.5f,0.5f},{0,0.5f,0},{},
                {0.5f,0.5f,0.5f},{0,5.0f,0},{},
                0.04f,candidates,count) || count!=0)
            throw std::runtime_error("PCM SAT accepted separated boxes");
        (void)fixed;
    }
    static void verify_source_pcm_plane_box_lifecycle() {
        // Refresh is shared with subsequent convex PCM generation:
        // the signed separation and tangential eviction must be evaluated
        // in B-local coordinates using the stored B-local normal.
        {
            Scene::PcmManifoldCache cache;
            cache.count=2;
            cache.points[0]={{1.0f,0.0f,0.0f},
                             {0.0f,1.0f,0.0f},
                             {1.0f,0.0f,0.0f},0.0f};
            cache.points[1]={{0.0f,0.0f,0.0f},
                             {2.0f,0.0f,0.0f},
                             {0.0f,0.0f,1.0f},0.0f};
            const float root_half=std::sqrt(0.5f);
            Scene::refresh_pcm_contact_points(
                cache,{0.1f,0.0f,0.0f},
                {0.0f,0.0f,root_half,root_half},0.1f);
            if(cache.count!=1 ||
               std::abs(cache.points[0].separation-0.1f)>1.0e-5f ||
               cache.points[0].normal_b.x!=1.0f)
                throw std::runtime_error(
                    "PCM local-normal refresh or tangential eviction diverged");
        }
        // GuPCMContactPlaneBox::pcmContactPlaneBox + source persistent
        // refresh/invalidation/reduction execute in the actual Scene
        // publication path. The same Sc manager EdgeIndex owns its
        // cache and subsequent Dy constraint rows.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_pcm_plane_box(true);
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 || scene.contacts()[0].count!=4)
            throw std::runtime_error(
                "PCM floor contact did not admit four source cached witnesses");
        const auto edge=scene.contacts()[0].edge_handle;
        if(!scene.island_edges_[edge].pcm_cache.initialized ||
           scene.island_edges_[edge].pcm_cache.count!=4 ||
           !scene.gu_cache_records_.empty())
            throw std::runtime_error(
                "PCM manager used independent Gu cache or lost source manifold");
        const auto first=scene.contacts()[0].points[0];
        if(std::abs(first.position.y)>1.0e-6f ||
           first.normal.y!=1.0f ||
           std::abs(first.separation)>1.0e-5f)
            throw std::runtime_error(
                "PCM plane contact must project B onto plane, not legacy Gu box vertex");
        // Sc authored pose edits reset PxsContactManager's cache,
        // irrespective of whether the translation is inside the PCM
        // refresh threshold. Source first contact B moves accordingly.
        scene.set_position(box,{0.004f,0.5f,0});
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].edge_handle!=edge ||
           std::abs(scene.island_edges_[edge].
                    pcm_cache.relative_position.x-0.004f)>1.0e-6f ||
           std::abs(scene.contacts()[0].points[0].position.x-
                    (first.position.x+0.004f))>1.0e-6f)
            throw std::runtime_error(
                "PCM authored pose failed Sc manager cache reset");
        // Idempotent authored poses still reach
        // Sc::ShapeInteraction::resetManagerCachedState. A normal,
        // non-forced geometry advance must rebuild that invalidated
        // manager before returning contact spans.
        scene.set_position(box,{0.004f,0.5f,0});
        if(scene.island_edges_[edge].pcm_cache.initialized ||
           !scene.dirty_)
            throw std::runtime_error(
                "Idempotent Sc PCM pose failed manager invalidation");
        scene.advance_collision();
        if(!scene.island_edges_[edge].pcm_cache.initialized ||
           scene.contacts().size()!=1)
            throw std::runtime_error(
                "Idempotent Sc PCM pose did not republish Gu contacts");
        // Dy integration instead advances the SAME actor pose without
        // Sc's authored-pose callback. Reproduce that internal handoff
        // in this native-source fixture; the cached B point must remain
        // anchored in plane coordinates across a small motion.
        scene.actors_[box].position.x=0.008f;
        scene.actors_[box].cached_bounds_current=false;
        scene.dirty_=true;
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           scene.island_edges_[edge].pcm_cache.relative_position.x!=0.004f ||
           std::abs(scene.contacts()[0].points[0].position.x-
                    (first.position.x+0.004f))>1.0e-6f)
            throw std::runtime_error(
                "Solver-integrated PCM translation did not refresh retained B");
        scene.set_position(box,{0.05f,0.5f,0});
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].edge_handle!=edge ||
           std::abs(scene.island_edges_[edge].pcm_cache.relative_position.x-
                    0.05f)>1.0e-6f ||
           std::abs(scene.contacts()[0].points[0].position.x-
                    (first.position.x+0.05f))>1.0e-5f)
            throw std::runtime_error(
                "PCM displacement invalidation omitted source regeneration");
        // Internal integrated rotation (not an authored pose reset)
        // must trigger the actual PCM 0.99996 quaternion invalidation.
        const float theta=0.14f;
        scene.actors_[box].orientation={
            0,0,std::sin(theta/2),std::cos(theta/2)};
        scene.actors_[box].cached_bounds_current=false;
        scene.dirty_=true;
        scene.advance_collision(true);
        if(scene.contacts().empty() ||
           scene.contacts()[0].count>4 ||
           std::abs(scene.island_edges_[edge].
                    pcm_cache.relative_rotation.z)<0.01f)
            throw std::runtime_error(
                "PCM plane-box rotation did not rebuild four-point manifold");
        for(const auto& point:scene.contacts()[0].points)
            if(std::abs(point.position.y)>1.0e-5f ||
               point.normal.y!=1.0f)
                throw std::runtime_error(
                    "PCM rotated plane-box output lost plane projection");
        scene.set_position(box,{0.05f,3.0f,0});
        scene.advance_collision(true);
        if(!scene.contacts().empty() ||
           scene.island_edges_[edge].pcm_cache.count!=0)
            throw std::runtime_error(
                "PCM persistent manager did not retire separated points");
        scene.set_position(box,{0.05f,0.5f,0});
        scene.advance_collision(true);
        if(scene.contacts().empty() ||
           scene.contacts()[0].edge_handle!=edge)
            throw std::runtime_error(
                "PCM persistent manager did not regenerate after reentry");
        scene.remove(box);
        scene.advance_collision(true);
        if(scene.contacts().size()!=0)
            throw std::runtime_error(
                "Destroyed PCM actor retained published touching manager");
        const auto replacement=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,0.5f,0},1);
        scene.advance_collision(true);
        if(scene.contacts().size()!=1 ||
           !scene.island_edges_[scene.contacts()[0].edge_handle].
                pcm_cache.initialized ||
           std::abs(scene.contacts()[0].points[0].position.x)<1.0f)
            throw std::runtime_error(
                "Reused contact-manager handle retained the old PCM manifold");
        (void)replacement;
        (void)floor;

        // All eight corners are accepted under deep penetration,
        // then the original four-witness clustering must reduce the
        // point cache BEFORE Sc/PGS and contact reports consume it.
        Scene deep{Scene::ContactGenerator::DiagnosticLegacyGu};
        deep.set_pcm_plane_box(true);
        deep.add_static_floor(0);
        const auto buried=deep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,-0.6f,0},1);
        deep.advance_collision(true);
        if(deep.contacts().size()!=1 ||
           deep.contacts()[0].count!=4 ||
           deep.island_edges_[deep.contacts()[0].edge_handle].
               pcm_cache.count!=4)
            throw std::runtime_error(
                "PCM plane-box failed source eight-corner four-witness reduction");
        for(const auto& point:deep.contacts()[0].points)
            if(std::abs(point.position.y)>1.0e-6f ||
               point.normal.y!=1.0f ||
               point.separation>=0)
                throw std::runtime_error(
                    "PCM deep contacts must be projected onto the plane");
        // A shape contact-offset change must invalidate cached PCM
        // generation even if the relative transform did not move.
        const auto deep_edge=deep.contacts()[0].edge_handle;
        deep.set_shape_offsets(buried,0.2f,0.0f);
        if(deep.island_edges_[deep_edge].pcm_cache.initialized)
            throw std::runtime_error(
                "Authored PCM shape distance retained stale candidate set");
        deep.advance_collision(true);
        if(deep.contacts().size()!=1 ||
           deep.contacts()[0].count!=4 ||
           !deep.island_edges_[deep_edge].pcm_cache.initialized)
            throw std::runtime_error(
                "PCM edited shape distance did not regenerate manifold");
    }
    static void verify_legacy_gu_contact_cache() {
        // PxcCacheLocalContacts is independent from PCM and
        // DyFrictionCorrelation. A local-contact cache HIT moves raw
        // geometry by the midpoint of the two shape motions; the
        // original cached generating transforms are not advanced.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        const auto id=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.5f,0.0f},1.0f);
        scene.advance_collision(true);
        if(scene.gu_cache_records_.size()!=1 ||
           scene.gu_cache_records_[0].count!=4 ||
           scene.manifolds_.size()!=1)
            throw std::runtime_error(
                "Native legacy-Gu contact cache initial contact admission failed");
        // Raw Gu normals belong to plane-first geometry. The source
        // Sc contact manager instead orders the dynamic body first,
        // irrespective of which actor was admitted first.
        if(scene.gu_cached_points_[0].normal.y!=-1.0f ||
           scene.manifolds_[0].points[0].normal.y!=1.0f ||
           !scene.island_edges_[scene.manifolds_[0].edge_handle].gu_order_reversed)
            throw std::runtime_error(
                "Pxc floor-first Gu cache or source Sc body order mismatch");
        const float source_x=scene.manifolds_[0].points[0].position.x;
        scene.set_position(id,{0.004f,0.5f,0.0f});
        scene.advance_collision(true);
        if(scene.gu_cache_records_[0].pose1.position.x!=0.0f ||
           scene.gu_cache_records_[0].count!=4 ||
           std::abs(scene.manifolds_[0].points[0].position.x-
               (source_x+0.002f))>3.0e-5f)
            throw std::runtime_error(
                "Pxc legacy cache hit replaced original geometry or omitted midpoint update");
        scene.set_position(id,{0.020f,0.5f,0.0f});
        scene.advance_collision(true);
        if(scene.gu_cache_records_[0].pose1.position.x!=0.020f ||
           std::abs(scene.manifolds_[0].points[0].position.x-
               (source_x+0.020f))>3.0e-5f)
            throw std::runtime_error(
                "Pxc legacy cache did not regenerate contacts at source motion threshold");
        scene.set_position(id,{0.022f,0.5f,0.0f});
        scene.advance_collision(true);
        if(scene.gu_cache_records_[0].pose1.position.x!=0.020f ||
           std::abs(scene.manifolds_[0].points[0].position.x-
               (source_x+0.021f))>3.0e-5f)
            throw std::runtime_error(
                "Pxc legacy contact reference frame advanced incorrectly on repeated hit");
        // No-contact overlaps also own source cache records. The
        // infinite plane participates in BP even far above the floor.
        Scene separated{Scene::ContactGenerator::DiagnosticLegacyGu};
        separated.add_static_floor(0.0f);
        const auto distant=separated.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1);
        separated.advance_collision(true);
        if(separated.gu_cache_records_.size()!=1 ||
           separated.gu_cache_records_[0].count!=0)
            throw std::runtime_error(
                "Pxc legacy cache lost zero-contact BP overlap");
        separated.set_position(distant,{0,4.995f,0});
        separated.advance_collision(true);
        if(separated.gu_cache_records_[0].pose1.position.y!=5.0f ||
           separated.gu_cache_records_[0].count!=0)
            throw std::runtime_error(
                "Pxc zero-contact cache should retain original generating pose");

        // When the floor is created AFTER its dynamic partner,
        // PxcNpBatch sorts the Gu pair plane-first, caches its raw -Y
        // contact normal, and only then flips the published Sc contact
        // normal to +Y. A cached upward movement must INCREASE the
        // separation; caching the flipped normal reverses that sign.
        Scene late_floor{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto late_box=late_floor.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1);
        late_floor.add_static_floor(0.0f);
        late_floor.advance_collision(true);
        if(late_floor.gu_cached_points_.size()!=4 ||
           late_floor.manifolds_.size()!=1 ||
           late_floor.gu_cached_points_[0].normal.y!=-1.0f ||
           late_floor.manifolds_[0].points[0].normal.y!=1.0f ||
           late_floor.island_edges_[
               late_floor.manifolds_[0].edge_handle].gu_order_reversed)
            throw std::runtime_error(
                "Pxc plane-first cache or published Sc flip has incorrect normal");
        const float old_separation=late_floor.manifolds_[0].points[0].separation;
        late_floor.set_position(late_box,{0,0.504f,0});
        late_floor.advance_collision(true);
        if(late_floor.gu_cache_records_.size()!=1 ||
           late_floor.gu_cache_records_[0].pose1.position.y!=0.5f ||
           late_floor.gu_cached_points_[0].normal.y!=-1.0f ||
           late_floor.manifolds_[0].points[0].normal.y!=1.0f ||
           std::abs(late_floor.manifolds_[0].points[0].separation-
               (old_separation+0.004f))>2.0e-5f)
            throw std::runtime_error(
                "Pxc plane-second cache hit reversed relative-motion separation");

        // Box-box contacts use the source Sc actor-order frame, not a
        // separately ordered cache stream or canonical BodyId geometry.
        Scene stack{Scene::ContactGenerator::DiagnosticLegacyGu};
        stack.add_dynamic_box({0.5f,0.5f,0.5f},{0,0.5f,0},1);
        const auto upper=stack.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.10f,1.52f,0.10f},1);
        stack.advance_collision(true);
        if(stack.gu_cache_records_.size()!=1 ||
           stack.gu_cache_records_[0].count==0)
            throw std::runtime_error(
                "Pxc box/box cache did not capture speculative Gu contacts");
        const auto old=stack.gu_cache_records_[0];
        const auto old_p=stack.manifolds_[0].points[0].position;
        stack.set_position(upper,{0.103f,1.52f,0.1f});
        stack.advance_collision(true);
        if(stack.gu_cache_records_[0].pose0.position.x!=old.pose0.position.x ||
           std::abs(stack.manifolds_[0].points[0].position.x-
                    (old_p.x+0.0015f))>3.0e-5f)
            throw std::runtime_error(
                "Pxc source-oriented box/box contact cache midpoint mismatch");
    }
    static void verify_replaced_cached_friction_anchor() {
        // DyFrictionCorrelation::growPatches can replace anchor zero
        // while growing an existing one-anchor patch to two. Both
        // body-local frames must follow the actual selected contact IDs
        // or the friction Jacobians use different lever arms.
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        scene.step(dt);
        if(scene.manifolds_.size()!=1 ||
           scene.friction_patches_.size()!=1 ||
           scene.friction_patches_[0].anchor_count!=2)
            throw std::runtime_error(
                "Friction replacement fixture lacks the initial floor patch");
        auto& old=scene.friction_patches_[0];
        old.anchor_count=1;
        old.broken=false;
        old.contact_ids.fill(0xffff);
        // Source body0 is the dynamic box, not the canonical first
        // actor (floor). Seed a valid center anchor equidistant from
        // the four corners to require search replacement of slot 0.
        const auto before=scene.body_state(box);
        old.body0_anchors[0]={0,-0.5f,0};
        old.body1_anchors[0]={
            before.position.x,before.position.y-0.5f,
            before.position.z};
        scene.step(dt);
        if(scene.friction_patches_.size()!=1 ||
           scene.friction_patches_[0].anchor_count!=2 ||
           scene.manifolds_.size()!=1)
            throw std::runtime_error(
                "Cached one-anchor friction patch did not grow to two");
        const auto& patch=scene.friction_patches_[0];
        const auto& manifold=scene.manifolds_[0];
        if(patch.contact_ids[0]==0xffff ||
           patch.contact_ids[1]==0xffff)
            throw std::runtime_error(
                "Cached friction anchor search did not replace the retained slot");
        // DyContactPrep freezes its contact/anchor transforms BEFORE
        // integration. The externally published body_state here is
        // post-integration, so use the saved pre-solve transform when
        // reconstructing the body0 local anchor against Gu contacts.
        const auto& ground=scene.actors_[floor];
        const auto close=[](Vec3 a,Vec3 b) {
            const float dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;
            return dx*dx+dy*dy+dz*dz<2.5e-7f;
        };
        const auto rotate_anchor=[](Quat q,Vec3 v) {
            const Vec3 t{
                2.0f*(q.y*v.z-q.z*v.y),
                2.0f*(q.z*v.x-q.x*v.z),
                2.0f*(q.x*v.y-q.y*v.x)};
            return Vec3{
                v.x+q.w*t.x+q.y*t.z-q.z*t.y,
                v.y+q.w*t.y+q.z*t.x-q.x*t.z,
                v.z+q.w*t.z+q.x*t.y-q.y*t.x};
        };
        for(std::size_t k=0;k<2;++k) {
            const auto id=patch.contact_ids[k];
            if(id>=manifold.count)
                throw std::runtime_error(
                    "Cached friction anchor selected an invalid Gu contact");
            const auto& expected=manifold.points[id].position;
            const Vec3 local0=patch.body0_anchors[k];
            const Vec3 local1=patch.body1_anchors[k];
            const Vec3 r0=rotate_anchor(before.orientation,local0);
            const Vec3 r1=rotate_anchor(ground.orientation,local1);
            const Vec3 world0{
                before.position.x+r0.x,
                before.position.y+r0.y,
                before.position.z+r0.z};
            const Vec3 world1{
                ground.position.x+r1.x,
                ground.position.y+r1.y,
                ground.position.z+r1.z};
            if(!close(world0,expected) || !close(world1,expected))
                throw std::runtime_error(
                    "Selected cached friction anchor differs from its body-local frames");
        }
    }
    static void verify_source_retained_acceleration_lifecycle() {
        // ScBodySim::updateForces preserves per-second VelocityMod
        // entries only with eRETAIN_ACCELERATIONS, while each per-step
        // velocity change is always consumed exactly once.
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto body=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.linear_damping=0.0f;
        dynamics.angular_damping=0.0f;
        dynamics.retain_accelerations=true;
        scene.set_dynamics(body,dynamics);
        scene.add_spatial_velocity_mod(body,{1.5f,0,0},
                                       ForceMode::Acceleration,false);
        scene.add_spatial_velocity_mod(body,{0,0,0.75f},
                                       ForceMode::Acceleration,true);
        scene.add_spatial_velocity_mod(body,{0.2f,0,0},
                                       ForceMode::VelocityChange,false);
        for(unsigned i=1;i<=5;++i) {
            scene.step(dt);
            const auto state=scene.body_state(body);
            if(std::abs(state.velocity.x-(0.2f+i*1.5f*dt))>1e-5f ||
               std::abs(state.angular_velocity.z-i*0.75f*dt)>1e-5f ||
               scene.actors_[body].pending_linear_acceleration.x!=1.5f ||
               scene.actors_[body].pending_angular_acceleration.z!=0.75f ||
               scene.actors_[body].pending_linear_velocity_delta.x!=0.0f)
                throw std::runtime_error(
                    "Retained PhysX per-second acceleration or one-shot velocity mod diverged");
        }
        scene.clear_spatial_velocity_mod(body,ForceMode::Acceleration,false);
        scene.clear_spatial_velocity_mod(body,ForceMode::Acceleration,true);
        const auto previous=scene.body_state(body);
        scene.step(dt);
        const auto final=scene.body_state(body);
        if(std::abs(final.velocity.x-previous.velocity.x)>1e-6f ||
           std::abs(final.angular_velocity.z-previous.angular_velocity.z)>1e-6f)
            throw std::runtime_error(
                "Cleared retained PhysX acceleration continued after the next step");

        Scene default_scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto once=default_scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        auto no_retention=dynamics;
        no_retention.retain_accelerations=false;
        default_scene.set_dynamics(once,no_retention);
        default_scene.add_spatial_velocity_mod(once,{2.0f,0,0},
                                               ForceMode::Acceleration,false);
        default_scene.step(dt);
        default_scene.step(dt);
        if(std::abs(default_scene.body_state(once).velocity.x-
                    2.0f*dt)>1e-6f)
            throw std::runtime_error(
                "Default source non-retaining acceleration was applied twice");
    }
    static void verify_after_integration_cached_shape_bounds() {
        // ScPipeline::afterIntegration::updateScBodyAndShapeSim publishes
        // each integrated active body's pose-derived bounds before the
        // frame completes. The next SAP update must consume those bounds
        // rather than recomputing a stale pre-integration transform.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        scene.advance_collision(true);
        const auto before=scene.actors_[box].encoded_bounds;
        if(!scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "Sc initial Gu bounds were not cached for rigid actor");
        scene.step(1.0f/60.0f);
        const auto integrated=scene.actors_[box].encoded_bounds;
        if(!scene.actors_[box].cached_bounds_current ||
           integrated==before ||
           scene.solved_result_frame()!=scene.frame_index())
            throw std::runtime_error(
                "Sc afterIntegration did not publish integrated shape bounds");
        scene.advance_collision(true);
        if(scene.actors_[box].encoded_bounds!=integrated ||
           !scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "SAP recomputed or changed an already published integrated AABB");
        // Explicit shape pose and contact-offset edits invalidate the
        // cache, without requiring a separate shape/bounds owner.
        scene.set_position(box,{0,8,0});
        if(scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "Authored shape pose failed to invalidate Sc cached bounds");
        scene.advance_collision(true);
        const auto moved=scene.actors_[box].encoded_bounds;
        if(moved==integrated ||
           !scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "Gu broadphase ignored an authored shape relocation");
        scene.set_shape_offsets(box,0.15f,0.0f);
        if(scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "Authored contact offset failed to invalidate Sc shape bounds");
        scene.advance_collision(true);
        if(scene.actors_[box].encoded_bounds==moved ||
           !scene.actors_[box].cached_bounds_current)
            throw std::runtime_error(
                "Gu cached bounds omitted the new shape contact offset");
    }
    static void verify_source_sleep_velocity_mod_retirement() {
        // ScBodySim::deactivate calls setForcesToDefaults and
        // putToSleep clears the complete VelocityMod. A pending
        // per-step or per-second contribution cannot survive the
        // island's authoritative sleep-to-wake boundary.
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto id=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        scene.set_dynamics(id,dynamics);
        bool ready=false;
        for(unsigned frame=0;frame<90;++frame) {
            scene.step(dt);
            if(scene.body_ready_for_sleep(id) &&
               !scene.body_is_sleeping(id)) {
                ready=true;
                break;
            }
        }
        if(!ready)
            throw std::runtime_error(
                "Source sleep velocity-mod fixture never reached Dy readiness");
        // Arrange retained stale writes at the moment the previously
        // published Dy readiness is consumed by the NEXT island pass.
        // This probes the deactivation cleanup directly; public
        // authored force calls correctly clear readiness themselves.
        auto& actor=scene.actors_[id];
        actor.pending_linear_velocity_delta={1.0f,0,0};
        actor.pending_angular_velocity_delta={0,0.7f,0};
        actor.pending_linear_acceleration={0,2.0f,0};
        actor.pending_angular_acceleration={0,0,3.0f};
        scene.step(dt);
        if(!scene.body_is_sleeping(id) ||
           actor.pending_linear_velocity_delta.x!=0 ||
           actor.pending_angular_velocity_delta.y!=0 ||
           actor.pending_linear_acceleration.y!=0 ||
           actor.pending_angular_acceleration.z!=0)
            throw std::runtime_error(
                "Sc island deactivation retained source velocity modifications");
        scene.set_velocity(id,{0,0,0});
        scene.step(dt);
        const auto state=scene.body_state(id);
        if(state.sleeping ||
           std::abs(state.velocity.x)>1e-6f ||
           std::abs(state.velocity.y)>1e-6f ||
           std::abs(state.angular_velocity.y)>1e-6f ||
           std::abs(state.angular_velocity.z)>1e-6f)
            throw std::runtime_error(
                "Source awakened rigid body replayed sleeping VelocityMod");
    }
    static void verify_source_post_solver_continuations() {
        // ScPipeline::updateDynamics wires processLostContactsTask3
        // and all PxsSolverEndTask chains to mPostSolver. No handle
        // retirement/result publication may pass that join while an
        // island solver chain is still integrating its rigid bodies.
        Scene::RigidStepJoin join;
        const auto rejected=[&](auto&& attempt) {
            try { attempt(); }
            catch(const std::logic_error&) { return true; }
            return false;
        };
        if(!rejected([&]{ join.require_post_solver(); }) ||
           !rejected([&]{ join.finish_solver_task(0); }) ||
           !rejected([&]{ join.finish_third_pass(); }) ||
           !rejected([&]{ join.finish_lost_contacts(); }) ||
           !rejected([&]{ join.finish_post_solver(); }) ||
           !rejected([&]{ join.finish_after_integration(); }) ||
           !rejected([&]{ join.finish_finalization(); }) ||
           !rejected([&]{ join.publish_results(); }))
            throw std::runtime_error(
                "ScPipeline postSolver accepted an uncompleted producer");
        // Original PhysX runs the two producer branches independently.
        // Dispatch and finish one Dy chain BEFORE the lost-touch branch.
        std::array<std::uint32_t,2> completed{};
        join.dispatch_solver_tasks(completed,1);
        // Completing one island chain twice cannot substitute for
        // the missing completion of another active island chain.
        join.finish_solver_task(1);
        if(!rejected([&]{ join.finish_solver_task(1); }) ||
           !rejected([&]{ join.require_post_solver(); }))
            throw std::runtime_error(
                "ScPipeline postSolver retired an incomplete lost-touch branch");
        join.finish_speculative_third_pass();
        if(!rejected([&]{ join.finish_third_pass(); }) ||
           !rejected([&]{ join.finish_lost_contacts(); }) ||
           !rejected([&]{ join.finish_speculative_third_pass(); }))
            throw std::runtime_error(
                "ScPipeline postThirdPassTask ran before accurate IslandSim completion");
        join.finish_accurate_third_pass();
        join.finish_third_pass();
        join.finish_lost_contacts();
        if(!rejected([&]{ join.require_post_solver(); }) ||
           !rejected([&]{ join.finish_third_pass(); }) ||
           !rejected([&]{ join.finish_accurate_third_pass(); }))
            throw std::runtime_error(
                "ScPipeline postSolver omitted the final Dy task completion");
        join.finish_solver_task(0);
        join.finish_post_solver();
        join.finish_after_integration();
        if(!rejected([&]{ join.publish_results(); }))
            throw std::runtime_error(
                "ScPipeline published results before finalization");
        join.finish_finalization();
        join.publish_results();
        if(!rejected([&]{ join.finish_finalization(); }) ||
           !rejected([&]{ join.finish_after_integration(); }) ||
           !rejected([&]{ join.publish_results(); }) ||
           !rejected([&]{ join.finish_post_solver(); }) ||
           !rejected([&]{ join.finish_lost_contacts(); }) ||
           !rejected([&]{ join.finish_solver_task(0); }) ||
           !rejected([&]{
               std::array<std::uint32_t,1> duplicate{};
               join.dispatch_solver_tasks(duplicate,2);
           }))
            throw std::runtime_error(
                "ScPipeline postSolver accepted duplicate task completion");
        // Source permits zero active solver task chains in an empty
        // scene, but still requires lost-contact completion and Dy
        // task dispatch before publishing a finished frame.
        Scene::RigidStepJoin empty;
        std::array<std::uint32_t,0> no_tasks{};
        empty.dispatch_solver_tasks(no_tasks,1);
        if(!rejected([&]{ empty.finish_solver_task(0); }) ||
           !rejected([&]{ empty.finish_lost_contacts(); }) ||
           !rejected([&]{ empty.finish_post_solver(); }))
            throw std::runtime_error(
                "Empty source task stream bypassed third-pass completion");
        // Independent island completions may arrive in the opposite
        // order, even when no rigid solver tasks are admitted.
        empty.finish_accurate_third_pass();
        if(!rejected([&]{ empty.finish_third_pass(); }))
            throw std::runtime_error(
                "Empty source task stream skipped speculative IslandSim");
        empty.finish_speculative_third_pass();
        empty.finish_third_pass();
        empty.finish_lost_contacts();
        empty.finish_post_solver();
        empty.finish_after_integration();
        empty.finish_finalization();
        empty.publish_results();

        // Distinct real Dy end-task completions can arrive from two
        // execution lanes concurrently, not merely in reverse serial
        // order. The shared outstanding producer edge must account for
        // every task exactly once before postSolver can publish.
        {
            Scene::RigidStepJoin concurrent_join;
            std::array<std::uint32_t,64> generations{};
            concurrent_join.dispatch_solver_tasks(generations,17);
            concurrent_join.finish_accurate_third_pass();
            concurrent_join.finish_speculative_third_pass();
            concurrent_join.finish_third_pass();
            concurrent_join.finish_lost_contacts();
            std::jthread worker([&] {
                for(std::size_t i=1;i<generations.size();i+=2)
                    concurrent_join.finish_solver_task(i);
            });
            for(std::size_t i=0;i<generations.size();i+=2)
                concurrent_join.finish_solver_task(i);
            worker.join();
            if(concurrent_join.outstanding_solver_tasks!=0 ||
               !std::all_of(generations.begin(),generations.end(),
                   [](std::uint32_t v){return v==17;}))
                throw std::runtime_error(
                    "Concurrent PxsSolverEndTask producers lost completion edges");
            concurrent_join.finish_post_solver();
            concurrent_join.finish_after_integration();
            concurrent_join.finish_finalization();
            concurrent_join.publish_results();
        }
        // Real Scene task IDs remain safe across uint32 stamp rollover;
        // a completed old generation cannot mask a current completion.
        {
            Scene rollover{Scene::ContactGenerator::DiagnosticLegacyGu};
            rollover.add_dynamic_box({0.5f,0.5f,0.5f},{0,8,0},1.0f);
            rollover.solver_task_completion_generation_=
                std::numeric_limits<std::uint32_t>::max();
            rollover.solver_task_completion_stamps_.assign(1,1);
            rollover.step(1.0f/120.0f);
            if(rollover.solver_task_completion_generation_!=1 ||
               rollover.solver_tasks_.size()!=1 ||
               rollover.solver_task_completion_stamps_[0]!=1 ||
               rollover.solved_result_frame()!=rollover.frame_index())
                throw std::runtime_error(
                    "PxsSolverEndTask generation rollover blocked source publication");
        }

        // Production-path test: enhanced determinism prevents merging
        // three independently owned active islands into one Dy task.
        // Two islands have floor contact rows, and a third free-flight
        // island has no constraints but STILL requires its own solver
        // task completion after body integration.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_enhanced_determinism(true);
        scene.add_static_floor(0.0f);
        const auto left=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-4,0.5f,0},1.0f);
        const auto right=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,0.5f,0},1.0f);
        const auto free=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        scene.step(1.0f/120.0f);
        if(scene.solver_tasks_.size()!=3 ||
           scene.solved_result_frame()!=scene.frame_index() ||
           scene.contacts().size()!=2 ||
           !(scene.body_state(free).position.y<10.0f) ||
           !std::isfinite(scene.body_state(left).velocity.y) ||
           !std::isfinite(scene.body_state(right).velocity.y))
            throw std::runtime_error(
                "Real Dy solver task groups did not join after all island integration");
        for(const auto& task:scene.solver_tasks_)
            if(task.batch_end-task.first_batch!=1)
                throw std::runtime_error(
                    "Source enhanced-determinism task range crossed island ownership");
        scene.remove(right);
        scene.step(1.0f/120.0f);
        if(scene.solved_result_frame()!=scene.frame_index() ||
           scene.contacts().size()!=1 ||
           scene.solver_tasks_.size()!=2 ||
           !std::isfinite(scene.body_state(free).position.y))
            throw std::runtime_error(
                "Sc lost-contact retirement or Dy task completion leaked across frames");
    }
    static void verify_enhanced_determinism_tasks() {
        // Original DyDynamicsContext::updatePostKinematic stops island
        // accumulation after EACH active island under
        // PxSceneFlag::eENABLE_ENHANCED_DETERMINISM. The default groups
        // these four islands until the 128-body threshold. Independently
        // authored solver iteration counts must never leak across
        // isolated enhanced-determinism task chains.
        const auto prepare=[](bool enhanced) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            scene.set_enhanced_determinism(enhanced);
            scene.add_static_floor(0.0f);
            const std::array<std::pair<std::uint8_t,std::uint8_t>,4> counts{{
                {4,1},{9,3},{2,0},{6,2}}};
            for(std::size_t i=0;i<counts.size();++i) {
                const auto id=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},
                    {static_cast<float>(i)*3.0f,i==3?8.0f:0.5f,0},1.0f);
                RigidBodyDynamics dynamics{};
                dynamics.position_iterations=counts[i].first;
                dynamics.velocity_iterations=counts[i].second;
                scene.set_dynamics(id,dynamics);
            }
            scene.step(1.0f/120.0f);
            if(scene.indexed_managers_.size()!=3 ||
               scene.solver_tasks_.size()!=(enhanced?4u:1u) ||
               scene.solver_partition_count()!=(enhanced?3u:1u))
                throw std::runtime_error(
                    "Enhanced PhysX accurate-island task grouping mismatched source");
            if(!enhanced) {
                const auto& task=scene.solver_tasks_.front();
                if(task.position_iterations!=9 ||
                   task.velocity_iterations!=3 ||
                   task.batch_end!=4 ||
                   task.manager_end!=3)
                    throw std::runtime_error(
                        "Default Dy task failed to merge island iteration maxima");
            } else {
                std::array<std::pair<std::uint8_t,std::uint8_t>,4> actual{};
                std::size_t previous_managers=0;
                std::size_t zero_contact_tasks=0;
                for(std::size_t i=0;i<actual.size();++i) {
                    const auto& task=scene.solver_tasks_[i];
                    if(task.batch_end!=i+1 || task.first_batch!=i)
                        throw std::runtime_error(
                            "Enhanced Dy task chain crossed source island boundary");
                    actual[i]={task.position_iterations,task.velocity_iterations};
                    if(task.manager_end==previous_managers)
                        ++zero_contact_tasks;
                    previous_managers=task.manager_end;
                }
                auto expected=counts;
                std::sort(expected.begin(),expected.end());
                std::sort(actual.begin(),actual.end());
                if(actual!=expected || zero_contact_tasks!=1)
                    throw std::runtime_error(
                        "Enhanced Dy task iterations or no-contact integration lost");
            }
            // Task-local solve changes no public Gu publication or
            // per-contact writeback layout across the source flag.
            const auto normals=scene.solved_normal_impulses();
            for(const auto& manifold:scene.contacts()) {
                double total=0.0;
                for(std::size_t k=0;k<manifold.count;++k)
                    total+=normals[manifold.contact_offset+k];
                if(!std::isfinite(total) || total<=0.0)
                    throw std::runtime_error(
                        "Enhanced Dy group dropped its solved Gu contact forces");
            }
        };
        prepare(false);
        prepare(true);
    }
    static void verify() {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        scene.add_dynamic_box({0.5f,0.5f,0.5f},
                              {0.0f,0.49f,0.0f},1.0f);
        constexpr float dt=1.0f/120.0f;
        scene.step(dt);
        if(scene.manifolds_.size()!=1 ||
           scene.manifolds_[0].count<4)
            throw std::runtime_error(
                "Source header probe lacks four live Gu floor contacts");
        auto& manifold=scene.manifolds_[0];
        const std::size_t offset=manifold.contact_offset;
        // A and C are within PXC_SAME_NORMAL (0.999), while B is
        // distinct. createContactPatches yields A, B, C; the source
        // correlatePatches prepends C ahead of A in patch 0.
        const Vec3 a{0.0f,-1.0f,0.0f};
        const Vec3 b{1.0f,0.0f,0.0f};
        const Vec3 c{0.02f,-std::sqrt(1.0f-0.0004f),0.0f};
        scene.contact_points_[offset].normal=a;
        scene.contact_points_[offset+1].normal=b;
        for(std::size_t j=2;j<manifold.count;++j)
            scene.contact_points_[offset+j].normal=c;
        // Cached anchor data from the preceding frame is irrelevant
        // to this fixture's uncached C-to-A patch chain.
        scene.friction_patches_.clear();
        scene.prepare_constraints(dt);
        if(manifold.friction_patch_count!=2 ||
           scene.friction_patches_.size()!=2 ||
           scene.solver_contact_order_.size()!=manifold.count ||
           scene.solver_contact_order_[0]!=offset+2 ||
           scene.solver_contact_order_[1]!=offset+3 ||
           scene.solver_contact_order_[2]!=offset ||
           scene.solver_contact_order_[3]!=offset+1)
            throw std::runtime_error(
                "Source header probe lost reverse linked contact-patch order");
        const auto close=[](Vec3 x,Vec3 y) {
            const float dx=x.x-y.x,dy=x.y-y.y,dz=x.z-y.z;
            return dx*dx+dy*dy+dz*dz<1.0e-8f;
        };
        for(std::size_t j=0;j<manifold.count;++j) {
            const Vec3 expected=j==1?b:c;
            if(!close(scene.solver_contact_points_[
                          manifold.solver_offset+j].normal,expected))
                throw std::runtime_error(
                    "Correlated normal constraint ignored source header normal");
        }
        // Contact 0 (normal A) must nevertheless use the later
        // correlation head C; test the actual friction basis produced
        // from that same head, not from FrictionPatch's cached A.
        const auto& patch=scene.friction_patches_[0];
        if(patch.anchor_count==0)
            throw std::runtime_error(
                "Source header probe lacks active friction constraints");
        for(std::size_t i=0;i<patch.anchor_count*2;++i) {
            const Vec3 tangent=scene.solver_friction_rows_[
                patch.row_offset+i].tangent;
            const float projection=tangent.x*c.x+tangent.y*c.y+
                                   tangent.z*c.z;
            if(!std::isfinite(projection) ||
               std::abs(projection)>1.0e-4f)
                throw std::runtime_error(
                    "Correlated friction tangent not orthogonal to header normal");
        }
        // Reprepare with the first pass's unbroken local anchors. The
        // source retains their original cached frame normal A while
        // constructing the CURRENT solver header and tangents from C.
        const Vec3 original_cached=patch.body0_normal;
        scene.prepare_constraints(dt);
        if(!close(scene.friction_patches_[0].body0_normal,
                  original_cached) ||
           !close(scene.solver_contact_points_[
                      manifold.solver_offset].normal,c))
            throw std::runtime_error(
                "Source cached friction normal overwrote the current header");
        const auto& reused=scene.friction_patches_[0];
        if(reused.anchor_count==0)
            throw std::runtime_error(
                "Source cached header probe lost valid strong-friction anchors");
        for(std::size_t i=0;i<reused.anchor_count*2;++i) {
            const Vec3 tangent=scene.solver_friction_rows_[
                reused.row_offset+i].tangent;
            const float projection=tangent.x*c.x+tangent.y*c.y+
                                   tangent.z*c.z;
            if(!std::isfinite(projection) ||
               std::abs(projection)>1.0e-4f)
                throw std::runtime_error(
                    "Reused friction anchors used stale cached normal for tangent");
        }
    }
    static void verify_post_solver_manager_retirement() {
        // ScPipeline::destroyManagers schedules the accurate and
        // speculative third pass, but Pxs contact-manager allocations
        // are not reusable until the lost-contact continuation has
        // joined their solver consumer. A serial CPU lowering must
        // preserve that ownership boundary without a fake task queue.
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        const Pair old_pair{a,b};
        scene.advance_collision();
        const auto original=scene.contact_edge_index(old_pair);
        if(original==Scene::invalid_edge)
            throw std::runtime_error(
                "Post-solver retirement fixture failed to admit old overlap");
        scene.set_position(b,{6,8,0});
        scene.step_checkpoint_.capture(scene);
        scene.advance_collision_internal(false,true);
        scene.third_pass_island_gen();
        if(scene.retired_edge_handles_.size()!=1 ||
           scene.retired_edge_handles_[0]!=original ||
           original>=scene.island_edges_.size() ||
           !scene.island_edges_[original].retiring ||
           std::find(scene.free_edge_handles_.begin(),
                     scene.free_edge_handles_.end(),original)!=
               scene.free_edge_handles_.end())
            throw std::runtime_error(
                "Lost manager handle was recycled before source solver/lost-contact join");
        // Emulate completion of both consumers, then verify the exact
        // same retirement operation used by Scene::step. A rejected
        // frame must reverse handle retirement and intrusive history
        // without introducing a second contact-manager owner.
        scene.retire_destroyed_contact_edges();
        if(!scene.retired_edge_handles_.empty() ||
           scene.free_edge_handles_.empty() ||
           scene.free_edge_handles_.back()!=original ||
           scene.island_edges_[original].live)
            throw std::runtime_error(
                "Source post-third-pass manager retirement did not publish");
        scene.step_checkpoint_.rollback(scene);
        if(scene.contact_edge_index(old_pair)!=original ||
           std::find(scene.free_edge_handles_.begin(),
                     scene.free_edge_handles_.end(),original)!=
               scene.free_edge_handles_.end())
            throw std::runtime_error(
                "Rejected post-solver manager retirement lost published handle identity");
        // The complete production solve retires only after Dy consumes
        // this frame; the next collision may then recycle the handle.
        scene.step(dt);
        if(scene.contact_edge_index(old_pair)!=Scene::invalid_edge ||
           scene.free_edge_handles_.empty() ||
           scene.free_edge_handles_.back()!=original)
            throw std::runtime_error(
                "Completed source PGS frame retained a destroyed contact manager");
        const auto c=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        scene.advance_collision();
        if(scene.contact_edge_index(Pair{a,c})!=original)
            throw std::runtime_error(
                "Completed post-solver retirement failed to recycle source manager handle");
        // A collision-only transaction has no Dy consumer. It still
        // completes both island passes and retires before publishing.
        Scene query{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto x=query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto y=query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        query.advance_collision();
        const auto retired=query.contact_edge_index(Pair{x,y});
        query.set_position(y,{6,8,0});
        query.advance_collision();
        if(retired==Scene::invalid_edge ||
           query.free_edge_handles_.empty() ||
           query.free_edge_handles_.back()!=retired ||
           !query.retired_edge_handles_.empty())
            throw std::runtime_error(
                "Collision-only island join leaked a destroyed source manager");
    }
    static void verify_post_solver_lost_report_join() {
        // Exercise the single production frame joining original
        // Sc lost-touch events, Dy force-threshold loss, island-edge
        // retirement and completed-frame publication. The lost manager
        // must not outlive a successful step or leak its prior force
        // into a later report.
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_contact_notifications(true);
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        const Pair pair{a,b};
        scene.set_contact_report_threshold(a,0.0f);
        scene.set_contact_report_threshold(b,0.0f);
        scene.set_velocity(b,{-2,0,0});
        scene.step(dt);
        const auto handle=scene.contact_edge_index(pair);
        if(handle==Scene::invalid_edge ||
           scene.contact_events().size()!=1 ||
           scene.contact_events()[0].kind!=ContactEventKind::Found ||
           scene.force_threshold_changes().size()!=1 ||
           scene.force_threshold_changes()[0].change!=
               ForceThresholdChange::Found)
            throw std::runtime_error(
                "Source completion join fixture lacks touching force/report history");
        scene.set_position(b,{6,8,0});
        const auto published_frame=scene.frame_index();
        scene.step(dt);
        if(scene.frame_index()!=published_frame+1 ||
           scene.contact_edge_index(pair)!=Scene::invalid_edge ||
           scene.contact_events().size()!=1 ||
           scene.contact_events()[0].kind!=ContactEventKind::Lost ||
           scene.contact_events()[0].normal_impulse!=0.0 ||
           scene.force_threshold_changes().size()!=1 ||
           scene.force_threshold_changes()[0].change!=
               ForceThresholdChange::Lost ||
           std::find(scene.free_edge_handles_.begin(),
                     scene.free_edge_handles_.end(),handle)==
               scene.free_edge_handles_.end())
            throw std::runtime_error(
                "Source completion join lost a report or retained its manager handle");
        scene.step(dt);
        if(!scene.contact_events().empty() ||
           !scene.force_threshold_changes().empty())
            throw std::runtime_error(
                "Completed lost-contact reports replayed on the next frame");
    }
    static void verify_reversible_cache_checkpoint() {
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        scene.set_contact_report_threshold(box,0.01f);
        for(unsigned i=0;i<8;++i)scene.step(dt);
        if(scene.friction_patches_.empty() ||
           scene.threshold_previous_.empty())
            throw std::runtime_error(
                "Source cache fixture has no persistent contact histories");
        // A quiet Gu publication journals the public solver offsets as
        // well, so this direct transaction exercises the exact rollback
        // boundary without leaving modified manifold metadata behind.
        const std::vector<double> preceding_forces(
            scene.solved_normal_impulses().begin(),
            scene.solved_normal_impulses().end());
        scene.advance_collision();
        const auto old_patches=scene.friction_patches_;
        const auto old_spans=scene.friction_by_edge_;
        const auto old_prepared_epoch=scene.manifolds_[0].prepared_friction_epoch;
        const auto old_threshold=scene.threshold_previous_;
        const auto* const patches_ptr=scene.friction_patches_.data();
        const auto* const spans_ptr=scene.friction_by_edge_.data();
        const auto* const threshold_ptr=scene.threshold_previous_.data();
        const auto old_friction_epoch=scene.friction_cache_epoch_;
        const auto old_threshold_epoch=scene.threshold_cache_epoch_;
        const auto old_frame=scene.frame_index();
        scene.step_checkpoint_.capture(scene);
        // The capture moves the previous indexed-manager and active-batch
        // arrays into undo storage. Recreate this frame's REAL source
        // task input before probing the ordinary correlation/force
        // stages; otherwise the test would correlate zero managers.
        scene.prepare_active_accurate_island_batches();
        scene.prepare_solver_bodies();
        scene.partition_constraints();
        if(scene.indexed_managers_.size()!=1)
            throw std::runtime_error(
                "Quiet cache fixture did not rebuild the active PhysX manager");
        scene.prepare_constraints(dt);
        if(scene.step_checkpoint_.quiet_manifold_metadata.size()!=1 ||
           scene.step_checkpoint_.quiet_manifold_metadata[0].index!=
               scene.indexed_managers_[0])
            throw std::runtime_error(
                "Quiet rollback failed to journal the single mutated Dy manager");
        // A collision-only publication deliberately clears solver
        // writeback. Feed the actual preceding solved impulses into
        // this isolated threshold-stage transaction.
        scene.normal_impulse_writeback_.assign(
            preceding_forces.begin(),preceding_forces.end());
        scene.update_threshold_stream(dt);
        if(scene.friction_cache_epoch_!=old_friction_epoch+1 ||
           scene.threshold_cache_epoch_!=old_threshold_epoch+1)
            throw std::runtime_error(
                "Production contact/force streams were not rotated");
        scene.step_checkpoint_.rollback(scene);
        if(scene.friction_cache_epoch_!=old_friction_epoch ||
           scene.threshold_cache_epoch_!=old_threshold_epoch ||
           scene.friction_patches_.data()!=patches_ptr ||
           scene.friction_by_edge_.data()!=spans_ptr ||
           scene.friction_by_edge_.size()!=old_spans.size() ||
           scene.manifolds_[0].prepared_friction_epoch!=old_prepared_epoch ||
           scene.threshold_previous_.data()!=threshold_ptr ||
           scene.friction_patches_.size()!=old_patches.size() ||
           scene.threshold_previous_.size()!=old_threshold.size() ||
           scene.frame_index()!=old_frame)
            throw std::runtime_error(
                "Rejected contact/force cache stream lost old allocation");
        for(std::size_t i=0;i<old_spans.size();++i) {
            const auto& a=old_spans[i];
            const auto& b=scene.friction_by_edge_[i];
            if(a.pair!=b.pair || a.offset!=b.offset ||
               a.count!=b.count || a.epoch!=b.epoch)
                throw std::runtime_error(
                    "Rejected source friction index changed manager identity");
        }
        for(std::size_t i=0;i<old_patches.size();++i) {
            const auto& a=old_patches[i];
            const auto& b=scene.friction_patches_[i];
            if(a.pair!=b.pair || a.anchor_count!=b.anchor_count ||
               a.static_friction!=b.static_friction ||
               a.dynamic_friction!=b.dynamic_friction ||
               a.broken!=b.broken)
                throw std::runtime_error(
                    "Rejected correlation changed cached friction patch");
            for(std::size_t j=0;j<a.anchor_count;++j)
                if(a.body0_anchors[j].x!=b.body0_anchors[j].x ||
                   a.body0_anchors[j].y!=b.body0_anchors[j].y ||
                   a.body0_anchors[j].z!=b.body0_anchors[j].z ||
                   a.body1_anchors[j].x!=b.body1_anchors[j].x ||
                   a.body1_anchors[j].y!=b.body1_anchors[j].y ||
                   a.body1_anchors[j].z!=b.body1_anchors[j].z)
                    throw std::runtime_error(
                        "Rejected correlation changed cached friction anchors");
        }
        for(std::size_t i=0;i<old_threshold.size();++i)
            if(scene.threshold_previous_[i].key!=old_threshold[i].key ||
               scene.threshold_previous_[i].impulse!=old_threshold[i].impulse ||
               scene.threshold_previous_[i].threshold!=old_threshold[i].threshold)
                throw std::runtime_error(
                    "Rejected force stream changed prior exceeded pairs");
        scene.step(dt);
        if(scene.friction_patches_.empty() ||
           scene.friction_cache_epoch_!=old_friction_epoch+1 ||
           scene.threshold_cache_epoch_!=old_threshold_epoch+1)
            throw std::runtime_error(
                "Retried native PGS did not restore cached correlation");
    }
    static void verify_source_gu_manager_lookup_lifecycle() {
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        const Pair pair{floor,box};
        scene.advance_collision();
        const auto handle=scene.contact_edge_index(pair);
        if(handle==Scene::invalid_edge || scene.manifolds_.size()!=1 ||
           scene.manifold_by_edge_.size()<=handle ||
           scene.manifold_by_edge_[handle].index!=0 ||
           scene.manifold_by_edge_[handle].generation!=
               scene.manifold_lookup_generation_)
            throw std::runtime_error(
                "Gu source contact publication did not index live manager");
        const auto* const original_mapping=scene.manifold_by_edge_.data();
        const auto published_generation=scene.manifold_lookup_generation_;
        scene.advance_collision();
        if(scene.manifold_by_edge_.data()!=original_mapping ||
           scene.manifold_lookup_generation_!=published_generation)
            throw std::runtime_error(
                "Quiet source contact publication unnecessarily rebuilt Gu index");
        scene.step(dt);
        if(scene.manifold_lookup_generation_!=published_generation ||
           scene.indexed_managers_.size()!=1 ||
           scene.indexed_managers_[0]!=0)
            throw std::runtime_error(
                "Quiet PGS did not reuse previously published Gu manager lookup");
        // Loss of physical touch keeps the speculative infinite-plane
        // manager handle alive, but its old Gu index must be rejected.
        scene.set_position(box,{0,5,0});
        scene.advance_collision();
        if(scene.contact_edge_index(pair)!=handle ||
           scene.contact_edge_connected(pair) ||
           !scene.manifolds_.empty() ||
           scene.manifold_by_edge_[handle].generation==
               scene.manifold_lookup_generation_)
            throw std::runtime_error(
                "Gu index retained a stale manager after accurate touch loss");
        scene.set_position(box,{0,0.49f,0});
        scene.advance_collision();
        if(scene.contact_edge_index(pair)!=handle ||
           !scene.contact_edge_connected(pair) ||
           scene.manifolds_.size()!=1 ||
           scene.manifold_by_edge_[handle].index!=0 ||
           scene.manifold_by_edge_[handle].generation!=
               scene.manifold_lookup_generation_)
            throw std::runtime_error(
                "Gu touch reentry did not rebind its original EdgeIndex");

        // Stage a new dirty collision/Gu publication, then fail during
        // real Dy material combination. Failure must rotate the prior
        // Gu arena and its manager index back together (same pointers).
        const auto* const retained_map=scene.manifold_by_edge_.data();
        const auto* const retained_manifold=scene.manifolds_.data();
        const auto retained_epoch=scene.contact_epoch_;
        const auto retained_generation=scene.manifold_lookup_generation_;
        Material overflow{};
        overflow.static_friction=1.0e30f;
        overflow.dynamic_friction=1.0e30f;
        overflow.friction_combine=MaterialCombineMode::Multiply;
        scene.set_material(floor,overflow);
        scene.set_material(box,overflow);
        scene.set_position(box,{0,0.48f,0});
        bool failed=false;
        try {scene.step(dt);}
        catch(const std::overflow_error&) {failed=true;}
        if(!failed || scene.contact_epoch_!=retained_epoch ||
           scene.manifold_lookup_generation_!=retained_generation ||
           scene.manifold_by_edge_.data()!=retained_map ||
           scene.manifolds_.data()!=retained_manifold ||
           scene.manifold_by_edge_[handle].generation!=retained_generation)
            throw std::runtime_error(
                "Rejected PGS frame lost published Gu manager lookup ownership");
        scene.set_material(floor,Material{});
        scene.set_material(box,Material{});
        scene.step(dt);
        if(scene.indexed_managers_.size()!=1 ||
           scene.indexed_managers_[0]!=0 ||
           scene.manifold_by_edge_[handle].generation!=
               scene.manifold_lookup_generation_ ||
           scene.manifold_lookup_generation_==retained_generation)
            throw std::runtime_error(
                "Retried source Gu publication did not rebuild usable active mapping");
    }
    static void verify_source_friction_task_order() {
        // Source island contact-manager order C,B,D differs from Gu's
        // canonical B,C,D. Friction patch preparation, retained anchors
        // and per-manager source handles must follow the former; Gu
        // force writeback must retain its independent original offsets.
        constexpr float dt=1.0f/480.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto center=scene.add_dynamic_box(
            {1,1,1},{0,10,0},1.0f);
        const auto b=scene.add_static_box(
            {0.25f,0.25f,0.25f},{4,10,0});
        const auto c=scene.add_static_box(
            {0.25f,0.25f,0.25f},{-1.1f,10,0});
        const auto d=scene.add_static_box(
            {0.25f,0.25f,0.25f},{0,10,4});
        scene.advance_collision();
        scene.set_position(b,{1.1f,10,0});
        scene.set_position(d,{0,10,1.1f});
        scene.step(dt);
        const std::array<Pair,3> expected{
            Pair{center,c},Pair{center,b},Pair{center,d}};
        if(scene.indexed_managers_.size()!=expected.size() ||
           scene.friction_patches_.size()<expected.size())
            throw std::runtime_error(
                "Source friction task-order fixture lacks three live managers");
        for(std::size_t i=0;i<expected.size();++i) {
            const auto& manifold=scene.manifolds_[
                scene.indexed_managers_[i]];
            if(!(manifold.pair==expected[i]) ||
               !manifold.friction_patch_count ||
               !(scene.friction_patches_[
                   manifold.friction_patch_offset].pair==expected[i]))
                throw std::runtime_error(
                    "PhysX friction prep followed Gu order instead of island order");
            const auto& record=scene.friction_by_edge_[
                manifold.edge_handle];
            if(record.epoch!=scene.friction_cache_epoch_ ||
               !(record.pair==expected[i]) ||
               record.offset!=manifold.friction_patch_offset ||
               record.count!=manifold.friction_patch_count)
                throw std::runtime_error(
                    "PhysX cached friction index lost source manager identity");
        }
        scene.step(dt);
        for(const Pair pair:expected) {
            bool retained=false;
            for(const auto& patch:scene.friction_patches_)
                if(patch.pair==pair && patch.anchor_count)
                    retained=true;
            if(!retained)
                throw std::runtime_error(
                    "Noncanonical PhysX manager lost cached friction anchors");
        }
        // Explicit shape editing compacts only B's old patch entries.
        // The remaining C and D cached offsets need rebuilding, but
        // no normal PGS step should scan or re-sort the whole cache.
        scene.set_position(b,{1.15f,10,0});
        scene.step(dt);
        for(const Pair pair:expected) {
            bool retained=false;
            for(const auto& patch:scene.friction_patches_)
                if(patch.pair==pair && patch.anchor_count)
                    retained=true;
            if(!retained)
                throw std::runtime_error(
                    "Shape-edit compaction corrupted source friction history");
        }
    }
    static void verify_source_contact_slop() {
        // This probes the production Gu->Dy constraint stream, rather
        // than using a symmetric floor's final angular velocity as a
        // false oracle (the four point torques can cancel).
        const auto probe=[](float slop) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            scene.add_static_floor(0.0f);
            const auto box=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            RigidBodyDynamics d{};
            d.contact_slop_coefficient=slop;
            scene.set_dynamics(box,d);
            scene.step(1.0f/120.0f);
            if(scene.solver_contact_points_.size()<4 ||
               scene.solver_friction_rows_.empty())
                throw std::runtime_error(
                    "Source contact slop fixture lacks live normal/friction rows");
            const auto max_abs=[](Vec3 v) {
                return std::max({std::abs(v.x),std::abs(v.y),std::abs(v.z)});
            };
            float normal=0.0f,friction=0.0f;
            for(const auto& row:scene.solver_contact_points_)
                normal=std::max(normal,max_abs(row.ra_x_n_sqrt_inertia));
            for(const auto& row:scene.solver_friction_rows_)
                friction=std::max(friction,max_abs(row.ra_x_t_sqrt_inertia));
            return std::pair{normal,friction};
        };
        const auto ordinary=probe(0.0f);
        const auto slopped=probe(1.0f);
        if(!(ordinary.first>0.1f && ordinary.second>0.1f &&
             slopped.first==0.0f && slopped.second==0.0f))
            throw std::runtime_error(
                "Source pair contact slop did not mask Dy angular Jacobians");
    }
    static void verify_signed_compliant_friction_solver() {
        // Codex's two-floor repro enters the signed normal-force path
        // after dozens of ordinary PGS steps. The original source
        // applies FMin(maxDyn,FMax(-maxDyn,candidate)) even when
        // maxDyn is negative; std::clamp has undefined behavior there.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        scene.add_static_floor(0.001f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.495876f,0.0f},1.0f);
        fyzz::Material spring{};
        spring.restitution=-95.11f;
        spring.damping=3.33204f;
        scene.set_material(box,spring);
        scene.set_velocity(box,{-1.39922f,4.64631f,0.0f});
        // The source FMin/FMax limiter must be checked independently
        // of whether a contact patch's final sum is negative on one
        // particular frame. A naturally negative per-Gu-point force
        // does not imply the ENTIRE correlated patch is negative.
        // These are numerical source invariants of the SAME inline
        // expression used in the production PGS solve.
        const auto source_limit=source_contact_friction_limit;
        if(source_limit(0.0,-0.125,true)!=-0.125 ||
           source_limit(0.75,-0.125,true)!=-0.125 ||
           source_limit(-0.75,-0.125,true)!=-0.125 ||
           source_limit(0.1,0.125,false)!=0.1 ||
           source_limit(0.75,0.125,true)!=0.125 ||
           source_limit(-0.75,0.125,true)!=-0.125)
            throw std::runtime_error(
                "Production Dy signed friction limiter diverges from original FMin/FMax");
        bool saw_signed=false;
        for(unsigned frame=0;frame<100;++frame) {
            scene.step(1.0f/60.0f);
            for(double force:scene.solved_normal_impulses()) {
                if(!std::isfinite(force))
                    throw std::runtime_error(
                        "Source signed spring emitted nonfinite Gu force");
                saw_signed|=force<0.0;
            }
            // Inspect the FINAL production solver rows, not merely
            // the output's finiteness. Every live friction patch with
            // negative summed normal force must satisfy the exact
            // DySolverConstraints::solveContact signed FMin/FMax
            // limiter: negative maxDynamic makes the final result
            // maxDynamic for EVERY executed tangential row. Replacing
            // this with std::clamp(-abs(maxDyn),abs(maxDyn)) fails.
            for(std::size_t patch_id=0;
                patch_id<scene.friction_patches_.size();++patch_id) {
                const auto& patch=scene.friction_patches_[patch_id];
                if(!patch.anchor_count)continue;
                double normal_sum=0.0;
                for(const auto& row:scene.solver_contact_points_)
                    if(row.friction_patch==patch_id)
                        normal_sum+=row.applied_force;
                if(normal_sum>=-1.0e-8)continue;
                const double max_dynamic=patch.dynamic_friction*
                    (patch.anchor_count==2 ? 0.5:1.0)*normal_sum;
                if(!(max_dynamic<-1.0e-8))continue;
                bool inspected=false;
                for(std::size_t k=patch.row_offset;
                    k<patch.row_offset+2*patch.anchor_count;++k) {
                    const double result=scene.solver_friction_rows_[k].applied_force;
                    if(!std::isfinite(result) ||
                       std::abs(result-max_dynamic)>1.0e-9*
                           std::max(1.0,std::abs(max_dynamic)))
                        throw std::runtime_error(
                            "Signed Dy friction row differs from source FMin/FMax limiter");
                    inspected=true;
                }
                (void)inspected;
            }
            for(const auto& patch:scene.solved_friction_patches())
                for(std::size_t i=0;i<patch.anchor_count;++i) {
                    const auto v=patch.anchor_impulses[i];
                    if(!std::isfinite(v.x)||!std::isfinite(v.y)||
                       !std::isfinite(v.z))
                        throw std::runtime_error(
                            "Source signed friction bound emitted nonfinite impulse");
                }
        }
        if(!saw_signed)
            throw std::runtime_error(
                "Production compliant PGS did not generate a signed Gu force");
    }
    static void verify_signed_compliant_source_writeback() {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        fyzz::Material compliant{};
        compliant.restitution=-100.0f;
        scene.set_material(box,compliant);
        scene.set_velocity(box,{1.0f,1.0f,0.0f});
        scene.step(1.0f/60.0f);
        if(scene.manifolds_.size()!=1 ||
           scene.manifolds_[0].count==0 ||
           scene.solver_contact_points_.empty())
            throw std::runtime_error(
                "Signed compliant force test lacks real Gu/Dy contact rows");
        const auto& manifold=scene.manifolds_[0];
        // Original DySolverConstraintsShared::solveDynamicContacts can
        // produce this signed result after the compliant multiplier;
        // its original writeBackContact does not clamp the force slot.
        constexpr double source_signed_force=-0.000160085;
        scene.solver_contact_points_[manifold.solver_offset].applied_force=
            source_signed_force;
        scene.begin_contact_writeback();
        scene.write_back_contacts(0,scene.indexed_managers_.size());
        if(scene.solved_normal_impulses().size()!=scene.contact_points_.size() ||
           scene.solved_normal_impulses()[manifold.contact_offset]!=
               source_signed_force)
            throw std::runtime_error(
                "Signed source compliant force was clipped at Dy writeback");
    }
    static void verify_compliant_contact_source() {
        using fyzz::Material;
        using fyzz::MaterialCombineMode;
        constexpr float dt=1.0f/120.0f;
        // Inspect actual Gu-correlated Dy rows after the production
        // Scene::step rather than reproducing the spring implementation
        // in an isolated test solver.
        const auto inspect=[&](Material floor_material,
                               Material body_material, float height=0.49f,
                               float density=2.0f) {
            Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto floor=scene.add_static_floor(0.0f);
            const auto box=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,height,0},density);
            floor_material.disable_friction=true;
            body_material.disable_friction=true;
            scene.set_material(floor,floor_material);
            scene.set_material(box,body_material);
            fyzz::RigidBodyDynamics dynamics{};
            dynamics.gravity_enabled=false;
            scene.set_dynamics(box,dynamics);
            scene.step(dt);
            if(scene.contacts().size()!=1 ||
               scene.friction_patches_.size()!=1 ||
               scene.solver_contact_points_.empty())
                throw std::runtime_error(
                    "Compliant PGS test did not exercise live Gu/Dy contacts");
            const auto patch=scene.friction_patches_[0];
            const auto row=scene.solver_contact_points_[0];
            const auto& point=scene.contacts()[0].points[0];
            const auto sq=[](Vec3 v) {
                return static_cast<double>(v.x)*v.x+
                    static_cast<double>(v.y)*v.y+
                    static_cast<double>(v.z)*v.z;
            };
            const double response=scene.actors_[box].inverse_mass+
                sq(row.ra_x_n_sqrt_inertia)+sq(row.rb_x_n_sqrt_inertia);
            if(!(response>0.0))
                throw std::runtime_error(
                    "Compliant PGS prepared zero effective response");
            if(patch.restitution<0.0f) {
                const double nrdt=static_cast<double>(dt)*patch.restitution;
                // Both pre-solver velocities are zero (gravity disabled).
                const bool separated=point.separation>=0.0f;
                const double enabled_damping=separated?0.0:patch.damping;
                const double coeff=static_cast<double>(dt)*
                    (enabled_damping-nrdt);
                const double mass=patch.acceleration_spring
                    ? 1.0/response:1.0;
                const double x=1.0/(1.0+coeff*
                    (patch.acceleration_spring?1.0:response));
                const double bias=x*(-nrdt*point.separation*mass);
                const double expected_velocity=x*coeff*mass;
                const double expected_impulse=1.0-x;
                const auto near=[](double x,double y) {
                    return std::abs(x-y)<1.0e-6*
                        std::max({1.0,std::abs(x),std::abs(y)});
                };
                if(!near(row.velocity_multiplier,expected_velocity) ||
                   !near(row.impulse_multiplier,expected_impulse) ||
                   !near(row.biased_error,-bias) ||
                   !near(row.unbiased_error,-bias))
                    throw std::runtime_error(
                        "Dy compliant spring coefficients differ from original source");
            } else if(row.impulse_multiplier!=1.0)
                throw std::runtime_error(
                    "Ordinary restitution incorrectly entered compliant PGS");
            return patch;
        };
        Material hard{};
        hard.restitution=0.8f;
        hard.damping=0.0f;
        Material force{};
        force.restitution=-120.0f;
        force.damping=9.0f;
        Material acceleration=force;
        acceleration.compliant_acceleration_spring=true;
        acceleration.restitution=-70.0f;
        acceleration.damping=4.0f;
        const auto mixed=inspect(hard,force);
        if(mixed.restitution!=-120.0f || mixed.damping!=9.0f ||
           mixed.acceleration_spring)
            throw std::runtime_error(
                "PxsMaterialCombiner did not give compliant material priority");
        const auto mixed_reverse=inspect(force,hard);
        if(mixed_reverse.restitution!=-120.0f ||
           mixed_reverse.damping!=9.0f)
            throw std::runtime_error(
                "Mixed compliant material combination depends on body order");
        const auto fast_spring=inspect(force,acceleration);
        if(fast_spring.restitution!=-70.0f || fast_spring.damping!=4.0f ||
           !fast_spring.acceleration_spring)
            throw std::runtime_error(
                "PxsMaterialCombiner acceleration-spring priority missing");
        const auto reversed=inspect(acceleration,force);
        if(reversed.restitution!=-70.0f || reversed.damping!=4.0f ||
           !reversed.acceleration_spring)
            throw std::runtime_error(
                "Acceleration-spring priority depends on body order");
        Material multiplication=force;
        multiplication.restitution=-3.0f;
        multiplication.damping=2.0f;
        multiplication.restitution_combine=MaterialCombineMode::Multiply;
        multiplication.damping_combine=MaterialCombineMode::Maximum;
        const auto multiplied=inspect(multiplication,force);
        if(multiplied.restitution!=-360.0f ||
           multiplied.damping!=9.0f)
            throw std::runtime_error(
                "PxsMaterialCombiner lost negative spring stiffness in multiply");
        // Damping must be disabled for a separated, non-closing spring
        // contact, even when a material authors a substantial damper.
        const auto not_yet_colliding=inspect(hard,force,0.525f);
        if(not_yet_colliding.restitution!=-120.0f)
            throw std::runtime_error(
                "Separated compliant contact was not correlated");
        const auto ordinary=inspect(hard,hard);
        if(ordinary.restitution!=0.8f)
            throw std::runtime_error(
                "Ordinary material preparation regressed");
    }
    static void verify_zero_capped_dynamic_partition_lifecycle() {
        // Source DyConstraintPartition::initSolverProgress resets the
        // three scheduler fields of every admitted active rigid body,
        // even when DyContactPrepShared::extractContacts filters ALL
        // its zero-maxImpulse rows. This is a real dynamic/dynamic
        // manager, not a static-floor case that bypasses mask logic.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,2.0f,0.0f},1.0f);
        const auto b=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.9f,2.0f,0.0f},1.0f);
        RigidBodyDynamics capped{};
        capped.gravity_enabled=false;
        capped.max_contact_impulse=0.0f;
        scene.set_dynamics(a,capped);
        scene.set_dynamics(b,capped);
        constexpr float dt=1.0f/480.0f;
        const Pair pair{a,b};
        for(unsigned frame=0;frame<112;++frame) {
            scene.step(dt);
            if(scene.manifolds_.size()!=1 ||
               scene.manifolds_[0].pair!=pair ||
               !scene.contact_edge_connected(pair) ||
               scene.indexed_managers_.size()!=1 ||
               scene.ordered_constraints_.size()!=1 ||
               scene.solver_contact_points_.size()!=0 ||
               !scene.solver_contact_order_.empty() ||
               !scene.solver_active_ids_.empty())
                throw std::runtime_error(
                    "Zero-capped dynamic pair lost source Gu manager or created Dy rows");
            for(const BodyId id:{a,b}) {
                const auto& row=scene.solver_rows_[id];
                if(row.prepared_generation==scene.solver_preparation_generation_ ||
                   row.solver.partition_mask!=1 ||
                   row.solver.max_dynamic_partition!=1 ||
                   row.solver.static_constraint_count!=0)
                    throw std::runtime_error(
                        "Source zero-response dynamic partition retained prior-frame progress");
            }
            const auto& contact=scene.manifolds_[0];
            if(!scene.solved_normal_impulses().size() ||
               contact.contact_offset>scene.solved_normal_impulses().size() ||
               contact.count>scene.solved_normal_impulses().size()-
                   contact.contact_offset)
                throw std::runtime_error(
                    "Zero-response partition lost original Gu writeback offsets");
            for(std::size_t j=0;j<contact.count;++j)
                if(scene.solved_normal_impulses()[contact.contact_offset+j]!=0)
                    throw std::runtime_error(
                        "Zero-response dynamic manager wrote nonzero Gu impulse");
        }
        // Move from zero response to live Dy rows, then back again,
        // without changing the source edge or actor IDs.
        RigidBodyDynamics live=capped;
        live.max_contact_impulse=std::numeric_limits<float>::max();
        scene.set_dynamics(a,live);
        scene.set_dynamics(b,live);
        scene.step(dt);
        if(scene.manifolds_.size()!=1 ||
           scene.solver_contact_points_.empty() ||
           scene.solver_active_ids_.size()!=2 ||
           scene.solver_rows_[a].prepared_generation!=
               scene.solver_preparation_generation_ ||
           scene.solver_rows_[b].prepared_generation!=
               scene.solver_preparation_generation_)
            throw std::runtime_error(
                "Zero-capped dynamic pair did not reenter source PGS");
        scene.set_dynamics(a,capped);
        for(unsigned frame=0;frame<48;++frame) {
            // The first 24 frames have ONE capped body and a live
            // partner; the second 24 have BOTH sides capped. Source
            // extractContacts applies the minimum across the pair.
            if(frame==24)scene.set_dynamics(b,capped);
            // Keep the same two source contact managers overlapping
            // through readmission's prior impact displacement.
            scene.set_position(a,{0.0f,2.0f,0.0f});
            scene.set_position(b,{0.9f,2.0f,0.0f});
            scene.set_velocity(a,{});
            scene.set_velocity(b,{});
            scene.step(dt);
            if(scene.manifolds_.size()!=1 ||
               scene.ordered_constraints_.size()!=1 ||
               !scene.solver_contact_points_.empty() ||
               !scene.solver_active_ids_.empty() ||
               scene.solver_rows_[a].solver.partition_mask!=1 ||
               scene.solver_rows_[b].solver.partition_mask!=1)
                throw std::runtime_error(
                    "Re-capping original PGS manager leaked stale partition or rows");
        }
    }
    static void verify_zero_capped_stream_compaction() {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto zero=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-3.0f,0.49f,0},1.0f);
        const auto live=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3.0f,0.49f,0},1.0f);
        RigidBodyDynamics d{};
        d.max_contact_impulse=0.0f;
        scene.set_dynamics(zero,d);
        constexpr float dt=1.0f/120.0f;
        scene.step(dt);
        if(scene.manifolds_.size()!=2 ||
           scene.manifolds_[0].pair!=Pair{floor,zero} ||
           scene.manifolds_[1].pair!=Pair{floor,live})
            throw std::runtime_error(
                "Source compact-zero fixture lost ordered Gu managers");
        const auto first=scene.manifolds_[0];
        const auto second=scene.manifolds_[1];
        if(first.count<2 || second.count<2 ||
           first.friction_patch_count!=0 ||
           second.friction_patch_count==0 ||
           first.solver_offset!=0 ||
           second.solver_offset!=0 ||
           scene.solver_contact_points_.size()!=second.count ||
           scene.solver_contact_order_.size()!=second.count)
            throw std::runtime_error(
                "Zero-capped Gu manager allocated forbidden source Dy rows");
        for(std::size_t j=0;j<first.count;++j)
            if(scene.normal_impulse_writeback_[first.contact_offset+j]!=0)
                throw std::runtime_error(
                    "Zero-capped source Gu force buffer not explicitly zeroed");
        double live_sum=0.0;
        for(std::size_t j=0;j<second.count;++j)
            live_sum+=scene.normal_impulse_writeback_[
                second.contact_offset+j];
        if(!(live_sum>0.0) || !scene.contact_edge_connected({floor,zero}))
            throw std::runtime_error(
                "Compacted source Dy rows lost later live Gu writeback or edge");
        d.max_contact_impulse=std::numeric_limits<float>::max();
        scene.set_dynamics(zero,d);
        scene.step(dt);
        if(scene.manifolds_.size()!=2 ||
           scene.solver_contact_points_.size()!=
               scene.manifolds_[0].count+scene.manifolds_[1].count)
            throw std::runtime_error(
                "Previously zero-capped source manager failed solver readmission");
        d.max_contact_impulse=0.0f;
        scene.set_dynamics(zero,d);
        scene.step(dt);
        if(scene.solver_contact_points_.size()!=scene.manifolds_[1].count)
            throw std::runtime_error(
                "Zero-cap retry retained stale source Dy rows");
        for(std::size_t j=0;j<scene.manifolds_[0].count;++j)
            if(scene.normal_impulse_writeback_[
                scene.manifolds_[0].contact_offset+j]!=0)
                throw std::runtime_error(
                    "Zero-cap retry leaked prior frame impulse into Gu writeback");
    }
};
}

namespace fyzz::native {
// Verify the actual production DyDynamics task-chain descriptors. The
// source computes maximum solverIterationCounts across all bodies in
// each task (including unconstrained members), not across the scene.
struct SolverIterationSourceFixture {
    static void verify_active_manager_admission() {
        constexpr float dt=1.0f/120.0f;
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_contact_notifications(true);
        const auto floor=scene.add_static_floor(0.0f);
        // Retain a substantial accurate Gu contact history that is
        // entirely dormant. Its managers must not be traversed again
        // for either rigid-body solver admission or Dy sleepCheck.
        constexpr unsigned resting=48;
        for(unsigned i=0;i<resting;++i)
            scene.add_dynamic_box({0.5f,0.5f,0.5f},
                {static_cast<float>(i)*3.0f,0.5f,0.0f},1.0f);
        for(unsigned i=0;i<360;++i)scene.step(dt);
        if(!scene.accurate_islands_.active_islands.empty() ||
           scene.manifolds_.size()<resting)
            throw std::runtime_error(
                "Active-manager admission fixture has no dormant Gu history");
        const auto first=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1000.0f,0.49f,0.0f},1.0f);
        const auto second=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1000.8f,0.49f,0.0f},1.0f);
        scene.step(dt);
        if(scene.manifolds_.size()<=3 ||
           scene.indexed_managers_.size()!=3 ||
           scene.solver_active_ids_.size()!=2 ||
           std::find(scene.solver_active_ids_.begin(),
                     scene.solver_active_ids_.end(),first)==
               scene.solver_active_ids_.end() ||
           std::find(scene.solver_active_ids_.begin(),
                     scene.solver_active_ids_.end(),second)==
               scene.solver_active_ids_.end() ||
           scene.actors_[first].num_counted_interactions!=2 ||
           scene.actors_[second].num_counted_interactions!=2)
            throw std::runtime_error(
                "PhysX active island contact edges did not drive both Dy stages");
        // Dormant Gu managers remain queryable, but the unified input
        // carries only this frame's three active source descriptors.
        // It no longer allocates or clears stamps for dormant Gu slots.
        if(scene.constraint_descriptors_.size()!=3 ||
           scene.constraint_partitions_.size()!=3 ||
           scene.ordered_constraints_.size()!=3)
            throw std::runtime_error(
                "Source partition stream includes dormant Gu history");
        for(const auto& descriptor:scene.constraint_descriptors_) {
            if(descriptor.distance ||
               std::find(scene.indexed_managers_.begin(),
                         scene.indexed_managers_.end(),
                         descriptor.index)==scene.indexed_managers_.end())
                throw std::runtime_error(
                    "Dormant Gu manager was repartitioned by active PGS");
        }
        for(std::size_t i=0;i<scene.indexed_managers_.size();++i) {
            const auto pair=scene.manifolds_[
                scene.indexed_managers_[i]].pair;
            if(pair.a!=first && pair.b!=first &&
               pair.a!=second && pair.b!=second)
                throw std::runtime_error(
                    "Dormant Gu manager leaked into active PGS task chain");
        }
        // The dense original Gu force-buffer remains entirely indexed
        // by contact generation. Every dormant historical manager must
        // contribute zero force without being visited by source Dy
        // writeback or acquiring fresh solver descriptors.
        const auto forces=scene.solved_normal_impulses();
        bool active_force=false;
        std::size_t dormant_contacts=0;
        for(const auto& manager:scene.manifolds_) {
            for(std::size_t j=0;j<manager.count;++j) {
                const std::size_t index=manager.contact_offset+j;
                if(index>=forces.size())
                    throw std::runtime_error(
                        "Gu writeback omitted its original dense contact slot");
                if(manager.pair.b<first) {
                    ++dormant_contacts;
                    if(forces[index]!=0.0 ||
                       manager.friction_patch_count!=0 ||
                       manager.prepared_friction_epoch!=0)
                        throw std::runtime_error(
                            "Dormant Gu contact retained previous Dy response");
                } else if(forces[index]>0.0)active_force=true;
            }
        }
        if(dormant_contacts<resting || !active_force)
            throw std::runtime_error(
                "Active Gu force writeback lost either the dormant zeroes or the current solve");
        // Sc's opt-in found-touch and active-manager producers must
        // publish exactly these three reports, in canonical Gu order.
        // None of the 48 retained sleeping Gu managers may reappear.
        const auto events=scene.contact_events();
        if(events.size()!=3 ||
           events[0].body0!=floor || events[0].body1!=first ||
           events[1].body0!=floor || events[1].body1!=second ||
           events[2].body0!=first || events[2].body1!=second ||
           scene.contact_report_managers_.size()!=3)
            throw std::runtime_error(
                "Sc active contact notification included dormant Gu history or lost original order");
        for(const auto& event:events)
            if(event.kind!=ContactEventKind::Found ||
               event.contact_count==0)
                throw std::runtime_error(
                    "Sc active-manager notification omitted a newly touching pair");
        // Even when a live accurate manager has no Dy response rows,
        // Sc retains its contact edge and counted interactions. Only
        // the remaining floor contact should require solver-body data.
        RigidBodyDynamics disabled{};
        disabled.max_contact_impulse=0.0f;
        scene.set_dynamics(second,disabled);
        scene.step(dt);
        if(scene.constraint_descriptors_.size()!=3 ||
           scene.constraint_partitions_.size()!=3 ||
           scene.ordered_constraints_.size()!=3)
            throw std::runtime_error(
                "Source PGS partition retained stale prior-frame descriptors");
        if(scene.solver_active_ids_.size()!=1 ||
           scene.solver_active_ids_[0]!=first ||
           scene.indexed_managers_.size()!=3 ||
           scene.actors_[first].num_counted_interactions!=2 ||
           scene.actors_[second].num_counted_interactions!=2)
            throw std::runtime_error(
                "Zero-response accurate edge changed source body admission or sleep count");
        scene.set_dynamics(second,RigidBodyDynamics{});
        scene.step(dt);
        if(scene.solver_active_ids_.size()!=2 ||
           scene.indexed_managers_.size()!=3)
            throw std::runtime_error(
                "Source active manager admission failed after response reenable");
    }
    static void verify_parallel_solver_end_rollback() {
        // Force overflow inside the worker-owned integrateCore rather
        // than during PGS. A completed serial task and an outstanding
        // worker must both join before the single scene checkpoint can
        // roll back motion and the preceding published Gu result.
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_enhanced_determinism(true);
        scene.parallel_third_pass_enabled_=false;
        scene.parallel_solver_min_descriptors_=0;
        scene.add_static_floor(0.0f);
        std::array<BodyId,8> ids{};
        for(std::size_t i=0;i<ids.size();++i) {
            ids[i]=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},
                {float(i)*3.0f,i==7?10.0f:0.48f,0},1.0f);
            RigidBodyDynamics d{};
            d.gravity_enabled=false;
            if(i==7)d.max_linear_velocity=1.0e18f;
            scene.set_dynamics(ids[i],d);
        }
        scene.set_velocity(ids.back(),{1.0e18f,0,0});
        const auto frame=scene.frame_index();
        std::array<BodyState,8> before{};
        for(std::size_t i=0;i<ids.size();++i)
            before[i]=scene.body_state(ids[i]);
        bool rejected=false;
        try {scene.step(1.0e25f);}
        catch(const std::overflow_error& e) {
            rejected=std::string_view(e.what())==
                "Native position-iteration motion overflow";
        }
        if(!rejected || scene.frame_index()!=frame ||
           scene.solved_result_frame()!=0 ||
           scene.concurrent_solver_worker_jobs_!=1 ||
           !scene.speculative_ready_nodes_.empty() ||
           (scene.third_pass_worker_state_ &&
            scene.third_pass_worker_state_->outstanding))
            throw std::runtime_error(
                "Concurrent source solver-end overflow was not failure-atomic");
        for(std::size_t i=0;i<ids.size();++i) {
            const auto after=scene.body_state(ids[i]);
            const auto& old=before[i];
            if(after.position.x!=old.position.x ||
               after.position.y!=old.position.y ||
               after.position.z!=old.position.z ||
               after.velocity.x!=old.velocity.x ||
               after.velocity.y!=old.velocity.y ||
               after.velocity.z!=old.velocity.z ||
               after.sleeping!=old.sleeping)
                throw std::runtime_error(
                    "Concurrent solver-end rejection retained partial rigid motion");
        }
        scene.set_velocity(ids.back(),{});
        scene.step(1.0f/120.0f);
        if(scene.concurrent_integration_dispatches_!=1 ||
           scene.concurrent_solver_worker_jobs_!=2 ||
           scene.solved_result_frame()!=scene.frame_index() ||
           scene.solver_tasks_.size()!=ids.size())
            throw std::runtime_error(
                "Concurrent source solver-end retry did not complete every task");
    }
    static void verify_parallel_pgs_groups() {
        constexpr float dt=1.0f/120.0f;
        constexpr std::size_t count=12;
        auto serial_owner=std::make_unique<Scene>(
            Scene::ContactGenerator::DiagnosticLegacyGu);
        auto concurrent_owner=std::make_unique<Scene>(
            Scene::ContactGenerator::DiagnosticLegacyGu);
        Scene& serial=*serial_owner;
        Scene& concurrent=*concurrent_owner;
        serial.parallel_solver_enabled_=false;
        concurrent.parallel_solver_min_descriptors_=0;
        std::array<BodyId,count> serial_ids{}, concurrent_ids{};
        for(Scene* scene:{&serial,&concurrent}) {
            scene->set_enhanced_determinism(true);
            scene->add_static_floor(0.0f);
            for(std::size_t i=0;i<count;++i) {
                const auto id=scene->add_dynamic_box(
                    {0.5f,0.5f,0.5f},
                    {static_cast<float>(i)*3.0f,0.48f,0},1.0f);
                (scene==&serial?serial_ids:concurrent_ids)[i]=id;
                RigidBodyDynamics dynamics{};
                dynamics.position_iterations=i&1u?6:4;
                dynamics.velocity_iterations=i&1u?2:1;
                scene->set_dynamics(id,dynamics);
            }
        }
        const auto same=[](Vec3 a,Vec3 b) {
            return a.x==b.x && a.y==b.y && a.z==b.z;
        };
        for(std::size_t frame=0;frame<6;++frame) {
            serial.step(dt);
            concurrent.step(dt);
            if(serial.solver_tasks_.size()!=count ||
               concurrent.solver_tasks_.size()!=count ||
               concurrent.concurrent_solver_dispatches_!=frame+1 ||
               concurrent.concurrent_integration_dispatches_!=frame+1 ||
               concurrent.concurrent_solver_worker_jobs_!=frame+1 ||
               concurrent.third_pass_worker_start_count_!=1 ||
               concurrent.solved_result_frame()!=concurrent.frame_index() ||
               serial.solved_result_frame()!=serial.frame_index())
                throw std::runtime_error(
                    "Independent Dy PGS task chains did not execute concurrently");
            for(std::size_t i=0;i<count;++i) {
                const auto lhs=serial.body_state(serial_ids[i]);
                const auto rhs=concurrent.body_state(concurrent_ids[i]);
                if(!same(lhs.position,rhs.position) ||
                   !same(lhs.velocity,rhs.velocity) ||
                   !same(lhs.angular_velocity,rhs.angular_velocity) ||
                   lhs.orientation.x!=rhs.orientation.x ||
                   lhs.orientation.y!=rhs.orientation.y ||
                   lhs.orientation.z!=rhs.orientation.z ||
                   lhs.orientation.w!=rhs.orientation.w ||
                   lhs.sleeping!=rhs.sleeping ||
                   serial.accurate_island_id(serial_ids[i])!=
                       concurrent.accurate_island_id(concurrent_ids[i]))
                    throw std::runtime_error(
                        "Concurrent Dy PGS changed independent rigid body/island state");
            }
            if(serial.speculative_ready_nodes_!=
                   concurrent.speculative_ready_nodes_)
                throw std::runtime_error(
                    "Parallel source sleepCheck changed next-frame readiness order");
            // Source preIntegration must have journaled every awake
            // island node before the two task lanes may mutate actors.
            for(BodyId id:concurrent.accurate_island_nodes_)
                if(!concurrent.actors_[id].sleeping &&
                   concurrent.motion_undo_generations_[id]!=
                       concurrent.motion_undo_generation_)
                    throw std::runtime_error(
                        "Source preIntegration skipped concurrent actor rollback");
            for(std::size_t i=0;i<count;++i) {
                const auto& a=serial.actors_[serial_ids[i]];
                const auto& b=concurrent.actors_[concurrent_ids[i]];
                if(a.ready_for_sleep!=b.ready_for_sleep ||
                   a.wake_counter!=b.wake_counter ||
                   !same(a.sleep_linear_acc,b.sleep_linear_acc) ||
                   !same(a.sleep_angular_acc,b.sleep_angular_acc) ||
                   !same(a.last_ccd_position,b.last_ccd_position) ||
                   a.last_ccd_generation!=b.last_ccd_generation)
                    throw std::runtime_error(
                        "Parallel source solver-end sleep/integration state differs");
            }
            const auto left=serial.solved_normal_impulses();
            const auto right=concurrent.solved_normal_impulses();
            if(left.size()!=right.size() ||
               !std::equal(left.begin(),left.end(),right.begin()))
                throw std::runtime_error(
                    "Concurrent Dy PGS changed Gu-indexed contact force writeback");
        }
        // A task-local static-world row must never retain another
        // island's solver deltas or publish a nonzero world velocity.
        if(!same(concurrent.world_solver_row_.solver.linear_velocity_delta,{}) ||
           !same(concurrent.world_solver_row_.solver.angular_state,{}))
            throw std::runtime_error(
                "Concurrent Dy PGS modified the shared static solver body");

        // Two rigid nodes coupled by a source 1D distance row belong
        // to ONE accurate island, but different joints are independent
        // solver task groups. Validate the mixed 1D/contact PGS stream
        // through all position and velocity iterations, not only
        // isolated static-contact row updates.
        auto joints_serial_owner=std::make_unique<Scene>(
            Scene::ContactGenerator::DiagnosticLegacyGu);
        auto joints_parallel_owner=std::make_unique<Scene>(
            Scene::ContactGenerator::DiagnosticLegacyGu);
        Scene& joints_serial=*joints_serial_owner;
        Scene& joints_parallel=*joints_parallel_owner;
        joints_serial.set_enhanced_determinism(true);
        joints_parallel.set_enhanced_determinism(true);
        joints_serial.parallel_solver_enabled_=false;
        joints_parallel.parallel_solver_min_descriptors_=0;
        joints_serial.parallel_third_pass_enabled_=false;
        joints_parallel.parallel_third_pass_enabled_=false;
        constexpr std::size_t pairs=8;
        std::array<BodyId,pairs*2> left_ids{},right_ids{};
        for(Scene* scene:{&joints_serial,&joints_parallel}) {
            scene->add_static_floor(0.0f);
            auto& ids=scene==&joints_serial?left_ids:right_ids;
            for(std::size_t i=0;i<pairs;++i) {
                const float base=float(i)*6.0f;
                const auto a=scene->add_dynamic_box(
                    {0.5f,0.5f,0.5f},{base,0.48f,0},1.0f);
                const auto b=scene->add_dynamic_box(
                    {0.5f,0.5f,0.5f},{base+1.4f,0.48f,0},1.0f);
                ids[2*i]=a;
                ids[2*i+1]=b;
                scene->add_distance_constraint(a,b,1.4f);
                RigidBodyDynamics dynamics{};
                dynamics.position_iterations=(i&1u)?7:4;
                dynamics.velocity_iterations=(i&1u)?2:1;
                scene->set_dynamics(a,dynamics);
                scene->set_dynamics(b,dynamics);
                scene->set_velocity(a,{0.3f,-0.3f,0});
                scene->set_velocity(b,{-0.2f,-0.1f,0});
            }
        }
        for(std::size_t frame=0;frame<8;++frame) {
            joints_serial.step(dt);
            joints_parallel.step(dt);
            if(joints_serial.solver_tasks_.size()!=pairs ||
               joints_parallel.solver_tasks_.size()!=pairs ||
               joints_parallel.concurrent_solver_dispatches_!=frame+1 ||
               joints_parallel.concurrent_integration_dispatches_!=frame+1 ||
               joints_parallel.concurrent_solver_worker_jobs_!=frame+1 ||
               joints_parallel.third_pass_worker_start_count_!=1 ||
               joints_serial.indexed_distance_constraints_.size()!=pairs ||
               joints_parallel.indexed_distance_constraints_.size()!=pairs)
                throw std::runtime_error(
                    "Independent contact/1D Dy task chains failed concurrent admission");
            for(std::size_t i=0;i<left_ids.size();++i) {
                const auto a=joints_serial.body_state(left_ids[i]);
                const auto b=joints_parallel.body_state(right_ids[i]);
                if(!same(a.position,b.position) ||
                   !same(a.velocity,b.velocity) ||
                   !same(a.angular_velocity,b.angular_velocity) ||
                   a.orientation.x!=b.orientation.x ||
                   a.orientation.y!=b.orientation.y ||
                   a.orientation.z!=b.orientation.z ||
                   a.orientation.w!=b.orientation.w ||
                   a.sleeping!=b.sleeping ||
                   joints_serial.accurate_island_id(left_ids[i])!=
                       joints_parallel.accurate_island_id(right_ids[i]))
                    throw std::runtime_error(
                        "Concurrent mixed 1D/contact task altered rigid state");
            }
            if(joints_serial.speculative_ready_nodes_!=
                   joints_parallel.speculative_ready_nodes_)
                throw std::runtime_error(
                    "Parallel mixed source task changed ready-node publication");
            const auto serial_impulses=joints_serial.solved_normal_impulses();
            const auto parallel_impulses=joints_parallel.solved_normal_impulses();
            if(serial_impulses.size()!=parallel_impulses.size() ||
               !std::equal(serial_impulses.begin(),serial_impulses.end(),
                           parallel_impulses.begin()) ||
               joints_serial.distance_rows_.size()!=
                   joints_parallel.distance_rows_.size())
                throw std::runtime_error(
                    "Concurrent mixed task changed source Gu/1D writeback");
            for(std::size_t i=0;i<joints_serial.distance_rows_.size();++i)
                if(joints_serial.distance_rows_[i].applied_force!=
                   joints_parallel.distance_rows_[i].applied_force)
                    throw std::runtime_error(
                        "Concurrent source 1D impulse disagreed with serial PGS");
        }

        // Unlike the source static-world body, a kinematic actor has an
        // individually prepared row and may touch two separate accurate
        // rigid islands. Do not concurrently solve those shared rows.
        auto mixed_owner=std::make_unique<Scene>(
            Scene::ContactGenerator::DiagnosticLegacyGu);
        Scene& mixed=*mixed_owner;
        mixed.set_enhanced_determinism(true);
        mixed.parallel_third_pass_enabled_=false;
        mixed.parallel_solver_min_descriptors_=0;
        mixed.add_static_floor(0.0f);
        const auto kinematic=mixed.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.48f,0},1.0f);
        mixed.set_kinematic(kinematic,true);
        for(float x:{-0.85f,0.85f,4.0f,8.0f})
            mixed.add_dynamic_box(
                {0.5f,0.5f,0.5f},{x,0.48f,0},1.0f);
        mixed.step(dt);
        bool found_kinematic_constraint=false;
        for(const auto& descriptor:mixed.ordered_constraints_) {
            if(descriptor.distance)continue;
            const auto pair=mixed.manifolds_[descriptor.index].pair;
            found_kinematic_constraint |=
                pair.a==kinematic || pair.b==kinematic;
        }
        if(!found_kinematic_constraint || mixed.solver_tasks_.size()<2 ||
           mixed.concurrent_solver_dispatches_!=0 ||
           mixed.concurrent_solver_worker_jobs_!=0 ||
           mixed.third_pass_worker_start_count_!=0)
            throw std::runtime_error(
                "Shared kinematic solver row escaped serial task admission");
    }
    static void verify() {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        constexpr std::size_t bodies=130;
        for(std::size_t i=0;i<bodies;++i) {
            const auto id=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},
                {static_cast<float>(i)*3.0f,0.49f,0},1.0f);
            RigidBodyDynamics d{};
            if(i==0) {d.position_iterations=7;d.velocity_iterations=2;}
            if(i==127) {d.position_iterations=9;d.velocity_iterations=3;}
            if(i==128) {d.position_iterations=5;d.velocity_iterations=4;}
            scene.set_dynamics(id,d);
        }
        scene.step(1.0f/120.0f);
        if(scene.solver_tasks_.size()!=2 ||
           scene.solver_tasks_[0].manager_end!=128 ||
           scene.solver_tasks_[1].manager_end!=130 ||
           scene.solver_tasks_[0].position_iterations!=9 ||
           scene.solver_tasks_[0].velocity_iterations!=3 ||
           scene.solver_tasks_[1].position_iterations!=5 ||
           scene.solver_tasks_[1].velocity_iterations!=4 ||
           scene.solver_tasks_[0].solver_body_end!=128 ||
           scene.solver_tasks_[1].solver_body_end!=130 ||
           scene.solver_tasks_[0].constraint_end!=128 ||
           scene.solver_tasks_[1].constraint_end!=130)
            throw std::runtime_error(
                "Source PGS task iteration maxima leaked across 128-body boundary");
        // Iteration settings must never alter the permanent island graph
        // or repartition independent body contacts into a false conflict.
        if(scene.active_accurate_islands().size()!=bodies ||
           scene.solver_partition_count()!=2)
            throw std::runtime_error(
                "Per-body source solver iterations changed island ownership");
        scene.step(1.0f/120.0f);
        if(scene.solver_tasks_.size()!=2 ||
           scene.solver_tasks_[0].position_iterations!=9 ||
           scene.solver_tasks_[1].position_iterations!=5)
            throw std::runtime_error(
                "Per-task source iteration maxima did not persist across steps");
        // Both bodies in the second task request ZERO velocity passes.
        // This must remain zero in the authoritative Dy task maximum,
        // while the solver still performs its final velocity/writeback
        // solve. An unrelated 3-velocity request in task 0 must not
        // promote the second task's source count.
        RigidBodyDynamics zero{};
        zero.velocity_iterations=0;
        zero.position_iterations=5;
        scene.set_dynamics(static_cast<BodyId>(bodies-1),zero);
        scene.set_dynamics(static_cast<BodyId>(bodies),zero);
        scene.step(1.0f/120.0f);
        if(scene.solver_tasks_.size()!=2 ||
           scene.solver_tasks_[0].velocity_iterations!=3 ||
           scene.solver_tasks_[1].velocity_iterations!=0 ||
           scene.solver_tasks_[1].position_iterations!=5 ||
           scene.solver_tasks_[1].constraint_end!=bodies)
            throw std::runtime_error(
                "Zero requested source velocity count leaked across task boundary");
        const auto forces=scene.solved_normal_impulses();
        const auto contacts=scene.contacts();
        std::size_t last_task_contacts=0;
        double last_task_force=0.0;
        for(const auto& manager:contacts) {
            if(manager.pair.b<bodies-1)continue;
            ++last_task_contacts;
            for(std::size_t j=0;j<manager.count;++j)
                last_task_force+=forces[manager.contact_offset+j];
        }
        if(last_task_contacts!=2 || !std::isfinite(last_task_force) ||
           last_task_force<=0.0)
            throw std::runtime_error(
                "Zero-velocity task lost its Gu-indexed contact force buffer");

        // A positive force buffer is NOT evidence of a velocity pass:
        // position iterations already populate applied_force. Exercise
        // the actual final unbiased phase in a fresh source 128+2 task
        // arrangement. The last two contacts begin with a substantial
        // closing velocity and penetration bias, so their position
        // motion differs measurably from the unbiased final velocity.
        struct FinalPassSample {
            Vec3 motion{}, final{};
            double normal_impulse=0.0;
            std::size_t contact_count=0;
            std::uint8_t requested_velocity_iterations=255;
        };
        const auto run_final_pass=[](std::uint8_t request) {
            Scene tested{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto floor=tested.add_static_floor(0.0f);
            // Suppress restitution, which removes the position bias in
            // DyContactPrep. We need an actual penetration-bias delta
            // that the final UNBIASED pass must undo, not a bouncing
            // contact where both phases may legitimately coincide.
            Material no_bounce{};
            no_bounce.restitution=0.0f;
            tested.set_material(floor,no_bounce);
            constexpr BodyId body_count=130;
            for(BodyId i=0;i<body_count;++i) {
                const auto id=tested.add_dynamic_box(
                    {0.5f,0.5f,0.5f},
                    {static_cast<float>(i)*3.0f,
                     i>=128 ? 0.45f : 0.49f,0.0f},1.0f);
                RigidBodyDynamics d{};
                d.position_iterations=5;
                d.velocity_iterations=i>=128 ? request : 3;
                tested.set_dynamics(id,d);
                if(i>=128) {
                    tested.set_material(id,no_bounce);
                    tested.set_velocity(id,{0.0f,-2.0f,0.0f});
                }
            }
            tested.step(1.0f/120.0f);
            if(tested.solver_tasks_.size()!=2 ||
               tested.solver_tasks_[0].velocity_iterations!=3 ||
               tested.solver_tasks_[1].velocity_iterations!=request ||
               tested.solver_tasks_[1].constraint_end!=body_count)
                throw std::runtime_error(
                    "Final-pass fixture lost source 128+2 task isolation");
            const BodyId last=body_count;
            const auto& row=tested.solver_rows_[last];
            if(row.prepared_generation!=tested.solver_preparation_generation_)
                throw std::runtime_error(
                    "Final-pass fixture omitted second-task solver body");
            FinalPassSample sample{};
            sample.motion={
                row.data.linear_velocity.x+row.motion_linear_delta.x,
                row.data.linear_velocity.y+row.motion_linear_delta.y,
                row.data.linear_velocity.z+row.motion_linear_delta.z};
            sample.final=tested.body_state(last).velocity;
            sample.requested_velocity_iterations=
                tested.solver_tasks_[1].velocity_iterations;
            for(const auto& manager:tested.contacts()) {
                if(manager.pair.b<body_count-1)continue;
                ++sample.contact_count;
                for(std::size_t j=0;j<manager.count;++j)
                    sample.normal_impulse+=
                        tested.solved_normal_impulses()[
                            manager.contact_offset+j];
            }
            return sample;
        };
        const auto actual_zero=run_final_pass(0);
        const auto actual_one=run_final_pass(1);
        const auto close=[](float a,float b) {
            return std::isfinite(a) && std::isfinite(b) &&
                   std::abs(a-b)<=2.0e-5f;
        };
        const auto close_vec=[&](Vec3 a,Vec3 b) {
            return close(a.x,b.x) && close(a.y,b.y) && close(a.z,b.z);
        };
        if(actual_zero.contact_count!=2 ||
           actual_one.contact_count!=2)
            throw std::runtime_error(
                "Final-velocity fixture lacks two source task contacts");
        if(actual_zero.requested_velocity_iterations!=0 ||
           actual_one.requested_velocity_iterations!=1)
            throw std::runtime_error(
                "Final-velocity fixture did not preserve source requested counts");
        if(!close_vec(actual_zero.motion,actual_one.motion))
            throw std::runtime_error(
                "Position-phase snapshots changed between zero/one requests");
        if(!close_vec(actual_zero.final,actual_one.final))
            throw std::runtime_error(
                "Zero/one velocity requests produced different final contact motion");
        if(!std::isfinite(actual_zero.normal_impulse) ||
           actual_zero.normal_impulse<=0 ||
           std::abs(actual_zero.normal_impulse-
                    actual_one.normal_impulse)>1.0e-5)
            throw std::runtime_error(
                "Zero/one velocity requests changed source Gu impulse writeback");
        if(std::abs(actual_one.final.y-actual_one.motion.y)<=1.0e-3f)
            throw std::runtime_error(
                "Reference one-pass contact fixture has no unbiased velocity correction");
        if(std::abs(actual_zero.final.y-actual_zero.motion.y)<=1.0e-3f)
            throw std::runtime_error(
                "Zero-request task omitted the final unbiased velocity correction");
    }
};
}

void verify_source_zero_velocity_iteration_request() {
    using namespace fyzz;
    using fyzz::native::Scene;
    constexpr float dt=1.0f/120.0f;
    const auto near=[](float a,float b,float epsilon=2.0e-5f) {
        return std::isfinite(a) && std::isfinite(b) &&
               std::abs(a-b)<=epsilon;
    };
    const auto equivalent=[&](const BodyState& a,const BodyState& b) {
        return near(a.position.x,b.position.x) &&
               near(a.position.y,b.position.y) &&
               near(a.position.z,b.position.z) &&
               near(a.velocity.x,b.velocity.x) &&
               near(a.velocity.y,b.velocity.y) &&
               near(a.velocity.z,b.velocity.z) &&
               near(a.angular_velocity.x,b.angular_velocity.x) &&
               near(a.angular_velocity.y,b.angular_velocity.y) &&
               near(a.angular_velocity.z,b.angular_velocity.z);
    };
    // A real contact with closing speed and restitution is decisive:
    // skipping the mandatory final velocity/writeback pass must
    // change the published state. A free-flight-only comparison would
    // not distinguish the source's zero/one velocity-iteration rule.
    struct ContactStep {
        BodyState body;
        double normal_impulse=0;
    };
    const auto native=[&](std::uint8_t vel_iterations) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.49f,0.0f},1.0f);
        RigidBodyDynamics d{};
        d.position_iterations=1;
        d.velocity_iterations=vel_iterations;
        scene.set_dynamics(box,d);
        scene.set_velocity(box,{0.35f,-4.0f,0.0f});
        scene.step(dt);
        double impulse=0.0;
        for(const double value:scene.solved_normal_impulses())
            impulse+=value;
        return ContactStep{scene.body_state(box),impulse};
    };
    const auto zero=native(0), one=native(1);
    if(!equivalent(zero.body,one.body) ||
       !std::isfinite(zero.normal_impulse) ||
       zero.normal_impulse<=0 ||
       std::abs(zero.normal_impulse-one.normal_impulse)>1.0e-6)
        throw std::runtime_error(
            "Source zero velocity iterations must execute final PGS/writeback");

    const auto public_body=[&](ExecutionMode mode,
                              std::uint8_t vel_iterations) {
        World world(WorldOptions{mode});
        world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.49f,0.0f},1.0f);
        RigidBodyDynamics d{};
        d.position_iterations=1;
        d.velocity_iterations=vel_iterations;
        world.set_body_dynamics(box,d);
        world.set_body_velocity(box,{0.35f,-4.0f,0.0f});
        world.step(dt);
        return world.body_state(box);
    };
    if(!equivalent(public_body(ExecutionMode::NativeCpu,0),
                   public_body(ExecutionMode::NativeCpu,1)))
        throw std::runtime_error(
            "World NativeCpu rejected or skipped zero requested velocity passes");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    if(!equivalent(public_body(ExecutionMode::ReferencePhysXCpuLegacyGu,0),
                   public_body(ExecutionMode::ReferencePhysXCpuLegacyGu,1)))
        throw std::runtime_error(
            "Reference PhysX CPU zero/one velocity iteration contract changed");
#endif
}

void verify_source_rigid_contact_limits() {
    using fyzz::native::Scene;
    using fyzz::Material;
    using fyzz::RigidBodyDynamics;
    constexpr float dt=1.0f/120.0f;
    // Each source SolverContactPoint has its OWN impulse clamp, and
    // Dy writeback must retain the corresponding original Gu order.
    const auto impulse_case=[&](float cap) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        Material no_friction{};
        no_friction.static_friction=0;
        no_friction.dynamic_friction=0;
        no_friction.restitution=0;
        scene.set_material(floor,no_friction);
        scene.set_material(box,no_friction);
        RigidBodyDynamics d{};
        d.max_contact_impulse=cap;
        scene.set_dynamics(box,d);
        scene.set_velocity(box,{0,-5.0f,0});
        scene.step(dt);
        const auto contacts=scene.contacts();
        if(contacts.size()!=1 || contacts[0].count<2)
            throw std::runtime_error(
                "Source impulse-limit fixture lacks a multipoint Gu manager");
        double sum=0;
        for(std::size_t i=0;i<contacts[0].count;++i) {
            const double impulse=scene.solved_normal_impulses()[
                contacts[0].contact_offset+i];
            if(!std::isfinite(impulse) || impulse<0.0 ||
               impulse>static_cast<double>(cap)+1.0e-6)
                throw std::runtime_error(
                    "Source per-contact impulse clamp exceeded body limit");
            sum+=impulse;
        }
        if(!scene.contact_edge_connected({floor,box}))
            throw std::runtime_error(
                "Source impulse limit severed persistent accurate edge");
        return std::pair{scene.body_state(box),sum};
    };
    const auto capped=impulse_case(0.05f);
    const auto unrestricted=impulse_case(
        std::numeric_limits<float>::max());
    if(!(capped.second>0.05+1.0e-5 &&
         capped.second<=0.2+1.0e-6 &&
         unrestricted.second>capped.second*4.0 &&
         capped.first.velocity.y<unrestricted.first.velocity.y-1.0f))
        throw std::runtime_error(
            "Source body impulse cap did not restrict multipoint response");
    const auto disabled=impulse_case(0.0f);
    if(disabled.second!=0.0 ||
       !(disabled.first.velocity.y<-4.9f))
        throw std::runtime_error(
            "Source zero maxContactImpulse did not omit Dy contact response");

    // A dynamic/dynamic manager takes the SMALLER of the two per-body
    // caps, independent of which actor is first in the canonical pair.
    const auto dynamic_pair=[&](float lower_cap,float upper_cap) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto lower=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto upper=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,1.49f,0},1.0f);
        RigidBodyDynamics d{};
        d.max_contact_impulse=lower_cap;
        scene.set_dynamics(lower,d);
        d.max_contact_impulse=upper_cap;
        scene.set_dynamics(upper,d);
        scene.set_velocity(upper,{0,-3.0f,0});
        scene.step(dt);
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].count<1)
            throw std::runtime_error(
                "Source two-body contact cap fixture lacks a live Gu pair");
        double total=0.0;
        for(std::size_t i=0;i<scene.contacts()[0].count;++i) {
            const double impulse=scene.solved_normal_impulses()[
                scene.contacts()[0].contact_offset+i];
            if(!std::isfinite(impulse) || impulse<0 ||
               impulse>std::min(lower_cap,upper_cap)+1.0e-6)
                throw std::runtime_error(
                    "Source two-body minimum contact cap not enforced per point");
            total+=impulse;
        }
        return total;
    };
    const auto loose_pair=dynamic_pair(0.09f,0.07f);
    const auto tight_pair=dynamic_pair(0.02f,0.09f);
    if(!(loose_pair>tight_pair && tight_pair>0.0))
        throw std::runtime_error(
            "Source dynamic pair did not apply minimum body contact cap");

    // Source penBiasClamp is NEGATIVE maxDepenetrationVelocity.
    // Under deep penetration, limiting either actor must limit biased
    // position motion, without clamping the ordinary final velocity.
    const auto bias_case=[&](float cap) {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.25f,0},1.0f);
        Material mat{};
        mat.restitution=0;
        mat.static_friction=0;
        mat.dynamic_friction=0;
        scene.set_material(floor,mat);
        scene.set_material(box,mat);
        RigidBodyDynamics d{};
        d.max_depenetration_velocity=cap;
        scene.set_dynamics(box,d);
        // PxsBodyCore owns this clamp on dynamic rigid bodies.
        // The static world solver row has the source unrestricted cap.
        scene.step(dt);
        return scene.body_state(box);
    };
    const auto limited=bias_case(0.25f);
    const auto wide=bias_case(1.0e32f);
    if(!(std::isfinite(limited.position.y) &&
         std::isfinite(wide.position.y) &&
         wide.position.y>limited.position.y+0.02f))
        throw std::runtime_error(
            "Source rigid depenetration limit did not clamp position bias");

    // Reject invalid authored inputs at the public contract boundary.
    fyzz::World world(fyzz::WorldOptions{
        fyzz::ExecutionMode::NativeCpu});
    const auto id=world.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,5,0},1.0f);
    auto rejected=[&](RigidBodyDynamics d) {
        try {world.set_body_dynamics(id,d);}
        catch(const std::invalid_argument&) {return true;}
        return false;
    };
    RigidBodyDynamics bad{};
    bad.max_depenetration_velocity=0;
    if(!rejected(bad))
        throw std::runtime_error(
            "Source zero maxDepenetrationVelocity was admitted");
    bad=RigidBodyDynamics{};
    bad.max_contact_impulse=-1;
    if(!rejected(bad))
        throw std::runtime_error(
            "Source negative maxContactImpulse was admitted");
    auto zero=RigidBodyDynamics{};
    zero.max_contact_impulse=0;
    world.set_body_dynamics(id,zero);

#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    // The isolated source reference consumes the same real
    // PxRigidBody::setMaxContactImpulse/DepenetrationVelocity API.
    // Compare qualitative source behavior, without pretending PCM and
    // legacy Gu have already reached exact multiframe CPU parity.
    const auto reference_case=[](float impulse_cap,float pen_cap) {
        fyzz::World reference(fyzz::WorldOptions{
            fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu});
        reference.add_static_floor(0.0f);
        const auto box=reference.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.25f,0},1.0f);
        RigidBodyDynamics d{};
        d.max_contact_impulse=impulse_cap;
        d.max_depenetration_velocity=pen_cap;
        reference.set_body_dynamics(box,d);
        reference.step(dt);
        return reference.body_state(box);
    };
    const auto reference_limited=reference_case(
        std::numeric_limits<float>::max(),0.25f);
    const auto reference_wide=reference_case(
        std::numeric_limits<float>::max(),1.0e32f);
    if(!(std::isfinite(reference_limited.position.y) &&
         std::isfinite(reference_wide.position.y) &&
         reference_wide.position.y>reference_limited.position.y))
        throw std::runtime_error(
            "Source PhysX reference depenetration cap ignored authoring");
    const auto reference_impulse_case=[](float cap) {
        fyzz::World reference(fyzz::WorldOptions{
            fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu});
        reference.add_static_floor(0.0f);
        const auto box=reference.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        RigidBodyDynamics d{};
        d.max_contact_impulse=cap;
        reference.set_body_dynamics(box,d);
        reference.set_body_velocity(box,{0,-5.0f,0});
        reference.step(dt);
        return reference.body_state(box).velocity.y;
    };
    const float source_disabled=reference_impulse_case(0.0f);
    const float source_capped=reference_impulse_case(0.05f);
    const float source_unlimited=reference_impulse_case(
        std::numeric_limits<float>::max());
    if(!(std::isfinite(source_disabled) &&
         std::isfinite(source_capped) &&
         std::isfinite(source_unlimited) &&
         source_disabled<source_capped &&
         source_capped<source_unlimited))
        throw std::runtime_error(
            "Real PhysX CPU reference contact impulse limits lack ordered response");
#endif
}

void verify_source_sleep_wake_generator_matrix() {
    using namespace fyzz;
    // Sleep/wake acceptance was previously confined to default PCM.
    // Exercise the same Sc island deactivation/reactivation through
    // BOTH original CPU contact generators, with body/shape admission
    // ordering reversed and unequal dynamic masses.
    struct Sample {
        BodyState settled{},woken{};
        std::array<int,4> events{};
    };
    const auto run=[](ExecutionMode mode,bool pcm,bool reversed) {
        WorldOptions opts{mode};
        opts.contact_notifications=true;
        opts.pcm_plane_box=pcm;
        opts.pcm_box_box=pcm;
        World world(opts);
        BodyId floor=0,body=0;
        const auto add_floor=[&] {floor=world.add_static_floor(0.0f);};
        const auto add_body=[&] {
            body=world.add_dynamic_box({0.5f,0.5f,0.5f},
                                       {0,0.5f,0},
                                       reversed?2.3f:0.7f);
        };
        if(reversed){add_body();add_floor();}
        else{add_floor();add_body();}
        Material material{};
        material.restitution=0.0f;
        material.static_friction=0.5f;
        material.dynamic_friction=0.5f;
        world.set_body_material(floor,material);
        world.set_body_material(body,material);
        constexpr float dt=1.0f/120.0f;
        for(int frame=0;frame<360;++frame)
            world.step(dt);
        Sample sample{};
        sample.settled=world.body_state(body);
        if(!sample.settled.sleeping)
            throw std::runtime_error(
                "Original rigid island failed to sleep after 360 settled frames");
        const auto record=[&](std::size_t index) {
            const auto events=world.contact_events();
            if(events.size()>1 ||
               (!events.empty() &&
                (events[0].body0!=std::min(floor,body) ||
                 events[0].body1!=std::max(floor,body))))
                throw std::runtime_error(
                    "Sleep/wake source contact reporting changed actor ownership");
            sample.events[index]=events.empty()?-1:
                static_cast<int>(events[0].kind);
        };
        world.step(dt);
        record(0);
        world.set_body_velocity(body,{0.8f,0,0});
        world.step(dt);
        sample.woken=world.body_state(body);
        record(1);
        if(sample.woken.sleeping)
            throw std::runtime_error(
                "Sc wake did not reactivate a resting rigid contact island");
        world.set_body_position(body,{0,4,0});
        world.step(dt);
        record(2);
        world.step(dt);
        record(3);
        const std::array<int,4> expected{
            -1,static_cast<int>(ContactEventKind::Persists),
            static_cast<int>(ContactEventKind::Lost),-1};
        if(sample.events!=expected)
            throw std::runtime_error(
                "Source sleep/wake contact notifications violated Sc lifetime");
        return sample;
    };
    const auto max_error=[](Vec3 a,Vec3 b) {
        return std::max({std::abs(a.x-b.x),std::abs(a.y-b.y),
                         std::abs(a.z-b.z)});
    };
    for(const bool pcm:{false,true})
        for(const bool reversed:{false,true}) {
            const auto native=run(ExecutionMode::NativeCpu,pcm,reversed);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
            const auto source=run(pcm
                ? ExecutionMode::ReferencePhysXCpu
                : ExecutionMode::ReferencePhysXCpuLegacyGu,pcm,reversed);
            if(native.events!=source.events ||
               max_error(native.settled.position,
                         source.settled.position)>0.01f ||
               max_error(native.woken.position,
                         source.woken.position)>0.03f ||
               max_error(native.woken.velocity,
                         source.woken.velocity)>0.10f)
                throw std::runtime_error(
                    "Native/PhysX CPU sleep/wake parity diverged by contact generator");
#endif
        }
}

void verify_source_combined_lifecycle_matrix() {
    using namespace fyzz;
    // One real World and one independent PhysX CPU World per cell.
    // Preserve original mass/ActorID ordering across the SAME sequence:
    // 1D constraint admission -> multi-dt solver -> joint retirement ->
    // frictional dynamic pair -> shape-contact retirement. This probes
    // source Sc/Gu/Dy handoffs, not isolated arithmetic helper functions.
    struct Frame {
        BodyState left{},right{};
        int event=-1;
        std::size_t points=0;
    };
    const auto run=[](ExecutionMode mode,bool pcm,bool reversed) {
        WorldOptions opts{mode};
        opts.contact_notifications=true;
        opts.pcm_plane_box=pcm;
        opts.pcm_box_box=pcm;
        World world(opts);
        const Vec3 half{0.4f,0.4f,0.4f};
        const Vec3 left_position{0,6,0};
        const Vec3 right_position{1.8f,6,0};
        BodyId left=0,right=0;
        const auto add_left=[&] {
            left=world.add_dynamic_box(
                half,left_position,reversed?2.3f:0.7f);
        };
        const auto add_right=[&] {
            right=world.add_dynamic_box(
                half,right_position,reversed?0.7f:2.3f);
        };
        if(reversed){add_right();add_left();}
        else{add_left();add_right();}
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0.0f;
        d.angular_damping=0.0f;
        world.set_body_dynamics(left,d);
        world.set_body_dynamics(right,d);
        Material m{};
        m.static_friction=0.75f;
        m.dynamic_friction=0.6f;
        m.restitution=0.0f;
        world.set_body_material(left,m);
        world.set_body_material(right,m);
        world.set_body_velocity(left,{0.4f,0,0});
        world.set_body_velocity(right,{-0.15f,0,0});
        const auto constraint=world.add_center_distance_constraint(
            left,right,1.35f);
        std::array<Frame,12> observations{};
        for(std::size_t frame=0;frame<observations.size();++frame) {
            if(frame==6)world.remove_constraint(constraint);
            if(frame==7) {
                world.set_body_position(left,left_position);
                world.set_body_position(right,{0.72f,6,0});
                world.set_body_velocity(left,{0,0,0.4f});
                world.set_body_velocity(right,{0,0,-0.3f});
            }
            if(frame==9)
                world.set_body_position(right,{5.0f,6,0});
            const float dt=frame%3==0 ? 1.0f/90.0f :
                frame%3==1 ? 1.0f/240.0f : 1.0f/120.0f;
            world.step(dt);
            auto& result=observations[frame];
            result.left=world.body_state(left);
            result.right=world.body_state(right);
            const auto events=world.contact_events();
            if(events.size()>1)
                throw std::runtime_error(
                    "Source integrated lifecycle reported extra shape interactions");
            if(!events.empty()) {
                if(events[0].body0!=std::min(left,right) ||
                   events[0].body1!=std::max(left,right))
                    throw std::runtime_error(
                        "Source integrated lifecycle reported wrong actor pair");
                result.event=static_cast<int>(events[0].kind);
                result.points=events[0].contact_count;
            }
            if(result.left.sleeping || result.right.sleeping)
                throw std::runtime_error(
                    "Live Sc constraint/contact lifecycle unexpectedly deactivated");
        }
        return observations;
    };
    const auto max_component=[](Vec3 a,Vec3 b) {
        return std::max({std::abs(a.x-b.x),std::abs(a.y-b.y),
                         std::abs(a.z-b.z)});
    };
    const auto validate=[&](const std::array<Frame,12>& native,
                            bool pcm,bool reversed) {
        for(std::size_t frame=0;frame<native.size();++frame) {
            const int expected=frame==7
                ? static_cast<int>(ContactEventKind::Found)
                : frame==8
                    ? static_cast<int>(ContactEventKind::Persists)
                    : frame==9
                        ? static_cast<int>(ContactEventKind::Lost):-1;
            if(native[frame].event!=expected ||
               (expected==static_cast<int>(ContactEventKind::Lost)
                   ? native[frame].points!=0
                   : expected!=-1 && native[frame].points==0))
                throw std::runtime_error(
                    "Sc/Gu/Dy integrated contact lifecycle differs at frame "+
                    std::to_string(frame)+", pcm="+std::to_string(pcm)+
                    ", reversed="+std::to_string(reversed));
        }
        if(!(native[0].left.position.x>0.0f) ||
           !(native[0].right.position.x<1.8f))
            throw std::runtime_error(
                "Combined 1D fixture never exercised source position PGS");
    };
    for(const bool pcm:{false,true})
        for(const bool reversed:{false,true}) {
            const auto native=run(ExecutionMode::NativeCpu,pcm,reversed);
            validate(native,pcm,reversed);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
            const auto source=run(pcm
                ? ExecutionMode::ReferencePhysXCpu
                : ExecutionMode::ReferencePhysXCpuLegacyGu,pcm,reversed);
            for(std::size_t frame=0;frame<native.size();++frame) {
                if(native[frame].event!=source[frame].event)
                    throw std::runtime_error(
                        "Original Sc contact event ordering diverged in "
                        "the combined source lifecycle at frame "+
                        std::to_string(frame));
                for(const auto side:{0,1}) {
                    const auto& a=side?native[frame].right:native[frame].left;
                    const auto& b=side?source[frame].right:source[frame].left;
                    const float pos=max_component(a.position,b.position);
                    const float vel=max_component(a.velocity,b.velocity);
                    const float ang=max_component(
                        a.angular_velocity,b.angular_velocity);
                    const float dot=std::abs(
                        a.orientation.x*b.orientation.x+
                        a.orientation.y*b.orientation.y+
                        a.orientation.z*b.orientation.z+
                        a.orientation.w*b.orientation.w);
                    if(!std::isfinite(pos) || !std::isfinite(vel) ||
                       !std::isfinite(ang) || !std::isfinite(dot) ||
                       pos>0.025f || vel>0.075f || ang>0.10f ||
                       dot<0.999f) {
                        std::cerr<<"N1 Sc/Gu/Dy matrix"
                            <<" generator="<<(pcm?"PCM":"Gu")
                            <<" reversed="<<reversed
                            <<" frame="<<frame<<" body="<<side
                            <<" position="<<pos<<" velocity="<<vel
                            <<" angular="<<ang<<" dot="<<dot<<'\n';
                        throw std::runtime_error(
                            "Native coupled source lifecycle diverged from PhysX");
                    }
                }
            }
#endif
        }
}

void verify_source_multiframe_completion() {
    using namespace fyzz;
    using fyzz::native::Scene;
    constexpr float dt=1.0f/120.0f;
    // A Gu collision clock is not the scene's completed Dy/fetch clock.
    // The marker must survive a failed solve with the actual previously
    // published result buffers and become invalid on a collision-only
    // query, even when that query does not change contact geometry.
    {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        if(scene.solved_result_frame()!=0)
            throw std::runtime_error(
                "Scene exposed a completed source solver result before its first step");
        scene.step(dt);
        const auto committed=scene.solved_result_frame();
        if(committed==0 || committed!=scene.frame_index() ||
           scene.solved_normal_impulses().empty())
            throw std::runtime_error(
                "Dy result was published before solver and Gu force writeback completed");
        const auto* const forces=scene.solved_normal_impulses().data();
        const std::vector<double> old_forces(
            scene.solved_normal_impulses().begin(),
            scene.solved_normal_impulses().end());
        Material invalid{};
        invalid.static_friction=1.0e30f;
        invalid.dynamic_friction=1.0e30f;
        invalid.friction_combine=MaterialCombineMode::Multiply;
        scene.set_material(floor,invalid);
        scene.set_material(box,invalid);
        bool rejected=false;
        try { scene.step(dt); }
        catch(const std::overflow_error&) { rejected=true; }
        if(!rejected || scene.frame_index()!=committed ||
           scene.solved_result_frame()!=committed ||
           scene.solved_normal_impulses().data()!=forces ||
           scene.solved_normal_impulses().size()!=old_forces.size() ||
           !std::equal(old_forces.begin(),old_forces.end(),
                       scene.solved_normal_impulses().begin()))
            throw std::runtime_error(
                "Rejected Dy step advanced completed-result generation or replaced its force storage");
        scene.set_material(floor,Material{});
        scene.set_material(box,Material{});
        scene.advance_collision();
        if(scene.solved_result_frame()!=0 ||
           !scene.solved_normal_impulses().empty())
            throw std::runtime_error(
                "Gu-only collision incorrectly retained the completed solver result");
        scene.step(dt);
        if(scene.solved_result_frame()!=scene.frame_index() ||
           scene.solved_result_frame()<=committed)
            throw std::runtime_error(
                "Native Dy completion did not republish after collision-only invalidation");
    }

    // ScPipeline::afterIntegration restores integrated poses ONLY when
    // island generation can deactivate a body concurrently with Dy.
    // The admitted serial CPU path completes third-pass deactivation
    // before Dy body admission: the exact equivalent is zero integration
    // on the deactivation frame, followed by an explicit wake handoff.
    {
        Scene serial{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=serial.add_static_floor(0.0f);
        const auto sleeper=serial.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        Material no_friction{};
        no_friction.static_friction=0.0f;
        no_friction.dynamic_friction=0.0f;
        no_friction.restitution=0.0f;
        serial.set_material(floor,no_friction);
        serial.set_material(sleeper,no_friction);
        bool deactivated=false;
        for(int frame=0;frame<600;++frame) {
            const bool previously_ready=serial.body_ready_for_sleep(sleeper);
            const BodyState before=serial.body_state(sleeper);
            serial.step(dt);
            if(!serial.body_is_sleeping(sleeper))continue;
            const auto after=serial.body_state(sleeper);
            const auto& deactivations=serial.deactivating_accurate_nodes();
            if(!previously_ready ||
               after.position.x!=before.position.x ||
               after.position.y!=before.position.y ||
               after.position.z!=before.position.z ||
               after.orientation.x!=before.orientation.x ||
               after.orientation.y!=before.orientation.y ||
               after.orientation.z!=before.orientation.z ||
               after.orientation.w!=before.orientation.w ||
               after.velocity.x!=0.0f || after.velocity.y!=0.0f ||
               after.velocity.z!=0.0f ||
               after.angular_velocity.x!=0.0f ||
               after.angular_velocity.y!=0.0f ||
               after.angular_velocity.z!=0.0f ||
               !serial.active_accurate_nodes().empty() ||
               !serial.indexed_contact_manager_order().empty() ||
               deactivations.size()!=1 || deactivations[0]!=sleeper ||
               serial.solved_result_frame()!=serial.frame_index())
                throw std::runtime_error(
                    "Serial Dy integration moved a rigid body deactivated by the previous island third pass");
            deactivated=true;
            break;
        }
        if(!deactivated)
            throw std::runtime_error(
                "Multiframe serial source deactivation did not reach a completed sleep publication");
        const BodyState sleeping=serial.body_state(sleeper);
        serial.step(dt);
        const auto dormant=serial.body_state(sleeper);
        if(!dormant.sleeping ||
           dormant.position.y!=sleeping.position.y ||
           !serial.active_accurate_nodes().empty() ||
           !serial.deactivating_accurate_nodes().empty())
            throw std::runtime_error(
                "Next completed frame re-integrated a dormant source rigid body");
        serial.set_velocity(sleeper,{2.0f,0.0f,0.0f});
        serial.step(dt);
        if(serial.body_is_sleeping(sleeper) ||
           serial.active_accurate_nodes().size()!=1 ||
           serial.body_state(sleeper).position.x<=
               sleeping.position.x+0.005f ||
           serial.solved_result_frame()!=serial.frame_index())
            throw std::runtime_error(
                "Authoritative source wake did not readmit sleeping rigid body to Dy");
    }

    // Multi-frame free-body comparison against the ORIGINAL PhysX CPU
    // implementation. Disparate masses, rotations, damping and changing
    // dt all pass through the real public World backend. No Gu contact
    // is possible; drift here belongs to body integration, not PCM.
    const auto freeflight=[](ExecutionMode mode) {
        World world(WorldOptions{mode});
        const auto first=world.add_dynamic_box(
            {0.4f,0.7f,0.3f},{-6,25,0},0.75f);
        const auto second=world.add_dynamic_box(
            {0.8f,0.3f,0.6f},{6,28,0},3.5f);
        RigidBodyDynamics a{},b{};
        a.linear_damping=0.12f;
        a.angular_damping=0.09f;
        b.linear_damping=0.025f;
        b.angular_damping=0.16f;
        a.position_iterations=5;
        b.position_iterations=8;
        world.set_body_dynamics(first,a);
        world.set_body_dynamics(second,b);
        world.set_body_velocity(first,{0.85f,-0.4f,0.15f});
        world.set_body_velocity(second,{-0.35f,0.2f,-0.8f});
        world.set_body_angular_velocity(first,{0.4f,0.9f,-0.2f});
        world.set_body_angular_velocity(second,{0.6f,-0.3f,0.7f});
        std::vector<std::array<BodyState,2>> frames;
        frames.reserve(96);
        for(unsigned i=0;i<96;++i) {
            const float timestep=i%3==0?1.0f/90.0f:
                (i%3==1?1.0f/240.0f:1.0f/120.0f);
            world.step(timestep);
            const std::array<BodyState,2> state{
                world.body_state(first),world.body_state(second)};
            for(const auto& body:state) {
                const auto finite=[](Vec3 v) {
                    return std::isfinite(v.x) &&
                           std::isfinite(v.y) && std::isfinite(v.z);
                };
                const auto& q=body.orientation;
                const float q2=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
                if(!finite(body.position) || !finite(body.velocity) ||
                   !finite(body.angular_velocity) ||
                   !std::isfinite(q2) || std::abs(q2-1.0f)>0.01f ||
                   body.sleeping)
                    throw std::runtime_error(
                        "Multiframe PhysX integration lost a live rigid-body state");
            }
            frames.push_back(state);
        }
        return frames;
    };
    const auto native=freeflight(ExecutionMode::NativeCpu);

    // More than one frame of actual contact lifetime, with alternating
    // timesteps and repeated Found/Persists/Lost transitions. Zero
    // contact response deliberately isolates Gu/Sc publication ordering
    // from the still separately gated strict PGS impulse parity.
    const auto contact_timeline=[](ExecutionMode mode) {
        WorldOptions opts{mode};
        opts.contact_notifications=true;
        World world(opts);
        const auto floor=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        dynamics.max_contact_impulse=0.0f;
        world.set_body_dynamics(box,dynamics);
        std::vector<int> events;
        events.reserve(40);
        for(unsigned cycle=0;cycle<10;++cycle) {
            for(unsigned phase=0;phase<4;++phase) {
                if(phase==0)world.set_body_position(box,{0,0.49f,0});
                if(phase==2)world.set_body_position(box,{0,4,0});
                world.step((cycle+phase)%2?1.0f/120.0f:1.0f/240.0f);
                const auto reports=world.contact_events();
                if(reports.size()>1 ||
                   (!reports.empty() &&
                    (reports[0].body0!=floor || reports[0].body1!=box)))
                    throw std::runtime_error(
                        "Sc multiframe touch lifecycle reported an unrelated contact");
                const int kind=reports.empty()?-1:
                    static_cast<int>(reports[0].kind);
                const int expected=phase==0
                    ?static_cast<int>(ContactEventKind::Found)
                    :phase==1?static_cast<int>(ContactEventKind::Persists)
                    :phase==2?static_cast<int>(ContactEventKind::Lost):-1;
                if(kind!=expected ||
                   (!reports.empty() && phase!=2 &&
                    (reports[0].contact_count==0 ||
                     reports[0].normal_impulse!=0.0)))
                    throw std::runtime_error(
                        "Sc multiframe zero-response touch or fetch order diverged at cycle "+
                        std::to_string(cycle)+", phase "+
                        std::to_string(phase));
                events.push_back(kind);
            }
        }
        return events;
    };
    const auto native_events=contact_timeline(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto reference=freeflight(ExecutionMode::ReferencePhysXCpuLegacyGu);
    for(std::size_t frame=0;frame<native.size();++frame)
    for(std::size_t body=0;body<2;++body) {
        const auto& a=native[frame][body];
        const auto& b=reference[frame][body];
        const auto delta=[](Vec3 x,Vec3 y) {
            return std::max({
                std::abs(x.x-y.x),std::abs(x.y-y.y),
                std::abs(x.z-y.z)});
        };
        const float location_error=delta(a.position,b.position);
        const float velocity_error=delta(a.velocity,b.velocity);
        const float angular_error=delta(a.angular_velocity,b.angular_velocity);
        const auto& q=a.orientation;
        const auto& p=b.orientation;
        const float q_dot=std::abs(
            q.x*p.x+q.y*p.y+q.z*p.z+q.w*p.w);
        if(location_error>0.004f || velocity_error>0.004f ||
           angular_error>0.004f || !std::isfinite(q_dot) ||
           1.0f-q_dot>0.003f)
            throw std::runtime_error(
                "Multiframe native/PhysX CPU free-flight state diverged on frame "+
                std::to_string(frame)+", actor "+std::to_string(body)+
                " (pos "+std::to_string(location_error)+
                ", vel "+std::to_string(velocity_error)+
                ", angular "+std::to_string(angular_error)+
                ", quaternion "+std::to_string(q_dot)+")");
    }
    for(const auto mode:{ExecutionMode::ReferencePhysXCpuLegacyGu,
                         ExecutionMode::ReferencePhysXCpu}) {
        const auto source=contact_timeline(mode);
        if(source!=native_events)
            throw std::runtime_error(
                "Original PhysX and native Sc touch lifecycles diverged across 40 frames");
    }
#else
    (void)native;
    (void)native_events;
#endif
}

void verify_source_pose_authoring() {
    constexpr float dt=1.0f/120.0f;
    constexpr float sine=0.38268343236f;
    constexpr float cosine=0.92387953251f;
    const fyzz::Quat rotated{0,-sine,0,cosine};
    // A slender static box's rotation brings it into a previously
    // separated dynamic box. Exercise original Gu SAT and SAP bounds,
    // not a rotated AABB approximation or a manually injected row.
    {
        fyzz::native::Scene scene{fyzz::native::Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto fixed=scene.add_static_box(
            {1.0f,0.4f,0.15f},{0,2,0});
        const auto moving=scene.add_dynamic_box(
            {0.3f,0.3f,0.3f},{0.6f,2.0f,0.6f},1.0f);
        scene.advance_collision();
        if(!scene.pairs().empty() || !scene.contacts().empty())
            throw std::runtime_error(
                "Rotated-geometry fixture initially intersects");
        scene.set_pose(fixed,{0,2,0},rotated);
        scene.advance_collision();
        if(scene.pairs().size()!=1 || scene.contacts().size()!=1 ||
           scene.contacts()[0].count==0 ||
           !scene.contact_edge_connected({fixed,moving}))
            throw std::runtime_error(
                "Authored box rotation did not reach native Gu SAT and SAP");
        scene.set_pose(fixed,{0,2,0},{});
        scene.advance_collision();
        if(!scene.contacts().empty())
            throw std::runtime_error(
                "Reversed authored pose retained stale rotated Gu contacts");
    }
    for(const auto mode:{fyzz::ExecutionMode::NativeCpu
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
                         ,fyzz::ExecutionMode::ReferencePhysXCpuLegacyGu
                         ,fyzz::ExecutionMode::ReferencePhysXCpu
#endif
        }) {
        fyzz::World world(fyzz::WorldOptions{mode});
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,3,0},1.0f);
        const auto floor=world.add_static_floor(0.0f);
        const auto invalid=[&](fyzz::Quat q) {
            try {world.set_body_pose(box,{0,3,0},q);}
            catch(const std::invalid_argument&) {return true;}
            return false;
        };
        if(!invalid({0,0,0,0}) ||
           !invalid({0,0,0,std::numeric_limits<float>::infinity()}) ||
           !invalid({0,0,0,0.8f}))
            throw std::runtime_error(
                "PxQuat::isSane pose validation admitted invalid rotation");
        const fyzz::Quat near_unit{
            rotated.x*1.003f,rotated.y*1.003f,
            rotated.z*1.003f,rotated.w*1.003f};
        world.set_body_pose(box,{0,3,0},near_unit);
        auto pose=world.body_state(box);
        if(std::abs(pose.orientation.y-rotated.y)>2.0e-5f ||
           std::abs(pose.orientation.w-rotated.w)>2.0e-5f)
            throw std::runtime_error(
                "Source getNormalized authored quaternion was not published");
        world.set_body_position(box,{0,4,0});
        pose=world.body_state(box);
        if(std::abs(pose.orientation.y-rotated.y)>2.0e-5f ||
           pose.position.y!=4.0f)
            throw std::runtime_error(
                "Translation-only pose edit destroyed rigid orientation");
        bool plane_rejected=false;
        try {world.set_body_pose(floor,{0,0,0},rotated);}
        catch(const std::logic_error&) {plane_rejected=true;}
        if(!plane_rejected)
            throw std::runtime_error(
                "Fixed horizontal floor admitted unsupported rotation");
        world.set_body_pose(floor,{0,0.25f,0},{0,0,0,-1.0f});
        const auto signed_floor=world.body_state(floor);
        if(signed_floor.position.y!=0.25f ||
           signed_floor.orientation.x!=0.0f ||
           signed_floor.orientation.y!=0.0f ||
           signed_floor.orientation.z!=0.0f ||
           signed_floor.orientation.w!=1.0f)
            throw std::runtime_error(
                "Signed-equivalent floor identity changed its logical pose");
        world.set_body_pose(floor,{0,0,0},{0,0,0,-1.001f});
        const auto restored_floor=world.body_state(floor);
        if(restored_floor.position.y!=0.0f ||
           restored_floor.orientation.w!=1.0f)
            throw std::runtime_error(
                "Sane signed floor pose failed canonical restoration");
        world.step(dt);
        if(!std::isfinite(world.body_state(box).position.y) ||
           std::abs(world.body_state(box).orientation.y-rotated.y)>5.0e-4f)
            throw std::runtime_error(
                "Source authored rigid orientation did not survive integration");
    }
}

void verify_source_shape_offsets() {
    using fyzz::native::Scene;
    using fyzz::ExecutionMode;
    using fyzz::World;
    using fyzz::WorldOptions;
    using fyzz::RigidBodyDynamics;
    using fyzz::Material;
    constexpr float dt=1.0f/120.0f;
    // Shape-specific contact distances must affect both the SAP
    // candidate bounds and original Gu box-box contact generation.
    {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto fixed=scene.add_static_box(
            {0.5f,0.5f,0.5f},{0.0f,2.0f,0.0f});
        const auto moving=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.07f,2.0f,0.0f},1.0f);
        scene.advance_collision();
        if(!scene.pairs().empty() || !scene.contacts().empty())
            throw std::runtime_error(
                "Default box contact offsets admitted separated pair");
        scene.set_shape_offsets(moving,0.10f,0.0f);
        scene.advance_collision();
        if(scene.pairs().size()!=1 || scene.contacts().size()!=1 ||
           scene.contacts()[0].pair!=fyzz::native::Pair{fixed,moving} ||
           scene.contacts()[0].count==0 ||
           scene.found_contacts().size()!=1 ||
           !scene.contact_edge_connected({fixed,moving}))
            throw std::runtime_error(
                "Shape contact offsets did not reach SAP and Gu box contact");
        scene.set_shape_offsets(moving,0.02f,0.0f);
        scene.advance_collision();
        if(!scene.contacts().empty() || scene.lost_contacts().size()!=1 ||
           !scene.pairs().empty())
            throw std::runtime_error(
                "Restored contact offset failed to retire Gu touch and SAP pair");
    }
    // Plane-box Gu uses exactly the same pair distance, even though the
    // infinite plane enters broadphase as an unconditional candidate.
    {
        Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.56f,0},1.0f);
        scene.advance_collision();
        if(!scene.contacts().empty())
            throw std::runtime_error(
                "Plane contact appeared outside default contact distance");
        scene.set_shape_offsets(box,0.08f,0.0f);
        scene.advance_collision();
        if(scene.contacts().size()!=1 ||
           scene.contacts()[0].pair!=fyzz::native::Pair{floor,box} ||
           scene.contacts()[0].count==0)
            throw std::runtime_error(
                "Plane Gu ignored authored shape contact offset");
        scene.set_shape_offsets(box,0.02f,0.0f);
        scene.advance_collision();
        if(!scene.contacts().empty())
            throw std::runtime_error(
                "Plane Gu retained contact after offset reduction");
    }
    // Rest offset is NOT the Gu contact-generation margin: Dy subtracts
    // the pair rest distance from the already-generated separation.
    // Both independent CPU backends must respond more strongly to a
    // larger rest distance without changing contact-offset admission.
    const auto solve=[&](ExecutionMode execution,float rest) {
        World world(WorldOptions{execution});
        const auto floor=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        Material material{};
        material.static_friction=0;
        material.dynamic_friction=0;
        material.restitution=0;
        world.set_body_material(floor,material);
        world.set_body_material(box,material);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        world.set_body_dynamics(box,d);
        world.set_body_shape_offsets(box,0.08f,rest);
        world.step(dt);
        return world.body_state(box);
    };
    const auto native_zero=solve(ExecutionMode::NativeCpu,0.0f);
    const auto native_rest=solve(ExecutionMode::NativeCpu,0.035f);
    if(!std::isfinite(native_rest.position.y) ||
       !std::isfinite(native_zero.position.y) ||
       !(native_rest.position.y>native_zero.position.y+1.0e-3f))
        throw std::runtime_error(
            "Dy contact bias ignored source PxShape rest distance");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto ref_zero=solve(ExecutionMode::ReferencePhysXCpuLegacyGu,0.0f);
    const auto ref_rest=solve(ExecutionMode::ReferencePhysXCpuLegacyGu,0.035f);
    if(!std::isfinite(ref_zero.position.y) ||
       !std::isfinite(ref_rest.position.y) ||
       !(ref_rest.position.y>ref_zero.position.y+1.0e-3f))
        throw std::runtime_error(
            "Reference PhysX shape rest distance did not change contact bias");
#endif
    // Validate the full authored pair BEFORE either backend mutates it.
    for(const auto mode:{ExecutionMode::NativeCpu
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
            ,ExecutionMode::ReferencePhysXCpuLegacyGu
#endif
        }) {
        World world(WorldOptions{mode});
        const auto id=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto invalid=[&](float contact,float rest) {
            try {world.set_body_shape_offsets(id,contact,rest);}
            catch(const std::invalid_argument&) {return true;}
            return false;
        };
        if(!invalid(0.02f,0.02f) || !invalid(-0.01f,-0.02f) ||
           !invalid(std::numeric_limits<float>::infinity(),0.0f) ||
           !invalid(0.02f,std::numeric_limits<float>::quiet_NaN()))
            throw std::runtime_error(
                "Invalid source shape offset pair was admitted");
        // Both source PxShape setters validate against the *old*
        // opposite field; transitions must work in both directions.
        world.set_body_shape_offsets(id,0.10f,0.08f);
        world.set_body_shape_offsets(id,0.02f,-0.01f);
        world.step(dt);
        if(!std::isfinite(world.body_state(id).position.y))
            throw std::runtime_error(
                "Valid shape offset transition corrupted integration");
    }
}

void verify_source_compliant_world_parity() {
    using namespace fyzz;
    constexpr float dt=1.0f/60.0f;
    struct Outcome { float height, velocity; };
    const auto run=[](ExecutionMode mode, bool acceleration) {
        World world(WorldOptions{mode});
        const auto floor=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.47f,0.0f},5.0f);
        Material rigid{};
        rigid.static_friction=0.0f;
        rigid.dynamic_friction=0.0f;
        rigid.restitution=0.0f;
        Material spring=rigid;
        spring.restitution=-600.0f;
        spring.damping=4.0f;
        spring.compliant_acceleration_spring=acceleration;
        world.set_body_material(floor,rigid);
        world.set_body_material(box,spring);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        world.set_body_dynamics(box,dynamics);
        world.set_body_velocity(box,{0.0f,-0.4f,0.0f});
        world.step(dt);
        const auto state=world.body_state(box);
        if(!std::isfinite(state.position.y) ||
           !std::isfinite(state.velocity.y))
            throw std::runtime_error(
                "Compliant material corrupted World solver publication");
        return Outcome{state.position.y,state.velocity.y};
    };
    const auto force=run(ExecutionMode::NativeCpu,false);
    const auto acceleration=run(ExecutionMode::NativeCpu,true);
    if(!(force.velocity>-0.4f && acceleration.velocity>force.velocity &&
         force.height>0.45f && acceleration.height>force.height))
        throw std::runtime_error(
            "Native compliant spring did not distinguish source force/acceleration");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto ref_force=run(ExecutionMode::ReferencePhysXCpuLegacyGu,false);
    const auto ref_acceleration=run(ExecutionMode::ReferencePhysXCpuLegacyGu,true);
    // Geometry and solver phase should agree in direction and magnitude.
    // Strict matching of the original packed SIMD rounding is separately
    // gated by the N1 source descriptor admission criteria.
    const auto compare=[](Outcome a,Outcome b) {
        return std::abs(a.velocity-b.velocity)<0.2f &&
               std::abs(a.height-b.height)<0.025f;
    };
    if(!(ref_force.velocity>-0.4f &&
         ref_acceleration.velocity>ref_force.velocity &&
         compare(force,ref_force) &&
         compare(acceleration,ref_acceleration)))
        throw std::runtime_error(
            "Compliant PGS World differs from independent PhysX CPU oracle");
#endif
    World check(WorldOptions{ExecutionMode::NativeCpu});
    const auto id=check.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,4,0},1.0f);
    Material bad{};
    bad.damping=-1.0f;
    bool rejected=false;
    try {check.set_body_material(id,bad);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected)
        throw std::runtime_error(
            "Invalid source compliant damping was accepted");
    bad.damping=0.0f;
    bad.restitution=-200.0f;
    check.set_body_material(id,bad);
    check.step(dt);
    if(!std::isfinite(check.body_state(id).position.y))
        throw std::runtime_error(
            "Compliant material incorrectly disabled free-flight integration");
}


void verify_kinematic_anchored_constraint_handoff() {
    using namespace fyzz;
    struct Result {BodyState dynamic{},kinematic{};};
    const auto run=[](ExecutionMode mode) {
        World world(WorldOptions{mode});
        const auto dynamic=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto kinematic=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,7,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0;
        d.angular_damping=0;
        world.set_body_dynamics(dynamic,d);
        world.set_body_kinematic(kinematic,true);
        const auto joint=world.add_anchored_distance_constraint(
            dynamic,kinematic,{0,0.25f,0},{0,-0.2f,0},1.5f);
        const Quat target_q{0,0,std::sin(0.12f),std::cos(0.12f)};
        world.set_body_kinematic_target(
            kinematic,{2.2f,7.15f,0},target_q);
        world.step(1.0f/120.0f);
        const Result result{
            world.body_state(dynamic),world.body_state(kinematic)};
        if(!(result.dynamic.position.x>0.01f) ||
           std::abs(result.dynamic.orientation.z)<0.001f ||
           std::abs(result.kinematic.position.x-2.2f)>1.0e-5f ||
           std::abs(result.kinematic.orientation.z-target_q.z)>1.0e-5f)
            throw std::runtime_error(
                "Source kinematic/1D constraint postSolver handoff failed");
        world.remove_constraint(joint);
        return result;
    };
    const auto native=run(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto reference=run(ExecutionMode::ReferencePhysXCpu);
    for(const auto& pair:{
        std::pair{native.dynamic.position.x,reference.dynamic.position.x},
        std::pair{native.dynamic.orientation.z,reference.dynamic.orientation.z},
        std::pair{native.dynamic.velocity.x,reference.dynamic.velocity.x},
        std::pair{native.kinematic.position.x,reference.kinematic.position.x}
    }) if(std::abs(pair.first-pair.second)>0.03f)
        throw std::runtime_error(
            "Source anchored kinematic/rigid solver differs from PhysX");
#endif
}

void verify_anchored_distance_constraint_angular_jacobians() {
    using namespace fyzz;
    struct Result {BodyState a{},b{};};
    const auto run=[](ExecutionMode mode) {
        World world(WorldOptions{mode});
        const auto a=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,7,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0;
        d.angular_damping=0;
        world.set_body_dynamics(a,d);
        world.set_body_dynamics(b,d);
        const auto joint=world.add_anchored_distance_constraint(
            a,b,{0,0.35f,0},{0,-0.25f,0},1.4f);
        world.step(1.0f/120.0f);
        const Result result{world.body_state(a),world.body_state(b)};
        if(!std::isfinite(result.a.angular_velocity.z) ||
           !std::isfinite(result.b.angular_velocity.z) ||
           std::abs(result.a.orientation.z)<0.002f ||
           std::abs(result.b.orientation.z)<0.002f ||
           !(result.a.position.x>0 && result.b.position.x<2)) {
            std::cerr<<"Anchored 1D mode="<<static_cast<int>(mode)
                <<" pa="<<result.a.position.x
                <<" pb="<<result.b.position.x
                <<" qa="<<result.a.orientation.z
                <<" qb="<<result.b.orientation.z
                <<" wa="<<result.a.angular_velocity.z
                <<" wb="<<result.b.angular_velocity.z<<'\n';
            throw std::runtime_error(
                "Anchored source 1D joint omitted position-solver angular motion");
        }
        world.remove_constraint(joint);
        return result;
    };
    const auto native=run(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto original=run(ExecutionMode::ReferencePhysXCpu);
    for(const auto& pair:{
        std::pair{native.a.position.x,original.a.position.x},
        std::pair{native.b.position.x,original.b.position.x},
        std::pair{native.a.orientation.z,original.a.orientation.z},
        std::pair{native.b.orientation.z,original.b.orientation.z},
        std::pair{native.a.angular_velocity.z,original.a.angular_velocity.z},
        std::pair{native.b.angular_velocity.z,original.b.angular_velocity.z}
    }) if(std::abs(pair.first-pair.second)>0.02f)
        throw std::runtime_error(
            "Anchored rigid 1D Jacobian diverged from stock PhysX");
#endif
}

void verify_public_center_distance_constraint_lifecycle() {
    using namespace fyzz;
    struct Observation {
        BodyState a{},b{};
    };
    const auto run=[](ExecutionMode mode) -> Observation {
        World world(WorldOptions{mode});
        const auto a=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,7,0},1.0f);
        const auto b=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{2,7,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        d.linear_damping=0;
        d.angular_damping=0;
        world.set_body_dynamics(a,d);
        world.set_body_dynamics(b,d);
        bool bad=false;
        try {world.add_center_distance_constraint(a,a,1.0f);}
        catch(const std::invalid_argument&){bad=true;}
        if(!bad)throw std::runtime_error(
            "Public constraint allowed self-coupled rigid actors");
        const auto id=world.add_center_distance_constraint(a,b,1.0f);
        world.step(1.0f/120.0f);
        const Observation first{world.body_state(a),world.body_state(b)};
        if(!(first.a.position.x>0 && first.b.position.x<2) ||
           !(first.a.velocity.x<0.001f && first.a.velocity.x>-0.001f) ||
           !std::isfinite(first.b.velocity.x))
            throw std::runtime_error(
                "Public distance constraint did not execute source PGS");
        world.remove_constraint(id);
        bool stale=false;
        try {world.remove_constraint(id);}
        catch(const std::out_of_range&){stale=true;}
        if(!stale)throw std::runtime_error(
            "Retired public constraint accepted duplicate release");
        world.set_body_position(b,{6,7,0});
        world.step(1.0f/120.0f);
        if(std::abs(world.body_state(b).position.x-6.0f)>1.0e-4f)
            throw std::runtime_error(
                "Retired constraint continued influencing Dy motion");

        // Destruction of an incident body must also retire source
        // ScConstraintSim/PxDistanceJoint ownership.
        const auto second_id=world.add_center_distance_constraint(a,b,2.0f);
        if(second_id==id)
            throw std::runtime_error(
                "Recycled ScConstraintSim reused a public constraint token");
        stale=false;
        try {world.remove_constraint(id);}
        catch(const std::out_of_range&){stale=true;}
        if(!stale)
            throw std::runtime_error(
                "Stale public constraint ID removed a replacement");
        world.remove_body(b);
        stale=false;
        try {world.remove_constraint(second_id);}
        catch(const std::out_of_range&){stale=true;}
        if(!stale)throw std::runtime_error(
            "Actor retirement leaked a public constraint");
        return first;
    };
    const Observation native=run(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const Observation source=run(ExecutionMode::ReferencePhysXCpu);
    for(const auto& [x,y]:{
        std::pair{native.a.position.x,source.a.position.x},
        std::pair{native.b.position.x,source.b.position.x},
        std::pair{native.a.velocity.x,source.a.velocity.x},
        std::pair{native.b.velocity.x,source.b.velocity.x}
    }) if(std::abs(x-y)>1.0e-3f)
        throw std::runtime_error(
            "World center-distance PGS diverged from original PhysX");
#endif
}

void verify_source_simulation_shape_flag_lifecycle() {
    using namespace fyzz;
    constexpr float dt=1.0f/120.0f;
    const auto probe=[](ExecutionMode mode) {
        WorldOptions options{mode};
        options.contact_notifications=true;
        World world(options);
        const auto floor=world.add_static_floor(0);
        const auto body=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        RigidBodyDynamics dynamics{};
        dynamics.gravity_enabled=false;
        world.set_body_dynamics(body,dynamics);
        world.step(dt);
        if(world.contact_events().size()!=1 ||
           world.contact_events()[0].kind!=ContactEventKind::Found)
            throw std::runtime_error(
                "Source shape flag fixture failed initial Sc contact admission");
        world.set_body_simulation_shape_enabled(floor,false);
        world.set_body_velocity(body,{0,-0.2f,0});
        world.step(dt);
        if(world.body_state(body).velocity.y>-0.18f)
            throw std::runtime_error(
                "Disabled source simulation shape continued solver contact");
        // Sc filtering/retirement differs from an authored actor remove.
        // Both actors and their stable BodyIds must remain available.
        if(!std::isfinite(world.body_state(floor).position.y) ||
           !std::isfinite(world.body_state(body).position.y))
            throw std::runtime_error(
                "Shape flag edit retired its owning rigid actor");
        world.set_body_simulation_shape_enabled(floor,true);
        world.step(dt);
        if(world.body_state(body).velocity.y<=-0.18f)
            throw std::runtime_error(
                "Restored source simulation shape failed contact response");
        world.set_body_simulation_shape_enabled(body,false);
        world.step(dt);
        world.set_body_simulation_shape_enabled(body,true);
        world.step(dt);
        if(!std::isfinite(world.body_state(body).velocity.y))
            throw std::runtime_error(
                "Repeated source simulation shape admission corrupted Dy");
    };
    probe(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    probe(ExecutionMode::ReferencePhysXCpu);
#endif

    // Native Sc must retain one actor, remove the former manager,
    // and admit a newly filtered pair with no orphaned EdgeInstance.
    native::Scene scene{};
    const auto floor=scene.add_static_floor(0);
    const auto body=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
    const native::Pair key{floor,body};
    scene.advance_collision(true);
    const auto prior=scene.contact_edge_index(key);
    if(prior==std::numeric_limits<std::uint32_t>::max() ||
       scene.contacts().size()!=1)
        throw std::runtime_error(
            "Native shape filter lacked its original contact manager");
    scene.set_shape_simulation_enabled(floor,false);
    scene.advance_collision(true);
    if(!scene.pairs().empty() || !scene.contacts().empty() ||
       std::find(scene.lost_pairs().begin(),scene.lost_pairs().end(),key)==
           scene.lost_pairs().end() ||
       scene.contact_edge_index(key)!=std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(
            "Shape flag suppression failed source SAP and Sc retirement");
    scene.set_shape_simulation_enabled(floor,true);
    scene.advance_collision(true);
    if(scene.contacts().size()!=1 ||
       std::find(scene.found_pairs().begin(),scene.found_pairs().end(),key)==
           scene.found_pairs().end() ||
       scene.contact_edge_index(key)==std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(
            "Shape reenable did not admit a source contact manager");
}

void verify_source_contact_notification_world() {
    using namespace fyzz;
    constexpr float dt=1.0f/120.0f;
    const auto verify=[&](ExecutionMode mode) {
        WorldOptions enabled{mode};
        enabled.contact_notifications=true;
        World world(enabled);
        const auto floor=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.49f,0.0f},1.0f);
        RigidBodyDynamics no_gravity{};
        no_gravity.gravity_enabled=false;
        world.set_body_dynamics(box,no_gravity);
        const auto only_event=[&](ContactEventKind expected) {
            const auto events=world.contact_events();
            if(events.size()!=1 ||
               events[0].body0!=floor || events[0].body1!=box ||
               events[0].kind!=expected)
                throw std::runtime_error(
                    "Sc touch notification did not match PhysX event lifecycle");
            const auto& event=events[0];
            if(!std::isfinite(event.normal_impulse) ||
               (expected==ContactEventKind::Lost &&
                (event.contact_count!=0 || event.normal_impulse!=0.0)) ||
               (expected!=ContactEventKind::Lost &&
                event.contact_count==0))
                throw std::runtime_error(
                    "Sc contact notification has invalid Gu/Dy force fields");
            // Original PxContactPair::extractContacts publishes real
            // per-point geometry and its ordered normal-force slots.
            const auto points=world.contact_event_points();
            if(event.point_offset>points.size() ||
               event.contact_count>points.size()-event.point_offset ||
               points.size()!=event.contact_count)
                throw std::runtime_error(
                    "Sc touch event did not preserve per-point report range");
            double sum=0.0;
            for(std::size_t i=0;i<event.contact_count;++i) {
                const auto& point=points[event.point_offset+i];
                const auto finite=[](Vec3 v) {
                    return std::isfinite(v.x) &&
                           std::isfinite(v.y) && std::isfinite(v.z);
                };
                if(!finite(point.position) || !finite(point.normal) ||
                   !std::isfinite(point.separation) ||
                   !std::isfinite(point.normal_impulse))
                    throw std::runtime_error(
                        "Sc event point geometry or force is nonfinite");
                // Canonical IDs are floor then box, so Gu
                // normal must face downward from box toward floor
                // in BOTH original PhysX and native Scene modes.
                if(!(point.normal.y < -0.9f &&
                     point.position.y > -0.1f &&
                     point.position.y < 0.6f))
                    throw std::runtime_error(
                        "Contact event normal/geometry disagrees with canonical shape order");
                sum+=point.normal_impulse;
            }
            if(std::abs(sum-event.normal_impulse)>1.0e-6*
               std::max({1.0,std::abs(sum),
                         std::abs(event.normal_impulse)}))
                throw std::runtime_error(
                    "Sc event point impulses disagree with original force buffer");
        };
        if(!world.contact_events().empty())
            throw std::runtime_error(
                "Contact event stream reported before first source fetch");
        world.step(dt);
        only_event(ContactEventKind::Found);
        world.step(dt);
        only_event(ContactEventKind::Persists);
        world.set_body_position(box,{0,4,0});
        world.step(dt);
        only_event(ContactEventKind::Lost);
        world.step(dt);
        if(!world.contact_events().empty())
            throw std::runtime_error(
                "Sc contact events replayed after touch was already lost");
        world.set_body_position(box,{0,0.49f,0});
        world.step(dt);
        only_event(ContactEventKind::Found);
        // A zero max-contact-impulse skips Dy entirely, but it still
        // belongs to the source Gu and Sc touch reporting lifecycle.
        no_gravity.max_contact_impulse=0.0f;
        world.set_body_dynamics(box,no_gravity);
        world.step(dt);
        only_event(ContactEventKind::Persists);
        if(world.contact_events()[0].normal_impulse!=0.0)
            throw std::runtime_error(
                "Zero-response source contact falsely reported a solved impulse");

        World disabled(WorldOptions{mode});
        const auto static_body=disabled.add_static_floor(0.0f);
        const auto dynamic_body=disabled.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        disabled.step(dt);
        if(!disabled.contact_events().empty() ||
           !disabled.contact_event_points().empty())
            throw std::runtime_error(
                "Default PxDefaultSimulationFilterShader emitted opt-in touch event");
        (void)static_body;
        (void)dynamic_body;
    };
    // The second source acceptance axis is packed event geometry:
    // distinct shape interactions must not alias point offsets even if
    // the PhysX callback order differs from native Gu manager order.
    const auto verify_multi_manager=[&](ExecutionMode mode) {
        WorldOptions opts{mode};
        opts.contact_notifications=true;
        World world(opts);
        // Two spatially separated interactions, rather than nearly
        // coplanar floors: swapping the two complete point windows
        // would still pass nonoverlap, count and force-sum checks,
        // but MUST fail the pair-local geometry check below.
        constexpr float left_x=-4.0f,right_x=4.0f;
        const auto left_floor=world.add_static_box(
            {0.5f,0.5f,0.5f},{left_x,-0.5f,0.0f});
        const auto right_floor=world.add_static_box(
            {0.5f,0.5f,0.5f},{right_x,-0.5f,0.0f});
        const auto left_box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{left_x,0.49f,0.0f},1.0f);
        const auto right_box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{right_x,0.49f,0.0f},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        world.set_body_dynamics(left_box,d);
        world.set_body_dynamics(right_box,d);
        world.step(dt);
        const auto events=world.contact_events();
        const auto points=world.contact_event_points();
        if(events.size()!=2 || points.empty())
            throw std::runtime_error(
                "Source contact callbacks lost an independent shape interaction");
        std::vector<bool> claimed(points.size(),false);
        bool saw_left=false,saw_right=false;
        for(const auto& e:events) {
            if(e.kind!=ContactEventKind::Found ||
               e.contact_count==0 || e.point_offset>points.size() ||
               e.contact_count>points.size()-e.point_offset)
                throw std::runtime_error(
                    "Source contact callback lost Gu contact point window");
            float expected_x;
            if(e.body0==left_floor && e.body1==left_box) {
                expected_x=left_x;
                saw_left=true;
            } else if(e.body0==right_floor && e.body1==right_box) {
                expected_x=right_x;
                saw_right=true;
            } else throw std::runtime_error(
                "Source event misidentified its interacting rigid-body pair");
            double sum=0.0;
            for(std::size_t i=e.point_offset;
                i<e.point_offset+e.contact_count;++i) {
                if(claimed[i])
                    throw std::runtime_error(
                        "Multiple shape interactions alias published contact points");
                claimed[i]=true;
                const auto& point=points[i];
                if(std::abs(point.position.x-expected_x)>0.55f ||
                   !(point.position.y>-0.1f && point.position.y<0.1f) ||
                   point.normal.y>-0.9f ||
                   !std::isfinite(point.separation) ||
                   !std::isfinite(point.normal_impulse))
                    throw std::runtime_error(
                        "Sc event's point range belongs to a different actor pair");
                sum+=point.normal_impulse;
            }
            if(std::abs(sum-e.normal_impulse)>1.0e-5*
               std::max({1.0,std::abs(sum),std::abs(e.normal_impulse)}))
                throw std::runtime_error(
                    "Multiple shape interactions lost Gu-indexed impulse totals");
        }
        if(!saw_left || !saw_right ||
           !std::all_of(claimed.begin(),claimed.end(),
                        [](bool x){return x;}))
            throw std::runtime_error(
                "Sc callback packing dropped one of the source contact point ranges");
    };
    verify_multi_manager(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    verify_multi_manager(ExecutionMode::ReferencePhysXCpuLegacyGu);
    verify_multi_manager(ExecutionMode::ReferencePhysXCpu);
#endif
    verify(ExecutionMode::NativeCpu);
    // The externally returned span is the LAST successfully published
    // simulation result. A rejected step must not replace that storage
    // with partially prepared Sc/Dy event data.
    {
        fyzz::native::Scene scene{fyzz::native::Scene::ContactGenerator::DiagnosticLegacyGu};
        scene.set_contact_notifications(true);
        const auto floor=scene.add_static_floor(0.0f);
        const auto box=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        scene.step(dt);
        if(scene.contact_events().size()!=1 ||
           scene.contact_events()[0].kind!=ContactEventKind::Found)
            throw std::runtime_error(
                "Native source touch event baseline not published");
        const auto* const old_storage=scene.contact_events().data();
        const auto* const old_point_storage=
            scene.contact_event_points().data();
        const auto old_point=scene.contact_event_points()[0];
        const auto old=scene.contact_events()[0];
        // Reuse original admission failure already tested for materials:
        // eMULTIPLY 1e30*1e30 overflows PxsMaterialCombiner's PxReal.
        Material huge{};
        huge.static_friction=1.0e30f;
        huge.dynamic_friction=1.0e30f;
        huge.friction_combine=MaterialCombineMode::Multiply;
        scene.set_material(floor,huge);
        scene.set_material(box,huge);
        bool threw=false;
        try {scene.step(dt);}
        catch(const std::overflow_error&) {threw=true;}
        const auto published=scene.contact_events();
        if(!threw || published.data()!=old_storage ||
           scene.contact_event_points().data()!=old_point_storage ||
           scene.contact_event_points()[0].normal_impulse!=
               old_point.normal_impulse ||
           scene.contact_event_points()[0].separation!=old_point.separation ||
           published.size()!=1 ||
           published[0].body0!=old.body0 ||
           published[0].body1!=old.body1 ||
           published[0].kind!=old.kind ||
           published[0].normal_impulse!=old.normal_impulse)
            throw std::runtime_error(
                "Failed native PGS step replaced published Sc contact events");
        scene.set_material(floor,Material{});
        scene.set_material(box,Material{});
        scene.step(dt);
        if(scene.contact_events().size()!=1 ||
           scene.contact_events()[0].kind!=ContactEventKind::Persists ||
           scene.contact_events()[0].contact_count!=
               scene.contact_event_points().size())
            throw std::runtime_error(
                "Source touch event publication did not recover on retry");
    }
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    verify(ExecutionMode::ReferencePhysXCpuLegacyGu);
    verify(ExecutionMode::ReferencePhysXCpu);
#endif
}

void verify_removed_actor_touch_notifications() {
    using namespace fyzz;
    constexpr float dt=1.0f/120.0f;
    const auto verify=[](ExecutionMode mode) {
        WorldOptions opts{mode};
        opts.contact_notifications=true;
        World world(opts);
        const auto floor=world.add_static_floor(0.0f);
        const auto left=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-4.0f,0.49f,0},1.0f);
        const auto right=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4.0f,0.49f,0},1.0f);
        RigidBodyDynamics d{};
        d.gravity_enabled=false;
        world.set_body_dynamics(left,d);
        world.set_body_dynamics(right,d);
        world.step(dt);
        if(world.contact_events().size()!=2)
            throw std::runtime_error(
                "Source removed-actor fixture lacks both initial contacts");
        // Both actors are retired before the next source Sc fetch. The
        // original callback marks dangling actor pointers with
        // eREMOVED_ACTOR_*. A stale PxActor address must be compared
        // only against IDs preserved on the retirement path.
        world.remove_body(left);
        world.remove_body(right);
        world.step(dt);
        const auto events=world.contact_events();
        if(events.size()!=2 || !world.contact_event_points().empty())
            throw std::runtime_error(
                "Retiring rigid actors did not publish both Lost contacts");
        bool saw_left=false,saw_right=false;
        for(const auto& event:events) {
            if(event.body0!=floor ||
               event.kind!=ContactEventKind::Lost ||
               event.contact_count!=0 ||
               event.normal_impulse!=0.0)
                throw std::runtime_error(
                    "Removed-actor Sc Lost callback has invalid event fields");
            if(event.body1==left)saw_left=true;
            else if(event.body1==right)saw_right=true;
            else throw std::runtime_error(
                "Removed-actor callback corrupted stable BodyId tokens");
        }
        if(!saw_left || !saw_right)
            throw std::runtime_error(
                "Two removed actors were aliased in original callback identity");
        world.step(dt);
        if(!world.contact_events().empty())
            throw std::runtime_error(
                "Removed actor Lost callbacks replayed on a later fetch");

        // The first support plane is no longer part of the scenario.
        // Retire it before introducing the next pair, or a new dynamic
        // box would legitimately contact BOTH coincident planes.
        world.remove_body(floor);
        world.step(dt);
        // Removal of the STATIC endpoint must retain the same
        // identity contract, not only the removed dynamic endpoint.
        const auto support=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        world.set_body_dynamics(box,d);
        world.step(dt);
        if(world.contact_events().size()!=1 ||
           world.contact_events()[0].kind!=ContactEventKind::Found)
            throw std::runtime_error(
                "Removed-static callback fixture missed initial contact");
        world.remove_body(support);
        world.step(dt);
        if(world.contact_events().size()!=1 ||
           world.contact_events()[0].kind!=ContactEventKind::Lost ||
           world.contact_events()[0].body0!=support ||
           world.contact_events()[0].body1!=box)
            throw std::runtime_error(
                "Retired static actor lost its stable Sc contact identity");
    };
    verify(ExecutionMode::NativeCpu);
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    verify(ExecutionMode::ReferencePhysXCpuLegacyGu);
    verify(ExecutionMode::ReferencePhysXCpu);
#endif
}

void verify_source_signed_compliant_contact_writeback() {
    using namespace fyzz;
    constexpr float dt=1.0f/60.0f;
    World world(WorldOptions{ExecutionMode::NativeCpu});
    const auto ground=world.add_static_floor(0.0f);
    const auto box=world.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,0.49f,0.0f},1.0f);
    Material spring{};
    spring.restitution=-100.0f;
    world.set_body_material(box,spring);
    world.set_body_velocity(box,{1.0f,1.0f,0.0f});
    // Codex's actual public-API repro fails on frame 27 if native
    // writeback rejects a small signed compliant spring impulse.
    // Original DySolverConstraintsShared does not clamp the result to
    // zero after applying the spring's impulse multiplier.
    for(unsigned frame=0;frame<80;++frame) {
        world.step(dt);
        const auto state=world.body_state(box);
        if(!std::isfinite(state.position.x) ||
           !std::isfinite(state.position.y) ||
           !std::isfinite(state.velocity.x) ||
           !std::isfinite(state.velocity.y))
            throw std::runtime_error(
                "Source compliant impulse caused nonfinite body publication");
    }
    (void)ground;
}

void verify_source_body_dynamics_api() {
    using namespace fyzz;
    RigidBodyDynamics configured{};
    configured.linear_damping=2.0f;
    configured.angular_damping=1.0f;
    configured.gravity_enabled=false;
    configured.max_linear_velocity=1.5f;
    configured.max_angular_velocity=2.0f;
    configured.position_iterations=6;
    configured.velocity_iterations=2;

    const auto motion=[](ExecutionMode mode,RigidBodyDynamics d) {
        World scene(WorldOptions{mode});
        const auto id=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,12,0},1.0f);
        scene.set_body_dynamics(id,d);
        scene.set_body_velocity(id,{3.0f,2.0f,-1.0f});
        scene.set_body_angular_velocity(id,{0,0,4.0f});
        scene.step(0.1f);
        return scene.body_state(id);
    };
    const auto native=motion(ExecutionMode::NativeCpu,configured);
    const auto lin=std::sqrt(
        native.velocity.x*native.velocity.x+
        native.velocity.y*native.velocity.y+
        native.velocity.z*native.velocity.z);
    const auto ang=std::sqrt(
        native.angular_velocity.x*native.angular_velocity.x+
        native.angular_velocity.y*native.angular_velocity.y+
        native.angular_velocity.z*native.angular_velocity.z);
    if(!std::isfinite(lin) || !std::isfinite(ang) ||
       std::abs(lin-1.5f)>1.0e-4f ||
       std::abs(ang-2.0f)>1.0e-4f ||
       native.position.y<12.0f)
        throw std::runtime_error(
            "Native source rigid damping/gravity/speed clamps not applied");
    RigidBodyDynamics default_d{};
    const auto baseline=motion(ExecutionMode::NativeCpu,default_d);
    if(!(baseline.angular_velocity.z>native.angular_velocity.z &&
         baseline.velocity.x>native.velocity.x))
        throw std::runtime_error(
            "Source rigid body damping and clamp authoring had no effect");
    // Isolate gravity from the deliberate speed limit. Compare otherwise
    // identical bodies with a high clamp, so a stronger downward force
    // cannot be hidden by a different velocity-length normalization.
    RigidBodyDynamics no_gravity=configured;
    no_gravity.max_linear_velocity=100.0f;
    RigidBodyDynamics with_gravity=no_gravity;
    with_gravity.gravity_enabled=true;
    const auto floating=motion(ExecutionMode::NativeCpu,no_gravity);
    const auto falling=motion(ExecutionMode::NativeCpu,with_gravity);
    if(!(floating.velocity.y>falling.velocity.y+0.7f &&
         floating.position.y>falling.position.y+0.07f))
        throw std::runtime_error(
            "Source per-body eDISABLE_GRAVITY did not change free-flight motion");
    World scene(WorldOptions{ExecutionMode::NativeCpu});
    const auto floor=scene.add_static_floor(0.0f);
    const auto id=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
    bool static_rejected=false,invalid_rejected=false;
    try {scene.set_body_dynamics(floor,configured);}
    catch(const std::logic_error&) {static_rejected=true;}
    RigidBodyDynamics bad=configured;
    bad.position_iterations=0;
    try {scene.set_body_dynamics(id,bad);}
    catch(const std::invalid_argument&) {invalid_rejected=true;}
    if(!static_rejected || !invalid_rejected)
        throw std::runtime_error("Source rigid body dynamics admission invalid");
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    const auto reference=motion(ExecutionMode::ReferencePhysXCpuLegacyGu,
                                configured);
    const auto close=[](Vec3 a,Vec3 b,float epsilon) {
        return std::abs(a.x-b.x)<=epsilon &&
               std::abs(a.y-b.y)<=epsilon &&
               std::abs(a.z-b.z)<=epsilon;
    };
    if(!close(native.velocity,reference.velocity,0.0004f) ||
       !close(native.angular_velocity,reference.angular_velocity,0.0004f))
        throw std::runtime_error(
            "Native rigid body source preintegration differs from PhysX CPU");
#endif
}

namespace {
void expect(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
bool has(std::span<const fyzz::native::Pair> p, fyzz::BodyId a, fyzz::BodyId b) {
    const fyzz::native::Pair match{std::min(a,b), std::max(a,b)};
    return std::find(p.begin(), p.end(), match) != p.end();
}
}
// PxRigidDynamic::setGlobalPose(autowake=true) resets sleep state even
// when the authored translation equals the current pose. Geometry-only
// caches and manager handles must remain untouched by that no-op edit.
void verify_idempotent_pose_autowake() {
    using fyzz::native::Scene;
    {
        Scene connected{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=connected.add_static_floor(0.0f);
        const auto left=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            connected.step(1.0f/120.0f);
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right) &&
               connected.active_accurate_nodes().empty() &&
               connected.active_speculative_nodes().empty(),
               "Same-pose autowake fixture never reached a dormant island");
        const auto before=connected.body_state(left);
        const auto original_edge=connected.contact_edge_index({left,right});
        const auto rebuilds=connected.island_full_rebuild_count();
        const auto revision=connected.island_topology_revision();
        const auto manager_count=connected.contacts().size();
        connected.set_position(left,before.position);
        expect(!connected.body_is_sleeping(left) &&
               !connected.body_ready_for_sleep(left) &&
               connected.body_wake_counter(left)>0.0f &&
               connected.body_state(left).position.x==before.position.x &&
               connected.body_state(left).position.y==before.position.y &&
               connected.body_state(left).position.z==before.position.z,
               "Same-position dynamic pose authoring skipped PhysX autowake");
        connected.advance_collision();
        expect(!connected.body_is_sleeping(left) &&
               !connected.body_is_sleeping(right) &&
               connected.active_accurate_nodes().size()==2 &&
               connected.active_speculative_nodes().size()==2 &&
               connected.island_full_rebuild_count()==rebuilds &&
               connected.island_topology_revision()==revision &&
               connected.contact_edge_index({left,right})==original_edge &&
               connected.contacts().size()==manager_count &&
               connected.found_pairs().empty() &&
               connected.lost_pairs().empty() &&
               connected.found_contacts().empty() &&
               connected.lost_contacts().empty(),
               "Same-pose autowake rebuilt unchanged collision topology");
        // A matching static pose must not wake its actual contacting
        // sleepers; the automatic wake is specific to dynamic actors.
        for(int frame=0;frame<360;++frame)
            connected.step(1.0f/120.0f);
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right),
               "Static same-pose fixture never returned to sleep");
        const auto quiet_revision=connected.island_topology_revision();
        connected.set_position(floor,connected.body_state(floor).position);
        connected.advance_collision();
        expect(connected.body_is_sleeping(left) &&
               connected.body_is_sleeping(right) &&
               connected.island_topology_revision()==quiet_revision,
               "Same-position static pose changed contacting sleep state");
    }
    {
        // Source Dy::sleepCheck can mark a body ready one frame before
        // the third pass deactivates it. The same-pose write must reset
        // that readiness as well as waking bodies already asleep.
        Scene ready{Scene::ContactGenerator::DiagnosticLegacyGu};
        ready.add_static_floor(0.0f);
        const auto body=ready.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool reached_ready=false;
        for(int frame=0;frame<600;++frame) {
            ready.step(1.0f/120.0f);
            if(ready.body_ready_for_sleep(body)) {
                reached_ready=true;
                break;
            }
        }
        expect(reached_ready && !ready.body_is_sleeping(body),
               "Same-pose readiness fixture did not reach the source boundary");
        const auto old=ready.body_state(body);
        ready.set_position(body,old.position);
        expect(!ready.body_ready_for_sleep(body) &&
               !ready.body_is_sleeping(body) &&
               ready.body_wake_counter(body)>0.0f,
               "Same-position authoring retained stale Dy sleep readiness");
        ready.step(1.0f/120.0f);
        expect(!ready.body_is_sleeping(body),
               "Same-position wake was lost at the next source third pass");
    }
}
// PxsIslandSim::deactivateNodeInternal emits eCONTACT_MANAGER edges
// only when the final awake endpoint sleeps. Static world connections
// count; destroyed edges never become sleep-deactivation events.
void verify_source_contact_edge_deactivations() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=scene.add_static_floor(0.0f);
    const auto left=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1);
    const auto right=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1);
    scene.advance_collision();
    const auto lr=scene.contact_edge_index(Pair{left,right});
    const auto fl=scene.contact_edge_index(Pair{floor,left});
    const auto fr=scene.contact_edge_index(Pair{floor,right});
    expect(lr!=std::numeric_limits<std::uint32_t>::max() &&
           fl!=std::numeric_limits<std::uint32_t>::max() &&
           fr!=std::numeric_limits<std::uint32_t>::max() &&
           scene.deactivating_accurate_contact_edges().empty() &&
           scene.deactivating_speculative_contact_edges().empty(),
           "Contact edges deactivated before rigid sleep");
    bool entered_sleep=false;
    for(int frame=0;frame<600;++frame) {
        scene.step(1.0f/120.0f);
        if(scene.body_is_sleeping(left) &&
           scene.body_is_sleeping(right)) {
            entered_sleep=true;
            break;
        }
    }
    expect(entered_sleep,"Source contact-edge fixture never slept");
    for(const auto edges:{
        scene.deactivating_accurate_contact_edges(),
        scene.deactivating_speculative_contact_edges()}) {
        expect(edges.size()==3 &&
               std::count(edges.begin(),edges.end(),lr)==1 &&
               std::count(edges.begin(),edges.end(),fl)==1 &&
               std::count(edges.begin(),edges.end(),fr)==1,
               "Rigid sleep did not deactivate each contact edge once");
    }
    expect(scene.contact_edge_connected(Pair{left,right}) &&
           scene.contact_edge_connected(Pair{floor,left}) &&
           scene.contact_edge_connected(Pair{floor,right}),
           "Rigid sleep incorrectly retired persistent contact managers");
    scene.advance_collision();
    expect(scene.deactivating_accurate_contact_edges().empty() &&
           scene.deactivating_speculative_contact_edges().empty(),
           "Quiet source publication replayed edge deactivations");
    scene.set_position(left,scene.body_state(left).position);
    scene.advance_collision();
    expect(!scene.body_is_sleeping(left) &&
           !scene.body_is_sleeping(right) &&
           scene.deactivating_accurate_contact_edges().empty(),
           "Authored wake replayed stale edge deactivations");
    Scene lost{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=lost.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,12,0},1);
    const auto b=lost.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,12,0},1);
    lost.advance_collision();
    lost.set_position(b,{5,12,0});
    lost.advance_collision();
    expect(has(lost.lost_contacts(),a,b) &&
           lost.deactivating_accurate_contact_edges().empty() &&
           lost.deactivating_speculative_contact_edges().empty(),
           "Lost contact edge incorrectly entered sleep output");
}

// NpRigidDynamic::setGlobalPose refreshes an awake body's wake counter
// without clearing DySleep's accumulated low-motion samples. Use the
// existing setLinearVelocity autowake path as a source-compatible control,
// with an unchanged velocity to preserve the same physical state.
void verify_active_pose_preserves_sleep_filter() {
    using fyzz::native::Scene;
    Scene authored, control;
    const auto populate=[](Scene& scene) {
        const auto floor=scene.add_static_floor(0.0f);
        const auto body=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material material{};
        material.disable_friction=true;
        scene.set_material(floor,material);
        scene.set_material(body,material);
        scene.set_velocity(body,{0.006f,0,0});
        return body;
    };
    const auto a=populate(authored);
    const auto b=populate(control);
    constexpr float dt=1.0f/120.0f;
    // Once the source wake counter is below 0.2 seconds, DySleep is
    // integrating small but nonzero lateral velocities into its filter.
    // Preserve those samples through a same-pose authoring operation.
    for(int frame=0;frame<40;++frame) {
        authored.step(dt);
        control.step(dt);
    }
    const auto before=authored.body_state(a);
    const auto control_before=control.body_state(b);
    expect(!before.sleeping && !control_before.sleeping &&
           authored.body_wake_counter(a)>0.0f &&
           authored.body_wake_counter(a)<0.2f &&
           authored.body_wake_counter(a)==control.body_wake_counter(b) &&
           before.velocity.x>0.005f &&
           before.velocity.x==control_before.velocity.x,
           "Active pose fixture did not accumulate low-motion sleep samples");
    authored.set_position(a,before.position);
    control.set_velocity(b,control_before.velocity);
    expect(authored.body_wake_counter(a)==control.body_wake_counter(b) &&
           !authored.body_ready_for_sleep(a),
           "Active pose write did not refresh the source wake counter");
    bool renewed=false;
    float last_control_counter=control.body_wake_counter(b);
    for(int frame=0;frame<42;++frame) {
        authored.step(dt);
        control.step(dt);
        const float current=control.body_wake_counter(b);
        renewed |= current>last_control_counter+0.02f;
        last_control_counter=current;
        expect(authored.body_wake_counter(a)==current &&
               authored.body_ready_for_sleep(a)==control.body_ready_for_sleep(b) &&
               authored.body_is_sleeping(a)==control.body_is_sleeping(b),
               "Active pose write discarded source DySleep accumulator history");
    }
    expect(renewed,
           "Active pose filter fixture did not exercise accumulated-energy renewal");
}
// Keep multi-island lifetime regressions in separate stack frames. The
// native scene smoke runner contains many Scene fixtures{Scene::ContactGenerator::DiagnosticLegacyGu}; combining all
// of their local allocations into one MSVC frame can exhaust the
// default Windows stack even though the individual tests are small.
void verify_simultaneous_initial_sleep_compaction() {
    using fyzz::native::Scene;
    // Multiple initially active rigid nodes may enter the third-pass
    // deactivation list together while a new rigid is being admitted.
    // markInactive relocates each initial member to the initial-prefix
    // tail, then removes it using ordinary swap-last. The traversal
    // must recheck the ORIGINAL compacted position, including any
    // other dormant initial member moved there; otherwise [A, B, Cnew]
    // can leave dormant B in mActiveNodes and misreport the suffix.
    // Exercise both numerical and collision-only publication paths.
    for(const bool query_only : {false,true}) {
        Scene multi_sleep{Scene::ContactGenerator::DiagnosticLegacyGu};
        multi_sleep.add_static_floor(0.0f);
        const auto first=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto second=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
        multi_sleep.advance_collision();
        expect(multi_sleep.active_accurate_nodes().size()==2 &&
               multi_sleep.accurate_active_node_index(first)==0 &&
               multi_sleep.accurate_active_node_index(second)==1,
               "Multi-sleep source fixture did not index two active islands");
        bool both_ready=false;
        for(int frame=0;frame<600;++frame) {
            multi_sleep.step(1.0f/120.0f);
            if(multi_sleep.body_ready_for_sleep(first) &&
               multi_sleep.body_ready_for_sleep(second)) {
                both_ready=true;
                break;
            }
        }
        expect(both_ready && !multi_sleep.body_is_sleeping(first) &&
               !multi_sleep.body_is_sleeping(second),
               "Multi-sleep source fixture missed shared readiness boundary");
        const auto newcomer=multi_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,10,0},1.0f);
        if(query_only)
            multi_sleep.advance_collision();
        else
            multi_sleep.step(1.0f/120.0f);
        const auto active=multi_sleep.active_accurate_nodes();
        const auto fresh=multi_sleep.newly_activated_accurate_nodes();
        const auto retired=multi_sleep.deactivating_accurate_nodes();
        expect(multi_sleep.body_is_sleeping(first) &&
               multi_sleep.body_is_sleeping(second) &&
               active.size()==1 && active[0]==newcomer &&
               fresh.size()==1 && fresh[0]==newcomer &&
               retired.size()==2 &&
               std::find(retired.begin(),retired.end(),first)!=retired.end() &&
               std::find(retired.begin(),retired.end(),second)!=retired.end() &&
               multi_sleep.accurate_active_node_index(first)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.accurate_active_node_index(second)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.accurate_active_node_index(newcomer)==0,
               "Source initial-prefix compaction skipped another dormant body");
        // The wake path must not replay the stale sleeper or lose the
        // surviving initial active node when it starts a new pass.
        multi_sleep.set_velocity(first,{0.25f,0,0});
        multi_sleep.advance_collision();
        const auto reactivated=multi_sleep.active_accurate_nodes();
        expect(reactivated.size()==2 &&
               std::find(reactivated.begin(),reactivated.end(),first)!=
                   reactivated.end() &&
               std::find(reactivated.begin(),reactivated.end(),newcomer)!=
                   reactivated.end() &&
               multi_sleep.accurate_active_node_index(second)==
                   std::numeric_limits<std::size_t>::max() &&
               multi_sleep.newly_activated_accurate_nodes().size()==1 &&
               multi_sleep.newly_activated_accurate_nodes()[0]==first,
               "Multi-sleep compaction corrupted later authored wake suffix");
    }

}
// Independent speculative IslandSim active indices must follow the
// original wakeIslands/markInactive initial-prefix contract, including
// quiet authored wakes, actor admission and failed graph reconstruction.
void verify_speculative_active_node_lifecycle() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    scene.add_static_floor(0.0f);
    const auto first=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto sleeper=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
    scene.advance_collision();
    expect(scene.active_speculative_nodes().size()==2 &&
           scene.newly_activated_speculative_nodes().size()==2 &&
           scene.speculative_active_node_index(first)==0 &&
           scene.speculative_active_node_index(sleeper)==1,
           "Speculative IslandSim did not index initial dynamic actors");
    for(int i=0;i<360;++i)scene.step(1.0f/120.0f);
    expect(scene.body_is_sleeping(first) && scene.body_is_sleeping(sleeper) &&
           scene.active_speculative_nodes().empty() &&
           scene.active_accurate_nodes().empty() &&
           scene.speculative_active_node_index(first)==
               std::numeric_limits<std::size_t>::max() &&
           scene.speculative_active_node_index(sleeper)==
               std::numeric_limits<std::size_t>::max(),
           "Speculative graph retained source-deactivated rigid nodes");

    scene.set_velocity(first,{0.2f,0,0});
    scene.advance_collision();
    const auto waking=scene.newly_activated_speculative_nodes();
    expect(scene.active_speculative_nodes().size()==1 &&
           waking.size()==1 && waking[0]==first &&
           scene.speculative_active_node_index(first)==0 &&
           scene.speculative_active_node_index(sleeper)==
               std::numeric_limits<std::size_t>::max(),
           "Quiet authored wake failed speculative active-index publication");
    scene.advance_collision();
    expect(scene.newly_activated_speculative_nodes().empty() &&
           scene.active_speculative_nodes().size()==1,
           "Quiet collision query replayed speculative activation suffix");

    const auto admitted=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{20,5,0},1.0f);
    scene.advance_collision();
    const auto suffix=scene.newly_activated_speculative_nodes();
    expect(scene.active_speculative_nodes().size()==2 &&
           suffix.size()==1 && suffix[0]==admitted &&
           scene.speculative_active_node_index(admitted)==1 &&
           scene.newly_activated_accurate_nodes().size()==1,
           "Speculative active prefix was lost during actor admission");

    const auto published=scene.active_speculative_nodes();
    const auto old_frame=scene.frame_index();
    const auto candidate=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{30,5,0},1.0f);
    bool failed=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {failed=true;}
    const auto restored=scene.active_speculative_nodes();
    expect(failed && scene.frame_index()==old_frame &&
           restored.data()==published.data() &&
           restored.size()==published.size() &&
           restored[0]==first && restored[1]==admitted &&
           scene.speculative_active_node_index(candidate)==
               std::numeric_limits<std::size_t>::max(),
           "Rejected graph admission replaced published speculative active span");
    scene.step(1.0f/120.0f);
    expect(scene.active_speculative_nodes().size()==3 &&
           scene.speculative_active_node_index(candidate)<
               scene.active_speculative_nodes().size(),
           "Speculative IslandSim could not admit actor after rejected frame");
    scene.remove(first);
    scene.advance_collision();
    expect(scene.active_speculative_nodes().size()==2 &&
           scene.speculative_active_node_index(first)==
               std::numeric_limits<std::size_t>::max(),
           "Speculative graph kept a retired rigid in its active list");
}

// Two orphan components can share one original indexed-manager array.
// PhysX processes their dirty nodes in ascending mDirtyMap index,
// NOT in the order of the smallest BodyId in each new component. These
// orders disagree: right G is dirty ID 3; left B is dirty ID 5, while
// left A has smaller BodyId 0 than right F's BodyId 2. Swap-last
// compaction therefore moves F-G before A-B out of the old root array.
void verify_dirty_order_multisplit_manager_remainder() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    // Form all three islands separately first. The central C-D-E
    // source root must acquire size three before the two size-two
    // arms connect, otherwise mergeIslands could retain an arm root.
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-5.0f,10,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto f=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{6.2f,10,0},1);
    const auto g=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5.4f,10,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-4.2f,10,0},1);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==3 &&
           scene.contacts().size()==4 &&
           scene.accurate_island_root(c)==d,
           "Two-way split fixture did not establish its larger central root");
    scene.set_position(a,{-1.6f,10,0});
    scene.set_position(b,{-0.8f,10,0});
    scene.set_position(f,{3.2f,10,0});
    scene.set_position(g,{2.4f,10,0});
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.accurate_island_root(a)==d &&
           scene.accurate_island_root(g)==d &&
           has(scene.found_contacts(),b,c) &&
           has(scene.found_contacts(),g,e),
           "Two-way split fixture lost its central source root during merges");
    // Append root-owned static managers only *after* both arm arrays
    // have joined the central island. Their suffix is essential:
    // removing F-G before A-B swaps these later root managers into
    // a different order than removing A-B before F-G.
    const auto sc=scene.add_static_box(
        {0.1f,0.1f,0.1f},{0,10,0.54f});
    const auto sd=scene.add_static_box(
        {0.1f,0.1f,0.1f},{0.8f,10,0.54f});
    const auto se=scene.add_static_box(
        {0.1f,0.1f,0.1f},{1.6f,10,0.54f});
    scene.advance_collision();
    scene.step(1.0f/480.0f);
    const auto initial_managers=scene.indexed_contact_manager_order();
    const auto initial_gu=scene.contacts();
    expect(scene.accurate_island_count()==1 &&
           initial_gu.size()==9 && initial_managers.size()==9,
           "Dirty-ordered two-way split fixture lacked nine managers");
    std::vector<Pair> root_array;
    for(std::size_t index:initial_managers)
        root_array.push_back(initial_gu[index].pair);

    // Source lost-bridge publication occurs before processLostEdges,
    // in the deterministic Gu manager stream. Thereafter the right
    // dirty G component splits first, then left dirty B.
    const auto swap_remove=[&](Pair pair) {
        const auto found=std::find(root_array.begin(),root_array.end(),pair);
        expect(found!=root_array.end(),
               "Source split oracle missing its old indexed manager");
        *found=root_array.back();
        root_array.pop_back();
    };
    swap_remove(Pair{c,b});
    swap_remove(Pair{g,e});
    swap_remove(Pair{f,g});
    swap_remove(Pair{a,b});
    std::vector<Pair> expected;
    for(const Pair pair:root_array)
        if(pair==Pair{c,d} || pair==Pair{d,e} ||
           pair==Pair{c,sc} || pair==Pair{d,sd} ||
           pair==Pair{e,se})
            expected.push_back(pair);
    expect(expected.size()==5,
           "Dirty-order source oracle did not retain five root managers");

    scene.set_position(a,{-2.5f,10,0});
    scene.set_position(b,{-1.7f,10,0});
    scene.set_position(f,{4.1f,10,0});
    scene.set_position(g,{3.3f,10,0});
    scene.step(1.0f/480.0f);
    expect(scene.accurate_island_count()==3 &&
           has(scene.lost_contacts(),b,c) &&
           has(scene.lost_contacts(),g,e) &&
           scene.contacts().size()==7,
           "Dirty-order fixture did not detach both orphan components");
    const auto gu=scene.contacts();
    const auto managers=scene.indexed_contact_manager_order();
    std::vector<Pair> retained;
    for(std::size_t index:managers) {
        const Pair pair=gu[index].pair;
        if(pair==Pair{c,d} || pair==Pair{d,e} ||
           pair==Pair{c,sc} || pair==Pair{d,sd} ||
           pair==Pair{e,se})
            retained.push_back(pair);
    }
    expect(retained==expected,
           "Source mDirtyMap multi-split order changed retained manager sequence");
}

// Unequal original mHopCounts must change *which branch is popped*
// before the other branch. Starting at dirty B, the source EdgeInstance
// head discovers A before X, but a prior successful lost-chord repair
// gives X a finite route estimate and leaves A without one. The queue
// therefore expands X-H before A-G; FIFO would reverse that ordering.
// PhysX addNode starts hop counts at zero, then processNewEdges
// adjusts them through source mergeIslandsInternal. Here X joins the
// root island through C and has fewer hops than A, which later merges
// through B. The older B-X edge is behind newer B-A in B's adjacency;
// when C-X and B-C vanish together, the hop-priority queue expands X
// before A despite having *discovered* A first. The discovered child H
// must then precede G in the source mVisitedNodes migration list.
// FIFO would reverse their manager order.
void verify_hop_priority_split_migration() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene split{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.8f,10,0},1);
    const auto x=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.35f,10.65f,0},1);
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-3.2f,9.6f,0},1);
    const auto g=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-2.4f,9.6f,0},1);
    const auto h=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-1.1f,11.6f,0},1);
    const auto sg=split.add_static_box(
        {0.1f,0.1f,0.1f},{-2.4f,9.6f,0.54f});
    const auto sh=split.add_static_box(
        {0.1f,0.1f,0.1f},{-1.1f,11.6f,0.54f});
    split.advance_collision();
    expect(has(split.found_contacts(),c,x) &&
           has(split.found_contacts(),b,x) &&
           has(split.found_contacts(),x,h) &&
           has(split.found_contacts(),a,g) &&
           has(split.found_contacts(),g,sg) &&
           has(split.found_contacts(),h,sh) &&
           split.accurate_island_count()==2 &&
           split.accurate_cached_route_hops(x)>0 &&
           split.accurate_cached_route_hops(a)>0,
           "Hop-priority fixture did not establish source merge estimates");

    // A-G is the smaller island: merging it at the newer B-A manager
    // gives A the original mergeIslandsInternal endpoint hop B+1.
    // X remains connected by the older B-X and C-X edges.
    split.set_position(a,{-1.6f,9.6f,0});
    split.advance_collision();
    expect(has(split.found_contacts(),a,b) &&
           split.accurate_island_count()==1 &&
           split.accurate_cached_route_hops(x)<
               split.accurate_cached_route_hops(a),
           "Hop-priority fixture did not establish unequal source merge hops");

    // Lose C-X and B-C in one publication. An earlier independent
    // lost-chord repair would raise X's estimate to B+1, obscuring the
    // valid shorter source estimate that existed when both edges broke.
    split.set_position(x,{-1.1f,10.8f,0});
    split.set_position(c,{1.5f,10,0});
    split.set_position(d,{2.3f,10,0});
    split.set_position(e,{3.1f,10,0});
    split.step(1.0f/480.0f);
    const auto gu=split.contacts();
    const auto managers=split.indexed_contact_manager_order();
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),b,c) &&
           has(split.lost_contacts(),c,x) &&
           !has(split.lost_contacts(),b,x) &&
           !has(split.lost_contacts(),x,h) &&
           gu.size()==managers.size(),
           "Hop-priority fixture did not detach its original branched island");
    std::size_t h_position=managers.size();
    std::size_t g_position=managers.size();
    for(std::size_t i=0;i<managers.size();++i) {
        const Pair pair=gu[managers[i]].pair;
        if(pair==Pair{h,sh})h_position=i;
        if(pair==Pair{g,sg})g_position=i;
    }
    expect(h_position<g_position && g_position<managers.size(),
           "Split manager migration ignored source unequal merge-hop priorities");
}

// CmPriorityQueue::pop favors the left child on equal priorities.
// With five equal-hop branches, it expands the first, second, fourth,
// fifth and third queued branches (not FIFO). Each branch owns a leaf
// with a static contact manager: their migration order makes this heap
// detail observable on the actual Dy indexed constraint input.
void verify_equal_hop_split_queue_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    using fyzz::Vec3;
    Scene split{Scene::ContactGenerator::DiagnosticLegacyGu};
    const Vec3 origin{0,10,0};
    // Admit root-anchor first so the two-node original root survives
    // the star's later one-node merges. Losing root-center then forces
    // a real dirty-route split, rather than visiting an already-rooted
    // star whose dirty seed is trivially its root.
    const auto root=split.add_dynamic_box(
        {0.4f,0.4f,0.4f},{0,10,-1.1f},1);
    const auto anchor=split.add_dynamic_box(
        {0.4f,0.4f,0.4f},{0,10,-1.75f},1);
    const auto center=split.add_dynamic_box(
        {0.75f,0.75f,0.75f},origin,1);
    const std::array<Vec3,5> axes{{
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1}}};
    std::array<fyzz::BodyId,5> arms{},leaves{},markers{};
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        arms[i]=split.add_dynamic_box(
            {0.4f,0.4f,0.4f},
            {a.x*1.1f,10+a.y*1.1f,a.z*1.1f},1);
    }
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        leaves[i]=split.add_dynamic_box(
            {0.3f,0.3f,0.3f},
            {a.x*1.7f,10+a.y*1.7f,a.z*1.7f},1);
    }
    for(std::size_t i=0;i<axes.size();++i) {
        const Vec3& a=axes[i];
        markers[i]=split.add_static_box(
            {0.08f,0.08f,0.08f},
            {a.x*1.96f,10+a.y*1.96f,a.z*1.96f});
    }
    split.advance_collision();
    expect(split.accurate_island_count()==1 &&
           has(split.found_contacts(),root,center) &&
           has(split.found_contacts(),root,anchor) &&
           split.accurate_island_root(center)==anchor,
           "Equal-hop source fixture did not preserve the original root");
    for(std::size_t i=0;i<axes.size();++i) {
        expect(has(split.found_contacts(),center,arms[i]) &&
               has(split.found_contacts(),arms[i],leaves[i]) &&
               has(split.found_contacts(),leaves[i],markers[i]),
               "Equal-hop fixture lacked an isolated marked branch");
    }
    // All five child islands joined their parent through the same
    // hop depth. No newer redundant chords should alter the estimates.
    const auto hop=split.accurate_cached_route_hops(arms[0]);
    for(auto arm:arms)
        expect(split.accurate_cached_route_hops(arm)==hop,
               "Equal-hop priority fixture has unequal source estimates");

    split.set_position(root,{0,10,-1.4f});
    split.set_position(anchor,{0,10,-2.05f});
    split.step(1.0f/480.0f);
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),root,center) &&
           split.accurate_island_root(center)==center,
           "Equal-hop queue fixture did not separate the dirty star");
    const auto gu=split.contacts();
    const auto indices=split.indexed_contact_manager_order();
    std::vector<fyzz::BodyId> visited_leaves;
    for(auto index:indices) {
        const auto pair=gu[index].pair;
        for(std::size_t i=0;i<axes.size();++i)
            if(pair==Pair{leaves[i],markers[i]})
                visited_leaves.push_back(leaves[i]);
    }
    // Center's intrusive adjacency enumerates arms in reverse insertion
    // order: 4,3,2,1,0. Source CmPriorityQueue pops the five equal-hop
    // branches in order 4,3,1,0,2, so their grandchildren and marker
    // managers migrate in that order. FIFO/tie-by-discovery differs.
    const std::vector<fyzz::BodyId> expected{
        leaves[4],leaves[3],leaves[1],leaves[0],leaves[2]};
    expect(visited_leaves==expected,
           "Equal-hop source heap tie order changed split manager migration");
}

// A root-bridge loss and an internal orphan chord loss occur in the
// same publication. PhysX processLostEdges splits on the first failed
// dirty route, then *skips* other dirty nodes already present in that
// split's shared mVisitedState. The initial findRoute visitation supplies
// the new node/manager ordering directly; it must not traverse twice.
void verify_source_dirty_witness_split() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,16,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,16,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,16,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.4f,16,0},1);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.55f,16,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.speculative_island_count()==1 &&
           scene.accurate_island_root(c)==b &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),b,c) &&
           has(scene.found_contacts(),c,d) &&
           has(scene.found_contacts(),c,e) &&
           has(scene.found_contacts(),d,e),
           "Dirty witness split fixture lacks its pre-loss root and chord");
    const auto old_handle=scene.accurate_island_id(b);
    const auto checks=scene.island_dirty_route_check_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    const auto splits=scene.island_local_split_count();
    // Preserve the A-B root island, but disconnect C-D-E at B-C.
    // Remove C-E while preserving C-D-E's alternate route.
    scene.set_position(a,{-2,16,0});
    scene.set_position(b,{-1.2f,16,0});
    scene.set_position(e,{3.35f,16,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),b,c) &&
           has(scene.lost_contacts(),c,e) &&
           !has(scene.lost_contacts(),d,e) &&
           scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==2 &&
           scene.accurate_island_id(b)==old_handle &&
           scene.accurate_island_root(b)==b &&
           scene.accurate_island_root(e)==c &&
           scene.accurate_island_id(c)!=old_handle &&
           scene.accurate_island_id(c)==scene.accurate_island_id(e) &&
           scene.accurate_cached_route_hops(c)==0 &&
           scene.accurate_cached_route_hops(d)==1 &&
           scene.accurate_cached_route_hops(e)==2 &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()==splits+2,
           "Immediate source dirty split lost root, routes or handle");
    // Dirty bitmap B,C,E per graph. B is old root; C splits all of
    // C,D,E. E has already been witnessed and should *not* be searched.
    // A deferred split or a second island-wide traversal does not
    // implement this source third-pass contract.
    expect(scene.island_dirty_route_check_count()==checks+4,
           "Dirty orphan witnesses were re-searched after their split");
    scene.advance_collision();
    expect(scene.accurate_island_count()==2 &&
           scene.island_local_split_count()==splits+2 &&
           scene.island_full_rebuild_count()==rebuilds,
           "Quiet publication replayed the source dirty split");
}

// After two redundant chords disappear, the first dirty-node search
// reaches its original root through C while also visiting later-dirty D.
// Source unwindRoute repairs the A-C-B path, then processLostEdges
// updates D's off-path estimate via A and treats D as an established
// witness instead of searching D a second time.
void verify_successful_route_branch_witness() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,18,0},1);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.4f,18,0},1);
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,18,0},1);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.4f,18,0},1);
    scene.advance_collision();
    expect(scene.accurate_island_count()==1 &&
           scene.accurate_island_root(d)==b &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),b,d) &&
           has(scene.found_contacts(),a,d) &&
           has(scene.found_contacts(),a,c) &&
           has(scene.found_contacts(),b,c),
           "Shared-witness fixture lacks its redundant root chords");
    const auto checks=scene.island_dirty_route_check_count();
    const auto splits=scene.island_local_split_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    scene.set_position(b,{1.1f,18,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),a,b) &&
           has(scene.lost_contacts(),b,d) &&
           !has(scene.lost_contacts(),b,c) &&
           scene.accurate_island_count()==1 &&
           scene.speculative_island_count()==1 &&
           scene.accurate_island_root(d)==b &&
           scene.accurate_cached_route_hops(c)==1 &&
           scene.accurate_cached_route_hops(a)==2 &&
           scene.accurate_cached_route_hops(d)==3 &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()==splits,
           "Source successful route failed to update off-path witnesses");
    // A and B are processed in each graph; D was already reached in
    // A's heap search and is skipped by the source visited bitmap.
    expect(scene.island_dirty_route_check_count()==checks+4,
           "Successful source search repeated a witnessed dirty node");
}

// A split in one island must not abort the rest of either IslandSim's
// dirty bitmap. Replacing a temporary root chord in an unrelated island
// requires actual source findRoute repair, rather than just retaining the
// same island handle after the bounded component reconstruction.
void verify_route_repair_continues_after_split() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    // The initial two-node island chooses B (the second node) as its
    // PhysX merge-tie root. Losing A-B makes the earlier dirty A fail.
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,14,0},1.0f);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,14,0},1.0f);
    // A separate chain C-D-E-F has source root D. Its eventual D-F
    // shortcut sets F's hop to one; removing D-F leaves F-E-D intact,
    // requiring its cached hop to increase back to two.
    const auto c=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10,14,0},1.0f);
    const auto d=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{10.8f,14,0},1.0f);
    const auto e=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{11.6f,14,0},1.0f);
    const auto f=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{12.4f,14,0},1.0f);
    scene.advance_collision();
    expect(scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==2 &&
           scene.accurate_island_root(a)==b &&
           scene.accurate_island_root(f)==d &&
           scene.accurate_cached_route_hops(f)==2 &&
           has(scene.found_contacts(),a,b) &&
           has(scene.found_contacts(),c,d) &&
           has(scene.found_contacts(),d,e) &&
           has(scene.found_contacts(),e,f),
           "Multi-island dirty fixture did not establish source root hops");
    scene.set_position(f,{11.55f,14,0});
    scene.advance_collision();
    expect(has(scene.found_contacts(),d,f) &&
           scene.accurate_cached_route_hops(f)==1 &&
           scene.accurate_island_root(f)==d,
           "Added redundant D-F chord failed to shorten source hop");
    const auto checks=scene.island_dirty_route_check_count();
    const auto rebuilds=scene.island_full_rebuild_count();
    const auto local_splits=scene.island_local_split_count();
    const auto surviving=scene.accurate_island_id(d);
    scene.set_position(b,{4,14,0});
    scene.set_position(f,{12.4f,14,0});
    scene.advance_collision();
    expect(has(scene.lost_contacts(),a,b) &&
           has(scene.lost_contacts(),d,f) &&
           !has(scene.lost_contacts(),e,f) &&
           scene.accurate_island_count()==3 &&
           scene.speculative_island_count()==3 &&
           scene.accurate_island_id(d)==surviving &&
           scene.accurate_island_root(f)==d &&
           scene.island_full_rebuild_count()==rebuilds &&
           scene.island_local_split_count()>local_splits,
           "Concurrent source-local split rebuilt unrelated island graphs");
    // Four distinct dirty actors per graph: A fails, B and D are
    // roots, and F searches its surviving route via E. Both source
    // graphs must complete all four checks even when A fails.
    expect(scene.island_dirty_route_check_count()==checks+8 &&
           scene.accurate_cached_route_hops(f)==2,
           "First failed route aborted later source dirty-node repairs");
    scene.advance_collision();
    expect(scene.accurate_cached_route_hops(f)==2 &&
           scene.accurate_island_id(d)==surviving,
           "Source route repair did not survive quiet island publication");
}

// Source mDirtyMap begins a detached component's findRoute at its
// lowest dirty node, not its lowest BodyId. Source mVisitedNodes records
// discovered entries as they enter the hop-priority queue. In this
// arrangement B is the dirty endpoint but A is the smaller node ID;
// moving A's static manager ahead of B's is a real descriptor-order bug.
void verify_dirty_seed_split_manager_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene split{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1);
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-1.6f,10,0},1);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-0.8f,10,0},1);
    const auto sa=split.add_static_box(
        {0.1f,0.1f,0.1f},{-1.6f,10,0.54f});
    const auto sb=split.add_static_box(
        {0.1f,0.1f,0.1f},{-0.8f,10,0.54f});
    split.advance_collision();
    expect(split.accurate_island_count()==1 &&
           split.contacts().size()==6 &&
           has(split.found_contacts(),b,c) &&
           has(split.found_contacts(),a,sa) &&
           has(split.found_contacts(),b,sb),
           "Dirty-seed source split fixture did not connect six managers");

    // Move only the retained-root C-D-E side. The orphan's three
    // contact managers must survive with unchanged contact geometry.
    split.set_position(c,{1.0f,10,0});
    split.set_position(d,{1.8f,10,0});
    split.set_position(e,{2.6f,10,0});
    split.step(1.0f/480.0f);
    const auto gu=split.contacts();
    const auto indices=split.indexed_contact_manager_order();
    expect(split.accurate_island_count()==2 &&
           has(split.lost_contacts(),b,c) &&
           gu.size()==5 && indices.size()==5 &&
           split.accurate_island_root(a)==b &&
           split.accurate_island_root(b)==b &&
           split.accurate_island_root(c)==d &&
           split.accurate_island_root(d)==d,
           "Dirty-seed split did not preserve original root and promote B");

    std::vector<Pair> orphan_managers;
    for(std::size_t index:indices) {
        const Pair pair=gu[index].pair;
        if(pair==Pair{b,sb} || pair==Pair{a,sa} ||
           pair==Pair{a,b})
            orphan_managers.push_back(pair);
    }
    expect(orphan_managers.size()==3 &&
           orphan_managers[0]==Pair{b,sb} &&
           orphan_managers[1]==Pair{a,sa} &&
           orphan_managers[2]==Pair{a,b},
           "Split manager order ignored source first-dirty B visitation");
}

// PxsIslandSim::processLostEdges collects a separated component's
// managers by visiting its node adjacency, then splitEdges appends that
// collected sequence to its new island. Replaying globally preserved
// manager order gives a different finite-iteration PGS input here.
void verify_split_contact_manager_migration() {
    using fyzz::native::Scene;
    Scene split{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,10,0},1.0f);
    const auto b=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.8f,10,0},1.0f);
    const auto c=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{1.6f,10,0},1.0f);
    const auto d=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{2.4f,10,0},1.0f);
    const auto e=split.add_dynamic_box(
        {0.5f,0.5f,0.5f},{3.2f,10,0},1.0f);
    split.advance_collision();
    expect(split.contacts().size()==4 &&
           split.accurate_island_count()==1 &&
           has(split.contacts().size()?split.found_contacts():
                                     split.lost_contacts(),b,c),
           "Split manager fixture did not form its five-body chain");
    // Losing B-C leaves A-B on the original root-B island. The
    // orphan C-D-E side must migrate contacts in visited C,D,E
    // adjacency order: C-D followed by D-E. The source's earlier
    // array swap-last on B-C would leave D-E before C-D if we
    // simply filtered/replayed the old array.
    split.set_position(c,{2.0f,10,0});
    split.step(1.0f/480.0f);
    const auto managers=split.indexed_contact_manager_order();
    const auto gu=split.contacts();
    expect(split.accurate_island_count()==2 &&
           gu.size()==3 && managers.size()==3 &&
           gu[0].pair==fyzz::native::Pair{a,b} &&
           gu[1].pair==fyzz::native::Pair{c,d} &&
           gu[2].pair==fyzz::native::Pair{d,e} &&
           managers[0]==0 && managers[1]==1 && managers[2]==2,
           "Split migration replayed old manager order instead of visited edges");
}

// Source contact-edge ordering and removal conformance remains in its
// own stack frame, alongside the other multi-island lifecycle fixtures.
void verify_island_contact_manager_order() {
    using fyzz::native::Scene;
    // PhysX indexed-manager order belongs to the accurate island's
    // eCONTACT_MANAGER edge array, not the pair-sorted Gu stream.
    // Build C first, then B and D, producing source [C,B,D] against
    // canonical Gu [B,C,D]. Removing C must swap-last to [D,B], not
    // perform a stable erase or rebuild from canonical pair order.
    {
        Scene island_managers{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto center=island_managers.add_dynamic_box(
            {1,1,1},{0,10,0},1);
        const auto b=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{4,10,0});
        const auto c=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{-1.1f,10,0});
        const auto d=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{0,10,4});
        island_managers.advance_collision();
        expect(island_managers.contacts().size()==1 &&
               has(island_managers.found_contacts(),center,c),
               "Island manager fixture did not establish the first touch");
        island_managers.set_position(b,{1.1f,10,0});
        island_managers.set_position(d,{0,10,1.1f});
        island_managers.step(1.0f/480.0f);
        const auto initial=island_managers.indexed_contact_manager_order();
        const auto contacts=island_managers.contacts();
        expect(contacts.size()==3 && initial.size()==3 &&
               contacts[0].pair==fyzz::native::Pair{center,b} &&
               contacts[1].pair==fyzz::native::Pair{center,c} &&
               contacts[2].pair==fyzz::native::Pair{center,d} &&
               initial[0]==1 && initial[1]==0 && initial[2]==2,
               "Dy partitioning ignored island-owned manager insertion order");
        // A failed solver frame must restore the exact published manager
        // allocation, not just the same values. Admit an additional contact
        // in the *failed* frame so the transient indexed manager sequence
        // has four entries while the published sequence still has three.
        // The source material-combination overflow occurs in
        // prepare_constraints(), after partition_constraints() has rebuilt
        // this four-manager descriptor input.
        const auto published_pointer=initial.data();
        const auto published_contacts=contacts.data();
        const auto published_frame=island_managers.frame_index();
        const auto e=island_managers.add_static_box(
            {0.25f,0.25f,0.25f},{0,10,-1.1f});
        fyzz::Material overflowing{};
        overflowing.static_friction=1.0e30f;
        overflowing.dynamic_friction=1.0e30f;
        overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
        island_managers.set_material(center,overflowing);
        island_managers.set_material(b,overflowing);
        bool rejected=false;
        try { island_managers.step(1.0f/480.0f); }
        catch(const std::overflow_error& error) {
            rejected=std::string(error.what())==
                "Native Pxs material combination overflow";
        }
        const auto restored=island_managers.indexed_contact_manager_order();
        const auto restored_contacts=island_managers.contacts();
        expect(rejected &&
               island_managers.frame_index()==published_frame &&
               restored.data()==published_pointer &&
               restored.size()==3 &&
               restored[0]==1 && restored[1]==0 && restored[2]==2 &&
               restored_contacts.data()==published_contacts &&
               restored_contacts.size()==3,
               "Post-partition rejected PGS step invalidated published manager span");
        island_managers.set_material(center,fyzz::Material{});
        island_managers.set_material(b,fyzz::Material{});
        // The same pending authored fourth contact must publish on a
        // clean retry. This also proves the failed step would have rebuilt
        // a different (four-entry) indexed-manager stream before throwing.
        island_managers.step(1.0f/480.0f);
        const auto admitted=island_managers.indexed_contact_manager_order();
        const auto admitted_contacts=island_managers.contacts();
        const fyzz::native::Pair fourth_pair{std::min(center,e),
                                             std::max(center,e)};
        expect(admitted_contacts.size()==4 &&
               std::any_of(admitted_contacts.begin(),admitted_contacts.end(),
                   [&](const auto& contact) {
                       return contact.pair==fourth_pair;
                   }) &&
               admitted.size()==4 &&
               admitted[0]==1 && admitted[1]==0 &&
               admitted[2]==2 && admitted[3]==3,
               "Post-partition failure fixture did not admit its fourth manager");
        island_managers.set_position(center,{0,10,0});
        island_managers.set_position(c,{-4,10,0});
        island_managers.set_position(e,{0,10,-4});
        island_managers.step(1.0f/480.0f);
        const auto after=island_managers.indexed_contact_manager_order();
        const auto remaining=island_managers.contacts();
        expect(remaining.size()==2 && after.size()==2 &&
               remaining[0].pair==fyzz::native::Pair{center,b} &&
               remaining[1].pair==fyzz::native::Pair{center,d} &&
               after[0]==1 && after[1]==0,
               "PhysX island edge swap-last was replaced by stable order");
    }

    // PxsIslandSim::mergeEdges concatenates the retained winner's
    // manager array with the losing island's array before appending
    // the newly connected bridge. Equal-size merges preserve the
    // second island, independent of canonical Gu BodyId ordering.
    {
        Scene merge_order{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1);
        const auto b=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,10,0},1);
        const auto c=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,10,0},1);
        const auto d=merge_order.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3.8f,10,0},1);
        merge_order.advance_collision();
        expect(merge_order.contacts().size()==2 &&
               has(merge_order.found_contacts(),a,b) &&
               has(merge_order.found_contacts(),c,d),
               "Indexed manager merge fixture did not form separate islands");
        merge_order.set_position(c,{1.6f,10,0});
        merge_order.set_position(d,{2.4f,10,0});
        merge_order.step(1.0f/480.0f);
        const auto sequence=merge_order.indexed_contact_manager_order();
        const auto gu=merge_order.contacts();
        expect(gu.size()==3 && sequence.size()==3 &&
               gu[0].pair==fyzz::native::Pair{a,b} &&
               gu[1].pair==fyzz::native::Pair{b,c} &&
               gu[2].pair==fyzz::native::Pair{c,d} &&
               sequence[0]==2 && sequence[1]==0 && sequence[2]==1,
               "Source size-tied island merge did not concatenate manager order");
        // The tie retains C-D's D-root island. Source
        // mergeIslandsInternal computes extraPath =
        // hops[C](1) + hops[B](0) + 1 = 2, adds that to
        // the losing A-B island, then pins B to hops[C]+1.
        expect(merge_order.accurate_cached_route_hops(a)==3 &&
               merge_order.accurate_cached_route_hops(b)==2 &&
               merge_order.accurate_cached_route_hops(c)==1 &&
               merge_order.accurate_cached_route_hops(d)==0,
               "Source mergeIslandsInternal did not translate loser hop offsets");
    }

}

// Keep the long-running actor, sleep and rollback suites in a separate
// frame from the broadphase and constraint fixture groups on Windows.
// The source graph's rollback ownership is a cold-path concern. Steady
// PGS steps may update active-node publication and motion, but must not
// copy every persistent island node/handle array when no topology changes.
void verify_sparse_motion_checkpoint() {
    using namespace fyzz;
    using namespace fyzz::native;
    constexpr float dt=1.0f/120.0f;
    // Retired BodyIds remain allocated for stable identity. They must
    // not force historical actor scans or motion copies in every frame.
    Scene history{Scene::ContactGenerator::DiagnosticLegacyGu};
    for(unsigned i=0;i<512;++i) {
        const auto id=history.add_dynamic_box(
            {0.2f,0.2f,0.2f},{float(i)*3.0f,3.0f,0},1.0f);
        history.remove(id);
    }
    const auto live=history.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,3,0},1.0f);
    RigidBodyDynamics without_gravity{};
    without_gravity.gravity_enabled=false;
    history.set_dynamics(live,without_gravity);
    history.step(dt);
    expect(history.motion_rollback_journal_size()==1,
           "Retired rigid bodies expanded the per-step motion checkpoint");
    history.step(dt);
    expect(history.motion_rollback_journal_size()==1,
           "Quiet free-flight frame copied more than its active body");

    // A dormant scene needs no motion records. A newly admitted body
    // can subsequently wake an old sleeping neighbor, and both bodies
    // must roll back to their pre-step states if contact preparation fails.
    Scene wakes{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=wakes.add_static_floor(0.0f);
    const auto sleeper=wakes.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    for(int i=0;i<360;++i)wakes.step(dt);
    expect(wakes.body_is_sleeping(sleeper),
           "Sparse motion checkpoint fixture did not reach source sleep");
    wakes.step(dt);
    expect(wakes.motion_rollback_journal_size()==0,
           "Dormant island unnecessarily copied rigid-body motion");
    const auto newcomer=wakes.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.4f,0.49f,0},1.0f);
    const auto sleeping_before=wakes.body_state(sleeper);
    const auto newcomer_before=wakes.body_state(newcomer);
    Material exploding{};
    exploding.static_friction=1.0e30f;
    exploding.dynamic_friction=1.0e30f;
    exploding.friction_combine=MaterialCombineMode::Multiply;
    wakes.set_material(floor,exploding);
    wakes.set_material(newcomer,exploding);
    bool rejected=false;
    try {wakes.step(dt);}
    catch(const std::overflow_error&) {rejected=true;}
    const auto sleeping_after=wakes.body_state(sleeper);
    const auto newcomer_after=wakes.body_state(newcomer);
    expect(rejected && wakes.motion_rollback_journal_size()==2 &&
           sleeping_after.sleeping==sleeping_before.sleeping &&
           sleeping_after.position.x==sleeping_before.position.x &&
           sleeping_after.position.y==sleeping_before.position.y &&
           newcomer_after.position.x==newcomer_before.position.x &&
           newcomer_after.position.y==newcomer_before.position.y &&
           newcomer_after.velocity.y==newcomer_before.velocity.y,
           "Failed PGS admission lost the sparse source wake/motion rollback");
    wakes.set_material(floor,Material{});
    wakes.set_material(newcomer,Material{});
    wakes.step(dt);
    expect(!wakes.body_is_sleeping(sleeper) &&
           wakes.motion_rollback_journal_size()==2,
           "Retried new-contact wake did not journal both mutated rigid bodies");
}

void verify_demand_driven_island_ownership_checkpoint() {
    using fyzz::native::Scene;
    Scene quiet{Scene::ContactGenerator::DiagnosticLegacyGu};
    for(int i=0;i<24;++i)
        quiet.add_dynamic_box({0.5f,0.5f,0.5f},
                              {static_cast<float>(i*4),150,0},1.0f);
    quiet.advance_collision();
    const auto baseline=quiet.island_graph_checkpoint_count();
    const auto revision=quiet.island_topology_revision();
    const auto rebuilds=quiet.island_full_rebuild_count();
    expect(baseline>0 && quiet.accurate_island_count()==24,
           "Quiet checkpoint fixture did not admit all source islands");
    for(int frame=0;frame<80;++frame)
        quiet.step(1.0f/240.0f);
    expect(quiet.island_graph_checkpoint_count()==baseline &&
           quiet.island_full_rebuild_count()==rebuilds &&
           quiet.island_topology_revision()==revision &&
           quiet.accurate_island_count()==24,
           "Steady PGS copied persistent island topology without a mutation");
    const auto added=quiet.add_dynamic_box(
        {0.5f,0.5f,0.5f},{200,150,0},1.0f);
    quiet.advance_collision();
    expect(quiet.island_graph_checkpoint_count()==baseline+1 &&
           quiet.island_full_rebuild_count()==rebuilds+1 &&
           quiet.accurate_island_count()==25 &&
           quiet.accurate_island_root(added)==added,
           "Source node admission skipped the demand-driven ownership snapshot");
    const auto after=quiet.island_graph_checkpoint_count();
    quiet.advance_collision();
    expect(quiet.island_graph_checkpoint_count()==after,
           "Unchanged geometry republished a redundant graph checkpoint");
}

// Independent source edge-activity regressions live outside the full
// rigid integration fixture to keep their stack frames and failures isolated.
void verify_source_active_island_handoff() {
    using fyzz::native::Scene;
    Scene active{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=active.add_static_floor(0.0f);
    const auto a=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto b=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
    for(int frame=0;frame<360;++frame)
        active.step(1.0f/120.0f);
    expect(active.body_is_sleeping(a) && active.body_is_sleeping(b) &&
           active.active_accurate_islands().empty() &&
           active.active_speculative_islands().empty(),
           "Sleeping isolated rigid islands retained source active handles");
    // Source markIslandActive appends on the first genuine node wake.
    // Each independent graph owns its own active-index permutation.
    active.set_position(a,active.body_state(a).position);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_accurate_islands()[0]==active.accurate_island_id(a) &&
           active.active_speculative_islands().size()==1 &&
           active.active_speculative_islands()[0]==
               active.speculative_island_id(a),
           "Same-pose wake failed to append source island handles");
    const auto a_accurate=active.accurate_island_id(a);
    const auto a_speculative=active.speculative_island_id(a);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1,
           "Quiet publication duplicated already active source island");
    active.set_position(b,active.body_state(b).position);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands()[0]==a_speculative,
           "Additional awake island lost existing source activation order");
    const auto* accurate_published=active.active_accurate_islands().data();
    const auto* speculative_published=active.active_speculative_islands().data();
    const std::vector<std::uint32_t> accurate_before(
        active.active_accurate_islands().begin(),
        active.active_accurate_islands().end());
    const std::vector<std::uint32_t> speculative_before(
        active.active_speculative_islands().begin(),
        active.active_speculative_islands().end());
    bool rejected=false;
    try { active.step(std::numeric_limits<float>::max()); }
    catch(const std::overflow_error&) { rejected=true; }
    expect(rejected &&
           active.active_accurate_islands().data()==accurate_published &&
           active.active_speculative_islands().data()==speculative_published &&
           std::equal(accurate_before.begin(),accurate_before.end(),
                      active.active_accurate_islands().begin(),
                      active.active_accurate_islands().end()) &&
           std::equal(speculative_before.begin(),speculative_before.end(),
                      active.active_speculative_islands().begin(),
                      active.active_speculative_islands().end()),
           "Rejected source island handoff changed published storage/order");
    active.remove(b);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands().size()==1 &&
           active.active_speculative_islands()[0]==a_speculative,
           "Retiring rigid actor left stale mActiveIslands handles");
    const auto c=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{6,3,0},1.0f);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.active_accurate_islands()[0]==a_accurate &&
           active.active_speculative_islands()[0]==a_speculative &&
           active.accurate_island_id(c)!=a_accurate,
           "Actor admission rebuilt instead of preserving active island owner");
    const auto c_accurate=active.accurate_island_id(c);
    const auto c_speculative=active.speculative_island_id(c);
    // Retire the FIRST active island, not the last. Source
    // markIslandInactive replaces it with the final active handle.
    active.remove(a);
    active.advance_collision();
    expect(active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1 &&
           active.active_accurate_islands()[0]==c_accurate &&
           active.active_speculative_islands()[0]==c_speculative,
           "First-slot source active-island retirement broke swap-last");
    // Rejected actor admission rebuilds both IslandSims and changes their
    // visited-island scratch capacities. The old source permutation,
    // handles and scratch watermark must all survive and work on retry.
    const auto d=active.add_dynamic_box(
        {0.5f,0.5f,0.5f},{9,3,0},1.0f);
    bool failed_rebuild=false;
    try { active.step(std::numeric_limits<float>::max()); }
    catch(const std::overflow_error&) { failed_rebuild=true; }
    expect(failed_rebuild &&
           active.active_accurate_islands().size()==1 &&
           active.active_speculative_islands().size()==1 &&
           active.active_accurate_islands()[0]==c_accurate &&
           active.active_speculative_islands()[0]==c_speculative,
           "Rejected full island rebuild corrupted source active handles");
    active.advance_collision();
    expect(active.active_accurate_islands().size()==2 &&
           active.active_speculative_islands().size()==2 &&
           active.accurate_island_id(d)!=c_accurate &&
           active.speculative_island_id(d)!=c_speculative,
           "Retried source island rebuild reused stale active-list slots");
    (void)floor;
}

void verify_source_mixed_split_and_sleep_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    // Source mergeIslands keeps the SECOND handle on equal-sized
    // singletons. Admit the eventually sleeping rod first, then the
    // awake winner. Losing their SAP-only bridge must detach the
    // sleeping rod and append it to speculative mActiveIslands after
    // the independent ready island, exercising the pre-sync index.
    const auto dormant=scene.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,5},1.0f);
    const auto awake=scene.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,0},1.0f);
    for(const auto id:{awake,dormant}) {
        scene.set_velocity(id,{0,9.81f,0});
        scene.set_angular_velocity(id,{0,0.83f,0});
    }
    scene.step(1.0f);
    for(const auto id:{awake,dormant}) {
        scene.set_angular_velocity(id,{});
        scene.set_velocity(id,{});
    }
    scene.set_position(awake,{0.6f,0.1f,0.6f});
    scene.set_position(dormant,{0,0.1f,0});
    const auto floor=scene.add_static_floor(0.0f);
    scene.advance_collision();
    expect(has(scene.pairs(),awake,dormant) &&
           scene.accurate_island_count()==2 &&
           scene.speculative_island_count()==1,
           "Mixed source sleep fixture lacks its SAP-only bridge");
    bool dormant_ready=false;
    for(int frame=0;frame<160;++frame) {
        scene.set_velocity(awake,{0,0,0.25f});
        scene.step(1.0f/120.0f);
        if(scene.body_is_sleeping(dormant) &&
           !scene.body_is_sleeping(awake) &&
           scene.speculative_island_count()==1) {
            dormant_ready=true;
            break;
        }
    }
    expect(dormant_ready,
           "Mixed source fixture could not sleep one SAP-only endpoint");
    const auto independent=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{30,0.5f,0},1.0f);
    const Pair dormant_floor{dormant,floor};
    const Pair independent_floor{std::min(independent,floor),
                                 std::max(independent,floor)};
    bool third_pass_ready=false;
    for(int frame=0;frame<150;++frame) {
        scene.set_velocity(awake,{0,0,0.25f});
        scene.step(1.0f/120.0f);
        if(scene.body_ready_for_sleep(independent) &&
           !scene.body_is_sleeping(independent)) {
            third_pass_ready=true;
            break;
        }
    }
    expect(third_pass_ready,
           "Mixed source fixture independent island never became ready");
    expect(scene.body_is_sleeping(dormant),
           "Mixed source fixture dormant rod woke before independent readiness");
    expect(!scene.body_is_sleeping(awake),
           "Mixed source fixture bridge root slept before independent readiness");
    expect(has(scene.pairs(),awake,dormant),
           "Mixed source fixture lost its SAP-only bridge before splitting");
    expect(scene.speculative_contact_edge_active(dormant_floor),
           "Mixed source fixture dormant floor edge not speculative-active");
    expect(scene.speculative_contact_edge_active(independent_floor),
           "Mixed source fixture independent floor edge not speculative-active");
    const auto dormant_edge=scene.contact_edge_index(dormant_floor);
    const auto independent_edge=scene.contact_edge_index(independent_floor);
    expect(dormant_edge!=std::numeric_limits<std::uint32_t>::max() &&
           independent_edge!=std::numeric_limits<std::uint32_t>::max(),
           "Mixed source fixture lost its floor-manager source handles");
    // This collision-only handoff must split the sleeping component
    // while Dy's previously ready independent node deactivates.
    // Source speculative reverse mActiveIslands visits the appended
    // split handle BEFORE the still-indexed independent island.
    scene.set_position(awake,{12,0.1f,12});
    scene.advance_collision();
    const auto events=scene.deactivating_speculative_contact_edges();
    const auto a=std::find(events.begin(),events.end(),dormant_edge);
    const auto b=std::find(events.begin(),events.end(),independent_edge);
    expect(a!=events.end(),
           "Mixed source split did not publish dormant component floor edge");
    expect(b!=events.end(),
           "Mixed source third pass missed independent ready floor edge");
    expect(a<b,
           "Mixed source split/sleep reversed pre-reconciliation speculative order");
    expect(scene.body_is_sleeping(dormant) &&
           scene.body_is_sleeping(independent),
           "Mixed source third pass did not retire both independent sleepers");
    expect(scene.speculative_island_count()==3,
           "Mixed source split did not detach sleeping speculative component");
    scene.advance_collision();
    expect(scene.deactivating_speculative_contact_edges().empty() &&
           !scene.speculative_contact_edge_active(dormant_floor) &&
           !scene.speculative_contact_edge_active(independent_floor),
           "Mixed split/sleep replayed the one-shot speculative events");
}

void verify_source_speculative_reverse_sleep_order() {
    using fyzz::native::Scene;
    using fyzz::native::Pair;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=scene.add_dynamic_box({2.0f,0.1f,0.2f},{0,10,0},1.0f);
    const auto b=scene.add_dynamic_box({2.0f,0.1f,0.2f},{20,10,0},1.0f);
    const auto c=scene.add_dynamic_box({2.0f,0.1f,0.2f},{0,10,5},1.0f);
    // Two long rotated boxes overlap in SAP, but the Gu boxes remain
    // separate. Source speculative merge swaps the first active handle
    // with the last while accurate singleton handles retain their order.
    for(const auto id:{a,b,c}) {
        scene.set_velocity(id,{0,9.81f,0});
        scene.set_angular_velocity(id,{0,0.83f,0});
    }
    scene.step(1.0f);
    for(const auto id:{a,b,c}) {
        scene.set_angular_velocity(id,{});
        scene.set_velocity(id,{});
    }
    scene.set_position(a,{0,0.1f,0});
    scene.set_position(b,{20,0.1f,0});
    scene.set_position(c,{0.6f,0.1f,0.6f});
    const auto floor=scene.add_static_floor(0.0f);
    scene.advance_collision();
    expect(has(scene.pairs(),a,c) &&
           !has(scene.found_contacts(),a,c) &&
           scene.accurate_island_count()==3 &&
           scene.speculative_island_count()==2 &&
           scene.active_accurate_islands().size()==3 &&
           scene.active_speculative_islands().size()==2,
           "Speculative sleep fixture did not establish SAP-only source merge");

    const std::array<fyzz::BodyId,3> bodies{a,b,c};
    const std::array<std::uint32_t,3> floor_edges{
        scene.contact_edge_index({a,floor}),
        scene.contact_edge_index({b,floor}),
        scene.contact_edge_index({c,floor})};
    for(const auto edge:floor_edges)
        expect(edge!=std::numeric_limits<std::uint32_t>::max(),
               "Speculative sleep fixture lacks an active floor manager");
    std::array<std::uint32_t,3> accurate_handles{};
    for(std::size_t i=0;i<bodies.size();++i)
        accurate_handles[i]=scene.accurate_island_id(bodies[i]);

    // Verify that the fixture would fail with candidate-order
    // publication. An accurate reverse traversal must yield a different
    // speculative order from the speculative reverse active list.
    std::vector<std::uint32_t> accurate_derived;
    for(auto it=scene.active_accurate_islands().rbegin();
        it!=scene.active_accurate_islands().rend();++it)
        for(std::size_t j=0;j<bodies.size();++j)
            if(*it==accurate_handles[j]) {
                const auto handle=scene.speculative_island_id(bodies[j]);
                if(std::find(accurate_derived.begin(),
                             accurate_derived.end(),handle)==
                   accurate_derived.end())
                    accurate_derived.push_back(handle);
            }
    const std::vector<std::uint32_t> source_reverse{
        scene.active_speculative_islands().rbegin(),
        scene.active_speculative_islands().rend()};
    expect(accurate_derived!=source_reverse &&
           source_reverse.size()==2,
           "Speculative sleep fixture does not discriminate graph ordering");

    bool retired=false;
    for(int frame=0;frame<360;++frame) {
        scene.step(1.0f/120.0f);
        const auto events=scene.deactivating_speculative_contact_edges();
        if(events.empty())continue;
        expect(!retired,"Speculative source sleep re-published retired edges");
        std::vector<std::uint32_t> observed;
        for(const auto edge:events)
            for(std::size_t i=0;i<bodies.size();++i)
                if(edge==floor_edges[i]) {
                    const auto handle=scene.speculative_island_id(bodies[i]);
                    if(std::find(observed.begin(),observed.end(),handle)==
                       observed.end())
                        observed.push_back(handle);
                }
        expect(observed==source_reverse &&
               scene.body_is_sleeping(a) &&
               scene.body_is_sleeping(b) &&
               scene.body_is_sleeping(c),
               "Source speculative deactivation ignored reverse active-island order");
        retired=true;
    }
    expect(retired && scene.active_speculative_islands().empty(),
           "Speculative source island retirement did not complete");
    scene.advance_collision();
    expect(scene.deactivating_speculative_contact_edges().empty(),
           "Quiet source publication repeated speculative retirement");
}

void verify_source_correlated_friction_patch_lifecycle() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=scene.add_static_floor(0.0f);
    const auto left=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{-3.0f,0.49f,0.0f},1.0f);
    const auto right=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{3.0f,0.49f,0.0f},1.0f);
    constexpr float dt=1.0f/120.0f;
    scene.step(dt);
    scene.step(dt);
    const auto& contacts=scene.contacts();
    const auto patches=scene.solved_friction_patches();
    expect(contacts.size()==2 && patches.size()==2,
           "Dy correlation did not publish two independent source managers");
    expect(contacts[0].count>=2 && contacts[1].count>=2,
           "Dy correlation source fixture lacks multipoint Gu contacts");
    expect(patches[0].anchor_count==2 && patches[1].anchor_count==2,
           "Dy correlation failed to grow two source friction anchors");
    // A patch may legitimately break in the final velocity solve
    // while correcting a deeply penetrating initial contact. The
    // source caches only unbroken anchors; this fixture asserts that
    // fresh two-anchor contact correlation survives either outcome,
    // rather than assuming the one-frame Coulomb limit was not reached.
    // eDISABLE_FRICTION disables all tangent rows on precisely one
    // manager without removing that manager's normal contact header.
    // Cached strong-friction anchors of the other manager must survive
    // independent shape-material authoring and the compact source
    // friction patch stream must remain in Gu pair order.
    fyzz::Material disabled{};
    disabled.disable_friction=true;
    scene.set_material(left,disabled);
    scene.step(dt);
    const auto disabled_patches=scene.solved_friction_patches();
    const auto impulses=scene.solved_normal_impulses();
    expect(disabled_patches.size()==2 &&
           disabled_patches[0].anchor_count==0 &&
           disabled_patches[1].anchor_count==2 &&
           scene.contacts().size()==2,
           "Source cached friction correlation leaked disabled anchor rows");
    for(const auto& manager:scene.contacts()) {
        double force=0;
        for(std::size_t j=0;j<manager.count;++j)
            force+=impulses[manager.contact_offset+j];
        expect(std::isfinite(force) && force>0.0,
               "Source friction patch compaction lost Gu-indexed normal writeback");
    }
    scene.set_material(left,fyzz::Material{});
    scene.step(dt);
    const auto restored=scene.solved_friction_patches();
    expect(restored.size()==2 &&
           restored[0].anchor_count==2 &&
           restored[1].anchor_count==2,
           "Source patch correlation failed to rebuild missing friction anchors");
    // A quiet publication removes transient force writeback, but does
    // not invalidate the prior source friction cache or manager identity.
    const auto a_edge=scene.contact_edge_index({floor,left});
    const auto b_edge=scene.contact_edge_index({floor,right});
    scene.advance_collision();
    expect(scene.solved_normal_impulses().empty() &&
           scene.contact_edge_index({floor,left})==a_edge &&
           scene.contact_edge_index({floor,right})==b_edge,
           "Quiet Gu publication changed source persistent contact managers");
    scene.step(dt);
    expect(scene.solved_friction_patches().size()==2 &&
           scene.solved_friction_patches()[0].anchor_count==2 &&
           scene.solved_friction_patches()[1].anchor_count==2,
           "Source friction cache failed across a quiet contact publication");
}

void verify_source_solver_task_batch_partitioning() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    scene.add_static_floor(0.0f);
    // DyDynamicsContext::updatePostKinematic uses the default
    // solverBatchSize=128 and requires at least one contact before
    // ending a task group. These 130 independent rigid contact islands
    // must form two descriptor groups, not one global partition zero.
    constexpr std::size_t bodies=130;
    for(std::size_t i=0;i<bodies;++i)
        scene.add_dynamic_box({0.5f,0.5f,0.5f},
            {static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    scene.step(1.0f/120.0f);
    expect(scene.contacts().size()==bodies &&
           scene.indexed_contact_manager_order().size()==bodies &&
           scene.solver_partition_count()==2,
           "Source Dy task-group boundary failed to partition 128+2 islands");
    const auto impulses=scene.solved_normal_impulses();
    std::size_t total=0;
    for(std::size_t i=0;i<scene.contacts().size();++i) {
        const auto& manager=scene.contacts()[i];
        expect(manager.contact_offset==total &&
               total+manager.count<=impulses.size(),
               "Task-group source descriptor writeback lost its Gu offset");
        double force=0.0;
        for(std::size_t j=0;j<manager.count;++j)
            force+=impulses[total+j];
        expect(std::isfinite(force) && force>0.0,
               i>=128
                   ? "Final source task group dropped rigid solver work"
                   : "First source task group dropped rigid solver work");
        total+=manager.count;
    }
    expect(total==impulses.size(),
           "Source task groups did not write all rigid contact impulses");
    scene.step(1.0f/120.0f);
    expect(scene.solver_partition_count()==2 &&
           scene.contacts().size()==bodies,
           "Source task-group boundaries leaked partition state across steps");
}

void verify_source_reverse_active_island_sleep_retirement() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    scene.add_static_floor(0.0f);
    std::array<fyzz::BodyId,4> bodies{};
    for(std::size_t i=0;i<bodies.size();++i)
        bodies[i]=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},
            {static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    scene.advance_collision();
    expect(scene.active_accurate_islands().size()==bodies.size(),
           "Sleep order fixture failed to publish independent active islands");
    bool deactivated=false;
    for(int frame=0;frame<360;++frame) {
        const std::vector<std::uint32_t> before(
            scene.active_accurate_islands().begin(),
            scene.active_accurate_islands().end());
        scene.step(1.0f/120.0f);
        const auto retired=scene.deactivating_accurate_nodes();
        if(retired.empty())continue;
        expect(!deactivated && retired.size()==bodies.size() &&
               before.size()==bodies.size(),
               "Source independent sleep islands did not retire together");
        // PxsIslandSim::processLostEdges walks mActiveIslands backwards.
        // The old forward scan of the full island-handle table must
        // fail this assertion even though it sleeps the same bodies.
        for(std::size_t i=0;i<retired.size();++i)
            expect(scene.accurate_island_id(retired[i])==
                   before[before.size()-i-1],
                   "Third-pass deactivation lost source reverse active-island order");
        expect(scene.active_accurate_islands().empty() &&
               scene.active_speculative_islands().empty(),
               "Third-pass sleep left stale published active islands");
        deactivated=true;
    }
    expect(deactivated,
           "Source reverse active-island retirement never reached sleep");
    scene.advance_collision();
    expect(scene.deactivating_accurate_nodes().empty() &&
           scene.active_accurate_islands().empty(),
           "Quiet publication replayed already retired active islands");
}

void verify_source_reverse_wake_order_and_rollback_capacity() {
    using fyzz::native::Scene;
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    scene.add_static_floor(0.0f);
    std::array<fyzz::BodyId,4> ids{};
    for(std::size_t i=0;i<ids.size();++i)
        ids[i]=scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{static_cast<float>(i)*3.0f,0.5f,0},1.0f);
    for(int frame=0;frame<360;++frame)
        scene.step(1.0f/120.0f);
    expect(scene.active_accurate_islands().empty() &&
           scene.active_speculative_islands().empty(),
           "Reverse wake fixture did not retire the dormant source islands");
    std::array<std::uint32_t,4> accurate{},speculative{};
    for(std::size_t i=0;i<ids.size();++i) {
        accurate[i]=scene.accurate_island_id(ids[i]);
        speculative[i]=scene.speculative_island_id(ids[i]);
        expect(scene.body_is_sleeping(ids[i]),
               "Reverse wake fixture did not put all bodies to sleep");
    }
    // Queue multiple distinct authored wakes before a single source
    // wakeIslands publication, explicitly against the sorted BodyId order.
    // Reject the publication first: its rollback must restore both
    // graphs' pending event sequences and reserve no-throw list capacity.
    const std::array<std::size_t,4> order{3,1,2,0};
    // Exercise all four public wake sources. The event sequence must
    // survive the failed step exactly as authored, rather than silently
    // falling back to accurate graph handle or BodyId ordering.
    scene.set_velocity(ids[3],{});
    scene.set_angular_velocity(ids[1],{});
    scene.apply_torque_impulse(ids[2],{});
    scene.set_position(ids[0],scene.body_state(ids[0]).position);
    bool rejected=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {rejected=true;}
    expect(rejected &&
           scene.active_accurate_islands().empty() &&
           scene.active_speculative_islands().empty(),
           "Rejected multi-wake step prematurely published source islands");
    scene.advance_collision();
    const auto actual=scene.active_accurate_islands();
    const auto sap=scene.active_speculative_islands();
    expect(actual.size()==order.size() && sap.size()==order.size(),
           "Wake events omitted some previously dormant island");
    for(std::size_t i=0;i<order.size();++i)
        expect(actual[i]==accurate[order[i]] &&
               sap[i]==speculative[order[i]],
               "Source mActivatingNodes order lost in graph-specific handoff");
    scene.advance_collision();
    for(std::size_t i=0;i<order.size();++i)
        expect(scene.active_accurate_islands()[i]==accurate[order[i]] &&
               scene.active_speculative_islands()[i]==speculative[order[i]],
               "Quiet handoff duplicated or sorted source active handles");
}

void verify_source_contact_interaction_consumption() {
    using fyzz::native::Scene;
    // Sc::InteractionFlag::eIS_ACTIVE follows the speculative graph,
    // not the accurate Gu contact or an actor's individual sleep flag.
    Scene scene{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=scene.add_static_floor(0.0f);
    const auto a=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
    const auto b=scene.add_dynamic_box(
        {0.5f,0.5f,0.5f},{4,0.5f,0},1.0f);
    const fyzz::native::Pair floor_a{std::min(a,floor),std::max(a,floor)};
    scene.advance_collision();
    expect(scene.speculative_contact_edge_active(floor_a) &&
           scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==2,
           "Sc activation did not consume source speculative contact edges");
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==2,
           "Sc interaction was spuriously reactivated on a quiet query");
    for(int i=0;i<360;++i)scene.step(1.0f/120.0f);
    expect(scene.body_is_sleeping(a) &&
           !scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==0 &&
           !scene.speculative_contact_edge_active(floor_a),
           "Sc third-pass failed to consume sleeping speculative edges");
    // Wake only one dormant island; its other floor interaction must
    // remain inactive and cannot be woken by accurate island ordering.
    scene.set_velocity(a,{});
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Sc speculative interaction did not reactivate on authored wake");
    // Force a Dy overflow after a published interaction transition.
    scene.set_position(a,{0,3,0});
    bool rejected=false;
    try {scene.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {rejected=true;}
    expect(rejected && scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Rejected Sc pipeline step corrupted interaction activity");
    scene.advance_collision();
    expect(scene.contact_interaction_active(floor_a) &&
           !scene.contact_edge_connected(floor_a) &&
           scene.active_contact_interaction_count()==1,
           "Gu-only touch loss incorrectly retired the speculative interaction");
    // A plane has unbounded broadphase coverage. Removing it, not
    // translating the dynamic box, actually destroys the SAP overlap.
    scene.remove(floor);
    scene.advance_collision();
    expect(!scene.contact_interaction_active(floor_a) &&
           scene.active_contact_interaction_count()==0,
           "Destroyed source overlap retained stale Sc interaction activity");

    // A rejected frame can discover a NEW contact-manager EdgeIndex,
    // activate its speculative interaction, then throw in Dy. Rollback
    // must not publish either the new manager or the interaction count;
    // retry must admit the same logical contact exactly once.
    Scene admission{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto wall=admission.add_static_box(
        {0.5f,0.5f,0.5f},{0,0.5f,0});
    const auto box=admission.add_dynamic_box(
        {0.5f,0.5f,0.5f},{5,0.5f,0},1.0f);
    const fyzz::native::Pair new_pair{
        std::min(wall,box),std::max(wall,box)};
    admission.advance_collision();
    expect(admission.active_contact_interaction_count()==0 &&
           !admission.contact_interaction_active(new_pair),
           "Sc new-contact rollback fixture started with active interaction");
    admission.set_position(box,{0.9f,0.5f,0});
    bool admission_failed=false;
    try {admission.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {admission_failed=true;}
    expect(admission_failed &&
           admission.active_contact_interaction_count()==0 &&
           !admission.contact_interaction_active(new_pair) &&
           admission.contact_edge_index(new_pair)==
               std::numeric_limits<std::uint32_t>::max(),
           "Failed first-touch frame published Sc interaction prematurely");
    admission.advance_collision();
    expect(admission.active_contact_interaction_count()==1 &&
           admission.contact_interaction_active(new_pair) &&
           admission.speculative_contact_edge_active(new_pair),
           "Retried first-touch failed to consume source edge activation");
}

void verify_source_active_reference_lifecycle() {
    using fyzz::native::Scene;
    Scene refs{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto floor=refs.add_static_floor(0.0f);
    const auto a=refs.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.0f,0.5f,0},1.0f);
    const auto b=refs.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1.0f);
    const fyzz::native::Pair pair{std::min(a,b),std::max(a,b)};
    refs.advance_collision();
    const auto handle=refs.contact_edge_index(pair);
    const auto valid=std::numeric_limits<std::uint32_t>::max();
    expect(handle!=valid &&
           refs.speculative_contact_edge_active(pair) &&
           refs.accurate_contact_edge_active(pair) &&
           refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1 &&
           refs.speculative_active_reference_count(floor)==0 &&
           refs.accurate_active_reference_count(floor)==0,
           "Source dynamic contact references incorrectly include static edges");
    // A persistent active edge cannot double-count its endpoints during
    // unchanged collision publications or another dynamic actor's entry.
    refs.advance_collision();
    refs.add_dynamic_box({0.5f,0.5f,0.5f},{50,50,0},1.0f);
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Source active-edge references drifted on quiet/rebuilt graph");
    refs.set_position(b,{6,0.5f,0});
    bool failed=false;
    try {refs.step(std::numeric_limits<float>::max());}
    catch(const std::overflow_error&) {failed=true;}
    expect(failed && refs.contact_edge_index(pair)==handle &&
           refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Rejected lost-contact step corrupted source endpoint references");
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==0 &&
           refs.speculative_active_reference_count(b)==0 &&
           refs.accurate_active_reference_count(a)==0 &&
           refs.accurate_active_reference_count(b)==0,
           "Retired dynamic contact did not release both source references");
    refs.set_position(b,{0.85f,0.5f,0});
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==1 &&
           refs.speculative_active_reference_count(b)==1 &&
           refs.accurate_active_reference_count(a)==1 &&
           refs.accurate_active_reference_count(b)==1,
           "Reconnected source manager did not reacquire endpoint references");
    refs.remove(b);
    refs.advance_collision();
    expect(refs.speculative_active_reference_count(a)==0 &&
           refs.accurate_active_reference_count(a)==0,
           "Actor retirement left stale source endpoint references");
}

void verify_source_speculative_only_active_references() {
    using fyzz::native::Scene;
    Scene separated{Scene::ContactGenerator::DiagnosticLegacyGu};
    const auto a=separated.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,0},1.0f);
    const auto b=separated.add_dynamic_box(
        {2.0f,0.1f,0.2f},{0,10,5},1.0f);
    for(fyzz::BodyId id:{a,b}) {
        separated.set_velocity(id,{0,9.81f,0});
        separated.set_angular_velocity(id,{0,0.83f,0});
    }
    separated.step(1.0f);
    for(fyzz::BodyId id:{a,b}) {
        separated.set_velocity(id,{});
        separated.set_angular_velocity(id,{});
    }
    separated.set_position(a,{0,0.1f,0});
    separated.set_position(b,{0.6f,0.1f,0.6f});
    separated.advance_collision();
    const fyzz::native::Pair candidate{a,b};
    expect(has(separated.pairs(),a,b) &&
           !has(separated.found_contacts(),a,b) &&
           separated.speculative_island_count()==1 &&
           separated.accurate_island_count()==2 &&
           separated.speculative_active_reference_count(a)==1 &&
           separated.speculative_active_reference_count(b)==1 &&
           separated.accurate_active_reference_count(a)==0 &&
           separated.accurate_active_reference_count(b)==0,
           "SAP-only source contact references leaked into accurate graph");
    separated.set_position(b,{12.0f,0.1f,12.0f});
    separated.advance_collision();
    expect(has(separated.lost_pairs(),a,b) &&
           separated.speculative_active_reference_count(a)==0 &&
           separated.speculative_active_reference_count(b)==0 &&
           separated.accurate_active_reference_count(a)==0 &&
           separated.accurate_active_reference_count(b)==0,
           "SAP-only source reference retirement missed the lost edge");
}

void verify_source_retired_bridge_deactivation() {
    using fyzz::native::Scene;
    {
        // Retiring an awake actor from an otherwise dormant SAP-only
        // bridge invokes the FULL island rebuild in the first pass.
        // The retained floor manager must still publish one speculative
        // deactivation in the third pass, without requiring a new
        // accurate sleep event or a local lost-edge split.
        Scene retired_bridge{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto sleeping_rod=retired_bridge.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto active_rod=retired_bridge.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(fyzz::BodyId id:{sleeping_rod,active_rod}) {
            retired_bridge.set_velocity(id,{0,9.81f,0});
            retired_bridge.set_angular_velocity(id,{0,0.83f,0});
        }
        retired_bridge.step(1.0f);
        for(fyzz::BodyId id:{sleeping_rod,active_rod}) {
            retired_bridge.set_angular_velocity(id,{});
            retired_bridge.set_velocity(id,{});
        }
        retired_bridge.set_position(sleeping_rod,{0,0.1f,0});
        retired_bridge.set_position(active_rod,{0.6f,0.1f,0.6f});
        const auto floor=retired_bridge.add_static_floor(0.0f);
        retired_bridge.advance_collision();
        bool independent_sleep=false;
        for(int frame=0;frame<160;++frame) {
            retired_bridge.set_velocity(active_rod,{0,0,0.25f});
            retired_bridge.step(1.0f/120.0f);
            if(retired_bridge.body_is_sleeping(sleeping_rod) &&
               !retired_bridge.body_is_sleeping(active_rod) &&
               retired_bridge.speculative_island_count()==1 &&
               retired_bridge.accurate_island_count()==2) {
                independent_sleep=true;
                break;
            }
        }
        const fyzz::native::Pair floor_pair{
            std::min(floor,sleeping_rod),std::max(floor,sleeping_rod)};
        const auto floor_edge=retired_bridge.contact_edge_index(floor_pair);
        const auto prior_spec_count=
            retired_bridge.active_speculative_contact_edge_count();
        if(!independent_sleep) {
            const auto a=retired_bridge.body_state(sleeping_rod);
            const auto b=retired_bridge.body_state(active_rod);
            std::cerr<<"N1CACHE sleep fixture: sleeping="
                     <<a.sleeping<<" active="<<b.sleeping
                     <<" speculative="<<retired_bridge.speculative_island_count()
                     <<" accurate="<<retired_bridge.accurate_island_count()
                     <<" quiet_vel=("<<a.velocity.x<<","
                     <<a.velocity.y<<","<<a.velocity.z<<")"
                     <<" moving_vel=("<<b.velocity.x<<","
                     <<b.velocity.y<<","<<b.velocity.z<<")"
                     <<" quiet_angular=("<<a.angular_velocity.x<<","
                     <<a.angular_velocity.y<<","<<a.angular_velocity.z<<")"
                     <<" quiet_y="<<a.position.y
                     <<" wake_counter="<<retired_bridge.body_wake_counter(sleeping_rod)
                     <<" ready_for_sleep="<<retired_bridge.body_ready_for_sleep(sleeping_rod)
                     <<" contacts="<<retired_bridge.contacts().size()
                     <<" pairs="<<retired_bridge.pairs().size()
                     <<" floor_active="<<retired_bridge.contact_interaction_active(floor_pair)
                     <<'\n';
        }
        expect(independent_sleep &&
               floor_edge!=std::numeric_limits<std::uint32_t>::max() &&
               retired_bridge.speculative_contact_edge_active(floor_pair) &&
               retired_bridge.contact_interaction_active(floor_pair) &&
               !retired_bridge.accurate_contact_edge_active(floor_pair),
               "Retirement fixture did not establish independent island sleep");
        retired_bridge.remove(active_rod);
        retired_bridge.advance_collision();
        const auto events=retired_bridge.deactivating_speculative_contact_edges();
        expect(retired_bridge.body_is_sleeping(sleeping_rod) &&
               !retired_bridge.speculative_contact_edge_active(floor_pair) &&
               !retired_bridge.contact_interaction_active(floor_pair) &&
               std::count(events.begin(),events.end(),floor_edge)==1 &&
               retired_bridge.deactivating_accurate_contact_edges().empty() &&
               prior_spec_count>0 &&
               retired_bridge.active_speculative_contact_edge_count()==0,
               "Full rebuild skipped speculative retirement of dormant manager");
        retired_bridge.advance_collision();
        expect(retired_bridge.deactivating_speculative_contact_edges().empty() &&
               !retired_bridge.speculative_contact_edge_active(floor_pair),
               "Retired-actor rebuild replayed speculative deactivation");
    }

}
void verify_source_activity_rollback() {
    using fyzz::native::Scene;
    {
        // A failed Gu touch-loss must restore the active contact-manager
        // bits as well as each IslandSim's counts and indexed worklists.
        // Structural EdgeUndo snapshots are taken *after* the edge's
        // activity was cleared; they cannot supersede ActivityUndo.
        Scene activity_rollback{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=activity_rollback.add_static_floor(0.0f);
        const auto body=activity_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const fyzz::native::Pair touch{std::min(floor,body),std::max(floor,body)};
        activity_rollback.advance_collision();
        const auto original_handle=activity_rollback.contact_edge_index(touch);
        const auto accurate_active=
            activity_rollback.active_accurate_contact_edge_count();
        const auto speculative_active=
            activity_rollback.active_speculative_contact_edge_count();
        expect(activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch),
               "Source activity rollback fixture failed to create active edge");
        activity_rollback.set_position(body,{0,3,0});
        bool failed=false;
        try {activity_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed &&
               activity_rollback.contact_edge_index(touch)==original_handle &&
               activity_rollback.contact_edge_connected(touch) &&
               activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch) &&
               activity_rollback.active_accurate_contact_edge_count()==
                   accurate_active &&
               activity_rollback.active_speculative_contact_edge_count()==
                   speculative_active,
               "Failed touch-loss step did not restore source edge activity");
        activity_rollback.advance_collision();
        expect(!activity_rollback.contact_edge_connected(touch) &&
               !activity_rollback.accurate_contact_edge_active(touch) &&
               activity_rollback.speculative_contact_edge_active(touch) &&
               activity_rollback.active_accurate_contact_edge_count()+1==
                   accurate_active &&
               activity_rollback.active_speculative_contact_edge_count()==
                   speculative_active,
               "Retried touch loss did not retire accurate activity exactly once");
    }

}

void verify_native_rigid_integration_and_island_lifecycle() {
    using fyzz::native::Scene;
    // End-to-end native CPU execution, without a reference PxScene.
    {
        Scene falling_scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        falling_scene.add_static_floor(0);
        const auto falling = falling_scene.add_dynamic_box(
            {0.25f,0.25f,0.25f}, {0,2,0}, 1);
        for (int frame = 0; frame < 360; ++frame)
            falling_scene.step(1.0f / 120.0f);
        const auto state = falling_scene.body_state(falling);
        if (!(std::isfinite(state.position.y) &&
              std::abs(state.position.y - 0.25f) < 0.06f &&
              std::abs(state.velocity.y) < 0.6f))
            std::fprintf(stderr,
                "floor y=%g vy=%g spin=(%g,%g,%g) quat=(%g,%g,%g,%g) contacts=%zu\n",
                state.position.y,state.velocity.y,
                state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z,
                state.orientation.x,state.orientation.y,state.orientation.z,state.orientation.w,
                falling_scene.contacts().size());
        expect(std::isfinite(state.position.y) &&
               std::abs(state.position.y - 0.25f) < 0.06f &&
               std::abs(state.velocity.y) < 0.6f,
               "Native CPU floor contact did not settle");
        expect(falling_scene.frame_index() == 360 &&
               falling_scene.contacts().size() == 1,
               "Native step must publish exactly one collision frame");
        falling_scene.remove(falling);
        falling_scene.step(1.0f / 120.0f);
        expect(falling_scene.contacts().empty() &&
               !falling_scene.lost_contacts().empty(),
               "Native CPU actor retirement failed to publish touch loss");
    }
    {
        // The provisional translation-only mass admission must not
        // silently turn a constrained rigid body into a rotation-locked
        // one when source PxReal angular inertia is not representable.
        Scene heavy{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto obstacle=heavy.add_static_box({1,1,1},{0,5,0});
        const auto body = heavy.add_dynamic_box(
            {1,1,1},{1.5f,5,0},std::numeric_limits<float>::max());
        heavy.set_velocity(body,{-1,0,0});
        bool rejected_contact=false;
        try { heavy.step(1.0f/120.0f); }
        catch(const std::logic_error&) { rejected_contact=true; }
        expect(rejected_contact &&
               heavy.frame_index()==0 &&
               heavy.body_state(body).position.x==1.5f &&
               heavy.body_state(body).velocity.x==-1.0f,
               "Nonrepresentable heavy-body contact was solved or leaked state");
        bool angular_rejected = false;
        try { heavy.apply_torque_impulse(body,{1,0,0}); }
        catch(const std::logic_error&) { angular_rejected = true; }
        expect(angular_rejected,
               "Heavy body with nonrepresentable angular inertia accepted torque");
        heavy.remove(obstacle);
        heavy.step(1.0f/120.0f);
        expect(heavy.body_state(body).position.x<1.5f &&
               std::isfinite(heavy.body_state(body).velocity.x),
               "Admitted free translation failed after unsupported contact rejection");
    }

    {
        Scene stacked{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto ground=stacked.add_static_box({2,0.5f,2},{0,-0.5f,0});
        const auto lower = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,2,0},1);
        const auto upper = stacked.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,3.5f,0},1);
        // Source PGS, independently reproduced on Linux and Windows:
        // at frame 86, the lower box/ground patch saturates during a
        // position pass, then recovers (no clamp) during final velocity
        // writeback. The previous "broken |= clamped" across ALL passes
        // incorrectly reports a broken patch. Its local anchors must
        // also remain eligible for reuse in the following frame.
        const fyzz::native::Pair support_pair{ground,lower};
        std::array<fyzz::Vec3,2> saved_support_anchors{};
        std::uint8_t saved_anchor_count=0;
        for (int frame=0;frame<600;++frame) {
            stacked.step(1.0f/120.0f);
            if (stacked.frame_index()==86 ||
                stacked.frame_index()==87) {
                const auto reports=stacked.solved_friction_patches();
                const auto it=std::find_if(reports.begin(),reports.end(),
                    [&](const auto& patch) {
                        return patch.pair==support_pair;
                    });
                expect(it!=reports.end() && it->anchor_count>0,
                       "Stack contact lost the frame-86 friction anchors");
                if (stacked.frame_index()==86) {
                    expect(!it->broken,
                           "Position-only clamp incorrectly persisted as final friction break");
                    saved_anchor_count=it->anchor_count;
                    saved_support_anchors=it->anchor_positions;
                } else {
                    expect(it->anchor_count==saved_anchor_count,
                           "Unbroken friction anchors were discarded next frame");
                    for (std::size_t k=0;k<saved_anchor_count;++k) {
                        const auto& a=it->anchor_positions[k];
                        const auto& b=saved_support_anchors[k];
                        expect(std::abs(a.x-b.x)<0.001f &&
                               std::abs(a.y-b.y)<0.001f &&
                               std::abs(a.z-b.z)<0.001f,
                               "Recovered patch did not reuse its cached local anchors");
                    }
                }
            }
        }
        const auto lo = stacked.body_state(lower);
        const auto hi = stacked.body_state(upper);
        if (!(std::isfinite(lo.position.y) &&
               std::isfinite(hi.position.y) &&
               lo.position.y > 0.37f && lo.position.y < 0.65f &&
               hi.position.y > 1.25f && hi.position.y < 1.70f &&
               std::abs(lo.velocity.y) < 0.8f &&
               std::abs(hi.velocity.y) < 0.8f))
            std::fprintf(stderr,
                "stack lo(y=%g vy=%g spin=%g,%g,%g) hi(y=%g vy=%g spin=%g,%g,%g) contacts=%zu\n",
                lo.position.y,lo.velocity.y,lo.angular_velocity.x,
                lo.angular_velocity.y,lo.angular_velocity.z,
                hi.position.y,hi.velocity.y,hi.angular_velocity.x,
                hi.angular_velocity.y,hi.angular_velocity.z,
                stacked.contacts().size());
        expect(std::isfinite(lo.position.y) &&
               std::isfinite(hi.position.y) &&
               lo.position.y > 0.37f && lo.position.y < 0.65f &&
               hi.position.y > 1.25f && hi.position.y < 1.70f &&
               std::abs(lo.velocity.y) < 0.8f &&
               std::abs(hi.velocity.y) < 0.8f,
               "Native two-body contact chain failed");
    }

    {
        // Tiny mass can still have a representable inverse mass while
        // its source angular tensor is not representable. Contact PGS
        // must reject this geometry instead of using a zero tensor.
        Scene very_light{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto left = very_light.add_dynamic_box(
            {1,1,1},{0,5,0},5.0e-40f);
        const auto right = very_light.add_dynamic_box(
            {1,1,1},{1.5f,5,0},5.0e-40f);
        bool rejected_angular = false;
        try { very_light.apply_torque_impulse(left, {1,0,0}); }
        catch (const std::logic_error&) { rejected_angular = true; }
        expect(rejected_angular,
               "Unrepresentable angular inertia silently used a substitute");
        rejected_angular = false;
        try { very_light.set_angular_velocity(left, {1,0,0}); }
        catch (const std::logic_error&) { rejected_angular = true; }
        expect(rejected_angular,
               "Tiny-mass body admitted an unrepresentable angular state");
        bool rejected_contact=false;
        try { very_light.step(1.0f/120.0f); }
        catch(const std::logic_error&) { rejected_contact=true; }
        expect(rejected_contact &&
               very_light.frame_index()==0 &&
               very_light.body_state(left).position.x==0.0f &&
               very_light.body_state(right).position.x==1.5f,
               "Tiny-mass contact substituted a zero angular tensor");
        // Zero maxContactImpulse is extracted before Dy constraint
        // construction. Even an unrepresentable angular tensor must
        // remain translation-capable when the existing Gu touch has
        // no response rows. This also checks a retry after rejection.
        fyzz::RigidBodyDynamics no_response{};
        no_response.max_contact_impulse=0.0f;
        very_light.set_dynamics(left,no_response);
        very_light.step(1.0f/120.0f);
        const auto left_free=very_light.body_state(left);
        const auto right_free=very_light.body_state(right);
        constexpr float dt=1.0f/120.0f;
        const float free_vy=-9.81f*dt;
        const float free_y=5.0f+free_vy*dt;
        expect(very_light.frame_index()==1 &&
               very_light.contacts().size()==1 &&
               very_light.contact_edge_connected({left,right}) &&
               very_light.solved_normal_impulses().size()==
                   very_light.contacts()[0].count &&
               std::all_of(very_light.solved_normal_impulses().begin(),
                           very_light.solved_normal_impulses().end(),
                           [](double value){return value==0.0;}) &&
               std::abs(left_free.velocity.y-free_vy)<1.0e-5f &&
               std::abs(right_free.velocity.y-free_vy)<1.0e-5f &&
               std::abs(left_free.position.y-free_y)<1.0e-5f &&
               std::abs(right_free.position.y-free_y)<1.0e-5f &&
               left_free.position.x==0.0f &&
               right_free.position.x==1.5f,
               "Zero-capped Gu touch altered or rejected free translation");
        very_light.set_position(right,{5,5,0});
        very_light.set_velocity(left,{1,0,0});
        very_light.step(1.0f/120.0f);
        expect(std::isfinite(very_light.body_state(left).velocity.x) &&
               very_light.body_state(left).position.x>0.0f,
               "Rejected tiny-mass contact poisoned later free translation");
    }

    {
        // Inverse mass can remain representable when a source PxReal box
        // inertia component underflows, overflows or has an
        // unrepresentable inverse. Those are translation-only bodies,
        // not grounds for rejecting the existing mass admission.
        const std::array<std::pair<fyzz::Vec3, float>, 4> cases{{
            {{1.0e-13f,1.0e-13f,1.0e-13f},1.0f},
            {{1.0e-13f,1.0e-13f,1.0f},1.0f},
            {{1.0f,1.0f,1.0f},5.0e-40f},
            {{1.0e13f,1.0e13f,1.0e13f},1.0e-39f}
        }};
        for (const auto& [half_extents,density] : cases) {
            Scene body{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto id = body.add_dynamic_box(
                half_extents,{0,5,0},density);
            bool rejected = false;
            try { body.set_angular_velocity(id,{1,0,0}); }
            catch(const std::logic_error&) { rejected = true; }
            expect(rejected,
                   "Unrepresentable box inertia incorrectly admitted spin");
            rejected = false;
            try { body.apply_torque_impulse(id,{1,0,0}); }
            catch(const std::logic_error&) { rejected = true; }
            expect(rejected,
                   "Unrepresentable box inertia incorrectly admitted torque");
            body.set_velocity(id,{1,0,0});
            body.step(1.0f/120.0f);
            const auto state=body.body_state(id);
            expect(std::isfinite(state.position.x) &&
                   std::isfinite(state.position.y) &&
                   std::isfinite(state.velocity.x) &&
                   std::isfinite(state.velocity.y) &&
                   state.position.x>0.0f && state.velocity.x>0.0f &&
                   state.orientation.w==1.0f,
                   "Valid translational body rejected or corrupted by inertia");
        }
    }

    {
        // Two individually finite extreme inputs must pass through the
        // exact PhysX unconstrained-velocity clamp before contact prep.
        // Preserve finite impulse state without a custom wide dot product.
        Scene extreme{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a = extreme.add_dynamic_box({50,50,50}, {0,60,0}, 1);
        const auto b = extreme.add_dynamic_box({50,50,50}, {80,60,0}, 1);
        extreme.set_velocity(a, {2.0e38f,0,0});
        extreme.set_velocity(b, {-2.0e38f,0,0});
        // Smallest normal float: portable even with flush-to-zero.
        extreme.step(std::numeric_limits<float>::min());
        const auto va = extreme.body_state(a).velocity;
        const auto vb = extreme.body_state(b).velocity;
        expect(std::isfinite(va.x) && std::isfinite(va.y) &&
               std::isfinite(va.z) && std::isfinite(vb.x) &&
               std::isfinite(vb.y) && std::isfinite(vb.z) &&
               va.x < 0.0f && vb.x > 0.0f &&
               std::abs(va.x) < 1.0e38f && std::abs(vb.x) < 1.0e38f,
               "Finite high-speed contact overflowed or corrupted velocities");
        expect(extreme.contacts().size() == 1,
               "High-speed dynamic pair lost its contact");
    }
    {
        // Regression for a second float overflow: relative velocity and
        // impulse remain finite in double, but unequal inverse masses
        // previously overflowed one body's float velocity on writeback.
        // Mirror the PhysX dynamic-body maxLinearVelocitySq=1e32 default.
        // The unconstrained speed clamp precedes the contact solver.
        // Use touching faces (zero penetration bias) so this tests that
        // clamp rather than incorrectly capping post-solver correction.
        Scene unequal_mass{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto light = unequal_mass.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1.0e-6f);
        const auto heavy = unequal_mass.add_dynamic_box(
            {1,1,1}, {2.0f,5,0}, 1.0e6f);
        const float max_v = std::numeric_limits<float>::max();
        unequal_mass.set_velocity(light, {max_v,0,0});
        unequal_mass.set_velocity(heavy, {-max_v,0,0});
        unequal_mass.step(std::numeric_limits<float>::min());
        const auto vlight = unequal_mass.body_state(light).velocity;
        const auto vheavy = unequal_mass.body_state(heavy).velocity;
        expect(std::isfinite(vlight.x) && std::isfinite(vlight.y) &&
               std::isfinite(vlight.z) && std::isfinite(vheavy.x) &&
               std::isfinite(vheavy.y) && std::isfinite(vheavy.z),
               "Unequal-mass contact published infinite or NaN velocity");
        expect(std::abs(vlight.x) < 2.0e16f &&
               std::abs(vheavy.x) < 2.0e16f &&
               unequal_mass.contacts().size() == 1,
               "Source unconstrained linear-speed clamp failed zero-bias contact");
    }
    {
        // A free body uses the same source default velocity clamp, so
        // safety is independent of whether contacts are generated.
        Scene unopposed{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto b = unopposed.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1);
        unopposed.set_velocity(b, {2.0e38f,2.0e38f,2.0e38f});
        unopposed.step(std::numeric_limits<float>::min());
        const auto vel = unopposed.body_state(b).velocity;
        // PhysX's PxReal squared speed overflows to infinity here.
        // Consequently sqrt(1e32f / inf) is zero, not 1e16 normalization.
        expect(vel.x == 0.0f && vel.y == 0.0f && vel.z == 0.0f,
               "Native high-speed clamp diverged from PhysX float arithmetic");
        Scene finite_limit{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto limited = finite_limit.add_dynamic_box(
            {1,1,1}, {0,5,0}, 1);
        finite_limit.set_velocity(limited, {2.0e16f,0,0});
        finite_limit.step(1.0f / 120.0f);
        const auto moderate = finite_limit.body_state(limited).velocity;
        expect(std::isfinite(moderate.x) &&
               std::abs(moderate.x - 1.0e16f) < 1.0e10f,
               "Finite squared velocity failed PhysX source clamp");
    }
    {
        // Velocity authoring has no full-manifold invalidation scan:
        // an isolated edit changes only the selected actor's velocity.
        Scene authoring{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto static_id = authoring.add_static_floor(0);
        const auto dynamic_id = authoring.add_dynamic_box(
            {0.25f,0.25f,0.25f}, {0,1,0}, 1);
        authoring.set_velocity(dynamic_id, {2,0,3});
        expect(authoring.body_state(dynamic_id).velocity.x == 2,
               "Native dynamic-body velocity setter failed");
        bool rejected = false;
        try { authoring.set_velocity(static_id, {1,0,0}); }
        catch (const std::logic_error&) { rejected = true; }
        expect(rejected, "Native static body accepted linear velocity");
        bool invalid = false;
        try { authoring.set_velocity(dynamic_id, {INFINITY,0,0}); }
        catch (const std::invalid_argument&) { invalid = true; }
        expect(invalid, "Native scene accepted nonfinite velocity");
    }

    {
        // Source inertia is directional and density-scaled. The same
        // world torque impulse must produce different angular velocity
        // for a rectangular body on its distinct principal axes.
        Scene torque{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto id = torque.add_dynamic_box(
            {0.5f,1.0f,2.0f}, {0,10,0}, 2.0f);
        torque.apply_torque_impulse(id, {1,1,1});
        torque.step(1.0f/120.0f);
        const auto w = torque.body_state(id).angular_velocity;
        // 8*0.5*1*2*2 = 16 kg; inertia = mass/3*(hy^2+hz^2...).
        const float angular_damp = 1.0f-0.05f/120.0f;
        const float ix=16.0f*(1.0f+4.0f)/3.0f;
        const float iy=16.0f*(0.25f+4.0f)/3.0f;
        const float iz=16.0f*(0.25f+1.0f)/3.0f;
        expect(std::abs(w.x-angular_damp/ix)<0.0001f &&
               std::abs(w.y-angular_damp/iy)<0.0001f &&
               std::abs(w.z-angular_damp/iz)<0.0001f &&
               w.x < w.y && w.y < w.z,
               "Native box impulse failed source mass/inertia axes");
        torque.step(1.0f/120.0f);
        const auto twice=torque.body_state(id).angular_velocity;
        expect(twice.z < w.z,
               "Per-step angular impulse was replayed on the next frame");
        bool invalid=false;
        try {torque.apply_torque_impulse(id,{INFINITY,0,0});}
        catch(const std::invalid_argument&) {invalid=true;}
        expect(invalid,"Native torque accepted nonfinite input");
        const auto floor_id=torque.add_static_floor(0);
        bool rejected=false;
        try {torque.apply_torque_impulse(floor_id,{1,0,0});}
        catch(const std::logic_error&) {rejected=true;}
        expect(rejected,"Static floor accepted angular impulse");
    }
    {
        // Queued angular impulses are consumed at the next scene step
        // even when static contacts are present. Normal contact rows
        // must now accept the resulting rotated body without replaying
        // the pending torque on later frames.
        Scene rotating_contact{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto id = rotating_contact.add_dynamic_box(
            {0.5f,1.0f,2.0f},{0,2,0},1);
        rotating_contact.add_static_floor(0);
        rotating_contact.apply_torque_impulse(id,{1,0,0});
        const auto before = rotating_contact.body_state(id);
        rotating_contact.step(1.0f/120.0f);
        const auto after = rotating_contact.body_state(id);
        expect(after.position.y < before.position.y &&
               after.angular_velocity.x > before.angular_velocity.x &&
               std::isfinite(after.angular_velocity.x),
               "Queued angular impulse was not consumed in active scene");
        rotating_contact.step(1.0f/120.0f);
        expect(std::isfinite(rotating_contact.body_state(id).angular_velocity.x),
               "Angular motion failed to continue without replaying torque");
    }
    {
        // Source ScBodyCore creation counter and Dy::sleepCheck: motion
        // readiness precedes whole-island deactivation. Once a rigid
        // body rests on its static floor it must stop integrating gravity
        // and remain stationary, but an explicit velocity edit wakes it.
        Scene sleeping{Scene::ContactGenerator::DiagnosticLegacyGu};
        sleeping.add_static_floor(0.0f);
        const auto id=sleeping.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        expect(std::abs(sleeping.body_wake_counter(id)-0.4f)<1e-6f &&
               !sleeping.body_ready_for_sleep(id) &&
               !sleeping.body_is_sleeping(id),
               "Source rigid-body creation did not initialize wake counter");
        for(int frame=0;frame<360;++frame)
            sleeping.step(1.0f/120.0f);
        const auto still=sleeping.body_state(id);
        expect(sleeping.body_is_sleeping(id) &&
               sleeping.body_ready_for_sleep(id) &&
               still.sleeping &&
               sleeping.body_wake_counter(id)==0.0f &&
               still.velocity.y==0.0f &&
               sleeping.solver_partition_count()==0,
               "PhysX island did not deactivate a settled floor body");
        for(double impulse:sleeping.solved_normal_impulses())
            expect(impulse==0.0,
                   "Sleeping contact manager emitted an active solver impulse");
        // A failed later step must restore all existing sleep state,
        // not just actor motion and topology. An independently admitted
        // awake body forces a finite-input overflow before publication.
        const auto awake=sleeping.add_dynamic_box(
            {0.25f,0.25f,0.25f},{20,10,0},1.0f);
        const auto before_failure=sleeping.body_state(awake);
        bool failed=false;
        try {sleeping.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        const auto after_failure=sleeping.body_state(awake);
        expect(failed && sleeping.body_is_sleeping(id) &&
               sleeping.body_wake_counter(id)==0.0f &&
               sleeping.body_state(id).sleeping &&
               after_failure.position.y==before_failure.position.y &&
               sleeping.body_wake_counter(awake)==0.4f,
               "Failed source step corrupted dormant island or wake counter");
        sleeping.set_velocity(id,{1,0,0});
        expect(!sleeping.body_is_sleeping(id) &&
               !sleeping.body_ready_for_sleep(id) &&
               sleeping.body_wake_counter(id)==0.4f,
               "Source velocity authoring did not wake the rigid body");
        sleeping.step(1.0f/120.0f);
        const auto moved=sleeping.body_state(id);
        expect(!moved.sleeping &&
               moved.position.x>still.position.x &&
               std::isfinite(moved.velocity.x),
               "Woken source island failed to reenter native solver");
    }

    {
        // PxsIslandSim::processLostEdges(third pass) consumes the *prior*
        // frame's Dy::sleepCheck readiness. The frame in which the body
        // first becomes ready is not allowed to deactivate it immediately.
        // The next third pass then deactivates before gravity and PGS:
        // that frame must not integrate one extra position increment.
        Scene phase_boundary{Scene::ContactGenerator::DiagnosticLegacyGu};
        phase_boundary.add_static_floor(0.0f);
        const auto sleeper=phase_boundary.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool saw_ready_before_sleep=false;
        for(int frame=0;frame<600;++frame) {
            phase_boundary.step(1.0f/120.0f);
            if(phase_boundary.body_ready_for_sleep(sleeper)) {
                expect(!phase_boundary.body_is_sleeping(sleeper),
                       "New Dy sleep readiness deactivated during its own solver frame");
                saw_ready_before_sleep=true;
                const auto ready_pose=phase_boundary.body_state(sleeper);
                phase_boundary.step(1.0f/120.0f);
                const auto inactive=phase_boundary.body_state(sleeper);
                expect(inactive.sleeping &&
                       inactive.position.x==ready_pose.position.x &&
                       inactive.position.y==ready_pose.position.y &&
                       inactive.position.z==ready_pose.position.z &&
                       inactive.velocity.x==0.0f &&
                       inactive.velocity.y==0.0f &&
                       inactive.velocity.z==0.0f,
                       "Third-pass deactivation integrated an already-ready island");
                break;
            }
        }
        expect(saw_ready_before_sleep,
               "Source third-pass fixture never produced Dy sleep readiness");
    }

    // A user-authored dynamic teleport can remove a contact edge and
    // split the original island before its waking node is propagated.
    // Its dormant former contact partner must be reactivated directly
    // from the previously published Gu manager, just as for a static
    // support edit, even when the new topology no longer connects them.
    {
        Scene moved_peer{Scene::ContactGenerator::DiagnosticLegacyGu};
        moved_peer.add_static_floor(0.0f);
        const auto left=moved_peer.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=moved_peer.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            moved_peer.step(1.0f/120.0f);
        expect(moved_peer.body_is_sleeping(left) &&
               moved_peer.body_is_sleeping(right) &&
               moved_peer.accurate_island_id(left)==
                   moved_peer.accurate_island_id(right),
               "Dynamic support edit fixture lacked dormant contact island");
        moved_peer.set_position(left,{-10,0.5f,0});
        expect(!moved_peer.body_is_sleeping(right) &&
               !moved_peer.body_ready_for_sleep(right) &&
               moved_peer.body_wake_counter(right)>0.0f,
               "Dynamic pose edit left old contact peer dormant");
        moved_peer.step(1.0f/120.0f);
        expect(moved_peer.accurate_island_id(left)!=
                   moved_peer.accurate_island_id(right) &&
               !moved_peer.body_is_sleeping(right),
               "Contact island split discarded the old peer wake");
    }

    // Third-pass sleep readiness belongs to the previous solver frame.
    // A support edit between the ready and inactive passes must revoke
    // the ready flag while the body is still awake. Verify both static
    // movement and retirement, with and without an intermediate
    // collision-only publication. These cases must integrate gravity on
    // their next actual step, not suspend the body in midair.
    for(bool retire_support : {false,true}) {
        for(bool query_first : {false,true}) {
            Scene unsupported{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto floor=unsupported.add_static_floor(0.0f);
            const auto body=unsupported.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            bool became_ready=false;
            for(int frame=0;frame<600;++frame) {
                unsupported.step(1.0f/120.0f);
                if(unsupported.body_ready_for_sleep(body)) {
                    became_ready=true;
                    break;
                }
            }
            expect(became_ready && !unsupported.body_is_sleeping(body),
                   "Support-edit fixture missed ready-but-awake third-pass boundary");
            const auto settled=unsupported.body_state(body);
            if(retire_support)
                unsupported.remove(floor);
            else
                unsupported.set_position(floor,{0,-10,0});
            expect(!unsupported.body_ready_for_sleep(body) &&
                   !unsupported.body_is_sleeping(body) &&
                   unsupported.body_wake_counter(body)>0.0f,
                   "Support edit retained stale ready-but-awake sleep state");
            if(query_first) {
                unsupported.advance_collision();
                expect(!unsupported.body_ready_for_sleep(body) &&
                       !unsupported.body_is_sleeping(body),
                       "Collision query restored obsolete third-pass readiness");
            }
            unsupported.step(1.0f/120.0f);
            const auto falling=unsupported.body_state(body);
            expect(!falling.sleeping &&
                   !unsupported.body_ready_for_sleep(body) &&
                   falling.position.y<settled.position.y-1.0e-5f &&
                   falling.velocity.y<0.0f,
                   "Support edit deactivated ready actor before gravity");
        }
    }

    {
        // A geometry-only query also publishes changed island membership
        // and may consume an authored node wake, without a numerical
        // solver step. Its accurate active-node index must not vanish
        // after a graph rebuild or remain stale after an island split.
        Scene queried{Scene::ContactGenerator::DiagnosticLegacyGu};
        queried.add_static_floor(0.0f);
        const auto sleeper=queried.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.accurate_active_node_index(sleeper)==0,
               "Geometry-only island admission did not index its active rigid");
        for(int frame=0;frame<360;++frame)
            queried.step(1.0f/120.0f);
        expect(queried.body_is_sleeping(sleeper) &&
               queried.active_accurate_nodes().empty(),
               "Geometry query fixture never deactivated its first node");
        queried.set_velocity(sleeper,{0.25f,0,0});
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.active_accurate_nodes()[0]==sleeper &&
               queried.accurate_active_node_index(sleeper)==0,
               "Quiet authored wake failed to update the accurate active map");
        const auto added=queried.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==2 &&
               queried.accurate_active_node_index(added)<
                   queried.active_accurate_nodes().size(),
               "Geometry-only rebuild lost an active rigid node");
        queried.remove(sleeper);
        queried.advance_collision();
        expect(queried.active_accurate_nodes().size()==1 &&
               queried.active_accurate_nodes()[0]==added &&
               queried.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               queried.accurate_active_node_index(added)==0,
               "Actor retirement left a stale active-node reverse index");
    }

    {
        // Rejected steps must preserve retained source active-node
        // spans, their actual storage, and their published elements.
        // A failure after a newly ready body deactivates exercises
        // swap-last active index retirement before rollback.
        Scene retained_active{Scene::ContactGenerator::DiagnosticLegacyGu};
        retained_active.add_static_floor(0.0f);
        const auto sleeper=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            retained_active.step(1.0f/120.0f);
            if(retained_active.body_ready_for_sleep(sleeper)) {
                ready=true;
                break;
            }
        }
        expect(ready && !retained_active.body_is_sleeping(sleeper),
               "Retained active span fixture did not become ready");
        const auto published=retained_active.active_accurate_nodes();
        const auto published_deactivations=
            retained_active.deactivating_accurate_nodes();
        expect(published.size()==1 && published[0]==sleeper &&
               published_deactivations.empty(),
               "Retained active span fixture was not an active rigid");
        const auto* const published_ptr=published.data();
        const auto* const deact_ptr=published_deactivations.data();
        // The ready body will deactivate at the next third pass, so a
        // second, unconnected active body is required to force the
        // subsequent integration to reject an enormous timestep.
        const auto overflow_body=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10,5,0},1.0f);
        const auto frame=retained_active.frame_index();
        for(int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {retained_active.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected && retained_active.frame_index()==frame &&
                   !retained_active.body_is_sleeping(sleeper) &&
                   retained_active.active_accurate_nodes().data()==
                       published_ptr &&
                   retained_active.deactivating_accurate_nodes().data()==
                       deact_ptr &&
                   published.size()==1 && published[0]==sleeper &&
                   retained_active.active_accurate_nodes().size()==1 &&
                   retained_active.accurate_active_node_index(sleeper)==0 &&
                   retained_active.deactivating_accurate_nodes().empty(),
                   "Failed third-pass deactivation invalidated active-node span");
        }
        retained_active.step(1.0f/120.0f);
        expect(retained_active.body_is_sleeping(sleeper) &&
               retained_active.deactivating_accurate_nodes().size()==1 &&
               retained_active.deactivating_accurate_nodes()[0]==sleeper,
               "Valid retry failed to emit accurate third-pass deactivation");
        // The next rejected frame can rebuild topology after a newly
        // admitted awake body; retained deactivation storage remains
        // readable even if the working graph was replaced.
        const auto old_output=retained_active.deactivating_accurate_nodes();
        const auto* const old_output_ptr=old_output.data();
        const auto newcomer=retained_active.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        const auto next_frame=retained_active.frame_index();
        bool rejected=false;
        try {retained_active.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected && retained_active.frame_index()==next_frame &&
               old_output.data()==old_output_ptr && old_output.size()==1 &&
               old_output[0]==sleeper &&
               retained_active.deactivating_accurate_nodes().data()==
                   old_output_ptr &&
               retained_active.deactivating_accurate_nodes().size()==1 &&
               retained_active.deactivating_accurate_nodes()[0]==sleeper,
               "Rejected island rebuild invalidated deactivation report span");
        retained_active.step(1.0f/120.0f);
        const auto active_after_retry=retained_active.active_accurate_nodes();
        expect(retained_active.deactivating_accurate_nodes().empty() &&
               active_after_retry.size()==2 &&
               std::find(active_after_retry.begin(),active_after_retry.end(),
                         overflow_body)!=active_after_retry.end() &&
               std::find(active_after_retry.begin(),active_after_retry.end(),
                         newcomer)!=active_after_retry.end(),
               "Post-rollback deactivation output was replayed on retry");
    }

    {
        // PxsIslandSim::wakeIslands captures its initially active rigid
        // prefix before admitting newly activated nodes. Deactivating
        // an initial member must use source markInactive's prefix-tail
        // relocation so only genuinely new rigid nodes remain in
        // getNodesToActivate's active suffix. Rebuilding the graph for
        // actor admission must preserve the existing active permutation.
        Scene partitions{Scene::ContactGenerator::DiagnosticLegacyGu};
        partitions.add_static_floor(0.0f);
        const auto sleeper=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto flyer=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10,10,0},1.0f);
        partitions.advance_collision();
        expect(partitions.newly_activated_accurate_nodes().size()==2,
               "First source wake did not admit both new rigid bodies");
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            // Keep the independent free-flight body awake while the
            // resting island reaches its prior-frame sleep boundary.
            partitions.set_velocity(flyer,{0,9.81f,0});
            partitions.step(1.0f/120.0f);
            if(partitions.body_ready_for_sleep(sleeper)) {
                ready=true;
                break;
            }
        }
        expect(ready && !partitions.body_is_sleeping(sleeper),
               "Initial-active partition fixture did not reach readiness");
        // Admit a new island in the very pass where the old floor
        // island deactivates. The original two-node initial prefix must
        // shrink to the still-active flyer, while the newcomer belongs
        // strictly to the newly activated suffix. This fails if
        // markInactive uses unconditional swap-last without the
        // source initial-partition tail permutation.
        const auto newcomer=partitions.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,10,0},1.0f);
        partitions.set_velocity(flyer,{0,9.81f,0});
        partitions.step(1.0f/120.0f);
        expect(partitions.body_is_sleeping(sleeper) &&
               partitions.deactivating_accurate_nodes().size()==1 &&
               partitions.deactivating_accurate_nodes()[0]==sleeper &&
               partitions.active_accurate_nodes().size()==2 &&
               partitions.active_accurate_nodes()[0]==flyer &&
               partitions.active_accurate_nodes()[1]==newcomer &&
               partitions.newly_activated_accurate_nodes().size()==1 &&
               partitions.newly_activated_accurate_nodes()[0]==newcomer &&
               partitions.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               partitions.accurate_active_node_index(flyer)==0 &&
               partitions.accurate_active_node_index(newcomer)==1,
               "Simultaneous source markInactive/admission broke initial-active partition");
        const auto published=partitions.active_accurate_nodes();
        const auto prior_new=partitions.newly_activated_accurate_nodes();
        const auto* const original=published.data();
        bool failed=false;
        try {partitions.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed &&
               partitions.active_accurate_nodes().data()==original &&
               published[0]==flyer && published[1]==newcomer &&
               prior_new.size()==1 && prior_new[0]==newcomer,
               "Rejected frame damaged published active-node partitions");
        partitions.set_velocity(flyer,{0,9.81f,0});
        partitions.step(1.0f/120.0f);
        expect(partitions.newly_activated_accurate_nodes().empty() &&
               partitions.active_accurate_nodes().size()==2,
               "Previous initial-active suffix replayed on next frame");
    }

    verify_idempotent_pose_autowake();
    verify_active_pose_preserves_sleep_filter();
    verify_simultaneous_initial_sleep_compaction();
    verify_speculative_active_node_lifecycle();

    {
        // Source accurate active-node indexing must stay dense and
        // bidirectional through third-pass sleep, user wake, and
        // reactivation. The separate deactivation output is published
        // for exactly the boundary in which ready nodes go inactive.
        Scene indexed{Scene::ContactGenerator::DiagnosticLegacyGu};
        indexed.add_static_floor(0.0f);
        const auto body=indexed.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        indexed.step(1.0f/120.0f);
        expect(indexed.active_accurate_nodes().size()==1 &&
               indexed.active_accurate_nodes()[0]==body &&
               indexed.accurate_active_node_index(body)==0,
               "Source active-node index missed the initially awake body");
        bool deactivated=false;
        for(int frame=0;frame<600;++frame) {
            indexed.step(1.0f/120.0f);
            if(indexed.body_is_sleeping(body)) {
                const auto retired=indexed.deactivating_accurate_nodes();
                expect(retired.size()==1 && retired[0]==body &&
                       indexed.active_accurate_nodes().empty() &&
                       indexed.accurate_active_node_index(body)==
                           std::numeric_limits<std::size_t>::max(),
                       "Third pass did not publish indexed rigid deactivation");
                deactivated=true;
                break;
            }
        }
        expect(deactivated,"Source indexed active node never slept");
        // A public collision-only query consumes a new publication clock
        // without invoking another ready-island transition. Source
        // mNodesToPutToSleep is per-pass output, not sticky sleep state.
        indexed.advance_collision();
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().empty() &&
               indexed.accurate_active_node_index(body)==
                   std::numeric_limits<std::size_t>::max(),
               "Quiet collision publication replayed a deactivation event");
        indexed.advance_collision();
        expect(indexed.deactivating_accurate_nodes().empty(),
               "Repeated quiet publication resurrected an old deactivation");
        indexed.step(1.0f/120.0f);
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().empty(),
               "Dormant third pass repeated an already published deactivation");
        indexed.set_velocity(body,{1,0,0});
        indexed.step(1.0f/120.0f);
        expect(indexed.deactivating_accurate_nodes().empty() &&
               indexed.active_accurate_nodes().size()==1 &&
               indexed.active_accurate_nodes()[0]==body &&
               indexed.accurate_active_node_index(body)==0,
               "Source node activation did not rebuild the active index");
    }

    {
        // The source active-island list is a strict subset of all accurate
        // islands. Dormant bodies remain in the persistent source graph
        // but are excluded from unconstrained integration and PGS. An
        // isolated awake node must retain its own active island batch.
        Scene active_handoff{Scene::ContactGenerator::DiagnosticLegacyGu};
        active_handoff.add_static_floor(0.0f);
        const auto resting=active_handoff.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto flyer=active_handoff.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        active_handoff.advance_collision();
        expect(active_handoff.accurate_island_count()==2,
               "Active-island fixture did not admit both source islands");
        // Isolate the dormant transition from the free-flight wake state:
        // first sleep the floor body, then introduce the distant flyer.
        Scene staged{Scene::ContactGenerator::DiagnosticLegacyGu};
        staged.add_static_floor(0.0f);
        const auto dormant=staged.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            staged.step(1.0f/120.0f);
        expect(staged.body_is_sleeping(dormant) &&
               staged.active_accurate_island_count()==0,
               "Dormant accurate island remained on Dy active list");
        const auto dynamic=staged.add_dynamic_box(
            {0.5f,0.5f,0.5f},{20,5,0},1.0f);
        staged.step(1.0f/120.0f);
        expect(staged.accurate_island_count()==2 &&
               staged.active_accurate_island_count()==1 &&
               staged.body_is_sleeping(dormant) &&
               staged.body_state(dynamic).position.y<5.0f,
               "Dy active-island handoff admitted sleeper or skipped free flight");
        staged.set_velocity(dormant,{1,0,0});
        staged.step(1.0f/120.0f);
        expect(staged.active_accurate_island_count()==2 &&
               !staged.body_is_sleeping(dormant),
               "Wake did not restore accurate active-island admission");
        (void)resting;
        (void)flyer;
    }

    // A quiet collision query must consume genuinely authored source
    // wake requests without scanning the island graph for every query.
    // Ensure a rejected step does not clear that pending wake indicator:
    // the next quiet query still propagates to the sleeping island peer.
    {
        Scene pending_query{Scene::ContactGenerator::DiagnosticLegacyGu};
        pending_query.add_static_floor(0.0f);
        const auto left=pending_query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=pending_query.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int i=0;i<360;++i)pending_query.step(1.0f/120.0f);
        pending_query.advance_collision();
        expect(pending_query.body_is_sleeping(left) &&
               pending_query.body_is_sleeping(right),
               "Quiet wake indicator fixture failed to settle");
        const auto revision=pending_query.island_topology_revision();
        for(int i=0;i<4;++i)pending_query.advance_collision();
        expect(pending_query.island_topology_revision()==revision,
               "Quiet collision queries modified source island topology");
        pending_query.set_velocity(left,{0.3f,0,0});
        bool rejected=false;
        try {pending_query.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected && pending_query.body_is_sleeping(right),
               "Rejected frame corrupted queued wake");
        pending_query.advance_collision();
        expect(!pending_query.body_is_sleeping(right) &&
               pending_query.island_topology_revision()==revision,
               "Queued authored wake was lost after quiet rollback");
    }

    {
        // A rejected Dy step rebuilds the active accurate island view
        // after source wake/deactivation. The previous published active
        // batch count must survive rollback without copying the list.
        Scene active_rollback{Scene::ContactGenerator::DiagnosticLegacyGu};
        active_rollback.add_static_floor(0.0f);
        const auto sleeper=active_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int i=0;i<360;++i)active_rollback.step(1.0f/120.0f);
        expect(active_rollback.active_accurate_island_count()==0 &&
               active_rollback.body_is_sleeping(sleeper),
               "Active-batch rollback fixture did not become dormant");
        active_rollback.set_velocity(sleeper,{1,0,0});
        // Prior published active view remains empty until the next
        // simulation publication, despite the authored actor wake.
        bool rejected=false;
        try {active_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               active_rollback.active_accurate_island_count()==0 &&
               active_rollback.active_accurate_nodes().empty() &&
               active_rollback.accurate_active_node_index(sleeper)==
                   std::numeric_limits<std::size_t>::max() &&
               active_rollback.deactivating_accurate_nodes().empty() &&
               !active_rollback.body_is_sleeping(sleeper),
               "Rejected Dy frame replaced the published active-node/index state");
        active_rollback.step(1.0f/120.0f);
        expect(active_rollback.active_accurate_island_count()==1 &&
               active_rollback.active_accurate_nodes().size()==1 &&
               active_rollback.accurate_active_node_index(sleeper)==0,
               "Active-island retry did not republish the indexed source node");
    }

    {
        // IslandSim::activateNode clears an already-awake node's ready
        // flag when a new contact wakes its island. A body that just
        // reached readiness must still resolve the newly added static
        // contact, not sleep before submitting the new Gu rows.
        Scene new_touch{Scene::ContactGenerator::DiagnosticLegacyGu};
        new_touch.add_static_floor(0.0f);
        const auto candidate=new_touch.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        bool ready=false;
        for(int frame=0;frame<600;++frame) {
            new_touch.step(1.0f/120.0f);
            if(new_touch.body_ready_for_sleep(candidate)) {
                ready=true;
                break;
            }
        }
        expect(ready && !new_touch.body_is_sleeping(candidate),
               "New-touch activation fixture missed the ready-before-sleep frame");
        const float previous_x=new_touch.body_state(candidate).position.x;
        new_touch.add_static_box(
            {0.5f,0.5f,0.5f},{previous_x+0.9f,0.5f,0});
        new_touch.step(1.0f/120.0f);
        const auto moved=new_touch.body_state(candidate);
        expect(!moved.sleeping &&
               moved.position.x<previous_x-0.001f,
               "New Gu touch failed to revoke an awake node's sleep readiness");
    }

    {
        // IslandSim::activateIslandInternal wakes connected rigid nodes,
        // not only the node explicitly edited by a user. A persistent
        // dynamic contact must admit both nodes together on reactivation.
        Scene coupled_sleep{Scene::ContactGenerator::DiagnosticLegacyGu};
        coupled_sleep.add_static_floor(0.0f);
        const auto left=coupled_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=coupled_sleep.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            coupled_sleep.step(1.0f/120.0f);
        expect(coupled_sleep.accurate_island_count()==1 &&
               coupled_sleep.body_is_sleeping(left) &&
               coupled_sleep.body_is_sleeping(right),
               "Source dynamic contact island did not deactivate together");
        coupled_sleep.set_velocity(left,{0.5f,0,0});
        expect(!coupled_sleep.body_is_sleeping(left) &&
               coupled_sleep.body_is_sleeping(right),
               "Direct user wake bypassed source island activation stage");
        coupled_sleep.step(1.0f/120.0f);
        expect(!coupled_sleep.body_is_sleeping(left) &&
               !coupled_sleep.body_is_sleeping(right),
               "Source island wake did not reactivate contact-connected peer");
    }

    {
        // IslandSim::wakeIslands consumes an actual authored activation,
        // including when a collision-only query is otherwise quiet.
        // An already-awake neighbor is not itself a perpetual wake event.
        Scene queued_wake{Scene::ContactGenerator::DiagnosticLegacyGu};
        queued_wake.add_static_floor(0.0f);
        const auto left=queued_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-0.5f,0.5f,0},1.0f);
        const auto right=queued_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.5f,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            queued_wake.step(1.0f/120.0f);
        expect(queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Activation queue fixture did not reach dormant island");
        queued_wake.advance_collision();
        expect(queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Quiet query spuriously activated a dormant island");
        queued_wake.set_velocity(left,{0.3f,0,0});
        expect(!queued_wake.body_is_sleeping(left) &&
               queued_wake.body_is_sleeping(right),
               "Authored activation did not retain per-node handoff");
        // A failed Dy frame may consume an activating node while
        // traversing the speculative island. The checkpoint must restore
        // both the dormant peer and the queued activating flag so the
        // subsequent collision-only query still wakes the peer.
        const auto queued_frame=queued_wake.frame_index();
        for(int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {queued_wake.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected &&
                   queued_wake.frame_index()==queued_frame &&
                   !queued_wake.body_is_sleeping(left) &&
                   queued_wake.body_is_sleeping(right),
                   "Failed Dy frame consumed a pending source activation");
        }
        queued_wake.advance_collision();
        expect(!queued_wake.body_is_sleeping(right),
               "Quiet collision publication discarded the queued source wake");
    }

    {
        // The speculative graph may contain a pair of overlapping AABBs
        // whose rotated Gu boxes never touch. PhysX's third pass uses
        // accurate islands to decide rigid-body deactivation; only
        // contact interactions use speculative deactivation state.
        // Keep one rod awake without translating far enough to destroy
        // the SAP overlap, while its accurate neighbor is allowed to sleep.
        Scene distinct_sleep{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto rod_a=distinct_sleep.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,0},1.0f);
        const auto rod_b=distinct_sleep.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,10,5},1.0f);
        for(fyzz::BodyId id:{rod_a,rod_b}) {
            distinct_sleep.set_velocity(id,{0,9.81f,0});
            distinct_sleep.set_angular_velocity(id,{0,0.83f,0});
        }
        distinct_sleep.step(1.0f);
        for(fyzz::BodyId id:{rod_a,rod_b}) {
            distinct_sleep.set_angular_velocity(id,{});
            distinct_sleep.set_velocity(id,{});
        }
        distinct_sleep.set_position(rod_a,{0,0.1f,0});
        distinct_sleep.set_position(rod_b,{0.6f,0.1f,0.6f});
        const auto source_floor=distinct_sleep.add_static_floor(0.0f);
        distinct_sleep.advance_collision();
        expect(has(distinct_sleep.pairs(),rod_a,rod_b) &&
               distinct_sleep.speculative_island_count()==1 &&
               distinct_sleep.accurate_island_count()==2,
               "Sleep fixture failed to separate speculative and accurate islands");
        bool independent_sleep=false;
        for(int frame=0;frame<160;++frame) {
            distinct_sleep.set_velocity(rod_b,{0,0,0.25f});
            distinct_sleep.step(1.0f/120.0f);
            if(distinct_sleep.body_is_sleeping(rod_a) &&
               !distinct_sleep.body_is_sleeping(rod_b) &&
               distinct_sleep.speculative_island_count()==1 &&
               distinct_sleep.accurate_island_count()==2) {
                const auto active=distinct_sleep.active_accurate_nodes();
                expect(active.size()==1 && active[0]==rod_b &&
                       distinct_sleep.accurate_active_node_index(rod_b)==0 &&
                       distinct_sleep.accurate_active_node_index(rod_a)==
                           std::numeric_limits<std::size_t>::max(),
                       "Accurate active node list included dormant speculative-only peer");
                const auto floor_edge=distinct_sleep.contact_edge_index(
                    fyzz::native::Pair{rod_a,source_floor});
                const auto accurate_events=
                    distinct_sleep.deactivating_accurate_contact_edges();
                expect(floor_edge!=std::numeric_limits<std::uint32_t>::max(),
                       "Independent accurate sleep lost floor manager handle");
                expect(std::count(accurate_events.begin(),
                                  accurate_events.end(),floor_edge)==1 &&
                       !distinct_sleep.accurate_contact_edge_active(
                           fyzz::native::Pair{rod_a,source_floor}) &&
                       distinct_sleep.speculative_contact_edge_active(
                           fyzz::native::Pair{rod_a,source_floor}) &&
                       distinct_sleep.contact_interaction_active(
                           fyzz::native::Pair{rod_a,source_floor}),
                       "Accurate sleep did not transition only its active edge");
                expect(distinct_sleep.deactivating_speculative_contact_edges().empty(),
                       "Speculative sleep edges published while its island remained awake");
                independent_sleep=true;
                break;
            }
        }
        expect(independent_sleep,
               "Accurate rigid island could not deactivate independently of SAP candidate");
        // Source processLostEdges also deactivates a speculative island
        // that becomes entirely inactive after its awake SAP-only bridge
        // is removed. No new accurate rigid node falls asleep in this
        // publication, so sourcing candidates solely from the accurate
        // deactivation output misses this edge event.
        const auto sleeping_floor=distinct_sleep.contact_edge_index(
            fyzz::native::Pair{rod_a,source_floor});
        distinct_sleep.set_position(rod_b,{12.0f,0.1f,12.0f});
        distinct_sleep.advance_collision();
        const auto speculative_after_split=
            distinct_sleep.deactivating_speculative_contact_edges();
        expect(has(distinct_sleep.lost_pairs(),rod_a,rod_b) &&
               distinct_sleep.speculative_island_count()==2 &&
               distinct_sleep.body_is_sleeping(rod_a) &&
               !distinct_sleep.body_is_sleeping(rod_b) &&
               sleeping_floor!=std::numeric_limits<std::uint32_t>::max() &&
               std::count(speculative_after_split.begin(),
                          speculative_after_split.end(),sleeping_floor)==1 &&
               distinct_sleep.deactivating_accurate_contact_edges().empty(),
               "Lost SAP-only bridge missed source speculative island sleep event");
        distinct_sleep.advance_collision();
        expect(distinct_sleep.deactivating_speculative_contact_edges().empty() &&
               !distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               !distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}),
               "Quiet publication replayed an already inactive edge");
        const auto before_wake_spec=
            distinct_sleep.active_speculative_contact_edge_count();
        const auto before_wake_acc=
            distinct_sleep.active_accurate_contact_edge_count();
        distinct_sleep.set_velocity(rod_a,{0,0,0.25f});
        distinct_sleep.advance_collision();
        const auto activated_spec=
            distinct_sleep.activated_speculative_contact_edges();
        const auto activated_acc=
            distinct_sleep.activated_accurate_contact_edges();
        expect(distinct_sleep.contact_edge_index(
                   fyzz::native::Pair{rod_a,source_floor})==sleeping_floor &&
               distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               std::count(activated_spec.begin(),activated_spec.end(),
                          sleeping_floor)==1 &&
               std::count(activated_acc.begin(),activated_acc.end(),
                          sleeping_floor)==1 &&
               distinct_sleep.active_speculative_contact_edge_count()==
                   before_wake_spec+1 &&
               distinct_sleep.active_accurate_contact_edge_count()==
                   before_wake_acc+1,
               "Retained manager did not publish source edge activation");
        distinct_sleep.advance_collision();
        expect(distinct_sleep.activated_speculative_contact_edges().empty() &&
               distinct_sleep.activated_accurate_contact_edges().empty() &&
               distinct_sleep.speculative_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}),
               "eACTIVATING was replayed or persistent eACTIVE was lost");
        // A complete membership rebuild may change island handles but
        // must not reset Edge::eACTIVE or mActiveEdgeCount. The new
        // distant actor creates no accurate contact with the old floor.
        distinct_sleep.add_dynamic_box(
            {0.25f,0.25f,0.25f},{50.0f,50.0f,50.0f},1.0f);
        distinct_sleep.advance_collision();
        expect(distinct_sleep.accurate_contact_edge_active(
                   fyzz::native::Pair{rod_a,source_floor}) &&
               distinct_sleep.active_accurate_contact_edge_count()==
                   before_wake_acc+1,
               "Source edge active count was reset by actor admission");
        distinct_sleep.set_velocity(rod_a,{0,0,0.25f});
        distinct_sleep.advance_collision();
        expect(distinct_sleep.activated_accurate_contact_edges().empty() &&
               distinct_sleep.activated_speculative_contact_edges().empty(),
               "Already awake rigid body incorrectly reactivated active edges");
    }

    {
        // Newly published contacts and island activation precede
        // unconstrained velocity integration. On a later overflow,
        // failure atomicity must roll back both the fresh contact
        // manager and the previously sleeping peer's activation. The
        // identical valid retry must admit both bodies to the solver.
        Scene wake_rollback{Scene::ContactGenerator::DiagnosticLegacyGu};
        wake_rollback.add_static_floor(0.0f);
        const auto sleeper=wake_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            wake_rollback.step(1.0f/120.0f);
        expect(wake_rollback.body_is_sleeping(sleeper),
               "Wake rollback source fixture did not reach sleep");
        const auto newcomer=wake_rollback.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,0.5f,0},1.0f);
        const auto prior_sleeper=wake_rollback.body_state(sleeper);
        const auto prior_newcomer=wake_rollback.body_state(newcomer);
        const auto prior_contacts=wake_rollback.contacts().size();
        bool rejected=false;
        try {wake_rollback.step(std::numeric_limits<float>::max());}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               wake_rollback.body_is_sleeping(sleeper) &&
               wake_rollback.body_wake_counter(sleeper)==0.0f &&
               wake_rollback.body_state(sleeper).position.y==
                   prior_sleeper.position.y &&
               wake_rollback.body_state(newcomer).position.x==
                   prior_newcomer.position.x &&
               wake_rollback.contacts().size()==prior_contacts,
               "Failed activation frame published wake or new Gu contact");
        wake_rollback.step(1.0f/120.0f);
        expect(!wake_rollback.body_is_sleeping(sleeper) &&
               wake_rollback.accurate_island_id(sleeper)==
                   wake_rollback.accurate_island_id(newcomer),
               "Source activation rollback lost new-touch retry");
        // PxvDynamics.h initializes the unconfigured rigid body's
        // maxPenBias to -1e32f, not an implicit -3 m/s clamp.
        // The source default is effectively unbounded for this contact.
        // An incorrect finite -8 or -10 m/s clamp also must not pass.
        // The reference-enabled suite checks a stronger deep-contact
        // short-timestep case against actual PhysX legacy Gu/PGS.
        expect(wake_rollback.body_state(sleeper).position.x < -0.05f,
               "Source default penetration bias was spuriously capped");

    }

    {
        // Sc static-actor pose authoring invalidates interactions with
        // an already sleeping rigid body. Reusing the existing accurate
        // contact managers wakes the touched actor before subsequent
        // Gu/PGS progression; quiet static-world contacts do not.
        Scene moving_ground{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=moving_ground.add_static_floor(0.0f);
        const auto body=moving_ground.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            moving_ground.step(1.0f/120.0f);
        expect(moving_ground.body_is_sleeping(body),
               "Source static wake fixture never reached sleep");
        moving_ground.set_position(floor,{0,0.1f,0});
        expect(!moving_ground.body_is_sleeping(body),
               "Static pose change left its touching rigid body asleep");
        moving_ground.step(1.0f/120.0f);
        expect(!moving_ground.body_state(body).sleeping &&
               std::isfinite(moving_ground.body_state(body).position.y),
               "Native PGS could not advance rigid body after static wake");
    }

    {
        // A geometry-only Gu publication can create a static touch
        // before the next step. IslandSim::wakeIslands consumes it at
        // publication, not from a one-frame event stream that may be
        // cleared by a later quiet collision query.
        Scene collision_wake{Scene::ContactGenerator::DiagnosticLegacyGu};
        collision_wake.add_static_floor(0);
        const auto body=collision_wake.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1);
        for(int frame=0;frame<360;++frame)
            collision_wake.step(1.0f/120.0f);
        expect(collision_wake.body_is_sleeping(body),
               "Geometry-only wake fixture failed to reach sleep");
        collision_wake.add_static_box(
            {0.5f,0.5f,0.5f},{0.85f,0.5f,0});
        collision_wake.advance_collision();
        expect(!collision_wake.body_is_sleeping(body) &&
               collision_wake.accurate_island_static_touches(body)==2,
               "Gu-only publication lost the static contact wake");
        collision_wake.advance_collision();
        expect(!collision_wake.body_is_sleeping(body),
               "Quiet collision query lost previously published activation");
        collision_wake.step(1.0f/120.0f);
        expect(!collision_wake.body_state(body).sleeping &&
               std::isfinite(collision_wake.body_state(body).velocity.y),
               "Geometry-published source activation failed solver admission");
    }

    {
        // Sc::Scene destroys contact-manager links when a supporting
        // static actor is retired. A sleeping box must wake and resume
        // falling rather than remain indefinitely suspended in space.
        Scene removed_support{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=removed_support.add_static_floor(0);
        const auto box=removed_support.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        for(int frame=0;frame<360;++frame)
            removed_support.step(1.0f/120.0f);
        expect(removed_support.body_is_sleeping(box),
               "Retired-support fixture failed to reach source sleep");
        const float old_y=removed_support.body_state(box).position.y;
        removed_support.remove(floor);
        expect(!removed_support.body_is_sleeping(box),
               "Retiring static support failed to reactivate sleeping box");
        for(int frame=0;frame<8;++frame)
            removed_support.step(1.0f/120.0f);
        expect(removed_support.body_state(box).position.y<old_y-0.005f &&
               removed_support.body_state(box).velocity.y<0.0f,
               "Source rigid body did not fall after losing static support");
    }

}

void verify_source_rigid_lock_flags() {
    using namespace fyzz;
    World native(WorldOptions{ExecutionMode::NativeCpu});
    const auto id=native.add_dynamic_box({0.35f,0.45f,0.6f},{0,5,0},1.0f);
    RigidBodyDynamics d{};
    d.lock_flags=LockLinearY | LockAngularZ;
    native.set_body_dynamics(id,d);
    native.set_body_velocity(id,{0.7f,2.0f,-0.4f});
    native.set_body_angular_velocity(id,{0.8f,-1.1f,3.0f});
    const auto start=native.body_state(id);
    for(int frame=0;frame<96;++frame) {
        native.step(frame<48?1.0f/120.0f:1.0f/60.0f);
        const auto state=native.body_state(id);
        expect(std::abs(state.position.y-start.position.y)<1.0e-6f &&
               state.velocity.y==0.0f &&
               state.angular_velocity.z==0.0f,
               "Source locks failed in native unconstrained integration");
    }
    const auto locked=native.body_state(id);
    expect(locked.position.x>start.position.x+0.1f &&
           std::abs(locked.angular_velocity.x)>0.01f,
           "Source locks incorrectly suppressed unlocked motion");
    auto invalid=d;
    invalid.lock_flags=0x40u;
    bool rejected=false;
    try {native.set_body_dynamics(id,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"Undefined PhysX rigid lock flag was admitted");
    native.step(1.0f/120.0f);
    expect(native.body_state(id).velocity.y==0.0f,
           "Rejected lock edit mutated the active body");
    d.lock_flags=0;
    native.set_body_dynamics(id,d);
    native.step(1.0f/120.0f);
    expect(native.body_state(id).position.y<locked.position.y,
           "Releasing source linear lock did not restore gravity");
    World contact(WorldOptions{ExecutionMode::NativeCpu});
    contact.add_static_floor(0.0f);
    const auto body=contact.add_dynamic_box({0.5f,0.5f,0.5f},
                                            {0,0.54f,0},1.0f);
    d.lock_flags=LockAngularX | LockAngularY | LockAngularZ;
    contact.set_body_dynamics(body,d);
    contact.set_body_velocity(body,{0.45f,-0.3f,0});
    for(int frame=0;frame<64;++frame) {
        contact.step(1.0f/120.0f);
        const auto state=contact.body_state(body);
        expect(state.angular_velocity.x==0.0f &&
               state.angular_velocity.y==0.0f &&
               state.angular_velocity.z==0.0f,
               "Locked angular axes were bypassed by production PGS");
    }
}

void verify_source_gyroscopic_motion() {
    using namespace fyzz;
    World enabled(WorldOptions{ExecutionMode::NativeCpu});
    World disabled(WorldOptions{ExecutionMode::NativeCpu});
    constexpr Vec3 half{0.3f,0.5f,0.9f};
    const auto a=enabled.add_dynamic_box(half,{0,10,0},1.0f);
    const auto b=disabled.add_dynamic_box(half,{0,10,0},1.0f);
    RigidBodyDynamics d{};
    d.gravity_enabled=false;
    d.gyroscopic_forces_enabled=true;
    enabled.set_body_dynamics(a,d);
    d.gyroscopic_forces_enabled=false;
    disabled.set_body_dynamics(b,d);
    enabled.set_body_angular_velocity(a,{1.2f,1.6f,-1.9f});
    disabled.set_body_angular_velocity(b,{1.2f,1.6f,-1.9f});
    float maximum_delta=0.0f;
    for(int frame=0;frame<128;++frame) {
        const float dt=frame<64?1.0f/120.0f:1.0f/240.0f;
        enabled.step(dt);
        disabled.step(dt);
        const auto e=enabled.body_state(a);
        const auto off=disabled.body_state(b);
        maximum_delta=std::max(maximum_delta,
            std::abs(e.angular_velocity.x-off.angular_velocity.x));
        maximum_delta=std::max(maximum_delta,
            std::abs(e.angular_velocity.y-off.angular_velocity.y));
        maximum_delta=std::max(maximum_delta,
            std::abs(e.angular_velocity.z-off.angular_velocity.z));
        expect(std::isfinite(e.angular_velocity.x) &&
               std::isfinite(e.angular_velocity.y) &&
               std::isfinite(e.angular_velocity.z),
               "Source gyroscopic angular update became nonfinite");
    }
    expect(maximum_delta>0.05f,
           "Native gyroscopic flag had no observable inertia response");
}

void verify_source_spatial_velocity_mods() {
    using namespace fyzz;
    constexpr float dt=1.0f/60.0f;
    World native(WorldOptions{ExecutionMode::NativeCpu});
    const auto id=native.add_dynamic_box({0.5f,0.5f,0.5f},{0,10,0},1.0f);
    RigidBodyDynamics d{};
    d.gravity_enabled=false;
    d.angular_damping=0.0f;
    native.set_body_dynamics(id,d);
    // Four PhysX force modes share two accumulation lifetimes.
    native.add_body_force(id,{6,0,0},ForceMode::Force);
    native.add_body_force(id,{6,0,0},ForceMode::Force);
    native.add_body_force(id,{0.3f,0,0},ForceMode::Impulse);
    native.add_body_force(id,{2,0,0},ForceMode::Acceleration);
    native.add_body_force(id,{0.25f,0,0},ForceMode::VelocityChange);
    native.add_body_torque(id,{0,0,0.1f},ForceMode::Impulse);
    native.add_body_torque(id,{0,0,0.2f},ForceMode::Force);
    native.step(dt);
    const auto first=native.body_state(id);
    const float expected=12.0f*dt+0.3f+2.0f*dt+0.25f;
    expect(std::abs(first.velocity.x-expected)<2.0e-5f,
           "Native force modes did not combine per-step and per-second motion");
    expect(first.angular_velocity.z>0.0f,
           "PhysX angular force/impulse did not reach native integration");
    native.step(dt);
    const auto second=native.body_state(id);
    expect(std::abs(second.velocity.x-first.velocity.x)<1.0e-6f &&
           std::abs(second.angular_velocity.z-first.angular_velocity.z)<1.0e-6f,
           "PhysX VelocityMod streams incorrectly persisted past one step");
    // Each source clear operation retires its own accumulation category.
    native.add_body_force(id,{600,0,0},ForceMode::Force);
    native.clear_body_force(id,ForceMode::Acceleration);
    native.add_body_force(id,{1,0,0},ForceMode::Impulse);
    native.clear_body_force(id,ForceMode::VelocityChange);
    native.add_body_torque(id,{0,0,40},ForceMode::Force);
    native.clear_body_torque(id,ForceMode::Force);
    native.step(dt);
    expect(std::abs(native.body_state(id).velocity.x-second.velocity.x)<1.0e-6f,
           "Clearing matching PhysX velocity-mod categories failed");
    bool rejected=false,static_rejected=false;
    try {native.add_body_force(id,{0,0,0},
            static_cast<ForceMode>(100));}
    catch(const std::invalid_argument&){rejected=true;}
    const auto floor=native.add_static_floor(0);
    try {native.add_body_torque(floor,{0,1,0},ForceMode::Impulse);}
    catch(const std::logic_error&){static_rejected=true;}
    expect(rejected && static_rejected,
           "PhysX force-mode and static-body rejection regressed");

    // An invalid Dy frame cannot consume either per-second forces or
    // per-step impulses. Replaying with the same smaller dt must match
    // a separately initialized scene, including angular correction.
    World retry(WorldOptions{ExecutionMode::NativeCpu});
    World baseline(WorldOptions{ExecutionMode::NativeCpu});
    const auto make=[&](World& world) {
        const auto body=world.add_dynamic_box(
            {0.3f,0.4f,0.8f},{0,8,0},1.0f);
        world.set_body_dynamics(body,d);
        world.add_body_force(body,{0.7f,1.1f,0.3f},ForceMode::Impulse);
        world.add_body_force(body,{1.5f,0.2f,0.1f},ForceMode::Force);
        world.add_body_torque(body,{0.02f,0.03f,0.04f},ForceMode::Impulse);
        world.add_body_torque(body,{0.1f,0.2f,0.3f},ForceMode::Force);
        return body;
    };
    const auto r=make(retry),b=make(baseline);
    bool failed=false;
    try {retry.step(std::numeric_limits<float>::max());}
    catch(const std::exception&) {failed=true;}
    expect(failed,"Force rollback fixture did not reject overflowing frame");
    retry.step(dt);
    baseline.step(dt);
    const auto a=retry.body_state(r),z=baseline.body_state(b);
    const auto close=[](Vec3 x,Vec3 y) {
        return std::abs(x.x-y.x)<1.0e-6f &&
               std::abs(x.y-y.y)<1.0e-6f &&
               std::abs(x.z-y.z)<1.0e-6f;
    };
    expect(close(a.position,z.position) &&
           close(a.velocity,z.velocity) &&
           close(a.angular_velocity,z.angular_velocity),
           "Rejected solver frame consumed pending ScBodySim force state");

    // Deliver authored force and impulse through real contact PGS,
    // not merely the no-contact Dy integrator.
    World contact(WorldOptions{ExecutionMode::NativeCpu});
    contact.add_static_floor(0);
    const auto box=contact.add_dynamic_box(
        {0.5f,0.5f,0.5f},{0,0.53f,0},1.0f);
    for(int frame=0;frame<80;++frame) {
        if(frame%7==0)
            contact.add_body_force(box,{0,-1.5f,0},ForceMode::Force);
        if(frame==20)
            contact.add_body_force(box,{0.1f,-0.2f,0},ForceMode::Impulse);
        contact.step(1.0f/120.0f);
        const auto state=contact.body_state(box);
        expect(std::isfinite(state.position.y) &&
               std::isfinite(state.velocity.y) &&
               std::isfinite(state.angular_velocity.x),
               "ScBodySim force integration made production PGS nonfinite");
    }
}

int run_native_scene_smoke() {
    using fyzz::native::Scene;
    Scene scene, independent;
    auto wall = scene.add_static_box({1,1,1}, {0,0,0});
    auto static2 = scene.add_static_box({1,1,1}, {1,0,0});
    auto moving = scene.add_dynamic_box({0.5f,0.5f,0.5f}, {5,0,0}, 1);
    auto second = independent.add_dynamic_box({0.5f,0.5f,0.5f},{1,1,1},1);
    scene.advance_collision();
    expect(scene.pairs().empty(), "static/static or distant pair admitted");
    scene.set_position(moving, {1.5f,0,0});
    scene.advance_collision();
    expect(has(scene.found_pairs(), wall, moving) &&
           has(scene.found_pairs(), static2, moving) &&
           scene.pairs().size() == 2 && scene.lost_pairs().empty(),
           "source-ordered found pairs absent");
    scene.advance_collision();
    expect(scene.found_pairs().empty() && scene.lost_pairs().empty(),
           "quiet frame changed incremental overlap state");
    scene.advance_collision(true);
    expect(scene.found_pairs().empty() && scene.lost_pairs().empty(),
           "forced unchanged update emitted pairs");
    scene.set_position(moving,{10,0,0});
    scene.advance_collision();
    expect(scene.pairs().empty() && scene.lost_pairs().size() == 2,
           "movement did not retire previous pair state");
    scene.set_position(moving,{1,0,0});
    scene.advance_collision();
    expect(scene.found_pairs().size() == 2, "overlap re-entry absent");
    scene.remove(wall);
    scene.advance_collision();
    expect(scene.lost_pairs().size() == 1 &&
           has(scene.lost_pairs(), wall, moving), "retired actor leaked pair");
    bool stale = false;
    try { (void)scene.body_state(wall); }
    catch (const std::out_of_range&) { stale = true; }
    expect(stale, "retired body ID still accessible");
    expect(scene.add_static_box({1,1,1},{100,0,0}) > moving,
           "retired body identifier reused");
    scene.advance_collision();
    expect(scene.found_pairs().empty(), "distant actor created false pair");
    expect(independent.frame_index() == 0 &&
           independent.body_state(second).position.x == 1,
           "separate native scene was modified");
    auto floor = independent.add_static_floor(-100);
    independent.set_position(floor, {0, 0.75f, 0});
    expect(independent.body_state(floor).position.y == 0.75f,
           "Native static plane failed to relocate");
    independent.advance_collision();
    expect(has(independent.found_pairs(), floor, second),
           "infinite floor broadphase candidate missing");
    independent.remove(second);
    independent.advance_collision();
    expect(has(independent.lost_pairs(), floor, second),
           "infinite floor lost pair missing");
    bool invalid = false;
    try { (void)scene.add_dynamic_box({1,1,1},{INFINITY,0,0},1); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "invalid native geometry admitted");
    expect(scene.frame_index() == 8, "scene frame counter drift");


    // An unforced quiet geometry publication has no throwing work and
    // must not checkpoint the complete edge/island state. It still
    // advances the publication frame, clears transient previous-step
    // results/events, and retains the original contact manager identity.
    {
        Scene quiet{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=quiet.add_static_floor(0.0f);
        const auto box=quiet.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        quiet.step(1.0f/120.0f);
        const fyzz::native::Pair pair{floor,box};
        const auto handle=quiet.contact_edge_index(pair);
        const auto revision=quiet.island_topology_revision();
        const auto rebuilds=quiet.island_full_rebuild_count();
        const auto start=quiet.frame_index();
        expect(!quiet.solved_normal_impulses().empty() &&
               handle!=std::numeric_limits<std::uint32_t>::max(),
               "Quiet collision fixture has no prior solved contact");
        for(int i=0;i<4;++i) {
            quiet.advance_collision();
            expect(quiet.frame_index()==start+static_cast<std::uint64_t>(i+1) &&
                   quiet.found_pairs().empty() && quiet.lost_pairs().empty() &&
                   quiet.found_contacts().empty() && quiet.lost_contacts().empty() &&
                   quiet.solved_normal_impulses().empty() &&
                   quiet.solved_friction_patches().empty() &&
                   quiet.force_threshold_changes().empty() &&
                   quiet.contact_edge_index(pair)==handle &&
                   quiet.contact_edge_connected(pair) &&
                   quiet.island_topology_revision()==revision &&
                   quiet.island_full_rebuild_count()==rebuilds,
                   "Quiet collision publication changed contact or island state");
        }
        quiet.advance_collision(true);
        expect(quiet.frame_index()==start+5 &&
               quiet.contact_edge_index(pair)==handle &&
               quiet.island_topology_revision()==revision,
               "Forced collision refresh lost persistent contact edge identity");
    }

    // SimpleIslandManager::addContactManager obtains an EdgeIndex once
    // per SAP candidate; setEdgeDisconnected does not retire it on Gu
    // touch loss. removeConnection defers release until the joined
    // post-third-pass retirement, including overlapping found/lost pairs.
    {
        Scene handles{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=handles.add_static_floor(0.0f);
        const auto first=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const fyzz::native::Pair first_pair{floor,first};
        handles.advance_collision();
        const auto first_edge=handles.contact_edge_index(first_pair);
        const auto invalid=std::numeric_limits<std::uint32_t>::max();
        expect(first_edge!=invalid &&
               handles.contact_edge_connected(first_pair) &&
               handles.active_contact_edge_count()==1,
               "PhysX contact manager did not acquire a persistent edge");
        handles.set_position(first,{0,8,0});
        handles.advance_collision();
        expect(handles.contact_edge_index(first_pair)==first_edge &&
               !handles.contact_edge_connected(first_pair) &&
               handles.active_contact_edge_count()==1,
               "Physical touch loss discarded the speculative EdgeIndex");
        handles.set_position(first,{0,0.5f,0});
        handles.advance_collision();
        expect(handles.contact_edge_index(first_pair)==first_edge &&
               handles.contact_edge_connected(first_pair),
               "Touch reentry allocated a different contact-manager edge");
        // Retiring one overlap and discovering another in the same
        // publication cannot reuse the old source handle prematurely.
        handles.remove(first);
        const auto second=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
        handles.advance_collision();
        const fyzz::native::Pair second_pair{floor,second};
        const auto second_edge=handles.contact_edge_index(second_pair);
        expect(second_edge!=invalid && second_edge!=first_edge &&
               handles.contact_edge_index(first_pair)==invalid &&
               handles.active_contact_edge_count()==1,
               "Removed contact edge was reused before third-pass retirement");
        handles.remove(second);
        const auto third=handles.add_dynamic_box(
            {0.5f,0.5f,0.5f},{6,0.5f,0},1.0f);
        handles.advance_collision();
        const fyzz::native::Pair third_pair{floor,third};
        expect(handles.contact_edge_index(third_pair)==first_edge &&
               handles.contact_edge_connected(third_pair) &&
               handles.active_contact_edge_count()==1,
               "Retired contact edge handle was not returned to source pool");
    }

    // PxsSimpleIslandManager's two distinct IslandSim graphs, for the
    // currently admitted all-awake rigid-body subset. Broadphase
    // candidates build speculative connectivity, while only resolved
    // touching dynamic pairs join accurate solver islands. The static
    // world is never a dynamic connectivity bridge.
    {
        Scene connected{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,5.0f,0.0f},1.0f);
        const auto b=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,5.0f,0.0f},1.0f);
        const auto c=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,5.0f,0.0f},1.0f);
        const auto isolated=connected.add_dynamic_box(
            {0.5f,0.5f,0.5f},{10.0f,5.0f,0.0f},1.0f);
        connected.advance_collision();
        expect(connected.contacts().size()==2 &&
               connected.speculative_island_count()==2 &&
               connected.accurate_island_count()==2,
               "Source first/second island graphs failed to merge the touching chain");
        const auto revision=connected.island_topology_revision();
        const auto initial_rebuilds=connected.island_full_rebuild_count();
        const auto initial_local_splits=connected.island_local_split_count();
        expect(initial_rebuilds==1,
               "First active island topology was not built once");
        connected.advance_collision(true);
        expect(connected.island_topology_revision()==revision,
               "Unchanged island connectivity rebuilt expensive graph state");
        // Removing the center's two dynamic contact edges must split the
        // component into three independent accurate islands, and the
        // still-alive isolated node contributes its own singleton.
        connected.set_position(b,{5.0f,5.0f,0.0f});
        connected.advance_collision();
        expect(connected.speculative_island_count()==4 &&
               connected.accurate_island_count()==4 &&
               connected.island_topology_revision()==revision+1 &&
               connected.island_full_rebuild_count()==initial_rebuilds &&
               connected.island_local_split_count()>initial_local_splits,
               "Lost source contact edges rebuilt unrelated island graphs");
        connected.set_position(b,{0.85f,5.0f,0.0f});
        connected.advance_collision();
        expect(connected.speculative_island_count()==2 &&
               connected.accurate_island_count()==2 &&
               connected.island_full_rebuild_count()==initial_rebuilds,
               "Restored source contact edges did not incrementally rejoin islands");
        connected.remove(b);
        connected.advance_collision();
        expect(connected.speculative_island_count()==3 &&
               connected.accurate_island_count()==3 &&
               connected.island_full_rebuild_count()==initial_rebuilds+1,
               "Destroyed source island node remained active in the graph");
        connected.step(1.0f/120.0f);
        expect(std::isfinite(connected.body_state(a).position.y) &&
               std::isfinite(connected.body_state(c).position.y) &&
               std::isfinite(connected.body_state(isolated).position.y) &&
               connected.body_state(isolated).position.y<5.0f,
               "Native island-node admission omitted free-flight integration");

        // Source task continuations can overlap the numerical solve.
        // The admitted synchronous schedule processes completed lost
        // contact events before solving, and the split result must be
        // published for the new frame through the same native scene.
        Scene phased{Scene::ContactGenerator::DiagnosticLegacyGu};
        phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto middle=phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.85f,5,0},1.0f);
        phased.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.7f,5,0},1.0f);
        phased.advance_collision();
        expect(phased.accurate_island_count()==1,
               "Third-pass fixture must begin with one accurate island");
        phased.set_position(middle,{5.0f,5,0});
        phased.step(1.0f/120.0f);
        expect(phased.accurate_island_count()==3 &&
               phased.largest_accurate_island_size()==1 &&
               phased.contacts().empty(),
               "Source lost-contact retirement did not split accurate islands");
        phased.step(1.0f/120.0f);
        expect(phased.accurate_island_count()==3 &&
               phased.contacts().empty(),
               "Next-frame island admission reused retired contact edges");

        // IslandSim::findRoute: deleting a non-bridge edge must not
        // split a three-node cycle. After that, deleting the surviving
        // two-edge route really must split the dynamic components.
        // Both speculative SAP and accurate Gu graphs must agree on
        // the independently published resulting connectivity.
        Scene cycle{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto ca=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,8,0},1.0f);
        const auto cb=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,8,0},1.0f);
        const auto cc=cycle.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,8,0},1.0f);
        cycle.advance_collision();
        expect(cycle.pairs().size()==3 &&
               cycle.contacts().size()==3 &&
               cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1,
               "Lost-edge route fixture did not begin as a connected cycle");
        const auto cycle_rebuilds=cycle.island_full_rebuild_count();
        cycle.set_position(cc,{1.3f,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().size()==2 &&
               cycle.contacts().size()==2 &&
               cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.largest_accurate_island_size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Source lost-edge route was discarded despite an alternate path");
        // A-C is the only removed chord. Source processNewEdges merges
        // A-B with a size tie, retaining B's island root; both A and C
        // are non-root dirty endpoints. Reintroducing A-C must retain
        // both cached routes through B in each graph (four hits total).
        const auto cycle_fast_hits=cycle.island_fast_route_hit_count();
        cycle.set_position(cc,{0.8f,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().size()==3 &&
               cycle.contacts().size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Redundant cycle chord insertion rebuilt the island");
        cycle.set_position(cc,{1.3f,8,0});
        cycle.advance_collision();
        expect(cycle.lost_pairs().size()==1 &&
               cycle.lost_contacts().size()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.speculative_island_count()==1 &&
               cycle.island_fast_route_hit_count()==cycle_fast_hits+4 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Redundant cycle chord invalidated its alternate fast route");
        // A-C is no longer connected. Removing B's remaining two
        // contacts must trigger actual split repair, not the route skip.
        cycle.set_position(cb,{5,8,0});
        cycle.advance_collision();
        expect(cycle.pairs().empty() && cycle.contacts().empty() &&
               cycle.speculative_island_count()==3 &&
               cycle.accurate_island_count()==3 &&
               cycle.largest_accurate_island_size()==1 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds &&
               cycle.island_local_split_count()>0,
               "Source island split lost its final surviving route");
        cycle.set_position(cb,{0.4f,8,0});
        cycle.advance_collision();
        expect(cycle.speculative_island_count()==1 &&
               cycle.accurate_island_count()==1 &&
               cycle.largest_accurate_island_size()==3 &&
               cycle.island_full_rebuild_count()==cycle_rebuilds,
               "Source lost-edge repair did not admit incremental rejoin");

        // Multiple dynamic losses in one collision publication: each
        // removed edge has a surviving replacement path in the final
        // graph. The proof must not stop at the first still-connected
        // pair, nor treat multiple redundant removals as a split.
        Scene redundant_losses{Scene::ContactGenerator::DiagnosticLegacyGu};
        redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,12,0},1.0f);
        redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,12,0},1.0f);
        const auto rc=redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,12,0},1.0f);
        const auto rd=redundant_losses.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.2f,12,0},1.0f);
        redundant_losses.advance_collision();
        expect(redundant_losses.pairs().size()==5 &&
               redundant_losses.contacts().size()==5 &&
               redundant_losses.accurate_island_count()==1,
               "Multi-edge source route fixture did not begin connected");
        const auto multi_rebuilds=
            redundant_losses.island_full_rebuild_count();
        redundant_losses.set_position(rc,{1.2f,12,0});
        redundant_losses.set_position(rd,{1.5f,12,0});
        redundant_losses.advance_collision();
        expect(redundant_losses.lost_pairs().size()==2 &&
               redundant_losses.lost_contacts().size()==2 &&
               redundant_losses.pairs().size()==3 &&
               redundant_losses.contacts().size()==3 &&
               redundant_losses.speculative_island_count()==1 &&
               redundant_losses.accurate_island_count()==1 &&
               redundant_losses.largest_accurate_island_size()==4 &&
               redundant_losses.island_full_rebuild_count()==multi_rebuilds,
               "Multiple redundant edge losses incorrectly rebuilt an intact island");

        // Source EdgeInstance adjacency must survive handle retirement
        // and reuse without retaining stale next/prev endpoint links.
        // Retire the A-C edge of a three-node cycle, recycle its handle
        // for A-D, then delete A-D while B still provides another route.
        // Finally remove B and verify real component split repair.
        Scene relink{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto aa=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,15,0},1.0f);
        const auto bb=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,15,0},1.0f);
        const auto relink_c=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,15,0},1.0f);
        relink.advance_collision();
        const fyzz::native::Pair ac{aa,relink_c};
        const auto recycled=relink.contact_edge_index(ac);
        expect(relink.pairs().size()==3 &&
               relink.contacts().size()==3 &&
               recycled!=std::numeric_limits<std::uint32_t>::max(),
               "Intrusive adjacency fixture did not establish a three-edge cycle");
        relink.set_position(relink_c,{1.3f,15,0});
        relink.advance_collision();
        expect(relink.contact_edge_index(ac)==
                   std::numeric_limits<std::uint32_t>::max() &&
               relink.accurate_island_count()==1,
               "Retired source graph edge leaked into surviving component");
        const auto dd=relink.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,15,0},1.0f);
        relink.advance_collision();
        const fyzz::native::Pair ad{aa,dd};
        expect(relink.contact_edge_index(ad)==recycled &&
               relink.accurate_island_count()==1 &&
               relink.largest_accurate_island_size()==4,
               "Recycled edge handle failed to link its new island endpoints");
        // Adding a dynamic node necessarily rebuilds the admitted island
        // roster; compare only the subsequent redundant edge retirement.
        const auto before_recycle=relink.island_full_rebuild_count();
        relink.set_position(dd,{1.3f,15,0});
        relink.advance_collision();
        expect(relink.contact_edge_index(ad)==
                   std::numeric_limits<std::uint32_t>::max() &&
               relink.accurate_island_count()==1 &&
               relink.island_full_rebuild_count()==before_recycle,
               "Recycled edge unlink corrupted an alternate island route");
        relink.set_position(bb,{5,15,0});
        relink.advance_collision();
        // Removing B strands both A and B as singleton islands,
        // while the overlapping C-D pair remains a third island.
        expect(relink.speculative_island_count()==3 &&
               relink.accurate_island_count()==3 &&
               relink.largest_accurate_island_size()==2 &&
               relink.island_full_rebuild_count()==before_recycle &&
               relink.island_local_split_count()>0,
               "Recycled intrusive graph failed to split on lost bridge");

        // Source mDirtyMap deduplicates endpoints of overlapping lost
        // edges: A-tip and B-tip remove three distinct dynamic nodes
        // in each graph, not four entries with a repeated tip.
        // Re-adding redundant chords must retain the cached tip->C->root
        // route for the next removal; only a real union changes the root.
        Scene dirty_routes{Scene::ContactGenerator::DiagnosticLegacyGu};
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,22,0},1.0f);
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.2f,22,0},1.0f);
        dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.4f,22,0},1.0f);
        const auto route_tip=dirty_routes.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.6f,22,0},1.0f);
        dirty_routes.advance_collision();
        expect(dirty_routes.pairs().size()==6 &&
               dirty_routes.contacts().size()==6 &&
               dirty_routes.accurate_island_count()==1,
               "Dirty-route fixture did not begin fully connected");
        const auto before_rebuild=dirty_routes.island_full_rebuild_count();
        const auto before_fast=dirty_routes.island_fast_route_hit_count();
        const auto before_checks=dirty_routes.island_dirty_route_check_count();
        dirty_routes.set_position(route_tip,{1.3f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.lost_pairs().size()==2 &&
               dirty_routes.lost_contacts().size()==2 &&
               dirty_routes.accurate_island_count()==1 &&
               dirty_routes.speculative_island_count()==1 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_dirty_route_check_count()==before_checks+6 &&
               dirty_routes.island_fast_route_hit_count()==before_fast,
               "Source mDirtyMap repeated an endpoint during lost-edge repair");

        // Reintroducing these redundant chords cannot change island
        // membership. Source processNewEdges *can*, however, shorten a
        // root-directed hop estimate through one of the new edges.
        // Removing a shortened chord invalidates its own fast route.
        // Source-style shortening retains one alternate root fast path
        // per IslandSim graph, so the next loss yields two hits, not
        // the four hits from the older non-shortening implementation.
        dirty_routes.set_position(route_tip,{0.6f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.pairs().size()==6 &&
               dirty_routes.contacts().size()==6 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild,
               "Redundant chord readdition rebuilt an existing island");
        const auto fast_before_retry=dirty_routes.island_fast_route_hit_count();
        const auto checks_before_retry=dirty_routes.island_dirty_route_check_count();
        dirty_routes.set_position(route_tip,{1.3f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.lost_pairs().size()==2 &&
               dirty_routes.lost_contacts().size()==2 &&
               dirty_routes.accurate_island_count()==1 &&
               dirty_routes.speculative_island_count()==1 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_dirty_route_check_count()==checks_before_retry+6 &&
               dirty_routes.island_fast_route_hit_count()==fast_before_retry+2,
               "Redundant chord removal changed source fast-route reuse");

        // Retiring the last surviving tip edge is a real split.
        dirty_routes.set_position(route_tip,{3.0f,22,0});
        dirty_routes.advance_collision();
        expect(dirty_routes.accurate_island_count()==2 &&
               dirty_routes.speculative_island_count()==2 &&
               dirty_routes.island_full_rebuild_count()==before_rebuild &&
               dirty_routes.island_local_split_count()>0,
               "Dirty fast route falsely preserved a disconnected island");

        // IslandSim::addNode initializes only its own route state.
        // Rebuilding membership for an unrelated actor must not evict
        // the alternate paths learned in an established dynamic island.
        // Two root-directed fast hits (one per graph) must survive.
        {
            Scene admitted_routes{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto a=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.0f,26,0},1.0f);
            admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.2f,26,0},1.0f);
            admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.4f,26,0},1.0f);
            const auto tip=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.6f,26,0},1.0f);
            admitted_routes.advance_collision();
            admitted_routes.set_position(tip,{1.3f,26,0});
            admitted_routes.advance_collision();
            expect(admitted_routes.lost_pairs().size()==2 &&
                   admitted_routes.lost_contacts().size()==2,
                   "Actor-admission route fixture did not prime lost chords");
            admitted_routes.set_position(tip,{0.6f,26,0});
            admitted_routes.advance_collision();
            const auto root_accurate=admitted_routes.accurate_island_id(a);
            const auto root_speculative=admitted_routes.speculative_island_id(a);
            const auto before_rebuild=
                admitted_routes.island_full_rebuild_count();
            const auto before_hits=admitted_routes.island_fast_route_hit_count();
            const auto unrelated=admitted_routes.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,26,0},1.0f);
            admitted_routes.advance_collision();
            expect(admitted_routes.island_full_rebuild_count()==
                       before_rebuild+1 &&
                   admitted_routes.accurate_island_count()==2 &&
                   admitted_routes.speculative_island_count()==2 &&
                   admitted_routes.accurate_island_id(a)==root_accurate &&
                   admitted_routes.speculative_island_id(a)==root_speculative &&
                   admitted_routes.accurate_island_id(unrelated)!=root_accurate,
                   "Unrelated node admission changed source island ownership");
            admitted_routes.set_position(tip,{1.3f,26,0});
            admitted_routes.advance_collision();
            expect(admitted_routes.lost_pairs().size()==2 &&
                   admitted_routes.lost_contacts().size()==2 &&
                   admitted_routes.island_fast_route_hit_count()==before_hits+2 &&
                   admitted_routes.island_full_rebuild_count()==before_rebuild+1,
                   "Unrelated node admission evicted valid root routes");
        }

        // PxsIslandSim owns stable island handles, not transient
        // union/find roots. A split preserves the root's handle, an
        // unaffected island retains its handle, and a merge keeps the
        // larger island handle. Node admission/retirement must not
        // renumber unrelated source islands in either graph.
        {
            Scene lifetime{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto a=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,30,0},1.0f);
            const auto b=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.4f,30,0},1.0f);
            const auto c=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0.8f,30,0},1.0f);
            const auto isolated=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{8,30,0},1.0f);
            lifetime.advance_collision();
            const auto island=lifetime.accurate_island_id(a);
            const auto speculative=lifetime.speculative_island_id(a);
            const auto quiet=lifetime.accurate_island_id(isolated);
            const auto quiet_spec=lifetime.speculative_island_id(isolated);
            expect(lifetime.accurate_island_count()==2 &&
                   lifetime.speculative_island_count()==2 &&
                   island!=quiet && speculative!=quiet_spec &&
                   island==lifetime.accurate_island_id(b) &&
                   island==lifetime.accurate_island_id(c) &&
                   speculative==lifetime.speculative_island_id(c),
                   "Source IslandSim did not publish persistent node handles");
            lifetime.set_position(c,{3,30,0});
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.speculative_island_count()==3 &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.speculative_island_id(a)==speculative &&
                   lifetime.accurate_island_id(b)==island &&
                   lifetime.accurate_island_id(c)!=island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(isolated)==quiet_spec,
                   "Source island split renumbered old-root or unrelated island");
            lifetime.set_position(c,{0.8f,30,0});
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==2 &&
                   lifetime.speculative_island_count()==2 &&
                   lifetime.accurate_island_id(c)==island &&
                   lifetime.speculative_island_id(c)==speculative &&
                   lifetime.accurate_island_id(isolated)==quiet,
                   "Source island merge failed to retain larger island identity");
            const auto newcomer=lifetime.add_dynamic_box(
                {0.5f,0.5f,0.5f},{12,30,0},1.0f);
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(a)==speculative &&
                   lifetime.speculative_island_id(isolated)==quiet_spec &&
                   lifetime.accurate_island_id(newcomer)!=island &&
                   lifetime.accurate_island_id(newcomer)!=quiet,
                   "Source node admission renumbered existing island handles");
            lifetime.remove(b);
            lifetime.advance_collision();
            expect(lifetime.accurate_island_count()==3 &&
                   lifetime.speculative_island_count()==3 &&
                   lifetime.accurate_island_id(b)==
                       std::numeric_limits<std::uint32_t>::max() &&
                   lifetime.speculative_island_id(b)==
                       std::numeric_limits<std::uint32_t>::max() &&
                   lifetime.accurate_island_id(a)==island &&
                   lifetime.accurate_island_id(c)==island &&
                   lifetime.accurate_island_id(isolated)==quiet &&
                   lifetime.speculative_island_id(isolated)==quiet_spec,
                   "Source node retirement invalidated a live island handle");
        }

        // Actor admission may coincide with a merge of two existing
        // islands. Source mergeIslands retains the *larger* island handle
        // even if the smaller island owns the lower BodyId and appears
        // first in the full component rebuild's traversal.
        {
            Scene concurrent{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto singleton=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,36,0},1.0f);
            const auto large_root=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,36,0},1.0f);
            const auto large_member=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4.4f,36,0},1.0f);
            concurrent.advance_collision();
            const auto accurate_large=concurrent.accurate_island_id(large_root);
            const auto speculative_large=
                concurrent.speculative_island_id(large_root);
            expect(concurrent.accurate_island_count()==2 &&
                   concurrent.speculative_island_count()==2 &&
                   concurrent.accurate_island_id(singleton)!=accurate_large &&
                   concurrent.accurate_island_id(large_member)==accurate_large,
                   "Rebuild merge fixture did not begin with unequal islands");
            const auto unrelated=concurrent.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,36,0},1.0f);
            concurrent.set_position(singleton,{3.4f,36,0});
            concurrent.advance_collision();
            expect(concurrent.accurate_island_count()==2 &&
                   concurrent.speculative_island_count()==2 &&
                   concurrent.accurate_island_id(singleton)==accurate_large &&
                   concurrent.accurate_island_id(large_root)==accurate_large &&
                   concurrent.accurate_island_id(large_member)==accurate_large &&
                   concurrent.speculative_island_id(singleton)==speculative_large &&
                   concurrent.speculative_island_id(large_member)==speculative_large &&
                   concurrent.accurate_island_id(unrelated)!=accurate_large,
                   "Rebuild-time merge discarded the larger source island handle");
        }

        // IslandSim::processNewEdges grows island sizes after each
        // found edge. Two old 2-node islands merge before a 3-node
        // island joins: the newly 4-node handle wins the next merge.
        // Unrelated actor admission must not change that sequence or
        // either graph's retained source island identity.
        {
            auto populate=[](Scene& scene) {
                return std::array<fyzz::BodyId,7>{
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{0,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{0.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{4,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{4.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8.3f,42,0},1),
                    scene.add_dynamic_box({0.5f,0.5f,0.5f},{8.6f,42,0},1)};
            };
            Scene incremental, admitted;
            const auto ci=populate(incremental);
            const auto ai=populate(admitted);
            incremental.advance_collision();
            admitted.advance_collision();
            const auto survivor=incremental.accurate_island_id(ci[2]);
            const auto speculative_survivor=
                incremental.speculative_island_id(ci[2]);
            const auto largest_original=incremental.accurate_island_id(ci[4]);
            expect(incremental.accurate_island_count()==3 &&
                   incremental.speculative_island_count()==3 &&
                   survivor!=largest_original &&
                   survivor==incremental.accurate_island_id(ci[3]) &&
                   incremental.accurate_island_id(ci[0])!=survivor,
                   "Accumulated source merge fixture lacks three old islands");
            const auto newcomer=admitted.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,42,0},1);
            const std::array<float,7> joined_x{
                0.0f,0.3f,1.0f,1.3f,2.0f,2.3f,2.6f};
            for(std::size_t i=0;i<ci.size();++i) {
                incremental.set_position(ci[i],{joined_x[i],42,0});
                admitted.set_position(ai[i],{joined_x[i],42,0});
            }
            incremental.advance_collision();
            admitted.advance_collision();
            expect(incremental.accurate_island_count()==1 &&
                   incremental.speculative_island_count()==1 &&
                   admitted.accurate_island_count()==2 &&
                   admitted.speculative_island_count()==2 &&
                   incremental.accurate_island_id(ci[0])==survivor &&
                   incremental.speculative_island_id(ci[0])==
                       speculative_survivor &&
                   admitted.accurate_island_id(newcomer)!=survivor,
                   "Incremental source did not retain accumulated merge winner");
            for(std::size_t i=0;i<ci.size();++i)
                expect(admitted.accurate_island_id(ai[i])==
                           incremental.accurate_island_id(ci[i]) &&
                       admitted.speculative_island_id(ai[i])==
                           incremental.speculative_island_id(ci[i]),
                       "Actor admission changed source accumulated merge identity");
        }

        // Found edges are processed before the third-pass lost-edge
        // split, including when independent node admission forces the
        // full source island-membership rebuild. A-B first forms the
        // two-node island rooted at B. In one publication A-B is lost,
        // B-C is found, and an unrelated node is admitted in one scene.
        // Both execution paths must preserve B's island handle for B-C.
        {
            auto populate=[](Scene& scene) {
                const auto a=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,48,0},1.0f);
                const auto b=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.4f,48,0},1.0f);
                const auto c=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{3,48,0},1.0f);
                scene.advance_collision();
                return std::array{a,b,c};
            };
            Scene normal, admitted;
            const auto n=populate(normal);
            const auto a=populate(admitted);
            const auto root_accurate=normal.accurate_island_id(n[1]);
            const auto root_speculative=normal.speculative_island_id(n[1]);
            expect(normal.accurate_island_id(n[0])==root_accurate &&
                   normal.speculative_island_id(n[0])==root_speculative &&
                   normal.accurate_island_id(n[2])!=root_accurate,
                   "Source mixed-loss fixture lacks the initial A-B island");
            const auto unrelated=admitted.add_dynamic_box(
                {0.5f,0.5f,0.5f},{30,48,0},1.0f);
            for(auto* scene:{&normal,&admitted}) {
                const auto& ids=scene==&normal?n:a;
                scene->set_position(ids[0],{-3,48,0});
                scene->set_position(ids[2],{0.8f,48,0});
                scene->advance_collision();
                expect(scene->lost_pairs().size()==1 &&
                       scene->found_pairs().size()==1 &&
                       scene->lost_contacts().size()==1 &&
                       scene->found_contacts().size()==1,
                       "Mixed first-pass lost/found fixture did not swap contacts");
            }
            expect(normal.accurate_island_count()==2 &&
                   normal.speculative_island_count()==2 &&
                   admitted.accurate_island_count()==3 &&
                   admitted.speculative_island_count()==3 &&
                   normal.accurate_island_id(n[1])==root_accurate &&
                   normal.accurate_island_id(n[2])==root_accurate &&
                   normal.speculative_island_id(n[1])==root_speculative &&
                   normal.speculative_island_id(n[2])==root_speculative &&
                   admitted.accurate_island_id(a[1])==root_accurate &&
                   admitted.accurate_island_id(a[2])==root_accurate &&
                   admitted.speculative_island_id(a[1])==root_speculative &&
                   admitted.speculative_island_id(a[2])==root_speculative &&
                   admitted.accurate_island_id(a[0])!=root_accurate &&
                   admitted.speculative_island_id(a[0])!=root_speculative &&
                   admitted.accurate_island_id(unrelated)!=root_accurate,
                   "Actor admission changed first-pass merge and lost-edge split identity");
        }

        // A failed step must restore pre-existing source route hints,
        // not merely topology and diagnostics. Prime both IslandSim
        // caches with a redundant chord removal, reinsert the chords,
        // then trigger repeated solver failures on the next removal.
        // Redundant chords can shorten source fast routes upon
        // reintroduction; the retried scene must match the clean
        // reference's hints, counters, and motion instead of an
        // obsolete absolute fast-hit count.
        {
            auto populate=[](Scene& scene) {
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.0f,22,0},1.0f);
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.2f,22,0},1.0f);
                scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.4f,22,0},1.0f);
                const auto tip=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.6f,22,0},1.0f);
                const auto floor=scene.add_static_floor(0);
                const auto load=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
                scene.advance_collision();
                return std::tuple{tip,floor,load};
            };
            Scene rejected, clean;
            const auto [rt,rf,rl]=populate(rejected);
            const auto [ct,cf,cl]=populate(clean);
            for(auto* scene:{&rejected,&clean}) {
                const auto tip=scene==&rejected?rt:ct;
                scene->set_position(tip,{1.3f,22,0});
                scene->advance_collision();
                expect(scene->island_dirty_route_check_count()==6 &&
                       scene->island_fast_route_hit_count()==0 &&
                       scene->accurate_island_count()==2 &&
                       scene->speculative_island_count()==2,
                       "Fast-route rollback fixture did not prime both graphs");
                scene->set_position(tip,{0.6f,22,0});
                scene->advance_collision();
                // Six box/box candidates plus the source plane's
                // five dynamic-body candidates; only the load touches
                // that plane, giving seven physical contact managers.
                expect(scene->pairs().size()==11 &&
                       scene->contacts().size()==7 &&
                       scene->accurate_island_count()==2 &&
                       scene->speculative_island_count()==2,
                       "Fast-route rollback fixture did not restore redundant chords");
                scene->set_position(tip,{1.3f,22,0});
            }
            const auto checks=rejected.island_dirty_route_check_count();
            const auto hits=rejected.island_fast_route_hit_count();
            const auto revision=rejected.island_topology_revision();
            const auto frame=rejected.frame_index();
            expect(checks==clean.island_dirty_route_check_count() &&
                   hits==clean.island_fast_route_hit_count(),
                   "Primed route histories disagree before rollback test");
            auto overflowing=fyzz::Material{};
            overflowing.static_friction=1.0e30f;
            overflowing.dynamic_friction=1.0e30f;
            overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
            rejected.set_material(rf,overflowing);
            rejected.set_material(rl,overflowing);
            for(int attempt=0;attempt<2;++attempt) {
                bool threw=false;
                try {rejected.step(1.0f/120.0f);}
                catch(const std::overflow_error&) {threw=true;}
                expect(threw &&
                       rejected.frame_index()==frame &&
                       rejected.island_topology_revision()==revision &&
                       rejected.island_dirty_route_check_count()==checks &&
                       rejected.island_fast_route_hit_count()==hits,
                       "Failed PGS publication committed island route diagnostics");
            }
            rejected.set_material(rf,fyzz::Material{});
            rejected.set_material(rl,fyzz::Material{});
            rejected.step(1.0f/120.0f);
            clean.step(1.0f/120.0f);
            expect(rejected.island_dirty_route_check_count()==checks+6 &&
                   rejected.island_dirty_route_check_count()==
                       clean.island_dirty_route_check_count() &&
                   rejected.island_fast_route_hit_count()==hits+2 &&
                   rejected.island_fast_route_hit_count()==
                       clean.island_fast_route_hit_count() &&
                   rejected.accurate_island_count()==clean.accurate_island_count() &&
                   rejected.speculative_island_count()==clean.speculative_island_count() &&
                   rejected.frame_index()==clean.frame_index() &&
                   rejected.body_state(rt).position.x==
                       clean.body_state(ct).position.x &&
                   rejected.body_state(rl).velocity.y==
                       clean.body_state(cl).velocity.y,
                   "Failed island route repair changed subsequent source retry");
        }

        // A failed solver frame after genuine *local* island splitting
        // must roll back handle/node-chain migration, indexed managers,
        // route hints, split diagnostics and publication. Retrying must
        // then perform exactly the same split as a clean production scene.
        {
            auto populate=[](Scene& scene) {
                const auto a=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,20,0},1);
                const auto b=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0.85f,20,0},1);
                const auto c=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{1.7f,20,0},1);
                const auto floor=scene.add_static_floor(0);
                const auto load=scene.add_dynamic_box(
                    {0.5f,0.5f,0.5f},{0,0.49f,0},1);
                scene.advance_collision();
                expect(scene.accurate_island_count()==2 &&
                       scene.speculative_island_count()==2 &&
                       has(scene.found_contacts(),a,b) &&
                       has(scene.found_contacts(),b,c),
                       "Local split rollback fixture did not form the source chain");
                scene.set_position(b,{5,20,0});
                return std::tuple{a,b,c,floor,load};
            };
            Scene rejected,clean;
            const auto [ra,rb,rc,rf,rl]=populate(rejected);
            const auto [ca,cb,cc,cf,cl]=populate(clean);
            const auto before_frame=rejected.frame_index();
            const auto before_revision=rejected.island_topology_revision();
            const auto before_local=rejected.island_local_split_count();
            const auto before_global=rejected.island_full_rebuild_count();
            const auto before_graph_checkpoint=
                rejected.island_graph_checkpoint_count();
            const auto original_root=rejected.accurate_island_id(rb);
            auto overflowing=fyzz::Material{};
            overflowing.static_friction=1.0e30f;
            overflowing.dynamic_friction=1.0e30f;
            overflowing.friction_combine=fyzz::MaterialCombineMode::Multiply;
            rejected.set_material(rf,overflowing);
            rejected.set_material(rl,overflowing);
            for(int attempt=0;attempt<2;++attempt) {
                bool threw=false;
                try {rejected.step(1.0f/120.0f);}
                catch(const std::overflow_error&) {threw=true;}
                expect(threw &&
                       rejected.frame_index()==before_frame &&
                       rejected.island_topology_revision()==before_revision &&
                       rejected.island_local_split_count()==before_local &&
                       rejected.island_full_rebuild_count()==before_global &&
                       rejected.island_graph_checkpoint_count()==
                           before_graph_checkpoint &&
                       rejected.accurate_island_id(rb)==original_root &&
                       rejected.accurate_island_count()==2 &&
                       rejected.contacts().size()==clean.contacts().size(),
                       "Failed local split published partial island migration");
            }
            rejected.set_material(rf,fyzz::Material{});
            rejected.set_material(rl,fyzz::Material{});
            rejected.step(1.0f/120.0f);
            clean.step(1.0f/120.0f);
            expect(rejected.island_full_rebuild_count()==before_global &&
                   rejected.island_graph_checkpoint_count()>
                       before_graph_checkpoint &&
                   rejected.island_graph_checkpoint_count()==
                       clean.island_graph_checkpoint_count() &&
                   rejected.island_local_split_count()>before_local &&
                   rejected.island_local_split_count()==
                       clean.island_local_split_count() &&
                   rejected.accurate_island_count()==4 &&
                   rejected.speculative_island_count()==4 &&
                   rejected.accurate_island_id(rb)==original_root &&
                   rejected.accurate_island_id(ra)!=original_root &&
                   rejected.accurate_island_id(rc)!=original_root &&
                   rejected.accurate_island_id(ra)==
                       clean.accurate_island_id(ca) &&
                   rejected.accurate_island_id(rc)==
                       clean.accurate_island_id(cc) &&
                   rejected.frame_index()==clean.frame_index() &&
                   rejected.body_state(rl).velocity.y==
                       clean.body_state(cl).velocity.y,
                   "Local PhysX split rollback did not reproduce clean migration");
        }

        // Two nearly parallel diagonal thin boxes may overlap in their
        // rotated SAP AABBs without touching in Gu SAT. This discriminates
        // the speculative source graph from the accurate touch graph;
        // replacing both with Gu manager connectivity fails here.
        Scene candidate_only{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto rod_a=candidate_only.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,20,0},1.0f);
        const auto rod_b=candidate_only.add_dynamic_box(
            {2.0f,0.1f,0.2f},{0,25,0},1.0f);
        candidate_only.set_velocity(rod_a,{0,9.81f,0});
        candidate_only.set_velocity(rod_b,{0,9.81f,0});
        candidate_only.set_angular_velocity(rod_a,{0,0,0.83f});
        candidate_only.set_angular_velocity(rod_b,{0,0,0.83f});
        candidate_only.step(1.0f);
        candidate_only.set_angular_velocity(rod_a,{0,0,0});
        candidate_only.set_angular_velocity(rod_b,{0,0,0});
        const auto candidate_rebuilds=candidate_only.island_full_rebuild_count();
        candidate_only.set_position(rod_a,{0,20,0});
        candidate_only.set_position(rod_b,{0,22,0});
        candidate_only.advance_collision();
        expect(has(candidate_only.pairs(),rod_a,rod_b) &&
               candidate_only.contacts().empty() &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==2 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds,
               "Speculative island insertion rebuilt or conflated accurate topology");
        candidate_only.set_position(rod_b,{0,20.1f,0});
        candidate_only.advance_collision();
        expect(candidate_only.contacts().size()==1 &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==1 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds,
               "Accurate island insertion rebuilt or missed a touching edge");
        candidate_only.set_position(rod_b,{0,22,0});
        candidate_only.advance_collision();
        expect(candidate_only.contacts().empty() &&
               candidate_only.speculative_island_count()==1 &&
               candidate_only.accurate_island_count()==2 &&
               candidate_only.island_full_rebuild_count()==candidate_rebuilds &&
               candidate_only.island_local_split_count()>0,
               "Accurate island removal did not repair separated components");

        // Three coupled dynamic bodies generate two competing contact
        // managers, not a single order-insensitive pair. Check both the
        // contiguous Dy island-node batch and the finite-iteration PGS
        // impulses against a scene which began with the same topology.
        Scene joined, initially_joined;
        const auto j0=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto j1=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{4,5,0},1.0f);
        const auto j2=joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{8,5,0},1.0f);
        const auto b0=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto b1=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,5,0},1.0f);
        const auto b2=initially_joined.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.6f,5,0},1.0f);
        joined.advance_collision();
        expect(joined.accurate_island_count()==3 &&
               joined.largest_accurate_island_size()==1,
               "Disjoint source nodes were not initially isolated");
        const auto before_join=joined.island_full_rebuild_count();
        joined.set_position(j1,{0.8f,5,0});
        joined.set_position(j2,{1.6f,5,0});
        joined.step(1.0f/120.0f);
        initially_joined.step(1.0f/120.0f);
        expect(joined.accurate_island_count()==1 &&
               joined.largest_accurate_island_size()==3 &&
               joined.island_full_rebuild_count()==before_join &&
               joined.contacts().size()==2 &&
               joined.solver_partition_count()>=2 &&
               joined.solved_normal_impulses().size()==
                    initially_joined.solved_normal_impulses().size(),
               "Incremental merge did not form a coupled accurate PGS island");
        // Both managers must contribute positive forces. Merely solving
        // the first contact would leave the outer body unconstrained.
        for (const auto& manager:joined.contacts()) {
            double impulse=0.0;
            for (std::size_t i=0;i<manager.count;++i)
                impulse+=joined.solved_normal_impulses()[
                    manager.contact_offset+i];
            expect(impulse>0.0,
                   "Coupled incremental island omitted a source contact solve");
        }
        const auto p=joined.solved_normal_impulses();
        const auto q=initially_joined.solved_normal_impulses();
        for(std::size_t i=0;i<p.size();++i)
            expect(std::abs(p[i]-q[i])<0.0001,
                   "Incremental island changed coupled PGS force ordering");
        for (const auto [incremental,initial] :
             {std::pair{j0,b0},std::pair{j1,b1},std::pair{j2,b2}}) {
            const auto lhs=joined.body_state(incremental);
            const auto rhs=initially_joined.body_state(initial);
            expect(std::abs(lhs.velocity.x-rhs.velocity.x)<0.0001f &&
                   std::abs(lhs.velocity.y-rhs.velocity.y)<0.0001f &&
                   std::abs(lhs.position.x-rhs.position.x)<0.0001f &&
                   std::abs(lhs.position.y-rhs.position.y)<0.0001f,
                   "Incremental accurate-island merge changed coupled PGS motion");
        }

        // IslandSim::Node::mStaticTouchCount and
        // mIslandStaticTouchCount: source SAP candidates against an
        // infinite plane remain speculative edges even when Gu reports
        // no touch. Dynamic-island merges accumulate static contacts;
        // lost dynamic edges split them back into individual counters.
        // None of these static-only changes should fabricate an island
        // topology revision.
        {
            Scene static_counts{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto ground=static_counts.add_static_floor(0);
            const auto a=static_counts.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1);
            const auto b=static_counts.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,0.5f,0},1);
            static_counts.advance_collision();
            expect(static_counts.speculative_island_count()==2 &&
                   static_counts.accurate_island_count()==2 &&
                   static_counts.speculative_island_static_touches(a)==1 &&
                   static_counts.speculative_island_static_touches(b)==1 &&
                   static_counts.accurate_island_static_touches(a)==1 &&
                   static_counts.accurate_island_static_touches(b)==1,
                   "PhysX static touches not initialized in both island graphs");
            const auto stable_revision=static_counts.island_topology_revision();
            static_counts.set_position(b,{4,3,0});
            static_counts.advance_collision();
            expect(static_counts.island_topology_revision()==stable_revision &&
                   static_counts.speculative_island_static_touches(b)==1 &&
                   static_counts.accurate_island_static_touches(b)==0 &&
                   static_counts.accurate_island_static_touches(a)==1,
                   "Gu touch loss changed SAP static counts or island identity");
            static_counts.set_position(b,{4,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.island_topology_revision()==stable_revision &&
                   static_counts.accurate_island_static_touches(b)==1,
                   "Gu touch reentry did not restore static-touch ownership");
            static_counts.set_position(b,{0.6f,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.speculative_island_count()==1 &&
                   static_counts.accurate_island_count()==1 &&
                   static_counts.speculative_island_static_touches(a)==2 &&
                   static_counts.accurate_island_static_touches(a)==2 &&
                   static_counts.accurate_island_static_touches(b)==2,
                   "Dynamic island merge lost source static-touch counts");
            static_counts.set_position(b,{4,0.5f,0});
            static_counts.advance_collision();
            expect(static_counts.accurate_island_count()==2 &&
                   static_counts.speculative_island_count()==2 &&
                   static_counts.accurate_island_static_touches(a)==1 &&
                   static_counts.accurate_island_static_touches(b)==1 &&
                   static_counts.speculative_island_static_touches(a)==1,
                   "Lost-edge split duplicated or lost static-touch counts");
            static_counts.remove(ground);
            static_counts.advance_collision();
            expect(static_counts.accurate_island_static_touches(a)==0 &&
                   static_counts.accurate_island_static_touches(b)==0 &&
                   static_counts.speculative_island_static_touches(a)==0 &&
                   static_counts.speculative_island_static_touches(b)==0,
                   "Retired static actor retained ghost island touches");
        }

        Scene static_world{Scene::ContactGenerator::DiagnosticLegacyGu};
        static_world.add_static_floor(0.0f);
        const auto ground_box=static_world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.5f,0.0f},1.0f);
        static_world.add_dynamic_box({0.5f,0.5f,0.5f},
                                     {4.0f,0.5f,0.0f},1.0f);
        static_world.advance_collision();
        expect(static_world.contacts().size()==2 &&
               static_world.speculative_island_count()==2 &&
               static_world.accurate_island_count()==2,
               "Shared static world incorrectly merged independent dynamic islands");
        const auto static_rev=static_world.island_topology_revision();
        const auto static_rebuilds=static_world.island_full_rebuild_count();
        const auto fixture=static_world.add_static_box(
            {1.0f,1.0f,1.0f},{20.0f,20.0f,20.0f});
        static_world.advance_collision();
        expect(static_world.island_topology_revision()==static_rev &&
               static_world.accurate_island_count()==2,
               "Static actor creation rebuilt dynamic island graph");
        static_world.remove(fixture);
        static_world.advance_collision();
        expect(static_world.island_topology_revision()==static_rev &&
               static_world.accurate_island_count()==2,
               "Static actor retirement rebuilt dynamic island graph");
        static_world.set_position(ground_box,{0,5,0});
        static_world.advance_collision();
        expect(static_world.contacts().size()==1 &&
               static_world.island_topology_revision()==static_rev,
               "Static-only contact loss unnecessarily rebuilt dynamic islands");
        static_world.set_position(ground_box,{0,0.5f,0});
        static_world.advance_collision();
        expect(static_world.contacts().size()==2 &&
               static_world.island_topology_revision()==static_rev &&
               static_world.island_full_rebuild_count()==static_rebuilds,
               "Static-only contact reentry unnecessarily rebuilt dynamic islands");
        static_world.step(1.0f/120.0f);
        expect(static_world.accurate_island_count()==2 &&
               static_world.solver_partition_count()==1,
               "Static world island split changed conflict-free solver partitioning");
    }

    // The same scene must distinguish SAP candidate overlaps from actual
    // shape contacts, and preserve touch found/lost across quiet frames.
    {
        Scene contact_scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto distant_floor = contact_scene.add_static_floor(0.0f);
        const auto falling = contact_scene.add_dynamic_box(
            {1.0f,1.0f,1.0f}, {0,20,0}, 1.0f);
        contact_scene.advance_collision();
        expect(has(contact_scene.found_pairs(), distant_floor, falling),
               "Unbounded plane did not enter broadphase");
        expect(contact_scene.contacts().empty() &&
               contact_scene.found_contacts().empty(),
               "Far plane candidate was misreported as a contact");
        contact_scene.set_position(falling, {0,1.02f,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 4 &&
               has(contact_scene.found_contacts(), distant_floor, falling) &&
               contact_scene.lost_contacts().empty(),
               "PhysX plane/box near-contact vertex manifold absent");
        const auto& above = contact_scene.contacts()[0];
        expect(std::abs(above.points[0].separation - 0.02f) < 0.0001f &&
               above.points[0].normal.y == 1.0f,
               "Plane contact separation or normal incorrect");
        // GuContactPlaneBox enumerates +x,-y,+z before the other
        // three near-plane corners. Preserve source contact ordering
        // because contact force buffers are indexed per point.
        expect(std::abs(above.points[0].position.x - 1.0f) < 0.0001f &&
               std::abs(above.points[0].position.y - 0.02f) < 0.0001f &&
               std::abs(above.points[0].position.z - 1.0f) < 0.0001f,
               "GuContactPlaneBox source corner order or position diverged");
        contact_scene.advance_collision(true);
        expect(contact_scene.found_contacts().empty() &&
               contact_scene.lost_contacts().empty(),
               "Forced unchanged narrowphase published duplicate touch");
        contact_scene.set_position(falling, {0,0.8f,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 4 &&
               contact_scene.contacts()[0].points[0].separation < -0.19f &&
               contact_scene.found_contacts().empty(),
               "Plane penetration did not refresh contact manifold");
        contact_scene.set_position(falling, {0,20,0});
        contact_scene.advance_collision();
        expect(has(contact_scene.lost_contacts(), distant_floor, falling) &&
               contact_scene.lost_pairs().empty(),
               "Physical touch should be lost while plane BP candidate survives");
        contact_scene.advance_collision();
        expect(contact_scene.lost_contacts().empty(),
               "Quiet noncontact frame emitted repeated lost touch");
        contact_scene.set_position(falling, {0,-3,0});
        contact_scene.advance_collision();
        expect(contact_scene.contacts().size() == 1 &&
               contact_scene.contacts()[0].count == 8,
               "Plane penetration must retain all eight source vertices");
        contact_scene.remove(falling);
        contact_scene.advance_collision();
        expect(has(contact_scene.lost_contacts(), distant_floor, falling),
               "Actor retirement failed to report lost physical touch");
    }
    {
        // The ordered pair must not determine which way gravity would
        // be opposed: creating the box before the floor reverses IDs
        // but not the physical plane orientation.
        Scene reversed{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto box = reversed.add_dynamic_box(
            {1,1,1}, {0,1.02f,0}, 1);
        const auto floor = reversed.add_static_floor(0);
        reversed.advance_collision();
        expect(has(reversed.found_pairs(), box, floor) &&
               has(reversed.found_contacts(), box, floor) &&
               reversed.contacts().size() == 1 &&
               reversed.contacts()[0].count == 4,
               "Box-before-floor contact stream absent");
        for (unsigned i = 0; i < reversed.contacts()[0].count; ++i) {
            const auto& point = reversed.contacts()[0].points[i];
            expect(point.normal.x == 0.0f && point.normal.y == 1.0f &&
                   point.normal.z == 0.0f &&
                   std::abs(point.separation - 0.02f) < 0.0001f,
                   "Box-before-floor contact normal or separation incorrect");
        }
        reversed.set_position(box, {0,20,0});
        reversed.advance_collision();
        expect(has(reversed.lost_contacts(), box, floor) &&
               reversed.lost_pairs().empty(),
               "Box-before-floor touch loss emitted wrong events");
    }
    {
        Scene box_scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto fixed = box_scene.add_static_box(
            {1,1,1},{0,0,0});
        const auto moveable = box_scene.add_dynamic_box(
            {1,1,1},{1.5f,0,0},1);
        box_scene.advance_collision();
        expect(box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].count == 4 &&
               has(box_scene.found_contacts(), fixed, moveable),
               "Axis-aligned box contact manifold absent");
        const auto& manifold = box_scene.contacts()[0];
        expect(manifold.points[0].normal.x == 1.0f &&
               std::abs(manifold.points[0].separation + 0.5f) < 0.0001f,
               "Source-ordered box contact normal or signed penetration incorrect");
        box_scene.set_position(moveable, {2.03f,0,0});
        box_scene.advance_collision();
        expect(box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].points[0].separation > 0.029f,
               "Box proximity contacts must preserve positive separation");
        box_scene.set_position(moveable, {2.10f,0,0});
        box_scene.advance_collision();
        expect(has(box_scene.lost_contacts(), fixed, moveable) &&
               box_scene.contacts().empty(),
               "Box separation did not retire touch history");
        box_scene.set_position(moveable, {1.5f,0,0});
        box_scene.advance_collision();
        expect(has(box_scene.found_contacts(), fixed, moveable),
               "Box re-entry did not recreate narrowphase touch");

        // The original GuContactBoxBox face expansion retains near-corner
        // contacts inside per-axis contactDistance, even for diagonal gaps.
        // Preserve its generated point stream instead of the retired
        // face-only implementation's rejection of those candidates.
        box_scene.set_position(moveable, {2.03f,2.03f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count>0 &&
               box_scene.lost_contacts().empty(),
               "Source near-corner contact generation was not retained");
        const auto old_count=box_scene.contacts()[0].count;
        box_scene.advance_collision(true);
        expect(box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count==old_count &&
               box_scene.found_contacts().empty() &&
               box_scene.lost_contacts().empty(),
               "Forced near-corner update changed source touch history");
        box_scene.set_position(moveable, {2.01f,2.01f,0});
        box_scene.advance_collision();
        expect(has(box_scene.pairs(), fixed, moveable) &&
               box_scene.contacts().size()==1 &&
               box_scene.contacts()[0].count>0 &&
               box_scene.found_contacts().empty(),
               "Original Gu near-edge contact generation was lost");

        // A single-axis near contact with positive separation is still
        // supported when the two transverse face intervals overlap.
        box_scene.set_position(moveable, {1.5f,2.03f,0});
        box_scene.advance_collision();
        expect(box_scene.found_contacts().empty() &&
               box_scene.lost_contacts().empty() &&
               box_scene.contacts().size() == 1 &&
               box_scene.contacts()[0].count > 0 &&
               box_scene.contacts()[0].points[0].normal.y == 1.0f &&
               box_scene.contacts()[0].points[0].separation > 0.029f,
               "Source single-axis proximity or persistent-touch semantics drifted");
    }



    {
        // Direct production-path translation of the original PhysX
        // DyBodyCoreIntegrator closed-form quaternion step, limited to
        // an isolated free body until source angular contact constraints
        // and the original PGS execution are integrated.
        Scene angular{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto body = angular.add_dynamic_box(
            {0.25f,0.5f,0.75f},{0,10,0},1);
        angular.set_angular_velocity(body,{0,2,0});
        for (int i=0; i<120; ++i) angular.step(1.0f/120.0f);
        const auto state = angular.body_state(body);
        const float quat_norm = std::sqrt(
            state.orientation.x*state.orientation.x +
            state.orientation.y*state.orientation.y +
            state.orientation.z*state.orientation.z +
            state.orientation.w*state.orientation.w);
        expect(std::isfinite(quat_norm) &&
               std::abs(quat_norm-1.0f)<1.0e-4f &&
               std::abs(state.orientation.y)>0.5f &&
               state.angular_velocity.y>1.8f &&
               state.angular_velocity.y<2.0f,
               "Source quaternion integration/damping did not advance");
        bool invalid=false;
        try {angular.set_angular_velocity(body,{0,INFINITY,0});}
        catch(const std::invalid_argument&) {invalid=true;}
        expect(invalid,"Nonfinite native angular velocity was admitted");
        auto floor=angular.add_static_floor(0);
        bool static_rejected=false;
        try {angular.set_angular_velocity(floor,{0,1,0});}
        catch(const std::logic_error&) {static_rejected=true;}
        expect(static_rejected,"Static actor accepted angular velocity");
        // The source normal-contact rows include rotational Jacobians,
        // so a rotated actor may advance alongside static geometry.
        angular.step(1.0f/120.0f);
        angular.set_angular_velocity(body,{0,0,0});
        angular.step(1.0f/120.0f);
        // Collision-only Gu narrowphase remains independently usable.
        angular.advance_collision(true);
        expect(has(angular.pairs(),body,floor) &&
               angular.contacts().empty(),
               "Rotated distant floor candidate must not invent a contact");
    }

    // Per-point Dy rigid-body normals must exert angular response at an
    // offset contact; the retired one-impulse-per-manifold solver could
    // not produce contact torque at any rigid-body geometry.
    {
        Scene angular_contact{Scene::ContactGenerator::DiagnosticLegacyGu};
        angular_contact.add_static_box({1,1,1},{0,0,0});
        const auto body=angular_contact.add_dynamic_box(
            {0.5f,0.5f,0.5f},{1.3f,1.25f,0},1.0f);
        angular_contact.set_velocity(body,{-2,0,0});
        angular_contact.step(1.0f/120.0f);
        const auto result=angular_contact.body_state(body);
        expect(!angular_contact.contacts().empty() &&
               std::isfinite(result.angular_velocity.z) &&
               std::abs(result.angular_velocity.z)>1.0e-5f,
               "Per-point source contact angular Jacobian produced no torque");
    }

    // Original Gu box-box SAT/contact path must accept an oblique
    // body directly in the owning scene, not only the free-flight path.
    {
        Scene oblique{Scene::ContactGenerator::DiagnosticLegacyGu};
        auto moving=oblique.add_dynamic_box({0.5f,0.5f,0.5f},
                                             {0,5,0},1.0f);
        oblique.set_angular_velocity(moving,{0,0,2.0f});
        oblique.step(0.4f);
        oblique.set_angular_velocity(moving,{0,0,0});
        oblique.set_position(moving,{1.35f,0,0});
        auto fixed=oblique.add_static_box({1,1,1},{0,0,0});
        oblique.advance_collision();
        expect(has(oblique.found_pairs(),moving,fixed) &&
               has(oblique.found_contacts(),moving,fixed) &&
               oblique.contacts().size()==1 &&
               oblique.contacts()[0].count>0 &&
               oblique.contacts()[0].collision_axis>0,
               "Rotated Gu box contact generation or pair axis missing");
        for(const auto& pt:oblique.contacts()[0].points) {
            const float length=std::sqrt(pt.normal.x*pt.normal.x+
                 pt.normal.y*pt.normal.y+pt.normal.z*pt.normal.z);
            expect(std::isfinite(pt.separation) && std::abs(length-1.0f)<0.002f,
                   "Rotated box contact normal/separation invalid");
        }
        const auto axis=oblique.contacts()[0].collision_axis;
        oblique.advance_collision(true);
        expect(oblique.contacts().size()==1 &&
               oblique.contacts()[0].collision_axis==axis &&
               oblique.found_contacts().empty() &&
               oblique.lost_contacts().empty(),
               "Gu box-box collision axis cache was not retained");
        // Per-point angular contact constraints must now admit the
        // oblique geometry without the retired translation-only guard.
        oblique.step(1.0f/120.0f);
        expect(std::isfinite(oblique.body_state(moving).angular_velocity.z),
               "Rotated Dy normal-contact solver produced invalid angular motion");
    }
    // A pitched box has more than an AABB corner representation. The
    // source plane contact stream must carry its real rotated vertices.
    {
        Scene pitched{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto moving=pitched.add_dynamic_box(
            {0.4f,0.7f,0.3f},{0,5,0},1);
        pitched.set_angular_velocity(moving,{0,0,2.0f});
        pitched.step(0.4f);
        pitched.set_angular_velocity(moving,{0,0,0});
        pitched.set_position(moving,{0,0.7f,0});
        const auto floor=pitched.add_static_floor(0);
        pitched.advance_collision();
        expect(has(pitched.found_pairs(),moving,floor) &&
               pitched.contacts().size()==1 &&
               pitched.contacts()[0].count>0,
               "Rotated plane-box source contact stream absent");
        bool displaced=false;
        for(const auto& pt:pitched.contacts()[0].points) {
            if(std::abs(std::abs(pt.position.x)-0.4f)>0.05f)
                displaced=true;
            expect(pt.normal.y==1.0f && std::isfinite(pt.separation),
                   "Rotated plane-box normal or separation invalid");
        }
        expect(displaced,"Rotated plane-box contacts remained axis-aligned");
    }

    // The scene-owned source solver-body pool prepares only true contact
    // participants. An unrelated free-flight body must keep its original
    // unconstrained motion through repeated solver rounds and retirement.
    {
        Scene sparse{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor = sparse.add_static_floor(0.0f);
        const auto contact = sparse.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto free = sparse.add_dynamic_box(
            {0.25f,0.25f,0.25f},{8,12,0},1.0f);
        sparse.set_velocity(free,{3.0f,0.0f,0.0f});
        for (unsigned i=0;i<100;++i)
            sparse.step(1.0f/120.0f);
        const auto body = sparse.body_state(free);
        expect(std::abs(body.velocity.x-3.0f)<0.0001f &&
               std::abs(body.position.x-10.5f)<0.001f,
               "Contact solver body state modified unrelated free flight");
        for (const auto& manifold : sparse.contacts())
            expect(manifold.pair.a!=free && manifold.pair.b!=free,
                   "Distant free-flight body admitted to source contact solve");
        sparse.remove(contact);
        sparse.step(1.0f/120.0f);
        expect(std::abs(sparse.body_state(free).velocity.x-3.0f)<0.0001f &&
               sparse.contacts().empty() &&
               sparse.frame_index()==101,
               "Retiring a contact participant leaked stale solver-body state");
        (void)floor;
    }

    // Two static contact managers must reference the same zero-mass world
    // solver body, without duplicating or accumulating static velocity.
    {
        Scene supports{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto left = supports.add_static_box(
            {0.5f,0.5f,1.0f},{-0.5f,-0.5f,0});
        const auto right = supports.add_static_box(
            {0.5f,0.5f,1.0f},{0.5f,-0.5f,0});
        const auto load = supports.add_dynamic_box(
            {0.75f,0.5f,0.75f},{0,0.5f,0},1.0f);
        supports.set_velocity(load,{0,-1.0f,0});
        supports.step(1.0f/120.0f);
        const auto v = supports.body_state(load).velocity;
        expect(supports.contacts().size()==2 &&
               std::isfinite(v.y) && v.y>-1.1f &&
               supports.body_state(left).velocity.y==0.0f &&
               supports.body_state(right).velocity.y==0.0f,
               "Shared world solver body failed multi-static contact");
    }

    // DyFrictionCorrelation retains the floor anchors while contact
    // moves, and the two source tangent constraints must dissipate
    // sliding motion with the reference material (0.5/0.5 friction).
    // A normal-only solver preserves almost all horizontal velocity.
    {
        Scene sliding{Scene::ContactGenerator::DiagnosticLegacyGu};
        sliding.add_static_floor(0);
        const auto box=sliding.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        sliding.set_velocity(box,{2.0f,0,0});
        for (unsigned frame=0;frame<180;++frame)
            sliding.step(1.0f/120.0f);
        const auto state=sliding.body_state(box);
        expect(std::isfinite(state.velocity.x) &&
               std::abs(state.velocity.x)<0.9f &&
               std::isfinite(state.position.y) &&
               state.position.y>0.34f && state.position.y<0.68f &&
               sliding.contacts().size()==1,
               "Correlated PhysX friction anchors did not resist floor sliding");
    }

    // DyCpuGpuBiasCoefficient PGS rigidContact=0.8 is shared by
    // penetration bias and friction-row relaxation. A shallow symmetric
    // overlap must receive source-scale positional correction, rather
    // than the earlier unrelated 0.2 normal-bias approximation.
    {
        Scene correction{Scene::ContactGenerator::DiagnosticLegacyGu};
        correction.add_static_floor(0.0f);
        const auto body=correction.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.48f,0},1.0f);
        correction.step(1.0f/120.0f);
        const auto state=correction.body_state(body);
        expect(std::isfinite(state.position.y) &&
               state.position.y>0.488f &&
               state.position.y<0.55f,
               "Source 0.8 PGS rigid-contact penetration bias was not applied");
    }
    // When the Coulomb limit is not reached, source friction relaxation
    // removes tangential slip more strongly than the old 0.2 multiplier.
    // Use a low sliding speed so the limit does not dominate this check.
    {
        Scene relaxation{Scene::ContactGenerator::DiagnosticLegacyGu};
        relaxation.add_static_floor(0.0f);
        const auto body=relaxation.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        relaxation.set_velocity(body,{0.02f,0,0});
        relaxation.step(1.0f/120.0f);
        const auto state=relaxation.body_state(body);
        expect(std::isfinite(state.velocity.x) &&
               std::abs(state.velocity.x)<0.012f,
               "Source friction relaxation left excessive low-speed slip");
    }

    // DyConstraintPartition's rigid-body 32-bit partition mask must
    // roll into a second wave when one dynamic body has >32 independent
    // dynamic/dynamic contacts. None of the leaf bodies touch each other;
    // the center is the one genuine per-partition conflict.
    {
        Scene crowded{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto center = crowded.add_dynamic_box(
            {2.0f,2.0f,2.0f},{0,5,0},1.0f);
        std::vector<fyzz::BodyId> leaves;
        leaves.reserve(35);
        for (unsigned y=0;y<5;++y) {
            for (unsigned z=0;z<7;++z) {
                leaves.push_back(crowded.add_dynamic_box(
                    {0.1f,0.1f,0.1f},
                    {2.08f,3.6f+0.7f*y,-1.5f+0.5f*z},1.0f));
            }
        }
        crowded.advance_collision();
        expect(crowded.contacts().size()==35,
               "Dense source contact-manager star did not create 35 dynamic edges");
        crowded.step(1.0f/120.0f);
        expect(crowded.solver_partition_count()==35,
               "Source 32-bit partition rollover collapsed conflicting dynamic contacts");
        const auto center_state = crowded.body_state(center);
        expect(std::isfinite(center_state.position.x) &&
               std::isfinite(center_state.velocity.x) &&
               crowded.contacts().size()==35,
               "Source 32-partition overflow did not execute all contact managers");
        // Dy writeback indexes the original Gu manifold stream even
        // though the solver traverses reordered constraint partitions.
        // All 35 manager reports must have a positive final impulse,
        // including three managers assigned to the overflow wave.
        const auto reported=crowded.solved_normal_impulses();
        std::size_t expected_offset=0;
        for (std::size_t i=0;i<crowded.contacts().size();++i) {
            const auto& manager=crowded.contacts()[i];
            expect(manager.contact_offset==expected_offset &&
                   manager.contact_offset+manager.count<=reported.size(),
                   "Partitioned Dy force writeback lost Gu contact offsets");
            double normal_sum=0.0;
            for (std::size_t k=0;k<manager.count;++k)
                normal_sum+=reported[manager.contact_offset+k];
            expect(std::isfinite(normal_sum) && normal_sum>0.0,
                   i>=32
                       ? "Overflow-wave manager has no solved contact force writeback"
                       : "First-wave manager has no solved contact force writeback");
            expected_offset+=manager.count;
        }
        expect(expected_offset==reported.size(),
               "Dy force writeback omitted original Gu contact points");
        // Constructing 35 partitions is insufficient proof of execution:
        // these boxes start with no X velocity and gravity has no X
        // component. Every active center/leaf contact must therefore
        // push its leaf outward along +X on the *first* solve. In
        // particular leaves 32-34 are assigned to the second 32-bit
        // overflow wave; dropping their ordered descriptors must fail.
        for (std::size_t index=0;index<leaves.size();++index) {
            const auto state=crowded.body_state(leaves[index]);
            expect(std::isfinite(state.position.x) &&
                   std::isfinite(state.velocity.x) &&
                   state.velocity.x>0.01f && state.position.x>2.08f,
                   index>=32
                       ? "Source second-wave contact produced no outward impulse"
                       : "Source first-wave contact produced no outward impulse");
        }
        // A new frame must recompute partition masks without retaining
        // previous wave assignments or introducing another body owner.
        crowded.step(1.0f/120.0f);
        expect(crowded.solver_partition_count()==35 &&
               std::isfinite(crowded.body_state(center).position.x),
               "Source constraint partition did not reset on next frame");
    }

    verify_island_contact_manager_order();
    verify_split_contact_manager_migration();
    verify_dirty_seed_split_manager_order();
    verify_hop_priority_split_migration();
    verify_equal_hop_split_queue_order();
    verify_source_dirty_witness_split();
    verify_successful_route_branch_witness();
    verify_route_repair_continues_after_split();
    verify_dirty_order_multisplit_manager_remainder();

    // The source zero-mass world solver body does not create a dynamic
    // conflict between distinct static-floor contacts; all may share
    // partition zero, even though they reference one world solver row.
    {
        Scene detached{Scene::ContactGenerator::DiagnosticLegacyGu};
        detached.add_static_floor(0.0f);
        for (unsigned i=0;i<12;++i)
            detached.add_dynamic_box({0.25f,0.25f,0.25f},
                                     {float(i)*2.0f,0.25f,0},1.0f);
        detached.step(1.0f/120.0f);
        expect(detached.contacts().size()==12 &&
               detached.solver_partition_count()==1,
               "Static world solver row incorrectly serialized unrelated contacts");
    }

    // DySolverConstraints.cpp::writeBackContact must preserve Gu contact
    // offsets even when DyConstraintPartition changes constraint order.
    // Two disjoint floor contacts with different masses independently
    // verify final per-point normal impulse sums against velocity change.
    {
        constexpr float dt=1.0f/120.0f;
        Scene reported{Scene::ContactGenerator::DiagnosticLegacyGu};
        reported.add_static_floor(0.0f);
        const auto light=reported.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.0f,0.5f,0.0f},1.0f);
        const auto heavy=reported.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3.0f,0.5f,0.0f},2.0f);
        reported.advance_collision(true);
        expect(reported.contacts().size()==2 &&
               reported.solved_normal_impulses().empty(),
               "Collision-only Gu query exposed stale Dy force writeback");
        reported.step(dt);
        const auto manifolds=reported.contacts();
        const auto impulses=reported.solved_normal_impulses();
        expect(manifolds.size()==2 && impulses.size()==8,
               "Dy writeback did not preserve the flat Gu contact layout");
        double sums[2]{};
        for (std::size_t m=0;m<manifolds.size();++m) {
            const auto& manifold=manifolds[m];
            expect(manifold.count==4 &&
                   manifold.contact_offset+manifold.count<=impulses.size(),
                   "Dy contact output does not match original Gu offsets");
            for (std::size_t i=0;i<manifold.count;++i) {
                const double impulse=impulses[manifold.contact_offset+i];
                expect(std::isfinite(impulse) && impulse>=0.0,
                       "Dy reported an invalid final contact normal impulse");
                sums[m]+=impulse;
            }
        }
        const auto l=reported.body_state(light);
        const auto h=reported.body_state(heavy);
        // For unit-half boxes, mass is density (volume = 1).
        // Source solver normal impulse equals mass times vertical
        // velocity delta from pre-solver gravity, not a force in newtons.
        const double expected_light=l.velocity.y+9.81*dt;
        const double expected_heavy=2.0*(h.velocity.y+9.81*dt);
        expect(sums[0]>0.02 && sums[1]>sums[0]*1.5 &&
               std::abs(sums[0]-expected_light)<0.001 &&
               std::abs(sums[1]-expected_heavy)<0.001,
               "Source contact writeback does not reconcile solved body velocity");
        reported.advance_collision(true);
        expect(reported.solved_normal_impulses().empty(),
               "Geometry-only advance retained obsolete solver force results");
        reported.step(dt);
        expect(reported.solved_normal_impulses().size()==
                   reported.contacts()[0].count+
                   reported.contacts()[1].count,
               "Following Dy step did not republish fresh contact impulses");
        reported.set_position(light,{0.0f,5.0f,0.0f});
        reported.set_position(heavy,{3.0f,5.0f,0.0f});
        reported.advance_collision(true);
        expect(reported.contacts().empty() &&
               reported.solved_normal_impulses().empty(),
               "Lost contact retained previous normal force writeback");
    }

    // Genuine DyConstraintPartition permutation:
    // Gu order is (A,B), (A,C), (D,E), while two conflicting A edges
    // occupy partitions 0,1 and independent D/E falls back to 0.
    // Dy execution is therefore (A,B), (D,E), (A,C). Detect any
    // writeback accidentally appended in solver order by checking each
    // unique leaf's momentum against its original Gu manager offset.
    {
        Scene reordered{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto a=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {0,5,0},1.0f);
        const auto b=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {0.80f,5,0},1.0f);
        const auto c=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {-0.78f,5,0},2.0f);
        const auto d=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {4,5,0},3.0f);
        const auto e=reordered.add_dynamic_box({0.5f,0.5f,0.5f},
                                               {4.92f,5,0},4.0f);
        reordered.step(1.0f/120.0f);
        const auto contacts=reordered.contacts();
        const auto impulses=reordered.solved_normal_impulses();
        expect(contacts.size()==3 &&
               reordered.solver_partition_count()==2 &&
               contacts[0].pair==fyzz::native::Pair{a,b} &&
               contacts[1].pair==fyzz::native::Pair{a,c} &&
               contacts[2].pair==fyzz::native::Pair{d,e},
               "Native permutation regression did not construct [0,1,0] partitions");
        auto sum_for=[&](std::size_t m) {
            double sum=0;
            const auto& manager=contacts[m];
            expect(manager.contact_offset+manager.count<=impulses.size(),
                   "Reordered Dy force buffer does not match Gu offsets");
            for(std::size_t k=0;k<manager.count;++k)
                sum+=impulses[manager.contact_offset+k];
            return sum;
        };
        const double ib=sum_for(0),ic=sum_for(1),ie=sum_for(2);
        const double vb=std::abs(reordered.body_state(b).velocity.x);
        const double vc=std::abs(reordered.body_state(c).velocity.x);
        const double ve=std::abs(reordered.body_state(e).velocity.x);
        expect(ib>0 && ic>0 && ie>0 &&
               std::abs(ib-vb)<0.003 &&
               std::abs(ic-2.0*vc)<0.003 &&
               std::abs(ie-4.0*ve)<0.003 &&
               std::abs(ic-ie)>0.003,
               "Dy writeback followed partition order instead of Gu contact order");
    }

    // DyContactPrep.cpp::writeBackContactFriction reconstructs each
    // retained anchor impulse as tangent0*lambda0 + tangent1*lambda1.
    // Compare the aggregate world X impulse with each independent body's
    // tangential velocity change, and verify per-manager patch offsets.
    {
        Scene friction_report{Scene::ContactGenerator::DiagnosticLegacyGu};
        friction_report.add_static_floor(0.0f);
        const auto left=friction_report.add_dynamic_box(
            {0.5f,0.5f,0.5f},{-5.0f,0.5f,0.0f},1.0f);
        const auto right=friction_report.add_dynamic_box(
            {0.5f,0.5f,0.5f},{5.0f,0.5f,0.0f},2.0f);
        friction_report.set_velocity(left,{0.015f,0,0});
        friction_report.set_velocity(right,{-0.025f,0,0});
        friction_report.step(1.0f/120.0f);
        const auto contacts=friction_report.contacts();
        const auto patches=friction_report.solved_friction_patches();
        expect(contacts.size()==2 && patches.size()==2 &&
               contacts[0].friction_patch_offset==0 &&
               contacts[1].friction_patch_offset==1 &&
               contacts[0].friction_patch_count==1 &&
               contacts[1].friction_patch_count==1,
               "Source friction writeback did not retain Gu patch order");
        const auto left_after=friction_report.body_state(left);
        const auto right_after=friction_report.body_state(right);
        for (std::size_t m=0;m<2;++m) {
            const auto& patch=patches[m];
            expect(patch.pair==contacts[m].pair &&
                   patch.anchor_count>0 && patch.anchor_count<=2,
                   "Source friction writeback lost per-patch anchors");
            double total_x=0.0;
            for(std::size_t k=0;k<patch.anchor_count;++k) {
                const auto& impulse=patch.anchor_impulses[k];
                const auto& position=patch.anchor_positions[k];
                expect(std::isfinite(impulse.x) &&
                       std::isfinite(impulse.y) &&
                       std::isfinite(impulse.z) &&
                       std::isfinite(position.x) &&
                       std::isfinite(position.y) &&
                       std::abs(position.x-(m==0 ? -5.0 : 5.0))<0.6 &&
                       std::abs(position.y)<0.1 &&
                       std::abs(impulse.y)<0.001,
                       "Source friction impulse or body0 anchor position mismatch");
                total_x+=impulse.x;
            }
            const double velocity_before=m==0 ? 0.015 : -0.025;
            const double velocity_after=m==0
                ? left_after.velocity.x : right_after.velocity.x;
            const double mass=m==0 ? 1.0 : 2.0;
            // Source Dy applies the reported tangent impulse to
            // Sc body0. For plane/box, Sc body0 is always the dynamic
            // actor, independent of the canonical floor-first Pair.
            expect(std::abs(total_x-
                            mass*(velocity_after-velocity_before))<0.002 &&
                   (m==0 ? total_x<0.0 : total_x>0.0),
                   "Source friction anchor writeback disagrees with body impulse");
        }
        friction_report.advance_collision(true);
        expect(friction_report.solved_normal_impulses().empty() &&
               friction_report.solved_friction_patches().empty(),
               "Geometry-only collision advance left stale friction impulses");
        friction_report.step(1.0f/120.0f);
        expect(friction_report.solved_friction_patches().size()==2,
               "Source friction writeback did not republish after collision query");
    }

    // A failed *geometry-only* advance must preserve the exact backing
    // allocations of the previously published Gu pair/contact and
    // solved force streams. A finite static box can overflow its AABB
    // during source broadphase preparation without any solver step.
    {
        Scene geometry{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto floor=geometry.add_static_floor(0.0f);
        const auto box=geometry.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
        geometry.step(1.0f/120.0f);
        const auto existing=fyzz::native::Pair{floor,box};
        const auto old_edge=geometry.contact_edge_index(existing);
        const auto* const old_pairs=geometry.pairs().data();
        const auto* const old_contacts=geometry.contacts().data();
        const auto* const old_points=geometry.contacts()[0].points.data();
        const auto* const old_forces=geometry.solved_normal_impulses().data();
        const auto old_frame=geometry.frame_index();
        const auto old_impulse=geometry.solved_normal_impulses()[0];
        const auto giant=geometry.add_static_box(
            {std::numeric_limits<float>::max()*0.5f,1,1},
            {std::numeric_limits<float>::max()*0.75f,0,0});
        bool rejected=false;
        try {geometry.advance_collision();}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               geometry.frame_index()==old_frame &&
               geometry.pairs().data()==old_pairs &&
               geometry.contacts().data()==old_contacts &&
               geometry.contacts()[0].points.data()==old_points &&
               geometry.solved_normal_impulses().data()==old_forces &&
               geometry.solved_normal_impulses()[0]==old_impulse &&
               geometry.contact_edge_index(existing)==old_edge &&
               geometry.contact_edge_connected(existing),
               "Failed geometry publication invalidated previous source spans");
        geometry.remove(giant);
        geometry.advance_collision();
        expect(geometry.contacts().size()==1 &&
               geometry.contact_edge_index(existing)==old_edge,
               "Rejected geometry advance prevented subsequent publication");
    }

    // Quiet failed solver preparation must also restore Gu manifold
    // descriptor metadata, not merely the contact-point span and pair
    // event streams. Put the overflowing contact manager second so
    // preparation has already advanced the solver offset of its row.
    {
        Scene rejected, clean;
        auto construct=[](Scene& s) {
            const auto floor=s.add_static_floor(0.0f);
            const auto first=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            const auto second=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{5,0.49f,0},1.0f);
            s.advance_collision();
            return std::tuple{floor,first,second};
        };
        const auto [floor_bad,first_bad,second_bad]=construct(rejected);
        const auto [floor_good,first_good,second_good]=construct(clean);
        expect(rejected.contacts().size()==2 &&
               clean.contacts().size()==2,
               "Quiet manifold rollback fixture requires two Gu managers");
        const auto* const published_ptr=rejected.contacts().data();
        const auto* const points0=rejected.contacts()[0].points.data();
        const auto* const points1=rejected.contacts()[1].points.data();
        const auto original0=rejected.contacts()[0];
        const auto original1=rejected.contacts()[1];
        const auto frame=rejected.frame_index();
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(floor_bad,exploding);
        rejected.set_material(second_bad,exploding);
        bool failed=false;
        try {rejected.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {failed=true;}
        const auto after=rejected.contacts();
        expect(failed && rejected.frame_index()==frame &&
               after.size()==2 && after.data()==published_ptr &&
               after[0].points.data()==points0 &&
               after[1].points.data()==points1 &&
               after[0].solver_offset==original0.solver_offset &&
               after[1].solver_offset==original1.solver_offset &&
               after[0].friction_patch_offset==original0.friction_patch_offset &&
               after[1].friction_patch_offset==original1.friction_patch_offset &&
               after[0].friction_patch_count==original0.friction_patch_count &&
               after[1].friction_patch_count==original1.friction_patch_count,
               "Rejected quiet preparation leaked partial Gu/Dy descriptors");
        rejected.set_material(floor_bad,fyzz::Material{});
        rejected.set_material(second_bad,fyzz::Material{});
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        const auto bad=rejected.body_state(second_bad);
        const auto good=clean.body_state(second_good);
        expect(bad.position.y==good.position.y &&
               bad.velocity.y==good.velocity.y &&
               rejected.contacts().size()==clean.contacts().size(),
               "Quiet manifold rollback changed valid retry dynamics");
        (void)first_bad;
        (void)first_good;
    }

    // Rejected solver scratch must not be admitted again when the
    // checkpoint rolls back the frame clock and the author changes the
    // geometry before retry. Both scenes start with the same Gu contact;
    // only one attempts an overflowing material preparation. On retry,
    // their boxes are teleported out of contact and must follow the
    // identical unconstrained free-flight path.
    {
        constexpr float dt=1.0f/120.0f;
        Scene rejected, clean;
        auto create=[](Scene& s) {
            const auto floor=s.add_static_floor(0.0f);
            const auto box=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            s.advance_collision();
            return std::pair{floor,box};
        };
        const auto [floor_bad,body_bad]=create(rejected);
        const auto [floor_good,body_good]=create(clean);
        auto material=fyzz::Material{};
        material.static_friction=1.0e30f;
        material.dynamic_friction=1.0e30f;
        material.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(floor_bad,material);
        rejected.set_material(body_bad,material);
        bool failed=false;
        const auto old_frame=rejected.frame_index();
        try {rejected.step(dt);}
        catch(const std::overflow_error&) {failed=true;}
        expect(failed && rejected.frame_index()==old_frame,
               "Solver generation fixture failed after source row admission");
        rejected.set_material(floor_bad,fyzz::Material{});
        rejected.set_material(body_bad,fyzz::Material{});
        rejected.set_position(body_bad,{0,5,0});
        clean.set_position(body_good,{0,5,0});
        rejected.step(dt);
        clean.step(dt);
        const auto a=rejected.body_state(body_bad);
        const auto b=clean.body_state(body_good);
        expect(rejected.contacts().empty() && clean.contacts().empty() &&
               a.position.x==b.position.x &&
               a.position.y==b.position.y &&
               a.position.z==b.position.z &&
               a.velocity.x==b.velocity.x &&
               a.velocity.y==b.velocity.y &&
               a.velocity.z==b.velocity.z &&
               rejected.solved_normal_impulses().empty() &&
               rejected.frame_index()==clean.frame_index(),
               "Rejected solver row leaked into changed-contact retry");
    }

    // A quiet failed step must not erase Found/Lost events from a
    // preceding successful geometry-only publication. Only event streams
    // rotate on that path: contact managers, their point spans, and pair
    // buffers retain their original allocations and publication epochs.
    // Exercise both found and lost events, and two failed retries.
    for(const bool expect_lost:{false,true}) {
        Scene rejected, clean;
        auto setup=[&](Scene& s) {
            // A bounded static box allows both the SAP overlap and the
            // Gu contact to be lost. An infinite floor remains a SAP
            // candidate after the dynamic body moves away vertically.
            const auto ground=s.add_static_box(
                {2.0f,0.5f,2.0f},{0,-0.5f,0});
            const auto box=s.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            s.advance_collision();
            if(expect_lost) {
                s.set_position(box,{0,10.0f,0});
                s.advance_collision();
            }
            return std::pair{ground,box};
        };
        const auto [floor_a,box_a]=setup(rejected);
        const auto [floor_b,box_b]=setup(clean);
        const auto found_pairs=rejected.found_pairs();
        const auto lost_pairs=rejected.lost_pairs();
        const auto found_contacts=rejected.found_contacts();
        const auto lost_contacts=rejected.lost_contacts();
        expect((expect_lost
                    ? has(lost_pairs,floor_a,box_a) &&
                      has(lost_contacts,floor_a,box_a)
                    : has(found_pairs,floor_a,box_a) &&
                      has(found_contacts,floor_a,box_a)) &&
               rejected.body_state(box_a).position.y==
                   clean.body_state(box_b).position.y,
               "Quiet-event rollback fixture failed to publish events");
        const std::vector<fyzz::native::Pair> prior_found(
            found_pairs.begin(),found_pairs.end());
        const std::vector<fyzz::native::Pair> prior_lost(
            lost_pairs.begin(),lost_pairs.end());
        const std::vector<fyzz::native::Pair> prior_found_contacts(
            found_contacts.begin(),found_contacts.end());
        const std::vector<fyzz::native::Pair> prior_lost_contacts(
            lost_contacts.begin(),lost_contacts.end());
        const auto* const found_ptr=found_pairs.data();
        const auto* const lost_ptr=lost_pairs.data();
        const auto* const touch_ptr=found_contacts.data();
        const auto* const lost_touch_ptr=lost_contacts.data();
        const auto* const pairs_ptr=rejected.pairs().data();
        const auto* const manifolds_ptr=rejected.contacts().data();
        const auto frame=rejected.frame_index();
        const auto topology=rejected.island_topology_revision();
        const auto original=rejected.body_state(box_a);
        for(int attempt=0;attempt<2;++attempt) {
            bool threw=false;
            try {rejected.step(std::numeric_limits<float>::max());}
            catch(const std::overflow_error&) {threw=true;}
            const auto body=rejected.body_state(box_a);
            expect(threw &&
                   rejected.frame_index()==frame &&
                   rejected.island_topology_revision()==topology &&
                   rejected.pairs().data()==pairs_ptr &&
                   rejected.contacts().data()==manifolds_ptr &&
                   rejected.found_pairs().data()==found_ptr &&
                   rejected.lost_pairs().data()==lost_ptr &&
                   rejected.found_contacts().data()==touch_ptr &&
                   rejected.lost_contacts().data()==lost_touch_ptr &&
                   body.position.x==original.position.x &&
                   body.position.y==original.position.y &&
                   body.velocity.y==original.velocity.y,
                   "Rejected quiet step altered previous scene publication");
            const auto equal=[](auto span,const auto& expected) {
                return span.size()==expected.size() &&
                    std::equal(span.begin(),span.end(),expected.begin());
            };
            expect(equal(rejected.found_pairs(),prior_found) &&
                   equal(rejected.lost_pairs(),prior_lost) &&
                   equal(rejected.found_contacts(),prior_found_contacts) &&
                   equal(rejected.lost_contacts(),prior_lost_contacts),
                   "Rejected quiet step erased prior Found/Lost events");
        }
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        const auto recovered=rejected.body_state(box_a);
        const auto baseline=clean.body_state(box_b);
        expect(rejected.found_pairs().empty() &&
               rejected.lost_pairs().empty() &&
               rejected.found_contacts().empty() &&
               rejected.lost_contacts().empty() &&
               recovered.position.y==baseline.position.y &&
               recovered.velocity.y==baseline.velocity.y &&
               rejected.frame_index()==clean.frame_index(),
               "Quiet event rollback changed successful retry state");
        (void)floor_b;
    }

    // A failing Dy step must not publish its unconstrained velocities,
    // consume pending torque, advance topology/frame history, or replace
    // contact/force/threshold reports. Retrying with a valid material must
    // reproduce an identical scene which never attempted the failed step.
    // Material coefficients are individually finite at authoring, while
    // the source MULTIPLY combination can still overflow at contact prep.
    {
        constexpr float dt=1.0f/120.0f;
        Scene failed, clean;
        auto construct=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto box=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            scene.set_contact_report_threshold(box,0.1f);
            scene.step(1.0f/120.0f);
            scene.apply_torque_impulse(box,{0,0,0.1f});
            return std::pair{floor,box};
        };
        const auto [bad_floor,bad_box]=construct(failed);
        const auto [good_floor,good_box]=construct(clean);
        const auto prior=failed.body_state(bad_box);
        const auto frame=failed.frame_index();
        const auto topology=failed.island_topology_revision();
        const auto count=failed.accurate_island_count();
        const auto partitions=failed.solver_partition_count();
        const std::vector<double> forces(
            failed.solved_normal_impulses().begin(),
            failed.solved_normal_impulses().end());
        const auto prior_manifolds=failed.contacts();
        const auto* const manifold_memory=prior_manifolds.data();
        const auto* const pair_memory=failed.pairs().data();
        const auto* const found_memory=failed.found_pairs().data();
        const auto* const touch_memory=failed.found_contacts().data();
        const auto* const force_memory=failed.solved_normal_impulses().data();
        const auto* const threshold_memory=failed.force_threshold_changes().data();
        expect(prior_manifolds.size()==1 &&
               prior_manifolds[0].count==4,
               "Step failure fixture has no authoritative prior contacts");
        const auto* const point_memory=prior_manifolds[0].points.data();
        const auto first_contact=prior_manifolds[0].points[0];
        const std::vector<fyzz::native::ForceThresholdReport> threshold(
            failed.force_threshold_changes().begin(),
            failed.force_threshold_changes().end());
        const auto original_pair=prior_manifolds[0].pair;
        const auto original_edge=failed.contact_edge_index(original_pair);
        expect(original_edge!=std::numeric_limits<std::uint32_t>::max() &&
               failed.contact_edge_connected(original_pair),
               "Step rollback fixture did not establish an accurate edge");
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        failed.set_material(bad_floor,exploding);
        failed.set_material(bad_box,exploding);
        for (int attempt=0;attempt<2;++attempt) {
            bool rejected=false;
            try {failed.step(dt);}
            catch(const std::overflow_error&) {rejected=true;}
            expect(rejected,
                   "Combined material overflow did not fail source preparation");
            const auto after=failed.body_state(bad_box);
            const auto same_vec=[](fyzz::Vec3 a,fyzz::Vec3 b) {
                return a.x==b.x && a.y==b.y && a.z==b.z;
            };
            const auto same_quat=[](fyzz::Quat a,fyzz::Quat b) {
                return a.x==b.x && a.y==b.y &&
                       a.z==b.z && a.w==b.w;
            };
            expect(same_vec(after.position,prior.position) &&
                   same_vec(after.velocity,prior.velocity) &&
                   same_quat(after.orientation,prior.orientation) &&
                   same_vec(after.angular_velocity,prior.angular_velocity) &&
                   failed.frame_index()==frame &&
                   failed.island_topology_revision()==topology &&
                   failed.accurate_island_count()==count &&
                   failed.solver_partition_count()==partitions &&
                   failed.contact_edge_index(original_pair)==original_edge &&
                   failed.contact_edge_connected(original_pair) &&
                   failed.active_contact_edge_count()==1,
                   "Rejected Dy step published partial body or edge state");
            const auto current=failed.contacts();
            expect(current.size()==1 &&
                   current.data()==manifold_memory &&
                   current[0].points.data()==point_memory &&
                   failed.pairs().data()==pair_memory &&
                   failed.found_pairs().data()==found_memory &&
                   failed.found_contacts().data()==touch_memory &&
                   failed.solved_normal_impulses().data()==force_memory &&
                   failed.force_threshold_changes().data()==threshold_memory &&
                   current[0].pair==original_pair &&
                   current[0].count==4 &&
                   same_vec(current[0].points[0].position,
                            first_contact.position) &&
                   same_vec(current[0].points[0].normal,
                            first_contact.normal) &&
                   current[0].points[0].separation==first_contact.separation,
                   "Rejected step invalidated previous authoritative Gu contacts");
            const auto reported=failed.solved_normal_impulses();
            expect(reported.size()==forces.size() &&
                   std::equal(reported.begin(),reported.end(),forces.begin()),
                   "Rejected step changed previous published force writeback");
            const auto changes=failed.force_threshold_changes();
            expect(changes.size()==threshold.size(),
                   "Rejected step changed previous force-threshold events");
            for(std::size_t i=0;i<changes.size();++i)
                expect(changes[i].key==threshold[i].key &&
                       changes[i].normal_impulse==threshold[i].normal_impulse &&
                       changes[i].change==threshold[i].change,
                       "Rejected step modified previous threshold payload");
        }
        const fyzz::Material ordinary{};
        failed.set_material(bad_floor,ordinary);
        failed.set_material(bad_box,ordinary);
        clean.set_material(good_floor,ordinary);
        clean.set_material(good_box,ordinary);
        failed.step(dt);
        clean.step(dt);
        const auto a=failed.body_state(bad_box);
        const auto b=clean.body_state(good_box);
        const auto near=[](float x,float y) {
            return std::abs(x-y)<2.0e-5f;
        };
        expect(near(a.position.x,b.position.x) &&
               near(a.position.y,b.position.y) &&
               near(a.position.z,b.position.z) &&
               near(a.velocity.x,b.velocity.x) &&
               near(a.velocity.y,b.velocity.y) &&
               near(a.velocity.z,b.velocity.z) &&
               near(a.orientation.x,b.orientation.x) &&
               near(a.orientation.y,b.orientation.y) &&
               near(a.orientation.z,b.orientation.z) &&
               near(a.orientation.w,b.orientation.w) &&
               near(a.angular_velocity.z,b.angular_velocity.z) &&
               std::abs(a.angular_velocity.z)>1.0e-4f &&
               failed.frame_index()==clean.frame_index() &&
               failed.contact_edge_index(original_pair)==original_edge,
               "Failed step consumed an impulse or changed edge identity");
    }

    // The source EdgeInstance checkpoint is sparse. Stable contact
    // managers must not copy the historical handle watermark on every
    // simulation step merely to guarantee exception rollback.
    {
        Scene stable{Scene::ContactGenerator::DiagnosticLegacyGu};
        stable.add_static_floor(0.0f);
        for(int i=0;i<16;++i)
            stable.add_dynamic_box({0.5f,0.5f,0.5f},
                {static_cast<float>(i)*3.0f,0.49f,0},1.0f);
        stable.step(1.0f/120.0f);
        expect(stable.active_contact_edge_count()==16,
               "Sparse edge checkpoint fixture lacks persistent managers");
        stable.step(1.0f/120.0f);
        expect(stable.active_contact_edge_count()==16 &&
               stable.edge_rollback_journal_size()==0,
               "Unchanged source contact edges were copied during PGS");
    }

    // An exception *after* an existing accurate edge is disconnected
    // must restore both endpoint list links and the connected bit. A
    // second independent contact with invalid combined material forces
    // the failure in contact preparation, after Gu island publication.
    {
        auto initialize=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto first=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            const auto second=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{3,0.49f,0},1.0f);
            scene.step(1.0f/120.0f);
            return std::tuple{floor,first,second};
        };
        Scene rejected, clean;
        const auto [rf,ra,rb]=initialize(rejected);
        const auto [cf,ca,cb]=initialize(clean);
        const fyzz::native::Pair ra_pair{rf,ra}, rb_pair{rf,rb};
        const auto id=rejected.contact_edge_index(ra_pair);
        const auto* const old_contacts=rejected.contacts().data();
        const auto* const old_points=rejected.contacts()[0].points.data();
        const auto old_frame=rejected.frame_index();
        rejected.set_position(ra,{0,8,0});
        clean.set_position(ca,{0,8,0});
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        rejected.set_material(rf,exploding);
        rejected.set_material(rb,exploding);
        bool threw=false;
        try {rejected.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {threw=true;}
        expect(threw &&
               rejected.edge_rollback_journal_size()>0 &&
               rejected.frame_index()==old_frame &&
               rejected.contacts().data()==old_contacts &&
               rejected.contacts()[0].points.data()==old_points &&
               rejected.contact_edge_index(ra_pair)==id &&
               rejected.contact_edge_connected(ra_pair) &&
               rejected.contact_edge_connected(rb_pair),
               "Failed PGS step did not restore intrusive contact links");
        rejected.set_material(rf,fyzz::Material{});
        rejected.set_material(rb,fyzz::Material{});
        rejected.step(1.0f/120.0f);
        clean.step(1.0f/120.0f);
        expect(rejected.contact_edge_index(ra_pair)==id &&
               !rejected.contact_edge_connected(ra_pair) &&
               rejected.contact_edge_connected(rb_pair) &&
               rejected.contact_edge_connected(ra_pair)==
                   clean.contact_edge_connected({cf,ca}) &&
               rejected.contact_edge_connected(rb_pair)==
                   clean.contact_edge_connected({cf,cb}) &&
               rejected.accurate_island_count()==clean.accurate_island_count(),
               "Sparse intrusive rollback contaminated successful retry");
    }

    // A failed solver step can happen after SAP creates a new contact
    // edge. Rollback must not consume a handle or leave its accurate
    // connection bit in the scene. Retrying must produce the same edge
    // identity as a scene that never attempted the unsuccessful step.
    {
        auto make=[](Scene& scene) {
            const auto floor=scene.add_static_floor(0.0f);
            const auto a=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            scene.step(1.0f/120.0f);
            const auto b=scene.add_dynamic_box(
                {0.5f,0.5f,0.5f},{4,0.49f,0},1.0f);
            return std::tuple{floor,a,b};
        };
        Scene rollback_scene, reference_scene;
        const auto [rf,ra,rb]=make(rollback_scene);
        const auto [cf,ca,cb]=make(reference_scene);
        const fyzz::native::Pair old_pair{rf,ra}, new_pair{rf,rb};
        const auto old_handle=rollback_scene.contact_edge_index(old_pair);
        const auto old_count=rollback_scene.active_contact_edge_count();
        const auto invalid=std::numeric_limits<std::uint32_t>::max();
        auto exploding=fyzz::Material{};
        exploding.static_friction=1.0e30f;
        exploding.dynamic_friction=1.0e30f;
        exploding.friction_combine=fyzz::MaterialCombineMode::Multiply;
        for(const auto id:{rf,ra,rb})
            rollback_scene.set_material(id,exploding);
        bool rejected=false;
        try {rollback_scene.step(1.0f/120.0f);}
        catch(const std::overflow_error&) {rejected=true;}
        expect(rejected &&
               rollback_scene.active_contact_edge_count()==old_count &&
               rollback_scene.contact_edge_index(old_pair)==old_handle &&
               rollback_scene.contact_edge_index(new_pair)==invalid &&
               rollback_scene.contact_edge_connected(old_pair),
               "Failed contact preparation leaked a newly allocated EdgeIndex");
        for(const auto id:{rf,ra,rb})
            rollback_scene.set_material(id,fyzz::Material{});
        rollback_scene.step(1.0f/120.0f);
        reference_scene.step(1.0f/120.0f);
        expect(rollback_scene.contact_edge_index(new_pair)==
                   reference_scene.contact_edge_index({cf,cb}) &&
               rollback_scene.contact_edge_connected(new_pair) &&
               rollback_scene.contact_edge_index(old_pair)==
                   reference_scene.contact_edge_index({cf,ca}),
               "Rejected simulation changed subsequent contact-edge identity");
    }

    // Late integration failures also must not expose earlier body updates.
    // The second body's near-FLT_MAX origin plus the finite final motion
    // yields an unrepresentable position after the first body integrated.
    {
        Scene rollback_pose{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto first=rollback_pose.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,10,0},1.0f);
        const auto second=rollback_pose.add_dynamic_box(
            {0.5f,0.5f,0.5f},
            {std::numeric_limits<float>::max(),10,0},1.0f);
        rollback_pose.set_velocity(second,{1.0e18f,0,0});
        const auto before_a=rollback_pose.body_state(first);
        const auto before_b=rollback_pose.body_state(second);
        const auto frame=rollback_pose.frame_index();
        bool threw=false;
        try {rollback_pose.step(1.0e17f);}
        catch(const std::overflow_error&) {threw=true;}
        const auto after_a=rollback_pose.body_state(first);
        const auto after_b=rollback_pose.body_state(second);
        expect(threw &&
               before_a.position.x==after_a.position.x &&
               before_a.position.y==after_a.position.y &&
               before_a.velocity.y==after_a.velocity.y &&
               before_b.position.x==after_b.position.x &&
               before_b.velocity.x==after_b.velocity.x &&
               rollback_pose.frame_index()==frame,
               "Late pose integration error published earlier body changes");
        rollback_pose.set_velocity(second,{0,0,0});
        rollback_pose.step(1.0f/120.0f);
        expect(rollback_pose.frame_index()==frame+1 &&
               rollback_pose.body_state(first).position.y<10,
               "Failed integration prevented subsequent valid simulation");
    }

    // Authored pose edits invalidate the affected Dy friction cache.
    // A tangent-only teleport leaves old anchor separation projected
    // along the normal at zero, so Dy pointsAreClose alone cannot reject
    // it. Compare the next actual solve against an uncached Scene seeded
    // with the same physical state, for both collision participants.
    {
        constexpr float dt=1.0f/120.0f;
        for (const bool relocate_floor : {false,true}) {
            Scene cached{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto floor=cached.add_static_floor(0.0f);
            const auto box=cached.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            cached.step(dt);
            cached.step(dt);
            const auto patches=cached.solved_friction_patches();
            expect(patches.size()==1 &&
                   patches[0].anchor_count>0 && !patches[0].broken,
                   "Teleport test did not establish reusable friction anchors");
            const auto before=cached.body_state(box);
            expect(std::abs(before.orientation.x)<1.0e-5f &&
                   std::abs(before.orientation.z)<1.0e-5f,
                   "Teleport baseline has unexpected angular motion");
            const fyzz::Vec3 shifted_body{
                before.position.x+(relocate_floor?0.0f:0.6f),
                before.position.y,before.position.z};
            if (relocate_floor)
                cached.set_position(floor,{0.6f,0.0f,0.0f});
            else
                cached.set_position(box,shifted_body);

            Scene fresh{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto new_floor=fresh.add_static_floor(0.0f);
            const auto new_box=fresh.add_dynamic_box(
                {0.5f,0.5f,0.5f},shifted_body,1.0f);
            if (relocate_floor)
                fresh.set_position(new_floor,{0.6f,0.0f,0.0f});
            fresh.set_velocity(new_box,before.velocity);
            fresh.set_angular_velocity(new_box,before.angular_velocity);

            cached.step(dt);
            fresh.step(dt);
            const auto a=cached.body_state(box);
            const auto b=fresh.body_state(new_box);
            // The settled prior scene has a small nonidentity orientation
            // not expressible by the box-only fresh-scene API. This bound
            // tolerates its sub-millimeter-per-second contact differences
            // while rejecting teleport-induced friction kicks.
            constexpr float tolerance=2.0e-3f;
            expect(std::abs(a.velocity.x-b.velocity.x)<tolerance &&
                   std::abs(a.velocity.y-b.velocity.y)<tolerance &&
                   std::abs(a.velocity.z-b.velocity.z)<tolerance &&
                   std::abs(a.angular_velocity.x-b.angular_velocity.x)<tolerance &&
                   std::abs(a.angular_velocity.y-b.angular_velocity.y)<tolerance &&
                   std::abs(a.angular_velocity.z-b.angular_velocity.z)<tolerance,
                   "Explicit tangent teleport reused invalid friction anchors");
        }
    }

    // DySolverConstraints.cpp::solveContact(_BStatic) resets its
    // per-call friction break condition and only stores it during the
    // final velocity/writeback solve. With equal default static/dynamic
    // coefficients (0.5), a row that actually clamps on the FINAL pass
    // ends at the exact per-anchor Coulomb limit. Thus a broken patch
    // must have at least one reported anchor at that limit; saturation
    // during an earlier position iteration cannot justify "broken".
    // Exercise persistent anchors through a range of slip magnitudes so
    // false historical breaks also poison the next frame's cache.
    {
        bool observed_break=false, observed_unbroken=false;
        for (const float sliding_speed :
             {0.0f,0.01f,0.04f,0.08f,0.2f,0.6f,2.0f,4.0f}) {
            Scene final_pass{Scene::ContactGenerator::DiagnosticLegacyGu};
            final_pass.add_static_floor(0.0f);
            const auto box=final_pass.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.49f,0},1.0f);
            final_pass.set_velocity(box,{sliding_speed,0,0});
            for (unsigned frame=0;frame<32;++frame) {
                final_pass.step(1.0f/120.0f);
                if (final_pass.contacts().size()!=1 ||
                    final_pass.solved_friction_patches().size()!=1)
                    continue;
                const auto& patch=final_pass.solved_friction_patches()[0];
                if (!patch.anchor_count) continue;
                double normal_sum=0;
                for(const double impulse: final_pass.solved_normal_impulses())
                    normal_sum+=impulse;
                if (normal_sum<=1.0e-6) continue;
                const double anchor_limit=0.5*normal_sum/
                    static_cast<double>(patch.anchor_count);
                double largest_anchor_impulse=0;
                for(std::size_t k=0;k<patch.anchor_count;++k) {
                    const auto& f=patch.anchor_impulses[k];
                    const double magnitude=std::sqrt(
                        f.x*f.x+f.y*f.y+f.z*f.z);
                    expect(std::isfinite(magnitude),
                           "Friction writeback produced nonfinite anchor impulse");
                    largest_anchor_impulse=std::max(
                        largest_anchor_impulse,magnitude);
                }
                if (patch.broken) {
                    observed_break=true;
                    expect(largest_anchor_impulse>=anchor_limit*0.995,
                           "Position-pass friction clamp leaked into final writeback");
                } else {
                    observed_unbroken=true;
                }
            }
        }
        expect(observed_break && observed_unbroken,
               "Friction final-writeback test did not exercise both break outcomes");
    }

    // DySolverConstraints::writeBackContact -> DyThresholdTable ->
    // PxsForceThresholdTask: a per-body threshold uses normal IMPULSE,
    // compared strictly against force*dt, and retains the preceding
    // exceeded-force stream across unsolved collision-only queries.
    // Static contacts share the source world node, while dynamic/dynamic
    // contacts retain their canonical BodyId pair identity.
    {
        using Change=fyzz::native::ForceThresholdChange;
        constexpr float dt=1.0f/120.0f;
        Scene thresholds{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto ground=thresholds.add_static_floor(0);
        const auto monitored=thresholds.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        const auto unmonitored=thresholds.add_dynamic_box(
            {0.5f,0.5f,0.5f},{3,0.5f,0},1.0f);
        bool rejected=false;
        try { thresholds.set_contact_report_threshold(ground,1.0f); }
        catch(const std::logic_error&) {rejected=true;}
        expect(rejected,
               "Static actor admitted dynamic source contact threshold");
        rejected=false;
        try { thresholds.set_contact_report_threshold(monitored,-1.0f); }
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"Negative source contact threshold admitted");
        // A 1 kg supported body produces about 0.082 N*s per frame.
        // Threshold 5 N is exceeded only under PhysX's impulse > 5*dt
        // rule, NOT the erroneous impulse > 5 comparison.
        thresholds.set_contact_report_threshold(monitored,5.0f);
        thresholds.set_contact_report_threshold(unmonitored,30.0f);
        thresholds.step(dt);
        auto changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].change==Change::Found &&
               changes[0].normal_impulse>5.0*dt &&
               changes[0].normal_impulse<5.0 &&
               changes[0].threshold==5.0f,
               "Source force-threshold found stream missing/incorrect");
        const double first_impulse=changes[0].normal_impulse;
        thresholds.advance_collision(true);
        expect(thresholds.force_threshold_changes().empty(),
               "Unsolved Gu collision exposed stale threshold change stream");
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Persists &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].normal_impulse==first_impulse,
               "Source threshold persistence lost across collision-only query");
        thresholds.set_contact_report_threshold(monitored,
            std::numeric_limits<float>::max());
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==fyzz::native::ForceThresholdKey{monitored,0,true} &&
               changes[0].normal_impulse==0.0,
               "Disabling source threshold omitted lost-force report");
        thresholds.step(dt);
        expect(thresholds.force_threshold_changes().empty(),
               "Lost threshold was repeated after source state retirement");
        thresholds.set_contact_report_threshold(monitored,0.0f);
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Found,
               "Re-enabling threshold did not generate a new found report");
        thresholds.set_position(monitored,{0,5,0});
        thresholds.step(dt);
        changes=thresholds.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost,
               "Loss of contact failed to retire exceeded-force pair");
    }

    // DySolverConstraints::writeBackContact uses the MINIMUM of the
    // two dynamic bodies' thresholds for a rigid-body contact manager.
    // Raising the lower threshold beyond the solved pair force must
    // produce LOST without retiring the underlying Gu contact.
    {
        using Change=fyzz::native::ForceThresholdChange;
        Scene pair_threshold{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto first=pair_threshold.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,5,0},1.0f);
        const auto second=pair_threshold.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0.8f,5,0},1.0f);
        pair_threshold.set_velocity(second,{-2,0,0});
        pair_threshold.set_contact_report_threshold(first,1000.0f);
        pair_threshold.set_contact_report_threshold(second,0.0f);
        pair_threshold.step(1.0f/120.0f);
        auto changes=pair_threshold.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==fyzz::native::ForceThresholdKey{first,second,false} &&
               changes[0].change==Change::Found &&
               changes[0].threshold==0.0f,
               "Dy threshold stream did not use minimum dynamic-body threshold");
        pair_threshold.set_contact_report_threshold(second,1000.0f);
        pair_threshold.step(1.0f/120.0f);
        changes=pair_threshold.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==fyzz::native::ForceThresholdKey{first,second,false} &&
               changes[0].normal_impulse==0.0,
               "Dy exceeded stream failed to retire when pair threshold rose");
    }

    // PhysX PxRigidDynamic::getContactReportThreshold and
    // DyThresholdTable group all static contacts through the ONE static
    // world solver node. In this fixture neither support alone exceeds
    // a threshold strictly between the larger support impulse and their
    // SUM; aggregation by individual Gu shape Pair incorrectly emits
    // no Found event (or duplicates if each is above the threshold).
    {
        using Change=fyzz::native::ForceThresholdChange;
        constexpr float dt=1.0f/120.0f;
        auto populate=[](Scene& scene) {
            const auto left=scene.add_static_box(
                {0.5f,0.5f,1.0f},{-0.5f,-0.5f,0});
            const auto right=scene.add_static_box(
                {0.5f,0.5f,1.0f},{0.5f,-0.5f,0});
            const auto load=scene.add_dynamic_box(
                {0.75f,0.5f,0.75f},{0,0.5f,0},1.0f);
            scene.set_velocity(load,{0,-1.0f,0});
            return std::array<fyzz::BodyId,3>{left,right,load};
        };
        Scene baseline{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto ids=populate(baseline);
        baseline.step(dt);
        const auto managers=baseline.contacts();
        const auto impulses=baseline.solved_normal_impulses();
        expect(managers.size()==2 &&
               managers[0].pair==fyzz::native::Pair{ids[0],ids[2]} &&
               managers[1].pair==fyzz::native::Pair{ids[1],ids[2]},
               "Static-world threshold fixture failed to create two supports");
        std::array<double,2> sums{};
        for (std::size_t j=0;j<2;++j) {
            const auto& manager=managers[j];
            expect(manager.contact_offset+manager.count<=impulses.size(),
                   "Static support impulse offsets were invalid");
            for (std::size_t k=0;k<manager.count;++k)
                sums[j]+=impulses[manager.contact_offset+k];
        }
        const double total=sums[0]+sums[1];
        const double largest=std::max(sums[0],sums[1]);
        expect(sums[0]>0.0 && sums[1]>0.0 &&
               total>largest+0.005,
               "Both static supports must contribute normal impulse");
        const float threshold=static_cast<float>(
            (total+largest)/(2.0*dt));
        expect(largest < static_cast<double>(threshold)*dt &&
               static_cast<double>(threshold)*dt < total,
               "Threshold fixture did not separate individual and summed force");
        Scene combined{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto both=populate(combined);
        combined.set_contact_report_threshold(both[2],threshold);
        combined.step(dt);
        auto changes=combined.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{both[2],0,true} &&
               changes[0].change==Change::Found &&
               std::abs(changes[0].normal_impulse-total)<0.001,
               "Static world contacts were not summed before threshold comparison");
        // The shape-partner identity must not survive in the stream:
        // removing both supports causes exactly one Lost event, followed
        // by no further exceeded-world threshold reports.
        combined.remove(both[0]);
        combined.remove(both[1]);
        combined.step(dt);
        changes=combined.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{both[2],0,true} &&
               changes[0].change==Change::Lost &&
               changes[0].normal_impulse==0.0,
               "Shared static-world exceeded pair was not retired once");
        combined.step(dt);
        expect(combined.force_threshold_changes().empty(),
               "Lost static-world threshold repeated after retirement");
        // With zero threshold, both separately qualifying static
        // contacts must still generate ONE aggregated Found report.
        Scene duplicates{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto duplicate_ids=populate(duplicates);
        duplicates.set_contact_report_threshold(duplicate_ids[2],0.0f);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true} &&
               changes[0].change==Change::Found &&
               std::abs(changes[0].normal_impulse-total)<0.001,
               "Both qualifying static supports generated duplicate reports");
        const double prior_impulse=changes[0].normal_impulse;
        // Changing the static shape partner must preserve the ONE world
        // node identity. A per-shape key would incorrectly report Lost
        // for one static pair even while the other support persists.
        duplicates.remove(duplicate_ids[0]);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true} &&
               changes[0].change==Change::Persists &&
               changes[0].normal_impulse==prior_impulse,
               "Changing static support changed source world-node identity");
        duplicates.remove(duplicate_ids[1]);
        duplicates.step(dt);
        changes=duplicates.force_threshold_changes();
        expect(changes.size()==1 &&
               changes[0].change==Change::Lost &&
               changes[0].key==
                   fyzz::native::ForceThresholdKey{duplicate_ids[2],0,true},
               "Final static support removal omitted shared world-node Lost");
    }

    // PxsMaterialCombiner.h: a material belongs to each authored native
    // shape, not to a global/parallel material registry. The source uses
    // the higher-precedence combine mode: MAX beats MIN/AVERAGE. A disabled
    // friction flag on EITHER side wins and suppresses all tangent rows.
    {
        using fyzz::Material;
        using Mode=fyzz::MaterialCombineMode;
        auto friction_run=[](Mode mode, bool disable) {
            Scene sample{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto ground=sample.add_static_floor(0.0f);
            const auto box=sample.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
            Material floor_material{};
            floor_material.static_friction=0.0f;
            floor_material.dynamic_friction=0.0f;
            floor_material.friction_combine=Mode::Average;
            Material box_material{};
            box_material.static_friction=1.0f;
            box_material.dynamic_friction=1.0f;
            box_material.friction_combine=mode;
            box_material.disable_friction=disable;
            sample.set_material(ground,floor_material);
            sample.set_material(box,box_material);
            sample.set_velocity(box,{0.2f,0,0});
            sample.step(1.0f/120.0f);
            double total_tangential_impulse=0.0;
            for(const auto& patch:sample.solved_friction_patches()) {
                for(std::size_t i=0;i<patch.anchor_count;++i)
                    total_tangential_impulse+=
                        std::abs(patch.anchor_impulses[i].x);
            }
            return std::pair{sample.body_state(box).velocity.x,
                             total_tangential_impulse};
        };
        const auto minimum=friction_run(Mode::Minimum,false);
        const auto average=friction_run(Mode::Average,false);
        const auto multiply=friction_run(Mode::Multiply,false);
        const auto maximum=friction_run(Mode::Maximum,false);
        const auto disabled=friction_run(Mode::Maximum,true);
        expect(std::abs(minimum.first-0.2f)<0.002f &&
               minimum.second<1.0e-8 &&
               multiply.second<1.0e-8 &&
               average.second>0.001 &&
               maximum.second>average.second &&
               maximum.first<minimum.first-0.001f,
               "Source contact friction mode precedence was not applied");
        expect(std::abs(disabled.first-0.2f)<0.002f &&
               disabled.second==0.0,
               "PxsMaterial flag eDISABLE_FRICTION did not suppress tangent rows");
        Scene spring_scene{Scene::ContactGenerator::DiagnosticLegacyGu};
        const auto spring_box=spring_scene.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,2,0},1.0f);
        Material compliant{};
        compliant.restitution=-0.1f;
        spring_scene.set_material(spring_box,compliant);
        Material invalid=compliant;
        invalid.damping=-1.0f;
        bool refused=false;
        try {spring_scene.set_material(spring_box,invalid);}
        catch(const std::invalid_argument&) {refused=true;}
        expect(refused,
               "Negative damping was admitted into compliant spring material");
    }
    // For ordinary nonnegative restitution, PxsCombineMaterials'
    // restitution mode uses the higher-precedence combine mode
    // independently of the friction mode.
    {
        using fyzz::Material;
        using Mode=fyzz::MaterialCombineMode;
        auto rebound=[](Mode mode, float initial_height=0.49f) {
            Scene bounce{Scene::ContactGenerator::DiagnosticLegacyGu};
            const auto ground=bounce.add_static_floor(0.0f);
            const auto box=bounce.add_dynamic_box(
                {0.5f,0.5f,0.5f},{0,initial_height,0},1.0f);
            Material floor_material{};
            floor_material.restitution=0.0f;
            floor_material.restitution_combine=Mode::Average;
            Material box_material{};
            box_material.restitution=1.0f;
            box_material.restitution_combine=mode;
            bounce.set_material(ground,floor_material);
            bounce.set_material(box,box_material);
            bounce.set_velocity(box,{0,-4,0});
            bounce.step(1.0f/120.0f);
            const auto state=bounce.body_state(box);
            return std::pair{state.velocity.y,state.position.y};
        };
        const auto low=rebound(Mode::Minimum);
        const auto multiply=rebound(Mode::Multiply);
        const auto average=rebound(Mode::Average);
        const auto high=rebound(Mode::Maximum);
        const auto touching=rebound(Mode::Maximum,0.5f);
        expect(std::isfinite(low.first) && std::isfinite(high.first) &&
               std::isfinite(multiply.first) && std::isfinite(average.first) &&
               std::abs(low.first-multiply.first)<0.02f &&
               average.first>low.first+0.4f &&
               high.first>average.first+0.4f,
               "Source material restitution combine modes did not order rebound");
        // DyContactPrepShared clears scaledBias for a qualifying bounce
        // BEFORE the position-solver motion snapshot is captured. A
        // slightly penetrating body must not acquire an additional upward
        // motion correction compared with the otherwise identical touching
        // contact. Final unbiased velocity alone cannot detect this bug.
        expect(std::isfinite(high.second) && std::isfinite(touching.second) &&
               std::abs((high.second-touching.second)+0.01f)<0.002f,
               "Restitution contact retained penetration bias during position solve");
    }

    // Material authoring must flow through the real public World
    // NativeCpu backend, not just through native::Scene test fixtures.
    {
        fyzz::World world{
            fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu}};
        const auto ground=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material surface{};
        surface.static_friction=0.0f;
        surface.dynamic_friction=0.0f;
        surface.friction_combine=fyzz::MaterialCombineMode::Minimum;
        world.set_body_material(ground,surface);
        world.set_body_material(box,surface);
        world.set_body_velocity(box,{0.3f,0,0});
        world.step(1.0f/120.0f);
        expect(std::abs(world.body_state(box).velocity.x-0.3f)<0.002f,
               "Public World noncompliant material authoring missed native solver");
        surface.restitution=-0.2f;
        world.set_body_material(box,surface);
        surface.damping=-1.0f;
        bool invalid=false;
        try{world.set_body_material(box,surface);}
        catch(const std::invalid_argument&){invalid=true;}
        expect(invalid,
               "Public World accepted negative compliant damping");
    }

#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    // The reference adapter must bind a real PxMaterial to its existing
    // PxShape, preserving the very same public World authoring contract.
    {
        fyzz::World world{
            fyzz::WorldOptions{fyzz::ExecutionMode::ReferencePhysXCpu}};
        const auto ground=world.add_static_floor(0.0f);
        const auto box=world.add_dynamic_box(
            {0.5f,0.5f,0.5f},{0,0.5f,0},1.0f);
        fyzz::Material frictionless{};
        frictionless.static_friction=0.0f;
        frictionless.dynamic_friction=0.0f;
        world.set_body_material(ground,frictionless);
        world.set_body_material(box,frictionless);
        world.set_body_velocity(box,{0.3f,0,0});
        world.step(1.0f/120.0f);
        expect(std::abs(world.body_state(box).velocity.x-0.3f)<0.005f,
               "World reference adapter failed to author PhysX shape material");
    }
#endif

    verify_demand_driven_island_ownership_checkpoint();
    verify_source_contact_edge_deactivations();
    verify_source_retired_bridge_deactivation();
    verify_source_active_island_handoff();
    verify_source_speculative_reverse_sleep_order();
    verify_source_mixed_split_and_sleep_order();
    verify_source_solver_task_batch_partitioning();
    verify_source_correlated_friction_patch_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify();
    fyzz::native::ContactCorrelationSourceFixture::verify_kinematic_contact_manager_source_order();
    fyzz::native::ContactCorrelationSourceFixture::verify_kinematic_threshold_source_pairs();
    fyzz::native::ContactCorrelationSourceFixture::verify_kinematic_contact_reporting_source_lifetime();
    fyzz::native::ContactCorrelationSourceFixture::verify_kinematic_target_source_reference_parity();
    fyzz::native::ContactCorrelationSourceFixture::verify_settled_kinematic_contact_solver_input();
    fyzz::native::ContactCorrelationSourceFixture::verify_moving_kinematic_target_through_source_solver();
    fyzz::native::ContactCorrelationSourceFixture::verify_kinematic_node_is_not_a_solver_island();
    fyzz::native::ContactCorrelationSourceFixture::verify_wake_before_lost_edge_split();
    fyzz::native::ContactCorrelationSourceFixture::verify_default_native_pcm_scene();
    fyzz::native::ContactCorrelationSourceFixture::verify_default_pcm_failure_atomic_cache();
    fyzz::native::ContactCorrelationSourceFixture::verify_failed_pcm_regeneration_clears_retained_points();
    fyzz::native::ConstraintIslandSourceFixture::verify_source_distance_1d_execution();
    fyzz::native::ConstraintIslandSourceFixture::verify_mixed_joint_contact_partitions();
    fyzz::native::ConstraintIslandSourceFixture::verify_mixed_partition_overflow_waves();
    fyzz::native::ConstraintIslandSourceFixture::verify_mixed_task_local_writeback();
#if defined(FYZZXYZZY_ENABLE_PHYSX_SOURCE) && FYZZXYZZY_ENABLE_PHYSX_SOURCE
    fyzz::native::ConstraintIslandSourceFixture::verify_original_distance_joint_1d();
#endif
    fyzz::native::ConstraintIslandSourceFixture::verify_prepublication_constraint_retirement();
    fyzz::native::ConstraintIslandSourceFixture::verify_typed_constraint_edge_lifetime();
    fyzz::native::ConstraintIslandSourceFixture::verify_island_node_type_counts();
    fyzz::native::ConstraintIslandSourceFixture::verify_constraint_sleep_deactivation_generations();
    fyzz::native::ConstraintIslandSourceFixture::verify_post_third_pass_handle_retirement_rollback();
    fyzz::native::ConstraintIslandSourceFixture::verify_actor_retirement();
    fyzz::native::ConstraintIslandSourceFixture::verify_counted_interaction_lifetime();
    fyzz::native::ConstraintIslandSourceFixture::verify_joint_counted_sleep_energy();
    fyzz::native::ConstraintIslandSourceFixture::verify_graph_owned_kinematic_active_references();
    fyzz::native::ConstraintIslandSourceFixture::verify_independent_lost_route_producers();
    fyzz::native::ConstraintIslandSourceFixture::verify_concurrent_lost_route_repair_rollback();
    fyzz::native::ConstraintIslandSourceFixture::verify_short_route_repairs_stay_serial();
    fyzz::native::ConstraintIslandSourceFixture::verify_single_dispatch_for_split_and_sleep();
    fyzz::native::ConstraintIslandSourceFixture::verify_after_integration_deactivation_reconciliation();
    fyzz::native::ConstraintIslandSourceFixture::verify_dy_ready_event_reaches_both_islands();
    fyzz::native::ConstraintIslandSourceFixture::verify_reused_worker_shutdown();
    fyzz::native::ConstraintIslandSourceFixture::verify_parallel_third_pass_failure_rollback();
    fyzz::native::ConstraintIslandSourceFixture::verify_speculative_active_island_ownership();
    fyzz::native::ConstraintIslandSourceFixture::verify_speculative_ready_is_independent();
    fyzz::native::ConstraintIslandSourceFixture::verify_sc_activity_third_pass_publication();
    fyzz::native::ConstraintIslandSourceFixture::verify_active_kinematic_guards_rigid_island_sleep();
    fyzz::native::ConstraintIslandSourceFixture::verify_kinematic_speculative_sleep_veto();
    fyzz::native::ConstraintIslandSourceFixture::verify_inactive_kinematic_link_sleep_arbitration();
    fyzz::native::ConstraintIslandSourceFixture::verify_kinematic_endpoint_not_static_world();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_pcm_box_box_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_pcm_plane_box_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify_mixed_pcm_legacy_cache_ownership();
    fyzz::native::ContactCorrelationSourceFixture::verify_legacy_gu_contact_cache();
    fyzz::native::ContactCorrelationSourceFixture::verify_replaced_cached_friction_anchor();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_sleep_velocity_mod_retirement();
    fyzz::native::ContactCorrelationSourceFixture::verify_after_integration_cached_shape_bounds();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_retained_acceleration_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_post_solver_continuations();
    fyzz::native::ContactCorrelationSourceFixture::verify_enhanced_determinism_tasks();
    fyzz::native::ContactCorrelationSourceFixture::verify_reversible_cache_checkpoint();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_gu_manager_lookup_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_friction_task_order();
    fyzz::native::ContactCorrelationSourceFixture::verify_post_solver_manager_retirement();
    fyzz::native::ContactCorrelationSourceFixture::verify_post_solver_lost_report_join();
    fyzz::native::ContactCorrelationSourceFixture::verify_zero_capped_stream_compaction();
    fyzz::native::ContactCorrelationSourceFixture::verify_zero_capped_dynamic_partition_lifecycle();
    fyzz::native::ContactCorrelationSourceFixture::verify_compliant_contact_source();
    fyzz::native::ContactCorrelationSourceFixture::verify_signed_compliant_source_writeback();
    fyzz::native::ContactCorrelationSourceFixture::verify_signed_compliant_friction_solver();
    fyzz::native::ContactCorrelationSourceFixture::verify_source_contact_slop();
    fyzz::native::SolverIterationSourceFixture::verify_active_manager_admission();
    fyzz::native::SolverIterationSourceFixture::verify();
    fyzz::native::SolverIterationSourceFixture::verify_parallel_pgs_groups();
    fyzz::native::SolverIterationSourceFixture::verify_parallel_solver_end_rollback();
    verify_source_body_dynamics_api();
    verify_source_rigid_lock_flags();
    verify_source_gyroscopic_motion();
    verify_source_spatial_velocity_mods();
    verify_source_compliant_world_parity();
    verify_source_signed_compliant_contact_writeback();
    verify_kinematic_anchored_constraint_handoff();
    verify_anchored_distance_constraint_angular_jacobians();
    verify_public_center_distance_constraint_lifecycle();
    verify_source_simulation_shape_flag_lifecycle();
    verify_source_contact_notification_world();
    verify_removed_actor_touch_notifications();
    verify_source_rigid_contact_limits();
    verify_source_shape_offsets();
    verify_source_sleep_wake_generator_matrix();
    verify_source_combined_lifecycle_matrix();
    verify_source_multiframe_completion();
    verify_source_pose_authoring();
    verify_source_zero_velocity_iteration_request();
    verify_source_reverse_active_island_sleep_retirement();
    verify_source_reverse_wake_order_and_rollback_capacity();
    verify_source_active_reference_lifecycle();
    verify_source_contact_interaction_consumption();
    verify_source_speculative_only_active_references();
    verify_source_activity_rollback();
    verify_sparse_motion_checkpoint();
    verify_native_rigid_integration_and_island_lifecycle();

    // Exercise the actual World selection, not only the owner fixture.
    fyzz::World native_world(fyzz::WorldOptions{fyzz::ExecutionMode::NativeCpu});
    auto owned = native_world.add_dynamic_box({0.5f,0.5f,0.5f}, {0,1,0}, 1);
    expect(native_world.body_state(owned).position.y == 1,
           "SDK World did not own native actor state");
    native_world.set_body_position(owned, {0,3,0});
    expect(native_world.body_state(owned).position.y == 3,
           "World did not relocate its native actor");
    native_world.set_body_velocity(owned, {1,0,0});
    native_world.step(1.0f/60.0f);
    expect(native_world.body_state(owned).position.x > 0.0f &&
           native_world.body_state(owned).position.y < 3.0f,
           "Native World step did not advance gravity and actor state");
    native_world.remove_body(owned);
    bool retired = false;
    try { (void)native_world.body_state(owned); }
    catch (const std::out_of_range&) { retired = true; }
    expect(retired, "World did not retire the native actor");
    return 0;
}

int main() {
    try {
    fyzz::native::GpuSapSceneAdmissionFixture::verify();
    fyzz::native::GpuSapSceneAdmissionFixture::verify_world_installation();
    fyzz::native::SourceSapEndpointFixture::verify();
    fyzz::native::SourceSapEndpointFixture::verify_pair_manager_states();
    fyzz::native::SourceSapEndpointFixture::verify_kinematic_sap_refilter();
    fyzz::native::SourceSapEndpointFixture::verify_crossing_stress();
    fyzz::native::SourceSapEndpointFixture::verify_persistent_overlap_crossings();
    fyzz::native::SourceSapEndpointFixture::verify_static_only_full_sweep();
    fyzz::native::SourceSapEndpointFixture::verify_source_activity_pockets();
    fyzz::native::SourceSapEndpointFixture::verify_bounded_bulk_shift_publication();
    fyzz::native::SourceSapEndpointFixture::verify_orthogonal_bulk_crossing_filter();
    fyzz::native::SourceSapEndpointFixture::verify_comoving_transient_crossings();
        fyzz::native::SourceSapEndpointFixture::verify_bounded_merge_pair_transitions();
    fyzz::native::SourceSapEndpointFixture::verify_stable_multi_update_pair_lifetime();
    fyzz::native::SourceSapEndpointFixture::verify_adversarial_activity_pockets();
        return run_native_scene_smoke();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native scene smoke: %s\n", e.what());
        return 1;
    }
}
