// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_scene_frame_native.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fyzz;
using namespace fyzz::physx_gpu;
namespace {
void require(bool condition,const char* description) {
    if(!condition)throw std::runtime_error(description);
}
void test_source_scene_boundary() {
    // The original AABB manager owns handles (which deliberately do NOT
    // equal native actor IDs), the delta lists, groups and bounds arrays.
    std::array<physx::PxBounds3,4> bounds{};
    for(auto& item:bounds)item=physx::PxBounds3::empty();
    std::array<physx::PxReal,4> distances{};
    std::array<physx::Bp::FilterGroup::Enum,4> groups{};
    const SourceSapSceneBounds target{bounds,distances,groups};
    SourceSapSceneBoundsScratch scratch;
    std::array<native::GpuBroadphaseBox,2> actors{};
    actors[0]={11u,{2.0f,3.0f,4.0f},{1.0f,2.0f,3.0f},
               {},0.125f,native::GpuBroadphaseBox::Motion::Dynamic};
    actors[1]={4u,{-6.0f,0.0f,2.0f},{2.0f,1.0f,1.0f},
               {},0.25f,native::GpuBroadphaseBox::Motion::Static};
    const std::array<physx::Bp::ShapeHandle,2> handles{0u,3u};
    const std::array<physx::Bp::ShapeHandle,2> created{0u,3u};
    physx::Bp::BpFilter filter(false,false);
    const SourceSapSceneFrameChanges changes{created,{},{}};
    const auto update=prepare_source_sap_scene_update(
        actors,handles,target,changes,filter,true,true,scratch);
    require(update.getNumCreatedHandles()==2u &&
            update.getNumRemovedHandles()==0u &&
            update.getGpuStateChanged(),
            "Original Bp update delta flags were not retained");
    require(std::fabs(bounds[0].minimum.x-1.0f)<1.0e-6f &&
            std::fabs(bounds[0].maximum.z-7.0f)<1.0e-6f &&
            std::fabs(bounds[3].minimum.x+8.0f)<1.0e-6f &&
            distances[0]==0.125f && distances[3]==0.25f,
            "Sc box geometry did not enter original PhysX AABB manager arrays");
    const auto before=bounds;
    const auto bounds_equal=[&]() {
        for(std::size_t i=0;i<bounds.size();++i) {
            const auto& a=bounds[i];
            const auto& b=before[i];
            if(a.minimum.x!=b.minimum.x || a.minimum.y!=b.minimum.y ||
               a.minimum.z!=b.minimum.z || a.maximum.x!=b.maximum.x ||
               a.maximum.y!=b.maximum.y || a.maximum.z!=b.maximum.z)
                return false;
        }
        return true;
    };
    const auto beforeDistances=distances;
    const auto beforeGroups=groups;
    actors[1].half_extents.x=-1.0f;
    bool rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            actors,handles,target,changes,filter,true,true,scratch);
    }catch(const std::invalid_argument&){rejected=true;}
    require(rejected && bounds_equal() && distances==beforeDistances &&
            groups==beforeGroups,
            "Malformed late actor geometry partially changed source AABB buffers");
    actors[1].half_extents.x=2.0f;
    const std::array<physx::Bp::ShapeHandle,2> duplicate{0u,0u};
    rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            actors,duplicate,target,changes,filter,true,true,scratch);
    }catch(const std::invalid_argument&){rejected=true;}
    require(rejected && bounds_equal(),
            "Duplicated original volume handle changed source bounds");

    // AABB handle order is independent of Sc actor order and may be
    // recycled. PhysX pair ordering is by volume; Sc is by actor ID.
    const std::array<BodyId,4> volume_to_actor{
        11u,std::numeric_limits<BodyId>::max(),9u,4u};
    const std::array<physx::PxgBroadPhasePair,3> sourceFound{
        physx::PxgBroadPhasePair{0u,3u},
        physx::PxgBroadPhasePair{0u,3u},
        physx::PxgBroadPhasePair{2u,3u}};
    const std::array<physx::PxgBroadPhasePair,1> sourceLost{
        physx::PxgBroadPhasePair{0u,2u}};
    std::vector<native::Pair> found,lost;
    lower_source_sap_actor_reports_to_sc(
        sourceFound,sourceLost,volume_to_actor,found,lost);
    require(found.size()==2u &&
            found[0]==native::Pair{4u,9u} &&
            found[1]==native::Pair{4u,11u} &&
            lost.size()==1u && lost[0]==native::Pair{9u,11u},
            "Source found/lost pairs were not canonicalized for Sc");
    const std::array<physx::PxgBroadPhasePair,1> invalidLost{
        physx::PxgBroadPhasePair{1u,3u}};
    rejected=false;
    try {
        lower_source_sap_actor_reports_to_sc(
            sourceFound,invalidLost,volume_to_actor,found,lost);
    }catch(const std::out_of_range&){rejected=true;}
    require(rejected && found.size()==2u && lost.size()==1u &&
            found[0]==native::Pair{4u,9u} &&
            lost[0]==native::Pair{9u,11u},
            "Rejected source lost handle partially published Sc pair reports");
}
}
int main(){
    try{
        test_source_scene_boundary();
        std::cout<<"Original SAP scene AABB and report boundary PASS\n";
    }catch(const std::exception& e){
        std::cerr<<"Original SAP scene boundary FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
