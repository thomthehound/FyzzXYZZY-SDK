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

    // A quiet frame carries no created/updated source volumes. Geometry
    // validation still runs, but the persistent source AABB arrays must
    // not be repopulated from every Sc actor on each collision frame.
    actors[0].position.x=6.0f;
    actors[1].position.x=-10.0f;
    const SourceSapSceneFrameChanges quiet{{},{},{}};
    (void)prepare_source_sap_scene_update(
        actors,handles,target,quiet,filter,false,false,scratch);
    require(bounds_equal() && distances==beforeDistances &&
            groups==beforeGroups,
            "Quiet source SAP frame rewrote persistent AABB manager arrays");

    // The original GPU AABB manager never populates the CPU updated
    // handle array. Its mChangedHandleMap supplies changed volumes to
    // markUpdatedPairs, including when the bounds actor is not 'created'.
    actors[0].position.x=4.0f;
    actors[0].contact_offset=0.3125f;
    const std::array<physx::PxU32,1> gpu_changed{1u};
    const SourceSapSceneFrameChanges gpu_dirty{
        {},{},{},gpu_changed};
    const auto gpu_frame=prepare_source_sap_scene_update(
        actors,handles,target,gpu_dirty,filter,true,true,scratch);
    require(gpu_frame.getNumUpdatedHandles()==0u &&
            gpu_frame.getGpuStateChanged() &&
            std::fabs(bounds[0].minimum.x-3.0f)<1.0e-6f &&
            std::fabs(bounds[3].minimum.x+8.0f)<1.0e-6f &&
            distances[0]==0.3125f &&
            distances[3]==beforeDistances[3],
            "GPU changed bitmap with empty updated list lost moved Sc bounds");

    // Source bitmap word indices above 31 must preserve volume-handle
    // identity and leave unmarked peers untouched.
    std::array<physx::PxBounds3,40> high_bounds{};
    for(auto& b:high_bounds)b=physx::PxBounds3::empty();
    std::array<physx::PxReal,40> high_offsets{};
    std::array<physx::Bp::FilterGroup::Enum,40> high_groups{};
    high_groups.fill(physx::Bp::FilterGroup::eINVALID);
    SourceSapSceneBoundsScratch high_scratch;
    const SourceSapSceneBounds high_target{
        high_bounds,high_offsets,high_groups};
    const std::array<native::GpuBroadphaseBox,2> high_boxes{{
        {11u,{0.f,0.f,0.f},{1.f,1.f,1.f},{},0.02f,
            native::GpuBroadphaseBox::Motion::Dynamic},
        {4u,{3.f,0.f,0.f},{1.f,1.f,1.f},{},0.03f,
            native::GpuBroadphaseBox::Motion::Static}
    }};
    const std::array<physx::Bp::ShapeHandle,2> high_handles{36u,2u};
    const std::array<physx::Bp::ShapeHandle,2> high_created{2u,36u};
    (void)prepare_source_sap_scene_update(
        high_boxes,high_handles,high_target,{high_created,{},{}},
        filter,true,true,high_scratch);
    const auto initial_low=high_bounds[2].minimum.x;
    auto shifted=high_boxes;
    shifted[0].position.x=7.f;
    shifted[1].position.x=100.f;
    const std::array<physx::PxU32,2> high_dirty{0u,1u<<4u};
    const auto high_frame=prepare_source_sap_scene_update(
        shifted,high_handles,high_target,{{},{},{},high_dirty},
        filter,true,true,high_scratch);
    require(high_frame.getNumUpdatedHandles()==0u &&
            high_frame.getNumCreatedHandles()==0u &&
            std::fabs(high_bounds[36].minimum.x-6.f)<1.0e-6f &&
            high_bounds[2].minimum.x==initial_low,
            "Source GPU changed bitmap failed at 32-bit volume boundary");

    // Only the authoritative updated handle is reconstructed, even when
    // another live Sc actor's pose differs from its retained GPU bounds.
    const std::array<physx::Bp::ShapeHandle,1> moved{0u};
    const SourceSapSceneFrameChanges moving{{},moved,{}};
    actors[0].contact_offset=0.375f;
    (void)prepare_source_sap_scene_update(
        actors,handles,target,moving,filter,true,true,scratch);
    require(std::fabs(bounds[0].minimum.x-5.0f)<1.0e-6f &&
            std::fabs(bounds[3].minimum.x+8.0f)<1.0e-6f &&
            distances[0]==0.375f && distances[3]==beforeDistances[3],
            "Sparse source SAP update changed an unmodified volume");

    // A late invalid actor is rejected before publishing even the
    // earlier valid changed volume to the original source arrays.
    const auto sparseBefore=bounds;
    const auto sparseDistances=distances;
    const auto sparseGroups=groups;
    const std::array<physx::Bp::ShapeHandle,2> bothMoved{0u,3u};
    const SourceSapSceneFrameChanges invalidUpdate{{},bothMoved,{}};
    actors[0].position.x=9.0f;
    actors[1].half_extents.x=-1.0f;
    rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            actors,handles,target,invalidUpdate,filter,true,true,scratch);
    }catch(const std::invalid_argument&){rejected=true;}
    bool unchanged=rejected && distances==sparseDistances &&
                   groups==sparseGroups;
    for(std::size_t i=0;i<bounds.size();++i) {
        const auto& a=bounds[i];
        const auto& b=sparseBefore[i];
        unchanged=unchanged && a.minimum.x==b.minimum.x &&
                  a.minimum.y==b.minimum.y && a.minimum.z==b.minimum.z &&
                  a.maximum.x==b.maximum.x && a.maximum.y==b.maximum.y &&
                  a.maximum.z==b.maximum.z;
    }
    require(unchanged,
            "Malformed sparse source SAP update partially published bounds");

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
