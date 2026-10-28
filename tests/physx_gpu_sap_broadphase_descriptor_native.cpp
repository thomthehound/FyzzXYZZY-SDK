// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"
#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_port/gpubroadphase/PxgSapRadixDescriptorsNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpUploadsNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "BpBroadPhaseUpdate.h"
#include "physx_port/gpucommon/include/PxgRadixSortKernelIndices.h"
#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_physx_heap_roots.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_sap_scene_reports.h"
#include "physx_gpu/source_sap_scene_bounds.h"
#include "fyzzxyzzy/compute.h"
#include <PxPhysicsAPI.h>

#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <stdexcept>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

// The live scene-to-source preBroadPhase input uses original AABB-manager
// volume handles, true oriented box bounds and source filter group encoding.
// No CPU SAP, independent proxy allocator or second pair history is involved.
static void verify_source_sap_scene_bounds() {
    using Box=fyzz::native::GpuBroadphaseBox;
    constexpr float s=0.7071067811865475f;
    const std::array<Box,3> boxes{{
        {7u,{3.0f,4.0f,5.0f},{2.0f,1.0f,0.5f},
            {0.0f,0.0f,s,s},0.02f,Box::Motion::Dynamic},
        {3u,{-4.0f,0.0f,0.0f},{1.0f,2.0f,3.0f},
            {},0.05f,Box::Motion::Static},
        {9u,{0.0f,0.0f,1.0f},{0.5f,0.5f,0.5f},
            {},0.10f,Box::Motion::Kinematic}
    }};
    // Source volume handle order is independent of Sc BodyId order.
    const std::array<physx::Bp::ShapeHandle,3> handles{4u,1u,6u};
    std::array<physx::PxBounds3,7> bounds;
    bounds.fill(physx::PxBounds3::centerExtents(
        physx::PxVec3(0.0f),physx::PxVec3(1.0f)));
    std::array<physx::PxReal,7> distances;
    distances.fill(-1.0f);
    std::array<physx::Bp::FilterGroup::Enum,7> groups;
    groups.fill(physx::Bp::FilterGroup::eINVALID);
    SourceSapSceneBoundsScratch scratch;
    const SourceSapSceneBounds output{bounds,distances,groups};
    const auto check=[](float a,float b) {
        return std::abs(a-b)<2.0e-5f;
    };
    write_source_sap_scene_bounds(boxes,handles,output,scratch);
    if(!check(bounds[4].minimum.x,2.0f) ||
       !check(bounds[4].minimum.y,2.0f) ||
       !check(bounds[4].minimum.z,4.5f) ||
       !check(bounds[4].maximum.x,4.0f) ||
       !check(bounds[4].maximum.y,6.0f) ||
       !check(bounds[4].maximum.z,5.5f) ||
       !check(distances[4],0.02f) ||
       !check(bounds[1].minimum.x,-5.0f) ||
       !check(bounds[1].maximum.y,2.0f) ||
       !check(distances[1],0.05f) ||
       !check(bounds[6].minimum.z,0.5f) ||
       !check(distances[6],0.10f) ||
       groups[4]!=physx::Bp::getFilterGroup_Dynamics(7u,false) ||
       groups[1]!=physx::Bp::getFilterGroup_Statics() ||
       groups[6]!=physx::Bp::getFilterGroup_Dynamics(9u,true) ||
       distances[0]!=-1.0f ||
       groups[0]!=physx::Bp::FilterGroup::eINVALID)
        throw std::runtime_error(
            "Native Sc oriented bounds/contact offsets/source PhysX groups lowered incorrectly");

    // Quiet source frames do not modify the original AABB-manager arrays.
    write_source_sap_scene_bounds({}, {},output,scratch);
    if(!check(bounds[4].minimum.x,2.0f) || distances[4]!=0.02f)
        throw std::runtime_error("Quiet source bounds publication mutated existing volume");

    // Invalid late inputs cannot partially update an earlier valid handle.
    auto invalid=boxes;
    invalid[2].orientation.w=0.0f;
    bool rejected=false;
    try {write_source_sap_scene_bounds(invalid,handles,output,scratch);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected || !check(bounds[4].minimum.x,2.0f) ||
       distances[4]!=0.02f || groups[6]!=
          physx::Bp::getFilterGroup_Dynamics(9u,true))
        throw std::runtime_error(
            "Malformed late source geometry partially changed GPU AABB input");

    auto duplicates=handles;
    duplicates[2]=handles[0];
    rejected=false;
    try {write_source_sap_scene_bounds(boxes,duplicates,output,scratch);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected || distances[4]!=0.02f)
        throw std::runtime_error("Duplicate original source volume handle accepted");

    auto outside=handles;
    outside[2]=static_cast<physx::PxU32>(bounds.size());
    rejected=false;
    try {write_source_sap_scene_bounds(boxes,outside,output,scratch);}
    catch(const std::out_of_range&) {rejected=true;}
    if(!rejected || distances[4]!=0.02f)
        throw std::runtime_error("Out-of-range source AABB handle partially published");

    // Original Bp update inputs must keep ascending, disjoint source
    // handle deltas independent of the scene's unsorted actor order.
    const physx::Bp::BpFilter filter(false,false);
    const std::array<physx::Bp::ShapeHandle,3> created{1u,4u,6u};
    const std::array<physx::Bp::ShapeHandle,1> retired{0u};
    const SourceSapSceneFrameChanges initial{created,{},retired};
    const auto update=prepare_source_sap_scene_update(
        boxes,handles,output,initial,filter,true,true,scratch);
    if(update.getNumCreatedHandles()!=3u ||
       update.getNumUpdatedHandles()!=0u ||
       update.getNumRemovedHandles()!=1u ||
       update.getCreatedHandles()!=created.data() ||
       update.getRemovedHandles()!=retired.data() ||
       update.getCapacity()!=bounds.size() ||
       update.getAABBs()!=bounds.data() ||
       update.getGroups()!=groups.data() ||
       update.getContactDistance()!=distances.data() ||
       !update.getStateChanged() || !update.getGpuStateChanged())
        throw std::runtime_error("Original source Bp frame deltas were not preserved");

    auto moved=boxes;
    moved[0].position.x=12.0f;
    const std::array<physx::Bp::ShapeHandle,1> updated{4u};
    const SourceSapSceneFrameChanges badOverlap{updated,updated,{}};
    rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            moved,handles,output,badOverlap,filter,true,true,scratch);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected || !check(bounds[4].minimum.x,2.0f))
        throw std::runtime_error(
            "Overlapping original source handle deltas modified staged bounds");

    const std::array<physx::Bp::ShapeHandle,2> unordered{6u,4u};
    rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            moved,handles,output,{unordered,{},{}},filter,true,true,scratch);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected || !check(bounds[4].minimum.x,2.0f))
        throw std::runtime_error("Unordered source handle deltas partially published");

    rejected=false;
    try {
        (void)prepare_source_sap_scene_update(
            moved,handles,output,{{},{},updated},filter,true,true,scratch);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected || !check(bounds[4].minimum.x,2.0f))
        throw std::runtime_error("Live source handle was erroneously retired");

    const auto moving=prepare_source_sap_scene_update(
        moved,handles,output,{{},updated,{}},filter,true,false,scratch);
    if(moving.getNumCreatedHandles()!=0u ||
       moving.getNumUpdatedHandles()!=1u ||
       moving.getUpdatedHandles()!=updated.data() ||
       !check(bounds[4].minimum.x,11.0f))
        throw std::runtime_error("Source Bp moving-frame admission lost handle identity");
}

// Original GPU source pair handles are not necessarily Scene actor IDs.
// This proves the real fetch ABI maps volume handles and duplicate source
// reports into the Sc-ordered stream before its contact-edge publication.
static void verify_source_sap_sc_report_mapping() {
    using fyzz::native::Pair;
    constexpr std::array<fyzz::BodyId,5> ownership{50u,3u,25u,7u,100u};
    const std::array<physx::PxgBroadPhasePair,3> source_found{
        physx::PxgBroadPhasePair{0u,2u},
        physx::PxgBroadPhasePair{1u,4u},
        physx::PxgBroadPhasePair{0u,2u}};
    const std::array<physx::PxgBroadPhasePair,1> source_lost{
        physx::PxgBroadPhasePair{1u,3u}};
    std::vector<Pair> found,lost;
    lower_source_sap_actor_reports_to_sc(
        source_found,source_lost,ownership,found,lost);
    const std::vector<Pair> expected_found{{3u,100u},{25u,50u}};
    const std::vector<Pair> expected_lost{{3u,7u}};
    if(found!=expected_found || lost!=expected_lost)
        throw std::runtime_error(
            "Source GPU fetched pair handles did not map to canonical native Sc pairs");
    const std::array<physx::PxgBroadPhasePair,1> invalid{
        physx::PxgBroadPhasePair{1u,9u}};
    bool rejected=false;
    try {
        lower_source_sap_actor_reports_to_sc(
            source_found,invalid,ownership,found,lost);
    } catch(const std::out_of_range&) {
        rejected=true;
    }
    if(!rejected || found!=expected_found || lost!=expected_lost)
        throw std::runtime_error(
            "Invalid source GPU lost report partially published Sc pairs");
    auto retired=ownership;
    retired[2]=std::numeric_limits<fyzz::BodyId>::max();
    rejected=false;
    try {
        lower_source_sap_actor_reports_to_sc(
            source_found,source_lost,retired,found,lost);
    } catch(const std::out_of_range&) {
        rejected=true;
    }
    if(!rejected || found!=expected_found || lost!=expected_lost)
        throw std::runtime_error(
            "Source GPU pair mapping accepted an already retired volume handle");
}

int main() {
    try {
        verify_source_sap_sc_report_mapping();
        verify_source_sap_scene_bounds();
        physx::PxDefaultAllocator defaultAllocator;
        physx::PxDefaultErrorCallback errors;
        auto* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,defaultAllocator,errors);
        if(!foundation) throw std::runtime_error("PhysX foundation initialization failed");
        try {
#if defined(FYZZXYZZY_TEST_D3D12)
            Context context(Backend::D3D12);
#else
            Context context(Backend::Vulkan);
#endif
            CudaDeviceAddressMap addresses;
            SourceSapDmaNative stream(context,addresses);
            SourcePhysxHeapRoots roots(context,addresses);
            SourcePhysxDeviceRootCallback callback(roots,&stream);
            bool aborted=false;
            const auto abort=[](void* user) noexcept { *static_cast<bool*>(user)=true; };
            {
                physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
                constexpr int group=physx::PxsHeapStats::eBROADPHASE;
                // Exercise the source preBroadPhase GPU upload with
                // real Sc shape geometry, external source volume handles,
                // and original PhysX-owned buddy-heap buffers.
                {
                    using Box=fyzz::native::GpuBroadphaseBox;
                    const std::array<Box,2> boxes{{
                        {12u,{1.0f,3.0f,5.0f},{2.0f,1.0f,0.5f},
                            {},0.03f,Box::Motion::Dynamic},
                        {19u,{-1.0f,-2.0f,0.0f},{1.0f,2.0f,0.5f},
                            {},0.07f,Box::Motion::Kinematic}
                    }};
                    const std::array<physx::Bp::ShapeHandle,2> handles{3u,0u};
                    std::array<physx::PxBounds3,4> bounds;
                    bounds.fill(physx::PxBounds3::centerExtents(
                        physx::PxVec3(0.0f),physx::PxVec3(1.0f)));
                    std::array<physx::PxReal,4> distances{};
                    std::array<physx::Bp::FilterGroup::Enum,4> groups;
                    groups.fill(physx::Bp::FilterGroup::eINVALID);
                    SourceSapSceneBoundsScratch scratch;
                    const physx::Bp::BpFilter filter(false,false);
                    // Source creation/removal lists are sorted separately
                    // from the scene input and the manager's handle mapping.
                    const std::array<physx::Bp::ShapeHandle,2> created{0u,3u};
                    const auto update=prepare_source_sap_scene_update(
                        boxes,handles,{bounds,distances,groups},
                        {created,{},{}},filter,true,false,scratch);
                    physx::PxgTypedCudaBuffer<physx::PxBounds3> gpuBounds(heap,group);
                    physx::PxgTypedCudaBuffer<physx::PxReal> gpuDistances(heap,group);
                    physx::PxgTypedCudaBuffer<physx::PxU32> gpuGroups(heap,group);
                    physx::PxgTypedCudaBuffer<physx::PxU32> gpuEnvironments(heap,group);
                    physx::PxU32 sourceCapacity=0u;
                    pre_broadphase_native(update,{
                        gpuBounds,gpuDistances,gpuGroups,gpuEnvironments,
                        sourceCapacity},stream);
                    stream.wait_at_fetch();
                    const auto checkBytes=[&](const auto& gpu,const void* expected,
                                             std::size_t bytes,const char* diagnostic) {
                        const auto view=addresses.resolve(gpu.getDevicePtr(),bytes);
                        std::vector<std::uint8_t> observed(bytes);
                        context.download(view.buffer,view.offset,
                                         observed.data(),bytes);
                        if(std::memcmp(observed.data(),expected,bytes)!=0)
                            throw std::runtime_error(diagnostic);
                    };
                    if(sourceCapacity!=4u)
                        throw std::runtime_error("Source scene AABB input capacity changed");
                    checkBytes(gpuBounds,bounds.data(),sizeof(bounds),
                               "Live Sc shape AABB GPU upload differs from source input");
                    checkBytes(gpuDistances,distances.data(),sizeof(distances),
                               "Live Sc contact-distance GPU upload differs from source input");
                    checkBytes(gpuGroups,groups.data(),sizeof(groups),
                               "Live Sc filter-group GPU upload differs from source input");

                    // A persistent source handle can move without being
                    // created again. Its original GPU bounds allocation
                    // must receive the new authored geometry at preBP.
                    auto moved=boxes;
                    moved[0].position.x=8.0f;
                    const std::array<physx::Bp::ShapeHandle,1> changed{3u};
                    const auto movedFrame=prepare_source_sap_scene_update(
                        moved,handles,{bounds,distances,groups},
                        {{},changed,{}},filter,true,false,scratch);
                    pre_broadphase_native(movedFrame,{
                        gpuBounds,gpuDistances,gpuGroups,gpuEnvironments,
                        sourceCapacity},stream);
                    stream.wait_at_fetch();
                    checkBytes(gpuBounds,bounds.data(),sizeof(bounds),
                               "Source moving-volume update did not reach original GPU bounds");
                    if(std::abs(bounds[3].minimum.x-6.0f)>1.0e-5f)
                        throw std::runtime_error(
                            "Original source handle did not preserve moved Sc shape bounds");

                    // The source always uploads groups and distances, but
                    // skips the bounds upload for stateChanged=false.
                    const auto bytesBefore=stream.submission_stats().upload_bytes;
                    const auto idle=prepare_source_sap_scene_update(
                        moved,handles,{bounds,distances,groups},
                        {},filter,false,false,scratch);
                    pre_broadphase_native(idle,{
                        gpuBounds,gpuDistances,gpuGroups,gpuEnvironments,
                        sourceCapacity},stream);
                    stream.wait_at_fetch();
                    if(stream.submission_stats().upload_bytes-bytesBefore !=
                          sizeof(distances)+sizeof(groups))
                        throw std::runtime_error(
                            "Source idle preBroadPhase unexpectedly reuploaded AABBs");
                }
                physx::PxgCudaBufferN<27> singles(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxU32> created(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxU32> removed(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxU32> updated(heap,group);
                created.allocateElements(8u,PX_FL);
                removed.allocateElements(8u,PX_FL);
                updated.allocateElements(8u,PX_FL);
                physx::PxgCudaBufferN<3> sap(heap,group),newSap(heap,group),ranks(heap,group);
                physx::PxgCudaBufferN<3> projections(heap,group);
                // Source members are N<6>, not N<3>; only axes 0..2 are used.
                physx::PxgCudaBufferN<6> totalEnd(heap,group),blockTotalEnd(heap,group);
                physx::PxgCudaBufferN<3> incremental(heap,group),blockIncremental(heap,group);
                physx::PxgCudaBufferN<6> boxHandles(heap,group),endHist(heap,group),blockEndHist(heap,group);
                physx::PxgCudaBufferN<6> startHist(heap,group),blockStartHist(heap,group),endHandles(heap,group),startHandles(heap,group);
                physx::PxgCudaBufferN<2> aggBlocks(heap,group),actorBlocks(heap,group);
                physx::PxgCudaBufferN<6> managerBuffers(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxgBroadPhaseDesc> descriptorDevice(heap,group);
                for(physx::PxU32 i=0;i<27;++i) singles[i].allocate(32u,PX_FL);
                const std::array<physx::PxgCudaBufferN<3>*,6> axis3{
                    &sap,&newSap,&ranks,&projections,&incremental,&blockIncremental};
                for(auto* a:axis3)
                    for(physx::PxU32 i=0;i<3;++i) (*a)[i].allocate(32u,PX_FL);
                for(physx::PxU32 i=0;i<3;++i) {
                    totalEnd[i].allocate(32u,PX_FL);
                    blockTotalEnd[i].allocate(32u,PX_FL);
                }
                const std::array<physx::PxgCudaBufferN<6>*,7> axis6{
                    &boxHandles,&endHist,&blockEndHist,&startHist,&blockStartHist,&endHandles,&startHandles};
                for(auto* a:axis6)
                    for(physx::PxU32 i=0;i<6;++i) (*a)[i].allocate(32u,PX_FL);
                for(physx::PxU32 i=0;i<2;++i) {
                    aggBlocks[i].allocate(32u,PX_FL);
                    actorBlocks[i].allocate(32u,PX_FL);
                }
                for(physx::PxU32 i=0;i<6;++i) managerBuffers[i].allocate(32u,PX_FL);
                descriptorDevice.allocateElements(1u,PX_FL);

                SourceSapBroadphaseDescriptorBuffers buffers{
                    created,removed,updated,singles[3],singles[4],
                    singles[5],singles[6],singles[7],singles[8],
                    singles[9],singles[10],singles[11],singles[12],singles[13],singles[14],
                    sap,newSap,ranks,projections,boxHandles,totalEnd,blockTotalEnd,
                    endHist,blockEndHist,startHist,blockStartHist,endHandles,startHandles,
                    incremental,blockIncremental,aggBlocks,actorBlocks,
                    singles[15],singles[16],singles[17],singles[18],
                    singles[19],singles[20],singles[21],singles[22],
                    singles[23],singles[24],singles[25],singles[26]
                };
                SourceSapBroadphaseDescriptorInputs in{
                    4u,2u,1u,5u,
                    singles[13].getDevicePtr(),singles[14].getDevicePtr(),
                    13u,17u,100u,80u,nullptr
                };
                auto ptr=[](const physx::PxgCudaBuffer& b) {
                    return static_cast<std::uintptr_t>(b.getDevicePtr());
                };
                auto expect=[](const auto* actual,const physx::PxgCudaBuffer& b) {
                    if(reinterpret_cast<std::uintptr_t>(actual)!=b.getDevicePtr())
                        throw std::runtime_error("Source SAP broadphase GPU descriptor pointer binding mismatch");
                };
                physx::PxgBroadPhaseDesc desc{};
                std::memset(&desc,0xaa,sizeof(desc));
                prepare_sap_broadphase_descriptor_native(desc,buffers,in);
                expect(desc.updateData_createdHandles,created);
                expect(desc.updateData_removedHandles,removed);
                expect(desc.oldIntegerBounds,singles[3]);
                expect(desc.newIntegerBounds,singles[4]);
                expect(desc.updateData_fpBounds,singles[5]);
                expect(desc.updateData_contactDistances,singles[6]);
                expect(desc.updateData_groups,singles[7]);
                expect(desc.updateData_envIDs,singles[8]);
                const std::array<physx::PxgBroadPhasePair*,6> pairs{
                    desc.foundPairReport,desc.lostPairReport,desc.foundAggPairReport,
                    desc.lostAggPairReport,desc.foundActorPairReport,desc.lostActorPairReport};
                for(std::size_t i=0;i<pairs.size();++i)expect(pairs[i],singles[9+i]);
                if(reinterpret_cast<std::uintptr_t>(desc.foundPairReportMap)!=in.mapped_found_actor_pairs ||
                   reinterpret_cast<std::uintptr_t>(desc.lostPairReportMap)!=in.mapped_lost_actor_pairs)
                    throw std::runtime_error("Mapped PhysX GPU pair report alias binding changed");

                for(physx::PxU32 i=0;i<3;++i) {
                    expect(desc.boxSapBox1D[i],sap[i]);
                    expect(desc.boxNewSapBox1D[i],newSap[i]);
                    expect(desc.boxProjectionRanks[i],ranks[i]);
                    expect(desc.boxProjections[i],projections[i]);
                    expect(desc.totalEndPtHistogram[i],totalEnd[i]);
                    expect(desc.blockTotalEndPtHistogram[i],blockTotalEnd[i]);
                    expect(desc.incrementalComparisons[i],incremental[i]);
                    expect(desc.incrementalBlockComparisons[i],blockIncremental[i]);
                    for(physx::PxU32 j=0;j<2;++j){
                        const physx::PxU32 index=j*3+i;
                        expect(desc.boxHandles[j][i],boxHandles[index]);
                        expect(desc.endPtHistogram[j][i],endHist[index]);
                        expect(desc.blockEndPtHistogram[j][i],blockEndHist[index]);
                        expect(desc.startPtHistogram[j][i],startHist[index]);
                        expect(desc.blockStartPtHistogram[j][i],blockStartHist[index]);
                        expect(desc.endPointHandles[j][i],endHandles[index]);
                        expect(desc.startPointHandles[j][i],startHandles[index]);
                    }
                }
                for(physx::PxU32 i=0;i<2;++i) {
                    expect(desc.aggReportBlock[i],aggBlocks[i]);
                    expect(desc.actorReportBlock[i],actorBlocks[i]);
                }
                const std::array<physx::PxU32*,12> regions{
                    desc.activeRegionsHistogram,desc.startRegionsHistogram,
                    desc.orderedActiveRegionHandles,desc.orderedStartRegionHandles,
                    reinterpret_cast<physx::PxU32*>(desc.blockOverlapChecksRegion),
                    reinterpret_cast<physx::PxU32*>(desc.overlapChecksRegion),
                    reinterpret_cast<physx::PxU32*>(desc.overlapChecksHandleRegiones),
                    reinterpret_cast<physx::PxU32*>(desc.regionRange),
                    desc.startRegionAccum,desc.blockStartRegionAccum,
                    desc.regionAccum,desc.blockRegionAccum
                };
                for(std::size_t i=0;i<regions.size();++i)expect(regions[i],singles[15+i]);
                if(desc.numPreviousHandles!=3u || desc.numCreatedHandles!=2u ||
                   desc.numRemovedHandles!=1u || desc.sharedFoundPairIndex ||
                   desc.sharedLostPairIndex || desc.sharedFoundAggPairIndex ||
                   desc.sharedLostAggPairIndex || desc.startRegionAccumTotal ||
                   desc.regionAccumTotal!=13u || desc.overlapChecksTotalRegion!=17u ||
                   desc.max_found_lost_pairs!=100u ||
                   desc.max_found_lost_agg_pairs!=80u ||
                   desc.found_lost_pairs_overflow_flags)
                    throw std::runtime_error("Original source SAP descriptor scalar/reset contract changed");
                for(const auto count:desc.totalIncrementalComparisons)
                    if(count)throw std::runtime_error("Original descriptor zero initialization lost");
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
                expect(desc.updateData_updatedHandles,updated);
                if(desc.numUpdatedHandles!=5u)
                    throw std::runtime_error("Standalone SAP descriptor did not retain updated count");
#endif
                if(desc.aabbMngr_changedHandleMap || desc.aabbMngr_volumeData ||
                   desc.aabbMngr_changedHandleBitMapWordCounts)
                    throw std::runtime_error("Standalone SAP descriptor invented AABB manager");

                SourceSapAabbManagerDescriptorView manager{
                    managerBuffers[0].getDevicePtr(),3u,managerBuffers[1].getDevicePtr(),
                    managerBuffers[2].getDevicePtr(),managerBuffers[3].getDevicePtr(),
                    managerBuffers[4].getDevicePtr()
                };
                in.aabb_manager=&manager;
                std::memset(&desc,0xbb,sizeof(desc));
                prepare_sap_broadphase_descriptor_native(desc,buffers,in);
                expect(desc.aabbMngr_changedHandleMap,managerBuffers[0]);
                expect(desc.aabbMngr_addedHandleMap,managerBuffers[1]);
                expect(desc.aabbMngr_removedHandleMap,managerBuffers[2]);
                expect(desc.aabbMngr_aggregatedBoundHandles,managerBuffers[3]);
                expect(desc.aabbMngr_volumeData,managerBuffers[4]);
                if(desc.aabbMngr_changedHandleBitMapWordCounts!=3u ||
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
                   desc.updateData_updatedHandles || desc.numUpdatedHandles ||
#endif
                   desc.numPreviousHandles!=3u)
                    throw std::runtime_error("AABB-manager source descriptor used standalone update path");

                // Buffers may grow between frames. Rebuild pointers from
                // the CURRENT PhysX owner; no cached GPU raw address.
                singles[3].allocate(4096u,PX_FL);
                prepare_sap_broadphase_descriptor_native(desc,buffers,in);
                expect(desc.oldIntegerBounds,singles[3]);

                // Invalid counts must not corrupt the previous descriptor.
                auto invalid=in;
                invalid.num_boxes=std::numeric_limits<physx::PxU32>::max();
                invalid.created_count=0u; invalid.removed_count=1u;
                const auto before=desc;
                bool rejected=false;
                try {prepare_sap_broadphase_descriptor_native(desc,buffers,invalid);}
                catch(const std::overflow_error&) {rejected=true;}
                if(!rejected||std::memcmp(&before,&desc,sizeof(desc))!=0)
                    throw std::runtime_error("Invalid original SAP count corrupted descriptor");

                // Original descriptor is GPU-uploadable unmodified. The
                // production sequence will upload it FIRST, after handle
                // uploads and before the two six-entry radix arrays.
                const auto expected=desc;
                stream.queue_uploads({{descriptorDevice.getDevicePtr(),&desc,sizeof(desc)}});
                std::memset(&desc,0xcc,sizeof(desc));
                stream.wait_at_fetch();
                physx::PxgBroadPhaseDesc gpu{};
                const auto view=addresses.resolve(descriptorDevice.getDevicePtr(),sizeof(gpu));
                context.download(view.buffer,view.offset,&gpu,sizeof(gpu));
                if(std::memcmp(&expected,&gpu,sizeof(gpu))!=0)
                    throw std::runtime_error("Source broadphase descriptor ABI or GPU upload snapshot changed");

                // Integrated source host stage: all records are built BEFORE
                // the first changed-handle HtoD. The pinned PhysX SAP owner
                // includes PxgBroadPhaseDesc.h before declaring the optional
                // updated members, so these count/buffer views must match.
                static_assert(sizeof(SourceSapGpuDMAUpCounts)==8u*sizeof(physx::PxU32));
                physx::PxU32 liveBoxes=2u,boxesCapacity=0u;
                physx::PxU32 createdCount=0u,removedCount=0u,updatedCount=0u;
                SourceSapGpuDMAUpCountMembers members{
                    liveBoxes,boxesCapacity,createdCount,removedCount,updatedCount};
                SourceSapGpuDMAUpHandleBuffers handles{created,removed,updated};
                physx::Bp::BpFilter filter(false,false);
                std::array<physx::PxU32,2> createdInput{7u,8u};
                std::array<physx::PxU32,1> removedInput{0u};
                std::array<physx::PxU32,1> updatedInput{1u};
                const auto savedCreated=createdInput;
                const auto savedRemoved=removedInput;
                const auto savedUpdated=updatedInput;
                physx::Bp::BroadPhaseUpdateData frame(
                    createdInput.data(),2u,updatedInput.data(),1u,
                    removedInput.data(),1u,
                    nullptr,nullptr,nullptr,4u,filter,true,true);
                const auto counts=prepare_gpu_dma_up_counts_native(frame,members);
                if(counts.previous_boxes!=2u || counts.current_boxes!=3u ||
                   counts.updated!=1u || createdCount!=2u ||
                   removedCount!=1u || updatedCount!=1u)
                    throw std::runtime_error("Source SAP feature macro/layout broke updated-handle count binding");
                allocate_gpu_dma_up_handles_native(counts,handles);

                // Use original source buffer collections to build BOTH
                // forms of radix descriptors; their six-element host
                // arrays and GPU storage are owned by the SAP instance.
                physx::PxgCudaBufferN<3> tempKeys(heap,group),tempHandles(heap,group),radixCounts(heap,group);
                for(physx::PxU32 i=0;i<3u;++i) {
                    tempKeys[i].allocate(64u,PX_FL);
                    tempHandles[i].allocate(64u,PX_FL);
                    radixCounts[i].allocate(
                        sizeof(physx::PxU32) *
                        physx::PxgRadixSortKernelGridDim::RADIX_SORT * 16u,PX_FL);
                }
                physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> normalGpu(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> withoutRanksGpu(heap,group);
                normalGpu.allocateElements(6u,PX_FL);
                withoutRanksGpu.allocateElements(6u,PX_FL);
                std::array<physx::PxgRadixSortDesc,6> normal{},wor{};
                for(physx::PxU32 i=0;i<6u;++i) {
                    normal[i].count=10u+i;
                    wor[i].count=20u+i;
                }
                prepare_sap_radix_descriptors_native(
                    normal.data(),wor.data(),
                    {projections,ranks,tempKeys,tempHandles,radixCounts,boxHandles});

                in.num_boxes=counts.current_boxes;
                in.created_count=counts.created;
                in.removed_count=counts.removed;
                in.updated_count=counts.updated;
                in.aabb_manager=nullptr;
                prepare_sap_broadphase_descriptor_native(desc,buffers,in);
                if(desc.updateData_updatedHandles!=updated.getTypedPtr() ||
                   desc.numUpdatedHandles!=1u ||
                   desc.updateData_createdHandles!=created.getTypedPtr())
                    throw std::runtime_error("Source standalone update-list descriptor ABI disagrees with handles");
                in.aabb_manager=&manager;
                prepare_sap_broadphase_descriptor_native(desc,buffers,in);
                if(desc.updateData_updatedHandles || desc.numUpdatedHandles ||
                   desc.aabbMngr_changedHandleBitMapWordCounts!=manager.changed_bitmap_word_count)
                    throw std::runtime_error("Source manager descriptor incorrectly bound optional updated list");

                SourceSapDmaUpDescriptorBuffers recordBuffers{
                    descriptorDevice,normalGpu,withoutRanksGpu};
                bool rejectedNoRadix=false;
                try {
                    queue_sap_dma_up_descriptors_native(
                        desc,normal.data(),nullptr,recordBuffers,stream);
                } catch(const std::invalid_argument&) {
                    rejectedNoRadix=true;
                }
                if(!rejectedNoRadix)
                    throw std::runtime_error("PhysX SAP queued incomplete descriptor upload group");
                physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> missingRecord(heap,group);
                bool rejectedUnallocated=false;
                try {
                    queue_sap_dma_up_descriptors_native(
                        desc,normal.data(),wor.data(),
                        {descriptorDevice,normalGpu,missingRecord},stream);
                } catch(const std::logic_error&) {
                    rejectedUnallocated=true;
                }
                if(!rejectedUnallocated)
                    throw std::runtime_error("PhysX SAP uploaded into an unallocated descriptor destination");

                const auto expectedBp=desc;
                const auto expectedNormal=normal;
                const auto expectedWor=wor;
                // Actual source upload order: created, removed, updated,
                // broadphase, normal radix[6], WOR radix[6], same mStream.
                // Standalone helpers keep immediate-submit semantics for
                // unrelated callers; only the *combined source frame*
                // safely joins known adjacent HtoD transfers.
                queue_gpu_dma_up_handles_native(frame,counts,handles,stream);
                queue_sap_dma_up_descriptors_native(
                    desc,normal.data(),wor.data(),recordBuffers,stream);
                physx::PxgCudaBuffer copyAfterDescriptors(heap,group);
                copyAfterDescriptors.allocate(sizeof(savedCreated),PX_FL);
                stream.queue_device_copy(
                    copyAfterDescriptors.getDevicePtr(),created.getDevicePtr(),
                    sizeof(savedCreated));

                // The source may reuse host update arrays immediately. The
                // native submission snapshots ALL records before returning.
                createdInput.fill(0xeeeeeeeeu);
                removedInput.fill(0xeeeeeeeeu);
                updatedInput.fill(0xeeeeeeeeu);
                std::memset(&desc,0xdd,sizeof(desc));
                for(auto& r:normal)r.inputKeys=nullptr;
                for(auto& r:wor)r.inputRanks=nullptr;
                stream.wait_at_fetch();

                const auto checkGpu=[&](const physx::PxgCudaBuffer& owner,
                                       const void* expectedBytes,std::size_t bytes,
                                       const char* diagnostic) {
                    const auto v=addresses.resolve(owner.getDevicePtr(),bytes);
                    std::vector<std::uint8_t> got(bytes);
                    context.download(v.buffer,v.offset,got.data(),bytes);
                    if(std::memcmp(got.data(),expectedBytes,bytes)!=0)
                        throw std::runtime_error(diagnostic);
                };
                checkGpu(created,savedCreated.data(),sizeof(savedCreated),
                         "Source ordered created-handle GPU upload failed");
                checkGpu(removed,savedRemoved.data(),sizeof(savedRemoved),
                         "Source ordered removed-handle GPU upload failed");
                checkGpu(updated,savedUpdated.data(),sizeof(savedUpdated),
                         "Source ordered updated-handle GPU upload failed");
                checkGpu(descriptorDevice,&expectedBp,sizeof(expectedBp),
                         "Source ordered broadphase descriptor GPU upload failed");
                checkGpu(normalGpu,expectedNormal.data(),sizeof(expectedNormal),
                         "Source ordered normal radix descriptor GPU upload failed");
                checkGpu(withoutRanksGpu,expectedWor.data(),sizeof(expectedWor),
                         "Source ordered WOR radix descriptor GPU upload failed");
                checkGpu(copyAfterDescriptors,savedCreated.data(),sizeof(savedCreated),
                         "Source-ordered GPU copy after descriptor DMA lost handle upload");

                // Exercise the complete source gpuDMAUp call-site bridge,
                // rather than independently invoking the individual stage
                // helpers. PhysX still owns the SAME buffers and source
                // stream: there is no second broadphase or heap.
                const std::array<physx::PxU32,8> previousProjection{
                    0x9a11u,0x9b22u,0x9c33u,0x9d44u,
                    0x9e55u,0x9f66u,0xa077u,0xa188u
                };
                {
                    const auto view=addresses.resolve(
                        projections[0].getDevicePtr(),sizeof(previousProjection));
                    context.upload(view.buffer,view.offset,
                                   previousProjection.data(),sizeof(previousProjection));
                }
                const auto earlierProjectionSize=projections[0].getSize();
                std::array<physx::PxU32,2> additionalCreated{9u,10u};
                std::array<physx::PxU32,1> additionalRemoved{1u};
                std::array<physx::PxU32,1> additionalUpdated{2u};
                const auto sourceCreated=additionalCreated;
                const auto sourceRemoved=additionalRemoved;
                const auto sourceUpdated=additionalUpdated;
                physx::Bp::BroadPhaseUpdateData nextUpdate(
                    additionalCreated.data(),2u,
                    additionalUpdated.data(),1u,
                    additionalRemoved.data(),1u,
                    nullptr,nullptr,nullptr,6u,filter,true,true);

                std::array<SourceSapHistoryAxisBuffers,3> axes{{
                    {sap[0],newSap[0],projections[0],ranks[0],tempKeys[0],tempHandles[0]},
                    {sap[1],newSap[1],projections[1],ranks[1],tempKeys[1],tempHandles[1]},
                    {sap[2],newSap[2],projections[2],ranks[2],tempKeys[2],tempHandles[2]}
                }};
                SourceSapDmaUpFrameMembers borrowed{
                    members,handles,singles[3],axes,
                    {boxHandles,endHist,blockEndHist,endHandles,
                     startHist,blockStartHist,startHandles,
                     incremental,blockIncremental,totalEnd,blockTotalEnd},
                    {aggBlocks,actorBlocks,singles[15],singles[16],
                     singles[17],singles[18],singles[20],singles[19],
                     singles[21],singles[22],singles[23],singles[24],
                     singles[25],singles[26]},
                    singles[4],buffers,in,
                    {projections,ranks,tempKeys,tempHandles,radixCounts,boxHandles},
                    recordBuffers,desc,normal.data(),wor.data()
                };
                // A missing original constructor-owned GPU descriptor must
                // reject the ENTIRE frame before counters or GPU commands
                // change. Retrying valid storage then applies changes once.
                SourceSapDmaUpFrameMembers unallocatedFrame{
                    borrowed.counts,borrowed.handles,borrowed.old_integer_bounds,
                    borrowed.axes,borrowed.remainder,borrowed.post_axis,
                    borrowed.new_integer_bounds,borrowed.broadphase_buffers,
                    borrowed.descriptor_inputs,borrowed.radix_buffers,
                    {descriptorDevice,normalGpu,missingRecord},
                    borrowed.broadphase_descriptor,
                    borrowed.radix_descriptors,borrowed.radix_wor_descriptors
                };
                const auto untouchedLive=liveBoxes;
                const auto untouchedCapacity=boxesCapacity;
                const auto untouchedProjectionSize=projections[0].getSize();
                const auto untouchedCreatedCount=createdCount;
                const auto untouchedRemovedCount=removedCount;
                const auto untouchedUpdatedCount=updatedCount;
                bool rejectedUnallocatedFrame=false;
                try {
                    (void)gpu_dma_up_sap_frame_native(
                        nextUpdate,unallocatedFrame,stream);
                } catch(const std::logic_error&) {
                    rejectedUnallocatedFrame=true;
                }
                if(!rejectedUnallocatedFrame ||
                   liveBoxes!=untouchedLive || boxesCapacity!=untouchedCapacity ||
                   createdCount!=untouchedCreatedCount ||
                   removedCount!=untouchedRemovedCount ||
                   updatedCount!=untouchedUpdatedCount ||
                   projections[0].getSize()!=untouchedProjectionSize ||
                   !stream.commands().empty())
                    throw std::runtime_error(
                        "Rejected source SAP frame mutated counters or queued GPU copies");

                // Boundary of upstream totalNbProjectionRegions (PxU32):
                // 33,554,431 live+created boxes -> 67,108,862
                // projections -> 4,294,967,168 region slots (fits);
                // adding one more box would require 4,294,967,296 slots.
                // Neither case needs a multi-gigabyte GPU allocation:
                // this is a pure source count/region preflight test.
                constexpr physx::PxU32 maxSafeBoxes=33'554'431u;
                physx::PxU32 hugeBoxes=maxSafeBoxes;
                physx::PxU32 hugeCapacity=911u;
                physx::PxU32 hugeCreated=921u,hugeRemoved=931u,hugeUpdated=941u;
                SourceSapGpuDMAUpCountMembers hugeMembers{
                    hugeBoxes,hugeCapacity,hugeCreated,hugeRemoved,hugeUpdated};
                const std::array<physx::PxU32,1u> oneNewHandle{17u};
                const physx::Bp::BroadPhaseUpdateData overRegionLimit(
                    oneNewHandle.data(),1u,nullptr,0u,nullptr,0u,
                    nullptr,nullptr,nullptr,maxSafeBoxes+1u,
                    filter,true,true);
                SourceSapDmaUpFrameMembers overflowFrame{
                    hugeMembers,borrowed.handles,borrowed.old_integer_bounds,
                    borrowed.axes,borrowed.remainder,borrowed.post_axis,
                    borrowed.new_integer_bounds,borrowed.broadphase_buffers,
                    borrowed.descriptor_inputs,borrowed.radix_buffers,
                    borrowed.descriptor_gpu_buffers,borrowed.broadphase_descriptor,
                    borrowed.radix_descriptors,borrowed.radix_wor_descriptors
                };
                const auto unchangedHistorySize=projections[0].getSize();
                const auto unchangedOldBoundsSize=singles[3].getSize();
                const auto unchangedDescriptor=desc;
                bool rejectedOversizedRegion=false;
                try {
                    (void)gpu_dma_up_sap_frame_native(
                        overRegionLimit,overflowFrame,stream);
                } catch(const std::overflow_error&) {
                    rejectedOversizedRegion=true;
                }
                if(!rejectedOversizedRegion || hugeBoxes!=maxSafeBoxes ||
                   hugeCapacity!=911u || hugeCreated!=921u ||
                   hugeRemoved!=931u || hugeUpdated!=941u ||
                   projections[0].getSize()!=unchangedHistorySize ||
                   singles[3].getSize()!=unchangedOldBoundsSize ||
                   std::memcmp(&desc,&unchangedDescriptor,sizeof(desc))!=0 ||
                   !stream.commands().empty())
                    throw std::runtime_error(
                        "Region overflow committed SAP counters or queued partial GPU history");
                const physx::Bp::BroadPhaseUpdateData atRegionLimit(
                    nullptr,0u,nullptr,0u,nullptr,0u,
                    nullptr,nullptr,nullptr,maxSafeBoxes,
                    filter,false,false);
                const auto acceptedBoundary=prepare_gpu_dma_up_counts_native(
                    atRegionLimit,hugeMembers);
                if(acceptedBoundary.current_boxes!=maxSafeBoxes ||
                   acceptedBoundary.projections!=67'108'862u ||
                   checked_sap_region_slots_native(acceptedBoundary.projections)
                        !=4'294'967'168ull ||
                   hugeCapacity!=maxSafeBoxes ||
                   hugeCreated || hugeRemoved || hugeUpdated)
                    throw std::runtime_error(
                        "SAP region-slot preflight rejected its valid PxU32 boundary");
                if(!stream.commands().empty())
                    throw std::runtime_error(
                        "Valid source count preflight incorrectly enqueued GPU work");

                // Source PxgCudaBroadPhaseSap::update runs gpuDMAUp first;
                // a changed frame requests a final idle convergence update.
                bool forceUpdate=true;
                const auto beforeFrameBatches=stream.submission_stats();
                const auto nextAdmission=begin_sap_update_native(
                    nextUpdate,borrowed,stream,forceUpdate);
                const auto nextCounts=nextAdmission.counts;
                if(!nextAdmission.execute_gpu_stages || !forceUpdate)
                    throw std::runtime_error(
                        "Changed source SAP frame failed update admission");
                const auto afterFrameBatches=stream.submission_stats();
                if(afterFrameBatches.upload_batches!=beforeFrameBatches.upload_batches+1u ||
                   afterFrameBatches.upload_spans!=beforeFrameBatches.upload_spans+6u ||
                   afterFrameBatches.upload_bytes<=beforeFrameBatches.upload_bytes)
                    throw std::runtime_error(
                        "Source SAP full-frame handle+descriptor HtoD was split into extra submissions");
                if(nextCounts.previous_boxes!=3u ||
                   nextCounts.current_boxes!=4u ||
                   nextCounts.capacity!=6u ||
                   nextCounts.projections!=10u ||
                   nextCounts.padded_projections!=12u ||
                   nextCounts.updated!=1u ||
                   projections[0].getSize()<=earlierProjectionSize ||
                   desc.numPreviousHandles!=3u ||
                   desc.numCreatedHandles!=2u ||
                   desc.numRemovedHandles!=1u ||
                   desc.aabbMngr_changedHandleBitMapWordCounts!=3u ||
                   desc.updateData_updatedHandles ||
                   desc.numUpdatedHandles ||
                   desc.boxProjections[0]!=
                       reinterpret_cast<physx::PxU32*>(projections[0].getDevicePtr()))
                    throw std::runtime_error("Full source SAP gpuDMAUp frame order or owner bindings changed");

                const auto nextExpectedDesc=desc;
                const auto nextExpectedNormal=normal;
                const auto nextExpectedWor=wor;
                additionalCreated.fill(0xdeadbeefu);
                additionalRemoved.fill(0xdeadbeefu);
                additionalUpdated.fill(0xdeadbeefu);
                std::memset(&desc,0xa5,sizeof(desc));
                for(auto& r:normal)r.outputKeys=nullptr;
                for(auto& r:wor)r.outputRanks=nullptr;
                stream.wait_at_fetch();
                checkGpu(created,sourceCreated.data(),sizeof(sourceCreated),
                         "Source gpuDMAUp frame lost created-handle transfer");
                checkGpu(removed,sourceRemoved.data(),sizeof(sourceRemoved),
                         "Source gpuDMAUp frame lost removed-handle transfer");
                checkGpu(updated,sourceUpdated.data(),sizeof(sourceUpdated),
                         "Source gpuDMAUp frame lost updated-handle transfer");
                checkGpu(descriptorDevice,&nextExpectedDesc,sizeof(nextExpectedDesc),
                         "Source gpuDMAUp frame lost broadphase descriptor");
                checkGpu(normalGpu,nextExpectedNormal.data(),sizeof(nextExpectedNormal),
                         "Source gpuDMAUp frame lost normal radix descriptor");
                checkGpu(withoutRanksGpu,nextExpectedWor.data(),sizeof(nextExpectedWor),
                         "Source gpuDMAUp frame lost WOR radix descriptor");
                // The first 32 bytes of the old projection allocation must
                // survive a larger native copy-preserving allocation,
                // including DtoD GPU command ordering before descriptor HtoD.
                checkGpu(projections[0],previousProjection.data(),
                         sizeof(previousProjection),
                         "Source gpuDMAUp frame destroyed prior GPU projection history");

                // The source runs exactly one otherwise-idle GPU update
                // to converge bounds/history, then skips kernel stages on
                // later idle frames. Each frame still runs gpuDMAUp.
                const physx::Bp::BroadPhaseUpdateData idleFrame(
                    nullptr,0u,nullptr,0u,nullptr,0u,
                    nullptr,nullptr,nullptr,6u,filter,false,false);
                const auto beforeIdle=stream.submission_stats();
                const auto forced=begin_sap_update_native(
                    idleFrame,borrowed,stream,forceUpdate);
                if(!forced.execute_gpu_stages || forceUpdate ||
                   forced.counts.previous_boxes!=4u ||
                   forced.counts.current_boxes!=4u || liveBoxes!=4u)
                    throw std::runtime_error(
                        "Source SAP lost its single forced idle convergence pass");
                const auto afterForced=stream.submission_stats();
                const auto trulyIdle=begin_sap_update_native(
                    idleFrame,borrowed,stream,forceUpdate);
                const auto afterIdle=stream.submission_stats();
                if(trulyIdle.execute_gpu_stages || forceUpdate ||
                   trulyIdle.counts.previous_boxes!=4u ||
                   trulyIdle.counts.current_boxes!=4u ||
                   afterForced.upload_batches!=beforeIdle.upload_batches+1u ||
                   afterIdle.upload_batches!=afterForced.upload_batches+1u)
                    throw std::runtime_error(
                        "Quiet SAP frame bypassed source DMA or repeated GPU stages");
                stream.wait_at_fetch();

                // Rejecting a malformed frame cannot consume the source
                // force-update state or commit a new box population.
                bool rejectedForceUpdate=true;
                const auto unchanged=liveBoxes;
                bool rejectedBroken=false;
                try {
                    (void)begin_sap_update_native(
                        overRegionLimit,overflowFrame,stream,
                        rejectedForceUpdate);
                } catch(const std::overflow_error&) {
                    rejectedBroken=true;
                }
                if(!rejectedBroken || !rejectedForceUpdate ||
                   liveBoxes!=unchanged || !stream.commands().empty())
                    throw std::runtime_error(
                        "Rejected source SAP DMA consumed update admission");


            }
            if(aborted||roots.live_roots()||addresses.live_allocations())
                throw std::runtime_error("Original SAP descriptor leaked GPU buddy heap roots");
        }catch(...){foundation->release();throw;}
        foundation->release();
        std::cout<<"PhysX SAP original broadphase descriptor GPU ABI PASS\n";
    }catch(const std::exception& e) {
        std::cerr<<"PhysX SAP original broadphase descriptor FAILED: "<<e.what()<<'\n';
        return 1;
    }
    return 0;
}
