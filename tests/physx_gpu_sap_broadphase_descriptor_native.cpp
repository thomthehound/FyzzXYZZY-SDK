// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapBroadphaseDescriptorNative.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_physx_heap_roots.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "fyzzxyzzy/compute.h"
#include <PxPhysicsAPI.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

int main() {
    try {
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
                physx::PxgCudaBufferN<27> singles(heap,group);
                physx::PxgCudaBufferN<3> sap(heap,group),newSap(heap,group),ranks(heap,group);
                physx::PxgCudaBufferN<3> projections(heap,group),totalEnd(heap,group);
                physx::PxgCudaBufferN<3> blockTotalEnd(heap,group),incremental(heap,group),blockIncremental(heap,group);
                physx::PxgCudaBufferN<6> boxHandles(heap,group),endHist(heap,group),blockEndHist(heap,group);
                physx::PxgCudaBufferN<6> startHist(heap,group),blockStartHist(heap,group),endHandles(heap,group),startHandles(heap,group);
                physx::PxgCudaBufferN<2> aggBlocks(heap,group),actorBlocks(heap,group);
                physx::PxgCudaBufferN<6> managerBuffers(heap,group);
                physx::PxgTypedCudaBuffer<physx::PxgBroadPhaseDesc> descriptorDevice(heap,group);
                for(physx::PxU32 i=0;i<27;++i) singles[i].allocate(32u,PX_FL);
                const std::array<physx::PxgCudaBufferN<3>*,8> axis3{
                    &sap,&newSap,&ranks,&projections,&totalEnd,&blockTotalEnd,&incremental,&blockIncremental};
                for(auto* a:axis3)
                    for(physx::PxU32 i=0;i<3;++i) (*a)[i].allocate(32u,PX_FL);
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
                    singles[0],singles[1],singles[2],singles[3],singles[4],
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
                expect(desc.updateData_createdHandles,singles[0]);
                expect(desc.updateData_removedHandles,singles[1]);
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
                expect(desc.updateData_updatedHandles,singles[2]);
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
