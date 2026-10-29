// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sc_scene_elements_native.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "fyzzxyzzy/compute.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;

namespace {
void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
struct Errors final : physx::PxErrorCallback {
    void reportError(physx::PxErrorCode::Enum,const char* message,
                     const char*,int) override {
        throw std::runtime_error(message);
    }
};
struct Foundation {
    physx::PxDefaultAllocator allocator;
    Errors errors;
    physx::PxFoundation* instance=PxCreateFoundation(
        PX_PHYSICS_VERSION,allocator,errors);
    Foundation() {require(instance!=nullptr,"Original PhysX foundation unavailable");}
    ~Foundation() {if(instance)instance->release();}
};
fyzz::native::GpuBroadphaseBox box(fyzz::BodyId id,float x,
                                     fyzz::native::GpuBroadphaseBox::Motion motion) {
    fyzz::native::GpuBroadphaseBox b{};
    b.id=id;
    b.position={x,0.0f,0.0f};
    b.half_extents={0.5f,0.5f,0.5f};
    b.motion=motion;
    return b;
}
void run(Backend backend) {
    Foundation foundation;
    Context context(backend);
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback callback(roots,&stream);
    SourcePhysxMappedReportRoots pinned(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    SourcePhysxMappedReportRoots mapped(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* state) noexcept {
        *static_cast<bool*>(state)=true;
    };
    {
        physx::PxgHeapMemoryAllocatorManager heap(
            1u<<20u,callback,pinned,mapped,abort,&aborted);
        auto allocators=physx::PxgCreateAllocatorDesc(heap);
        {
            PxgAabbSourceMembersNative owner(allocators,8u,16u,32u);
            SourceScSceneElementsNative sc(owner);
            const auto root=sc.create_aggregate(
                3u,physx::Bp::getFilterGroup_Statics(),1u);
            const auto root_handle=sc.aggregate_root_handle(root);
            require(root_handle!=BP_INVALID_BP_HANDLE &&
                    owner.source_aggregated().boundedTest(root_handle)==0u,
                    "Original Sc empty aggregate manufactured a member handle");

            sc.admit_aggregate_member(
                root,0u,fyzz::native::GpuBroadphaseBox::Motion::Dynamic,0.15f);
            sc.admit_aggregate_member(
                root,1u,fyzz::native::GpuBroadphaseBox::Motion::Dynamic,0.25f);
            sc.admit(2u,fyzz::native::GpuBroadphaseBox::Motion::Static,0.0f);
            const auto member0=sc.handle(0u),member1=sc.handle(1u);
            const auto ordinary=sc.handle(2u);
            require(member0!=member1 && member0!=root_handle &&
                    member1!=root_handle && ordinary!=root_handle &&
                    owner.source_aggregated().boundedTest(member0) &&
                    owner.source_aggregated().boundedTest(member1) &&
                    !owner.source_aggregated().boundedTest(ordinary) &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[member1]==1u,
                    "Source Sc allocated duplicate or detached aggregate member IDs");

            const std::array<fyzz::native::GpuBroadphaseBox,3> boxes{{
                box(0u,-1.0f,fyzz::native::GpuBroadphaseBox::Motion::Dynamic),
                box(1u,1.0f,fyzz::native::GpuBroadphaseBox::Motion::Dynamic),
                box(2u,4.0f,fyzz::native::GpuBroadphaseBox::Motion::Static)
            }};
            const physx::Bp::BpFilter filter(false,false);
            const auto frame=owner.prepare(boxes,sc.handles(),filter);
            require(frame.getNumCreatedHandles()==2u &&
                    frame.getCreatedHandles()[0]==root_handle &&
                    frame.getCreatedHandles()[1]==ordinary &&
                    frame.getNumUpdatedHandles()==0u,
                    "Original Sc aggregate members were wrongly inserted as SAP roots");
            const auto* bounds=frame.getAABBs();
            require(std::abs(bounds[member0].minimum.x+1.5f)<1.0e-5f &&
                    std::abs(bounds[member1].maximum.x-1.5f)<1.0e-5f &&
                    bounds[root_handle].isEmpty(),
                    "Sc member geometry did not populate original AABB bounds");

            bool rejected=false;
            try {
                auto invalid=boxes;
                invalid[0].id=1u;
                (void)owner.prepare(invalid,sc.handles(),filter);
            }catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,
                "Original Sc aggregate geometry accepted incorrect member identity");

            sc.mark_changed(0u,0.4f);
            require(owner.source_changed().boundedTest(member0)!=0u,
                    "Original Sc member edit did not mark the source GPU bitmap");

            rejected=false;
            try {sc.destroy_aggregate(root);}
            catch(const std::logic_error&) {rejected=true;}
            require(rejected,
                    "Original Sc destroyed aggregate while source members are live");

            // Sc shape simulation disablement removes its AABB membership,
            // not its ElementSim attachment to the original aggregate.
            sc.set_simulation_enabled(0u,false);
            sc.set_simulation_enabled(1u,false);
            rejected=false;
            try {sc.destroy_aggregate(root);}
            catch(const std::logic_error&) {rejected=true;}
            require(rejected,
                "Original Sc recycled aggregate ID with disabled members still attached");
            // Both admits were canceled before GPU publication. The
            // original Sc element handles and aggregate ID survive.
            sc.set_simulation_enabled(0u,true);
            sc.set_simulation_enabled(1u,true);
            require(sc.handle(0u)==member0 && sc.handle(1u)==member1 &&
                    sc.aggregate_root_handle(root)==root_handle &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[member1]==1u,
                    "Source Sc aggregate simulation toggle lost stable element identity");

            sc.retire(1u);
            require(!owner.source_aggregated().boundedTest(member1) &&
                    owner.volume_to_actor()[member1]==
                        std::numeric_limits<fyzz::BodyId>::max(),
                    "Canceled member admission retained a source actor identity");
            sc.retire(0u);
            sc.destroy_aggregate(root);
            require(sc.element_deleted(root_handle),
                "Original Sc aggregate root ID was not deferred until postReportsCleanup");
            sc.post_reports_cleanup();
            require(!sc.element_deleted(root_handle),
                "Original Sc aggregate root ID was not retired at cleanup");
        }
        heap.flushDeferredDeallocs();
    }
    roots.close(&stream);
    require(!aborted,"Original source Sc aggregate heap aborted");
}
}
int main() {
    try {
#if defined(FYZZXYZZY_TEST_D3D12)
        run(Backend::D3D12);
#else
        run(Backend::Vulkan);
#endif
        std::cout<<"Original source Sc aggregate ElementSim lifecycle PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Original source Sc aggregate ElementSim lifecycle FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
