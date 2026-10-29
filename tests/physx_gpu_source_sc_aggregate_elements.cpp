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

#include <algorithm>
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
            const auto root2=sc.create_aggregate(
                1u,physx::Bp::getFilterGroup_Statics(),0u);
            const auto second_root_handle=sc.aggregate_root_handle(root2);
            const auto group0=owner.scene_bounds().groups[root_handle];
            const auto group1=owner.scene_bounds().groups[second_root_handle];
            require(group0!=group1 &&
                    group0!=physx::Bp::getFilterGroup_Statics() &&
                    group1!=physx::Bp::getFilterGroup_Statics() &&
                    (group0&BP_FILTERING_TYPE_MASK)==
                        physx::Bp::FilterType::AGGREGATE &&
                    (group1&BP_FILTERING_TYPE_MASK)==
                        physx::Bp::FilterType::AGGREGATE,
                    "Original source Sc aggregate roots reused static or duplicate BP groups");
            sc.destroy_aggregate(root2);
            require(member0!=member1 && member0!=root_handle &&
                    member1!=root_handle && ordinary!=root_handle &&
                    owner.source_aggregated().boundedTest(member0) &&
                    owner.source_aggregated().boundedTest(member1) &&
                    !owner.source_aggregated().boundedTest(ordinary) &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[member1]==1u,
                    "Source Sc allocated duplicate or detached aggregate member IDs");

            // ScBodySim::switchToKinematic and switchToDynamic keep the
            // original aggregate member ElementSim and reinsert it with
            // its newly calculated Bp motion group.
            const auto dynamic_group=owner.scene_bounds().groups[member0];
            sc.set_motion(0u,fyzz::native::GpuBroadphaseBox::Motion::Kinematic);
            require(sc.handle(0u)==member0 &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.source_aggregated().boundedTest(member0) &&
                    (owner.scene_bounds().groups[member0]&BP_FILTERING_TYPE_MASK)==
                        physx::Bp::FilterType::KINEMATIC &&
                    owner.scene_bounds().groups[member0]!=dynamic_group,
                    "Kinematic aggregate member lost original ID or Bp filtering group");
            sc.set_motion(0u,fyzz::native::GpuBroadphaseBox::Motion::Dynamic);
            require(owner.scene_bounds().groups[member0]==dynamic_group &&
                    owner.source_aggregated().boundedTest(member0) &&
                    sc.aggregate_root_handle(root)==root_handle,
                    "Dynamic aggregate member failed original source Bp reinsertion");

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

            rejected=false;
            try {
                auto wrong_motion=boxes;
                wrong_motion[0].motion=
                    fyzz::native::GpuBroadphaseBox::Motion::Kinematic;
                (void)owner.prepare(wrong_motion,sc.handles(),filter);
            }catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,
                "Source GPU accepted a Kinematic Sc actor with its old dynamic BP group");

            // Recover after a completed aggregate GPU generation. The
            // original AABB owner must republish only real SAP endpoints,
            // while every still-live member regains its GPU bounds history.
            owner.complete_fetch(frame);
            require(!owner.source_aggregate(root).isNew &&
                    owner.source_aggregate(root).prevSize==2u,
                    "Original source aggregate frame did not retire previous size");
            owner.reset_for_source_rebuild();
            const auto dirty=owner.dirty_aggregate_indices();
            require(dirty.size()==2u &&
                    std::find(dirty.begin(),dirty.end(),root)!=dirty.end() &&
                    std::find(dirty.begin(),dirty.end(),root2)!=dirty.end(),
                    "Source GPU rollback omitted the retired aggregate descriptor slot");
            require(owner.source_aggregate(root).isNew &&
                    owner.source_aggregate(root).prevSize==0u &&
                    owner.source_aggregated().boundedTest(member0) &&
                    owner.source_aggregated().boundedTest(member1) &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[member1]==1u,
                    "Aggregate recovery lost source membership or previous-size reset");
            const auto rebuilt=owner.prepare(boxes,sc.handles(),filter);
            require(rebuilt.getNumCreatedHandles()==2u &&
                    rebuilt.getCreatedHandles()[0]==root_handle &&
                    rebuilt.getCreatedHandles()[1]==ordinary &&
                    rebuilt.getNumRemovedHandles()==0u &&
                    rebuilt.getNumUpdatedHandles()==0u,
                    "Aggregate recovery replayed members as SAP endpoints");

            sc.mark_changed(0u,0.4f);
            require(owner.source_changed().boundedTest(member0)!=0u,
                    "Original Sc member edit did not mark the source GPU bitmap");

            // The source release contract returns member actors to
            // the scene. Re-admission acquires a new ElementSim identity
            // while old aggregate reports retain their retired handle.
            owner.complete_fetch(rebuilt);
            sc.remove_actor_from_aggregate(root,0u);
            const auto standalone0=sc.handle(0u);
            require(standalone0!=member0 &&
                    sc.element_deleted(member0) &&
                    !owner.source_aggregated().boundedTest(member0) &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[standalone0]==0u &&
                    owner.scene_bounds().groups[standalone0]==dynamic_group,
                    "Aggregate removeActor lost the surviving Sc actor or pending GPU identity");
            const auto detached=owner.prepare(boxes,sc.handles(),filter);
            require(detached.getNumCreatedHandles()==1u &&
                    detached.getCreatedHandles()[0]==standalone0,
                    "Source aggregate actor removal did not create a standalone SAP endpoint");
            owner.complete_fetch(detached);
            require(owner.volume_to_actor()[member0]==
                        std::numeric_limits<fyzz::BodyId>::max(),
                    "Source removed aggregate handle survived GPU report cleanup");

            // Simulation-disabled members remain attached to the
            // aggregate until release, but must return as disabled
            // standalone actors with fresh, unrecycled Sc identities.
            sc.set_simulation_enabled(1u,false);
            sc.destroy_aggregate(root);
            const auto standalone1=sc.handle(1u);
            require(standalone1!=member1 &&
                    sc.element_deleted(member1) &&
                    sc.element_deleted(root_handle) &&
                    !owner.source_aggregated().boundedTest(member1),
                    "Original aggregate release failed to return disabled Sc actor");
            sc.set_simulation_enabled(1u,true);
            const auto released=owner.prepare(boxes,sc.handles(),filter);
            require(released.getNumCreatedHandles()==1u &&
                    released.getCreatedHandles()[0]==standalone1,
                    "Released Sc actor failed standalone SAP reinsertion");
            owner.complete_fetch(released);
            require(owner.volume_to_actor()[standalone1]==1u,
                    "Released source Sc actor lost its standalone owner");
            sc.post_reports_cleanup();
            require(!sc.element_deleted(root_handle) &&
                    !sc.element_deleted(member0) &&
                    !sc.element_deleted(member1),
                "Released original Sc member IDs were not retired at cleanup");
            sc.retire(1u);
            sc.retire(0u);

            // A disabled shape still occupies its PxAggregate actor
            // attachment slot even though it is absent from BP members.
            const auto limited=sc.create_aggregate(
                1u,physx::Bp::getFilterGroup_Statics(),0u);
            sc.admit_aggregate_member(
                limited,3u,fyzz::native::GpuBroadphaseBox::Motion::Static,0.0f);
            sc.set_simulation_enabled(3u,false);
            bool full=false;
            try {
                sc.admit_aggregate_member(
                    limited,4u,
                    fyzz::native::GpuBroadphaseBox::Motion::Static,0.0f);
            }catch(const std::invalid_argument&) {full=true;}
            require(full,
                "Disabled original Sc actor incorrectly freed aggregate capacity");
            sc.destroy_aggregate(limited);
            sc.retire(3u);
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
