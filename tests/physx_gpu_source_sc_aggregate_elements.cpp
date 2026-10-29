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
        {
            // A rejected Sc/Dy frame discards GPU histories after the
            // preceding source fetch. Recreate only standalone SAP
            // volumes and aggregate roots; their members retain their
            // original Sc IDs and enter aggregate history admission.
            PxgAabbSourceMembersNative owner(allocators,16u,32u,64u);
            SourceScSceneElementsNative sc(owner);
            const auto root0=sc.create_aggregate(
                2u,physx::Bp::getFilterGroup_Statics(),1u);
            const auto root1=sc.create_aggregate(
                1u,physx::Bp::getFilterGroup_Statics(),0u);
            sc.admit_aggregate_member(
                root0,0u,fyzz::native::GpuBroadphaseBox::Motion::Dynamic,0.15f);
            sc.admit_aggregate_member(
                root0,1u,fyzz::native::GpuBroadphaseBox::Motion::Dynamic,0.15f);
            sc.admit_aggregate_member(
                root1,2u,fyzz::native::GpuBroadphaseBox::Motion::Static,0.0f);
            sc.admit(3u,fyzz::native::GpuBroadphaseBox::Motion::Static,0.0f);
            const auto root_handle0=sc.aggregate_root_handle(root0);
            const auto root_handle1=sc.aggregate_root_handle(root1);
            const auto member0=sc.handle(0u);
            const auto member1=sc.handle(1u);
            const auto member2=sc.handle(2u);
            const auto ordinary=sc.handle(3u);
            const std::array<fyzz::native::GpuBroadphaseBox,4> boxes{{
                box(0u,0.0f,fyzz::native::GpuBroadphaseBox::Motion::Dynamic),
                box(1u,0.4f,fyzz::native::GpuBroadphaseBox::Motion::Dynamic),
                box(2u,0.6f,fyzz::native::GpuBroadphaseBox::Motion::Static),
                box(3u,1.0f,fyzz::native::GpuBroadphaseBox::Motion::Static)
            }};
            const physx::Bp::BpFilter filter(false,false);
            const auto first=owner.prepare(boxes,sc.handles(),filter);
            require(first.getNumCreatedHandles()==3u,
                "Original first aggregate source fetch did not admit both roots");
            // Complete the source host fetch lifecycle; the GPU
            // reconvergence itself is exercised by the World bundle.
            owner.complete_fetch(first);
            require(owner.dirty_aggregate_indices().empty() &&
                    !owner.source_added().boundedTest(root_handle0),
                "Original successful fetch did not retire AABB dirty history");
            sc.mark_changed(0u,0.35f);
            owner.reset_for_source_rebuild();
            require(owner.source_added().boundedTest(root_handle0) &&
                    owner.source_added().boundedTest(root_handle1) &&
                    owner.source_added().boundedTest(ordinary) &&
                    !owner.source_added().boundedTest(member0) &&
                    !owner.source_added().boundedTest(member1) &&
                    !owner.source_added().boundedTest(member2) &&
                    owner.source_aggregated().boundedTest(member0) &&
                    owner.source_aggregated().boundedTest(member1) &&
                    owner.source_aggregated().boundedTest(member2) &&
                    owner.dirty_aggregate_indices().size()==2u &&
                    owner.source_aggregate(root0).isNew &&
                    owner.source_aggregate(root0).prevSize==0u &&
                    owner.source_aggregate(root1).isNew &&
                    sc.handle(0u)==member0 && sc.handle(1u)==member1 &&
                    sc.handle(2u)==member2,
                    "Source rollback reconstructed members as SAP endpoints or lost Sc identities");
            const auto restored=owner.prepare(boxes,sc.handles(),filter);
            require(restored.getNumCreatedHandles()==3u &&
                    restored.getCreatedHandles()[0]==root_handle0 &&
                    restored.getCreatedHandles()[1]==root_handle1 &&
                    restored.getCreatedHandles()[2]==ordinary &&
                    owner.volume_to_actor()[member0]==0u &&
                    owner.volume_to_actor()[member1]==1u &&
                    owner.volume_to_actor()[member2]==2u &&
                    owner.scene_bounds().groups[root_handle0]!=
                        owner.scene_bounds().groups[root_handle1],
                    "Original AABB rollback did not recreate source aggregate roots");
            owner.complete_fetch(restored);
            sc.retire(0u);
            sc.retire(1u);
            sc.retire(2u);
            sc.retire(3u);
            sc.destroy_aggregate(root0);
            sc.destroy_aggregate(root1);
            sc.post_reports_cleanup();
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
