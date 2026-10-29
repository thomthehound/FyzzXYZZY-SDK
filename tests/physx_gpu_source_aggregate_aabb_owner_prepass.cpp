// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
#include "physx_gpu/source_aggregate_post_sap_member_history_native.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "fyzzxyzzy/compute.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <utility>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;

namespace {
void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
struct Errors final:physx::PxErrorCallback {
    void reportError(physx::PxErrorCode::Enum,const char* message,
                     const char*,int) override {
        throw std::runtime_error(message);
    }
};
struct Foundation {
    physx::PxDefaultAllocator allocator;
    Errors errors;
    physx::PxFoundation* ptr=PxCreateFoundation(
        PX_PHYSICS_VERSION,allocator,errors);
    Foundation() {require(ptr!=nullptr,"Source PhysX Foundation unavailable");}
    ~Foundation() {if(ptr)ptr->release();}
};
ShaderBinary load(const char* path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Missing native aggregate source kernel");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
    ShaderBinary bin;
#if defined(FYZZXYZZY_TEST_D3D12)
    bin.format=ShaderFormat::Dxil;
#else
    bin.format=ShaderFormat::Spirv;
#endif
    bin.bytes.assign(bytes.begin(),bytes.end());
    return bin;
}
void verify(Backend backend,ShaderBinary bounds_shader,ShaderBinary mark_shader,
            ShaderBinary member_history_shader,ShaderBinary projection_shader,
            ShaderBinary self_collision_shader,ShaderBinary report_copy_shader) {
    Foundation foundation;
    Context context(backend);
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback device(roots,&stream);
    SourcePhysxMappedReportRoots pinned(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    SourcePhysxMappedReportRoots mapped(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* data)noexcept{
        *static_cast<bool*>(data)=true;
    };
    {
        physx::PxgHeapMemoryAllocatorManager heap(
            1u<<20u,device,pinned,mapped,abort,&aborted);
        auto allocators=physx::PxgCreateAllocatorDesc(heap);
        {
            PxgAabbSourceMembersNative aabb(allocators,8u,16u,32u);
            PxgSapSourceMembersNative sap(allocators,32u,16u);
            SourceAggregateGpuNativeStage stage(
                context,std::move(bounds_shader),std::move(mark_shader));
            SourceAggregateMemberHistoryNativeStage history_stage(
                context,std::move(member_history_shader));
            SourceAggregateProjectionNativeStage projection_stage(
                context,std::move(projection_shader));
            SourceAggregateSelfCollisionNativeStage self_stage(
                context,std::move(self_collision_shader));
            SourceAggregateReportCopyNativeStage copy_stage(
                context,std::move(report_copy_shader));
            physx::PxgCudaBuffer new_integer_bounds(
                allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE);
            physx::PxgCudaBuffer old_integer_bounds(
                allocators.deviceAlloc,physx::PxsHeapStats::eBROADPHASE);
            const std::size_t integer_words=std::size_t(aabb.capacity())*6u;
            new_integer_bounds.allocate(integer_words*4u,PX_FL);
            old_integer_bounds.allocate(integer_words*4u,PX_FL);
            std::vector<std::uint32_t> original_new(integer_words);
            std::vector<std::uint32_t> original_old(integer_words);
            for(std::size_t i=0;i<integer_words;++i) {
                const auto handle=static_cast<std::uint32_t>(i/6u);
                const auto axis=static_cast<std::uint32_t>(i%6u);
                original_new[i]=0x80001000u+handle*4u+
                    (axis<3u?axis*4u:0x41u+(axis-3u)*4u);
                original_old[i]=0x80002000u+handle*4u+
                    (axis<3u?axis*4u:0x41u+(axis-3u)*4u);
            }

            alignas(16) std::array<std::uint32_t,4> elements{};
            const auto id=aabb.create_aggregate(
                3u,physx::Bp::getFilterGroup_Statics(),elements.data(),3u,
                PX_INVALID_U32,1u);
            aabb.admit_aggregate_member(
                id,4u,14u,0.25f,
                physx::Bp::getFilterGroup_Dynamics(14u,false),
                elements.data());
            aabb.admit_aggregate_member(
                id,5u,15u,0.5f,
                physx::Bp::getFilterGroup_Dynamics(15u,false),
                elements.data());

            auto geometry=aabb.scene_bounds();
            geometry.bounds[4u]=physx::PxBounds3(
                physx::PxVec3(-2.0f,-1.0f,0.0f),
                physx::PxVec3(1.0f,1.0f,2.0f));
            geometry.bounds[5u]=physx::PxBounds3(
                physx::PxVec3(3.0f,-2.0f,-3.0f),
                physx::PxVec3(4.0f,2.0f,-1.0f));
            aabb.mark_aggregate_member_changed(4u,0.25f);
            aabb.mark_aggregate_member_changed(5u,0.5f);
            // Same-frame admission and removal cancel rather than creating
            // a spurious history-invalidating removed member.
            aabb.admit_aggregate_member(
                id,6u,16u,0.1f,
                physx::Bp::getFilterGroup_Dynamics(16u,false),
                elements.data());
            aabb.remove_aggregate_member(6u);
            require(aabb.actor_lifetime().volume_to_actor[6u]==
                        std::numeric_limits<fyzz::BodyId>::max(),
                    "Canceled aggregate admission retained an unpublished Sc actor");
            const physx::Bp::BpFilter filter(false,false);

            std::uint32_t projection_frame=0u;
            const auto run_source_prepass=[&](
                std::initializer_list<physx::PxU32> added_members,
                std::initializer_list<physx::PxU32> removed_members) {
                const auto arrays=aabb.scene_bounds();
                const physx::Bp::BroadPhaseUpdateData update(
                    nullptr,0u,nullptr,0u,nullptr,0u,
                    arrays.bounds.data(),arrays.groups.data(),
                    arrays.contact_distances.data(),nullptr,
                    static_cast<physx::PxU32>(arrays.bounds.size()),
                    filter,true,true);
                const auto manager=aabb.aggregate_pre_update(
                    update,sap.pre_broadphase_buffers(),stage,addresses,stream);
                require(manager.has_aggregates && manager.aggregate_stage==&stage,
                        "Original AABB manager bypassed aggregate GPU stage");
                const auto source_history=aabb.aggregate_member_history_buffers(
                    new_integer_bounds.getDevicePtr(),
                    old_integer_bounds.getDevicePtr());
                require(source_history.num_added==added_members.size() &&
                        source_history.num_removed==removed_members.size(),
                        "Original AABB manager member-history lists disagree with admission");
                auto& integer_uploads=stream.upload_scratch();
                integer_uploads.reserve(2u);
                integer_uploads.push_back({
                    new_integer_bounds.getDevicePtr(),original_new.data(),
                    integer_words*4u});
                integer_uploads.push_back({
                    old_integer_bounds.getDevicePtr(),original_old.data(),
                    integer_words*4u});
                stream.queue_uploads(integer_uploads);
                const auto before=stream.submission_stats().command_batches;
                pre_update_source_sap_aabb_manager_native(update,manager,stream);
                // The post-SAP stage consumes descriptor pointers, not the
                // swap-sensitive C++ names of the source GPU buffers.
                auto& bp=sap.descriptor();
                bp.newIntegerBounds=reinterpret_cast<
                    decltype(bp.newIntegerBounds)>(
                        static_cast<std::uintptr_t>(new_integer_bounds.getDevicePtr()));
                bp.oldIntegerBounds=reinterpret_cast<
                    decltype(bp.oldIntegerBounds)>(
                        static_cast<std::uintptr_t>(old_integer_bounds.getDevicePtr()));
                append_source_sap_post_aggregate_member_history_native(
                    bp,aabb,history_stage,stream,addresses);
                const auto retirement_buffers=
                    aabb.aggregate_pair_retirement_gpu_buffers();
                require(retirement_buffers.aggregate_pair_capacity==32u &&
                        retirement_buffers.aggregate_pairs!=0u &&
                        retirement_buffers.remove_bitmap!=0u &&
                        retirement_buffers.remove_histogram!=0u &&
                        retirement_buffers.removed_count!=0u,
                        "Original AABB GPU owner lacks persistent aggregate pair retirement storage");
                (void)addresses.resolve(
                    retirement_buffers.remove_bitmap,
                    32u*sizeof(physx::PxU32));
                (void)addresses.resolve(
                    retirement_buffers.remove_histogram,
                    sizeof(physx::PxU32));
                (void)addresses.resolve(
                    retirement_buffers.removed_count,
                    sizeof(physx::PxU32));
                const auto projection_gpu=aabb.aggregate_projection_gpu_buffers(
                    new_integer_bounds.getDevicePtr());
                projection_stage.append(stream,addresses,projection_gpu);
                const auto group_buffer=sap.pre_broadphase_buffers().groups.getDevicePtr();
                const auto collision_gpu=aabb.aggregate_persistent_collision_gpu_buffers(
                    new_integer_bounds.getDevicePtr(),
                    old_integer_bounds.getDevicePtr(),
                    manager.volume_data_device.getDevicePtr(),
                    group_buffer,manager.removed.device.getDevicePtr());
                require(collision_gpu.pair_capacity==32u &&
                        collision_gpu.report_capacity==16u &&
                        collision_gpu.volume_data==manager.volume_data_device.getDevicePtr() &&
                        collision_gpu.current_keys==projection_gpu.sorted_projections &&
                        collision_gpu.current_members==projection_gpu.bound_index_history &&
                        collision_gpu.current_keys!=collision_gpu.previous_keys &&
                        collision_gpu.remove_pair_bitmap==
                            retirement_buffers.remove_bitmap &&
                        collision_gpu.aggregate_pairs==retirement_buffers.aggregate_pairs,
                        "Original persistent aggregate root-pair collision did not borrow source owner");
                (void)addresses.resolve(collision_gpu.volume_data,
                    aabb.capacity()*sizeof(physx::Bp::VolumeData));
                const auto self_gpu=aabb.aggregate_self_collision_gpu_buffers(
                    new_integer_bounds.getDevicePtr(),
                    old_integer_bounds.getDevicePtr(),group_buffer);
                require(self_gpu.report_capacity==16u &&
                        self_gpu.aggregate_count>=1u,
                        "Source AABB owner aggregate report config lost");
                self_stage.append(stream,addresses,self_gpu);
                const auto copied=aabb.aggregate_report_copy_gpu_buffers(mapped);
                copy_stage.append(stream,addresses,copied);
                require(stream.commands().size()==
                            5u+std::uint32_t(added_members.size()!=0u ||
                                             removed_members.size()!=0u),
                        "Original AABB aggregate GPU stage order or member-history dispatch differs");
                stream.wait_at_fetch();
                std::vector<std::uint32_t> new_values(integer_words);
                std::vector<std::uint32_t> old_values(integer_words);
                const auto read_integer=[&](const physx::PxgCudaBuffer& device,
                                            std::vector<std::uint32_t>& output) {
                    const auto view=addresses.resolve(
                        device.getDevicePtr(),integer_words*4u);
                    context.download(view.buffer,view.offset,
                        output.data(),integer_words*4u);
                };
                read_integer(new_integer_bounds,new_values);
                read_integer(old_integer_bounds,old_values);
                for(physx::PxU32 handle=0;handle<aabb.capacity();++handle) {
                    const bool is_added=std::find(
                        added_members.begin(),added_members.end(),handle)!=added_members.end();
                    const bool is_removed=std::find(
                        removed_members.begin(),removed_members.end(),handle)!=removed_members.end();
                    for(physx::PxU32 axis=0;axis<6u;++axis) {
                        const auto word=std::size_t(handle)*6u+axis;
                        const auto empty=axis<3u?0xff7fffffu:0x00800000u;
                        require(new_values[word]==(is_removed?empty:original_new[word]) &&
                                old_values[word]==((is_added||is_removed)?empty:original_old[word]),
                                "Original AABB manager GPU member history differs after source fetch");
                    }
                }
                // The source AABB owner owns all persistent projection
                // histories, not the GPU test fixture.
                const auto download_words=[&](CudaDeviceAddressMap::Address ptr,
                                              std::size_t words) {
                    std::vector<physx::PxU32> result(words);
                    const auto view=addresses.resolve(ptr,words*sizeof(physx::PxU32));
                    context.download(view.buffer,view.offset,
                                     result.data(),words*sizeof(physx::PxU32));
                    return result;
                };
                const auto m=projection_gpu.member_capacity;
                const auto sorted=download_words(
                    projection_gpu.sorted_projections,4u*m);
                const auto sorted_handles=download_words(
                    projection_gpu.sorted_handles,4u*m);
                const auto sap_boxes=download_words(
                    projection_gpu.sap_boxes,4u*m);
                const auto masks=download_words(
                    projection_gpu.start_masks,2u*m);
                const auto member_history=download_words(
                    projection_gpu.bound_index_history,2u*m);
                const auto& aggregate=aabb.source_aggregate(id);
                if(aggregate.size) {
                    SourceAggregateGpuNativeRecord descriptor{};
                    const auto gpu_record=addresses.resolve(
                        projection_gpu.aggregate_slots+
                            std::uint64_t(id)*sizeof(descriptor),sizeof(descriptor));
                    context.download(gpu_record.buffer,gpu_record.offset,
                                     &descriptor,sizeof(descriptor));
                    require(descriptor.root==aggregate.mIndex &&
                            descriptor.size==aggregate.size &&
                            std::uint64_t(descriptor.first_member)+descriptor.size<=m,
                            "Original GPU aggregate slot/member descriptor differs from AABB owner");
                    std::vector<std::pair<physx::PxU32,physx::PxU32>> expected;
                    for(physx::PxU32 local=0;local<aggregate.size;++local) {
                        const auto handle=aggregate.updateBoundIndices[local];
                        expected.emplace_back(new_values[handle*6u],(local<<3u)|1u);
                        expected.emplace_back(new_values[handle*6u+3u],local<<3u);
                    }
                    std::stable_sort(expected.begin(),expected.end(),
                        [](const auto& a,const auto& b){return a.first<b.first;});
                    const auto projection_base=2u*descriptor.first_member;
                    const auto member_base=descriptor.first_member;
                    for(physx::PxU32 j=0;j<expected.size();++j) {
                        if(sorted[projection_base+j]!=expected[j].first ||
                           sorted_handles[projection_base+j]!=expected[j].second)
                            throw std::runtime_error(
                                "Original aggregate projection GPU radix order differs: frame="+
                                std::to_string(projection_frame)+
                                " slot="+std::to_string(projection_gpu.current_slot)+
                                " member="+std::to_string(j)+
                                " key="+std::to_string(sorted[projection_base+j])+
                                " expectedKey="+std::to_string(expected[j].first)+
                                " handle="+std::to_string(sorted_handles[projection_base+j])+
                                " expectedHandle="+std::to_string(expected[j].second));
                        const auto local=expected[j].second>>3u;
                        const bool is_start=(expected[j].second&1u)!=0u;
                        require(sap_boxes[projection_base+local*2u+
                                          (is_start?0u:1u)]==j,
                                "Original aggregate local GPU SAP indices differ");
                    }
                    for(physx::PxU32 local=0;local<aggregate.size;++local)
                        require(member_history[member_base+local]==
                                    aggregate.updateBoundIndices[local],
                                "Original aggregate GPU member history differs");
                    for(physx::PxU32 word=0;word<(expected.size()+31u)/32u;++word) {
                        physx::PxU32 ballot=0u;
                        for(physx::PxU32 bit=0;bit<32u;++bit) {
                            const auto j=word*32u+bit;
                            if(j<expected.size() && (expected[j].second&1u))
                                ballot|=1u<<bit;
                        }
                        require(masks[member_base+word]==ballot,
                                "Original aggregate GPU start masks differ");
                    }
                }
                std::array<physx::PxU32,2> aggregate_counts{};
                {
                    const auto view=addresses.resolve(
                        self_gpu.found_lost_counts,sizeof(aggregate_counts));
                    context.download(view.buffer,view.offset,
                                     aggregate_counts.data(),sizeof(aggregate_counts));
                }
                require(aggregate_counts[0]<=self_gpu.report_capacity &&
                        aggregate_counts[1]<=self_gpu.report_capacity,
                        "Original source aggregate member report overflow");
                const auto reports=aabb.fetch_aggregate_mapped_reports(
                    context,addresses,mapped);
                require(reports.found.size()==aggregate_counts[0] &&
                        reports.lost.size()==aggregate_counts[1],
                        "Source mapped aggregate report copy differs from shared counters");
                if(projection_frame==0u) {
                    require(reports.found.size()==1u &&
                            reports.found[0].mVolA==4u &&
                            reports.found[0].mVolB==5u,
                            "Original mapped report heap did not publish aggregate self-contact");
                    require(aggregate_counts[0]==1u && aggregate_counts[1]==0u,
                        "Source-owned aggregate GPU failed to publish its first self-contact");
                    std::array<physx::PxU32,2> found_member{};
                    const auto view=addresses.resolve(
                        self_gpu.found_pairs,sizeof(found_member));
                    context.download(view.buffer,view.offset,
                                     found_member.data(),sizeof(found_member));
                    require(found_member[0]==4u && found_member[1]==5u,
                        "Original GPU aggregate self-contact uses wrong Sc shape handles");
                }
                require(stream.submission_stats().command_batches==before+1u,
                        "Original AABB source aggregate stages split GPU command batches");
                aabb.complete_fetch(update);
                ++projection_frame;
                const auto next=aabb.aggregate_projection_gpu_buffers(
                    new_integer_bounds.getDevicePtr());
                require(next.current_slot==(projection_gpu.current_slot^1u),
                        "Original aggregate GPU projection history failed to rotate at fetch");
            };
            const auto download_bounds=[&](physx::PxU32 handle) {
                const auto sap_buffers=sap.pre_broadphase_buffers();
                const auto address=sap_buffers.bounds.getDevicePtr()+
                                   std::uint64_t(handle)*sizeof(physx::PxBounds3);
                const auto view=addresses.resolve(address,sizeof(physx::PxBounds3));
                physx::PxBounds3 value;
                context.download(view.buffer,view.offset,&value,sizeof(value));
                return value;
            };
            const auto download_changed=[&] {
                const auto desc=aabb.pre_update(sap.pre_broadphase_buffers());
                const auto ptr=desc.changed.device.getDevicePtr();
                const auto view=addresses.resolve(ptr,sizeof(physx::PxU32));
                physx::PxU32 bits=0u;
                context.download(view.buffer,view.offset,&bits,sizeof(bits));
                return bits;
            };
            const auto verify_bounds=[&](physx::PxBounds3 value,
                                         std::array<float,6> expected) {
                const std::array<float,6> actual{
                    value.minimum.x,value.minimum.y,value.minimum.z,
                    value.maximum.x,value.maximum.y,value.maximum.z};
                for(std::size_t i=0;i<actual.size();++i)
                    require(std::fabs(actual[i]-expected[i])<1e-5f,
                            "Original PhysX AABB aggregate bounds GPU union disagrees");
            };
            run_source_prepass({4u,5u},{});
            verify_bounds(download_bounds(3u),
                          {-2.25f,-2.5f,-3.5f,4.5f,2.5f,2.25f});
            const auto bit=download_changed();
            require((bit&(1u<<3u))!=0u &&
                    (bit&((1u<<4u)|(1u<<5u)))==0u,
                    "Original source aggregate GPU dirty bitmap did not promote root");

            // Grow reserved aggregate membership between GPU frames.
            // The next sort writes slot 1, and the former slot 0 must
            // remain byte-identical after every source heap allocation grows.
            const auto previous_slot=aabb.aggregate_projection_gpu_buffers(
                new_integer_bounds.getDevicePtr(),true);
            const auto sample_previous=[&](
                const SourceAggregateProjectionNativeBuffers& slot) {
                std::array<std::array<physx::PxU32,6>,5> saved{};
                const std::array<CudaDeviceAddressMap::Address,5> pointers{
                    slot.sorted_projections,slot.sorted_handles,
                    slot.sap_boxes,slot.start_masks,slot.bound_index_history};
                for(std::size_t i=0;i<pointers.size();++i) {
                    const auto view=addresses.resolve(
                        pointers[i],sizeof(saved[i]));
                    context.download(view.buffer,view.offset,
                                     saved[i].data(),sizeof(saved[i]));
                }
                return saved;
            };
            const auto before_growth=sample_previous(previous_slot);
            const auto empty_id=aabb.create_aggregate(
                7u,physx::Bp::getFilterGroup_Statics(),elements.data(),8u);

            // The second source frame uses existing membership storage.
            // Bounds and contact-offset changes are still authored by
            // the same AABB owner, not a test-specific aggregate resolver.
            geometry.bounds[5u]=physx::PxBounds3(
                physx::PxVec3(-5.0f,-1.0f,1.0f),
                physx::PxVec3(-3.0f,4.0f,3.0f));
            aabb.mark_aggregate_member_changed(5u,0.75f);
            run_source_prepass({},{});
            verify_bounds(download_bounds(3u),
                          {-5.75f,-1.75f,-0.25f,1.25f,4.75f,3.75f});
            const auto preserved_slot=aabb.aggregate_projection_gpu_buffers(
                new_integer_bounds.getDevicePtr());
            require(preserved_slot.member_capacity>previous_slot.member_capacity,
                    "Aggregate reserved membership did not grow across GPU frames");
            const auto after_growth=sample_previous(preserved_slot);
            require(after_growth==before_growth,
                    "Original previous aggregate projection, handles, SAP boxes, masks or member history was lost on GPU heap growth");
            require((download_changed()&(1u<<3u))!=0u,
                    "Aggregate root changed bit was lost on persistent frame");

            // An already-published member cannot reclaim its retired
            // source shape ID until GPU lost reports have been fetched.
            // Rejected readmission leaves the pending removal intact.
            aabb.remove_aggregate_member(5u);
            require(aabb.actor_lifetime().volume_to_actor[5u]==15u,
                    "Original aggregate lost report discarded its retired actor identity");
            bool rejected_readmission=false;
            try {
                aabb.admit_aggregate_member(
                    id,5u,15u,0.75f,
                    physx::Bp::getFilterGroup_Dynamics(15u,false),
                    elements.data());
            } catch(const std::logic_error&) {
                rejected_readmission=true;
            }
            require(rejected_readmission,
                    "Source retired aggregate member was readmitted before GPU fetch");
            run_source_prepass({},{5u});
            require(aabb.actor_lifetime().volume_to_actor[5u]==
                        std::numeric_limits<fyzz::BodyId>::max(),
                    "Source aggregate retired actor identity survived successful report fetch");
            verify_bounds(download_bounds(3u),
                          {-2.25f,-1.25f,-0.25f,1.25f,1.25f,2.25f});
            require((download_changed()&((1u<<4u)|(1u<<5u)))==0u,
                    "Retired aggregate member leaked GPU changed bits");
            aabb.remove_aggregate_member(4u);
            require(aabb.actor_lifetime().volume_to_actor[4u]==14u,
                    "Original aggregate final member lost its Sc actor before fetch");
            aabb.destroy_aggregate(id);
            aabb.destroy_aggregate(empty_id);
            run_source_prepass({},{4u});
            require(aabb.actor_lifetime().volume_to_actor[4u]==
                        std::numeric_limits<fyzz::BodyId>::max(),
                    "Original final aggregate member actor identity outlived fetch");
            require((download_changed()&(1u<<3u))==0u,
                    "Retired source aggregate root was reintroduced by GPU bitmap");

            stream.wait_at_fetch();
        }
        heap.flushDeferredDeallocs();
    }
    roots.close(&stream);
    require(!aborted,"Original PhysX GPU heap allocation aborted");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::invalid_argument("Expected aggregate bounds, mark, member-history, projection, self-collision and report-copy shaders");
        verify(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12,
#else
            Backend::Vulkan,
#endif
            load(argv[1]),load(argv[2]),load(argv[3]),load(argv[4]),
            load(argv[5]),load(argv[6]));
        std::cout<<"Original PhysX aggregate AABB GPU first/second pass PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original PhysX aggregate AABB GPU prepass FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
