// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_gpu/source_aggregate_gpu_native.h"
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
#include <stdexcept>
#include <vector>

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
void verify(Backend backend,ShaderBinary bounds_shader,ShaderBinary mark_shader) {
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
            PxgAabbSourceMembersNative aabb(allocators,8u);
            PxgSapSourceMembersNative sap(allocators,32u,16u);
            SourceAggregateGpuNativeStage stage(
                context,std::move(bounds_shader),std::move(mark_shader));

            alignas(16) std::array<std::uint32_t,4> elements{};
            const auto id=aabb.create_aggregate(
                3u,physx::Bp::getFilterGroup_Statics(),elements.data(),3u);
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
            const physx::Bp::BpFilter filter(false,false);

            const auto run_source_prepass=[&] {
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
                const auto before=stream.submission_stats().command_batches;
                pre_update_source_sap_aabb_manager_native(update,manager,stream);
                require(stream.commands().size()==2u,
                        "Original AABB aggregate GPU stage did not append both source kernels");
                stream.wait_at_fetch();
                require(stream.submission_stats().command_batches==before+1u,
                        "Original AABB source aggregate stages split GPU command batches");
                aabb.complete_fetch(update);
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
            run_source_prepass();
            verify_bounds(download_bounds(3u),
                          {-2.25f,-2.5f,-3.5f,4.5f,2.5f,2.25f});
            const auto bit=download_changed();
            require((bit&(1u<<3u))!=0u &&
                    (bit&((1u<<4u)|(1u<<5u)))==0u,
                    "Original source aggregate GPU dirty bitmap did not promote root");

            // The second source frame uses existing membership storage.
            // Bounds and contact-offset changes are still authored by
            // the same AABB owner, not a test-specific aggregate resolver.
            geometry.bounds[5u]=physx::PxBounds3(
                physx::PxVec3(-5.0f,-1.0f,1.0f),
                physx::PxVec3(-3.0f,4.0f,3.0f));
            aabb.mark_aggregate_member_changed(5u,0.75f);
            run_source_prepass();
            verify_bounds(download_bounds(3u),
                          {-5.75f,-1.75f,-0.25f,1.25f,4.75f,3.75f});
            require((download_changed()&(1u<<3u))!=0u,
                    "Aggregate root changed bit was lost on persistent frame");

            aabb.remove_aggregate_member(5u);
            run_source_prepass();
            verify_bounds(download_bounds(3u),
                          {-2.25f,-1.25f,-0.25f,1.25f,1.25f,2.25f});
            require((download_changed()&((1u<<4u)|(1u<<5u)))==0u,
                    "Retired aggregate member leaked GPU changed bits");
            aabb.remove_aggregate_member(4u);
            aabb.destroy_aggregate(id);
            run_source_prepass();
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
        if(argc!=3)throw std::invalid_argument("Expected source aggregate bounds/mark GPU shaders");
        verify(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12,
#else
            Backend::Vulkan,
#endif
            load(argv[1]),load(argv[2]));
        std::cout<<"Original PhysX aggregate AABB GPU first/second pass PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original PhysX aggregate AABB GPU prepass FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
