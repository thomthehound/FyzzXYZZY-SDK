// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_region_histograms_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
constexpr std::uint32_t kCanary=0xadafafafu;
void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
template<class F> void reject(F&& f,const char* why){
    bool raised=false;
    try{f();}catch(const std::exception&){raised=true;}
    check(raised,why);
}
ShaderBinary shader(const char* filename){
    std::ifstream stream(filename,std::ios::binary);
    if(!stream)throw std::runtime_error("Source region scan shader missing");
    const std::vector<char> content{std::istreambuf_iterator<char>{stream},
                                    std::istreambuf_iterator<char>{}};
    ShaderBinary data;
#if defined(FYZZXYZZY_TEST_D3D12)
    data.format=ShaderFormat::Dxil;
#else
    data.format=ShaderFormat::Spirv;
#endif
    data.bytes.assign(content.begin(),content.end());
    return data;
}
struct Allocation {
    std::uint32_t root=0, offset=0, words=0;
    CudaDeviceAddressMap::Address cuda=0;
};
struct Heap {
    BufferId id=0;
    std::vector<std::uint32_t> initial, expected;
};
void run_case(Context& ctx,const SourceRegionHistogramsNativeStage& stage,
              std::uint32_t boxes,std::uint32_t created,
              std::uint32_t removed,bool shared){
    // PhysX retains removed projections until the update's buffer swap.
    const auto live=boxes*2u;
    const auto retained=(boxes+removed)*2u;
    const auto slots=(retained+3u)&~3u;
    const std::array<std::uint32_t,5> sizes{
        slots,slots,32u,32u,
        static_cast<std::uint32_t>((sizeof(physx::PxgBroadPhaseDesc)+3u)/4u)};
    std::vector<Heap> roots(shared?1u:5u);
    std::array<Allocation,5> allocations{};
    for(std::uint32_t i=0;i<5u;++i){
        Allocation& entry=allocations[i];
        entry.root=shared?0u:i;
        auto& root=roots[entry.root];
        root.initial.insert(root.initial.end(),65u+i%5u,kCanary);
        entry.offset=static_cast<std::uint32_t>(root.initial.size());
        entry.words=sizes[i];
        root.initial.insert(root.initial.end(),entry.words,kCanary);
        root.initial.insert(root.initial.end(),16u,kCanary);
    }
    CudaDeviceAddressMap addresses;
    for(auto& root:roots){
        root.id=ctx.create_buffer(root.initial.size()*4u,MemoryPlacement::DeviceLocal);
    }
    for(auto& entry:allocations)
        entry.cuda=addresses.register_slice(roots[entry.root].id,
            std::size_t(entry.offset)*4u, std::size_t(entry.words)*4u);
    const auto at=[&](const Allocation& item,std::uint32_t i)->std::uint32_t&{
        return roots[item.root].initial[item.offset+i];
    };
    for(std::uint32_t i=0;i<live;++i){
        at(allocations[0],i)=((i*17u+i/7u)%9u);
        at(allocations[1],i)=((i*19u+i/11u)%17u);
    }
    for(std::uint32_t i=0;i<32u;++i){
        at(allocations[2],i)=0xfefefefeu;
        at(allocations[3],i)=0xfefefefeu;
    }
    physx::PxgBroadPhaseDesc desc{};
    desc.numPreviousHandles=boxes-created+removed;
    desc.numCreatedHandles=created;
    desc.numRemovedHandles=removed;
    desc.startRegionAccum=reinterpret_cast<decltype(desc.startRegionAccum)>(
        static_cast<std::uintptr_t>(allocations[0].cuda));
    desc.regionAccum=reinterpret_cast<decltype(desc.regionAccum)>(
        static_cast<std::uintptr_t>(allocations[1].cuda));
    desc.blockStartRegionAccum=reinterpret_cast<decltype(desc.blockStartRegionAccum)>(
        static_cast<std::uintptr_t>(allocations[2].cuda));
    desc.blockRegionAccum=reinterpret_cast<decltype(desc.blockRegionAccum)>(
        static_cast<std::uintptr_t>(allocations[3].cuda));
    desc.startRegionAccumTotal=0xbadbadbau;
    desc.regionAccumTotal=0xbadbadbau;
    static_assert(sizeof(physx::PxgBroadPhaseDesc)%4u==0u,
                  "Source descriptor byte size must be DWORD aligned");
    auto& descRoot=roots[allocations[4].root];
    std::memcpy(&descRoot.initial[allocations[4].offset],&desc,sizeof(desc));
    for(auto& root:roots)root.expected=root.initial;
    const auto expectedAt=[&](const Allocation& item,std::uint32_t i)->std::uint32_t&{
        return roots[item.root].expected[item.offset+i];
    };
    if(created){
        // Separate CPU reference: compute each contiguous source block's
        // exclusive prefix, publish 32 local counts, then add preceding
        // block totals to each value. Never run the GPU scan algorithm here.
        const auto iter=((live+255u)/256u+31u)/32u;
        const auto span=iter*256u;
        for(std::uint32_t kind=0;kind<2u;++kind){
            const auto& data=allocations[kind];
            const auto& blocks=allocations[kind+2u];
            std::uint32_t running=0u;
            for(std::uint32_t group=0u;group<32u;++group){
                const auto begin=group*span;
                const auto end=std::min(live,begin+span);
                std::uint32_t local=0u;
                for(auto i=begin;i<end;++i){
                    const auto original=expectedAt(data,i);
                    expectedAt(data,i)=running+local;
                    local+=original;
                }
                expectedAt(blocks,group)=local;
                running+=local;
            }
            const auto scalar=kind==0u ?
                offsetof(physx::PxgBroadPhaseDesc,startRegionAccumTotal) :
                offsetof(physx::PxgBroadPhaseDesc,regionAccumTotal);
            expectedAt(allocations[4],static_cast<std::uint32_t>(scalar/4u))=running;
        }
    }

    const SourceRegionHistogramsNativeLayout layout{
        allocations[4].cuda,slots};
    std::vector<BatchCommand> ops;
    if(created){
        // A bad LAST source resource cannot append partial prefix work.
        auto bad=desc;
        bad.blockRegionAccum=reinterpret_cast<decltype(bad.blockRegionAccum)>(
            std::uintptr_t{0xdeadbeef0ull});
        reject([&]{stage.append(ops,bad,addresses,layout);},
               "Invalid late region histogram pointer was accepted");
        check(ops.empty(),"Late source error appended a partial histogram chain");
        reject([&]{stage.append(ops,desc,addresses,
                {allocations[4].cuda,live-1u});},
               "Undersized PhysX native projection capacity was accepted");
        check(ops.empty(),"Short capacity appended partial histogram commands");
    }
    stage.append(ops,desc,addresses,layout);
    check(ops.size()==(created?4u:0u),
          "Original PhysX four histogram launch gate or count changed");
    if(created){
        for(std::uint32_t i=0u;i<4u;++i){
            const auto& c=std::get<DispatchCommand>(ops[i]);
            check(c.groups.x==32u&&c.groups.y==1u&&c.groups.z==1u&&
                  c.buffers.size()==(i%2u==0u?2u:3u)&&
                  c.push_constants.size()==32u,
                  "PhysX source histogram native GPU launch shape changed");
            check(c.buffers[0].access==BufferAccess::ReadWrite&&
                  c.buffers[1].access==(i%2u==0u?
                      BufferAccess::Write:BufferAccess::Read)&&
                  (i%2u==0u || c.buffers[2].access==BufferAccess::Write),
                  "PhysX source histogram read/write intent changed");
        }
    }
    std::vector<UploadSpan> uploads;
    for(const auto& root:roots)
        uploads.push_back({root.id,0u,root.initial.data(),root.initial.size()*4u});
    ctx.submit_upload_batch(uploads);
    if(created)ctx.wait_submission(ctx.submit_batch(ops));
    for(const auto& root:roots){
        std::vector<std::uint32_t> actual(root.initial.size());
        ctx.download(root.id,0u,actual.data(),actual.size()*4u);
        if(actual!=root.expected){
            const auto diff=std::mismatch(actual.begin(),actual.end(),root.expected.begin());
            throw std::runtime_error("PhysX source region prefix or untouched heap mismatch: boxes "+
                std::to_string(boxes)+", created "+std::to_string(created)+
                ", shared "+std::to_string(shared)+", first word "+
                std::to_string(std::distance(actual.begin(),diff.first)));
        }
    }
    for(const auto& root:roots)ctx.destroy_buffer(root.id);
}
} // namespace

int main(int argc,char** argv){
    try{
        if(argc!=4)throw std::invalid_argument("Expected portable/wave/second native region shaders");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& gpu=ctx.device_info();
        if(gpu.max_workgroup_invocations<256u ||
           gpu.max_workgroup_size[0]<256u ||
           gpu.max_workgroup_count[0]<32u){
            std::cout<<"SKIP: PhysX source 32x256 region scan unavailable\n";
            return 77;
        }
        SourceRegionHistogramsNativeStage optimal(
            ctx,shader(argv[1]),shader(argv[2]),shader(argv[3]));
        SourceRegionHistogramsNativeStage portable(
            ctx,shader(argv[1]),shader(argv[2]),shader(argv[3]),true);
        const bool expectedWave=gpu.supports_wave_arithmetic &&
            gpu.subgroup_size>=4u && 256u%gpu.subgroup_size==0u;
        check(optimal.using_subgroup_scan()==expectedWave&&
              !portable.using_subgroup_scan(),
              "Source native region prefix GPU wave admission changed");
        for(const auto* stage:{&optimal,&portable}){
            run_case(ctx,*stage,5u,0u,0u,false);
            run_case(ctx,*stage,1u,1u,0u,false);
            run_case(ctx,*stage,5u,1u,2u,false);
            run_case(ctx,*stage,129u,3u,4u,false);
            run_case(ctx,*stage,257u,257u,0u,false);
            run_case(ctx,*stage,4099u,11u,0u,false);
            run_case(ctx,*stage,8201u,19u,0u,false);
            run_case(ctx,*stage,5u,1u,2u,true);
            run_case(ctx,*stage,129u,3u,4u,true);
        }
        std::cout<<"Original PhysX source-owned four region histograms PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"PhysX source region histogram FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
