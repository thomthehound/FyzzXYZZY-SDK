// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_region_create_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
constexpr std::uint32_t kCanary=0xcdcdcdcd;

void check(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
template<class F> void reject(F&& f,const char* message) {
    bool caught=false;
    try{f();}catch(const std::exception&){caught=true;}
    check(caught,message);
}
ShaderBinary load(const char* path){
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Missing PhysX source-region shader");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{file},
                                  std::istreambuf_iterator<char>{}};
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(),bytes.end());
    return shader;
}
struct Root {
    BufferId buffer=0;
    CudaDeviceAddressMap::Address address=0;
    std::uint32_t offset=0u;
    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> expected;
};
struct Endpoint {std::uint32_t key, handle, tag;};

void exercise(Context& ctx,SourceRegionCreateNativeStage& stage,
              std::uint32_t boxes,std::uint32_t created){
    CudaDeviceAddressMap map;
    std::vector<Root> roots;
    roots.reserve(13u);
    auto allocate=[&](std::uint32_t words)->std::size_t{
        const auto ordinal=static_cast<std::uint32_t>(roots.size());
        const std::uint32_t offset=260u+((ordinal%5u)*4u);
        const std::size_t total=offset/4u+words+16u;
        Root item;
        item.offset=offset;
        item.input.resize(total,kCanary);
        item.buffer=ctx.create_buffer(total*4u,MemoryPlacement::DeviceLocal);
        item.address=map.register_slice(item.buffer,offset,words*4u);
        roots.push_back(std::move(item));
        return roots.size()-1u;
    };
    const auto projections=boxes*2u;
    const auto slots=(projections+3u)&~3u;
    const auto box_capacity=boxes+7u;
    const auto cleared=(boxes*2u)*64u;
    const auto handles=allocate(slots);
    std::array<std::size_t,3> sap{},ends{};
    for(std::uint32_t axis=0;axis<3u;++axis){
        sap[axis]=allocate(box_capacity*2u);
        ends[axis]=allocate(slots);
    }
    const auto ranges=allocate(box_capacity*8u);
    const auto starts=allocate(slots);
    const auto regions=allocate(slots);
    const auto active=allocate(cleared);
    const auto startHist=allocate(cleared);

    auto set=[&](std::size_t root,std::uint32_t word,std::uint32_t val) {
        roots[root].input[roots[root].offset/4u+word]=val;
    };
    auto get=[&](std::size_t root,std::uint32_t word)->std::uint32_t {
        return roots[root].input[roots[root].offset/4u+word];
    };
    for(std::uint32_t i=0;i<cleared;++i) {
        set(active,i,0xfeedcafeu);
        set(startHist,i,0xfeedcafeu);
    }
    physx::PxgBroadPhaseDesc desc{};
    desc.numPreviousHandles=0u;
    desc.numCreatedHandles=created;
    desc.numRemovedHandles=0u;
    const auto ptr=[&](std::size_t idx){return static_cast<std::uintptr_t>(roots[idx].address);};
    desc.boxHandles[0][0]=reinterpret_cast<decltype(desc.boxHandles[0][0])>(ptr(handles));
    for(std::uint32_t a=0;a<3u;++a){
        desc.boxSapBox1D[a]=reinterpret_cast<decltype(desc.boxSapBox1D[a])>(ptr(sap[a]));
        desc.totalEndPtHistogram[a]=reinterpret_cast<decltype(desc.totalEndPtHistogram[a])>(ptr(ends[a]));
    }
    desc.regionRange=reinterpret_cast<decltype(desc.regionRange)>(ptr(ranges));
    desc.startRegionAccum=reinterpret_cast<decltype(desc.startRegionAccum)>(ptr(starts));
    desc.regionAccum=reinterpret_cast<decltype(desc.regionAccum)>(ptr(regions));
    desc.activeRegionsHistogram=reinterpret_cast<decltype(desc.activeRegionsHistogram)>(ptr(active));
    desc.startRegionsHistogram=reinterpret_cast<decltype(desc.startRegionsHistogram)>(ptr(startHist));

    for(std::uint32_t a=0u;a<3u;++a){
        std::vector<Endpoint> list;
        list.reserve(projections);
        for(std::uint32_t h=0u;h<boxes;++h){
            const auto low=((h*(13u+a*8u)+(h/17u)*7u+a*11u)%101u)*8u;
            const auto high=low+((h+a*5u)%19u+1u)*8u+1u;
            const auto flag=(h%9u==0u)?2u:0u;
            list.push_back({low,h,(h<<3u)|flag|1u});
            list.push_back({high,h,(h<<3u)|flag});
        }
        std::stable_sort(list.begin(),list.end(),
            [](const Endpoint& x,const Endpoint& y){return x.key<y.key;});
        std::uint32_t endCount=0u;
        for(std::uint32_t i=0u;i<list.size();++i){
            const auto& e=list[i];
            const auto isEnd=(e.tag&1u)==0u;
            set(sap[a],e.handle*2u+(isEnd?1u:0u),i);
            set(ends[a],i,endCount);
            if(isEnd)++endCount;
            if(a==0u)set(handles,i,e.tag);
        }
        check(endCount==boxes,"Malformed source SAP fixture");
    }
    for(auto& root:roots)root.expected=root.input;
    auto setExpected=[&](std::size_t root,std::uint32_t word,std::uint32_t val){
        roots[root].expected[roots[root].offset/4u+word]=val;
    };
    if(created){
        for(std::uint32_t i=0u;i<cleared;++i){
            setExpected(active,i,0u);
            setExpected(startHist,i,0u);
        }
        const auto perRegion=(boxes+3u)/4u;
        for(std::uint32_t h=0u;h<boxes;++h){
            std::array<std::uint32_t,3> first{},last{};
            for(std::uint32_t a=0u;a<3u;++a){
                first[a]=get(ends[a],get(sap[a],h*2u))/perRegion;
                last[a]=get(ends[a],get(sap[a],h*2u+1u))/perRegion;
                setExpected(ranges,h*8u+a,first[a]);
                setExpected(ranges,h*8u+4u+a,last[a]);
            }
            setExpected(ranges,h*8u+3u,0u);
            setExpected(ranges,h*8u+7u,0u);
            const auto yz=(last[1]-first[1]+1u)*(last[2]-first[2]+1u);
            const auto startX=get(sap[0],h*2u);
            const auto endX=get(sap[0],h*2u+1u);
            setExpected(starts,startX,yz);
            setExpected(starts,endX,0u);
            setExpected(regions,startX,(last[0]-first[0]+1u)*yz);
            setExpected(regions,endX,0u);
        }
    }

    const SourceRegionCreateNativeLayout layout{box_capacity,slots};
    std::vector<BatchCommand> operations;
    if(created){
        // A bad last output pointer cannot commit the two earlier fills.
        auto bad=desc;
        bad.regionAccum=reinterpret_cast<decltype(bad.regionAccum)>(std::uintptr_t{0x1234});
        reject([&]{stage.append(operations,bad,map,layout);},
               "Invalid final BP_CREATE_REGIONS source buffer accepted");
        check(operations.empty(),"Source region creation partially appended prior clears");
        reject([&]{stage.append(operations,desc,map,{box_capacity,projections-1u});},
               "Short original projection allocation accepted");
        check(operations.empty(),"Invalid original source layout appended work");
    }else{
        auto bad=desc;
        bad.activeRegionsHistogram=nullptr;
        stage.append(operations,bad,map,{});
        check(operations.empty(),"Source create-region no-created gate dispatched work");
    }
    stage.append(operations,desc,map,layout);
    check(operations.size()==(created?3u:0u),
          "Original PhysX two-clears plus create-regions launch order changed");
    if(created){
        const auto& dispatch=std::get<DispatchCommand>(operations.back());
        check(dispatch.groups.x==32u&&dispatch.buffers.size()==10u,
              "Source create-region 32x256 launch or ten source views changed");
        for(std::uint32_t i=0u;i<7u;++i)
            check(dispatch.buffers[i].access==BufferAccess::Read,
                  "Source create-regions input unexpectedly writable");
        for(std::uint32_t i=7u;i<10u;++i)
            check(dispatch.buffers[i].access==BufferAccess::Write,
                  "Source create-regions output access changed");
    }
    std::vector<UploadSpan> uploads;
    for(const auto& root:roots)
        uploads.push_back({root.buffer,0u,root.input.data(),root.input.size()*4u});
    ctx.submit_upload_batch(uploads);
    if(!operations.empty())ctx.wait_submission(ctx.submit_batch(operations));
    std::vector<std::uint32_t> actual;
    for(const auto& root:roots) {
        actual.resize(root.input.size());
        ctx.download(root.buffer,0u,actual.data(),actual.size()*4u);
        if(actual!=root.expected){
            const auto mismatch=std::mismatch(actual.begin(),actual.end(),root.expected.begin());
            throw std::runtime_error("Source create-region output/canary mismatch at root "+
                std::to_string(&root-&roots[0])+" word "+
                std::to_string(std::distance(actual.begin(),mismatch.first))+
                " for "+std::to_string(boxes)+" boxes");
        }
    }
    // The source buddy allocator can place all ten producer inputs/outputs
    // and both clear targets inside ONE native GPU resource. D3D12 must
    // bind reads as UAVs too: mixed SRV/UAV states on this heap root would
    // be invalid even though the logical CUDA suballocations do not overlap.
    if(created && boxes==5u){
        CudaDeviceAddressMap sharedMap;
        std::vector<std::uint32_t> initial(73u,kCanary), wanted(73u,kCanary);
        std::vector<CudaDeviceAddressMap::Address> alias(roots.size());
        std::vector<std::uint32_t> placement(roots.size());
        for(std::size_t i=0;i<roots.size();++i){
            const auto words=static_cast<std::uint32_t>(
                roots[i].input.size()-roots[i].offset/4u-16u);
            const auto position=static_cast<std::uint32_t>(initial.size()+5u+i%3u);
            initial.resize(position+words+17u,kCanary);
            wanted.resize(position+words+17u,kCanary);
            placement[i]=position*4u;
            const auto old=roots[i].offset/4u;
            for(std::uint32_t k=0u;k<words;++k){
                initial[position+k]=roots[i].input[old+k];
                wanted[position+k]=roots[i].expected[old+k];
            }
        }
        const auto backing=ctx.create_buffer(initial.size()*4u,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<roots.size();++i) {
            const auto words=static_cast<std::uint32_t>(
                roots[i].input.size()-roots[i].offset/4u-16u);
            alias[i]=sharedMap.register_slice(backing,placement[i],words*4u);
        }
        auto shared=desc;
        const auto ptr2=[&](std::size_t i){
            return static_cast<std::uintptr_t>(alias[i]);};
        shared.boxHandles[0][0]=reinterpret_cast<decltype(shared.boxHandles[0][0])>(ptr2(handles));
        for(std::uint32_t a=0u;a<3u;++a){
            shared.boxSapBox1D[a]=reinterpret_cast<decltype(shared.boxSapBox1D[a])>(ptr2(sap[a]));
            shared.totalEndPtHistogram[a]=reinterpret_cast<decltype(shared.totalEndPtHistogram[a])>(ptr2(ends[a]));
        }
        shared.regionRange=reinterpret_cast<decltype(shared.regionRange)>(ptr2(ranges));
        shared.startRegionAccum=reinterpret_cast<decltype(shared.startRegionAccum)>(ptr2(starts));
        shared.regionAccum=reinterpret_cast<decltype(shared.regionAccum)>(ptr2(regions));
        shared.activeRegionsHistogram=reinterpret_cast<decltype(shared.activeRegionsHistogram)>(ptr2(active));
        shared.startRegionsHistogram=reinterpret_cast<decltype(shared.startRegionsHistogram)>(ptr2(startHist));
        std::vector<BatchCommand> aliasedCommands;
        stage.append(aliasedCommands,shared,sharedMap,layout);
        check(aliasedCommands.size()==3u,
              "Source buddy-heap aliased region sequence lost dispatches");
        const auto& producer=std::get<DispatchCommand>(aliasedCommands.back());
        for(const auto& binding:producer.buffers)
            check(binding.buffer==backing,
                  "Source buddy-heap allocations silently used different roots");
        ctx.submit_upload_buffer(backing,0u,initial.data(),initial.size()*4u);
        ctx.wait_submission(ctx.submit_batch(aliasedCommands));
        std::vector<std::uint32_t> observed(initial.size());
        ctx.download(backing,0u,observed.data(),observed.size()*4u);
        check(observed==wanted,
              "Aliased PhysX source heap-root region contents or canaries changed");
        ctx.destroy_buffer(backing);
    }
    for(auto& root:roots)ctx.destroy_buffer(root.buffer);
}
} // namespace

int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::invalid_argument("Expected source fill and region-create shaders");
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
            std::cout<<"SKIP: original PhysX 32x256 create-regions geometry unsupported\n";
            return 77;
        }
        SourceFillD32NativeStage fill(ctx,load(argv[1]));
        SourceRegionCreateNativeStage stage(ctx,load(argv[2]),fill);
        exercise(ctx,stage,5u,0u);
        for(const auto n:{1u,5u,129u,257u})
            exercise(ctx,stage,n,n);
        std::cout<<"Original PhysX clears and BP_CREATE_REGIONS source-native PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Source region creation FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
