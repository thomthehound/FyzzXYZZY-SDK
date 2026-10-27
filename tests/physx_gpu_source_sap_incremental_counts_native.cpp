// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_incremental_counts_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
constexpr std::uint32_t kBoxes=3u,kEndpoints=8u,kBlocks=32u;
constexpr std::uint32_t kSentinel=0xcdcdcdcdu;
void require(bool ok,const char* what){
    if(!ok)throw std::runtime_error(what);
}
ShaderBinary shader(const char* path){
    std::ifstream stream(path,std::ios::binary);
    if(!stream)throw std::runtime_error("Missing source comparison shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{stream},std::istreambuf_iterator<char>{}};
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(bytes.begin(),bytes.end());
    return result;
}
void run(Context& context,const SourceSapIncrementalCountsNativeStage& stage,
         bool shared) {
    std::array<std::vector<std::uint32_t>,24> initial{};
    std::array<std::vector<std::uint32_t>,24> expected{};
    for(std::uint32_t axis=0;axis<3u;++axis){
        const auto base=axis*8u;
        for(std::uint32_t i=0;i<8u;++i)
            initial[base+i].assign(i==7u?kBlocks:kEndpoints,kSentinel);
        const std::array<std::uint32_t,6> ranks{3u,0u,2u,5u,4u,1u};
        for(std::uint32_t j=0;j<6u;++j){
            initial[base+0u][j]=(j<<3u)|((j%2u)?1u:0u)|
                (j==2u?4u:0u);
            initial[base+1u][j]=ranks[j];
            initial[base+2u][j]=3u*j+axis;
            initial[base+3u][j]=2u*j+axis;
            initial[base+4u][j]=4u*j+axis;
            initial[base+5u][j]=5u*j+axis;
        }
        expected[base+6u]=initial[base+6u];
        expected[base+7u]=initial[base+7u];
        std::uint32_t total=0u;
        for(std::uint32_t j=0;j<6u;++j){
            const auto rank=ranks[j];
            const bool down=j<=rank;
            const bool start=(initial[base][j]&1u)!=0u;
            const bool removed=(initial[base][j]&4u)!=0u;
            const auto field=down?(start?5u:4u):(start?3u:2u);
            const auto low=down?j:rank+1u;
            const auto high=down?rank:j;
            expected[base+6u][j]=total;
            if(!removed)
                total+=initial[base+field][high]-initial[base+field][low];
        }
        expected[base+7u][0]=total;
        for(std::uint32_t block=1;block<kBlocks;++block)
            expected[base+7u][block]=0u;
    }
    CudaDeviceAddressMap addresses;
    std::array<BufferId,25> buffers{};
    std::array<std::size_t,25> offsets{};
    std::array<CudaDeviceAddressMap::Address,25> pointers{};
    BufferId root=0u;
    const auto bytes=[](std::size_t i){
        return i==24u?sizeof(physx::PxgBroadPhaseDesc):
            (i%8u==7u?kBlocks:kEndpoints)*sizeof(std::uint32_t);
    };
    if(shared){
        std::size_t space=0;
        for(std::size_t i=0;i<25u;++i){
            space=(space+255u)&~std::size_t{255u};
            offsets[i]=space;
            space+=bytes(i);
        }
        root=context.create_buffer(space,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<25u;++i){
            buffers[i]=root;
            pointers[i]=addresses.register_slice(root,offsets[i],bytes(i));
        }
    }else{
        for(std::size_t i=0;i<25u;++i){
            buffers[i]=context.create_buffer(bytes(i),MemoryPlacement::DeviceLocal);
            pointers[i]=addresses.register_allocation(buffers[i],bytes(i));
        }
    }
    for(std::size_t i=0;i<24u;++i)
        context.upload(buffers[i],offsets[i],initial[i].data(),bytes(i));
    physx::PxgBroadPhaseDesc desc{};
    desc.numPreviousHandles=kBoxes;
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address a){
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(a));
    };
    for(std::uint32_t axis=0u;axis<3u;++axis){
        const auto base=8u*axis;
        using U=std::remove_reference_t<decltype(desc.boxHandles[0][0])>;
        desc.boxHandles[0][axis]=ptr.operator()<U>(pointers[base]);
        desc.boxProjectionRanks[axis]=ptr.operator()<U>(pointers[base+1u]);
        desc.startPtHistogram[0][axis]=ptr.operator()<U>(pointers[base+2u]);
        desc.endPtHistogram[0][axis]=ptr.operator()<U>(pointers[base+3u]);
        desc.startPtHistogram[1][axis]=ptr.operator()<U>(pointers[base+4u]);
        desc.endPtHistogram[1][axis]=ptr.operator()<U>(pointers[base+5u]);
        desc.incrementalComparisons[axis]=ptr.operator()<U>(pointers[base+6u]);
        desc.incrementalBlockComparisons[axis]=ptr.operator()<U>(pointers[base+7u]);
    }
    context.upload(buffers[24],offsets[24],&desc,sizeof(desc));
    SourceSapDmaNative stream(context,addresses);
    auto& commands=stream.commands();
    auto bad=desc;
    bad.incrementalBlockComparisons[2]=ptr.operator()<
        std::remove_reference_t<decltype(desc.incrementalBlockComparisons[2])>>(
            static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    bool rejected=false;
    try{stage.append(commands,bad,pointers[24],addresses,{kEndpoints});}
    catch(const std::exception&){rejected=true;}
    require(rejected&&commands.empty(),"Late invalid source histogram partially queued");
    rejected=false;
    try{stage.append(commands,desc,pointers[24],addresses,{2u});}
    catch(const std::exception&){rejected=true;}
    require(rejected&&commands.empty(),"Insufficient original endpoint capacity accepted");
    rejected=false;
    try{stage.append(commands,desc,0x1234u,addresses,{kEndpoints});}
    catch(const std::exception&){rejected=true;}
    require(rejected&&commands.empty(),"Unmapped original GPU descriptor accepted");
    stage.append(commands,desc,pointers[24],addresses,{kEndpoints});
    require(commands.size()==2u,"Original incremental comparison stage count changed");
    for(const auto& command:commands){
        const auto& d=std::get<DispatchCommand>(command);
        require(d.groups.x==32u&&d.groups.y==1u&&d.buffers.size()==25u,
                "Original comparison GPU launch or buffer ABI changed");
    }
    stream.wait_at_fetch();
    require(stream.submission_stats().command_batches==1u,
            "Source comparison kernels needlessly split GPU submissions");
    for(std::uint32_t axis=0u;axis<3u;++axis){
        const auto base=axis*8u;
        for(const auto slot:{6u,7u}){
            std::vector<std::uint32_t> observed(initial[base+slot].size());
            context.download(buffers[base+slot],offsets[base+slot],
                observed.data(),observed.size()*4u);
            require(observed==expected[base+slot],
                    "Source incremental comparison prefix differs from reference");
        }
    }
    std::array<std::uint32_t,3> observed{};
    context.download(buffers[24],offsets[24]+
        offsetof(physx::PxgBroadPhaseDesc,totalIncrementalComparisons),
        observed.data(),sizeof(observed));
    for(std::uint32_t axis=0;axis<3u;++axis){
        require(observed[axis]==expected[axis*8u+7u][0],
                "Original device descriptor comparison total not published");
    }
    if(shared)context.destroy_buffer(root);
    else for(const auto buffer:buffers)context.destroy_buffer(buffer);
}
}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::invalid_argument("Require source incremental comparison shaders");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapIncrementalCountsNativeStage stage(
            context,shader(argv[1]),shader(argv[2]));
        run(context,stage,false);
        run(context,stage,true);
        std::cout<<"Source GPU incremental comparison prefixes PASS\n";
    }catch(const std::exception& e){
        std::cerr<<"Source GPU incremental comparison prefixes FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
