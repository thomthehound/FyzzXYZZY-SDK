// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_endpoint_histograms_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstddef>
#include <cstdint>
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
constexpr std::uint32_t sentinel=0xcdcdcdcdu;
constexpr std::uint32_t projections=8u,orderedSlots=4u,blocks=32u;
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
ShaderBinary read_shader(const char* file) {
    std::ifstream in(file,std::ios::binary);
    if(!in)throw std::runtime_error("Missing source endpoint histogram shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{in},std::istreambuf_iterator<char>{}};
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(bytes.begin(),bytes.end());
    return result;
}
std::size_t count(std::size_t slot) {
    const auto field=slot%16u;
    return field==5u||field==6u||field==15u?blocks:
           field==3u||field==4u||field==10u||field==11u?
               orderedSlots:projections;
}
void run(Context& ctx,const SourceSapEndpointHistogramsNativeStage& stage,
         bool incremental,bool shared) {
    std::array<std::vector<std::uint32_t>,48u> initial{};
    for(std::size_t slot=0;slot<48u;++slot)
        initial[slot].assign(count(slot),sentinel);
    std::array<std::vector<std::uint32_t>,48u> expected=initial;
    for(std::uint32_t axis=0u;axis<3u;++axis){
        for(std::uint32_t frame=0u;frame<2u;++frame){
            const auto base=axis*16u+frame*7u;
            for(std::uint32_t index=0u;index<6u;++index){
                const auto id=index/2u+axis*5u;
                const bool start=(index%2u)==0u;
                const bool deleted=incremental&&index>=4u;
                initial[base][index]=(id<<3u)|(start?1u:0u)|
                    (deleted?4u:0u);
            }
        }
    }
    expected=initial;
    for(std::uint32_t axis=0u;axis<3u;++axis){
        for(std::uint32_t frame=0u;frame<2u;++frame){
            if(frame==1u&&!incremental)continue;
            const auto base=axis*16u+frame*7u;
            std::uint32_t starts=0u,ends=0u,total=0u;
            for(std::uint32_t i=0u;i<6u;++i){
                const auto tagged=initial[base][i];
                const bool start=(tagged&1u)!=0u;
                const bool deleted=(tagged&4u)!=0u;
                expected[base+1u][i]=ends;
                expected[base+2u][i]=starts;
                if(frame==0u)expected[axis*16u+14u][i]=total;
                if(!deleted){
                    expected[base+(start?4u:3u)][start?starts:ends]=tagged>>3u;
                    if(start)++starts;else ++ends;
                }
                if(!start)++total;
            }
            expected[base+5u][0u]=ends;
            expected[base+6u][0u]=starts;
            for(std::uint32_t b=1u;b<blocks;++b){
                expected[base+5u][b]=0u;
                expected[base+6u][b]=0u;
            }
            if(frame==0u){
                expected[axis*16u+15u][0u]=total;
                for(std::uint32_t b=1u;b<blocks;++b)
                    expected[axis*16u+15u][b]=0u;
            }
        }
    }
    CudaDeviceAddressMap map;
    std::array<BufferId,48u> buffer{};
    std::array<std::size_t,48u> offsets{};
    std::array<CudaDeviceAddressMap::Address,48u> addresses{};
    BufferId root=0u;
    if(shared){
        std::size_t total=0u;
        for(std::size_t i=0u;i<48u;++i){
            total=(total+255u)&~std::size_t{255u};
            offsets[i]=total;
            total+=initial[i].size()*4u;
        }
        root=ctx.create_buffer(total,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0u;i<48u;++i){
            buffer[i]=root;
            addresses[i]=map.register_slice(
                root,offsets[i],initial[i].size()*4u);
        }
    }else{
        for(std::size_t i=0u;i<48u;++i){
            const auto size=initial[i].size()*4u;
            buffer[i]=ctx.create_buffer(size,MemoryPlacement::DeviceLocal);
            addresses[i]=map.register_allocation(buffer[i],size);
        }
    }
    for(std::size_t i=0u;i<48u;++i)
        ctx.upload(buffer[i],offsets[i],initial[i].data(),initial[i].size()*4u);
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address a){
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(a));
    };
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles=3u;
    source.numCreatedHandles=incremental?1u:0u;
    source.numRemovedHandles=incremental?1u:0u;
    using Word=std::remove_reference_t<decltype(source.boxHandles[0][0])>;
    for(std::uint32_t axis=0u;axis<3u;++axis){
        const auto base=axis*16u;
        for(std::uint32_t frame=0u;frame<2u;++frame){
            const auto off=base+frame*7u;
            source.boxHandles[frame][axis]=ptr.operator()<Word>(addresses[off]);
            source.endPtHistogram[frame][axis]=ptr.operator()<Word>(addresses[off+1u]);
            source.startPtHistogram[frame][axis]=ptr.operator()<Word>(addresses[off+2u]);
            source.endPointHandles[frame][axis]=ptr.operator()<Word>(addresses[off+3u]);
            source.startPointHandles[frame][axis]=ptr.operator()<Word>(addresses[off+4u]);
            source.blockEndPtHistogram[frame][axis]=ptr.operator()<Word>(addresses[off+5u]);
            source.blockStartPtHistogram[frame][axis]=ptr.operator()<Word>(addresses[off+6u]);
        }
        source.totalEndPtHistogram[axis]=ptr.operator()<Word>(addresses[base+14u]);
        source.blockTotalEndPtHistogram[axis]=ptr.operator()<Word>(addresses[base+15u]);
    }
    SourceSapDmaNative stream(ctx,map);
    auto& cmds=stream.commands();
    auto bad=source;
    bad.blockTotalEndPtHistogram[2u]=ptr.operator()<Word>(
        static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    bool rejected=false;
    try{stage.append(cmds,bad,map,{projections,orderedSlots},incremental);}
    catch(const std::exception&){rejected=true;}
    require(rejected&&cmds.empty(),"Late invalid source histogram pointer queued GPU work");
    rejected=false;
    try{stage.append(cmds,source,map,{projections,2u},incremental);}
    catch(const std::exception&){rejected=true;}
    require(rejected&&cmds.empty(),"Insufficient ordered-handle source capacity accepted");
    stage.append(cmds,source,map,{projections,orderedSlots},incremental);
    require(cmds.size()==2u,"Original endpoint histograms are not two source dispatches");
    for(const auto& command:cmds){
        const auto& d=std::get<DispatchCommand>(command);
        require(d.groups.x==32u&&d.groups.y==3u&&d.buffers.size()==48u,
                "Original 32x3 endpoint grid or source bindings changed");
    }
    stream.wait_at_fetch();
    require(stream.submission_stats().command_batches==1u,
            "Original endpoint histogram kernels split source GPU submission");
    for(std::size_t i=0u;i<48u;++i){
        std::vector<std::uint32_t> observed(initial[i].size());
        ctx.download(buffer[i],offsets[i],observed.data(),observed.size()*4u);
        require(observed==expected[i],
                "Source endpoint histogram or previous-frame state disagrees with reference");
    }
    if(shared)ctx.destroy_buffer(root);
    else for(const auto id:buffer)ctx.destroy_buffer(id);
}
}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::invalid_argument("Require both source histogram shader paths");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapEndpointHistogramsNativeStage stage(
            context,read_shader(argv[1]),read_shader(argv[2]));
        run(context,stage,true,false);
        run(context,stage,false,true);
        std::cout<<"Source-owned endpoint histogram GPU sequence PASS\n";
    }catch(const std::exception& e){
        std::cerr<<"Source-owned endpoint histograms FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
