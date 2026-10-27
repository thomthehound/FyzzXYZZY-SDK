// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_incremental_pairs_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
constexpr std::uint32_t unused=0xcdcdcdcdu,boxes=2u,slots=4u;
void require(bool ok,const char* message){
    if(!ok)throw std::runtime_error(message);
}
ShaderBinary shader(const char* name){
    std::ifstream f(name,std::ios::binary);
    if(!f)throw std::runtime_error("Missing original SAP incremental producer shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{f},std::istreambuf_iterator<char>{}};
    ShaderBinary s;
#if defined(FYZZXYZZY_TEST_D3D12)
    s.format=ShaderFormat::Dxil;
#else
    s.format=ShaderFormat::Spirv;
#endif
    s.bytes.assign(bytes.begin(),bytes.end());
    return s;
}
std::size_t count(std::uint32_t slot){
    if(slot<33u)return slot%11u>=7u?boxes:slots;
    if(slot==33u||slot==34u)return boxes*6u;
    if(slot==35u||slot==36u)return boxes;
    return slots*2u;
}
void run(Context& context,SourceSapIncrementalPairsNativeStage& stage,
         bool lose,bool shared){
    std::array<std::vector<std::uint32_t>,39> data{};
    for(std::uint32_t i=0u;i<39u;++i)
        data[i].assign(count(i),unused);
    // One crossing of a previous X endpoint, with no Y/Z crossings.
    // Both cases reproduce the candidate index/partner lookup from the
    // original SAP rank histories; the oracle is independently specified.
    if(!lose){
        // B.start moves before A.end: a newly overlapping pair (0,1).
        data[0u]={1u,9u,0u,8u}; // current sorted endpoint handles
        data[1u]={0u,2u,1u,3u}; // rank in previous sorted frame
        data[2u]={0u,0u,1u,1u}; // exclusive comparison prefixes
        data[6u]={0u,0u,1u,1u}; // previous exclusive end counts
        data[10u]={0u,unused}; // previous ordered END handles
    }else{
        // B.end moves before A.start: previously overlapping pair lost.
        data[0u]={8u,1u,9u,0u};
        data[1u]={3u,0u,1u,2u};
        data[2u]={0u,1u,1u,1u};
        data[5u]={0u,1u,1u,1u}; // previous exclusive START counts
        data[9u]={0u,unused}; // previous ordered START handles
    }
    for(std::uint32_t axis=1u;axis<3u;++axis){
        const auto base=axis*11u;
        data[base]={1u,0u,9u,8u};
        data[base+1u]={0u,1u,2u,3u};
        data[base+2u]={0u,0u,0u,0u};
    }
    for(std::uint32_t object=0u;object<boxes;++object)
        for(std::uint32_t axis=0u;axis<3u;++axis){
            const auto i=object*6u+axis;
            const std::uint32_t lo=(axis==0u?
                (object==0u?100u:110u):10u);
            const std::uint32_t hi=(axis==0u?
                (object==0u?120u:130u):30u);
            data[33u][i]=lo;
            data[33u][i+3u]=hi;
            data[34u][i]=lo;
            data[34u][i+3u]=hi;
        }
    if(lose){
        data[34u][6u]=70u;data[34u][9u]=90u;
    }else{
        data[33u][6u]=130u;data[33u][9u]=150u;
    }
    data[35u]={0u,1u};
    data[36u]={0xffffffffu,0xffffffffu};
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles=boxes;
    source.max_found_lost_pairs=slots;
    source.sharedFoundPairIndex=0u;
    source.sharedLostPairIndex=0u;
    source.totalIncrementalComparisons[0]=1u;
    source.totalIncrementalComparisons[1]=0u;
    source.totalIncrementalComparisons[2]=0u;
    CudaDeviceAddressMap map;
    std::array<BufferId,40u> buffer{};
    std::array<std::size_t,40u> offset{};
    std::array<CudaDeviceAddressMap::Address,40u> pointer{};
    BufferId root=0u;
    const auto allocation=[&](std::size_t index){
        return index==39u?sizeof(source):data[index].size()*4u;
    };
    if(shared){
        std::size_t total=0u;
        for(std::size_t i=0u;i<40u;++i){
            total=(total+255u)&~std::size_t{255u};
            offset[i]=total; total+=allocation(i);
        }
        root=context.create_buffer(total,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0u;i<40u;++i){
            buffer[i]=root;
            pointer[i]=map.register_slice(root,offset[i],allocation(i));
        }
    }else{
        for(std::size_t i=0u;i<40u;++i){
            buffer[i]=context.create_buffer(allocation(i),MemoryPlacement::DeviceLocal);
            pointer[i]=map.register_allocation(buffer[i],allocation(i));
        }
    }
    for(std::size_t i=0u;i<39u;++i)
        context.upload(buffer[i],offset[i],data[i].data(),allocation(i));
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address a){
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(a));
    };
    using Word=std::remove_reference_t<decltype(source.boxHandles[0][0])>;
    for(std::uint32_t axis=0u;axis<3u;++axis){
        const auto base=axis*11u;
        source.boxHandles[0][axis]=ptr.operator()<Word>(pointer[base+0u]);
        source.boxProjectionRanks[axis]=ptr.operator()<Word>(pointer[base+1u]);
        source.incrementalComparisons[axis]=ptr.operator()<Word>(pointer[base+2u]);
        source.startPtHistogram[0][axis]=ptr.operator()<Word>(pointer[base+3u]);
        source.endPtHistogram[0][axis]=ptr.operator()<Word>(pointer[base+4u]);
        source.startPtHistogram[1][axis]=ptr.operator()<Word>(pointer[base+5u]);
        source.endPtHistogram[1][axis]=ptr.operator()<Word>(pointer[base+6u]);
        source.startPointHandles[0][axis]=ptr.operator()<Word>(pointer[base+7u]);
        source.endPointHandles[0][axis]=ptr.operator()<Word>(pointer[base+8u]);
        source.startPointHandles[1][axis]=ptr.operator()<Word>(pointer[base+9u]);
        source.endPointHandles[1][axis]=ptr.operator()<Word>(pointer[base+10u]);
    }
    using AABB=decltype(source.oldIntegerBounds);
    using Pairs=decltype(source.foundPairReport);
    source.oldIntegerBounds=ptr.operator()<AABB>(pointer[33u]);
    source.newIntegerBounds=ptr.operator()<AABB>(pointer[34u]);
    source.updateData_groups=ptr.operator()<Word>(pointer[35u]);
    source.updateData_envIDs=nullptr;
    source.foundPairReport=ptr.operator()<Pairs>(pointer[37u]);
    source.lostPairReport=ptr.operator()<Pairs>(pointer[38u]);
    context.upload(buffer[39u],offset[39u],&source,sizeof(source));
    std::cerr<<"source pairs: allocation/upload complete; building commands\n";
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    auto bad=source;
    bad.lostPairReport=ptr.operator()<Pairs>(
        static_cast<CudaDeviceAddressMap::Address>(0x1234u));
    bool rejected=false;
    try{stage.append(commands,bad,pointer[39u],map,{boxes,slots,boxes,slots});}
    catch(const std::exception&){rejected=true;}
    require(rejected&&commands.empty(),
            "Invalid late PhysX incremental pair source pointer queued GPU commands");
    stage.append(commands,source,pointer[39u],map,{boxes,slots,boxes,slots});
    require(commands.size()==1u,"Original incremental SAP producer did not append one dispatch");
    const auto& d=std::get<DispatchCommand>(commands[0]);
    require(d.groups.x==256u&&d.groups.y==3u&&d.buffers.size()==41u,
            "Original 256x3 SAP producer dispatch or source ABI changed");
    std::cerr<<"source pairs: GPU submit lose="<<lose<<" shared="<<shared<<"\n";
    stream.wait_at_fetch();
    std::cerr<<"source pairs: GPU completion\n";
    std::array<std::uint32_t,2> reportCounts{};
    context.download(buffer[39u],offset[39u]+
        offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex),
        reportCounts.data(),sizeof(reportCounts));
    require(reportCounts[lose?1u:0u]==1u&&reportCounts[lose?0u:1u]==0u,
            "Source GPU report counters disagree with found/lost crossing");
    std::array<std::uint32_t,2> pair{};
    const auto slot=lose?38u:37u;
    context.download(buffer[slot],offset[slot],pair.data(),sizeof(pair));
    require(pair[0]==0u&&pair[1]==1u,
            "Source PxgBroadPhasePair constructor canonicalization differs");
    for(std::size_t i=0u;i<33u;++i){
        std::vector<std::uint32_t> observed(data[i].size());
        context.download(buffer[i],offset[i],observed.data(),allocation(i));
        require(observed==data[i],"Source incremental producer modified endpoint history");
    }
    std::cerr<<"source pairs: reference verified\n";
    if(!lose) {
        // A quiet frame must not replay the preceding found pair. The
        // original buffers, address mappings and GPU source descriptor
        // remain live across the two submissions and the fetch boundary.
        auto quiet=source;
        quiet.sharedFoundPairIndex=0u;
        quiet.sharedLostPairIndex=0u;
        quiet.totalIncrementalComparisons[0]=0u;
        stream.queue_uploads({{pointer[39u],&quiet,sizeof(quiet)}});
        stage.append(stream.commands(),quiet,pointer[39u],map,
                     {boxes,slots,boxes,slots});
        stream.wait_at_fetch();
        context.download(buffer[39u],offset[39u]+
            offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex),
            reportCounts.data(),sizeof(reportCounts));
        require(reportCounts[0u]==0u&&reportCounts[1u]==0u,
                "Quiet source SAP frame repeated a previous GPU pair report");

        // The next motion frame uses the prior frame's CURRENT bounds as
        // its previous bounds, by exchanging only the original owner
        // identities. The opposite crossing must now emit LOST, while
        // the same GPU buffers and descriptors remain allocated.
        auto next=source;
        std::swap(next.oldIntegerBounds,next.newIntegerBounds);
        next.sharedFoundPairIndex=0u;
        next.sharedLostPairIndex=0u;
        next.totalIncrementalComparisons[0u]=1u;
        data[0u]={8u,1u,9u,0u};
        data[1u]={3u,0u,1u,2u};
        data[2u]={0u,1u,1u,1u};
        data[5u]={0u,1u,1u,1u};
        data[9u]={0u,unused};
        // B's new bounds move to the left of A. The original
        // new-bounds buffer is now the prior old-bounds allocation.
        data[33u][6u]=70u;
        data[33u][9u]=90u;
        std::vector<SourceSapUpload> nextUploads;
        nextUploads.reserve(7u);
        nextUploads.push_back({pointer[39u],&next,sizeof(next)});
        for(const std::size_t i:{0u,1u,2u,5u,9u,33u})
            nextUploads.push_back({pointer[i],data[i].data(),allocation(i)});
        stream.queue_uploads(nextUploads);
        stage.append(stream.commands(),next,pointer[39u],map,
                     {boxes,slots,boxes,slots});
        stream.wait_at_fetch();
        context.download(buffer[39u],offset[39u]+
            offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex),
            reportCounts.data(),sizeof(reportCounts));
        require(reportCounts[0u]==0u&&reportCounts[1u]==1u,
                "Next source SAP frame did not convert found to lost");
        context.download(buffer[38u],offset[38u],
                         pair.data(),sizeof(pair));
        require(pair[0u]==0u&&pair[1u]==1u,
                "Next-frame lost pair changed original PhysX handles");
        const auto priorBounds=map.resolve(
            static_cast<CudaDeviceAddressMap::Address>(
                reinterpret_cast<std::uintptr_t>(next.oldIntegerBounds)),48u);
        std::array<std::uint32_t,12> preserved{};
        context.download(priorBounds.buffer,priorBounds.offset,
                         preserved.data(),sizeof(preserved));
        require(preserved[6u]==110u&&preserved[9u]==130u,
                "Frame-to-frame bounds owner swap did not preserve previous GPU geometry");
    }
    if(shared)context.destroy_buffer(root);
    else for(const auto id:buffer)context.destroy_buffer(id);
}
}
int main(int argc,char** argv) {
    try{
        if(argc!=2)throw std::invalid_argument("Require source incremental SAP shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        std::cerr<<"source pairs: native context initialized\n";
        SourceSapIncrementalPairsNativeStage stage(context,shader(argv[1]));
        std::cerr<<"source pairs: shader pipeline created\n";
        for(bool shared:{false,true})
            for(bool lose:{false,true})
                run(context,stage,lose,shared);
        std::cout<<"Source GPU incremental found/lost SAP producer PASS\n";
    }catch(const std::exception& error){
        std::cerr<<"Source GPU incremental SAP producer FAILED: "<<error.what()<<'\n';
        return 1;
    }
}
