// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_report_copy_native.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
ShaderBinary load_shader(const char* path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Original aggregate report copy shader missing");
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(std::istreambuf_iterator<char>{in},
                        std::istreambuf_iterator<char>{});
    require(!shader.bytes.empty(),"Original aggregate copy shader empty");
    return shader;
}
void run(Context& context,ShaderBinary shader) {
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourceAggregateReportCopyNativeStage stage(context,std::move(shader));
    constexpr std::uint32_t capacity=4u;
    constexpr std::uint32_t canary=0xcdab3412u;
    struct Buffer {BufferId gpu;CudaDeviceAddressMap::Address addr;};
    std::vector<Buffer> allocations;
    const auto create=[&](std::size_t bytes) {
        auto gpu=context.create_buffer(bytes);
        auto addr=addresses.register_slice(gpu,0u,bytes);
        allocations.push_back({gpu,addr});
        return addr;
    };
    const auto found=create(capacity*8u+4u);
    const auto lost=create(capacity*8u+4u);
    const auto mapFound=create(capacity*8u+4u);
    const auto mapLost=create(capacity*8u+4u);
    const auto counts=create(8u);
    const auto upload=[&](CudaDeviceAddressMap::Address addr,
                          const void* data,std::size_t bytes) {
        auto view=addresses.resolve(addr,bytes);
        context.upload(view.buffer,view.offset,data,bytes);
    };
    const auto download=[&](CudaDeviceAddressMap::Address addr,
                            void* data,std::size_t bytes) {
        auto view=addresses.resolve(addr,bytes);
        context.download(view.buffer,view.offset,data,bytes);
    };
    std::array<std::uint32_t,capacity*2u+1u> sourceFound{
        1u,2u,3u,4u,5u,6u,7u,8u,canary};
    std::array<std::uint32_t,capacity*2u+1u> sourceLost{
        0u,7u,2u,3u,6u,7u,4u,5u,canary};
    std::array<std::uint32_t,capacity*2u+1u> mapped{};
    mapped.fill(canary);
    upload(found,sourceFound.data(),sizeof(sourceFound));
    upload(lost,sourceLost.data(),sizeof(sourceLost));
    SourceAggregateReportCopyNativeBuffers buffers{
        found,lost,mapFound,mapLost,counts,capacity};
    auto invalid=buffers;
    invalid.found_mapped=invalid.lost_mapped;
    bool rejected=false;
    try {stage.append(stream,addresses,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected && stream.commands().empty(),
            "Original aggregate report copy admitted aliased mapped streams");

    for(std::uint32_t frame=0;frame<2u;++frame) {
        upload(mapFound,mapped.data(),sizeof(mapped));
        upload(mapLost,mapped.data(),sizeof(mapped));
        const std::array<std::uint32_t,2> values=frame==0u
            ? std::array<std::uint32_t,2>{{3u,2u}}
            : std::array<std::uint32_t,2>{{6u,3u}};
        upload(counts,values.data(),sizeof(values));
        stage.append(stream,addresses,buffers);
        require(stream.commands().size()==1u,
                "Original GPU aggregate report copy changed dispatch count");
        stream.wait_at_fetch();
        std::array<std::uint32_t,capacity*2u+1u> resultFound{},resultLost{};
        download(mapFound,resultFound.data(),sizeof(resultFound));
        download(mapLost,resultLost.data(),sizeof(resultLost));
        const std::uint32_t foundWords=2u*std::min(values[0],capacity);
        const std::uint32_t lostWords=2u*std::min(values[1],capacity);
        for(std::uint32_t i=0;i<resultFound.size();++i)
            require(resultFound[i]==(i<foundWords?sourceFound[i]:canary),
                    "Original GPU aggregate found mapped report differs");
        for(std::uint32_t i=0;i<resultLost.size();++i)
            require(resultLost[i]==(i<lostWords?sourceLost[i]:canary),
                    "Original GPU aggregate lost mapped report differs");
        std::array<std::uint32_t,2> raw{};
        download(counts,raw.data(),sizeof(raw));
        require(raw==values,
                "Original aggregate report copy hid pair overflow from fetch");
    }
    require(stream.submission_stats().command_batches==2u,
            "Original aggregate report copy introduced an extra GPU queue");
    for(auto it=allocations.rbegin();it!=allocations.rend();++it) {
        addresses.unregister_slice(it->addr);
        context.destroy_buffer(it->gpu);
    }
    require(addresses.live_allocations()==0u,
            "Original aggregate mapped report GPU views leaked");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument(
            "Expected original aggregate copyReports kernel");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        run(context,load_shader(argv[1]));
        std::cout<<"Original PhysX aggregate GPU mapped reports PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Original PhysX aggregate GPU mapped reports FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
