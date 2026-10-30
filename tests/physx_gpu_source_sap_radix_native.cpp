// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
namespace {
ShaderBinary shader(const char* path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Missing source SAP radix shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(bytes.begin(),bytes.end());
    return result;
}
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void run(Context& context,const SourceSapRadixNativeStage& stage,
         bool wr,bool shared) {
    // WR: 3 previous boxes -> 6 live endpoints, padded to 8.
    // WOR: 2 previous + 1 created -> 6 live endpoints, padded to 8.
    constexpr std::uint32_t boxes=3u,keys=8u;
    constexpr std::uint32_t perAxis=7u;
    constexpr std::uint32_t hist=512u;
    // [projection, rank, tempProjection, tempRank, radixCounts,
    //  previousHandles, currentHandles] for each of the three axes.
    std::array<std::vector<std::uint32_t>,21> initial{};
    std::array<std::vector<std::pair<std::uint32_t,std::uint32_t>>,3> expected{};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto base=axis*perAxis;
        for(std::uint32_t slot=0;slot<perAxis;++slot)
            initial[base+slot].assign(slot==4u?hist:keys,0xcdcdcdcdu);
        auto& projection=initial[base+0u];
        auto& rank=initial[base+1u];
        auto& old=initial[base+5u];
        auto& current=initial[base+6u];
        for(std::uint32_t i=0u;i<6u;++i) {
            // Deliberate equal projections to verify stable ordering.
            projection[i]=100u+16u*((7u*i+3u*axis)%4u);
            rank[i]=0xf0000000u;
            old[i]=0x1000u+axis*256u+i;
            current[i]=0x2000u+axis*256u+i;
            expected[axis].push_back({
                projection[i],wr?i:current[i]});
        }
        projection[6u]=projection[7u]=0xffffffffu;
        expected[axis].push_back({0xffffffffu,wr?6u:0x3fffffffu});
        expected[axis].push_back({0xffffffffu,wr?7u:0x3fffffffu});
        if(!wr)current[6u]=current[7u]=0x3fffffffu;
        std::stable_sort(expected[axis].begin(),expected[axis].end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
    }
    std::array<BufferId,21> buffers{};
    std::array<CudaDeviceAddressMap::Address,21> pointers{};
    std::array<std::size_t,21> offsets{};
    CudaDeviceAddressMap map;
    BufferId root=0u;
    if(shared) {
        std::size_t total=0u;
        for(std::size_t i=0;i<initial.size();++i) {
            total=(total+255u)&~std::size_t{255u};
            offsets[i]=total;
            total+=initial[i].size()*sizeof(std::uint32_t);
        }
        root=context.create_buffer(total,MemoryPlacement::DeviceLocal);
        for(std::size_t i=0;i<initial.size();++i) {
            buffers[i]=root;
            const auto bytes=initial[i].size()*4u;
            pointers[i]=map.register_slice(root,offsets[i],bytes);
            context.upload(root,offsets[i],initial[i].data(),bytes);
        }
    } else {
        for(std::size_t i=0;i<initial.size();++i) {
            const auto bytes=initial[i].size()*4u;
            buffers[i]=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
            pointers[i]=map.register_allocation(buffers[i],bytes);
            context.upload(buffers[i],0u,initial[i].data(),bytes);
        }
    }
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address p) {
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(p));
    };
    physx::PxgBroadPhaseDesc source{};
    std::array<physx::PxgRadixSortDesc,6> radix{};
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto start=axis*perAxis;
        using Rank=std::remove_reference_t<decltype(source.boxProjectionRanks[axis])>;
        using Handles=std::remove_reference_t<decltype(source.boxHandles[0][axis])>;
        source.boxProjectionRanks[axis]=ptr.operator()<Rank>(pointers[start+1u]);
        source.boxHandles[1][axis]=ptr.operator()<Handles>(pointers[start+5u]);
        source.boxHandles[0][axis]=ptr.operator()<Handles>(pointers[start+6u]);
        auto& forward=radix[axis];
        auto& reverse=radix[axis+3u];
        using Word=std::uint32_t*;
        forward.inputKeys=ptr.operator()<Word>(pointers[start+0u]);
        forward.inputRanks=ptr.operator()<Word>(
            pointers[start+(wr?1u:6u)]);
        forward.outputKeys=ptr.operator()<Word>(pointers[start+2u]);
        forward.outputRanks=ptr.operator()<Word>(pointers[start+3u]);
        forward.radixBlockCounts=ptr.operator()<Word>(pointers[start+4u]);
        reverse.inputKeys=forward.outputKeys;
        reverse.inputRanks=forward.outputRanks;
        reverse.outputKeys=forward.inputKeys;
        reverse.outputRanks=forward.inputRanks;
        reverse.radixBlockCounts=forward.radixBlockCounts;
    }
    SourceSapDmaNative stream(context,map);
    auto& commands=stream.commands();
    auto bad=radix;
    bad[5u].outputRanks=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(0x1234u));
    bool rejected=false;
    try {
        if(wr)stage.append_wr(commands,source,bad,map,boxes);
        else stage.append_wor(commands,bad,map,2u,1u);
    }catch(const std::exception&){rejected=true;}
    require(rejected && commands.empty(),
            "Invalid late original radix descriptor partially appended GPU work");
    if(wr)stage.append_wr(commands,source,radix,map,boxes);
    else stage.append_wor(commands,radix,map,2u,1u);
    require(commands.size()==(wr?18u:16u),
            "Source sort did not enqueue all eight count/scatter passes");
    const auto& sorted=std::get<DispatchCommand>(commands[wr?1u:0u]);
    require(sorted.groups.x==32u && sorted.groups.y==3u &&
            sorted.buffers.size()==6u,
            "Original source radix count dispatch shape changed");
    // No artificial per-pass synchronization: the existing GPU source
    // stream submits and reaps once at the actual fetch boundary.
    stream.wait_at_fetch();
    auto read=[&](std::size_t index) {
        std::vector<std::uint32_t> values(initial[index].size());
        context.download(buffers[index],shared?offsets[index]:0u,
                         values.data(),values.size()*4u);
        return values;
    };
    for(std::uint32_t axis=0u;axis<3u;++axis) {
        const auto base=axis*perAxis;
        const auto projections=read(base);
        const auto ranks=read(base+(wr?1u:6u));
        // The last source radix pass (bits 28..31) leaves six normal
        // keys in digit 0 and two padded sentinels in digit 15.
        // All 31 empty partitions MUST publish zero histogram bins,
        // even though their GPU workgroups skip shared-memory reduction.
        const auto histogram=read(base+4u);
        for(std::uint32_t digit=0u;digit<16u;++digit)
            for(std::uint32_t block=0u;block<32u;++block)
                require(histogram[digit*32u+block]==
                    (block==0u?(digit==0u?6u:(digit==15u?2u:0u)):0u),
                    "Original source radix emitted a nonzero empty partition or incorrect histogram");
        for(std::uint32_t i=0u;i<keys;++i) {
            require(projections[i]==expected[axis][i].first,
                    "Source radix stable projection order differs from PhysX");
            require(ranks[i]==expected[axis][i].second,
                    "Source radix rank/handle payload permutation is not stable");
        }
        require(read(base+5u)==initial[base+5u],
                "Source WR radix mutated previous-frame handle history");
        if(wr) {
            const auto curr=read(base+6u);
            for(std::uint32_t i=0u;i<6u;++i) {
                const auto rank=expected[axis][i].second;
                require(curr[i]==initial[base+5u][rank],
                        "Source WR handle gather used wrong previous rank");
            }
        } else {
            require(read(base+1u)==initial[base+1u],
                    "Source WOR radix overwrote previous WR rank history");
        }
    }
    const auto count=commands.size();
    if(wr)stage.append_wr(commands,source,radix,map,0u);
    else stage.append_wor(commands,radix,map,0u,0u);
    require(commands.size()==count,"Idle source radix frame appended work");
    if(shared)context.destroy_buffer(root);
    else for(auto buffer:buffers)context.destroy_buffer(buffer);
}
}

// Opt-in source GPU radix timing. This measures the original 18-command
// sortProjectionAndHandlesWR stage, not an invented complete SAP frame.
// Upload/reset, pipeline creation and allocation are outside all samples.
void profile_wr_radix(Context& context,const SourceSapRadixNativeStage& stage,
                      std::uint32_t boxes,
                      const SourceSapRadixNativeStage* interleaved=nullptr) {
    using Clock=std::chrono::steady_clock;
    constexpr std::size_t axes=3u, fields=7u, hist_words=512u;
    const std::uint32_t keys=boxes*2u;
    std::array<std::size_t,axes*fields> offsets{};
    std::array<CudaDeviceAddressMap::Address,axes*fields> pointers{};
    std::array<std::size_t,axes*fields> counts{};
    std::size_t bytes=0;
    for(std::size_t i=0;i<counts.size();++i) {
        counts[i]=i%fields==4u?hist_words:keys;
        bytes=(bytes+255u)&~std::size_t{255u};
        offsets[i]=bytes;
        bytes+=counts[i]*sizeof(std::uint32_t);
    }
    std::vector<std::uint32_t> initial(bytes/sizeof(std::uint32_t),0u);
    for(std::uint32_t axis=0u;axis<axes;++axis) {
        const std::size_t start=axis*fields;
        for(std::uint32_t i=0u;i<keys;++i) {
            // The source radix shader consumes projections and previous
            // handle identities, not a synthetic kernel for this benchmark.
            initial[offsets[start]/4u+i]=
                (i*2654435761u)^(axis*0x9e3779b9u);
            initial[offsets[start+5u]/4u+i]=
                axis*keys+i;
            initial[offsets[start+6u]/4u+i]=
                0xffffffffu;
        }
    }
    const auto root=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    CudaDeviceAddressMap map;
    for(std::size_t i=0;i<pointers.size();++i)
        pointers[i]=map.register_slice(
            root,offsets[i],counts[i]*sizeof(std::uint32_t));
    const auto ptr=[]<class T>(CudaDeviceAddressMap::Address address) {
        return reinterpret_cast<T>(static_cast<std::uintptr_t>(address));
    };
    physx::PxgBroadPhaseDesc source{};
    std::array<physx::PxgRadixSortDesc,6u> radix{};
    for(std::uint32_t axis=0u;axis<axes;++axis) {
        const std::size_t start=axis*fields;
        using Rank=std::remove_reference_t<
            decltype(source.boxProjectionRanks[axis])>;
        using Handles=std::remove_reference_t<
            decltype(source.boxHandles[0][axis])>;
        source.boxProjectionRanks[axis]=
            ptr.operator()<Rank>(pointers[start+1u]);
        source.boxHandles[1][axis]=
            ptr.operator()<Handles>(pointers[start+5u]);
        source.boxHandles[0][axis]=
            ptr.operator()<Handles>(pointers[start+6u]);
        auto& forward=radix[axis];
        auto& reverse=radix[axis+3u];
        using Word=std::uint32_t*;
        forward.inputKeys=ptr.operator()<Word>(pointers[start]);
        forward.inputRanks=ptr.operator()<Word>(pointers[start+1u]);
        forward.outputKeys=ptr.operator()<Word>(pointers[start+2u]);
        forward.outputRanks=ptr.operator()<Word>(pointers[start+3u]);
        forward.radixBlockCounts=ptr.operator()<Word>(pointers[start+4u]);
        reverse.inputKeys=forward.outputKeys;
        reverse.inputRanks=forward.outputRanks;
        reverse.outputKeys=forward.inputKeys;
        reverse.outputRanks=forward.inputRanks;
        reverse.radixBlockCounts=forward.radixBlockCounts;
    }

    auto elapsed_us=[](Clock::time_point begin,Clock::time_point end) {
        return std::chrono::duration<double,std::micro>(end-begin).count();
    };
    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());
        return values[values.size()/2u];
    };
    constexpr std::size_t warmup=3u,trials=11u;
    std::vector<double> host_append,host_submit,host_submit_wait,gpu_us;
    host_append.reserve(trials);
    host_submit.reserve(trials);
    host_submit_wait.reserve(trials);
    gpu_us.reserve(trials);
    bool timestamp_supported=true;
    std::vector<BatchCommand> commands;
    for(std::size_t trial=0;trial<warmup+trials;++trial) {
        context.upload(root,0u,initial.data(),bytes);
        commands.clear();
        const auto begin=Clock::now();
        stage.append_wr(commands,source,radix,map,boxes);
        const auto prepared=Clock::now();
        require(commands.size()==18u,
                "Source WR radix profiling command topology changed");
        const auto submitted=context.submit_batch(commands);
        const auto queued=Clock::now();
        context.wait_submission(submitted);
        const auto completed=Clock::now();
        if(trial>=warmup) {
            host_append.push_back(elapsed_us(begin,prepared));
            host_submit.push_back(elapsed_us(prepared,queued));
            host_submit_wait.push_back(elapsed_us(prepared,completed));
        }
        if(timestamp_supported) {
            // Reset/timestamp replay is independent of the CPU timing.
            context.upload(root,0u,initial.data(),bytes);
            const auto gpu_ns=context.profile_batch_gpu_nanoseconds(commands);
            if(!gpu_ns) {
                timestamp_supported=false;
                gpu_us.clear();
            } else if(trial>=warmup) {
                gpu_us.push_back(*gpu_ns*0.001);
            }
        }
        if(interleaved && trial+1u<warmup+trials) {
            // Each original WR radix stage consumes the unsorted projection
            // and previous handle stream. Its alternate stage needs the
            // same initial input, not the preceding stage's sorted output.
            // Neither this reset nor the alternate submission is timed.
            context.upload(root,0u,initial.data(),bytes);
            std::vector<BatchCommand> other;
            interleaved->append_wr(other,source,radix,map,boxes);
            require(other.size()==18u,
                    "Interleaved source WR radix command topology changed");
            context.wait_submission(context.submit_batch(other));
        }
    }
    // The measured real GPU stage must still produce source-correct
    // sorted projection/rank/handle streams at both workload sizes.
    for(std::uint32_t axis=0u;axis<axes;++axis) {
        const std::size_t start=axis*fields;
        std::vector<std::pair<std::uint32_t,std::uint32_t>> expected;
        expected.reserve(keys);
        for(std::uint32_t i=0u;i<keys;++i)
            expected.push_back({initial[offsets[start]/4u+i],i});
        std::stable_sort(expected.begin(),expected.end(),
            [](const auto& left,const auto& right){
                return left.first<right.first;
            });
        std::vector<std::uint32_t> projections(keys),ranks(keys),handles(keys);
        context.download(root,offsets[start],
                         projections.data(),keys*sizeof(std::uint32_t));
        context.download(root,offsets[start+1u],
                         ranks.data(),keys*sizeof(std::uint32_t));
        context.download(root,offsets[start+6u],
                         handles.data(),keys*sizeof(std::uint32_t));
        for(std::uint32_t i=0u;i<keys;++i)
            require(projections[i]==expected[i].first &&
                    ranks[i]==expected[i].second &&
                    handles[i]==initial[offsets[start+5u]/4u+expected[i].second],
                    "Profiled source radix stage produced incorrect sorted GPU streams");
    }
    const char* backend=
#if defined(FYZZXYZZY_TEST_D3D12)
        "d3d12";
#else
        "vulkan";
#endif
    const auto& device=context.device_info();
    if(timestamp_supported)
        std::printf(
            "source_sap_profile scope=wr_radix backend=%s device=\"%s\""
            " boxes=%u commands=18 trials=%zu"
            " interleaved=%u wave_count=%u wave_scatter=%u"
            " median_host_append_us=%.3f median_host_submit_us=%.3f"
            " median_submit_wait_us=%.3f median_gpu_batch_us=%.3f"
            " timed_upload_reset=no gpu_timestamp_replay=yes\n",
            backend,device.name.c_str(),boxes,trials,
            interleaved?1u:0u,stage.using_wave_count()?1u:0u,
            stage.using_wave_scatter()?1u:0u,
            median(host_append),median(host_submit),
            median(host_submit_wait),median(gpu_us));
    else
        std::printf(
            "source_sap_profile scope=wr_radix backend=%s device=\"%s\""
            " boxes=%u commands=18 trials=%zu"
            " interleaved=%u wave_count=%u wave_scatter=%u"
            " median_host_append_us=%.3f median_host_submit_us=%.3f"
            " median_submit_wait_us=%.3f median_gpu_batch_us=unavailable"
            " timed_upload_reset=no gpu_timestamp_replay=yes\n",
            backend,device.name.c_str(),boxes,trials,
            interleaved?1u:0u,stage.using_wave_count()?1u:0u,
            stage.using_wave_scatter()?1u:0u,
            median(host_append),median(host_submit),
            median(host_submit_wait));
    context.destroy_buffer(root);
}
int main(int argc,char** argv) {
    try {
        const bool benchmark=argc==8 &&
            std::string_view(argv[7])=="--benchmark";
        const bool benchmark_alternating=argc==8 &&
            std::string_view(argv[7])=="--benchmark-alternating";
        if(argc!=7 && !benchmark && !benchmark_alternating)
            throw std::invalid_argument(
                "Require six source radix shaders and optional profiling mode");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        SourceSapRadixNativeStage stage(
            context,shader(argv[1]),shader(argv[2]),
            shader(argv[3]),shader(argv[4]),shader(argv[5]),false,shader(argv[6]));
        SourceSapRadixNativeStage portable(
            context,shader(argv[1]),shader(argv[2]),
            shader(argv[3]),shader(argv[4]),shader(argv[5]),true,shader(argv[6]));
        run(context,stage,true,false);
        run(context,portable,false,true);
        std::cout<<"Source WR/WOR stable radix, ranks and original handles PASS\n";
        if(benchmark) {
            profile_wr_radix(context,stage,256u);
            profile_wr_radix(context,portable,256u);
            profile_wr_radix(context,stage,4096u);
            profile_wr_radix(context,portable,4096u);
        } else if(benchmark_alternating) {
            profile_wr_radix(context,stage,256u,&portable);
            profile_wr_radix(context,portable,256u,&stage);
            profile_wr_radix(context,stage,4096u,&portable);
            profile_wr_radix(context,portable,4096u,&stage);
        }
    }catch(const std::exception& e) {
        std::cerr<<"Source WR/WOR radix GPU FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
