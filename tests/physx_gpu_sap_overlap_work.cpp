// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/sap_overlap_work_arena.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
using namespace fyzz::compute;
using fyzz::physx_gpu::SapOverlapWorkArenaDesc;
using fyzz::physx_gpu::SapOverlapWorkArenaStage;
namespace {
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
std::vector<std::uint8_t> load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Overlap shader not found");
    std::vector<char> b{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
    return {b.begin(), b.end()};
}
void test(Context& context, SapOverlapWorkArenaStage& stage, bool created) {
    std::vector<std::uint32_t> input(1024u, kCanary);
    constexpr std::uint32_t sorted=64u, sap=96u, ranges=128u;
    constexpr std::uint32_t prefix=192u, total=224u, active=256u;
    constexpr std::uint32_t starts=1280u, counts=2304u, regions=2336u;
    SapOverlapWorkArenaDesc d{};
    d.sorted_x_handles_offset=sorted; d.x_sap_positions_offset=sap;
    d.region_ranges_offset=ranges; d.region_prefix_offset=prefix;
    d.region_total_offset=total; d.active_histograms_offset=active;
    d.start_histograms_offset=starts; d.overlap_counts_offset=counts;
    d.handle_regions_offset=regions;
    std::memcpy(input.data(), &d, sizeof(d));
    const auto set=[&](std::uint32_t base, std::uint32_t index, std::uint32_t value) {
        input[base/4u+index]=value;
    };
    // Two valid X start/end pairs; handle 0 has a new-projection tag.
    const std::uint32_t tags[]={3u,9u,2u,8u};
    for (std::uint32_t i=0;i<4u;++i) set(sorted,i,tags[i]);
    set(sap,0u,0u); set(sap,1u,2u);
    set(sap,2u,1u); set(sap,3u,3u);
    for (std::uint32_t i=0;i<16u;++i) set(ranges,i,0u);
    set(prefix,0u,0u); set(prefix,1u,1u);
    set(prefix,2u,2u); set(prefix,3u,2u);
    set(total,0u,2u);
    for (std::uint32_t i=0;i<4u;++i) {
        set(starts,i,i<2u?i:2u);
        set(active,i,i<2u?i:2u);
    }
    auto expected=input;
    if (created) {
        // Independently enumerate the two logical region work entries.
        for (std::uint32_t box=0;box<2u;++box) {
            const std::uint32_t x=box;
            const std::uint32_t end=box==0u?2u:3u;
            const std::uint32_t hist=box==0u?starts:active;
            const std::uint32_t n=input[hist/4u+end]-input[hist/4u+x+1u];
            expected[counts/4u+box*2u]=n;
            expected[counts/4u+box*2u+1u]=0u;
            expected[regions/4u+box*2u]=box;
            expected[regions/4u+box*2u+1u]=0u;
        }
    }
    auto bytes=static_cast<std::uint32_t>(input.size()*4u);
    auto gpu=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    auto copy=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    context.upload(gpu,0u,input.data(),bytes);
    std::vector<BatchCommand> commands;
    stage.append(commands,gpu,bytes,0u,2u,created?1u:0u);
    if (commands.size()!=(created?1u:0u))
        throw std::runtime_error("Overlap created-handle gate mismatch");
    if (created) {
        const auto& dcmd=std::get<DispatchCommand>(commands.front());
        if (dcmd.groups.x!=32u || dcmd.groups.y!=1u || dcmd.groups.z!=1u)
            throw std::runtime_error("Overlap original launch geometry mismatch");
    }
    commands.emplace_back(CopyCommand{gpu,0u,copy,0u,bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> result(input.size());
    context.download(copy,0u,result.data(),bytes);
    if (result!=expected) {
        auto it=std::mismatch(result.begin(),result.end(),expected.begin());
        throw std::runtime_error("Overlap producer mismatch at word "+
                                 std::to_string(std::distance(result.begin(),it.first)));
    }
    context.destroy_buffer(copy); context.destroy_buffer(gpu);
}
void invalid(SapOverlapWorkArenaStage& stage) {
    std::vector<BatchCommand> commands;
    stage.append(commands,0u,0u,3u,0u,0u);
    if (!commands.empty()) throw std::runtime_error("Unexpected quiet-frame dispatch");
    for (auto bad : {1u,64u,4096u}) {
        bool rejected=false;
        try { stage.append(commands,1u,64u,bad,2u,1u); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected || !commands.empty())
            throw std::runtime_error("Overlap descriptor bounds not checked");
    }
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("Expected overlap producer shader");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        shader.format=ShaderFormat::Dxil;
#else
        shader.format=ShaderFormat::Spirv;
#endif
        shader.bytes=load(argv[1]);
        SapOverlapWorkArenaStage stage(context,std::move(shader));
        invalid(stage); test(context,stage,false); test(context,stage,true);
        std::cout << "PhysX BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX overlap producer FAILED: " << e.what() << '\n';
        return 1;
    }
}
