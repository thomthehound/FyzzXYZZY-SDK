// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_sequence_native.h"
#include "physx_gpu/broadphase_completion.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;
namespace {
constexpr std::uint32_t kCanary=0xcdcdcdcd, kBoxes=2u, kSlots=6u;
constexpr std::uint32_t kRegions=kBoxes*64u;
constexpr std::size_t kRootBytes=65536u;

void check(bool ok,const char* msg) {
    if(!ok) throw std::runtime_error(msg);
}
template<class Fn> void reject(Fn&& fn,const char* msg) {
    bool threw=false;
    try{fn();}catch(const std::exception&){threw=true;}
    check(threw,msg);
}
ShaderBinary load(const char* filename) {
    std::ifstream f(filename,std::ios::binary);
    if(!f) throw std::runtime_error("Source native four-stage shader missing");
    const std::vector<char> data{std::istreambuf_iterator<char>(f),
                                 std::istreambuf_iterator<char>()};
    ShaderBinary b;
#if defined(FYZZXYZZY_TEST_D3D12)
    b.format=ShaderFormat::Dxil;
#else
    b.format=ShaderFormat::Spirv;
#endif
    b.bytes.assign(data.begin(),data.end());
    return b;
}
struct Root {
    BufferId buffer=0;
    std::vector<std::uint32_t> image=
        std::vector<std::uint32_t>(kRootBytes/4u,kCanary);
    std::uint32_t next=256u;
    std::uint32_t reserve(std::uint32_t bytes) {
        const auto offset=(next+127u)&~127u;
        if(!bytes||bytes>kRootBytes-offset)
            throw std::runtime_error("Source four-stage native root exhausted");
        next=offset+bytes;
        return offset;
    }
    void put(std::uint32_t at,std::uint32_t word,std::uint32_t value) {
        image.at(at/4u+word)=value;
    }
    void put64(std::uint32_t at,std::uint64_t value) {
        put(at,0u,static_cast<std::uint32_t>(value));
        put(at,1u,static_cast<std::uint32_t>(value>>32u));
    }
};
struct Case {
    bool shared=false;
    bool new_flag=true;
    bool environment=false;
    bool matching_group=false;
    bool empty=false;
    std::uint32_t prior=0u;
    std::uint32_t report_capacity=4u;
};
void run(Context& ctx, const SourceNewBoundsSequenceNativeStage& stage,Case c) {
    CudaDeviceAddressMap addresses;
    std::array<Root,4> roots;
    for(auto& root:roots)
        root.buffer=ctx.create_buffer(kRootBytes,MemoryPlacement::DeviceLocal);
    const auto r0=0u, r1=c.shared?0u:1u, r2=c.shared?0u:2u,
               r3=c.shared?0u:3u;
    auto allocate=[&](std::uint32_t r,std::uint32_t bytes) {
        auto& root=roots.at(r);
        const auto offset=root.reserve(bytes);
        return std::pair<CudaDeviceAddressMap::Address,std::uint32_t>{
            addresses.register_slice(root.buffer,offset,bytes),offset};
    };
    auto bind=[&](auto& field,std::uint32_t r,std::uint32_t bytes) {
        const auto [ptr,off]=allocate(r,bytes);
        using T=std::remove_reference_t<decltype(field)>;
        field=reinterpret_cast<T>(static_cast<std::uintptr_t>(ptr));
        return off;
    };
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles=kBoxes;
    source.numCreatedHandles=1u;
    source.numRemovedHandles=1u;
    source.max_found_lost_pairs=c.report_capacity;
    // Host shadow is deliberately stale; the source GPU scalar is authoritative.
    source.regionAccumTotal=0x7fffffffu;
    source.overlapChecksTotalRegion=0xfeedbadbeefull;
    source.sharedFoundPairIndex=0xffffffffu;
    const auto [descriptor,descriptorOffset]=allocate(r0,static_cast<std::uint32_t>(sizeof(source)));
    const auto sorted=bind(source.boxHandles[0][0],r1,kSlots*4u);
    const auto sap=bind(source.boxSapBox1D[0],r1,kBoxes*8u);
    const auto ranges=bind(source.regionRange,r1,kBoxes*32u);
    const auto regionPrefix=bind(source.regionAccum,r3,kSlots*4u);
    const auto activeHist=bind(source.activeRegionsHistogram,r2,64u*kSlots*4u);
    const auto startHist=bind(source.startRegionsHistogram,r2,64u*kSlots*4u);
    const auto activeOrder=bind(source.orderedActiveRegionHandles,r2,64u*kSlots*4u);
    const auto startOrder=bind(source.orderedStartRegionHandles,r2,64u*kSlots*4u);
    const auto counts=bind(source.overlapChecksRegion,r3,kRegions*8u);
    const auto handleRegions=bind(source.overlapChecksHandleRegiones,r3,kRegions*8u);
    const auto blocks=bind(source.blockOverlapChecksRegion,r3,32u*8u);
    const auto bounds=bind(source.newIntegerBounds,r1,kBoxes*24u);
    const auto groups=bind(source.updateData_groups,r1,kBoxes*4u);
    std::uint32_t env=0u;
    if(c.environment)env=bind(source.updateData_envIDs,r1,kBoxes*4u);
    const auto reports=bind(source.foundPairReport,r0,c.report_capacity*8u);

    auto resident=source;
    resident.regionAccumTotal=c.empty?0u:1u;
    resident.overlapChecksTotalRegion=0xfeedbadbeefull; // previous-frame value
    resident.sharedFoundPairIndex=c.prior;
    std::memcpy(reinterpret_cast<std::uint8_t*>(roots[r0].image.data())+
                descriptorOffset,&resident,sizeof(resident));

    // The prior SAP/region producer has already produced exactly one
    // candidate region for the new min-X endpoint of handle 0.
    for(std::uint32_t i=0u;i<kSlots;++i)
        roots[r3].put(regionPrefix,i,i?1u:0u);
    roots[r1].put(sorted,0u,c.new_flag?3u:1u);
    roots[r1].put(sorted,1u,9u);
    roots[r1].put(sorted,2u,0u);
    roots[r1].put(sorted,3u,8u);
    roots[r1].put(sap,0u,0u);
    roots[r1].put(sap,1u,2u);
    roots[r1].put(sap,2u,1u);
    roots[r1].put(sap,3u,3u);
    for(std::uint32_t i=0u;i<16u;++i)
        roots[r1].put(ranges,i,0u);
    const auto hist=c.new_flag?startHist:activeHist;
    const auto ordered=c.new_flag?startOrder:activeOrder;
    roots[r2].put(hist,1u,0u);
    roots[r2].put(hist,2u,1u);
    roots[r2].put(ordered,0u,1u);
    roots[r1].put(groups,0u,10u);
    roots[r1].put(groups,1u,c.matching_group?10u:11u);
    if(c.environment) {
        roots[r1].put(env,0u,3u);
        roots[r1].put(env,1u,4u); // suppress otherwise valid group-filtered pair
    }
    // PhysX integer AABB: three minima followed by three maxima.
    for(std::uint32_t h=0u;h<kBoxes;++h) {
        const auto x=h==0u?0u:1u;
        roots[r1].put(bounds,h*6u,x);
        roots[r1].put(bounds,h*6u+1u,0u);
        roots[r1].put(bounds,h*6u+2u,0u);
        roots[r1].put(bounds,h*6u+3u,10u);
        roots[r1].put(bounds,h*6u+4u,10u);
        roots[r1].put(bounds,h*6u+5u,10u);
    }
    auto expected=roots;
    // The first work kernel writes the per-region count and handle mapping;
    // both scan kernels consume this live GPU data and replace the count
    // with a 64-bit exclusive prefix. No host-created overlap prefix.
    const bool candidate=!c.empty;
    if(candidate) {
        expected[r3].put64(counts,0u);
        expected[r3].put(handleRegions,0u,0u);
        expected[r3].put(handleRegions,1u,0u);
    }
    for(std::uint32_t i=0u;i<32u;++i)
        expected[r3].put64(blocks+i*8u,i==0u&&candidate?1u:0u);
    expected[r0].put64(
        descriptorOffset+offsetof(physx::PxgBroadPhaseDesc,overlapChecksTotalRegion),
        candidate?1u:0u);

    const bool found=candidate&&!c.matching_group&&!c.environment;
    expected[r0].put(
        descriptorOffset+offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex),
        0u,c.prior+(found?1u:0u));
    if(found&&c.prior<c.report_capacity) {
        expected[r0].put(reports,c.prior*2u,0u);
        expected[r0].put(reports,c.prior*2u+1u,1u);
    }

    const SourceNewBoundsSequenceNativeLayout layout{
        descriptor,kBoxes,kBoxes,kSlots,kRegions};
    std::vector<BatchCommand> operations;
    auto idle=source;
    idle.numCreatedHandles=0u;
    stage.append(operations,idle,addresses,layout);
    check(operations.empty(),"Zero-created-handles frame appended PhysX GPU work");
    auto invalid=source;
    invalid.foundPairReport=nullptr;
    reject([&]{stage.append(operations,invalid,addresses,layout);},
           "Invalid fourth stage committed partial first three commands");
    check(operations.empty(),"Failed last-stage view left partial native batch");
    stage.append(operations,source,addresses,layout);
    check(operations.size()==4u,
          "Original PhysX 4-kernel source launch order changed");
    const std::array<std::uint32_t,4> xGroups{32u,32u,32u,64u};
    const std::array<std::uint32_t,4> bindings{9u,4u,4u,16u};
    for(std::size_t i=0u;i<operations.size();++i) {
        const auto& cmd=std::get<DispatchCommand>(operations[i]);
        check(cmd.groups.x==xGroups[i]&&cmd.groups.y==1u&&
              cmd.groups.z==1u&&cmd.buffers.size()==bindings[i],
              "Source PhysX launch grids or binding layouts diverged");
    }
    // Upload initial state once; then run all four native kernels as one
    // GPU batch with no intervening CPU wait, readback or repacking.
    std::vector<UploadSpan> uploads;
    for(const auto& root:roots)
        uploads.push_back({root.buffer,0u,root.image.data(),kRootBytes});
    ctx.submit_upload_batch(uploads);
    const auto sub=ctx.submit_batch(operations);
    BroadPhaseCompletionGate gate(ctx);
    gate.arm(sub);
    // The *same GPU submission* that carries the original four kernels
    // is the completion dependency for PhysX's postBroadPhase publication.
    // Nothing is downloaded before the fetch callback.
    std::vector<std::string> events;
    auto publish=[&] {
        std::vector<std::uint32_t> actual(kRootBytes/4u);
        for(std::uint32_t i=0u;i<roots.size();++i) {
            ctx.download(roots[i].buffer,0u,actual.data(),kRootBytes);
            check(actual==expected[i].image,
                  "Four-source-kernel result or untouched allocation canary mismatch");
        }
        events.emplace_back("published");
    };
    if(c.shared) {
        // Non-null continuation path: install the two original task
        // continuations before releasing either source task reference.
        struct PairTask final : physx::PxBaseTask {
            PairTask(std::vector<std::string>& log,const char* name)
                : events(log),label(name) {}
            void run() override {}
            const char* getName() const override { return label; }
            void addReference() override { ++refs; }
            void removeReference() override {
                if(refs!=1)throw std::runtime_error("PhysX pair task released twice");
                --refs;events.emplace_back(label);
            }
            std::int32_t getReference() const override { return refs; }
            void release() override {}
            std::vector<std::string>& events;
            const char* label;
            std::int32_t refs=1;
        };
        PairTask found(events,"found"),lost(events,"lost");
        gate.fetch_and_release(publish,
            [&]{events.emplace_back("continuations");},found,lost);
        check(events==std::vector<std::string>{
            "published","continuations","found","lost"},
            "Original PhysX continuation tasks released before source GPU publication");
    } else {
        // Null continuation path: directly execute the two existing
        // PxgAABBManager tasks *after* source-owned GPU result retrieval.
        gate.fetch_and_run_inline(publish,
            [&]{events.emplace_back("found");},
            [&]{events.emplace_back("lost");});
        check(events==std::vector<std::string>{"published","found","lost"},
              "Original inline pair tasks ran before source GPU publication");
    }
    check(!gate.armed(),"Broadphase source GPU frame published more than once");
    // Source uploads and all four broadphase launches share one ordered
    // native queue. Waiting on the later four-kernel submission, followed
    // by the publishing readback, already makes the earlier upload visible.
    // Vulkan download drains all tracked submissions, including the old
    // upload token; D3D12 readback uses its own fence. Neither backend
    // requires an additional wait for an already ordered producer.
    for(auto& root:roots)ctx.destroy_buffer(root.buffer);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=5)
            throw std::invalid_argument("Expected work, first scan, second scan and found-pair shaders");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& gpu=ctx.device_info();
        if(!gpu.supports_int64_shader||!gpu.supports_int64_wave||
           !gpu.supports_wave_arithmetic||gpu.subgroup_size<4u||
           gpu.subgroup_size>128u||256u%gpu.subgroup_size!=0u||
           gpu.max_workgroup_invocations<256u||
           gpu.max_workgroup_size[0]<256u||gpu.max_workgroup_count[0]<64u) {
            std::cout<<"SKIP: source new-bounds four-stage GPU requirements unavailable\n";
            return 77;
        }
        SourceOverlapWorkNativeStage work(ctx,load(argv[1]));
        SourceOverlapPrefixNativeStage scan(ctx,load(argv[2]),load(argv[3]));
        SourceNewBoundsPairsNativeStage pairs(ctx,load(argv[4]));
        SourceNewBoundsSequenceNativeStage stage(work,scan,pairs);
        run(ctx,stage,{});
        Case aliased;aliased.shared=true;run(ctx,stage,aliased);
        Case old;old.new_flag=false;run(ctx,stage,old);
        Case env;env.environment=true;run(ctx,stage,env);
        Case filtered;filtered.matching_group=true;run(ctx,stage,filtered);
        Case zero;zero.empty=true;run(ctx,stage,zero);
        Case capped;capped.prior=1u;capped.report_capacity=1u;run(ctx,stage,capped);
        std::cout<<"Original PhysX source-owned 4-kernel new-bounds GPU batch PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Source PhysX native four-kernel batch FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
