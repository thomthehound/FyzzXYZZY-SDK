// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_newbounds_pairs_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;
namespace {
constexpr std::size_t kRootSize = 1024u * 1024u;
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
template<class F>
void reject(F&& f, const char* why) {
    bool thrown = false;
    try { f(); } catch (const std::exception&) { thrown = true; }
    check(thrown, why);
}
ShaderBinary shader(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("PhysX native found-pairs shader missing");
    const std::vector<char> bytes{std::istreambuf_iterator<char>(f),
                                  std::istreambuf_iterator<char>()};
    ShaderBinary b;
#if defined(FYZZXYZZY_TEST_D3D12)
    b.format = ShaderFormat::Dxil;
#else
    b.format = ShaderFormat::Spirv;
#endif
    b.bytes.assign(bytes.begin(),bytes.end());
    return b;
}
struct Root {
    BufferId id = 0;
    std::uint32_t next = 256u;
    std::vector<std::uint32_t> image =
        std::vector<std::uint32_t>(kRootSize / 4u, kCanary);
    std::uint32_t reserve(std::uint32_t bytes) {
        const auto offset = (next + 127u) & ~127u;
        if (!bytes || bytes > kRootSize - offset)
            throw std::runtime_error("PhysX multi-root pair fixture exhausted");
        next = offset + bytes;
        return offset;
    }
    void put(std::uint32_t base, std::uint32_t index, std::uint32_t v) {
        image.at(base / 4u + index) = v;
    }
    void put64(std::uint32_t at, std::uint64_t value) {
        put(at,0u,static_cast<std::uint32_t>(value));
        put(at,1u,static_cast<std::uint32_t>(value >> 32u));
    }
};
struct Case {
    std::uint32_t boxes = 19u;
    bool env = false;
    bool filters = false;
    bool candidates = true;
    bool enabled = true;
    bool shared = false;
    bool newFlag = true;
    std::uint32_t prior = 0u;
    std::uint32_t capacity = 0u;
};
void execute(Context& context, SourceNewBoundsPairsNativeStage& stage, Case c) {
    std::cerr << "Source native pair case: boxes=" << c.boxes
              << " env=" << c.env << " filters=" << c.filters
              << " candidates=" << c.candidates << " enabled=" << c.enabled
              << " shared=" << c.shared << " new=" << c.newFlag
              << " prior=" << c.prior << " capacity=" << c.capacity << '\n';
    CudaDeviceAddressMap map;
    std::array<Root,4> roots;
    for (auto& r : roots)
        r.id=context.create_buffer(kRootSize,MemoryPlacement::DeviceLocal);
    std::cerr << "  buffers allocated\n";
    auto allocate=[&](std::uint32_t root,std::uint32_t bytes) {
        const auto offset=roots[root].reserve(bytes);
        return std::pair<CudaDeviceAddressMap::Address,std::uint32_t>{
            map.register_slice(roots[root].id,offset,bytes),offset};
    };
    auto bind=[&](auto& ptr,std::uint32_t root,std::uint32_t bytes) {
        const auto [address, offset]=allocate(root,bytes);
        using T=std::remove_reference_t<decltype(ptr)>;
        ptr=reinterpret_cast<T>(static_cast<std::uintptr_t>(address));
        return offset;
    };
    // PhysX gpuDMAUp retains removed projections until the update finishes.
    const std::uint32_t n=c.boxes, p=2u*(n+(c.enabled?1u:0u));
    const auto r0=0u, r1=c.shared?0u:1u, r2=c.shared?0u:2u,
               r3=c.shared?0u:3u;
    physx::PxgBroadPhaseDesc src{};
    src.numPreviousHandles=n;
    src.numCreatedHandles=c.enabled?1u:0u;
    src.numRemovedHandles=c.enabled?1u:0u;
    src.max_found_lost_pairs=c.capacity?c.capacity:n+8u;
    src.regionAccumTotal=0x7fffffffu; // deliberately stale CPU shadow
    src.overlapChecksTotalRegion=0xffffffffffffffffull;
    src.sharedFoundPairIndex=0xffffffffu;
    const auto [descriptor,descOffset]=allocate(r0,static_cast<std::uint32_t>(sizeof(src)));
    const auto bounds=bind(src.newIntegerBounds,r1,n*24u);
    const auto groups=bind(src.updateData_groups,r1,n*4u);
    std::uint32_t env=0u;
    if(c.env) env=bind(src.updateData_envIDs,r1,n*4u);
    const auto sorted=bind(src.boxHandles[0][0],r1,p*4u);
    const auto sap=bind(src.boxSapBox1D[0],r1,n*8u);
    const auto ranges=bind(src.regionRange,r1,n*32u);
    const auto prefix=bind(src.overlapChecksRegion,r3,64u*n*8u);
    const auto regions=bind(src.overlapChecksHandleRegiones,r3,64u*n*8u);
    const auto activeHist=bind(src.activeRegionsHistogram,r2,64u*p*4u);
    const auto startHist=bind(src.startRegionsHistogram,r2,64u*p*4u);
    const auto activeOrder=bind(src.orderedActiveRegionHandles,r2,64u*p*4u);
    const auto startOrder=bind(src.orderedStartRegionHandles,r2,64u*p*4u);
    const auto reports=bind(src.foundPairReport,r0,src.max_found_lost_pairs*8u);
    auto gpuSource=src;
    gpuSource.regionAccumTotal=c.candidates?1u:0u;
    gpuSource.overlapChecksTotalRegion=c.candidates?n-1u:0u;
    gpuSource.sharedFoundPairIndex=c.prior;
    std::memcpy(reinterpret_cast<std::uint8_t*>(roots[r0].image.data())+
                descOffset,&gpuSource,sizeof(gpuSource));
    roots[r3].put64(prefix,0u);
    roots[r3].put(regions,0u,0u);
    roots[r3].put(regions,1u,0u);
    for(std::uint32_t h=0u;h<n;++h) {
        const auto lo=h, hi=h==0u?n:n+h;
        roots[r1].put(sorted,lo, (h<<3u)|1u|
                                  ((h==0u&&c.newFlag)?2u:0u));
        roots[r1].put(sorted,hi,h<<3u);
        roots[r1].put(sap,h*2u,lo);
        roots[r1].put(sap,h*2u+1u,hi);
        for(std::uint32_t j=0;j<8u;++j) roots[r1].put(ranges,h*8u+j,0u);
        roots[r1].put(groups,h,h!=0u&&c.filters&&h%7u==0u?10u:10u+h);
        if(c.env) roots[r1].put(env,h,h==0u?3u:(h%5u==0u?4u:3u));
        roots[r1].put(bounds,h*6u,h==0u?0u:h);
        roots[r1].put(bounds,h*6u+1u,h!=0u&&c.filters&&h%11u==0u?100001u:0u);
        roots[r1].put(bounds,h*6u+2u,0u);
        roots[r1].put(bounds,h*6u+3u,h==0u?100000u:h+5u);
        roots[r1].put(bounds,h*6u+4u,h!=0u&&c.filters&&h%11u==0u?100010u:10u);
        roots[r1].put(bounds,h*6u+5u,10u);
    }
    const auto hist=c.newFlag?startHist:activeHist;
    const auto order=c.newFlag?startOrder:activeOrder;
    for(std::uint32_t x=0u;x<p;++x)
        roots[r2].put(hist,x,x==0u?0u:(x<=n?x-1u:n-1u));
    for(std::uint32_t i=0;i+1u<n;++i) roots[r2].put(order,i,i+1u);

    std::set<std::pair<std::uint32_t,std::uint32_t>> expectedPairs;
    if(c.candidates&&c.enabled) for(std::uint32_t h=1u;h<n;++h) {
        if(c.filters&&(h%7u==0u||h%11u==0u))continue;
        if(c.env&&h%5u==0u)continue;
        expectedPairs.emplace(0u,h);
    }
    std::cerr << "  host source prepared\n";
    SourceNewBoundsPairsNativeLayout layout{
        descriptor,n,n,p,64u*n,64u*n};
    std::vector<BatchCommand> cmds;
    auto idle=src;idle.numCreatedHandles=0u;
    stage.append(cmds,idle,map,layout);
    check(cmds.empty(),"No-created-handle native pair frame launched");
    if(c.enabled) {
        auto wrong=layout;wrong.overlap_prefix_capacity=1u;
        reject([&]{stage.append(cmds,src,map,wrong);},
               "Source overlap prefix shorter than source allocation accepted");
        auto bad=src;bad.foundPairReport=nullptr;
        reject([&]{stage.append(cmds,bad,map,layout);},
               "Missing PhysX found report source allocation accepted");
    }
    check(cmds.empty(),"Source pair rejection appended half a command");
    std::cerr << "  source validation checked\n";

    std::vector<UploadSpan> uploads;
    for(const auto& root:roots)
        uploads.push_back({root.id,0u,root.image.data(),kRootSize});
    const auto upload=context.submit_upload_batch(uploads);
    std::cerr << "  upload submitted\n";
    stage.append(cmds,src,map,layout);
    std::cerr << "  dispatch appended\n";
    check(cmds.size()==(c.enabled?1u:0u),"Original PhysX created-pair gate changed");
    if(c.enabled) {
        const auto& dispatch=std::get<DispatchCommand>(cmds.front());
        check(dispatch.groups.x==64u&&dispatch.groups.y==1u&&
              dispatch.groups.z==1u&&dispatch.buffers.size()==16u&&
              dispatch.push_constants.size()==96u,
              "Native pair source kernel ABI or original grid changed");
        check(dispatch.buffers[0].buffer==roots[r1].id&&
              dispatch.buffers[6].buffer==roots[r3].id&&
              dispatch.buffers[11].buffer==roots[r2].id&&
              dispatch.buffers[15].buffer==roots[r0].id,
              "PhysX native pair lowering repacked independent heap roots");
        // PhysX's scalar counters need not be 16-byte aligned in the
        // original GPU descriptor. D3D12 RAW UAV descriptors DO need a
        // 16-byte-aligned FirstElement, even for a four-byte field.
        for (const auto& view : dispatch.buffers)
            check(view.offset % 16u == 0u,
                  "D3D12 RAW UAV view base is not 16-byte aligned");
        std::uint32_t scalar_delta = 0u;
        std::memcpy(&scalar_delta,
                    dispatch.push_constants.data() + 15u * sizeof(std::uint32_t),
                    sizeof(scalar_delta));
        const auto actual_scalar = map.resolve(
            descriptor + offsetof(physx::PxgBroadPhaseDesc, sharedFoundPairIndex),
            sizeof(std::uint32_t));
        check(dispatch.buffers[15].offset + scalar_delta == actual_scalar.offset,
              "Shader-local scalar displacement did not preserve original PhysX pointer");
        const auto gpu=context.submit_batch(cmds);
        std::cerr << "  GPU dispatch submitted\n";
        context.wait_submission(gpu);
        std::cerr << "  GPU completion acknowledged\n";
    }
    context.wait_submission(upload);
    std::cerr << "  upload completion acknowledged\n";

    std::vector<std::uint32_t> actual(kRootSize/4u);
    for(std::uint32_t root=1u;root<4u;++root) {
        std::cerr << "  checking root " << root << '\n';
        context.download(roots[root].id,0u,actual.data(),kRootSize);
        check(actual==roots[root].image,
              "Original PhysX native pair read-only root or canaries changed");
    }
    std::cerr << "  checking report root\n";
    context.download(roots[0].id,0u,actual.data(),kRootSize);
    auto original=roots[0].image;
    const auto expectedCount=c.prior+static_cast<std::uint32_t>(expectedPairs.size());
    const auto observedCount=actual[(descOffset+
        offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex))/4u];
    check(observedCount==expectedCount,
          "GPU-resident original found-pair counter failed");
    const std::uint32_t written=std::min<std::uint32_t>(
        static_cast<std::uint32_t>(expectedPairs.size()),
        src.max_found_lost_pairs>c.prior?src.max_found_lost_pairs-c.prior:0u);
    std::set<std::pair<std::uint32_t,std::uint32_t>> observed;
    for(std::uint32_t i=0u;i<written;++i) {
        const auto at=reports/4u+2u*(c.prior+i);
        const auto pair=std::make_pair(actual[at],actual[at+1u]);
        check(expectedPairs.count(pair)&&observed.insert(pair).second,
              "Original native found-pair report missing, duplicate or invalid");
        actual[at]=original[at];
        actual[at+1u]=original[at+1u];
    }
    check(observed.size()==written,"Source found-pair capacity semantics changed");
    actual[(descOffset+offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex))/4u]
        =original[(descOffset+offsetof(physx::PxgBroadPhaseDesc,sharedFoundPairIndex))/4u];
    check(actual==original,"PhysX resident descriptor or found-pair canaries changed");
    for(auto& root:roots)context.destroy_buffer(root.id);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected source native pair shader");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& d=ctx.device_info();
        if(!d.supports_int64_shader||!d.supports_wave_arithmetic||
           d.subgroup_size<4u||d.subgroup_size>256u||
           256u%d.subgroup_size!=0u||
           d.max_workgroup_invocations<256u||
           d.max_workgroup_size[0]<256u||
           d.max_workgroup_count[0]<64u) {
            std::cout<<"SKIP: source found-pair wave kernel unsupported\n";
            return 77;
        }
        SourceNewBoundsPairsNativeStage stage(ctx,shader(argv[1]));
        for(const auto n:{2u,19u,301u}) {
            Case c;c.boxes=n;execute(ctx,stage,c);
            c.filters=true;execute(ctx,stage,c);
            c.env=true;execute(ctx,stage,c);
        }
        Case extra;extra.shared=true;execute(ctx,stage,extra);
        extra.newFlag=false;execute(ctx,stage,extra);
        extra.candidates=false;execute(ctx,stage,extra);
        extra.enabled=false;execute(ctx,stage,extra);
        extra.enabled=true;extra.candidates=true;
        extra.prior=1u;extra.capacity=3u;execute(ctx,stage,extra);
        std::cout<<"PhysX original source multi-root new-bounds pairs PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Source native PhysX new-bounds pairs FAILED: "<<e.what()<<'\n';
        return 1;
    }
}
