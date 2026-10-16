// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_report_classification_arena.h"
#include "physx_gpu/broadphase_copy_reports_arena.h"
#include "physx_gpu/source_sap_reports_native.h"
#include "physx_gpu/source_sap_dma_native.h"
#include "physx_gpu/source_physx_heap_roots.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include <PxPhysicsAPI.h>
#include "physx_gpu/source_sap_fetch_native.h"
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/broadphase_completion.h"
#include "BpVolumeData.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::BroadphaseReportClassificationArenaDesc;
using fyzz::physx_gpu::BroadphaseReportClassificationArenaStage;
using fyzz::physx_gpu::BroadphaseCopyReportsArenaDesc;
using fyzz::physx_gpu::BroadphaseCopyReportsArenaStage;
using fyzz::physx_gpu::SourceSapReportNativeStage;

constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kBlocks = 32u;
constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kVolumes = 128u;

std::vector<std::uint8_t> load_shader(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX report classification shader missing");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}

struct Packed {
    std::vector<std::uint32_t> memory;
    std::uint32_t classDescOffset = 0u, copyDescOffset = 0u;
    BroadphaseReportClassificationArenaDesc classification{};
    BroadphaseCopyReportsArenaDesc copy{};

    std::uint32_t reserve(std::uint32_t count) {
        while (memory.size() % 32u) memory.push_back(kCanary);
        const auto first = memory.size();
        memory.resize(first + count, kCanary);
        return static_cast<std::uint32_t>(first * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t index = 0u) {
        return memory[offset / 4u + index];
    }
    std::uint32_t at(std::uint32_t offset, std::uint32_t index = 0u) const {
        return memory[offset / 4u + index];
    }
    template<typename T>
    void put(std::uint32_t offset, const T& desc) {
        std::memcpy(memory.data() + offset / 4u, &desc, sizeof(desc));
    }
};

std::uint32_t raw_pairs(const Packed& p, std::uint32_t kind) {
    return p.at(p.classification.raw_pair_counts_offset, kind);
}
std::uint32_t raw_offset(const Packed& p, std::uint32_t kind) {
    return kind == 0u ? p.classification.found_pairs_offset
                      : p.classification.lost_pairs_offset;
}
std::uint32_t class_offset(const Packed& p, std::uint32_t kind,
                           bool aggregate) {
    const auto& d = p.classification;
    if (kind == 0u) return aggregate ? d.found_aggregate_output_offset :
                                       d.found_actor_output_offset;
    return aggregate ? d.lost_aggregate_output_offset :
                       d.lost_actor_output_offset;
}
std::uint32_t block_offset(const Packed& p, std::uint32_t kind,
                           bool aggregate) {
    const auto& d = p.classification;
    if (kind == 0u) return aggregate ? d.found_aggregate_block_offset :
                                       d.found_actor_block_offset;
    return aggregate ? d.lost_aggregate_block_offset :
                       d.lost_actor_block_offset;
}
bool is_aggregate_pair(const Packed& p, std::uint32_t handleA,
                       std::uint32_t handleB) {
    const auto& d = p.classification;
    if (!d.has_volume_data) return false;
    const auto tags = d.volume_aggregate_tags_offset;
    // PhysX Bp::VolumeData::isSingleActor checks mAggregate == PX_INVALID_U32.
    return p.at(tags, handleA) != kInvalid || p.at(tags, handleB) != kInvalid;
}

Packed pack(std::uint32_t found, std::uint32_t lost,
            std::uint32_t capacity, std::uint32_t aggregateCapacity,
            bool useVolumeData) {
    Packed p;
    p.classDescOffset = p.reserve(sizeof(p.classification) / 4u);
    p.copyDescOffset = p.reserve(sizeof(p.copy) / 4u);
    auto& d = p.classification;
    d.raw_pair_counts_offset = p.reserve(2u);
    d.aggregate_pair_counts_offset = p.reserve(2u);
    d.found_pairs_offset = p.reserve(std::max(2u, capacity * 2u));
    d.lost_pairs_offset = p.reserve(std::max(2u, capacity * 2u));
    d.found_aggregate_output_offset = p.reserve(std::max(2u, aggregateCapacity * 2u));
    d.lost_aggregate_output_offset = p.reserve(std::max(2u, aggregateCapacity * 2u));
    d.found_actor_output_offset = p.reserve(std::max(2u, capacity * 2u));
    d.lost_actor_output_offset = p.reserve(std::max(2u, capacity * 2u));
    d.found_aggregate_block_offset = p.reserve(kBlocks);
    d.found_actor_block_offset = p.reserve(kBlocks);
    d.lost_aggregate_block_offset = p.reserve(kBlocks);
    d.lost_actor_block_offset = p.reserve(kBlocks);
    d.volume_aggregate_tags_offset = useVolumeData ? p.reserve(kVolumes) : kInvalid;
    d.volume_capacity = kVolumes;
    d.raw_report_capacity = capacity;
    d.aggregate_report_capacity = aggregateCapacity;
    d.has_volume_data = useVolumeData ? 1u : 0u;

    if (useVolumeData) {
        for (std::uint32_t h = 0u; h < kVolumes; ++h) {
            // Single actor, aggregate volume, and actor-in-aggregate all
            // use the exact original mAggregate layout.
            const auto tag = (h % 5u == 0u || h % 5u == 4u)
                ? kInvalid : ((h << 1u) | (h % 5u == 1u ? 1u : 0u));
            p.at(d.volume_aggregate_tags_offset, h) = tag;
        }
    }
    p.at(d.raw_pair_counts_offset) = found;
    p.at(d.raw_pair_counts_offset, 1u) = lost;
    p.at(d.aggregate_pair_counts_offset) = kCanary;
    p.at(d.aggregate_pair_counts_offset, 1u) = kCanary;

    for (std::uint32_t kind = 0u; kind < 2u; ++kind) {
        const auto n = std::min(kind ? lost : found, capacity);
        const auto offset = raw_offset(p, kind);
        for (std::uint32_t i = 0u; i < n; ++i) {
            std::uint32_t a = (i * 13u + kind * 11u + 3u) % kVolumes;
            std::uint32_t b = (i * 23u + kind * 7u + 37u) % kVolumes;
            if (a == b) b = (b + 1u) % kVolumes;
            p.at(offset, 2u * i) = std::min(a, b);
            p.at(offset, 2u * i + 1u) = std::max(a, b);
        }
    }

    p.copy.raw_pair_counts_offset = d.raw_pair_counts_offset;
    p.copy.aggregate_pair_counts_offset = d.aggregate_pair_counts_offset;
    p.copy.found_actor_pairs_offset = d.found_actor_output_offset;
    p.copy.lost_actor_pairs_offset = d.lost_actor_output_offset;
    p.copy.found_map_offset = p.reserve(std::max(2u, capacity * 2u));
    p.copy.lost_map_offset = p.reserve(std::max(2u, capacity * 2u));
    p.copy.overflow_flag_offset = p.reserve(1u);
    p.copy.report_capacity = capacity;
    p.at(p.copy.overflow_flag_offset) = 0x12345678u;
    p.put(p.classDescOffset, p.classification);
    p.put(p.copyDescOffset, p.copy);
    return p;
}

void reference(Packed& p) {
    const auto& d = p.classification;
    for (std::uint32_t kind = 0u; kind < 2u; ++kind) {
        const auto raw = raw_pairs(p, kind);
        const auto n = std::min(raw, d.raw_report_capacity);
        const auto blocksRequired = (raw + 255u) / 256u;
        const auto iters = (blocksRequired + kBlocks - 1u) / kBlocks;
        const auto blockSize = iters * kThreads;

        for (const bool aggregate : {true, false}) {
            const auto out = class_offset(p, kind, aggregate);
            const auto blockOut = block_offset(p, kind, aggregate);
            const auto outputCapacity = aggregate
                ? d.aggregate_report_capacity : d.raw_report_capacity;
            std::uint32_t total = 0u;
            for (std::uint32_t block = 0u; block < kBlocks; ++block) {
                const auto startCount = total;
                const auto first = block * blockSize;
                const auto last = std::min(n, first + blockSize);
                for (std::uint32_t i = first; i < last; ++i) {
                    const auto a = p.at(raw_offset(p, kind), 2u * i);
                    const auto b = p.at(raw_offset(p, kind), 2u * i + 1u);
                    if (is_aggregate_pair(p, a, b) != aggregate) continue;
                    if (total < outputCapacity) {
                        p.at(out, 2u * total) = a;
                        p.at(out, 2u * total + 1u) = b;
                    }
                    ++total;
                }
                p.at(blockOut, block) = total - startCount;
            }
            if (aggregate)
                p.at(d.aggregate_pair_counts_offset, kind) = total;
        }
        const auto actorCount = n - p.at(d.aggregate_pair_counts_offset, kind);
        const auto actorOut = class_offset(p, kind, false);
        const auto mapped = kind == 0u ? p.copy.found_map_offset : p.copy.lost_map_offset;
        for (std::uint32_t i = 0u; i < 2u * actorCount; ++i)
            p.at(mapped, i) = p.at(actorOut, i);
    }
    p.at(p.copy.overflow_flag_offset) =
        (raw_pairs(p, 0u) > d.raw_report_capacity ||
         raw_pairs(p, 1u) > d.raw_report_capacity) ? 1u : 0u;
}

void run_case(Context& context, BroadphaseReportClassificationArenaStage& classifier,
              BroadphaseCopyReportsArenaStage& copy, std::uint32_t found,
              std::uint32_t lost, std::uint32_t capacity,
              std::uint32_t aggregateCapacity, bool hasVolumes) {
    auto source = pack(found, lost, capacity, aggregateCapacity, hasVolumes);
    auto expected = source;
    reference(expected);
    const auto bytes = static_cast<std::uint32_t>(source.memory.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto staging = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, source.memory.data(), bytes);

    std::vector<BatchCommand> commands;
    classifier.append(commands, gpu, bytes, source.classDescOffset);
    copy.append(commands, gpu, bytes, source.copyDescOffset);
    if (commands.size() != 3u)
        throw std::runtime_error("PhysX report classify→copy source sequence changed");
    for (std::uint32_t i = 0u; i < 3u; ++i) {
        const auto& cmd = std::get<DispatchCommand>(commands[i]);
        if (cmd.groups.x != (i == 2u ? 64u : 32u) ||
            cmd.groups.y != (i == 2u ? 1u : 4u) || cmd.groups.z != 1u)
            throw std::runtime_error("PhysX report launch geometry changed");
    }
    commands.emplace_back(CopyCommand{gpu, 0u, staging, 0u, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(source.memory.size());
    context.download(staging, 0u, actual.data(), bytes);
    if (actual != expected.memory) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(),
                                            expected.memory.begin());
        throw std::runtime_error(
            "PhysX report classify/copy mismatch at word " +
            std::to_string(std::distance(actual.begin(), mismatch.first)) +
            " (found=" + std::to_string(found) +
            ", lost=" + std::to_string(lost) +
            ", maxPairs=" + std::to_string(capacity) +
            ", aggMax=" + std::to_string(aggregateCapacity) +
            ", volumeData=" + std::to_string(hasVolumes) + ")");
    }
    context.destroy_buffer(staging);
    context.destroy_buffer(gpu);
}

void negative_cases(BroadphaseReportClassificationArenaStage& classifier) {
    for (const auto offset : {1u, 128u, 256u}) {
        std::vector<BatchCommand> commands;
        commands.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        bool rejected = false;
        try { classifier.append(commands, 1u, 160u, offset); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || commands.size() != 1u ||
            !std::holds_alternative<CopyCommand>(commands.front()))
            throw std::runtime_error("Bad classifier descriptor altered caller work");
    }
}


struct SourcePairTask final : physx::PxBaseTask {
    int releases = 0;
    void run() override {}
    const char* getName() const override { return "Original GPU broadphase pair task"; }
    void addReference() override {}
    void removeReference() override { ++releases; }
    int32_t getReference() const override { return 1; }
    void release() override {}
};

void run_original_source_report_fetch(
    Context& context, BroadphaseReportClassificationArenaStage& classifier,
    BroadphaseCopyReportsArenaStage& copy, bool useVolumeData, bool overflow,
    fyzz::physx_gpu::SourceSapReportNativeStage* nativeStage=nullptr,
    bool separateRoots=false, bool sourceMappedHeap=false,
    bool forceStaged=false) {
    using namespace fyzz::physx_gpu;
    using Pair=physx::PxgBroadPhasePair;
    constexpr std::uint32_t kBytes=32768u,kCapacity=8u,kVolumes=32u;
    constexpr std::uint32_t kInvalidHandle=0xffffffffu;
    constexpr std::size_t kVolumeStride=sizeof(physx::Bp::VolumeData);
    static_assert(sizeof(void*)==8u && kVolumeStride==16u,
                  "Original native Bp::VolumeData layout changed; inspect mAggregate offset");
    std::vector<std::uint32_t> initial(kBytes/4u,kCanary);
    const auto root=context.create_buffer(kBytes,MemoryPlacement::DeviceLocal);
    context.upload(root,0u,initial.data(),kBytes);
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative source(context,addresses);
    SourcePhysxHeapRoots deviceRoots(context,addresses);
    SourcePhysxDeviceRootCallback deviceCallback(deviceRoots,&source);
    SourcePhysxMappedReportRoots mappedCallback(
        context,addresses,source,
        forceStaged ? SourcePhysxMappedReportRoots::Mode::ForceStaged
                    : SourcePhysxMappedReportRoots::Mode::PreferDirect);
    bool aborted=false;
    const auto abort=[](void* p) noexcept { *static_cast<bool*>(p)=true; };
    std::unique_ptr<physx::PxgHeapMemoryAllocatorManager> sourceHeapManager;
    physx::PxgHeapMemoryAllocator* mappedHostHeap=nullptr;
    Pair* originalFound=nullptr;
    Pair* originalLost=nullptr;
    if(sourceMappedHeap) {
        if(!nativeStage)
            throw std::invalid_argument("Mapped PhysX source heap needs original multi-root report shaders");
        sourceHeapManager=std::make_unique<physx::PxgHeapMemoryAllocatorManager>(
            4096u,deviceCallback,mappedCallback,mappedCallback,abort,&aborted);
        mappedHostHeap=static_cast<physx::PxgHeapMemoryAllocator*>(
            sourceHeapManager->mPinnedHostMappedMemoryAllocator);
        originalFound=static_cast<Pair*>(mappedHostHeap->allocate(
            8u*sizeof(Pair),physx::PxsHeapStats::eBROADPHASE,__FILE__,__LINE__));
        originalLost=static_cast<Pair*>(mappedHostHeap->allocate(
            8u*sizeof(Pair),physx::PxsHeapStats::eBROADPHASE,__FILE__,__LINE__));
        if(!originalFound||!originalLost || mappedCallback.live_roots()!=1u ||
           mappedCallback.is_direct(originalFound)!=mappedCallback.is_direct(originalLost))
            throw std::runtime_error("Original PhysX host heap did not own GPU report destinations");
    }
    std::uint32_t cursor=256u;
    std::vector<BufferId> owned;
    const auto allocate=[&](std::size_t bytes) {
        if(separateRoots) {
            const auto id=context.create_buffer(std::max<std::size_t>(bytes,256u),
                                                MemoryPlacement::DeviceLocal);
            owned.push_back(id);
            return addresses.register_slice(id,0u,bytes);
        }
        cursor=(cursor+255u)&~255u;
        if(!bytes || cursor+bytes>kBytes)
            throw std::runtime_error("Original report native heap exhausted");
        auto result=addresses.register_slice(root,cursor,bytes);
        cursor+=static_cast<std::uint32_t>(bytes);
        return result;
    };
    const auto descriptor=allocate(sizeof(physx::PxgBroadPhaseDesc));
    const auto classifyDesc=allocate(sizeof(BroadphaseReportClassificationArenaDesc));
    const auto copyDesc=allocate(sizeof(BroadphaseCopyReportsArenaDesc));
    const auto rawFound=allocate(sizeof(Pair)*kCapacity);
    const auto rawLost=allocate(sizeof(Pair)*kCapacity);
    const auto aggFound=allocate(sizeof(Pair)*kCapacity);
    const auto aggLost=allocate(sizeof(Pair)*kCapacity);
    const auto actorFound=allocate(sizeof(Pair)*kCapacity);
    const auto actorLost=allocate(sizeof(Pair)*kCapacity);
    const auto mappedFound=sourceMappedHeap
        ? mappedCallback.device_address(originalFound,sizeof(Pair)*kCapacity)
        : allocate(sizeof(Pair)*kCapacity);
    const auto mappedLost=sourceMappedHeap
        ? mappedCallback.device_address(originalLost,sizeof(Pair)*kCapacity)
        : allocate(sizeof(Pair)*kCapacity);
    std::array<CudaDeviceAddressMap::Address,4> blocks{};
    for(auto& b:blocks)b=allocate(32u*sizeof(std::uint32_t));
    const auto volumesAddress=allocate(kVolumes*kVolumeStride);
    const auto ptr=[](CudaDeviceAddressMap::Address address) {
        return reinterpret_cast<Pair*>(static_cast<std::uintptr_t>(address));
    };
    physx::PxgBroadPhaseDesc gpu{};
    gpu.max_found_lost_pairs=kCapacity;
    gpu.max_found_lost_agg_pairs=kCapacity;
    gpu.sharedFoundPairIndex=overflow?10u:6u;
    gpu.sharedLostPairIndex=4u;
    gpu.foundPairReport=ptr(rawFound);
    gpu.lostPairReport=ptr(rawLost);
    gpu.foundAggPairReport=ptr(aggFound);
    gpu.lostAggPairReport=ptr(aggLost);
    gpu.foundActorPairReport=ptr(actorFound);
    gpu.lostActorPairReport=ptr(actorLost);
    gpu.foundPairReportMap=ptr(mappedFound);
    gpu.lostPairReportMap=ptr(mappedLost);
    gpu.aggReportBlock[0]=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(blocks[0]));
    gpu.actorReportBlock[0]=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(blocks[1]));
    gpu.aggReportBlock[1]=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(blocks[2]));
    gpu.actorReportBlock[1]=reinterpret_cast<std::uint32_t*>(
        static_cast<std::uintptr_t>(blocks[3]));
    std::array<physx::Bp::VolumeData,kVolumes> volumes{};
    for(auto& v:volumes)v.reset();
    volumes[4].setAggregate(2u);
    volumes[10].setAggregated(2u);
    std::uint32_t volAggregate=0;
    std::memcpy(&volAggregate,
                reinterpret_cast<const std::uint8_t*>(volumes.data())+8u,4u);
    if(volAggregate!=kInvalidHandle)
        throw std::runtime_error("Source VolumeData mAggregate offset changed");
    if(useVolumeData)
        gpu.aabbMngr_volumeData=reinterpret_cast<physx::Bp::VolumeData*>(
            static_cast<std::uintptr_t>(volumesAddress));
    const SourceSapNativeReportLayout layout{
        root,kBytes,descriptor,classifyDesc,copyDesc,kVolumes,
        static_cast<std::uint32_t>(kVolumeStride),8u
    };
    SourceSapNativeReportDescriptors lowered{};
    if(!nativeStage) {
        lowered=lower_source_sap_report_descriptors(gpu,addresses,layout);
        if(lowered.classification.raw_pair_counts_offset!=
           addresses.resolve(descriptor+offsetof(physx::PxgBroadPhaseDesc,
                                                 sharedFoundPairIndex),8u).offset ||
           lowered.copy.found_map_offset!=addresses.resolve(
               mappedFound,sizeof(Pair)*kCapacity).offset ||
           lowered.classification.volume_stride!=(useVolumeData?kVolumeStride:0u))
            throw std::runtime_error("Source report descriptor lost original GPU aliases");
        auto bad=layout;
        bad.backing_bytes=128u;
        bool invalidRejected=false;
        try{(void)lower_source_sap_report_descriptors(gpu,addresses,bad);}
        catch(const std::exception&){invalidRejected=true;}
        if(!invalidRejected)
            throw std::runtime_error("Source report lowering accepted invalid heap extent");
    }

    std::array<Pair,kCapacity> created{},removed{};
    const std::array<std::pair<std::uint32_t,std::uint32_t>,8> createdInputs{{
        {1u,2u},{3u,4u},{5u,6u},{7u,8u},
        {9u,10u},{11u,12u},{13u,14u},{15u,16u}
    }};
    const std::array<std::pair<std::uint32_t,std::uint32_t>,4> removedInputs{{
        {1u,3u},{2u,4u},{7u,9u},{12u,13u}
    }};
    for(std::size_t i=0;i<createdInputs.size();++i)
        created[i]=Pair(createdInputs[i].first,createdInputs[i].second);
    for(std::size_t i=0;i<removedInputs.size();++i)
        removed[i]=Pair(removedInputs[i].first,removedInputs[i].second);

    std::vector<SourceSapUpload> uploads{
        {descriptor,&gpu,sizeof(gpu)},
        {rawFound,created.data(),sizeof(created)},
        {rawLost,removed.data(),sizeof(removed)},
        {volumesAddress,volumes.data(),sizeof(volumes)}
    };
    if(!nativeStage) {
        uploads.push_back({classifyDesc,&lowered.classification,sizeof(lowered.classification)});
        uploads.push_back({copyDesc,&lowered.copy,sizeof(lowered.copy)});
    }
    source.queue_uploads(uploads);
    // Exact original BP_ACCUMULATE_REPORT_STAGE_1, STAGE_2,
    // BP_COPY_REPORTS launches, and no host synchronization between them.
    if(nativeStage) {
        const SourceSapReportNativeLayout nativeLayout{
            descriptor,kVolumes,static_cast<std::uint32_t>(kVolumeStride),8u};
        auto malformed=gpu;
        malformed.lostPairReportMap=reinterpret_cast<Pair*>(0x100u);
        bool rejected=false;
        try{nativeStage->append(source.commands(),malformed,addresses,nativeLayout);}
        catch(const std::exception&){rejected=true;}
        if(!rejected || !source.commands().empty())
            throw std::runtime_error("Invalid original report GPU pointer partially appended stages");
        nativeStage->append(source.commands(),gpu,addresses,nativeLayout);
    } else {
        classifier.append(source.commands(),root,kBytes,
                          static_cast<std::uint32_t>(addresses.resolve(
                              classifyDesc,sizeof(lowered.classification)).offset));
        copy.append(source.commands(),root,kBytes,
                    static_cast<std::uint32_t>(addresses.resolve(
                        copyDesc,sizeof(lowered.copy)).offset));
    }
    if(source.commands().size()!=3u)
        throw std::runtime_error("Source GPU reports did not append exactly three kernels");
    BroadPhaseCompletionGate gate(context);
    gate.arm_source_stream(source);
    SapFrameBufferRoles roles{};
    roles.box_handles={1u,2u,3u,4u,5u,6u};
    physx::PxgBroadPhaseDesc fetched{};
    std::array<Pair,kCapacity> hostFound{},hostLost{};
    auto* publishedFound = sourceMappedHeap ? originalFound : hostFound.data();
    auto* publishedLost = sourceMappedHeap ? originalLost : hostLost.data();
    SourceSapFetchedCounts counts{};
    SourcePairTask foundTask,lostTask;
    bool published=false,continued=false;
    gate.fetch_and_release([&]{
        counts=fetch_source_sap_results(context,addresses,descriptor,fetched,
            {publishedFound,kCapacity},
            {publishedLost,kCapacity},false,roles,
            sourceMappedHeap ? &mappedCallback : nullptr);
        published=true;
    },[&] {
        if(!published)throw std::runtime_error("Original continuation before reports");
        continued=true;
    },foundTask,lostTask);
    if(!published||!continued||foundTask.releases!=1||lostTask.releases!=1||
       roles.box_handles[0]!=4u||roles.box_handles[3]!=1u)
        throw std::runtime_error("Original SAP GPU report publication task order changed");
    const std::uint32_t rawRetained=overflow?8u:6u;
    std::vector<Pair> expectedFound,expectedLost;
    auto aggregate=[&](std::uint32_t a,std::uint32_t b) {
        return useVolumeData && (a==4u||b==4u||a==10u||b==10u);
    };
    for(std::uint32_t i=0u;i<rawRetained;++i)
        if(!aggregate(createdInputs[i].first,createdInputs[i].second))
            expectedFound.push_back(created[i]);
    for(std::uint32_t i=0u;i<removedInputs.size();++i)
        if(!aggregate(removedInputs[i].first,removedInputs[i].second))
            expectedLost.push_back(removed[i]);
    if(counts.found_actors!=expectedFound.size() ||
       counts.lost_actors!=expectedLost.size() ||
       counts.pairs_needed!=(overflow?10u:6u) ||
       counts.overflow!=overflow ||
       fetched.sharedFoundAggPairIndex!=(useVolumeData?2u:0u) ||
       fetched.sharedLostAggPairIndex!=(useVolumeData?1u:0u))
        throw std::runtime_error("Original SAP source classifier reported wrong actor or aggregate counts");
    for(std::size_t i=0;i<expectedFound.size();++i)
        if(publishedFound[i].mVolA!=expectedFound[i].mVolA ||
           publishedFound[i].mVolB!=expectedFound[i].mVolB)
            throw std::runtime_error("Source found-actor reports have wrong stable ordering");
    for(std::size_t i=0;i<expectedLost.size();++i)
        if(publishedLost[i].mVolA!=expectedLost[i].mVolA ||
           publishedLost[i].mVolB!=expectedLost[i].mVolB)
            throw std::runtime_error("Source lost-actor reports have wrong stable ordering");
    if(sourceMappedHeap) {
        // Source GPU report kernels wrote directly to this ORIGINAL
        // PhysX host buddy heap (or its capability-gated staged backing).
        if(mappedCallback.device_address(originalFound,sizeof(Pair)*kCapacity)!=mappedFound ||
           mappedCallback.device_address(originalLost,sizeof(Pair)*kCapacity)!=mappedLost)
            throw std::runtime_error("PhysX mapped host report alias moved across fetch");
        mappedHostHeap->deallocateDeferred(originalFound);
        mappedHostHeap->deallocateDeferred(originalLost);
        mappedHostHeap->flushDeferredDeallocs();
        sourceHeapManager.reset();
        if(mappedCallback.live_roots()!=0u || deviceRoots.live_roots()!=0u ||
           aborted)
            throw std::runtime_error("Original PhysX source report heap leaked or entered abort mode");
    }
    for(const auto id:owned)context.destroy_buffer(id);
    context.destroy_buffer(root);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 11)
            throw std::invalid_argument("Expected legacy five and native five PhysX report shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        auto shader = [](const char* path) {
            ShaderBinary out;
#if defined(FYZZXYZZY_TEST_D3D12)
            out.format = ShaderFormat::Dxil;
#else
            out.format = ShaderFormat::Spirv;
#endif
            out.bytes = load_shader(path);
            return out;
        };
        BroadphaseReportClassificationArenaStage classify(
            context, shader(argv[1]), shader(argv[2]),
            shader(argv[4]), shader(argv[5]));
        BroadphaseReportClassificationArenaStage portable(
            context, shader(argv[1]), shader(argv[2]),
            shader(argv[4]), shader(argv[5]), true);
        BroadphaseCopyReportsArenaStage copy(context, shader(argv[3]));
        SourceSapReportNativeStage native(context,shader(argv[6]),shader(argv[7]),
            shader(argv[8]),shader(argv[9]),shader(argv[10]));
        SourceSapReportNativeStage nativePortable(context,shader(argv[6]),shader(argv[7]),
            shader(argv[8]),shader(argv[9]),shader(argv[10]),true);
        const auto& caps = context.device_info();
        const bool capable = caps.supports_wave_arithmetic &&
            caps.subgroup_size >= 4u && 256u % caps.subgroup_size == 0u;
        if (classify.using_wave_reports() != capable ||
            portable.using_wave_reports() ||
            native.using_wave_reports()!=capable || nativePortable.using_wave_reports())
            throw std::runtime_error("Report classification skipped native-wave capability selection");

        // Compare entire original 2+1 kernel report stream against the
        // independent CPU source oracle, including cross-block stable
        // order, empty input, overflow, aggregate/actor split and tails.
        for (auto* stage : {&classify, &portable}) {
            negative_cases(*stage);
            run_case(context, *stage, copy, 0u, 0u, 0u, 0u, false);
            run_case(context, *stage, copy, 0u, 0u, 8u, 2u, true);
            run_case(context, *stage, copy, 1u, 1u, 1u, 1u, false);
            run_case(context, *stage, copy, 12u, 8u, 12u, 2u, true);
            run_case(context, *stage, copy, 17u, 39u, 16u, 3u, true);
            run_case(context, *stage, copy, 21u, 12u, 0u, 0u, true);
            run_case(context, *stage, copy, 129u, 257u, 300u, 4u, true);
            run_case(context, *stage, copy, 8193u, 19u, 8193u, 100u, true);
            run_case(context, *stage, copy, 16385u, 1u, 16385u, 160u, false);
            run_original_source_report_fetch(context,*stage,copy,true,false);
            run_original_source_report_fetch(context,*stage,copy,true,true);
            run_original_source_report_fetch(context,*stage,copy,false,false);
        }
        for(auto* stage:{&native,&nativePortable}) {
            for(const bool split:{false,true}) {
                run_original_source_report_fetch(context,classify,copy,true,false,stage,split);
                run_original_source_report_fetch(context,classify,copy,true,true,stage,split);
                run_original_source_report_fetch(context,classify,copy,false,false,stage,split);
            }
        }
        std::cout << "PhysX native-wave/fallback report classification + publication PASS\\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "PhysX report classification FAILED: " << ex.what() << '\n';
        return 1;
    }
}
