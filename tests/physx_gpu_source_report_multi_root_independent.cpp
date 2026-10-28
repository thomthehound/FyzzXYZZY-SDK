// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_sap_scene_reports.h"
#include "physx_port/gpubroadphase/include/PxgBroadPhasePairReport.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
using Pair = physx::PxgBroadPhasePair;
using Address = CudaDeviceAddressMap::Address;
constexpr std::uint32_t kCapacity = 8u;
constexpr std::uint32_t kVolumes = 32u;
constexpr std::uint32_t kVolumeStride = 16u;
constexpr std::uint32_t kAggregateOffset = 8u;
static_assert(sizeof(Pair) == 8u);

void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
ShaderBinary shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Missing source report shader");
    const std::vector<char> data{std::istreambuf_iterator<char>{stream},
                                 std::istreambuf_iterator<char>{}};
    ShaderBinary binary;
#if defined(FYZZXYZZY_TEST_D3D12)
    binary.format = ShaderFormat::Dxil;
#else
    binary.format = ShaderFormat::Spirv;
#endif
    binary.bytes.assign(data.begin(), data.end());
    return binary;
}

// Original virtual addresses remain the single point of binding. Both layouts
// exercise the production resolver; no report data is staged on the CPU.
struct Roots {
    Context& context;
    bool shared;
    CudaDeviceAddressMap addresses;
    BufferId backing = 0;
    std::size_t cursor = 0;
    std::vector<BufferId> independent;
    explicit Roots(Context& ctx, bool useShared) : context(ctx), shared(useShared) {
        if (shared) backing = context.create_buffer(65536u, MemoryPlacement::DeviceLocal);
    }
    ~Roots() {
        for (auto id : independent) context.destroy_buffer(id);
        if (backing) context.destroy_buffer(backing);
    }
    Address make(std::size_t bytes, const void* initial = nullptr) {
        BufferId buffer;
        std::size_t offset = 0;
        Address address;
        if (shared) {
            cursor = (cursor + 255u) & ~std::size_t{255u};
            check(bytes && cursor + bytes <= 65536u, "Shared root exhausted");
            buffer = backing;
            offset = cursor;
            address = addresses.register_slice(buffer, offset, bytes);
            cursor += bytes;
        } else {
            buffer = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
            independent.push_back(buffer);
            address = addresses.register_allocation(buffer, bytes);
        }
        if (initial) context.upload(buffer, offset, initial, bytes);
        return address;
    }
    template<class T> void read(Address from, T& result) {
        const auto resource = addresses.resolve(from, sizeof(result));
        context.download(resource.buffer, resource.offset, &result, sizeof(result));
    }
};

template<class T> T* ptr(Address address) {
    return reinterpret_cast<T*>(static_cast<std::uintptr_t>(address));
}
void expect_pairs(const std::array<Pair, kCapacity>& actual,
                  const std::array<Pair, kCapacity>& originals,
                  std::initializer_list<std::size_t> indices,
                  const char* why) {
    std::size_t i = 0;
    for (const auto index : indices) {
        if (actual[i].mVolA != originals[index].mVolA ||
            actual[i].mVolB != originals[index].mVolB)
            throw std::runtime_error(why);
        ++i;
    }
}

void run(Context& context, SourceSapReportNativeStage& stage,
         bool shared, bool volumes, bool overflow) {
    Roots root(context, shared);
    std::array<Pair, kCapacity> found{
        Pair{1u,2u},Pair{3u,4u},Pair{5u,6u},Pair{7u,8u},
        Pair{9u,10u},Pair{11u,12u},Pair{13u,14u},Pair{15u,16u}};
    std::array<Pair, kCapacity> lost{
        Pair{1u,3u},Pair{2u,4u},Pair{7u,9u},Pair{12u,13u}};
    std::array<std::uint32_t, kVolumes * 4u> volumeData{};
    for (std::size_t i = 0; i < kVolumes; ++i)
        volumeData[i * 4u + kAggregateOffset / 4u] = 0xffffffffu;
    volumeData[4u * 4u + kAggregateOffset / 4u] = 2u;
    volumeData[10u * 4u + kAggregateOffset / 4u] = 2u;
    auto rawFound = root.make(sizeof(found), found.data());
    auto rawLost = root.make(sizeof(lost), lost.data());
    auto aggFound = root.make(sizeof(found));
    auto aggLost = root.make(sizeof(lost));
    auto actorFound = root.make(sizeof(found));
    auto actorLost = root.make(sizeof(lost));
    auto mappedFound = root.make(sizeof(found));
    auto mappedLost = root.make(sizeof(lost));
    std::array<Address, 4u> blocks{};
    for (auto& b : blocks) b = root.make(32u * sizeof(std::uint32_t));
    auto volumeAddress = root.make(sizeof(volumeData), volumeData.data());

    physx::PxgBroadPhaseDesc source{};
    source.max_found_lost_pairs = kCapacity;
    source.max_found_lost_agg_pairs = kCapacity;
    source.sharedFoundPairIndex = overflow ? 10u : 6u;
    source.sharedLostPairIndex = 4u;
    source.foundPairReport = ptr<Pair>(rawFound);
    source.lostPairReport = ptr<Pair>(rawLost);
    source.foundAggPairReport = ptr<Pair>(aggFound);
    source.lostAggPairReport = ptr<Pair>(aggLost);
    source.foundActorPairReport = ptr<Pair>(actorFound);
    source.lostActorPairReport = ptr<Pair>(actorLost);
    source.foundPairReportMap = ptr<Pair>(mappedFound);
    source.lostPairReportMap = ptr<Pair>(mappedLost);
    for (std::size_t i = 0; i < 2u; ++i) {
        source.aggReportBlock[i] = ptr<std::uint32_t>(blocks[2u * i]);
        source.actorReportBlock[i] = ptr<std::uint32_t>(blocks[2u * i + 1u]);
    }
    if (volumes)
        source.aabbMngr_volumeData = ptr<physx::Bp::VolumeData>(volumeAddress);
    const Address descriptor = root.make(sizeof(source), &source);
    const SourceSapReportNativeLayout layout{
        descriptor, volumes ? kVolumes : 0u,
        volumes ? kVolumeStride : 0u, volumes ? kAggregateOffset : 0u};

    std::vector<BatchCommand> commands;
    auto invalid = source;
    invalid.lostPairReportMap = ptr<Pair>(Address{0x100u});
    bool rejected = false;
    try { stage.append(commands, invalid, root.addresses, layout); }
    catch (const std::exception&) { rejected = true; }
    check(rejected && commands.empty(), "Invalid report resource partially appended work");

    stage.append(commands, source, root.addresses, layout);
    check(commands.size() == 3u, "Report stage did not append original three passes");
    for (std::uint32_t pass = 0; pass < 3u; ++pass) {
        const auto& command = std::get<DispatchCommand>(commands[pass]);
        check(command.groups.x == (pass == 2u ? 64u : 32u) &&
              command.groups.y == (pass == 2u ? 1u : 4u) &&
              command.buffers.size() == (pass == 0u ? 8u : pass == 1u ? 13u : 7u),
              "Report stage altered dispatch geometry or bindings");
    }
    context.wait_submission(context.submit_batch(commands));

    physx::PxgBroadPhaseDesc completed{};
    root.read(descriptor, completed);
    check(completed.sharedFoundPairIndex == (overflow ? 10u : 6u) &&
          completed.sharedLostPairIndex == 4u &&
          completed.sharedFoundAggPairIndex == (volumes ? 2u : 0u) &&
          completed.sharedLostAggPairIndex == (volumes ? 1u : 0u) &&
          completed.found_lost_pairs_overflow_flags == overflow,
          "Original source report counters or overflow changed");

    std::array<Pair, kCapacity> foundActors{}, lostActors{}, foundAggregates{}, lostAggregates{};
    root.read(mappedFound, foundActors);
    root.read(mappedLost, lostActors);
    root.read(aggFound, foundAggregates);
    root.read(aggLost, lostAggregates);
    if (volumes) {
        if (overflow) expect_pairs(foundActors, found, {0,2,3,5,6,7}, "Overflow actor ordering changed");
        else expect_pairs(foundActors, found, {0,2,3,5}, "Found actor ordering changed");
        expect_pairs(lostActors, lost, {0,2,3}, "Lost actor ordering changed");
        expect_pairs(foundAggregates, found, {1,4}, "Found aggregate ordering changed");
        expect_pairs(lostAggregates, lost, {1}, "Lost aggregate ordering changed");
    } else {
        if (overflow) expect_pairs(foundActors, found, {0,1,2,3,4,5,6,7}, "Overflow ordering changed");
        else expect_pairs(foundActors, found, {0,1,2,3,4,5}, "Found ordering changed");
        expect_pairs(lostActors, lost, {0,1,2,3}, "Lost ordering changed");
    }
    if (!overflow) {
        // Map source volume handles to intentionally reversed Sc actor IDs.
        // GPU report order cannot be assumed to be Sc pair order.
        std::array<fyzz::BodyId,kVolumes> volumeToActor{};
        for (std::size_t i=0; i<volumeToActor.size(); ++i)
            volumeToActor[i] = static_cast<fyzz::BodyId>(31u-i);
        const std::size_t foundCount = 6u-(volumes ? 2u : 0u);
        const std::size_t lostCount = 4u-(volumes ? 1u : 0u);
        const std::span<const Pair> gpuFound{foundActors.data(),foundCount};
        const std::span<const Pair> gpuLost{lostActors.data(),lostCount};
        std::vector<fyzz::native::Pair> scFound,scLost;
        lower_source_sap_actor_reports_to_sc(
            gpuFound,gpuLost,volumeToActor,scFound,scLost);
        const std::vector<fyzz::native::Pair> expectedFound = volumes
            ? std::vector<fyzz::native::Pair>{{19u,20u},{23u,24u},
                                               {25u,26u},{29u,30u}}
            : std::vector<fyzz::native::Pair>{{19u,20u},{21u,22u},
                                               {23u,24u},{25u,26u},
                                               {27u,28u},{29u,30u}};
        const std::vector<fyzz::native::Pair> expectedLost = volumes
            ? std::vector<fyzz::native::Pair>{{18u,19u},{22u,24u},{28u,30u}}
            : std::vector<fyzz::native::Pair>{{18u,19u},{22u,24u},
                                               {27u,29u},{28u,30u}};
        check(scFound == expectedFound && scLost == expectedLost,
              "Original GPU actor reports did not lower into canonical Sc keys");

        // A corrupt lost report must reject BEFORE modifying either
        // already-published Sc stream.
        const std::array<Pair,1> badLost{Pair{1u,kVolumes}};
        bool rejectedSc = false;
        try {
            lower_source_sap_actor_reports_to_sc(
                gpuFound,badLost,volumeToActor,scFound,scLost);
        } catch (const std::out_of_range&) { rejectedSc = true; }
        check(rejectedSc && scFound == expectedFound && scLost == expectedLost,
              "Invalid GPU lost report partially changed Sc publication");

        // Source volumes need not be in actor order; duplicate raw pairs
        // must collapse to one canonical Sc key without a CPU SAP pass.
        const std::array<Pair,3> duplicated{
            Pair{7u,8u},Pair{1u,2u},Pair{7u,8u}};
        lower_source_sap_actor_reports_to_sc(
            duplicated,std::span<const Pair>{},volumeToActor,scFound,scLost);
        check(scFound == std::vector<fyzz::native::Pair>{
                  {23u,24u},{29u,30u}} && scLost.empty(),
              "Source duplicate GPU pair records escaped Sc canonicalization");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("Require original five report shader files");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        auto first = shader(argv[1]);
        auto second = shader(argv[2]);
        auto copy = shader(argv[3]);
        auto firstWave = shader(argv[4]);
        auto secondWave = shader(argv[5]);
        SourceSapReportNativeStage automatic(
            context, std::move(first), std::move(second), std::move(copy),
            std::move(firstWave), std::move(secondWave));
        for (bool shared : {false, true})
            for (bool volumes : {false, true})
                for (bool overflow : {false, true})
                    run(context, automatic, shared, volumes, overflow);
        std::cout << "Source report multi-root GPU classification PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "Source report multi-root GPU classification FAILED: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
