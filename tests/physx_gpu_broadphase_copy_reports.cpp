// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_copy_reports_arena.h"

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

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::BroadphaseCopyReportsArenaDesc;
using fyzz::physx_gpu::BroadphaseCopyReportsArenaStage;

constexpr std::uint32_t kCanary = 0xcdcdcdcd;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX copy-reports shader missing");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t descriptor_offset = 0;
    BroadphaseCopyReportsArenaDesc desc{};

    std::uint32_t allocate(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kCanary);
        const auto start = arena.size();
        arena.resize(start + words, kCanary);
        return static_cast<std::uint32_t>(start * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t index = 0u) {
        return arena[offset / 4u + index];
    }
};

Packed pack(std::uint32_t rawFound, std::uint32_t rawLost,
            std::uint32_t maxPairs, std::uint32_t foundAgg,
            std::uint32_t lostAgg) {
    const std::uint32_t clippedFound = std::min(rawFound, maxPairs);
    const std::uint32_t clippedLost = std::min(rawLost, maxPairs);
    if (foundAgg > clippedFound || lostAgg > clippedLost)
        throw std::invalid_argument("Aggregate counts exceed stored pairs");

    Packed p;
    p.descriptor_offset = p.allocate(sizeof(p.desc) / 4u);
    p.desc.raw_pair_counts_offset = p.allocate(2u);
    p.desc.aggregate_pair_counts_offset = p.allocate(2u);
    p.desc.found_actor_pairs_offset = p.allocate(std::max(2u, maxPairs * 2u));
    p.desc.lost_actor_pairs_offset = p.allocate(std::max(2u, maxPairs * 2u));
    p.desc.found_map_offset = p.allocate(std::max(2u, maxPairs * 2u));
    p.desc.lost_map_offset = p.allocate(std::max(2u, maxPairs * 2u));
    p.desc.overflow_flag_offset = p.allocate(1u);
    p.desc.report_capacity = maxPairs;
    p.at(p.desc.raw_pair_counts_offset) = rawFound;
    p.at(p.desc.raw_pair_counts_offset, 1u) = rawLost;
    p.at(p.desc.aggregate_pair_counts_offset) = foundAgg;
    p.at(p.desc.aggregate_pair_counts_offset, 1u) = lostAgg;
    p.at(p.desc.overflow_flag_offset) = 0x12345678u;
    for (std::uint32_t i = 0u; i < clippedFound - foundAgg; ++i) {
        p.at(p.desc.found_actor_pairs_offset, i * 2u) = i;
        p.at(p.desc.found_actor_pairs_offset, i * 2u + 1u) = maxPairs + i + 1u;
    }
    for (std::uint32_t i = 0u; i < clippedLost - lostAgg; ++i) {
        p.at(p.desc.lost_actor_pairs_offset, i * 2u) = i + 71u;
        p.at(p.desc.lost_actor_pairs_offset, i * 2u + 1u) = maxPairs + i + 72u;
    }
    std::memcpy(p.arena.data() + p.descriptor_offset / 4u,
                &p.desc, sizeof(p.desc));
    return p;
}

void run_case(Context& context, BroadphaseCopyReportsArenaStage& stage,
              std::uint32_t rawFound, std::uint32_t rawLost,
              std::uint32_t maxPairs, std::uint32_t foundAgg,
              std::uint32_t lostAgg) {
    auto data = pack(rawFound, rawLost, maxPairs, foundAgg, lostAgg);
    auto expected = data.arena;
    const auto actualFound = std::min(rawFound, maxPairs) - foundAgg;
    const auto actualLost = std::min(rawLost, maxPairs) - lostAgg;
    for (std::uint32_t i = 0u; i < actualFound * 2u; ++i) {
        expected[data.desc.found_map_offset / 4u + i] =
            data.at(data.desc.found_actor_pairs_offset, i);
    }
    for (std::uint32_t i = 0u; i < actualLost * 2u; ++i) {
        expected[data.desc.lost_map_offset / 4u + i] =
            data.at(data.desc.lost_actor_pairs_offset, i);
    }
    const bool overflow = rawFound > maxPairs || rawLost > maxPairs;
    expected[data.desc.overflow_flag_offset / 4u] = overflow ? 1u : 0u;

    const auto bytes = static_cast<std::uint32_t>(data.arena.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto staging = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, data.arena.data(), bytes);
    std::vector<BatchCommand> commands;
    stage.append(commands, gpu, bytes, data.descriptor_offset);
    if (commands.size() != 1u)
        throw std::runtime_error("PhysX copy-reports must enqueue one dispatch");
    const auto& dispatch = std::get<DispatchCommand>(commands[0]);
    if (dispatch.groups.x != 64u || dispatch.groups.y != 1u ||
        dispatch.groups.z != 1u || dispatch.push_constants.size() != 32u)
        throw std::runtime_error("PhysX BP_COPY_REPORTS dispatch geometry changed");
    commands.emplace_back(CopyCommand{gpu, 0u, staging, 0u, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> observed(data.arena.size());
    context.download(staging, 0u, observed.data(), bytes);
    if (observed != expected) {
        const auto mismatch = std::mismatch(observed.begin(), observed.end(),
                                            expected.begin());
        throw std::runtime_error(
            "PhysX report publication mismatch at word " +
            std::to_string(std::distance(observed.begin(), mismatch.first)) +
            " (found " + std::to_string(rawFound) +
            ", lost " + std::to_string(rawLost) +
            ", capacity " + std::to_string(maxPairs) + ")");
    }
    context.destroy_buffer(staging);
    context.destroy_buffer(gpu);
}

void negative_cases(BroadphaseCopyReportsArenaStage& stage) {
    for (const auto offset : {1u, 128u, 256u}) {
        std::vector<BatchCommand> ops;
        ops.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        bool rejected = false;
        try { stage.append(ops, 1u, 160u, offset); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || ops.size() != 1u ||
            !std::holds_alternative<CopyCommand>(ops[0]))
            throw std::runtime_error("Invalid report descriptor changed caller work");
    }
    std::vector<BatchCommand> ops;
    bool rejected = false;
    try { stage.append(ops, 0u, 0u, 0u); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || !ops.empty())
        throw std::runtime_error("Empty PhysX report skipped required dispatch validation");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Expected BP_COPY_REPORTS shader path");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        shader.format = ShaderFormat::Dxil;
#else
        shader.format = ShaderFormat::Spirv;
#endif
        shader.bytes = load_shader(argv[1]);
        BroadphaseCopyReportsArenaStage stage(context, std::move(shader));
        negative_cases(stage);
        run_case(context, stage, 0u, 0u, 0u, 0u, 0u);
        run_case(context, stage, 0u, 0u, 4u, 0u, 0u);
        run_case(context, stage, 1u, 0u, 1u, 0u, 0u);
        run_case(context, stage, 12u, 8u, 12u, 2u, 7u);
        run_case(context, stage, 17u, 39u, 16u, 3u, 9u);
        run_case(context, stage, 21u, 12u, 0u, 0u, 0u);
        run_case(context, stage, 16385u, 1u, 16385u, 3u, 0u);
        run_case(context, stage, 1u, 16385u, 16385u, 0u, 4u);
        std::cout << "PhysX BP_COPY_REPORTS source oracle PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX BP_COPY_REPORTS FAILED: " << error.what() << '\n';
        return 1;
    }
}
