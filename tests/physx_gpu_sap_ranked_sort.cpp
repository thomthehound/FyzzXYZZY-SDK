// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/radix_count_arena.h"
#include "physx_gpu/radix_scatter_arena.h"
#include "physx_gpu/sap_ranked_sort_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::RadixCountArenaDesc;
using fyzz::physx_gpu::RadixCountArenaStage;
using fyzz::physx_gpu::RadixScatterArenaDesc;
using fyzz::physx_gpu::RadixScatterArenaStage;
using fyzz::physx_gpu::SapRankedSortArenaDesc;
using fyzz::physx_gpu::SapRankedSortArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;

std::vector<std::uint8_t> load_shader(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX WR sort shader not found");
    const std::vector<char> buffer{std::istreambuf_iterator<char>{in},
                                   std::istreambuf_iterator<char>{}};
    return {buffer.begin(), buffer.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t rank_desc = 0;
    std::uint32_t count_desc[2]{};
    std::uint32_t scatter_desc[2]{};
    std::uint32_t projection[3][2]{};
    std::uint32_t ranks[3][2]{};
    std::uint32_t previous_handles[3]{};
    std::uint32_t current_handles[3]{};
    std::uint32_t histogram[3]{};
    std::uint32_t live = 0;
    std::uint32_t padded = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> original[3];

    std::uint32_t reserve(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kUnused);
        const auto offset = arena.size();
        if (offset + words > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("PhysX WR fixture arena overflow");
        arena.resize(offset + words, kUnused);
        return static_cast<std::uint32_t>(offset * 4u);
    }
    void put(std::uint32_t offset, const void* value, std::size_t bytes) {
        if (offset % 4u || bytes % 4u ||
            offset + bytes > arena.size() * 4u)
            throw std::runtime_error("PhysX WR fixture descriptor exceeds arena");
        std::memcpy(arena.data() + offset / 4u, value, bytes);
    }
    std::uint32_t& word(std::uint32_t offset, std::uint32_t index) {
        return arena[offset / 4u + index];
    }
};

std::uint32_t key(std::uint32_t i, std::uint32_t axis) {
    if ((i + axis) % 9u == 0u) return 0x80000000u;
    if ((i + axis) % 7u == 0u) return 0u;
    if ((i + axis) % 5u == 0u) return 0x10000000u;
    return 0x20000000u + (((i * 37u + axis * 19u) % 31u) << 4);
}

Packed make_fixture(std::uint32_t boxes) {
    Packed p;
    p.live = boxes * 2u;
    p.padded = (p.live + 3u) & ~3u;
    p.rank_desc = p.reserve(sizeof(SapRankedSortArenaDesc) / 4u);
    for (unsigned parity = 0; parity < 2; ++parity) {
        p.count_desc[parity] = p.reserve(sizeof(RadixCountArenaDesc) / 4u);
        p.scatter_desc[parity] = p.reserve(sizeof(RadixScatterArenaDesc) / 4u);
    }
    SapRankedSortArenaDesc wr{};
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        const auto size = std::max(4u, p.padded);
        for (std::uint32_t parity = 0; parity < 2u; ++parity) {
            p.projection[axis][parity] = p.reserve(size);
            p.ranks[axis][parity] = p.reserve(size);
        }
        p.previous_handles[axis] = p.reserve(size);
        p.current_handles[axis] = p.reserve(size);
        p.histogram[axis] = p.reserve(16u * 32u);
        for (std::uint32_t i = 0; i < p.live; ++i) {
            const auto k = key(i, axis);
            p.word(p.projection[axis][0], i) = k;
            // Stable tie-breaking must preserve the original endpoint
            // positions, not the numeric/encoded handle order.
            const auto tagged = ((i / 2u + axis * 104729u) << 3u) |
                ((i % 2u == 0u) ? 1u : 0u) |
                ((i % 7u == 0u) ? 2u : 0u);
            p.word(p.previous_handles[axis], i) = tagged;
            p.original[axis].emplace_back(k, i);
        }
        for (std::uint32_t i = p.live; i < p.padded; ++i) {
            p.word(p.projection[axis][0], i) = 0xffffffffu;
        }
        wr.axes[axis].projection_ranks_offset = p.ranks[axis][0];
        wr.axes[axis].previous_handles_offset = p.previous_handles[axis];
        wr.axes[axis].current_handles_offset = p.current_handles[axis];
    }
    p.put(p.rank_desc, &wr, sizeof(wr));
    for (unsigned parity = 0; parity < 2; ++parity) {
        RadixCountArenaDesc count{};
        RadixScatterArenaDesc scatter{};
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            count.axes[axis].input_keys_offset = p.projection[axis][parity];
            count.axes[axis].radix_counts_offset = p.histogram[axis];
            scatter.axes[axis].input_keys_offset = p.projection[axis][parity];
            scatter.axes[axis].input_ranks_offset = p.ranks[axis][parity];
            scatter.axes[axis].output_keys_offset =
                p.projection[axis][1u - parity];
            scatter.axes[axis].output_ranks_offset =
                p.ranks[axis][1u - parity];
            scatter.axes[axis].radix_counts_offset = p.histogram[axis];
        }
        p.put(p.count_desc[parity], &count, sizeof(count));
        p.put(p.scatter_desc[parity], &scatter, sizeof(scatter));
    }
    return p;
}

void verify(const Packed& p, const std::vector<std::uint32_t>& actual) {
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        auto expected = p.original[axis];
        for (std::uint32_t i = p.live; i < p.padded; ++i)
            expected.emplace_back(0xffffffffu, i);
        std::stable_sort(expected.begin(), expected.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        for (std::uint32_t i = 0; i < p.padded; ++i) {
            if (actual[p.projection[axis][0] / 4u + i] != expected[i].first ||
                actual[p.ranks[axis][0] / 4u + i] != expected[i].second)
                throw std::runtime_error("WR source rank sort differs from stable oracle; axis " +
                                         std::to_string(axis) + ", index " + std::to_string(i));
        }
        for (std::uint32_t i = 0; i < p.live; ++i) {
            const auto source_index = expected[i].second;
            if (actual[p.current_handles[axis] / 4u + i] !=
                p.arena[p.previous_handles[axis] / 4u + source_index])
                throw std::runtime_error("PhysX WR handle gather mismatches sorted source rank");
        }
        for (std::uint32_t i = p.live; i < std::max(4u, p.padded); ++i) {
            if (actual[p.current_handles[axis] / 4u + i] != kUnused)
                throw std::runtime_error("PhysX WR handle gather wrote past live projections");
        }
        for (std::uint32_t i = 0; i < std::max(4u, p.padded); ++i) {
            if (actual[p.previous_handles[axis] / 4u + i] !=
                p.arena[p.previous_handles[axis] / 4u + i])
                throw std::runtime_error("PhysX WR previous-frame handles modified");
        }
    }
    for (std::uint32_t i = 0; i < sizeof(SapRankedSortArenaDesc) / 4u; ++i) {
        if (actual[p.rank_desc / 4u + i] != p.arena[p.rank_desc / 4u + i])
            throw std::runtime_error("PhysX WR persistent descriptor modified");
    }
}

void run_case(Context& context, SapRankedSortArenaStage& wr,
              RadixCountArenaStage& count_stage, RadixScatterArenaStage& scatter,
              std::uint32_t boxes) {
    auto p = make_fixture(boxes);
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0, p.arena.data(), bytes);
    std::vector<BatchCommand> commands;
    wr.append_source_sequence(commands, count_stage, scatter,
                              gpu, bytes, p.rank_desc, p.count_desc,
                              p.scatter_desc, boxes);
    const auto expected = boxes ? 18u : 0u;
    if (commands.size() != expected)
        throw std::runtime_error("PhysX WR source sequence changed");
    if (boxes) {
        if (std::get<DispatchCommand>(commands[0]).groups.x != 64u ||
            std::get<DispatchCommand>(commands[17]).groups.x != 64u)
            throw std::runtime_error("PhysX WR original grid width changed");
        for (std::uint32_t pass = 0; pass < 8u; ++pass) {
            const auto& count = std::get<DispatchCommand>(commands[pass * 2u + 1u]);
            const auto& rank = std::get<DispatchCommand>(commands[pass * 2u + 2u]);
            if (count.groups.x != 32u || count.groups.y != 3u ||
                rank.groups.x != 32u || rank.groups.y != 3u)
                throw std::runtime_error("PhysX WR radix geometry changed");
        }
    }
    commands.emplace_back(CopyCommand{gpu, 0, mirror, 0, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(p.arena.size());
    context.download(mirror, 0, actual.data(), bytes);
    if (boxes) verify(p, actual);
    else if (actual != p.arena)
        throw std::runtime_error("PhysX WR zero-work path changed arena");
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(Context& context, SapRankedSortArenaStage& wr,
                    RadixCountArenaStage& count_stage,
                    RadixScatterArenaStage& scatter) {
    // A caller may already have recorded work before constructing the
    // source-ordered WR chain. Rejection must preserve the entire prefix.
    auto prefix = [] {
        std::vector<BatchCommand> ops;
        ops.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        return ops;
    };
    auto check_prefix = [](const std::vector<BatchCommand>& ops) {
        if (ops.size() != 1u || !std::holds_alternative<CopyCommand>(ops[0]))
            throw std::runtime_error("Rejected PhysX WR sort leaked partial GPU commands");
        const auto& copy = std::get<CopyCommand>(ops[0]);
        if (copy.source != 1u || copy.destination != 2u ||
            copy.source_offset != 0u || copy.destination_offset != 0u ||
            copy.bytes != 4u)
            throw std::runtime_error("Rejected PhysX WR sort modified prior batch work");
    };
    const std::uint32_t valid[2]{0u, 128u};
    for (const auto bad : {1u, 128u, 256u}) {
        auto ops = prefix();
        bool rejected = false;
        try {
            wr.append_source_sequence(ops, count_stage, scatter, 1u, 160u,
                                      bad, valid, valid, 1u);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected)
            throw std::runtime_error("Invalid PhysX WR top-level descriptor accepted");
        check_prefix(ops);
    }
    // Every parity and both delegated stages must be transactional, even
    // when the error occurs after several valid source-order dispatches.
    for (std::uint32_t parity = 0u; parity < 2u; ++parity) {
        for (std::uint32_t kind = 0u; kind < 2u; ++kind) {
            for (const auto bad : {1u, 480u}) {
                std::uint32_t count_offsets[2]{0u, 128u};
                std::uint32_t scatter_offsets[2]{0u, 128u};
                if (kind == 0u) count_offsets[parity] = bad;
                else scatter_offsets[parity] = bad;
                auto ops = prefix();
                bool rejected = false;
                try {
                    wr.append_source_sequence(ops, count_stage, scatter,
                                              1u, 512u, 0u,
                                              count_offsets, scatter_offsets, 3u);
                } catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected)
                    throw std::runtime_error("Invalid PhysX WR radix descriptor accepted");
                check_prefix(ops);
            }
        }
    }
    // The source's no-previous-boxes gate must not enqueue anything.
    {
        auto ops = prefix();
        const std::uint32_t invalid[2]{1u, 480u};
        wr.append_source_sequence(ops, count_stage, scatter, 1u, 512u,
                                  0u, invalid, invalid, 0u);
        check_prefix(ops);
    }
    (void)context;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument(
                "Expected radix count, scatter, initialize-ranks, update-handles shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        auto shader = [](const char* file) {
            ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
            result.format = ShaderFormat::Dxil;
#else
            result.format = ShaderFormat::Spirv;
#endif
            result.bytes = load_shader(file);
            return result;
        };
        RadixCountArenaStage count(context, shader(argv[1]), shader(argv[5]));
        RadixScatterArenaStage scatter(context, shader(argv[2]), shader(argv[6]));
        const auto& caps = context.device_info();
        const bool expectWave = caps.supports_wave_arithmetic &&
            caps.subgroup_size >= 4u && 1024u % caps.subgroup_size == 0u;
        if (count.using_wave_count() != expectWave ||
            scatter.using_wave_scan() != expectWave)
            throw std::runtime_error("SAP integration bypassed available packed radix fast path");
        SapRankedSortArenaStage wr(context, shader(argv[3]), shader(argv[4]));
        negative_cases(context, wr, count, scatter);
        run_case(context, wr, count, scatter, 0u);
        run_case(context, wr, count, scatter, 1u);
        run_case(context, wr, count, scatter, 2u);
        run_case(context, wr, count, scatter, 3u);
        run_case(context, wr, count, scatter, 129u);
        run_case(context, wr, count, scatter, 257u);
        std::cout << "PhysX WR initialize/radix/remap source sequence PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX WR source-order oracle FAILED: " << e.what() << '\n';
        return 1;
    }
}
