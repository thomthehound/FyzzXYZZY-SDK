// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/radix_count_arena.h"
#include "physx_gpu/radix_scatter_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
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
using fyzz::physx_gpu::RadixCountArenaDesc;
using fyzz::physx_gpu::RadixCountArenaStage;
using fyzz::physx_gpu::RadixScatterArenaDesc;
using fyzz::physx_gpu::RadixScatterArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kBlocks = 32;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX radix shader not found");
    const std::vector<char> values{std::istreambuf_iterator<char>{in},
                                   std::istreambuf_iterator<char>{}};
    return {values.begin(), values.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t count_descriptor_offsets[2]{};
    std::uint32_t scatter_descriptor_offsets[2]{};
    std::uint32_t keys_offsets[3][2]{};
    std::uint32_t ranks_offsets[3][2]{};
    std::uint32_t counts_offsets[3]{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> original[3];

    std::uint32_t reserve(std::size_t words) {
        while (arena.size() % 32u) arena.push_back(kUnused);
        const auto index = arena.size();
        arena.resize(index + words, kUnused);
        return static_cast<std::uint32_t>(index * 4u);
    }

    void put(std::uint32_t offset, const void* bytes, std::size_t size) {
        if (offset % 4u || size % 4u || (offset + size) > arena.size() * 4u)
            throw std::runtime_error("Invalid PhysX fixture arena write");
        std::memcpy(arena.data() + offset / 4u, bytes, size);
    }
};

std::uint32_t key_for(std::uint32_t i, std::uint32_t axis) {
    if (i % 13u == 0u) return 0xffffffffu;
    if (i % 11u == 0u) return 0x00000000u;
    if (i % 7u == 0u) return 0x80000000u;
    if (i % 5u == 0u) return 0x10203040u + axis;
    return ((i / 3u) * 2654435761u) ^ (axis * 0x9e3779b9u);
}

Packed pack(std::uint32_t num_keys, bool clustered = false) {
    Packed p;
    for (std::uint32_t parity = 0; parity < 2; ++parity) {
        p.count_descriptor_offsets[parity] =
            p.reserve(sizeof(RadixCountArenaDesc) / 4u);
        p.scatter_descriptor_offsets[parity] =
            p.reserve(sizeof(RadixScatterArenaDesc) / 4u);
    }
    const auto capacity = std::max(4u, (num_keys + 3u) & ~3u);
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        for (std::uint32_t parity = 0; parity < 2; ++parity) {
            p.keys_offsets[axis][parity] = p.reserve(capacity);
            p.ranks_offsets[axis][parity] = p.reserve(capacity);
        }
        p.counts_offsets[axis] = p.reserve(16u * kBlocks);

        for (std::uint32_t i = 0; i < num_keys; ++i) {
            // Adversarial skew: almost all keys occupy just two bins
            // across all eight radix passes; catches packed-field
            // carry and massive single-bin contention regressions.
            const auto key = clustered
                ? ((i % 7u == 0u) ? 0xffffffffu : 0u)
                : key_for(i, axis);
            const auto rank = 0xf0000000u + i * 3u + axis;
            p.original[axis].emplace_back(key, rank);
            p.arena[p.keys_offsets[axis][0] / 4u + i] = key;
            p.arena[p.ranks_offsets[axis][0] / 4u + i] = rank;
        }
    }
    for (std::uint32_t parity = 0; parity < 2; ++parity) {
        RadixCountArenaDesc count{};
        RadixScatterArenaDesc scatter{};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            count.axes[axis].input_keys_offset = p.keys_offsets[axis][parity];
            count.axes[axis].radix_counts_offset = p.counts_offsets[axis];
            scatter.axes[axis].input_keys_offset = p.keys_offsets[axis][parity];
            scatter.axes[axis].input_ranks_offset = p.ranks_offsets[axis][parity];
            scatter.axes[axis].output_keys_offset = p.keys_offsets[axis][1u - parity];
            scatter.axes[axis].output_ranks_offset = p.ranks_offsets[axis][1u - parity];
            scatter.axes[axis].radix_counts_offset = p.counts_offsets[axis];
        }
        p.put(p.count_descriptor_offsets[parity], &count, sizeof(count));
        p.put(p.scatter_descriptor_offsets[parity], &scatter, sizeof(scatter));
    }
    return p;
}

void check_dispatch(const BatchCommand& command,
                    std::uint32_t desc, std::uint32_t num_keys,
                    std::uint32_t start_bit, std::uint32_t arena_bytes) {
    const auto& dispatch = std::get<DispatchCommand>(command);
    if (dispatch.groups.x != 32u || dispatch.groups.y != 3u || dispatch.groups.z != 1u)
        throw std::runtime_error("PhysX 32x3 radix dispatch geometry changed");
    std::uint32_t parameters[4]{};
    std::memcpy(parameters, dispatch.push_constants.data(), sizeof(parameters));
    if (parameters[0] != desc || parameters[1] != num_keys ||
        parameters[2] != start_bit || parameters[3] != arena_bytes)
        throw std::runtime_error("PhysX radix parameter ABI changed");
}

// Source-oracle stable LSD radix permutation: one digit or all eight digits.
// Payload order on equal keys is observable by later SAP handle consumers.
void check_sorted(const Packed& p, const std::vector<std::uint32_t>& actual,
                  std::uint32_t num_keys, std::uint32_t passes) {
    const std::uint32_t final_parity = passes & 1u;
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        auto expected = p.original[axis];
        for (std::uint32_t pass = 0; pass < passes; ++pass) {
            const auto shift = pass * 4u;
            std::stable_sort(expected.begin(), expected.end(),
                [shift](const auto& a, const auto& b) {
                    return ((a.first >> shift) & 15u) <
                           ((b.first >> shift) & 15u);
                });
        }
        const auto keys = p.keys_offsets[axis][final_parity] / 4u;
        const auto ranks = p.ranks_offsets[axis][final_parity] / 4u;
        for (std::uint32_t i = 0; i < num_keys; ++i) {
            if (actual[keys + i] != expected[i].first ||
                actual[ranks + i] != expected[i].second)
                throw std::runtime_error(
                    "PhysX radix stable scatter mismatch at axis " +
                    std::to_string(axis) + ", key index " + std::to_string(i) +
                    ", keys " + std::to_string(num_keys) +
                    ", passes " + std::to_string(passes));
        }
    }
}

void check_guarded(const Packed& p, const std::vector<std::uint32_t>& actual,
                   std::uint32_t num_keys, std::uint32_t passes) {
    std::vector<bool> mutable_words(actual.size(), false);
    if (passes) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto histogram = p.counts_offsets[axis] / 4u;
            for (std::uint32_t i = 0; i < 16u * kBlocks; ++i)
                mutable_words[histogram + i] = true;
            for (std::uint32_t parity = 0; parity < 2; ++parity) {
                // A is the initial input, B the first output. Subsequent
                // passes alternate between them without changing the owner.
                if (parity == 0u && passes == 1u) continue;
                for (const auto buffer : {p.keys_offsets[axis][parity],
                                          p.ranks_offsets[axis][parity]})
                    for (std::uint32_t i = 0; i < num_keys; ++i)
                        mutable_words[buffer / 4u + i] = true;
            }
        }
    }
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!mutable_words[i] && actual[i] != p.arena[i])
            throw std::runtime_error(
                "PhysX radix corrupted reserved/descriptor/tail word " +
                std::to_string(i));
}

void run_case(Context& context, RadixCountArenaStage& count_stage,
              RadixScatterArenaStage& scatter_stage,
              std::uint32_t num_keys, std::uint32_t passes,
              bool clustered = false) {
    auto p = pack(num_keys, clustered);
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    const auto arena = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(arena, 0, p.arena.data(), bytes);

    std::vector<BatchCommand> ops;
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        const auto parity = pass & 1u;
        const auto bit = pass * 4u;
        count_stage.append(ops, arena, bytes, p.count_descriptor_offsets[parity],
                           num_keys, bit);
        scatter_stage.append(ops, arena, bytes, p.scatter_descriptor_offsets[parity],
                             num_keys, bit);
        if (num_keys) {
            if (ops.size() != (pass + 1u) * 2u)
                throw std::runtime_error("PhysX radix dispatch pair missing");
            check_dispatch(ops[pass * 2u], p.count_descriptor_offsets[parity],
                           num_keys, bit, bytes);
            check_dispatch(ops[pass * 2u + 1u], p.scatter_descriptor_offsets[parity],
                           num_keys, bit, bytes);
        }
    }
    if (!num_keys && !ops.empty())
        throw std::runtime_error("PhysX zero-key radix work was dispatched");

    ops.emplace_back(CopyCommand{arena, 0, mirror, 0, bytes});
    context.wait_submission(context.submit_batch(ops));

    std::vector<std::uint32_t> actual(p.arena.size());
    context.download(mirror, 0, actual.data(), bytes);
    if (num_keys) {
        check_sorted(p, actual, num_keys, passes);
        check_guarded(p, actual, num_keys, passes);
    } else if (actual != p.arena) {
        throw std::runtime_error("PhysX zero-key radix path modified arena");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(arena);
}

void negative_cases(RadixScatterArenaStage& stage) {
    std::vector<BatchCommand> ops;
    for (const auto offset : {1u, 128u, 256u}) {
        bool rejected = false;
        try {
            stage.append(ops, 1, 160u, offset, 4u, 0u);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Invalid PhysX scatter descriptor accepted");
    }
    for (const auto bit : {1u, 29u, 32u}) {
        bool rejected = false;
        try {
            stage.append(ops, 1, 160u, 0u, 4u, bit);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Invalid PhysX scatter pass accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("Expected count, scatter, wave count and wave scatter shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        ShaderBinary count_shader;
        ShaderBinary scatter_shader;
        ShaderBinary wave_count_shader;
        ShaderBinary wave_shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        count_shader.format = scatter_shader.format =
            wave_count_shader.format = wave_shader.format = ShaderFormat::Dxil;
#else
        count_shader.format = scatter_shader.format =
            wave_count_shader.format = wave_shader.format = ShaderFormat::Spirv;
#endif
        count_shader.bytes = load_shader(argv[1]);
        scatter_shader.bytes = load_shader(argv[2]);
        wave_count_shader.bytes = load_shader(argv[3]);
        wave_shader.bytes = load_shader(argv[4]);
        RadixCountArenaStage count_stage(context, count_shader, wave_count_shader);
        RadixCountArenaStage count_fallback(
            context, std::move(count_shader), std::move(wave_count_shader), true);
        RadixScatterArenaStage scatter_stage(
            context, scatter_shader, wave_shader);
        RadixScatterArenaStage fallback(
            context, std::move(scatter_shader), std::move(wave_shader), true);
        const auto& device = context.device_info();
        const bool capable = device.supports_wave_arithmetic &&
            device.subgroup_size >= 4u && 1024u % device.subgroup_size == 0u;
        if (scatter_stage.using_wave_scan() != capable ||
            count_stage.using_wave_count() != capable ||
            fallback.using_wave_scan() ||
            count_fallback.using_wave_count())
            throw std::runtime_error("Radix wave-shader capability selection changed");
        negative_cases(scatter_stage);
        negative_cases(fallback);

        // Cross-check two complete execution strategies, both with the
        // original two-dispatch count→scatter dependencies and payload
        // stability. The independent CPU reference is unchanged.
        for (const auto path : {0u, 1u}) {
            auto& counters = path ? count_fallback : count_stage;
            auto& scatter = path ? fallback : scatter_stage;
            run_case(context, counters, scatter, 0u, 0u);
            run_case(context, counters, scatter, 1u, 1u);
            run_case(context, counters, scatter, 5u, 8u);
            run_case(context, counters, scatter, 128u, 8u);
            run_case(context, counters, scatter, 4099u, 8u);
            run_case(context, counters, scatter, 8193u, 8u);
            run_case(context, counters, scatter, 131075u, 1u);
            run_case(context, counters, scatter, 4099u, 8u, true);
            run_case(context, counters, scatter, 8193u, 1u, true);
        }
        std::cout << "PhysX packed wave radix count/scatter and fallback source oracle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX radix scatter GPU oracle FAILED: " << e.what() << '\n';
        return 1;
    }
}
