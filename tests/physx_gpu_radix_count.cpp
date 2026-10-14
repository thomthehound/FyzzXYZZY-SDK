// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/radix_count_arena.h"

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

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kThreads = 1024;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX radix-count shader not found");
    const std::vector<char> values{std::istreambuf_iterator<char>{in},
                                   std::istreambuf_iterator<char>{}};
    return {values.begin(), values.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    RadixCountArenaDesc descriptor{};
    std::uint32_t descriptor_offset = 0;
    std::vector<std::uint32_t> keys[3];

    std::uint32_t reserve(std::size_t words) {
        while (arena.size() % 32) arena.push_back(kUnused);
        const auto at = arena.size();
        arena.resize(at + words, kUnused);
        return static_cast<std::uint32_t>(at * 4u);
    }
};

Packed pack(std::uint32_t num_keys, std::uint32_t descriptor_offset) {
    Packed p;
    p.arena.resize(descriptor_offset / 4u, kUnused);
    p.descriptor_offset = p.reserve(sizeof(RadixCountArenaDesc) / 4u);
    if (p.descriptor_offset != descriptor_offset)
        throw std::runtime_error("Radix descriptor offset mismatch");
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        auto& d = p.descriptor.axes[axis];
        d.input_keys_offset = p.reserve((num_keys + 3u) & ~3u);
        d.radix_counts_offset = p.reserve(16u * kBlocks);
        for (std::uint32_t i = 0; i < num_keys; ++i) {
            // Duplicates, alternating bins, all-one sentinels, and high bits.
            const std::uint32_t key = i % 17u == 0 ? 0xffffffffu :
                (i * 2654435761u) ^ (axis * 0x9e3779b9u) ^ (i / 3u);
            p.keys[axis].push_back(key);
            p.arena[d.input_keys_offset / 4u + i] = key;
        }
    }
    std::memcpy(p.arena.data() + descriptor_offset / 4u,
                &p.descriptor, sizeof(p.descriptor));
    return p;
}

void source_oracle(Packed& expected, std::uint32_t num_keys, std::uint32_t bit) {
    const std::uint32_t num_vectors = (num_keys + 3u) / 4u;
    const std::uint32_t required = (num_vectors + kThreads - 1u) / kThreads;
    const std::uint32_t iterations = (required + kBlocks - 1u) / kBlocks;
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        const auto counts = expected.descriptor.axes[axis].radix_counts_offset / 4u;
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
            std::uint32_t bins[16]{};
            const auto first = std::min(iterations * block * kThreads, num_vectors);
            const auto last = std::min(first + iterations * kThreads, num_vectors);
            for (std::uint32_t v = first; v < last; ++v) {
                for (std::uint32_t lane = 0; lane < 4; ++lane) {
                    const std::uint32_t index = v * 4u + lane;
                    const std::uint32_t key = index < num_keys ?
                        expected.keys[axis][index] : 0xffffffffu;
                    ++bins[(key >> bit) & 15u];
                }
            }
            for (std::uint32_t digit = 0; digit < 16u; ++digit)
                expected.arena[counts + digit * kBlocks + block] = bins[digit];
        }
    }
}

void run_case(Context& context, RadixCountArenaStage& stage,
              std::uint32_t num_keys, std::uint32_t bit, std::uint32_t offset) {
    auto initial = pack(num_keys, offset);
    auto expected = initial;
    if (num_keys) source_oracle(expected, num_keys, bit);
    const auto bytes = static_cast<std::uint32_t>(initial.arena.size() * 4u);
    const auto arena = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(arena, 0, initial.arena.data(), bytes);
    std::vector<BatchCommand> ops;
    stage.append(ops, arena, bytes, offset, num_keys, bit);
    if (ops.size() != (num_keys ? 1u : 0u))
        throw std::runtime_error("Radix-count empty-stage conditional changed");
    if (num_keys) {
        const auto& dispatch = std::get<DispatchCommand>(ops.front());
        if (dispatch.groups.x != 32u || dispatch.groups.y != 3u ||
            dispatch.groups.z != 1u)
            throw std::runtime_error("PhysX radix-count grid changed");
        std::uint32_t parameters[4]{};
        std::memcpy(parameters, dispatch.push_constants.data(), sizeof(parameters));
        if (parameters[0] != offset || parameters[1] != num_keys ||
            parameters[2] != bit || parameters[3] != bytes)
            throw std::runtime_error("PhysX radix-count root parameters changed");
    }
    ops.emplace_back(CopyCommand{arena, 0, mirror, 0, bytes});
    context.wait_submission(context.submit_batch(ops));
    std::vector<std::uint32_t> actual(initial.arena.size());
    context.download(mirror, 0, actual.data(), bytes);
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected.arena[i])
            throw std::runtime_error("PhysX radix-count mismatch at word " +
                 std::to_string(i) + ", keys " + std::to_string(num_keys) +
                 ", pass bit " + std::to_string(bit));
    context.destroy_buffer(mirror);
    context.destroy_buffer(arena);
}

void negative_cases(RadixCountArenaStage& stage) {
    std::vector<BatchCommand> ops;
    for (const auto offset : {1u, 128u, 256u}) {
        bool rejected = false;
        try {
            stage.append(ops, 1, 160u, offset, 4u, 0u);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Invalid PhysX radix descriptor accepted");
    }
    for (const auto bit : {1u, 29u, 32u}) {
        bool rejected = false;
        try {
            stage.append(ops, 1, 160u, 0u, 4u, bit);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Invalid radix pass accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected compiled radix shader");
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
        RadixCountArenaStage stage(context, std::move(shader));
        negative_cases(stage);
        run_case(context, stage, 0u, 0u, 0u);
        run_case(context, stage, 4u, 0u, 128u);
        run_case(context, stage, 5u, 4u, 128u);
        run_case(context, stage, 4099u, 12u, 256u);
        run_case(context, stage, 8193u, 28u, 128u);
        run_case(context, stage, 131075u, 20u, 256u);
        std::cout << "PhysX three-axis radix block count oracle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX radix-count GPU oracle FAILED: " << e.what() << '\n';
        return 1;
    }
}
