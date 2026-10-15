// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "physx_gpu/sap_overlap_prefix_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using fyzz::physx_gpu::SapOverlapPrefixArenaDesc;
using fyzz::physx_gpu::SapOverlapPrefixArenaStage;
namespace {
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
std::vector<std::uint8_t> load_shader(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Overlap prefix shader missing");
    std::vector<char> bytes{std::istreambuf_iterator<char>{file},
                            std::istreambuf_iterator<char>{}};
    return {bytes.begin(), bytes.end()};
}
struct Fixture {
    SapOverlapPrefixArenaDesc desc{};
    std::vector<std::uint32_t> words;
    std::uint32_t count = 0;
    std::uint32_t reserve(std::uint32_t wordCount) {
        while (words.size() % 4u) words.push_back(kCanary);
        const auto start = words.size();
        words.resize(start + wordCount, kCanary);
        return static_cast<std::uint32_t>(start * 4u);
    }
    void store32(std::uint32_t offset, std::uint32_t value) {
        words.at(offset / 4u) = value;
    }
    void store64(std::uint32_t offset, std::uint64_t value) {
        store32(offset, static_cast<std::uint32_t>(value));
        store32(offset + 4u, static_cast<std::uint32_t>(value >> 32u));
    }
    std::uint64_t load64(std::uint32_t offset) const {
        return std::uint64_t(words.at(offset / 4u)) |
               (std::uint64_t(words.at(offset / 4u + 1u)) << 32u);
    }
};
Fixture build(std::uint32_t count, bool large) {
    Fixture f;
    f.count = count;
    const auto desc = f.reserve(sizeof(f.desc) / 4u);
    if (desc != 0u) throw std::runtime_error("Unexpected descriptor alignment");
    f.desc.overlap_counts_offset = f.reserve(2u * (count + 9u));
    f.desc.block_counts_offset = f.reserve(2u * 32u);
    f.desc.total_offset = f.reserve(2u);
    f.desc.region_work_total_offset = f.reserve(1u);
    f.reserve(9u);
    f.store32(f.desc.region_work_total_offset, count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint64_t value = large
            ? (std::uint64_t(1u << 30u) + (i % 19u))
            : ((i * 13u + i / 7u) % 11u);
        f.store64(f.desc.overlap_counts_offset + i * 8u, value);
    }
    std::memcpy(f.words.data(), &f.desc, sizeof(f.desc));
    return f;
}
// Independent reference accumulates linear 64-bit prefixes, reconstructs
// PhysX's exact 32 contiguous block partitions and retains per-block totals.
void reference(Fixture& f) {
    const auto perBlock = (((f.count + 255u) / 256u) + 31u) / 32u * 256u;
    std::uint64_t sum = 0;
    for (std::uint32_t block = 0; block < 32u; ++block) {
        const auto start = sum;
        const auto begin = std::uint64_t(block) * perBlock;
        const auto end = std::min<std::uint64_t>(f.count, begin + perBlock);
        for (auto i = begin; i < end; ++i) {
            auto offset = f.desc.overlap_counts_offset +
                          static_cast<std::uint32_t>(i) * 8u;
            const auto v = f.load64(offset);
            f.store64(offset, sum);
            sum += v;
        }
        f.store64(f.desc.block_counts_offset + block * 8u, sum - start);
    }
    f.store64(f.desc.total_offset, sum);
}
void run(Context& context, SapOverlapPrefixArenaStage& stage,
         std::uint32_t n, bool wide, bool enabled) {
    auto fixture = build(n, wide);
    auto expected = fixture;
    if (enabled) reference(expected);
    const auto bytes = static_cast<std::uint32_t>(fixture.words.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto copy = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, fixture.words.data(), bytes);
    std::vector<BatchCommand> commands;
    stage.append(commands, gpu, bytes, 0u, enabled ? 1u : 0u);
    if (commands.size() != (enabled ? 2u : 0u))
        throw std::runtime_error("PhysX overlap prefix host gate violated");
    for (const auto& command : commands) {
        const auto& dispatch = std::get<DispatchCommand>(command);
        if (dispatch.groups.x != 32u || dispatch.groups.y != 1u ||
            dispatch.groups.z != 1u || dispatch.push_constants.size() != 32u)
            throw std::runtime_error("PhysX overlap prefix launch changed");
    }
    commands.emplace_back(CopyCommand{gpu, 0u, copy, 0u, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(fixture.words.size());
    context.download(copy, 0u, actual.data(), bytes);
    if (actual != expected.words) {
        const auto it = std::mismatch(actual.begin(), actual.end(), expected.words.begin());
        throw std::runtime_error(
            "PhysX 64-bit overlap prefix mismatch at word " +
            std::to_string(std::distance(actual.begin(), it.first)) +
            " (count " + std::to_string(n) + ", wide " + std::to_string(wide) + ")");
    }
    context.destroy_buffer(copy);
    context.destroy_buffer(gpu);
}
void negative(SapOverlapPrefixArenaStage& stage) {
    std::vector<BatchCommand> commands;
    stage.append(commands, 0u, 0u, 17u, 0u);
    if (!commands.empty()) throw std::runtime_error("Unexpected empty dispatch");
    for (auto offset : {1u, 40u, 64u}) {
        bool rejected = false;
        try { stage.append(commands, 1u, 64u, offset, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !commands.empty())
            throw std::runtime_error("Invalid prefix descriptor accepted");
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Expected two 64-bit wave shader files");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& info = context.device_info();
        if (!info.supports_int64_wave || !info.supports_wave_arithmetic ||
            info.subgroup_size < 4u || info.subgroup_size > 128u ||
            256u % info.subgroup_size != 0u) {
            std::cout << "SKIP: no enabled 64-bit subgroup arithmetic on this adapter\n";
            return 77;
        }
        ShaderBinary first, second;
#if defined(FYZZXYZZY_TEST_D3D12)
        first.format = second.format = ShaderFormat::Dxil;
#else
        first.format = second.format = ShaderFormat::Spirv;
#endif
        first.bytes = load_shader(argv[1]);
        second.bytes = load_shader(argv[2]);
        SapOverlapPrefixArenaStage stage(context, std::move(first), std::move(second));
        negative(stage);
        for (auto n : {0u, 1u, 5u, 31u, 32u, 255u, 256u, 257u,
                       8191u, 8192u, 8193u, 16385u}) {
            run(context, stage, n, false, true);
            run(context, stage, n, true, true);
        }
        run(context, stage, 129u, true, false);
        std::cout << "PhysX native 64-bit overlap prefix PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX native 64-bit overlap prefix FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
