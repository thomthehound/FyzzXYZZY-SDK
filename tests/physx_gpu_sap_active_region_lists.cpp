// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_active_region_lists_arena.h"

#include <algorithm>
#include <array>
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

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::SapActiveRegionListsArenaDesc;
using fyzz::physx_gpu::SapActiveRegionListsArenaStage;
constexpr std::uint32_t kCanary = 0xcdcdcdcd;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX active-region shader missing");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}
struct Endpoint {
    std::uint32_t key, tagged, handle;
};
struct Packed {
    std::vector<std::uint32_t> arena;
    SapActiveRegionListsArenaDesc desc{};
    std::uint32_t descriptor_offset = 0;
    std::uint32_t boxes = 0;
    std::uint32_t created = 0;

    std::uint32_t reserve(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kCanary);
        const auto start = arena.size();
        arena.resize(start + std::max(1u, words), kCanary);
        return static_cast<std::uint32_t>(start * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t index = 0u) {
        return arena[offset / 4u + index];
    }
    std::uint32_t at(std::uint32_t offset, std::uint32_t index = 0u) const {
        return arena[offset / 4u + index];
    }
};

Packed build(std::uint32_t boxes, bool anyCreated) {
    Packed p;
    p.boxes = boxes;
    p.created = anyCreated ? (boxes + 4u) / 5u : 0u;
    p.descriptor_offset = p.reserve(sizeof(p.desc) / 4u);
    const auto n = boxes * 2u;
    const auto dense = 64u * n;
    auto& d = p.desc;
    d.sorted_x_handles_offset = p.reserve(n);
    d.x_sap_positions_offset = p.reserve(n);
    d.region_ranges_offset = p.reserve(boxes * 8u);
    d.start_region_prefix_offset = p.reserve(n);
    d.start_region_total_offset = p.reserve(1u);
    d.active_histograms_offset = p.reserve(dense);
    d.start_histograms_offset = p.reserve(dense);
    d.ordered_active_handles_offset = p.reserve(dense);
    d.ordered_start_handles_offset = p.reserve(dense);
    for (std::uint32_t i = 0; i < dense; ++i) {
        p.at(d.active_histograms_offset, i) = 0u;
        p.at(d.start_histograms_offset, i) = 0u;
    }

    std::vector<Endpoint> ordered;
    ordered.reserve(n);
    for (std::uint32_t h = 0; h < boxes; ++h) {
        const auto lo = ((h * 17u + h / 19u) % 131u) * 8u;
        const auto hi = lo + ((h * 13u) % 23u + 1u) * 8u + 1u;
        const auto flags = anyCreated && h % 5u == 0u ? 2u : 0u;
        ordered.push_back({lo, (h << 3u) | flags | 1u, h});
        ordered.push_back({hi, (h << 3u) | flags, h});
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const Endpoint& x,
                                                        const Endpoint& y) {
        return x.key < y.key;
    });
    std::vector<std::array<std::uint32_t, 2>> sap(boxes);
    std::uint32_t totalEnds = 0u;
    const auto elementsPerRegion = (boxes + 3u) / 4u;
    // The source's all-endpoint prefix bins are used for the X range,
    // while deterministic Y/Z ranks exercise multi-region membership.
    std::vector<std::uint32_t> endPrefix(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto tag = ordered[i].tagged;
        const auto h = ordered[i].handle;
        endPrefix[i] = totalEnds;
        p.at(d.sorted_x_handles_offset, i) = tag;
        sap[h][(tag & 1u) != 0u ? 0u : 1u] = i;
        if ((tag & 1u) == 0u) ++totalEnds;
    }
    for (std::uint32_t h = 0; h < boxes; ++h) {
        const auto x0 = endPrefix[sap[h][0]] / elementsPerRegion;
        const auto x1 = endPrefix[sap[h][1]] / elementsPerRegion;
        const auto y0 = (h * 7u + h / 17u) % 4u;
        const auto z0 = (h * 11u + h / 13u) % 4u;
        const auto y1 = std::min(3u, y0 + (h % 3u));
        const auto z1 = std::min(3u, z0 + (h % 2u));
        const std::uint32_t first[] = {x0, y0, z0, 0u};
        const std::uint32_t last[] = {x1, y1, z1, 0u};
        for (std::uint32_t c = 0; c < 4u; ++c) {
            p.at(d.region_ranges_offset, h * 8u + c) = first[c];
            p.at(d.region_ranges_offset, h * 8u + c + 4u) = last[c];
        }
        p.at(d.x_sap_positions_offset, h * 2u) = sap[h][0];
        p.at(d.x_sap_positions_offset, h * 2u + 1u) = sap[h][1];
    }

    std::vector<std::uint32_t> contributions(n);
    for (std::uint32_t h = 0; h < boxes; ++h) {
        const auto range = d.region_ranges_offset / 4u + h * 8u;
        contributions[sap[h][0]] =
            (p.arena[range + 5u] - p.arena[range + 1u] + 1u) *
            (p.arena[range + 6u] - p.arena[range + 2u] + 1u);
    }
    std::uint32_t prefix = 0u;
    for (std::uint32_t i = 0u; i < n; ++i) {
        p.at(d.start_region_prefix_offset, i) = prefix;
        prefix += contributions[i];
    }
    p.at(d.start_region_total_offset) = prefix;
    std::memcpy(p.arena.data() + p.descriptor_offset / 4u, &p.desc, sizeof(p.desc));
    return p;
}

// CPU oracle deliberately iterates X-sorted logical *starts*, never
// the GPU's prefix binary-search work indices. It derives a sparse
// mask for each of the 64 regions, scans those masks linearly, then
// compacts the lists in source X-order.
void reference(Packed& p) {
    if (!p.created) return;
    const auto& d = p.desc;
    const auto n = p.boxes * 2u;
    for (std::uint32_t pos = 0u; pos < n; ++pos) {
        const auto tagged = p.at(d.sorted_x_handles_offset, pos);
        if (!(tagged & 1u)) continue;
        const auto h = tagged >> 3u;
        const auto first = d.region_ranges_offset / 4u + h * 8u;
        const auto x = p.arena[first], y0 = p.arena[first + 1u],
                   z0 = p.arena[first + 2u];
        const auto y1 = p.arena[first + 5u], z1 = p.arena[first + 6u];
        for (auto y = y0; y <= y1; ++y)
            for (auto z = z0; z <= z1; ++z) {
                const auto region = x + y * 4u + z * 16u;
                p.at(d.start_histograms_offset, region * n + pos) = 1u;
                p.at(d.active_histograms_offset, region * n + pos) =
                    (tagged >> 1u) & 1u;
            }
    }

    for (std::uint32_t region = 0u; region < 64u; ++region) {
        std::uint32_t startPrefix = 0u, activePrefix = 0u;
        for (std::uint32_t pos = 0u; pos < n; ++pos) {
            const auto index = region * n + pos;
            const auto start = p.at(d.start_histograms_offset, index);
            const auto active = p.at(d.active_histograms_offset, index);
            p.at(d.start_histograms_offset, index) = startPrefix;
            p.at(d.active_histograms_offset, index) = activePrefix;
            if (start) {
                const auto h = p.at(d.sorted_x_handles_offset, pos) >> 3u;
                p.at(d.ordered_start_handles_offset, region * n + startPrefix) = h;
                ++startPrefix;
            }
            if (active) {
                const auto h = p.at(d.sorted_x_handles_offset, pos) >> 3u;
                p.at(d.ordered_active_handles_offset, region * n + activePrefix) = h;
                ++activePrefix;
            }
        }
    }
}

void run_case(Context& context, SapActiveRegionListsArenaStage& stage,
              std::uint32_t boxes, bool hasCreated) {
    auto p = build(boxes, hasCreated);
    auto expected = p;
    reference(expected);
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, p.arena.data(), bytes);
    std::vector<BatchCommand> commands;
    stage.append(commands, gpu, bytes, p.descriptor_offset, boxes, p.created);
    if (commands.size() != (p.created ? 3u : 0u))
        throw std::runtime_error("PhysX active-region created gate changed");
    if (p.created) {
        const std::uint32_t workgroups[] = {32u, 64u, 64u};
        for (std::uint32_t i = 0; i < 3u; ++i) {
            const auto& dispatch = std::get<DispatchCommand>(commands[i]);
            if (dispatch.groups.x != workgroups[i] ||
                dispatch.groups.y != 1u || dispatch.groups.z != 1u ||
                dispatch.push_constants.size() != 32u)
                throw std::runtime_error("PhysX active-region launch geometry changed");
        }
    }
    commands.emplace_back(CopyCommand{gpu, 0u, mirror, 0u, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> observed(p.arena.size());
    context.download(mirror, 0u, observed.data(), bytes);
    if (observed != expected.arena) {
        const auto mismatch =
            std::mismatch(observed.begin(), observed.end(), expected.arena.begin());
        throw std::runtime_error(
            "PhysX active region mismatch at word " +
            std::to_string(std::distance(observed.begin(), mismatch.first)) +
            " (boxes " + std::to_string(boxes) +
            ", created " + std::to_string(p.created) + ")");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}
void negative_cases(SapActiveRegionListsArenaStage& stage) {
    for (const auto invalid : {1u, 128u, 160u}) {
        std::vector<BatchCommand> commands;
        commands.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        bool rejected = false;
        try { stage.append(commands, 1u, 160u, invalid, 1u, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || commands.size() != 1u ||
            !std::holds_alternative<CopyCommand>(commands.front()))
            throw std::runtime_error("Invalid PhysX active-region descriptor accepted");
    }
    std::vector<BatchCommand> exact;
    stage.append(exact, 1u, 160u, 96u, 1u, 1u);
    if (exact.size() != 3u)
        throw std::runtime_error("Valid exact-fit active descriptor rejected");
    for (const auto boxes : {0u, 8388608u, 0xffffffffu}) {
        std::vector<BatchCommand> commands;
        bool rejected = false;
        try { stage.append(commands, 1u, 160u, 96u, boxes, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !commands.empty())
            throw std::runtime_error("Invalid active-region dimensions accepted");
    }
    std::vector<BatchCommand> skipped;
    stage.append(skipped, 0u, 0u, 1u, 0u, 0u);
    if (!skipped.empty())
        throw std::runtime_error("No-created PhysX active-region gate changed");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("Expected three active-region shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto format =
#if defined(FYZZXYZZY_TEST_D3D12)
            ShaderFormat::Dxil;
#else
            ShaderFormat::Spirv;
#endif
        const auto shader = [&](const char* file) {
            ShaderBinary result;
            result.format = format;
            result.bytes = load_shader(file);
            return result;
        };
        SapActiveRegionListsArenaStage stage(
            context, shader(argv[1]), shader(argv[2]), shader(argv[3]));
        negative_cases(stage);
        run_case(context, stage, 0u, false);
        run_case(context, stage, 5u, false);
        run_case(context, stage, 1u, true);
        run_case(context, stage, 5u, true);
        run_case(context, stage, 129u, true);
        run_case(context, stage, 257u, true);
        run_case(context, stage, 1027u, true);
        std::cout << "PhysX SAP three-stage active-region lists PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX active-region stages FAILED: " << error.what() << '\n';
        return 1;
    }
}
