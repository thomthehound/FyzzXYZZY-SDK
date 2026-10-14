// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_endpoint_histogram_arena.h"
#include "physx_gpu/sap_incremental_comparisons_arena.h"

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
using fyzz::physx_gpu::SapEndpointHistogramArenaDesc;
using fyzz::physx_gpu::SapEndpointHistogramArenaStage;
using fyzz::physx_gpu::SapIncrementalComparisonArenaDesc;
using fyzz::physx_gpu::SapIncrementalComparisonsArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kBlocks = 32u;
constexpr std::uint32_t kThreads = 256u;

std::vector<std::uint8_t> load_shader(const char* name) {
    std::ifstream in(name, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX incremental shader not found");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}

struct Entry {
    std::uint32_t previous_key;
    std::uint32_t current_key;
    std::uint32_t tagged;
    std::uint32_t prior_index = 0u;
};

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t histogram_descriptor_offset = 0u;
    std::uint32_t comparison_descriptor_offset = 0u;
    SapEndpointHistogramArenaDesc hist{};
    SapIncrementalComparisonArenaDesc comparison{};
    std::uint32_t previous_boxes = 0u;
    std::uint32_t removed_boxes = 0u;

    std::uint32_t reserve(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kUnused);
        const auto index = arena.size();
        if (index + words > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("PhysX incremental arena exceeds offset limit");
        arena.resize(index + words, kUnused);
        return static_cast<std::uint32_t>(index * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t index = 0u) {
        return arena[offset / 4u + index];
    }
    std::uint32_t at(std::uint32_t offset, std::uint32_t index = 0u) const {
        return arena[offset / 4u + index];
    }
};

Packed pack(std::uint32_t previous, std::uint32_t removed) {
    Packed p;
    p.previous_boxes = previous;
    p.removed_boxes = removed;
    p.histogram_descriptor_offset = p.reserve(sizeof(p.hist) / 4u);
    p.comparison_descriptor_offset = p.reserve(sizeof(p.comparison) / 4u);

    const auto projections = previous * 2u;
    const auto capacity = std::max(4u, projections);
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        std::vector<Entry> previous_endpoints;
        previous_endpoints.reserve(projections);
        for (std::uint32_t h = 0u; h < previous; ++h) {
            const auto old_lo = (h * 13u + axis * 11u) % 257u;
            const auto new_lo = (h * 17u + axis * 23u + (h % 7u) * 19u) % 257u;
            const auto old_hi = old_lo + h % 11u + 3u;
            const auto new_hi = new_lo + (h + axis) % 13u + 2u;
            const auto tagged_id = (h * 3u + axis * 137u) << 3u;
            const bool deleted = removed && (h % 17u < removed % 17u);
            const auto flags = (deleted ? 4u : 0u) | ((h % 9u == 0u) ? 2u : 0u);
            previous_endpoints.push_back(Entry{old_lo, new_lo, tagged_id | flags | 1u, 0u});
            previous_endpoints.push_back(Entry{old_hi, new_hi, tagged_id | flags, 0u});
        }
        std::stable_sort(previous_endpoints.begin(), previous_endpoints.end(),
            [](const Entry& a, const Entry& b) { return a.previous_key < b.previous_key; });
        for (std::uint32_t i = 0u; i < projections; ++i)
            previous_endpoints[i].prior_index = i;
        auto current_endpoints = previous_endpoints;
        std::stable_sort(current_endpoints.begin(), current_endpoints.end(),
            [](const Entry& a, const Entry& b) { return a.current_key < b.current_key; });

        auto& axisHist = p.hist.axes[axis];
        auto& axisCmp = p.comparison.axes[axis];
        for (std::uint32_t frame = 0u; frame < 2u; ++frame) {
            auto& d = axisHist.frame[frame];
            d.handles_offset = p.reserve(capacity);
            d.end_histogram_offset = p.reserve(capacity);
            d.start_histogram_offset = p.reserve(capacity);
            d.ordered_end_handles_offset = p.reserve(capacity);
            d.ordered_start_handles_offset = p.reserve(capacity);
            d.block_end_histogram_offset = p.reserve(kBlocks);
            d.block_start_histogram_offset = p.reserve(kBlocks);
            const auto& sorted = frame == 0u ? current_endpoints : previous_endpoints;
            for (std::uint32_t i = 0u; i < projections; ++i)
                p.at(d.handles_offset, i) = sorted[i].tagged;
        }
        axisHist.total_end_histogram_offset = p.reserve(capacity);
        axisHist.block_total_end_histogram_offset = p.reserve(kBlocks);
        axisCmp.sorted_handles_offset = axisHist.frame[0].handles_offset;
        axisCmp.projection_ranks_offset = p.reserve(capacity);
        axisCmp.current_start_histogram_offset = axisHist.frame[0].start_histogram_offset;
        axisCmp.current_end_histogram_offset = axisHist.frame[0].end_histogram_offset;
        axisCmp.previous_start_histogram_offset = axisHist.frame[1].start_histogram_offset;
        axisCmp.previous_end_histogram_offset = axisHist.frame[1].end_histogram_offset;
        axisCmp.comparison_histogram_offset = p.reserve(capacity);
        axisCmp.block_comparison_histogram_offset = p.reserve(kBlocks);
        for (std::uint32_t i = 0u; i < projections; ++i)
            p.at(axisCmp.projection_ranks_offset, i) = current_endpoints[i].prior_index;
    }
    p.comparison.total_comparisons_offset = p.reserve(3u);
    std::memcpy(p.arena.data() + p.histogram_descriptor_offset / 4u,
                &p.hist, sizeof(p.hist));
    std::memcpy(p.arena.data() + p.comparison_descriptor_offset / 4u,
                &p.comparison, sizeof(p.comparison));
    return p;
}

void expected_histogram(Packed& p, std::uint32_t axis,
                        std::uint32_t frame) {
    const auto& d = p.hist.axes[axis].frame[frame];
    const auto& a = p.hist.axes[axis];
    const auto projections = p.previous_boxes * 2u;
    const auto required = (projections + kThreads - 1u) / kThreads;
    const auto iterations = (required + kBlocks - 1u) / kBlocks;
    const auto perBlock = iterations * kThreads;
    std::uint32_t start = 0u, end = 0u, totalEnd = 0u;

    for (std::uint32_t block = 0u; block < kBlocks; ++block) {
        const auto first = block * perBlock;
        const auto last = std::min(projections, first + perBlock);
        const auto priorStart = start, priorEnd = end, priorTotal = totalEnd;
        for (auto i = first; i < last; ++i) {
            const auto tagged = p.at(d.handles_offset, i);
            const bool isStart = (tagged & 1u) != 0u;
            const bool deleted = (tagged & 4u) != 0u;
            p.at(d.start_histogram_offset, i) = start;
            p.at(d.end_histogram_offset, i) = end;
            if (frame == 0u)
                p.at(a.total_end_histogram_offset, i) = totalEnd;
            if (!deleted) {
                const auto out = isStart ? d.ordered_start_handles_offset :
                                            d.ordered_end_handles_offset;
                p.at(out, isStart ? start : end) = tagged >> 3u;
                if (isStart) ++start;
                else ++end;
            }
            if (!isStart) ++totalEnd;
        }
        p.at(d.block_start_histogram_offset, block) = start - priorStart;
        p.at(d.block_end_histogram_offset, block) = end - priorEnd;
        if (frame == 0u)
            p.at(a.block_total_end_histogram_offset, block) = totalEnd - priorTotal;
    }
}

void expected_comparisons(Packed& p) {
    const auto projections = p.previous_boxes * 2u;
    const auto required = (projections + kThreads - 1u) / kThreads;
    const auto iterations = (required + kBlocks - 1u) / kBlocks;
    const auto perBlock = iterations * kThreads;

    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const auto& d = p.comparison.axes[axis];
        std::uint32_t global = 0u;
        for (std::uint32_t block = 0u; block < kBlocks; ++block) {
            const auto begin = block * perBlock;
            const auto finish = std::min(projections, begin + perBlock);
            const auto blockStart = global;
            for (auto i = begin; i < finish; ++i) {
                const auto tagged = p.at(d.sorted_handles_offset, i);
                const auto rank = p.at(d.projection_ranks_offset, i);
                const bool down = i <= rank;
                const bool startHandle = (tagged & 1u) != 0u;
                const auto histogram =
                    down ? (startHandle ? d.previous_end_histogram_offset :
                                           d.previous_start_histogram_offset)
                         : (startHandle ? d.current_end_histogram_offset :
                                           d.current_start_histogram_offset);
                const auto low = down ? i : rank + 1u;
                const auto high = down ? std::min(rank, projections - 1u) : i;
                p.at(d.comparison_histogram_offset, i) = global;
                if ((tagged & 4u) == 0u)
                    global += p.at(histogram, high) - p.at(histogram, low);
            }
            p.at(d.block_comparison_histogram_offset, block) = global - blockStart;
        }
        p.at(p.comparison.total_comparisons_offset, axis) = global;
    }
}

void run_case(Context& context, SapEndpointHistogramArenaStage& hist,
              SapIncrementalComparisonsArenaStage& cmp,
              std::uint32_t previous, std::uint32_t removed) {
    auto p = pack(previous, removed);
    // When no removals exist, the upstream histogram dispatch leaves the
    // previous frame untouched. It must already contain valid prior-frame
    // state from the preceding broadphase update.
    if (!removed)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            expected_histogram(p, axis, 1u);
    auto expected = p;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        expected_histogram(expected, axis, 0u);
        if (removed) expected_histogram(expected, axis, 1u);
    }
    expected_comparisons(expected);
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, p.arena.data(), bytes);

    std::vector<BatchCommand> ops;
    hist.append(ops, gpu, bytes, p.histogram_descriptor_offset,
                previous, 0u, removed, true);
    cmp.append(ops, gpu, bytes, p.comparison_descriptor_offset, previous);
    if (ops.size() != 4u)
        throw std::runtime_error("PhysX histogram-to-incremental-count schedule changed");
    for (std::size_t i = 0u; i < ops.size(); ++i) {
        const auto& cmd = std::get<DispatchCommand>(ops[i]);
        if (cmd.groups.x != kBlocks || cmd.groups.y != (i < 2u ? 3u : 1u) ||
            cmd.groups.z != 1u)
            throw std::runtime_error("PhysX incremental comparison dispatch geometry changed");
    }
    ops.emplace_back(CopyCommand{gpu, 0u, mirror, 0u, bytes});
    context.wait_submission(context.submit_batch(ops));

    std::vector<std::uint32_t> actual(p.arena.size());
    context.download(mirror, 0u, actual.data(), bytes);
    if (actual != expected.arena) {
        const auto mismatch =
            std::mismatch(actual.begin(), actual.end(), expected.arena.begin());
        throw std::runtime_error(
            "PhysX incremental histogram/comparison mismatch at word " +
            std::to_string(std::distance(actual.begin(), mismatch.first)) +
            " (previous " + std::to_string(previous) +
            ", removed " + std::to_string(removed) + ")");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(SapIncrementalComparisonsArenaStage& cmp) {
    for (const auto offset : {1u, 128u, 256u}) {
        std::vector<BatchCommand> commands;
        commands.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        bool rejected = false;
        try { cmp.append(commands, 1u, 112u, offset, 5u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || commands.size() != 1u ||
            !std::holds_alternative<CopyCommand>(commands[0]))
            throw std::runtime_error("Invalid PhysX incremental comparison descriptor accepted");
    }
    std::vector<BatchCommand> commands;
    bool overflow = false;
    try { cmp.append(commands, 1u, 112u, 0u, 0xffffffffu); }
    catch (const std::invalid_argument&) { overflow = true; }
    if (!overflow || !commands.empty())
        throw std::runtime_error("Invalid PhysX incremental projection overflow accepted");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("Expected endpoint-first, endpoint-second, comparison-first, comparison-second shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto shader = [](const char* path) {
            ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
            result.format = ShaderFormat::Dxil;
#else
            result.format = ShaderFormat::Spirv;
#endif
            result.bytes = load_shader(path);
            return result;
        };
        SapEndpointHistogramArenaStage hist(context, shader(argv[1]), shader(argv[2]));
        SapIncrementalComparisonsArenaStage cmp(context, shader(argv[3]), shader(argv[4]));
        negative_cases(cmp);
        run_case(context, hist, cmp, 0u, 0u);
        run_case(context, hist, cmp, 1u, 0u);
        run_case(context, hist, cmp, 5u, 1u);
        run_case(context, hist, cmp, 129u, 7u);
        run_case(context, hist, cmp, 257u, 0u);
        run_case(context, hist, cmp, 4099u, 13u);
        std::cout << "PhysX endpoint histograms → incremental comparison counts PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX incremental comparison oracle FAILED: " << e.what() << '\n';
        return 1;
    }
}
