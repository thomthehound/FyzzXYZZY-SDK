// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_endpoint_histogram_arena.h"

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

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kThreads = 256;

std::vector<std::uint8_t> load_shader(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX endpoint histogram shader not found");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t descriptor_offset = 0;
    SapEndpointHistogramArenaDesc desc{};
    std::uint32_t boxes[2]{};

    std::uint32_t reserve(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kUnused);
        const auto first = arena.size();
        if (first + words > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("Endpoint histogram fixture too large");
        arena.resize(first + words, kUnused);
        return static_cast<std::uint32_t>(first * 4u);
    }
    std::uint32_t& at(std::uint32_t byte_offset, std::uint32_t index = 0u) {
        return arena[byte_offset / 4u + index];
    }
    const std::uint32_t& at(std::uint32_t byte_offset,
                            std::uint32_t index = 0u) const {
        return arena[byte_offset / 4u + index];
    }
};

Packed pack(std::uint32_t previous, std::uint32_t created,
            std::uint32_t removed, bool incremental) {
    Packed p;
    p.boxes[0] = incremental ? previous : previous + created - removed;
    p.boxes[1] = previous;
    p.descriptor_offset = p.reserve(sizeof(p.desc) / 4u);

    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        auto& desc = p.desc.axes[axis];
        for (std::uint32_t frame = 0u; frame < 2u; ++frame) {
            const auto count = p.boxes[frame] * 2u;
            const auto bufferSize = std::max(4u, count);
            auto& d = desc.frame[frame];
            d.handles_offset = p.reserve(bufferSize);
            d.end_histogram_offset = p.reserve(bufferSize);
            d.start_histogram_offset = p.reserve(bufferSize);
            d.ordered_end_handles_offset = p.reserve(bufferSize);
            d.ordered_start_handles_offset = p.reserve(bufferSize);
            d.block_end_histogram_offset = p.reserve(kBlocks);
            d.block_start_histogram_offset = p.reserve(kBlocks);

            std::vector<std::pair<std::uint32_t, std::uint32_t>> endpoints;
            endpoints.reserve(count);
            for (std::uint32_t handle = 0u; handle < p.boxes[frame]; ++handle) {
                // Distinct frame tags, start/end encoding, and deletes.
                // Sorted projections deliberately have duplicate keys.
                const bool deleted = removed &&
                    (((handle + axis + frame) % 13u) == 0u);
                const bool newlyCreated = ((handle + axis) % 7u) == 0u;
                const auto flags = (deleted ? 4u : 0u) | (newlyCreated ? 2u : 0u);
                const auto id = handle * 3u + axis * 137u;
                const auto lo = (handle * 17u + axis * 11u) % 101u;
                const auto hi = lo + 3u + handle % 7u;
                endpoints.emplace_back(lo, (id << 3u) | flags | 1u);
                endpoints.emplace_back(hi, (id << 3u) | flags);
            }
            std::stable_sort(endpoints.begin(), endpoints.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::uint32_t i = 0; i < count; ++i)
                p.at(d.handles_offset, i) = endpoints[i].second;
        }
        const auto curSize = std::max(4u, p.boxes[0] * 2u);
        desc.total_end_histogram_offset = p.reserve(curSize);
        desc.block_total_end_histogram_offset = p.reserve(kBlocks);
    }

    std::memcpy(p.arena.data() + p.descriptor_offset / 4u,
                &p.desc, sizeof(p.desc));
    return p;
}

void oracle_frame(const Packed& p, std::vector<std::uint32_t>& expected,
                  std::uint32_t axis, std::uint32_t frame) {
    const auto& d = p.desc.axes[axis].frame[frame];
    const auto& axisDesc = p.desc.axes[axis];
    const std::uint32_t projections = p.boxes[frame] * 2u;
    const std::uint32_t required = (projections + 255u) / 256u;
    const std::uint32_t iterations = (required + 31u) / 32u;
    const std::uint32_t perBlock = iterations * 256u;

    std::uint32_t globalStart = 0u;
    std::uint32_t globalEnd = 0u;
    std::uint32_t globalAllEnd = 0u;
    for (std::uint32_t block = 0u; block < kBlocks; ++block) {
        const auto begin = block * perBlock;
        const auto limit = std::min(projections, begin + perBlock);
        const auto firstStart = globalStart;
        const auto firstEnd = globalEnd;
        const auto firstAllEnd = globalAllEnd;

        for (auto i = begin; i < limit; ++i) {
            const auto tagged = p.at(d.handles_offset, i);
            const bool isStart = (tagged & 1u) != 0u;
            const bool deleted = (tagged & 4u) != 0u;
            expected[d.start_histogram_offset / 4u + i] = globalStart;
            expected[d.end_histogram_offset / 4u + i] = globalEnd;
            if (frame == 0u)
                expected[axisDesc.total_end_histogram_offset / 4u + i] = globalAllEnd;

            if (!deleted) {
                const auto target = isStart ? d.ordered_start_handles_offset :
                                                d.ordered_end_handles_offset;
                const auto out = isStart ? globalStart : globalEnd;
                expected[target / 4u + out] = tagged >> 3u;
                if (isStart) ++globalStart;
                else ++globalEnd;
            }
            if (!isStart) ++globalAllEnd;
        }

        expected[d.block_start_histogram_offset / 4u + block] = globalStart - firstStart;
        expected[d.block_end_histogram_offset / 4u + block] = globalEnd - firstEnd;
        if (frame == 0u)
            expected[axisDesc.block_total_end_histogram_offset / 4u + block] =
                globalAllEnd - firstAllEnd;
    }
}

void run_case(Context& context, SapEndpointHistogramArenaStage& stage,
              std::uint32_t previous, std::uint32_t created,
              std::uint32_t removed, bool incremental) {
    auto p = pack(previous, created, removed, incremental);
    auto expected = p.arena;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        oracle_frame(p, expected, axis, 0u);
        if (incremental && removed) oracle_frame(p, expected, axis, 1u);
    }
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0, p.arena.data(), bytes);
    std::vector<BatchCommand> commands;
    stage.append(commands, gpu, bytes, p.descriptor_offset,
                 previous, created, removed, incremental);
    if (commands.size() != 2u)
        throw std::runtime_error("PhysX endpoint histogram source schedule changed");
    for (const auto& op : commands) {
        const auto& dispatch = std::get<DispatchCommand>(op);
        if (dispatch.groups.x != kBlocks || dispatch.groups.y != 3u ||
            dispatch.groups.z != 1u)
            throw std::runtime_error("PhysX endpoint histogram launch geometry changed");
        const std::uint32_t args[5] = {p.descriptor_offset, previous,
                                       created, removed, incremental ? 1u : 0u};
        if (dispatch.push_constants.size() != 32u ||
            std::memcmp(dispatch.push_constants.data(), args, sizeof(args)) != 0)
            throw std::runtime_error("PhysX endpoint histogram parameters changed");
    }
    commands.emplace_back(CopyCommand{gpu, 0u, mirror, 0u, bytes});
    context.wait_submission(context.submit_batch(commands));
    std::vector<std::uint32_t> actual(p.arena.size());
    context.download(mirror, 0u, actual.data(), bytes);
    if (actual != expected) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        throw std::runtime_error(
            "PhysX endpoint histogram mismatch at word " +
            std::to_string(std::distance(actual.begin(), mismatch.first)) +
            " (prev " + std::to_string(previous) +
            ", created " + std::to_string(created) +
            ", removed " + std::to_string(removed) +
            ", incremental " + std::to_string(incremental) + ")");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(SapEndpointHistogramArenaStage& stage) {
    const auto prefix = []() {
        std::vector<BatchCommand> batch;
        batch.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        return batch;
    };
    for (const auto offset : {1u, 128u, 256u}) {
        auto batch = prefix();
        bool rejected = false;
        try { stage.append(batch, 1u, 240u, offset, 1u, 0u, 0u, true); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || batch.size() != 1u)
            throw std::runtime_error("Invalid PhysX histogram descriptor accepted");
    }
    for (const auto sizes : {
             std::pair<std::uint32_t,std::uint32_t>{0u, 1u},
             std::pair<std::uint32_t,std::uint32_t>{1u, 2u},
             std::pair<std::uint32_t,std::uint32_t>{
                 std::numeric_limits<std::uint32_t>::max(), 0u}}) {
        auto batch = prefix();
        bool rejected = false;
        try { stage.append(batch, 1u, 240u, 0u,
                           sizes.first, 0u, sizes.second, false); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || batch.size() != 1u)
            throw std::runtime_error("Invalid PhysX histogram box count accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("Expected first and second PhysX histogram shaders");
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
        SapEndpointHistogramArenaStage stage(context, shader(argv[1]), shader(argv[2]));
        negative_cases(stage);
        run_case(context, stage, 0u, 0u, 0u, true);
        run_case(context, stage, 1u, 0u, 0u, true);
        run_case(context, stage, 5u, 0u, 1u, true);
        run_case(context, stage, 129u, 0u, 12u, true);
        run_case(context, stage, 257u, 0u, 7u, true);
        run_case(context, stage, 4099u, 0u, 0u, true);
        run_case(context, stage, 0u, 5u, 0u, false);
        run_case(context, stage, 7u, 3u, 2u, false);
        run_case(context, stage, 129u, 71u, 5u, false);
        std::cout << "PhysX SAP two-pass endpoint histogram oracle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX SAP endpoint histogram oracle FAILED: " << e.what() << '\n';
        return 1;
    }
}
