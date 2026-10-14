// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/sap_region_producer_arena.h"
#include "physx_gpu/sap_region_histograms_arena.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::SapRegionProducerArenaDesc;
using fyzz::physx_gpu::SapRegionProducerArenaStage;
using fyzz::physx_gpu::SapRegionHistogramsArenaDesc;
using fyzz::physx_gpu::SapRegionHistogramsArenaStage;

constexpr std::uint32_t kCanary = 0xcdcdcdcd;

std::vector<std::uint8_t> load_shader(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX BP_CREATE_REGIONS shader not found");
    const std::vector<char> data{std::istreambuf_iterator<char>{in},
                                 std::istreambuf_iterator<char>{}};
    return {data.begin(), data.end()};
}

struct Endpoint {
    std::uint32_t key, handle, tag;
};

struct Packed {
    std::vector<std::uint32_t> memory;
    SapRegionProducerArenaDesc desc{};
    SapRegionHistogramsArenaDesc hist{};
    std::uint32_t descriptor_offset = 0;
    std::uint32_t histogram_descriptor_offset = 0;

    std::uint32_t reserve(std::uint32_t words) {
        while (memory.size() % 32u) memory.push_back(kCanary);
        const auto base = memory.size();
        if (base + words > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("PhysX region fixture too large");
        memory.resize(base + words, kCanary);
        return static_cast<std::uint32_t>(base * 4u);
    }
    std::uint32_t& at(std::uint32_t offset, std::uint32_t i = 0u) {
        return memory[offset / 4u + i];
    }
    std::uint32_t at(std::uint32_t offset, std::uint32_t i = 0u) const {
        return memory[offset / 4u + i];
    }
};

Packed build(std::uint32_t boxes, std::uint32_t created) {
    Packed p;
    p.descriptor_offset = p.reserve(sizeof(p.desc) / 4u);
    p.histogram_descriptor_offset = p.reserve(sizeof(p.hist) / 4u);
    auto& d = p.desc;
    d.current_boxes = boxes;
    d.created_handles = created;
    d.box_capacity = boxes + 7u;
    d.endpoint_capacity = boxes * 2u + 11u;
    d.current_x_handles_offset = p.reserve(d.endpoint_capacity);
    for (std::uint32_t a = 0; a < 3u; ++a) {
        d.sap_positions_offset[a] = p.reserve(d.box_capacity * 2u);
        d.total_end_histogram_offset[a] = p.reserve(d.endpoint_capacity);
    }
    d.region_range_offset = p.reserve(d.box_capacity * 8u);
    d.start_region_accum_offset = p.reserve(d.endpoint_capacity);
    d.region_accum_offset = p.reserve(d.endpoint_capacity);
    auto& scan = p.hist;
    scan.start_region_accum_offset = d.start_region_accum_offset;
    scan.region_accum_offset = d.region_accum_offset;
    scan.block_start_region_accum_offset = p.reserve(32u);
    scan.block_region_accum_offset = p.reserve(32u);
    scan.start_region_total_offset = p.reserve(1u);
    scan.region_total_offset = p.reserve(1u);

    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        std::vector<Endpoint> endpoints;
        endpoints.reserve(boxes * 2u);
        for (std::uint32_t handle = 0; handle < boxes; ++handle) {
            // Distinct endpoint parity: start even, end odd, so tied
            // starts and ends still form valid nonzero-length intervals.
            // Different axis strides produce independent SAP orders.
            const auto low = ((handle * (13u + axis * 8u) +
                               (handle / 17u) * 7u + axis * 11u) % 101u) * 8u;
            const auto high = low + ((handle + axis * 5u) % 19u + 1u) * 8u + 1u;
            const auto newFlag = ((handle % 9u == 0u) ? 2u : 0u);
            endpoints.push_back({low, handle, (handle << 3u) | newFlag | 1u});
            endpoints.push_back({high, handle, (handle << 3u) | newFlag});
        }
        std::stable_sort(endpoints.begin(), endpoints.end(),
            [](const Endpoint& a, const Endpoint& b) {
                return a.key < b.key;
            });
        std::uint32_t endCount = 0u;
        for (std::uint32_t index = 0u; index < endpoints.size(); ++index) {
            const auto& entry = endpoints[index];
            const auto slot = ((entry.tag & 1u) != 0u) ? 0u : 1u;
            p.at(d.sap_positions_offset[axis], entry.handle * 2u + slot) = index;
            p.at(d.total_end_histogram_offset[axis], index) = endCount;
            if (slot == 1u) ++endCount;
            if (axis == 0u)
                p.at(d.current_x_handles_offset, index) = entry.tag;
        }
        if (endCount != boxes) throw std::runtime_error("Invalid fixture endpoint pairs");
    }
    std::memcpy(p.memory.data() + p.descriptor_offset / 4u,
                &p.desc, sizeof(p.desc));
    std::memcpy(p.memory.data() + p.histogram_descriptor_offset / 4u,
                &p.hist, sizeof(p.hist));
    return p;
}

// Independent CPU source oracle: for each logical box, derive its
// three region ranges from its own SAP positions and all-end prefix;
// then place the sparse contributions at the X endpoint indices.
// Unlike the shader, this does not iterate sorted-X projections.
void reference(Packed& p) {
    const auto& d = p.desc;
    if (!d.created_handles) return;
    const auto perRegion = (d.current_boxes + 3u) / 4u;
    for (std::uint32_t handle = 0u; handle < d.current_boxes; ++handle) {
        std::array<std::uint32_t, 3> first{}, last{};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const auto sap = d.sap_positions_offset[axis];
            const auto histogram = d.total_end_histogram_offset[axis];
            const auto minIndex = p.at(sap, 2u * handle);
            const auto maxIndex = p.at(sap, 2u * handle + 1u);
            first[axis] = p.at(histogram, minIndex) / perRegion;
            last[axis] = p.at(histogram, maxIndex) / perRegion;
            p.at(d.region_range_offset, handle * 8u + axis) = first[axis];
            p.at(d.region_range_offset, handle * 8u + 4u + axis) = last[axis];
        }
        p.at(d.region_range_offset, handle * 8u + 3u) = 0u;
        p.at(d.region_range_offset, handle * 8u + 7u) = 0u;
        const auto yz = (last[1] - first[1] + 1u) *
                        (last[2] - first[2] + 1u);
        const auto minX = p.at(d.sap_positions_offset[0], handle * 2u);
        const auto maxX = p.at(d.sap_positions_offset[0], handle * 2u + 1u);
        p.at(d.start_region_accum_offset, minX) = yz;
        p.at(d.start_region_accum_offset, maxX) = 0u;
        p.at(d.region_accum_offset, minX) =
            last[0] >= first[0] ? (last[0] - first[0] + 1u) * yz : 0u;
        p.at(d.region_accum_offset, maxX) = 0u;
    }
}


void reference_histograms(Packed& p) {
    if (!p.desc.created_handles) return;
    const auto n = 2u * p.desc.current_boxes;
    const auto perBlock = ((n + 255u) / 256u + 31u) / 32u * 256u;
    const auto run = [&](std::uint32_t data, std::uint32_t blocks,
                         std::uint32_t total) {
        std::uint32_t accumulated = 0u;
        for (std::uint32_t block = 0u; block < 32u; ++block) {
            const auto before = accumulated;
            const auto begin = block * perBlock;
            const auto end = std::min(n, begin + perBlock);
            for (std::uint32_t i = begin; i < end; ++i) {
                const auto count = p.at(data, i);
                p.at(data, i) = accumulated;
                accumulated += count;
            }
            p.at(blocks, block) = accumulated - before;
        }
        p.at(total) = accumulated;
    };
    run(p.hist.start_region_accum_offset,
        p.hist.block_start_region_accum_offset,
        p.hist.start_region_total_offset);
    run(p.hist.region_accum_offset,
        p.hist.block_region_accum_offset,
        p.hist.region_total_offset);
}
void run_case(Context& context, SapRegionProducerArenaStage& stage,
              SapRegionHistogramsArenaStage* scans,
              std::uint32_t boxes, std::uint32_t created) {
    auto p = build(boxes, created);
    auto expected = p;
    reference(expected);
    if (scans) reference_histograms(expected);
    const auto bytes = static_cast<std::uint32_t>(p.memory.size() * 4u);
    const auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0u, p.memory.data(), bytes);
    std::vector<BatchCommand> ops;
    stage.append(ops, gpu, bytes, p.descriptor_offset, boxes, created);
    if (scans) scans->append(ops, gpu, bytes, p.histogram_descriptor_offset, boxes, created);
    if (ops.size() != (created ? (scans ? 5u : 1u) : 0u))
        throw std::runtime_error("PhysX region host launch gate changed");
    if (created) {
        const auto& dispatch = std::get<DispatchCommand>(ops.front());
        if (dispatch.groups.x != 32u || dispatch.groups.y != 1u ||
            dispatch.groups.z != 1u)
            throw std::runtime_error("PhysX BP_CREATE_REGIONS geometry changed");
    }
    if (scans && created) {
        for (std::size_t i = 1u; i < ops.size(); ++i) {
            const auto& dispatch = std::get<DispatchCommand>(ops[i]);
            if (dispatch.groups.x != 32u || dispatch.groups.y != 1u ||
                dispatch.groups.z != 1u ||
                dispatch.push_constants.size() != 32u)
                throw std::runtime_error("PhysX region-prefix dispatch geometry changed");
        }
    }
    ops.emplace_back(CopyCommand{gpu, 0u, mirror, 0u, bytes});
    context.wait_submission(context.submit_batch(ops));
    std::vector<std::uint32_t> observed(p.memory.size());
    context.download(mirror, 0u, observed.data(), bytes);
    if (observed != expected.memory) {
        const auto mismatch =
            std::mismatch(observed.begin(), observed.end(), expected.memory.begin());
        throw std::runtime_error(
            "PhysX BP_CREATE_REGIONS mismatch at word " +
            std::to_string(std::distance(observed.begin(), mismatch.first)) +
            " (boxes " + std::to_string(boxes) +
            ", created " + std::to_string(created) + ")");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(SapRegionProducerArenaStage& stage,
                    SapRegionHistogramsArenaStage* scans) {
    for (const auto invalid : {1u, 128u, 256u}) {
        std::vector<BatchCommand> commands;
        commands.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        bool rejected = false;
        try { stage.append(commands, 1u, 160u, invalid, 5u, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || commands.size() != 1u ||
            !std::holds_alternative<CopyCommand>(commands.front()))
            throw std::runtime_error("Invalid region descriptor changed caller work");
    }
    for (const auto badBoxes : {0u, 0xffffffffu}) {
        std::vector<BatchCommand> commands;
        bool rejected = false;
        try { stage.append(commands, 1u, 160u, 0u, badBoxes, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !commands.empty())
            throw std::runtime_error("Invalid region count accepted");
    }
    // Original source host does nothing when there are no created handles.
    std::vector<BatchCommand> nothing;
    stage.append(nothing, 0u, 0u, 3u, 0u, 0u);
    if (!nothing.empty())
        throw std::runtime_error("Unexpected no-created region dispatch");
    if (scans) {
        scans->append(nothing, 0u, 0u, 7u, 0u, 0u);
        if (!nothing.empty())
            throw std::runtime_error("Unexpected no-created region scan");
        // The 32-byte descriptor at offset 128 fits exactly in a
        // 160-byte arena; it must be accepted. Earlier testing wrongly
        // treated this valid boundary as an invalid descriptor.
        std::vector<BatchCommand> boundary;
        boundary.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
        scans->append(boundary, 1u, 160u, 128u, 5u, 1u);
        if (boundary.size() != 5u ||
            !std::holds_alternative<CopyCommand>(boundary.front()))
            throw std::runtime_error("Valid exact-fit region descriptor not accepted");
        for (const auto invalid : {1u, 129u, 160u}) {
            std::vector<BatchCommand> commands;
            commands.emplace_back(CopyCommand{1u, 0u, 2u, 0u, 4u});
            bool rejected = false;
            try { scans->append(commands, 1u, 160u, invalid, 5u, 1u); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected || commands.size() != 1u ||
                !std::holds_alternative<CopyCommand>(commands.front()))
                throw std::runtime_error("Bad region scan changed caller batch");
        }
        std::vector<BatchCommand> ops;
        bool rejected = false;
        try { scans->append(ops, 1u, 160u, 0u, 0xffffffffu, 1u); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Region scan projection overflow accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 6)
            throw std::invalid_argument("Expected producer shader and optional four prefix shaders");
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
        const auto format = shader.format;
        SapRegionProducerArenaStage stage(context, std::move(shader));
        std::unique_ptr<SapRegionHistogramsArenaStage> scans;
        if (argc == 6) {
            const auto from = [&](const char* path) {
                ShaderBinary binary;
                binary.format = format;
                binary.bytes = load_shader(path);
                return binary;
            };
            scans = std::make_unique<SapRegionHistogramsArenaStage>(
                context, from(argv[2]), from(argv[3]),
                from(argv[4]), from(argv[5]));
        }
        negative_cases(stage, scans.get());
        run_case(context, stage, scans.get(), 0u, 0u);
        run_case(context, stage, scans.get(), 5u, 0u);
        run_case(context, stage, scans.get(), 1u, 1u);
        run_case(context, stage, scans.get(), 5u, 1u);
        run_case(context, stage, scans.get(), 17u, 3u);
        run_case(context, stage, scans.get(), 129u, 7u);
        run_case(context, stage, scans.get(), 257u, 257u);
        run_case(context, stage, scans.get(), 4099u, 11u);
        run_case(context, stage, scans.get(), 8201u, 19u);
        std::cout << (scans ? "PhysX region producer + four region scans PASS\\n" : "PhysX region producer PASS\\n");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "PhysX BP_CREATE_REGIONS FAILED: " << ex.what() << '\n';
        return 1;
    }
}
