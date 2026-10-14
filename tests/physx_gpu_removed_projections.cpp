// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/mark_removed_pairs_arena.h"
#include "physx_gpu/removed_projections_arena.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::MarkRemovedPairsArenaStage;
using fyzz::physx_gpu::RemovedProjectionsArenaDesc;
using fyzz::physx_gpu::RemovedProjectionsArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kInvalid = 0xffffffffu;

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX SAP shader not found");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{in},
                                  std::istreambuf_iterator<char>{}};
    return {bytes.begin(), bytes.end()};
}

struct Fixture {
    std::vector<std::uint32_t> words;
    RemovedProjectionsArenaDesc desc{};
    std::vector<std::uint32_t> removed;
    std::uint32_t descriptor_offset = 0;

    std::uint32_t reserve(std::size_t count, std::size_t alignment = 128) {
        while ((words.size() * 4) % alignment) words.push_back(kUnused);
        const std::size_t at = words.size();
        if (count > std::numeric_limits<std::uint32_t>::max()/4 - at)
            throw std::out_of_range("Arena offset exceeds 32-bit range");
        words.resize(at + count, kUnused);
        return static_cast<std::uint32_t>(at * 4);
    }
    std::uint32_t& word(std::uint32_t offset) {
        if (offset % 4 || offset / 4 >= words.size())
            throw std::out_of_range("Bad test arena address");
        return words[offset / 4];
    }
    void write_descriptor() {
        std::memcpy(words.data() + descriptor_offset / 4,
                    &desc, sizeof(desc));
    }
};

Fixture pack(std::uint32_t capacity, std::uint32_t removed_count,
             std::uint32_t descriptor_offset) {
    if (!capacity || removed_count > capacity || descriptor_offset % 128)
        throw std::invalid_argument("Invalid fixture dimensions");
    Fixture f;
    f.words.resize(descriptor_offset/4, kUnused);
    f.descriptor_offset = f.reserve(sizeof(RemovedProjectionsArenaDesc)/4, 16);
    if (f.descriptor_offset != descriptor_offset)
        throw std::runtime_error("Fixture descriptor position changed");
    auto& d = f.desc.common;
    d.box_capacity = capacity;
    d.endpoint_capacity = capacity * 2;
    d.num_removed_handles = removed_count;
    d.removed_handles_offset = f.reserve(std::max(1u, removed_count));
    for (std::uint32_t i = 0; i < removed_count; ++i) {
        const std::uint32_t handle = (i * 11u) % capacity;
        f.removed.push_back(handle);
        f.word(d.removed_handles_offset + i*4) = handle;
    }
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        d.sap_box_offsets[axis] = f.reserve(capacity*2);
        d.box_handle_offsets[0][axis] = f.reserve(capacity*2);
        d.box_handle_offsets[1][axis] = f.reserve(capacity*2);
        f.desc.projection_offsets[axis] = f.reserve(capacity*2);
        for (std::uint32_t h = 0; h < capacity; ++h) {
            const std::uint32_t rank = (h*7u + axis*3u) % capacity;
            const std::uint32_t low = 2*rank, high = low+1;
            const auto sap = d.sap_box_offsets[axis] + h*8u;
            f.word(sap) = low;
            f.word(sap+4) = high;
            const auto previous = d.box_handle_offsets[1][axis];
            const auto current = d.box_handle_offsets[0][axis];
            const auto projections = f.desc.projection_offsets[axis];
            f.word(previous + low*4) = (h<<3u) | 1u;
            f.word(previous + high*4) = (h<<3u);
            f.word(current + low*4) = (h<<3u) | 3u;
            f.word(current + high*4) = (h<<3u) | 2u;
            f.word(projections + low*4) = 0x10000000u + 3*h+axis;
            f.word(projections + high*4) = 0x20000000u + 3*h+axis;
        }
    }
    d.old_integer_bounds_offset = f.reserve(capacity*6);
    for (std::uint32_t h = 0; h < capacity; ++h)
        for (std::uint32_t c = 0; c < 6; ++c)
            f.word(d.old_integer_bounds_offset + (h*6+c)*4) =
                0x03000000u + h*6+c;
    f.write_descriptor();
    return f;
}

void upstream_reference(Fixture& f) {
    // PhysX broadphase.cu::markRemovedPairsLaunch first; later
    // ::markRemovedPairsProjectionsLaunch after the incremental SAP phase.
    for (const auto h : f.removed) {
        for (std::uint32_t axis=0; axis<3; ++axis) {
            const auto sap = f.desc.common.sap_box_offsets[axis] + h*8;
            const auto low = f.word(sap), high = f.word(sap+4);
            const auto prev = f.desc.common.box_handle_offsets[1][axis];
            f.word(prev+low*4) |= 4u;
            f.word(prev+high*4) |= 4u;
        }
        for (std::uint32_t c=0; c<6; ++c)
            f.word(f.desc.common.old_integer_bounds_offset+(h*6+c)*4) =
                c < 3 ? 0xff7fffffu : 0x00800000u;
    }
    for (const auto h : f.removed) {
        for (std::uint32_t axis=0; axis<3; ++axis) {
            const auto sap = f.desc.common.sap_box_offsets[axis] + h*8;
            const auto low = f.word(sap), high = f.word(sap+4);
            const auto proj = f.desc.projection_offsets[axis];
            f.word(proj+low*4) = kInvalid;
            f.word(proj+high*4) = kInvalid;
            f.word(sap) = kInvalid;
            f.word(sap+4) = kInvalid;
        }
    }
}

void run_case(Context& ctx, MarkRemovedPairsArenaStage& removed,
              RemovedProjectionsArenaStage& projections,
              std::uint32_t capacity, std::uint32_t count,
              std::uint32_t descriptor_offset) {
    auto input = pack(capacity, count, descriptor_offset);
    auto expected = input;
    upstream_reference(expected);
    const auto bytes = static_cast<std::uint32_t>(input.words.size()*4);
    const auto buffer = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto snapshot = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    ctx.upload(buffer, 0, input.words.data(), bytes);
    std::vector<BatchCommand> operations;
    removed.append(operations, buffer, bytes, descriptor_offset, count);
    projections.append(operations, buffer, bytes, descriptor_offset, count);
    if (operations.size() != (count ? 2u : 0u))
        throw std::runtime_error("PhysX removal dispatch gating diverged");
    if (count) {
        for (const auto& op : operations) {
            const auto& dispatch = std::get<DispatchCommand>(op);
            if (dispatch.groups.x != 32u || dispatch.groups.y != 1u ||
                dispatch.groups.z != 1u)
                throw std::runtime_error("PhysX deleted-pair launch grid diverged");
            std::uint32_t root[3]{};
            std::memcpy(root, dispatch.push_constants.data(), sizeof(root));
            if (root[1] != 32u * 256u)
                throw std::runtime_error("PhysX deleted-pair loop stride diverged");
        }
    }
    operations.emplace_back(CopyCommand{buffer,0,snapshot,0,bytes});
    ctx.wait_submission(ctx.submit_batch(operations));
    std::vector<std::uint32_t> actual(input.words.size());
    ctx.download(snapshot,0,actual.data(),bytes);
    for (std::size_t i=0; i<actual.size(); ++i)
        if (actual[i] != expected.words[i])
            throw std::runtime_error("PhysX deleted-projection arena mismatch at word "+
                                     std::to_string(i)+" (capacity "+
                                     std::to_string(capacity)+")");
    ctx.destroy_buffer(snapshot);
    ctx.destroy_buffer(buffer);
}

void reject_invalid(RemovedProjectionsArenaStage& stage) {
    std::vector<BatchCommand> pending;
    for (const std::uint32_t descriptor_offset : {1u, 16u, 128u}) {
        bool rejected = false;
        try {
            stage.append(pending, 1, 160, descriptor_offset, 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !pending.empty())
            throw std::runtime_error("Invalid PhysX projection descriptor accepted");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("Expected removed-pairs and projections shader binaries");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        ShaderBinary first;
        ShaderBinary second;
#if defined(FYZZXYZZY_TEST_D3D12)
        first.format = second.format = ShaderFormat::Dxil;
#else
        first.format = second.format = ShaderFormat::Spirv;
#endif
        first.bytes = read_shader(argv[1]);
        second.bytes = read_shader(argv[2]);
        MarkRemovedPairsArenaStage removed(ctx, std::move(first));
        RemovedProjectionsArenaStage projections(ctx, std::move(second));
        reject_invalid(projections);
        run_case(ctx, removed, projections, 1,1,0);
        run_case(ctx, removed, projections, 17,0,128);
        run_case(ctx, removed, projections, 17,5,128);
        run_case(ctx, removed, projections, 513,73,256);
        run_case(ctx, removed, projections, 8193,8193,128);
        std::cout << "PhysX removed-projection + deleted-pair source chain PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "PhysX removed-projection GPU chain FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
