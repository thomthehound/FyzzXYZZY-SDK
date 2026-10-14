// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/mark_removed_pairs_arena.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fyzz::compute;
using fyzz::physx_gpu::MarkRemovedPairsArenaDesc;
using fyzz::physx_gpu::MarkRemovedPairsArenaStage;

namespace {
constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kDeleteBit = 4u;

std::vector<std::uint8_t> shader_bytes(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("PhysX removed-pair shader is missing");
    std::vector<char> v(std::istreambuf_iterator<char>{stream},
                        std::istreambuf_iterator<char>{});
    return {v.begin(), v.end()};
}

struct PackedCase {
    std::vector<std::uint32_t> arena;
    MarkRemovedPairsArenaDesc desc{};
    std::uint32_t descriptor_offset = 0;
    std::vector<std::uint32_t> removed;

    std::uint32_t reserve(std::size_t count, std::size_t align_bytes = 128) {
        while ((arena.size() * sizeof(std::uint32_t)) % align_bytes)
            arena.push_back(kUnused);
        const auto start = arena.size();
        if (start + count > std::numeric_limits<std::uint32_t>::max() / 4u)
            throw std::out_of_range("PhysX arena test exceeds 32-bit offsets");
        arena.resize(start + count, kUnused);
        return static_cast<std::uint32_t>(start * 4);
    }

    std::uint32_t& word(std::uint32_t byte_offset) {
        if (byte_offset % 4 || byte_offset / 4 >= arena.size())
            throw std::out_of_range("Test packer invalid byte offset");
        return arena[byte_offset / 4];
    }
};

PackedCase pack(std::uint32_t capacity, std::uint32_t removed_count,
                std::uint32_t descriptor_offset) {
    if (!capacity || removed_count > capacity ||
        descriptor_offset % 128)
        throw std::invalid_argument("Invalid broadphase test dimensions");

    PackedCase result;
    result.arena.resize(descriptor_offset / 4, kUnused);
    result.descriptor_offset = result.reserve(sizeof(MarkRemovedPairsArenaDesc) / 4, 16);
    if (result.descriptor_offset != descriptor_offset)
        throw std::runtime_error("Arena descriptor placement unexpectedly changed");

    result.desc.box_capacity = capacity;
    result.desc.endpoint_capacity = 2 * capacity;
    result.desc.num_removed_handles = removed_count;
    result.desc.removed_handles_offset = result.reserve(std::max(1u, removed_count));
    for (std::uint32_t i = 0; i < removed_count; ++i) {
        // Unique handles in a deliberately non-contiguous removal order.
        const auto handle = (i * 11u) % capacity;
        result.removed.push_back(handle);
        result.word(result.desc.removed_handles_offset + i * 4) = handle;
    }

    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        result.desc.sap_box_offsets[axis] = result.reserve(capacity * 2);
        result.desc.box_handle_offsets[0][axis] = result.reserve(capacity * 2);
        result.desc.box_handle_offsets[1][axis] = result.reserve(capacity * 2);

        for (std::uint32_t h = 0; h < capacity; ++h) {
            // Permute the endpoints to force pointer chasing: all test
            // capacities are coprime to 7, so no endpoint aliases.
            const std::uint32_t rank = (h * 7u + axis * 3u) % capacity;
            const std::uint32_t lo = 2 * rank, hi = lo + 1;
            const auto sap = result.desc.sap_box_offsets[axis] + h * 8;
            result.word(sap) = lo;
            result.word(sap + 4) = hi;
            const auto curr = result.desc.box_handle_offsets[0][axis];
            const auto prev = result.desc.box_handle_offsets[1][axis];
            result.word(curr + lo * 4) = (h << 3) | 1u | 2u;
            result.word(curr + hi * 4) = (h << 3) | 2u;
            result.word(prev + lo * 4) = (h << 3) | 1u;
            result.word(prev + hi * 4) = (h << 3);
        }
    }

    result.desc.old_integer_bounds_offset = result.reserve(capacity * 6);
    for (std::uint32_t h = 0; h < capacity; ++h)
        for (std::uint32_t c = 0; c < 6; ++c)
            result.word(result.desc.old_integer_bounds_offset + (h * 6 + c) * 4) =
                0x01000000u + h * 16u + c;

    std::memcpy(result.arena.data() + descriptor_offset / 4,
                &result.desc, sizeof(result.desc));
    return result;
}

void reference_mark_removed(PackedCase& data) {
    // Source oracle: PxgBroadPhaseDesc::markDeleted, clearHandles,
    // markRemovedPairsLaunch, and PxgIntegerAABB::setEmpty.
    for (const std::uint32_t handle : data.removed) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const std::uint32_t sap = data.desc.sap_box_offsets[axis] + handle * 8;
            const auto lo = data.word(sap);
            const auto hi = data.word(sap + 4);
            const std::uint32_t prev = data.desc.box_handle_offsets[1][axis];
            data.word(prev + lo * 4) |= kDeleteBit;
            data.word(prev + hi * 4) |= kDeleteBit;
        }
        for (std::uint32_t c = 0; c < 6; ++c)
            data.word(data.desc.old_integer_bounds_offset + (handle * 6 + c) * 4) =
                c < 3 ? 0xff7fffffu : 0x00800000u;
    }
}

void run_case(Context& ctx, MarkRemovedPairsArenaStage& stage,
              std::uint32_t capacity, std::uint32_t removed,
              std::uint32_t descriptor_offset) {
    PackedCase packed = pack(capacity, removed, descriptor_offset);
    PackedCase expected = packed;
    reference_mark_removed(expected);
    const std::uint32_t bytes = static_cast<std::uint32_t>(packed.arena.size() * 4);
    const auto arena = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const auto snapshot = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    ctx.upload(arena, 0, packed.arena.data(), bytes);

    std::vector<BatchCommand> commands;
    stage.append(commands, arena, bytes, descriptor_offset, removed);
    if (!removed) {
        if (!commands.empty()) throw std::runtime_error("Empty removal unnecessarily dispatched");
        commands.emplace_back(CopyCommand{arena, 0, snapshot, 0, bytes});
    } else {
        if (commands.size() != 1 || !std::holds_alternative<DispatchCommand>(commands[0]))
            throw std::runtime_error("PhysX arena stage did not append its GPU dispatch");
        // Copy the GPU-written entire arena in the same submission; checks
        // dispatch -> transfer ordering on both APIs, including UAV barriers.
        commands.emplace_back(CopyCommand{arena, 0, snapshot, 0, bytes});
    }
    ctx.wait_submission(ctx.submit_batch(commands));

    std::vector<std::uint32_t> observed(packed.arena.size(), 0);
    ctx.download(snapshot, 0, observed.data(), bytes);
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != expected.arena[i])
            throw std::runtime_error("PhysX arena/source reference mismatch at word " +
                                     std::to_string(i) + " of " +
                                     std::to_string(observed.size()));
    }
    ctx.destroy_buffer(snapshot);
    ctx.destroy_buffer(arena);
}

void bench(Context& ctx, MarkRemovedPairsArenaStage& stage) {
    const auto packed = pack(8193, 2048, 256);
    const auto bytes = static_cast<std::uint32_t>(packed.arena.size() * 4);
    const auto arena = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    ctx.upload(arena, 0, packed.arena.data(), bytes);
    std::vector<BatchCommand> commands;
    for (int i = 0; i < 256; ++i)
        stage.append(commands, arena, bytes, packed.descriptor_offset,
                     packed.desc.num_removed_handles);
    const auto start = std::chrono::steady_clock::now();
    ctx.wait_submission(ctx.submit_batch(commands));
    const auto end = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Arena removed-pair 256 stages, end-to-end submit+wait (NOT GPU timestamps): "
              << us << " us; " << us / 256 << " us/stage, device "
              << ctx.device_info().name << '\n';
    ctx.destroy_buffer(arena);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3)
            throw std::invalid_argument("Expected shader path [--benchmark]");
        Context ctx(
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
        shader.bytes = shader_bytes(argv[1]);
        MarkRemovedPairsArenaStage stage(ctx, std::move(shader));
        run_case(ctx, stage, 1, 1, 0);
        run_case(ctx, stage, 17, 0, 128);
        run_case(ctx, stage, 17, 5, 128);
        run_case(ctx, stage, 513, 73, 256);
        run_case(ctx, stage, 4097, 513, 128);
        if (argc == 3) {
            if (std::string(argv[2]) != "--benchmark")
                throw std::invalid_argument("Unexpected arena test option");
            bench(ctx, stage);
        }
        std::cout << "PhysX incremental SAP removed-pair arena PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX removed-pair arena FAILED: " << e.what() << '\n';
        return 1;
    }
}
