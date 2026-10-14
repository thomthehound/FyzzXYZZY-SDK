// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/mark_removed_pairs_arena.h"

#include <algorithm>
#include <array>
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

// Dispatch-root comparison keeps the identical packed descriptor and GPU
// arithmetic, replacing only five per-dispatch pointer loads with offsets
// already known to the host. Three previous-frame handle offsets still
// come from the GPU descriptor. This is NOT a direct-buffer baseline.
void apply_resolved_subset(DispatchCommand& dispatch,
                           const MarkRemovedPairsArenaDesc& desc) {
    const std::array<std::uint32_t, 5> offsets{{
        desc.removed_handles_offset, desc.old_integer_bounds_offset,
        desc.sap_box_offsets[0], desc.sap_box_offsets[1], desc.sap_box_offsets[2]
    }};
    if (dispatch.push_constants.size() != 32)
        throw std::runtime_error("Unexpected PhysX arena root parameter size");
    std::memcpy(dispatch.push_constants.data() + 3 * sizeof(std::uint32_t),
                offsets.data(), sizeof(offsets));
}

void run_case(Context& ctx, MarkRemovedPairsArenaStage& stage,
              std::uint32_t capacity, std::uint32_t removed,
              std::uint32_t descriptor_offset, bool resolved = false) {
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
        if (resolved)
            apply_resolved_subset(std::get<DispatchCommand>(commands[0]), packed.desc);
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

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("No measurements");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] :
           (values[middle - 1] + values[middle]) * 0.5;
}

void bench(Context& ctx, MarkRemovedPairsArenaStage& descriptor_stage,
           MarkRemovedPairsArenaStage& resolved_stage) {
    constexpr int iterations = 128;
    constexpr int warmups = 2;
    constexpr int repeats = 7;
    const auto packed = pack(8193, 2048, 256);
    const auto bytes = static_cast<std::uint32_t>(packed.arena.size() * 4);
    const auto arena = ctx.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    ctx.upload(arena, 0, packed.arena.data(), bytes);

    auto commands_for = [&](MarkRemovedPairsArenaStage& stage, bool resolved) {
        std::vector<BatchCommand> ops;
        ops.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            stage.append(ops, arena, bytes, packed.descriptor_offset,
                         packed.desc.num_removed_handles);
            if (resolved)
                apply_resolved_subset(std::get<DispatchCommand>(ops.back()), packed.desc);
        }
        return ops;
    };
    const auto indirect = commands_for(descriptor_stage, false);
    const auto resolved = commands_for(resolved_stage, true);
    std::array<std::vector<double>, 2> gpu_ns, host_us;
    bool missing_gpu_timestamps = false;

    // Alternate order to mitigate clock, cache, and thermal drift.
    for (int sample = -warmups; sample < repeats; ++sample) {
        for (int offset = 0; offset < 2; ++offset) {
            const int variant = (sample + offset + warmups) % 2;
            const auto& ops = variant ? resolved : indirect;
            const auto begin = std::chrono::steady_clock::now();
            const auto gpu = ctx.profile_batch_gpu_nanoseconds(ops);
            if (!gpu) {
                // Profiling unavailable on this device: still execute work,
                // but do NOT claim the host-only number is a GPU measurement.
                missing_gpu_timestamps = true;
                ctx.execute_batch(ops);
            }
            const auto end = std::chrono::steady_clock::now();
            if (sample >= 0) {
                if (gpu) gpu_ns[variant].push_back(*gpu);
                host_us[variant].push_back(
                    std::chrono::duration<double, std::micro>(end - begin).count());
            }
        }
    }
    std::cout << "Device: " << ctx.device_info().name
              << "; " << iterations << " PhysX SAP removal dispatches/batch, "
              << repeats << " paired samples after " << warmups
              << " warmups; same input arena and physics writes.\n";
    for (int variant = 0; variant < 2; ++variant) {
        std::cout << (variant ? "5 host-resolved offsets" : "GPU descriptor offsets")
                  << ": host submit+wait median " << median(host_us[variant])
                  << " us/batch";
        if (!gpu_ns[variant].empty())
            std::cout << "; GPU command-batch median "
                      << median(gpu_ns[variant]) / 1000.0 << " us/batch"
                      << " (" << median(gpu_ns[variant]) / iterations
                      << " ns/dispatch, including in-batch barriers)";
        std::cout << '\n';
    }
    if (missing_gpu_timestamps)
        std::cout << "GPU timestamp queries unavailable; NO GPU performance comparison.\n";
    else if (median(gpu_ns[0]) > 0)
        std::cout << "GPU timing ratio (resolved / descriptor) = "
                  << median(gpu_ns[1]) / median(gpu_ns[0])
                  << "; lower is better. NOT a direct-bind/segmented/BDA comparison.\n";
    ctx.destroy_buffer(arena);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4)
            throw std::invalid_argument("Expected descriptor and resolved shader paths [--benchmark]");
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
        ShaderBinary resolved_shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        resolved_shader.format = ShaderFormat::Dxil;
#else
        resolved_shader.format = ShaderFormat::Spirv;
#endif
        resolved_shader.bytes = shader_bytes(argv[2]);
        MarkRemovedPairsArenaStage resolved_stage(ctx, std::move(resolved_shader));
        run_case(ctx, stage, 1, 1, 0);
        run_case(ctx, stage, 17, 0, 128);
        run_case(ctx, stage, 17, 5, 128);
        run_case(ctx, stage, 513, 73, 256);
        run_case(ctx, stage, 4097, 513, 128);
        run_case(ctx, resolved_stage, 1, 1, 0, true);
        run_case(ctx, resolved_stage, 17, 0, 128, true);
        run_case(ctx, resolved_stage, 17, 5, 128, true);
        run_case(ctx, resolved_stage, 513, 73, 256, true);
        run_case(ctx, resolved_stage, 4097, 513, 128, true);
        if (argc == 4) {
            if (std::string(argv[3]) != "--benchmark")
                throw std::invalid_argument("Unexpected arena test option");
            bench(ctx, stage, resolved_stage);
        }
        std::cout << "PhysX incremental SAP removed-pair arena PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX removed-pair arena FAILED: " << e.what() << '\n';
        return 1;
    }
}
