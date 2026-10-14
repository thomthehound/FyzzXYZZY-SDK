// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/aabb_encoding.h"

#include <algorithm>
#include <array>
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

namespace {

using fyzz::physx_gpu::AabbEncodingParameters;
using fyzz::physx_gpu::AabbEncodingStage;
using namespace fyzz::compute;

struct FloatBounds {
    float value[6]; // PxBounds3: minimum xyz, maximum xyz.
};

struct IntegerBounds {
    std::uint32_t value[6]; // PxgIntegerAABB::mMinMax.
};

static_assert(sizeof(FloatBounds) == 24, "PhysX bounds layout differs");
static_assert(sizeof(IntegerBounds) == 24, "PhysX integer bounds layout differs");

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX AABB shader was not found");
    std::vector<char> bytes(std::istreambuf_iterator<char>{in},
                            std::istreambuf_iterator<char>{});
    return {bytes.begin(), bytes.end()};
}

// Reference translation of PhysX broadphase.cu::translateAABBsLaunch,
// including its floating-point bit ordering and environment bounds behavior.
IntegerBounds upstream_reference(const FloatBounds& input, float distance,
                                 std::uint32_t env_id, bool has_env_ids,
                                 const AabbEncodingParameters& parameters) {
    const std::uint32_t shifts[3] = {
        parameters.shift_x, parameters.shift_y, parameters.shift_z
    };
    const std::uint32_t env_shifts[3] = {
        parameters.env_bits_x, parameters.env_bits_y, parameters.env_bits_z
    };
    IntegerBounds result{};
    for (std::uint32_t component = 0; component < 6; ++component) {
        const std::uint32_t axis = component % 3;
        const std::uint32_t mask = component >= 3 ? 1u : 0u;
        const float projection = component < 3
            ? input.value[component] - distance : input.value[component] + distance;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &projection, sizeof(bits));
        const std::uint32_t ordered = (bits & 0x80000000u)
            ? ~bits : (bits | 0x80000000u);
        std::uint32_t encoded = ((ordered >> shifts[axis]) & ~1u) | mask;
        if (has_env_ids && env_shifts[axis]) {
            if (env_id != std::numeric_limits<std::uint32_t>::max()) {
                encoded |= env_id << (32u - env_shifts[axis]);
            } else {
                encoded = mask ? 0xfe7fffffu : 0x01800000u;
            }
        }
        result.value[component] = encoded;
    }
    return result;
}

void run_case(Context& context, AabbEncodingStage& stage,
              const AabbEncodingParameters& p, bool with_env) {
    constexpr std::uint32_t padding = 19;
    constexpr std::uint32_t sentinel = 0xcdcdcdcd;
    const std::size_t capacity = p.count + padding;

    std::vector<FloatBounds> bounds(capacity);
    std::vector<float> distances(capacity);
    std::vector<std::uint32_t> env_ids(capacity);
    std::vector<IntegerBounds> result(capacity);

    for (std::size_t i = 0; i < capacity; ++i) {
        const float x = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.5f;
        const float y = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.25f;
        const float z = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.5f;
        bounds[i] = {{x, y, z, x + 0.5f, y + 1.0f, z + 1.5f}};
        distances[i] = static_cast<float>(i % 7) * 0.125f;
        env_ids[i] = i % 11 == 0 ? std::numeric_limits<std::uint32_t>::max()
                                  : static_cast<std::uint32_t>(i % 8);
        for (auto& n : result[i].value) n = sentinel;
    }

    const BufferId gpu_bounds = context.create_buffer(sizeof(FloatBounds) * capacity,
                                                      MemoryPlacement::DeviceLocal);
    const BufferId gpu_distances = context.create_buffer(sizeof(float) * capacity,
                                                         MemoryPlacement::DeviceLocal);
    const BufferId gpu_env = context.create_buffer(sizeof(std::uint32_t) * capacity,
                                                   MemoryPlacement::DeviceLocal);
    const BufferId gpu_result = context.create_buffer(sizeof(IntegerBounds) * capacity,
                                                      MemoryPlacement::DeviceLocal);

    context.upload(gpu_bounds, 0, bounds.data(), sizeof(FloatBounds) * capacity);
    context.upload(gpu_distances, 0, distances.data(), sizeof(float) * capacity);
    context.upload(gpu_env, 0, env_ids.data(), sizeof(std::uint32_t) * capacity);
    context.upload(gpu_result, 0, result.data(), sizeof(IntegerBounds) * capacity);

    std::vector<BatchCommand> commands;
    stage.append(commands, gpu_bounds, gpu_distances, with_env ? gpu_env : 0,
                 gpu_result, p);
    if (commands.size() != 1 || !std::holds_alternative<DispatchCommand>(commands[0])) {
        throw std::runtime_error("PhysX AABB translation did not enqueue its stage");
    }
    context.wait_submission(context.submit_batch(commands));
    context.download(gpu_result, 0, result.data(), sizeof(IntegerBounds) * capacity);

    for (std::size_t i = 0; i < capacity; ++i) {
        for (std::uint32_t component = 0; component < 6; ++component) {
            const std::uint32_t expected = i < p.count
                ? upstream_reference(bounds[i], distances[i], env_ids[i],
                                     with_env, p).value[component] : sentinel;
            if (result[i].value[component] != expected) {
                throw std::runtime_error(
                    "PhysX GPU AABB encoding mismatch at box " + std::to_string(i) +
                    " component " + std::to_string(component));
            }
        }
    }

    context.destroy_buffer(gpu_result);
    context.destroy_buffer(gpu_env);
    context.destroy_buffer(gpu_distances);
    context.destroy_buffer(gpu_bounds);
}

// Source-derived stage must participate in an ordered GPU batch, not
// implicitly synchronize or maintain a hidden secondary simulation state.
// Exercise shader write -> GPU copy -> shader write; both published results
// must match the original PhysX float-bit encoding for their respective
// environment-ID modes.
void run_ordered_batch(Context& context, AabbEncodingStage& stage) {
    const AabbEncodingParameters p{73, 4, 4, 4, 2, 3, 4};
    std::vector<FloatBounds> bounds(p.count);
    std::vector<float> distances(p.count);
    std::vector<std::uint32_t> env_ids(p.count);
    for (std::size_t i = 0; i < p.count; ++i) {
        const float x = static_cast<float>(i) * 0.125f - 5.0f;
        bounds[i] = {{x, -x, x * 0.5f, x + 1.0f, -x + 0.75f, x * 0.5f + 0.25f}};
        distances[i] = static_cast<float>(i % 5) * 0.03125f;
        env_ids[i] = i % 17 == 0 ? 0xffffffffu : static_cast<std::uint32_t>(i % 6);
    }
    const auto bound_bytes = bounds.size() * sizeof(FloatBounds);
    const auto scalar_bytes = distances.size() * sizeof(float);
    const BufferId box = context.create_buffer(bound_bytes, MemoryPlacement::DeviceLocal);
    const BufferId distance = context.create_buffer(scalar_bytes, MemoryPlacement::DeviceLocal);
    const BufferId env = context.create_buffer(scalar_bytes, MemoryPlacement::DeviceLocal);
    const BufferId output = context.create_buffer(bound_bytes, MemoryPlacement::DeviceLocal);
    const BufferId first_snapshot = context.create_buffer(bound_bytes, MemoryPlacement::DeviceLocal);
    context.upload(box, 0, bounds.data(), bound_bytes);
    context.upload(distance, 0, distances.data(), scalar_bytes);
    context.upload(env, 0, env_ids.data(), scalar_bytes);

    std::vector<BatchCommand> commands;
    stage.append(commands, box, distance, 0, output, p);
    commands.emplace_back(CopyCommand{output, 0, first_snapshot, 0, bound_bytes});
    stage.append(commands, box, distance, env, output, p);
    if (commands.size() != 3) throw std::runtime_error("PhysX batch did not record three operations");
    // The native D3D12 bounded AABB pipeline uses t0..t2 SRVs.
    // Explicitly claiming writes to those sources must be rejected before
    // recording GPU work, rather than silently disregarding metadata.
#if defined(FYZZXYZZY_TEST_D3D12)
    auto invalid_access = std::get<DispatchCommand>(commands.front());
    invalid_access.buffers[0].access = BufferAccess::Write;
    bool rejected_access = false;
    try {
        context.submit_batch({invalid_access});
    } catch (const std::invalid_argument&) {
        rejected_access = true;
    }
    if (!rejected_access)
        throw std::runtime_error("Writable PhysX AABB input SRV access was accepted");
#endif
    context.wait_submission(context.submit_batch(commands));

    std::vector<IntegerBounds> before(p.count), after(p.count);
    context.download(first_snapshot, 0, before.data(), bound_bytes);
    context.download(output, 0, after.data(), bound_bytes);
    for (std::size_t i = 0; i < p.count; ++i) {
        auto expected_first = upstream_reference(bounds[i], distances[i], env_ids[i], false, p);
        auto expected_last = upstream_reference(bounds[i], distances[i], env_ids[i], true, p);
        for (std::size_t axis = 0; axis < 6; ++axis) {
            if (before[i].value[axis] != expected_first.value[axis] ||
                after[i].value[axis] != expected_last.value[axis])
                throw std::runtime_error("PhysX batch producer/copy/consumer order mismatch");
        }
    }
    context.destroy_buffer(first_snapshot);
    context.destroy_buffer(output);
    context.destroy_buffer(env);
    context.destroy_buffer(distance);
    context.destroy_buffer(box);
}

// PhysX buffer-grow behavior: the old allocation remains valid to GPU work
// already submitted, but its public handle is invalid immediately. Waiting
// on a later submission also completes prior work on the ordered queue.
void run_deferred_retirement(Context& context) {
    std::array<std::uint32_t, 73> source_values{};
    for (std::size_t i = 0; i < source_values.size(); ++i)
        source_values[i] = 0x193f0000u + static_cast<std::uint32_t>(i * 17);
    constexpr std::size_t bytes = sizeof(source_values);

    const BufferId source = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId first = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId second = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(source, 0, source_values.data(), bytes);

    const auto earlier = context.submit_copy_buffer(source, 0, first, 0, bytes);
    const auto later = context.submit_copy_buffer(source, 0, second, 0, bytes);
    context.retire_buffer(source);

    // Immediate public-handle invalidation is essential: later dispatches
    // must not access an allocation already scheduled for deferred release.
    bool rejected = false;
    try {
        context.submit_copy_buffer(source, 0, second, 0, bytes);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Retired GPU buffer ID remained usable");
    rejected = false;
    try {
        context.retire_buffer(source);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("GPU buffer accepted duplicate retirement");

    // A fresh buffer should be legal before either submission completes,
    // without reusing the backing resource still needed by the GPU.
    const BufferId replacement = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    // Avoid a synchronous upload here: that would implicitly drain both
    // earlier copies and mask premature release by the retirement path.
    context.wait_submission(later);
    if (!context.submission_complete(earlier))
        throw std::runtime_error("Same-queue earlier copy did not complete");
    context.wait_submission(earlier);

    std::array<std::uint32_t, 73> observed{};
    context.download(first, 0, observed.data(), bytes);
    if (observed != source_values)
        throw std::runtime_error("Deferred GPU source retirement corrupted earlier copy");
    context.download(second, 0, observed.data(), bytes);
    if (observed != source_values)
        throw std::runtime_error("Deferred GPU source retirement corrupted later copy");

    // No pending submissions: retirement should release immediately.
    context.retire_buffer(replacement);
    context.destroy_buffer(second);
    context.destroy_buffer(first);
}

// Queue two source snapshots without waiting; the host immediately mutates
// the original byte array. GPU copies must retain their own staging storage
// until the final dependent submission completes.
void run_async_staging(Context& context) {
    constexpr std::size_t words = 73;
    constexpr std::size_t bytes = words * sizeof(std::uint32_t);
    std::array<std::uint32_t, words> host{};
    std::array<std::uint32_t, words> expected_a{}, expected_b{};
    for (std::size_t i = 0; i < words; ++i) {
        expected_a[i] = 0xabc10000u + static_cast<std::uint32_t>(i);
        expected_b[i] = 0xdef20000u + static_cast<std::uint32_t>(i * 3);
    }
    const BufferId destination_a = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId destination_b = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId snapshot = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);

    host = expected_a;
    const auto first = context.submit_upload_buffer(destination_a, 0, host.data(), bytes);
    host = expected_b;
    const auto second = context.submit_upload_buffer(destination_b, 0, host.data(), bytes);
    host.fill(0xcdcdcdcd);
    // PhysX-style handoff: dependent GPU work is submitted immediately,
    // while the old destination is retired before its copy can be waited.
    const auto third = context.submit_copy_buffer(destination_b, 0, snapshot, 0, bytes);
    context.retire_buffer(destination_b);

    bool rejected = false;
    try {
        context.submit_upload_buffer(destination_b, 0, host.data(), bytes);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Async upload accepted retired destination");

    // Ensure staging still belongs to the Context even though the caller
    // may have changed/reused its source bytes in the meantime.
    context.wait_submission(third);
    if (!context.submission_complete(first) || !context.submission_complete(second))
        throw std::runtime_error("Earlier queued uploads did not complete");
    context.wait_submission(first);
    context.wait_submission(second);
    std::array<std::uint32_t, words> observed{};
    context.download(destination_a, 0, observed.data(), bytes);
    if (observed != expected_a)
        throw std::runtime_error("Async GPU upload did not snapshot first source");
    context.download(snapshot, 0, observed.data(), bytes);
    if (observed != expected_b)
        throw std::runtime_error("Async GPU upload or dependent copy corrupted second source");

    rejected = false;
    try {
        context.submit_upload_buffer(destination_a, 0, host.data(), 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Empty async upload was accepted");
    rejected = false;
    try {
        context.submit_upload_buffer(destination_a, 0, host.data(), bytes - 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Misaligned async upload was accepted");
    context.destroy_buffer(snapshot);
    context.destroy_buffer(destination_a);
}

// Source-derived HtoD sequencing contract used by the PhysX broadphase:
// multiple changed arrays and descriptor subranges share one queue submission.
// Cross-backend tests verify caller snapshot, same-destination ordered overlap,
// two destinations, dependent GPU copy, and retirement before completion.
void run_batched_uploads(Context& context) {
    constexpr std::size_t count = 73;
    constexpr std::size_t bytes = count * sizeof(std::uint32_t);
    constexpr std::size_t patch_start = 11;
    constexpr std::size_t patch_count = 13;

    std::array<std::uint32_t, count> first{}, second{}, expected{};
    std::array<std::uint32_t, patch_count> patch{};
    for (std::size_t i = 0; i < count; ++i) {
        first[i] = 0x12340000u + static_cast<std::uint32_t>(i * 7);
        second[i] = 0x56780000u + static_cast<std::uint32_t>(i * 19);
    }
    for (std::size_t i = 0; i < patch_count; ++i)
        patch[i] = 0x87650000u + static_cast<std::uint32_t>(i * 23);
    expected = first;
    std::copy(patch.begin(), patch.end(), expected.begin() + patch_start);

    const BufferId a = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId b = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const BufferId mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    const std::vector<UploadSpan> spans{
        {a, 0, first.data(), bytes},
        {b, 0, second.data(), bytes},
        {a, patch_start * sizeof(std::uint32_t), patch.data(), sizeof(patch)}
    };
    const auto uploaded = context.submit_upload_batch(spans);
    first.fill(0xcdcdcdcd);
    second.fill(0xcdcdcdcd);
    patch.fill(0xcdcdcdcd);
    // Consume one of the buffers in the next queued operation before
    // waiting for any of the staged uploads.
    const auto copied = context.submit_copy_buffer(b, 0, mirror, 0, bytes);
    context.retire_buffer(b);

    bool rejected = false;
    try {
        context.submit_upload_batch({{a, 0, first.data(), bytes},
                                     {b, 0, second.data(), bytes}});
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Batched upload accepted a retired destination");

    context.wait_submission(copied);
    if (!context.submission_complete(uploaded))
        throw std::runtime_error("Ordered upload batch did not finish before GPU copy");
    context.wait_submission(uploaded);

    std::array<std::uint32_t, count> actual{};
    context.download(a, 0, actual.data(), bytes);
    if (actual != expected)
        throw std::runtime_error("Overlapping ordered batched uploads diverged");
    context.download(mirror, 0, actual.data(), bytes);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t correct = 0x56780000u + static_cast<std::uint32_t>(i * 19);
        if (actual[i] != correct)
            throw std::runtime_error("Dependent GPU copy lost batched HtoD input");
    }
    // Incorrect batches are rejected before recording any GPU commands.
    for (const auto& invalid : std::vector<std::vector<UploadSpan>>{
            {},
            {{a, 0, nullptr, bytes}},
            {{a, 0, actual.data(), 0}},
            {{a, 0, actual.data(), bytes - 1}},
            {{a, 1, actual.data(), bytes - 4}},
            {{a, 0, actual.data(), bytes},
             {a, bytes - 4, actual.data(), 8}}
        }) {
        rejected = false;
        try {
            context.submit_upload_batch(invalid);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Invalid batched HtoD request was accepted");
    }

    // Reuse the completed command/descriptor infrastructure with another
    // independent batch and verify it does not inherit stale staging bytes.
    std::array<std::uint32_t, count> repeat{};
    for (std::size_t i = 0; i < count; ++i)
        repeat[i] = 0xa5a50000u + static_cast<std::uint32_t>(i);
    auto next = context.submit_upload_batch({{a, 0, repeat.data(), bytes}});
    repeat.fill(0);
    context.wait_submission(next);
    context.download(a, 0, actual.data(), bytes);
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != 0xa5a50000u + i)
            throw std::runtime_error("Reused upload batch retained stale data");
    }
    context.destroy_buffer(mirror);
    context.destroy_buffer(a);
}

void reject_invalid(AabbEncodingStage& stage) {
    std::vector<BatchCommand> commands;
    commands.emplace_back(CopyCommand{1, 0, 2, 0, 16});

    const std::array<AabbEncodingParameters, 4> invalid{{
        {0, 4, 4, 4, 0, 0, 0},
        {1, 32, 4, 4, 0, 0, 0},
        {1, 4, 4, 4, 5, 0, 0},
        {1, 4, 4, 4, 0, 5, 0}
    }};
    for (const auto& p : invalid) {
        bool rejected = false;
        try {
            stage.append(commands, 1, 2, 0, 3, p);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || commands.size() != 1) {
            throw std::runtime_error("Invalid PhysX broadphase configuration accepted");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected compiled PhysX AABB shader path");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        const auto& info = context.device_info();
        if (info.max_workgroup_invocations < 128 ||
            info.max_workgroup_size[0] < 128) {
            throw std::runtime_error("Compute device below core group-size limits");
        }
        ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
        shader.format = ShaderFormat::Dxil;
#else
        shader.format = ShaderFormat::Spirv;
#endif
        shader.bytes = read_shader(argv[1]);
        AabbEncodingStage stage(context, std::move(shader));
        reject_invalid(stage);
        run_case(context, stage, {1, 4, 4, 4, 0, 0, 0}, false);
        run_case(context, stage, {513, 4, 4, 4, 0, 0, 0}, false);
        run_case(context, stage, {73, 0, 1, 8, 0, 0, 0}, false);
        run_case(context, stage, {513, 4, 4, 4, 2, 3, 4}, true);
        run_case(context, stage, {47, 6, 2, 4, 2, 1, 0}, true);
        run_case(context, stage, {37, 6, 2, 4, 2, 1, 0}, false);
        run_ordered_batch(context, stage);
        run_deferred_retirement(context);
        run_async_staging(context);
        run_batched_uploads(context);
        std::cout << "PhysX GPU broadphase AABB encoding PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX GPU broadphase AABB encoding FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
