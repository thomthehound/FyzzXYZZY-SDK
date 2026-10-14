// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/created_pairs_arena.h"
#include "physx_gpu/radix_count_arena.h"
#include "physx_gpu/radix_scatter_arena.h"
#include "physx_gpu/sap_box_indices_arena.h"

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
using fyzz::physx_gpu::CreatedPairsArenaDesc;
using fyzz::physx_gpu::CreatedPairsArenaStage;
using fyzz::physx_gpu::RadixCountArenaDesc;
using fyzz::physx_gpu::RadixCountArenaStage;
using fyzz::physx_gpu::RadixScatterArenaDesc;
using fyzz::physx_gpu::RadixScatterArenaStage;
using fyzz::physx_gpu::SapBoxIndicesArenaDesc;
using fyzz::physx_gpu::SapBoxIndicesArenaStage;

constexpr std::uint32_t kUnused = 0xcdcdcdcd;
constexpr std::uint32_t kInvalid = 0x3fffffffu;
constexpr std::uint32_t kInvalidProjection = 0xffffffffu;
constexpr std::uint32_t kBlocks = 32;
constexpr std::uint32_t kDigits = 16;

std::vector<std::uint8_t> load_shader(const char* file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("PhysX SAP shader not found");
    const std::vector<char> values{std::istreambuf_iterator<char>{in},
                                   std::istreambuf_iterator<char>{}};
    return {values.begin(), values.end()};
}

struct Packed {
    std::vector<std::uint32_t> arena;
    std::uint32_t created_desc_offset = 0;
    std::uint32_t count_desc_offsets[2]{};
    std::uint32_t scatter_desc_offsets[2]{};
    std::uint32_t sap_desc_offset = 0;
    std::uint32_t projection_offsets[3][2]{};
    std::uint32_t handle_offsets[3][2]{};
    std::uint32_t current_sap_offsets[3]{};
    std::uint32_t new_sap_offsets[3]{};
    std::uint32_t histogram_offsets[3]{};
    std::uint32_t old_bounds_offset = 0;
    std::uint32_t new_bounds_offset = 0;
    std::vector<std::uint32_t> handles;
    std::uint32_t capacity = 0;
    std::uint32_t num_keys = 0;

    std::uint32_t reserve(std::uint32_t words) {
        while (arena.size() % 32u) arena.push_back(kUnused);
        const auto index = arena.size();
        if (index > std::numeric_limits<std::uint32_t>::max() / 4u ||
            words > std::numeric_limits<std::uint32_t>::max() / 4u - index)
            throw std::out_of_range("SAP integration arena exceeds 32-bit offsets");
        arena.resize(index + words, kUnused);
        return static_cast<std::uint32_t>(index * 4u);
    }
    void put(std::uint32_t offset, const void* bytes, std::size_t count) {
        if (offset % 4u || count % 4u ||
            offset > arena.size() * 4u || count > arena.size() * 4u - offset)
            throw std::runtime_error("Invalid SAP integration fixture write");
        std::memcpy(arena.data() + offset / 4u, bytes, count);
    }
    std::uint32_t& at(std::uint32_t offset) {
        if (offset % 4u || offset / 4u >= arena.size())
            throw std::runtime_error("SAP integration fixture out of bounds");
        return arena[offset / 4u];
    }
};

Packed pack(std::uint32_t created_count) {
    Packed p;
    p.capacity = created_count + 9u;
    const std::uint32_t padded_boxes = created_count + (created_count & 1u);
    p.num_keys = padded_boxes * 2u;
    p.created_desc_offset = p.reserve(sizeof(CreatedPairsArenaDesc) / 4u);
    for (std::uint32_t pass = 0; pass < 2u; ++pass) {
        p.count_desc_offsets[pass] = p.reserve(sizeof(RadixCountArenaDesc) / 4u);
        p.scatter_desc_offsets[pass] = p.reserve(sizeof(RadixScatterArenaDesc) / 4u);
    }
    p.sap_desc_offset = p.reserve(sizeof(SapBoxIndicesArenaDesc) / 4u);
    const auto created_handles = p.reserve(std::max(1u, created_count));
    p.new_bounds_offset = p.reserve(p.capacity * 6u);
    p.old_bounds_offset = p.reserve(p.capacity * 6u);

    for (std::uint32_t h = 0; h < p.capacity; ++h) {
        for (std::uint32_t c = 0; c < 6u; ++c) {
            p.at(p.old_bounds_offset + (h * 6u + c) * 4u) =
                0x90000000u + h * 12u + c;
        }
    }
    for (std::uint32_t i = 0; i < created_count; ++i) {
        const std::uint32_t handle = p.capacity - 1u - i;
        p.handles.push_back(handle);
        p.at(created_handles + i * 4u) = handle;
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            // Packed integer AABBs, as produced by translateAABBsLaunch.
            // Deliberately repeat endpoints to test stable key/handle pairs.
            const std::uint32_t lo =
                0x10000000u + (((i * 19u + axis * 23u) % 37u) * 32u);
            const std::uint32_t hi = lo + 5u + ((i + axis) % 11u) * 2u;
            p.at(p.new_bounds_offset + (handle * 6u + axis) * 4u) = lo;
            p.at(p.new_bounds_offset + (handle * 6u + axis + 3u) * 4u) = hi;
        }
    }
    CreatedPairsArenaDesc created{};
    created.num_created_handles = created_count;
    created.num_previous_handles = 0;
    created.created_handles_offset = created_handles;
    created.new_integer_bounds_offset = p.new_bounds_offset;
    created.prior.common.box_capacity = p.capacity;
    created.prior.common.endpoint_capacity = p.num_keys;
    created.prior.common.old_integer_bounds_offset = p.old_bounds_offset;

    SapBoxIndicesArenaDesc sap{};
    sap.box_capacity = p.capacity;
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        for (std::uint32_t parity = 0; parity < 2u; ++parity) {
            p.projection_offsets[axis][parity] = p.reserve(std::max(4u, p.num_keys));
            p.handle_offsets[axis][parity] = p.reserve(std::max(4u, p.num_keys));
        }
        p.histogram_offsets[axis] = p.reserve(kDigits * kBlocks);
        p.current_sap_offsets[axis] = p.reserve(p.capacity * 2u);
        p.new_sap_offsets[axis] = p.reserve(p.capacity * 2u);
        for (std::uint32_t i = 0; i < p.capacity * 2u; ++i) {
            p.at(p.current_sap_offsets[axis] + i * 4u) = kInvalid;
            p.at(p.new_sap_offsets[axis] + i * 4u) = kInvalid;
        }
        created.prior.projection_offsets[axis] = p.projection_offsets[axis][0];
        created.prior.common.box_handle_offsets[0][axis] = p.handle_offsets[axis][0];
        sap.axes[axis].sorted_handles_offset = p.handle_offsets[axis][0];
        sap.axes[axis].current_sap_offset = p.current_sap_offsets[axis];
        sap.axes[axis].new_sap_offset = p.new_sap_offsets[axis];
    }
    p.put(p.created_desc_offset, &created, sizeof(created));
    p.put(p.sap_desc_offset, &sap, sizeof(sap));
    for (std::uint32_t parity = 0; parity < 2u; ++parity) {
        RadixCountArenaDesc count{};
        RadixScatterArenaDesc scatter{};
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            count.axes[axis].input_keys_offset = p.projection_offsets[axis][parity];
            count.axes[axis].radix_counts_offset = p.histogram_offsets[axis];
            scatter.axes[axis].input_keys_offset = p.projection_offsets[axis][parity];
            scatter.axes[axis].input_ranks_offset = p.handle_offsets[axis][parity];
            scatter.axes[axis].output_keys_offset =
                p.projection_offsets[axis][1u-parity];
            scatter.axes[axis].output_ranks_offset =
                p.handle_offsets[axis][1u-parity];
            scatter.axes[axis].radix_counts_offset = p.histogram_offsets[axis];
        }
        p.put(p.count_desc_offsets[parity], &count, sizeof(count));
        p.put(p.scatter_desc_offsets[parity], &scatter, sizeof(scatter));
    }
    return p;
}

void oracle(const Packed& p, const std::vector<std::uint32_t>& actual,
            std::uint32_t count) {
    // Original kernel always clears old bounds for newly created handles.
    for (const auto handle : p.handles) {
        for (std::uint32_t c = 0; c < 6u; ++c) {
            const auto expected = c < 3u ? 0xff7fffffu : 0x00800000u;
            if (actual[p.old_bounds_offset / 4u + handle * 6u + c] != expected)
                throw std::runtime_error("PhysX created handle old AABB was not emptied");
        }
    }
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> endpoints;
        endpoints.reserve(p.num_keys);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto handle = p.handles[i];
            const auto lo = p.arena[p.new_bounds_offset / 4u + handle * 6u + axis];
            const auto hi = p.arena[p.new_bounds_offset / 4u + handle * 6u + axis + 3u];
            endpoints.emplace_back(lo, (handle << 3u) | 3u);
            endpoints.emplace_back(hi, (handle << 3u) | 2u);
        }
        if (count & 1u) {
            endpoints.emplace_back(kInvalidProjection, kInvalid);
            endpoints.emplace_back(kInvalidProjection, kInvalid);
        }
        // Equivalent to the eight stable four-bit radix passes.
        std::stable_sort(endpoints.begin(), endpoints.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<std::uint32_t> expected_sap(p.capacity * 2u, kInvalid);
        for (std::uint32_t i = 0; i < p.num_keys; ++i) {
            if (actual[p.projection_offsets[axis][0] / 4u + i] !=
                    endpoints[i].first ||
                actual[p.handle_offsets[axis][0] / 4u + i] !=
                    endpoints[i].second)
                throw std::runtime_error("PhysX created→radix keys/handles mismatch");
            if (i < count * 2u) {
                const auto tagged = endpoints[i].second;
                const auto handle = tagged >> 3u;
                expected_sap[handle * 2u + ((tagged & 1u) ? 0u : 1u)] = i;
            }
        }
        for (std::uint32_t i = 0; i < p.capacity * 2u; ++i) {
            for (const auto sap : {p.current_sap_offsets[axis],
                                   p.new_sap_offsets[axis]}) {
                if (actual[sap / 4u + i] != expected_sap[i])
                    throw std::runtime_error("PhysX sorted SAP endpoint map mismatch");
            }
        }
        // Created data, source input, and descriptors are persistent/readonly.
        for (std::uint32_t i = 0; i < p.capacity * 6u; ++i) {
            if (actual[p.new_bounds_offset / 4u + i] !=
                p.arena[p.new_bounds_offset / 4u + i])
                throw std::runtime_error("PhysX new AABB data was modified");
        }
    }
}

void run_case(Context& context, CreatedPairsArenaStage& created,
              RadixCountArenaStage& count_stage, RadixScatterArenaStage& scatter,
              SapBoxIndicesArenaStage& sap, std::uint32_t count) {
    auto p = pack(count);
    const auto bytes = static_cast<std::uint32_t>(p.arena.size() * 4u);
    auto gpu = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    auto mirror = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
    context.upload(gpu, 0, p.arena.data(), bytes);
    std::vector<BatchCommand> ops;
    created.append(ops, gpu, bytes, p.created_desc_offset, count);
    if (count) {
        for (std::uint32_t pass = 0; pass < 8u; ++pass) {
            const auto parity = pass & 1u;
            count_stage.append(ops, gpu, bytes, p.count_desc_offsets[parity],
                               p.num_keys, pass * 4u);
            scatter.append(ops, gpu, bytes, p.scatter_desc_offsets[parity],
                           p.num_keys, pass * 4u);
        }
    }
    sap.append(ops, gpu, bytes, p.sap_desc_offset, count, false);
    sap.append(ops, gpu, bytes, p.sap_desc_offset, count, true);
    if (ops.size() != (count ? 19u : 0u))
        throw std::runtime_error("PhysX SAP creation-sort-index stage order changed");
    if (count) {
        const auto& first = std::get<DispatchCommand>(ops.front());
        if (first.groups.x != 32u || first.groups.y != 1u)
            throw std::runtime_error("PhysX created-handle launch changed");
        for (std::uint32_t j = 17u; j < 19u; ++j) {
            const auto& final = std::get<DispatchCommand>(ops[j]);
            const auto nb = (count * 2u + 255u) / 256u;
            if (final.groups.x != nb || final.groups.y != 1u)
                throw std::runtime_error("PhysX SAP endpoint dispatch changed");
        }
    }
    ops.emplace_back(CopyCommand{gpu, 0, mirror, 0, bytes});
    context.wait_submission(context.submit_batch(ops));
    std::vector<std::uint32_t> actual(p.arena.size());
    context.download(mirror, 0, actual.data(), bytes);
    if (count) oracle(p, actual, count);
    else if (actual != p.arena)
        throw std::runtime_error("Zero-work PhysX SAP path modified state");
    context.destroy_buffer(mirror);
    context.destroy_buffer(gpu);
}

void negative_cases(SapBoxIndicesArenaStage& sap) {
    std::vector<BatchCommand> ops;
    for (const auto offset : {1u, 128u, 256u}) {
        bool rejected = false;
        try { sap.append(ops, 1, 160u, offset, 1u, false); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || !ops.empty())
            throw std::runtime_error("Invalid SAP box descriptor accepted");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("Expected created, count, scatter, SAP shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        auto shader = [](const char* path) {
            ShaderBinary out;
#if defined(FYZZXYZZY_TEST_D3D12)
            out.format = ShaderFormat::Dxil;
#else
            out.format = ShaderFormat::Spirv;
#endif
            out.bytes = load_shader(path);
            return out;
        };
        CreatedPairsArenaStage created(context, shader(argv[1]));
        RadixCountArenaStage count(context, shader(argv[2]));
        RadixScatterArenaStage scatter(context, shader(argv[3]));
        SapBoxIndicesArenaStage sap(context, shader(argv[4]));
        negative_cases(sap);
        run_case(context, created, count, scatter, sap, 0u);
        run_case(context, created, count, scatter, sap, 1u);
        run_case(context, created, count, scatter, sap, 5u);
        run_case(context, created, count, scatter, sap, 129u);
        run_case(context, created, count, scatter, sap, 257u);
        std::cout << "PhysX created-handle → radix → SAP index oracle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PhysX SAP index integration FAILED: " << e.what() << '\n';
        return 1;
    }
}
