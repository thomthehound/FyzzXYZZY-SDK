// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_overlap_prefix_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;

namespace {
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::size_t kRootBytes = 262144u;

void check(bool yes, const char* description) {
    if (!yes) throw std::runtime_error(description);
}
template<class Fn>
void expect_throw(Fn&& fn, const char* description) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    check(caught, description);
}
ShaderBinary load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Native PhysX overlap scan shader missing");
    const std::vector<char> contents{std::istreambuf_iterator<char>(f),
                                     std::istreambuf_iterator<char>()};
    ShaderBinary binary;
#if defined(FYZZXYZZY_TEST_D3D12)
    binary.format = ShaderFormat::Dxil;
#else
    binary.format = ShaderFormat::Spirv;
#endif
    binary.bytes.assign(contents.begin(), contents.end());
    return binary;
}
struct Root {
    BufferId id = 0;
    std::vector<std::uint32_t> original =
        std::vector<std::uint32_t>(kRootBytes / 4u, kCanary);
    std::uint32_t cursor = 256u;
    std::uint32_t allocate(std::uint32_t bytes) {
        const auto at = (cursor + 127u) & ~127u;
        if (!bytes || bytes > kRootBytes - at)
            throw std::runtime_error("PhysX scan test root exhausted");
        cursor = at + bytes;
        return at;
    }
    void put64(std::uint32_t at, std::uint64_t value) {
        original.at(at/4u) = static_cast<std::uint32_t>(value);
        original.at(at/4u + 1u) = static_cast<std::uint32_t>(value >> 32u);
    }
    std::uint64_t get64(std::uint32_t at) const {
        return static_cast<std::uint64_t>(original.at(at/4u)) |
               (static_cast<std::uint64_t>(original.at(at/4u+1u)) << 32u);
    }
};
void run_case(Context& ctx, const char* first, const char* second,
              std::uint32_t n, bool wide, bool shared_root) {
    CudaDeviceAddressMap addresses;
    std::array<Root, 3> roots{};
    for (auto& r : roots) r.id = ctx.create_buffer(kRootBytes, MemoryPlacement::DeviceLocal);
    auto allocate = [&](std::uint32_t root, std::uint32_t size) {
        auto& native = roots.at(root);
        const auto local = native.allocate(size);
        const auto virtual_address = addresses.register_slice(native.id, local, size);
        return std::pair<CudaDeviceAddressMap::Address, std::uint32_t>{
            virtual_address, local};
    };
    auto bind = [&](auto& target, std::uint32_t root, std::uint32_t bytes) {
        const auto [gpu, offset] = allocate(root, bytes);
        using Pointer = std::remove_reference_t<decltype(target)>;
        target = reinterpret_cast<Pointer>(static_cast<std::uintptr_t>(gpu));
        return offset;
    };
    const auto currentBoxes = std::max(1u, (n + 63u) / 64u);
    const auto capacity = currentBoxes * 64u;
    physx::PxgBroadPhaseDesc source{};
    source.numPreviousHandles = currentBoxes;
    source.numCreatedHandles = 1u;
    source.numRemovedHandles = 1u;
    // The host copy deliberately does NOT know the GPU-updated count.
    source.regionAccumTotal = 0xffffffffu;
    source.overlapChecksTotalRegion = 0xfffffffffffffffeull;
    const std::uint32_t descriptorRoot = shared_root ? 0u : 2u;
    const auto [descriptor, descOffset] =
        allocate(descriptorRoot, static_cast<std::uint32_t>(sizeof(source)));
    const auto countsOffset = bind(source.overlapChecksRegion, 0u,
                                   capacity * sizeof(physx::regionOverlapType));
    const std::uint32_t blockRoot = shared_root ? 0u : 1u;
    const auto blocksOffset = bind(source.blockOverlapChecksRegion,
                                   blockRoot, 32u * sizeof(physx::regionOverlapType));
    auto gpuDescriptor = source;
    gpuDescriptor.regionAccumTotal = n;
    gpuDescriptor.overlapChecksTotalRegion = 0u;
    std::memcpy(reinterpret_cast<std::uint8_t*>(
                    roots[descriptorRoot].original.data()) + descOffset,
                &gpuDescriptor, sizeof(gpuDescriptor));

    std::uint64_t expectedTotal = 0u;
    std::array<std::uint64_t, 32> blockValues{};
    const std::uint64_t perBlock =
        std::uint64_t(((n + 255u) / 256u + 31u) / 32u) * 256u;
    for (std::uint32_t i = 0u; i < n; ++i) {
        const auto value = wide
            ? (std::uint64_t{1u << 30u} + (i % 19u))
            : std::uint64_t((i * 13u + i / 7u) % 11u);
        roots[0].put64(countsOffset + i * 8u, value);
    }
    auto expected = roots;
    for (std::uint32_t block = 0u; block < 32u; ++block) {
        const auto begin = std::uint64_t(block) * perBlock;
        const auto end = std::min<std::uint64_t>(n, begin + perBlock);
        const auto starting = expectedTotal;
        for (std::uint64_t j = begin; j < end; ++j) {
            const auto at = countsOffset + static_cast<std::uint32_t>(j) * 8u;
            const auto value = roots[0].get64(at);
            expected[0].put64(at, expectedTotal);
            expectedTotal += value;
        }
        blockValues[block] = expectedTotal - starting;
        expected[blockRoot].put64(blocksOffset + block * 8u, blockValues[block]);
    }
    expected[descriptorRoot].put64(
        descOffset + offsetof(physx::PxgBroadPhaseDesc, overlapChecksTotalRegion),
        expectedTotal);

    SourceOverlapPrefixNativeStage stage(ctx, load(first), load(second));
    const SourceOverlapPrefixLayout layout{descriptor, currentBoxes, capacity};
    std::vector<BatchCommand> cmds;
    auto idle = source; idle.numCreatedHandles = 0u;
    stage.append(cmds, idle, addresses, layout);
    check(cmds.empty(), "Original no-created-handles gate violated");
    auto invalid = layout;
    invalid.overlap_capacity = 1u;
    expect_throw([&]{stage.append(cmds, source, addresses, invalid);},
                 "Underallocated source overlap prefix accepted");
    auto bad_source = source;
    bad_source.blockOverlapChecksRegion = nullptr;
    expect_throw([&]{stage.append(cmds, bad_source, addresses, layout);},
                 "Missing source scan scratch buffer accepted");
    check(cmds.empty(), "Rejected source scan left a partial command batch");

    std::vector<UploadSpan> uploads;
    for (auto& root : roots)
        uploads.push_back({root.id, 0u, root.original.data(), kRootBytes});
    const auto up = ctx.submit_upload_batch(uploads);
    stage.append(cmds, source, addresses, layout);
    check(cmds.size() == 2u, "Original two-kernel scan chain was changed");
    for (std::size_t i = 0; i < 2u; ++i) {
        const auto& c = std::get<DispatchCommand>(cmds[i]);
        check(c.groups.x == 32u && c.groups.y == 1u && c.groups.z == 1u &&
              c.push_constants.size() == 32u && c.buffers.size() == 4u,
              "Source 32x256 scan launch geometry or ABI mismatch");
        check(c.buffers[0].buffer == roots[0].id &&
              c.buffers[1].buffer == roots[blockRoot].id &&
              c.buffers[2].buffer == roots[descriptorRoot].id &&
              c.buffers[3].buffer == roots[descriptorRoot].id,
              "Original source heap allocation roots were repacked");
    }
    // The two original kernels run consecutively without a host fence.
    ctx.wait_submission(ctx.submit_batch(cmds));
    ctx.wait_submission(up);

    std::vector<std::uint32_t> actual(kRootBytes / 4u);
    for (std::size_t i = 0; i < roots.size(); ++i) {
        ctx.download(roots[i].id, 0u, actual.data(), kRootBytes);
        check(actual == expected[i].original,
              "PhysX native source overlap scan output or untouched canary mismatch");
    }
    for (auto& root : roots) ctx.destroy_buffer(root.id);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("Expected first and second original source scan shaders");
        Context ctx(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& gpu = ctx.device_info();
        if (!gpu.supports_int64_wave || !gpu.supports_wave_arithmetic ||
            gpu.subgroup_size < 4u || gpu.subgroup_size > 128u ||
            256u % gpu.subgroup_size != 0u) {
            std::cout << "SKIP: 64-bit wave operations unavailable\n";
            return 77;
        }
        for (const auto n : {0u, 1u, 31u, 257u, 8193u, 16385u}) {
            run_case(ctx, argv[1], argv[2], n, false, false);
            run_case(ctx, argv[1], argv[2], n, true, false);
        }
        run_case(ctx, argv[1], argv[2], 257u, true, true);
        run_case(ctx, argv[1], argv[2], 8193u, true, true);
        std::cout << "Source-derived PhysX 64-bit multi-root scan chain PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Source-derived PhysX overlap scan FAILED: "
                  << ex.what() << '\n';
        return 1;
    }
}
