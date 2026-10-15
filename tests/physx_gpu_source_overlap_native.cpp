// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_overlap_work_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace fyzz::physx_gpu;
using namespace fyzz::compute;
namespace {
constexpr std::size_t kRootBytes = 8192u;
constexpr std::uint32_t kCanary = 0xcdcdcdcd;
constexpr std::uint32_t kBoxes = 2u, kProjectionSlots = 6u;
constexpr std::uint32_t kRegionCapacity = 64u * kBoxes;

void require(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
}
template<typename Fn>
void rejects(Fn&& fn, const char* why) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, why);
}
ShaderBinary load_shader(const char* file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("PhysX source overlap shader missing");
    const std::vector<char> bytes{std::istreambuf_iterator<char>{input},
                                  std::istreambuf_iterator<char>{}};
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format = ShaderFormat::Dxil;
#else
    shader.format = ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(), bytes.end());
    return shader;
}
struct Root {
    BufferId native = 0;
    std::uint32_t next = 256u;
    std::vector<std::uint32_t> image =
        std::vector<std::uint32_t>(kRootBytes / 4u, kCanary);

    std::uint32_t reserve(std::uint32_t bytes) {
        const auto at = (next + 127u) & ~127u; // PhysX aligned device pointers
        if (bytes > kRootBytes - at)
            throw std::runtime_error("Source multi-root test heap exhausted");
        next = at + bytes;
        return at;
    }
    void set(std::uint32_t start, std::uint32_t word, std::uint32_t value) {
        image[start / 4u + word] = value;
    }
};

void run_case(Context& context, const char* shader, bool shared_input_root) {
    CudaDeviceAddressMap map;
    std::array<Root, 3> roots;
    for (auto& r : roots)
        r.native = context.create_buffer(kRootBytes, MemoryPlacement::DeviceLocal);

    auto allocate = [&](int root, std::uint32_t bytes) {
        auto& r = roots[root];
        const auto offset = r.reserve(bytes);
        return std::pair<CudaDeviceAddressMap::Address, std::uint32_t>(
            map.register_slice(r.native, offset, bytes), offset);
    };
    auto bind = [&](auto& field, int root, std::uint32_t bytes) {
        const auto [pointer, offset] = allocate(root, bytes);
        using Pointer = std::remove_reference_t<decltype(field)>;
        field = reinterpret_cast<Pointer>(static_cast<std::uintptr_t>(pointer));
        return offset;
    };

    physx::PxgBroadPhaseDesc host{};
    host.numPreviousHandles = 2u;
    host.numCreatedHandles = 1u;
    host.numRemovedHandles = 1u;
    host.regionAccumTotal = std::numeric_limits<std::uint32_t>::max();
    const auto [resident, descriptorOffset] =
        allocate(0, static_cast<std::uint32_t>(sizeof(host)));

    const auto sorted = bind(host.boxHandles[0][0], 0, kProjectionSlots * 4u);
    const auto positions = bind(host.boxSapBox1D[0], 0, kBoxes * 8u);
    const auto ranges = bind(host.regionRange, 0, kBoxes * 32u);
    const auto prefix = bind(host.regionAccum, 0, kProjectionSlots * 4u);
    const int histRoot = shared_input_root ? 0 : 1;
    const auto active = bind(host.activeRegionsHistogram, histRoot,
                             64u * kProjectionSlots * 4u);
    const auto starts = bind(host.startRegionsHistogram, histRoot,
                             64u * kProjectionSlots * 4u);
    const auto counts = bind(host.overlapChecksRegion, 2, kRegionCapacity * 8u);
    const auto regions = bind(host.overlapChecksHandleRegiones, 2,
                              kRegionCapacity * 8u);

    // The source-owned descriptor is in root 0. The preceding PhysX producer
    // has written the *resident* total; the host snapshot is deliberately
    // stale. No host readback is permitted in field lowering.
    auto gpuDescriptor = host;
    gpuDescriptor.regionAccumTotal = 2u;
    std::memcpy(reinterpret_cast<std::uint8_t*>(roots[0].image.data()) +
                descriptorOffset, &gpuDescriptor, sizeof(gpuDescriptor));

    const std::uint32_t tags[] = {3u, 9u, 2u, 8u};
    for (std::uint32_t i = 0; i < 4u; ++i)
        roots[0].set(sorted, i, tags[i]);
    roots[0].set(positions, 0u, 0u);
    roots[0].set(positions, 1u, 2u);
    roots[0].set(positions, 2u, 1u);
    roots[0].set(positions, 3u, 3u);
    for (std::uint32_t i = 0; i < 16u; ++i)
        roots[0].set(ranges, i, 0u);
    for (std::uint32_t i = 0; i < kProjectionSlots; ++i)
        roots[0].set(prefix, i, i == 0u ? 0u : (i == 1u ? 1u : 2u));
    for (std::uint32_t i = 0; i < 4u; ++i) {
        roots[histRoot].set(active, i, i < 2u ? i : 2u);
        roots[histRoot].set(starts, i, i < 2u ? i : 2u);
    }

    SourceOverlapWorkLayout layout{resident, kBoxes, kBoxes,
                                    kProjectionSlots, kRegionCapacity,
                                    kRegionCapacity};
    SourceOverlapWorkNativeStage stage(context, load_shader(shader));
    std::vector<BatchCommand> commands;

    auto idle = host;
    idle.numCreatedHandles = 0u;
    stage.append(commands, idle, map, layout);
    require(commands.empty(), "Idle PhysX frame must not dispatch overlap work");
    auto missing = host;
    missing.regionAccum = nullptr;
    rejects([&]{ stage.append(commands, missing, map, layout); },
            "Missing source region prefix accepted");
    auto insufficient = layout;
    insufficient.region_work_capacity = 1u;
    rejects([&]{ stage.append(commands, host, map, insufficient); },
            "Source overlap allocation smaller than original capacity accepted");
    require(commands.empty(), "Rejected source dispatch modified command list");

    // Upload all source-owned roots in one ordered transfer, then submit the
    // original 32x256 shader without a host wait or per-kernel repacking.
    std::vector<UploadSpan> uploads;
    for (const auto& r : roots)
        uploads.push_back({r.native, 0u, r.image.data(), kRootBytes});
    const auto uploaded = context.submit_upload_batch(uploads);
    stage.append(commands, host, map, layout);
    require(commands.size() == 1u, "Source PhysX producer did not append exactly once");
    const auto& dispatch = std::get<DispatchCommand>(commands[0]);
    require(dispatch.groups.x == 32u && dispatch.groups.y == 1u &&
            dispatch.groups.z == 1u && dispatch.buffers.size() == 9u,
            "Source dispatch shape or binding manifest changed");
    require(dispatch.buffers[4].buffer == roots[0].native &&
            dispatch.buffers[5].buffer == roots[histRoot].native &&
            dispatch.buffers[7].buffer == roots[2].native &&
            dispatch.buffers[8].buffer == roots[2].native,
            "Original heap roots were packed or changed during lowering");
    for (std::uint32_t i = 0; i < 9u; ++i)
        require(dispatch.buffers[i].access ==
                    (i >= 7u ? BufferAccess::Write : BufferAccess::Read),
                "Native shader access manifest is incorrect");

    const auto completed = context.submit_batch(commands);
    context.wait_submission(completed);
    context.wait_submission(uploaded);

    auto expected = roots[2].image;
    expected[counts / 4u] = 1u;
    expected[counts / 4u + 1u] = 0u;
    expected[counts / 4u + 2u] = 0u;
    expected[counts / 4u + 3u] = 0u;
    expected[regions / 4u] = 0u;
    expected[regions / 4u + 1u] = 0u;
    expected[regions / 4u + 2u] = 1u;
    expected[regions / 4u + 3u] = 0u;

    std::vector<std::uint32_t> actual(kRootBytes / 4u);
    context.download(roots[2].native, 0u, actual.data(), kRootBytes);
    require(actual == expected,
            "Source-derived multi-root overlap-work output or canaries differ");

    // The original PhysX GPU descriptor and independent input roots remain
    // unmodified, including the GPU-owned inline regionAccumTotal field.
    for (int i = 0; i < 2; ++i) {
        context.download(roots[i].native, 0u, actual.data(), kRootBytes);
        require(actual == roots[i].image, "PhysX source input root was modified");
    }

    for (auto& r : roots) context.destroy_buffer(r.native);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Expected compiled source PhysX overlap shader");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        run_case(context, argv[1], false);
        run_case(context, argv[1], true);
        std::cout << "PhysX source-derived native multi-root overlap-work PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "PhysX source-derived overlap-work FAILED: "
                  << ex.what() << '\n';
        return 1;
    }
}
