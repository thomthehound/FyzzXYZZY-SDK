// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void exercise_source_roots(Context& context) {
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots device_roots(context, addresses);
    SourcePhysxDeviceRootCallback device_callback(device_roots, &stream);
    SourcePhysxMappedReportRoots host_roots(
        context, addresses, stream,
        SourcePhysxMappedReportRoots::Mode::ForceStaged);

    physx::Cm::VirtualAllocatorCallback& source_device = device_callback;
    physx::Cm::VirtualAllocatorCallback& source_mapped = host_roots;

    constexpr std::array<std::uint32_t, 8> expected{
        101u, 203u, 307u, 409u, 503u, 607u, 709u, 811u};
    constexpr std::size_t bytes = sizeof(expected);
    void* device_raw = source_device.allocate(4096u, 0, __FILE__, __LINE__);
    check(device_raw != nullptr && device_roots.live_roots() == 1u,
          "Original PhysX GPU root callback failed to allocate one page");
    const auto gpu_address = static_cast<CudaDeviceAddressMap::Address>(
        reinterpret_cast<std::uintptr_t>(device_raw));
    const auto gpu_view = addresses.resolve(gpu_address + 128u, bytes);
    context.upload(gpu_view.buffer, gpu_view.offset, expected.data(), bytes);

    void* report = source_mapped.allocate(256u, 0, __FILE__, __LINE__);
    check(report != nullptr && host_roots.live_roots() == 1u &&
          !host_roots.is_direct(report, bytes) &&
          (reinterpret_cast<std::uintptr_t>(report) & 255u) == 0u,
          "Original PhysX mapped-root allocator lost staged host allocation");
    auto* report_words = static_cast<std::uint32_t*>(report);
    const auto report_address = host_roots.device_address(report, bytes);
    const auto report_view = addresses.resolve(report_address, bytes);
    const auto inner_address = host_roots.device_address(report_words + 1u, bytes - 4u);
    const auto inner_view = addresses.resolve(inner_address, bytes - 4u);
    check(inner_view.buffer == report_view.buffer &&
          inner_view.offset == report_view.offset + 4u,
          "Original PhysX mapped heap lost interior report identity");

    bool rejected = false;
    try {
        (void)host_roots.device_address(report_words + 63u, bytes);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    check(rejected && host_roots.live_roots() == 1u,
          "Invalid original PhysX report view changed heap ownership");

    // Submit the real native device-to-device transfer on the source stream.
    // Retire its input root before the GPU completes; original source
    // submission ordering must retain the resource until its fence.
    stream.queue_device_copy(report_address, gpu_address + 128u, bytes);
    source_device.deallocate(device_raw);
    check(device_roots.live_roots() == 0u && addresses.live_allocations() == 1u,
          "Original PhysX root was not retired after enqueuing the GPU copy");
    host_roots.publish_at_fetch(report, bytes);
    for (std::size_t i = 0; i < expected.size(); ++i)
        check(report_words[i] == expected[i],
              "Original PhysX GPU-to-mapped-report publication lost source bytes");

    // The same pinned callback supplies CPU-written input descriptors.
    // The original source DMA takes a host snapshot before the caller
    // reuses its pinned allocation.
    constexpr std::array<std::uint32_t, 8> cpu_input{
        919u, 823u, 727u, 631u, 535u, 439u, 343u, 247u};
    for (std::size_t i = 0; i < cpu_input.size(); ++i)
        report_words[i] = cpu_input[i];
    const std::vector<SourceSapUpload> uploads{{report_address, report, bytes}};
    stream.queue_uploads(uploads);
    std::fill_n(report_words, cpu_input.size(), 0u);
    stream.wait_at_fetch();
    std::array<std::uint32_t, 8> device_readback{};
    context.download(report_view.buffer, report_view.offset,
                     device_readback.data(), bytes);
    check(device_readback == cpu_input,
          "Original PhysX pinned CPU descriptor was not snapshotted to GPU");

    source_mapped.deallocate(report);
    check(host_roots.live_roots() == 0u && addresses.live_allocations() == 0u,
          "Original mapped PhysX GPU report root leaked after fetch");
    device_roots.close(&stream);
    check(device_roots.live_roots() == 0u,
          "Original PhysX GPU root teardown retained native allocations");
}
} // namespace

int main() {
    try {
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        exercise_source_roots(context);
        std::cout << "Original PhysX GPU root and mapped callback lifecycle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original PhysX GPU root callback FAILED: " << e.what() << '\n';
        return 1;
    }
}
