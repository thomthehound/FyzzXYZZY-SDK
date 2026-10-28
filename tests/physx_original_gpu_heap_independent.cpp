// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_dma_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Errors final : physx::PxErrorCallback {
    bool reported = false;
    void reportError(physx::PxErrorCode::Enum,
                     const char* message, const char*, int) override {
        reported = true;
        std::cerr << "Original PhysX Foundation: " << message << '\n';
    }
};

struct Foundation {
    physx::PxDefaultAllocator allocator;
    Errors errors;
    physx::PxFoundation* instance;
    Foundation() : instance(PxCreateFoundation(
        PX_PHYSICS_VERSION, allocator, errors)) {
        require(instance != nullptr, "Source PhysX Foundation creation failed");
    }
    ~Foundation() { if(instance) instance->release(); }
};

void exercise(Context& context) {
    Foundation foundation;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted = false;
    const auto abort = [](void* data) noexcept {
        *static_cast<bool*>(data) = true;
    };
    constexpr auto group = physx::PxsHeapStats::eBROADPHASE;

    {
        // One original PhysX buddy page; all small allocations must share
        // its 128-byte block tree, not independently allocated GPU buffers.
        physx::PxgHeapMemoryAllocator heap(8192u, callback, abort, &aborted);
        require(roots.live_roots() == 1u,
                "Original PhysX buddy heap did not allocate its root page");
        physx::PxgCudaBuffer buffer(heap, group);
        physx::PxgCudaBuffer companion(heap, group);
        buffer.allocate(256u, PX_FL);
        companion.allocate(256u, PX_FL);
        require(heap.getHeapStats().stats[group] == 512u &&
                roots.live_roots() == 1u,
                "Original PhysX block splitting changed memory accounting");

        constexpr std::array<std::uint32_t, 8> contents{
            11u, 17u, 23u, 29u, 31u, 37u, 41u, 43u};
        const auto oldAddress = buffer.getDevicePtr();
        const auto oldView = addresses.resolve(oldAddress, sizeof(contents));
        context.upload(oldView.buffer, oldView.offset,
                       contents.data(), sizeof(contents));

        buffer.allocateCopyOldDataAsync(512u, stream, PX_FL);
        require(buffer.getSize() == 512u &&
                buffer.getDevicePtr() != oldAddress &&
                heap.getHeapStats().stats[group] == 1024u,
                "Original PhysX GPU buffer growth bypassed deferred heap ownership");
        stream.wait_at_fetch();
        std::array<std::uint32_t, 8> observed{};
        const auto newView = addresses.resolve(buffer.getDevicePtr(), sizeof(observed));
        context.download(newView.buffer, newView.offset,
                         observed.data(), sizeof(observed));
        require(observed == contents,
                "Original PhysX growth failed to copy GPU bytes on source stream");

        heap.flushDeferredDeallocs();
        require(heap.getHeapStats().stats[group] == 768u,
                "Original PhysX deferred free was not accounted at fetch");
        physx::PxgCudaBuffer reused(heap, group);
        reused.allocate(256u, PX_FL);
        require(reused.getDevicePtr() == oldAddress && roots.live_roots() == 1u,
                "Original PhysX buddy block was not reused after fetch");
        companion.deallocate();
        buffer.deallocate();
        reused.deallocate();
        require(heap.getHeapStats().stats[group] == 0u,
                "Original PhysX buddy blocks leaked after release");

        physx::PxgCudaBuffer exceptional(heap, group);
        exceptional.allocate(5000u, PX_FL);
        require(roots.live_roots() == 2u &&
                heap.getHeapStats().stats[group] == 5000u,
                "Original PhysX exceptional allocation did not use its own root");
        exceptional.deallocate();
        require(roots.live_roots() == 1u &&
                heap.getHeapStats().stats[group] == 0u,
                "Original PhysX exceptional root did not retire");
        require(!aborted && !foundation.errors.reported,
                "Original PhysX source heap reported a corrupted allocation");
    }

    roots.close(&stream);
    require(roots.live_roots() == 0u &&
            addresses.live_allocations() == 0u,
            "Original PhysX GPU heap failed to release its native root mappings");
}

void exercise_manager(Context& context) {
    Foundation foundation;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots deviceRoots(context, addresses);
    SourcePhysxDeviceRootCallback deviceCallback(deviceRoots, &stream);
    SourcePhysxMappedReportRoots pinnedRoots(
        context, addresses, stream,
        SourcePhysxMappedReportRoots::Mode::ForceStaged);
    SourcePhysxMappedReportRoots mappedRoots(
        context, addresses, stream,
        SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted = false;
    const auto abort = [](void* data) noexcept {
        *static_cast<bool*>(data) = true;
    };
    constexpr auto group = physx::PxsHeapStats::eBROADPHASE;

    // Normal PhysX mode shares a single mapped host heap between both
    // allocator-descriptor views. Device roots remain separate.
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            8192u, deviceCallback, pinnedRoots, mappedRoots, abort, &aborted);
        auto desc = physx::PxgCreateAllocatorDesc(manager);
        require(&desc.deviceAlloc == manager.mDeviceMemoryAllocator &&
                &desc.hostAlloc == &desc.hostMappedAlloc &&
                &desc.hostAlloc == manager.mPinnedHostMappedMemoryAllocator &&
                deviceRoots.live_roots() == 1u &&
                mappedRoots.live_roots() == 1u &&
                pinnedRoots.live_roots() == 0u,
                "Original PhysX normal heap-manager root aliasing changed");

        void* input = desc.hostAlloc.allocate(256u, group, PX_FL);
        void* report = desc.hostMappedAlloc.allocate(256u, group, PX_FL);
        require(input && report && input != report &&
                mappedRoots.live_roots() == 1u,
                "Original PhysX shared mapped host heap did not split a root");
        constexpr std::array<std::uint32_t, 8> contents{
            101u, 203u, 307u, 409u, 503u, 607u, 709u, 811u};
        const auto reportAddress = mappedRoots.device_address(
            report, sizeof(contents));
        const auto reportView = addresses.resolve(
            reportAddress, sizeof(contents));
        context.upload(reportView.buffer, reportView.offset,
                       contents.data(), sizeof(contents));
        mappedRoots.publish_at_fetch(report, sizeof(contents));
        require(std::equal(contents.begin(), contents.end(),
                           static_cast<std::uint32_t*>(report)),
                "Original PhysX shared host report did not publish GPU data");

        std::copy(contents.begin(), contents.end(),
                  static_cast<std::uint32_t*>(input));
        const auto inputAddress = mappedRoots.device_address(
            input, sizeof(contents));
        const auto inputView = addresses.resolve(inputAddress, sizeof(contents));
        stream.queue_uploads({{inputAddress, input, sizeof(contents)}});
        std::fill_n(static_cast<std::uint32_t*>(input), contents.size(), 0u);
        stream.wait_at_fetch();
        std::array<std::uint32_t, 8> observed{};
        context.download(inputView.buffer, inputView.offset,
                         observed.data(), sizeof(observed));
        require(observed == contents,
                "Original PhysX pinned input descriptor lost its source snapshot");

        desc.hostAlloc.deallocate(input);
        desc.hostMappedAlloc.deallocate(report);
        require(manager.getDeviceMemorySize() == 8192u &&
                manager.getDeviceHeapStats().stats[group] == 0u,
                "Original PhysX normal heap-manager accounting changed");
    }
    require(deviceRoots.live_roots() == 0u &&
            pinnedRoots.live_roots() == 0u &&
            mappedRoots.live_roots() == 0u &&
            addresses.live_allocations() == 0u,
            "Original PhysX normal heap-manager roots leaked");

    // Zero-capacity source debug mode uses distinct pinned and mapped host
    // heaps; each allocation is an exceptional root, not a buddy subblock.
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            0u, deviceCallback, pinnedRoots, mappedRoots, abort, &aborted);
        auto desc = physx::PxgCreateAllocatorDesc(manager);
        require(&desc.hostAlloc != &desc.hostMappedAlloc &&
                !deviceRoots.live_roots() && !pinnedRoots.live_roots() &&
                !mappedRoots.live_roots(),
                "Original PhysX debug heap-manager policy changed");
        void* device = desc.deviceAlloc.allocate(256u, group, PX_FL);
        void* pinned = desc.hostAlloc.allocate(256u, group, PX_FL);
        void* mapped = desc.hostMappedAlloc.allocate(256u, group, PX_FL);
        require(device && pinned && mapped &&
                deviceRoots.live_roots() == 1u &&
                pinnedRoots.live_roots() == 1u &&
                mappedRoots.live_roots() == 1u &&
                manager.getDeviceHeapStats().stats[group] == 256u,
                "Original PhysX debug heaps did not allocate distinct roots");
        desc.deviceAlloc.deallocate(device);
        desc.hostAlloc.deallocate(pinned);
        desc.hostMappedAlloc.deallocate(mapped);
        require(manager.getDeviceHeapStats().stats[group] == 0u,
                "Original PhysX debug heap-manager accounting failed");
    }
    deviceRoots.close(&stream);
    require(deviceRoots.live_roots() == 0u &&
            pinnedRoots.live_roots() == 0u &&
            mappedRoots.live_roots() == 0u &&
            addresses.live_allocations() == 0u &&
            !aborted && !foundation.errors.reported,
            "Original PhysX heap manager leaked roots or reported corruption");
}
}

int main() {
    try {
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        exercise(context);
        exercise_manager(context);
        std::cout << "Original PhysX source buddy heap GPU lifecycle PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Original PhysX source buddy heap FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
