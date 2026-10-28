// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_sap_dma_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"

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
        companion.deallocate();
        buffer.deallocate();
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
        std::cout << "Original PhysX source buddy heap GPU lifecycle PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Original PhysX source buddy heap FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
