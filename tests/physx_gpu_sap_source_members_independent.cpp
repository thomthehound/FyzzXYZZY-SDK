// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_port/gpucommon/include/PxgRadixSortKernelIndices.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct SourceErrors final : physx::PxErrorCallback {
    bool reported = false;
    void reportError(physx::PxErrorCode::Enum, const char* message,
                     const char*, int) override {
        reported = true;
        std::cerr << "PhysX SAP allocator: " << message << '\n';
    }
};

struct Foundation {
    physx::PxDefaultAllocator allocator;
    SourceErrors errors;
    physx::PxFoundation* instance = nullptr;
    Foundation() : instance(PxCreateFoundation(
        PX_PHYSICS_VERSION, allocator, errors)) {
        require(instance != nullptr, "PhysX Foundation initialization failed");
    }
    ~Foundation() { if (instance) instance->release(); }
};

void exercise(Context& context) {
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

    {
        physx::PxgHeapMemoryAllocatorManager manager(
            65536u, deviceCallback, pinnedRoots, mappedRoots,
            abort, &aborted);
        auto allocators = physx::PxgCreateAllocatorDesc(manager);
        require(&allocators.hostAlloc == &allocators.hostMappedAlloc,
                "Original SAP normal host allocations do not alias");
        {
            constexpr physx::PxU32 maxPairs = 32u;
            constexpr physx::PxU32 maxAggregatePairs = 16u;
            PxgSapSourceMembersNative sap(
                allocators, maxPairs, maxAggregatePairs);
            require(sap.force_update() && sap.num_boxes() == 0u &&
                    sap.max_found_lost_pairs() == maxPairs &&
                    sap.mapped_found_actor_pairs() &&
                    sap.mapped_lost_actor_pairs(),
                    "Original persistent SAP member initialization changed");

            auto members = sap.dma_up_members(nullptr, mappedRoots);
            constexpr auto radixBytes = sizeof(physx::PxU32) *
                physx::PxgRadixSortKernelGridDim::RADIX_SORT * 16u;
            for (physx::PxU32 axis = 0u; axis < 3u; ++axis)
                require(members.radix_buffers.radix_counts[axis].getSize() ==
                            radixBytes,
                        "Original SAP persistent radix allocations changed");
            require(
                members.descriptor_gpu_buffers.broadphase.getSize() ==
                    sizeof(physx::PxgBroadPhaseDesc) &&
                members.descriptor_gpu_buffers.radix.getSize() ==
                    6u * sizeof(physx::PxgRadixSortDesc) &&
                members.descriptor_gpu_buffers.radix_without_ranks.getSize() ==
                    6u * sizeof(physx::PxgRadixSortDesc) &&
                members.descriptor_inputs.max_found_lost_pairs == maxPairs &&
                members.descriptor_inputs.max_found_lost_aggregate_pairs ==
                    maxAggregatePairs,
                "Original SAP descriptor ownership/capacity changed");

            const auto reportAddress =
                members.descriptor_inputs.mapped_found_actor_pairs;
            const auto reportView = addresses.resolve(
                reportAddress, sizeof(physx::PxgBroadPhasePair));
            const physx::PxgBroadPhasePair pair(9u, 3u);
            context.upload(reportView.buffer, reportView.offset,
                           &pair, sizeof(pair));
            mappedRoots.publish_at_fetch(
                sap.mapped_found_actor_pairs(), sizeof(pair));
            const auto& fetched = *sap.mapped_found_actor_pairs();
            require(fetched.mVolA == 3u && fetched.mVolB == 9u,
                    "Original SAP mapped pair report lost GPU publication");

            // The persistent owner reuses the same buddy heap for its
            // per-frame GPU buffers, not an independent allocation arena.
            sap.new_integer_bounds().allocate(512u, PX_FL);
            require(sap.new_integer_bounds().getDevicePtr() &&
                    manager.getDeviceHeapStats()
                        .stats[physx::PxsHeapStats::eBROADPHASE] > 0u,
                    "Original SAP new-bounds buffer bypassed source heap");
        }
        require(manager.getDeviceHeapStats()
                    .stats[physx::PxsHeapStats::eBROADPHASE] == 0u,
                "Original SAP member teardown left live buddy blocks");
    }

    deviceRoots.close(&stream);
    require(deviceRoots.live_roots() == 0u &&
            pinnedRoots.live_roots() == 0u &&
            mappedRoots.live_roots() == 0u &&
            addresses.live_allocations() == 0u &&
            !aborted && !foundation.errors.reported,
            "Original SAP member teardown leaked roots or reported an error");
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
        exercise(context);
        std::cout << "Original SAP persistent member ownership PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original SAP persistent member ownership FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
