// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
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
        std::cerr << "PhysX SAP DMA-up: " << message << '\n';
    }
};

struct Foundation {
    physx::PxDefaultAllocator allocator;
    SourceErrors errors;
    physx::PxFoundation* instance = nullptr;
    Foundation() : instance(PxCreateFoundation(
        PX_PHYSICS_VERSION, allocator, errors)) {
        require(instance != nullptr, "Original PhysX Foundation initialization failed");
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

    constexpr auto group = physx::PxsHeapStats::eBROADPHASE;
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            65536u, deviceCallback, pinnedRoots, mappedRoots,
            abort, &aborted);
        auto allocators = physx::PxgCreateAllocatorDesc(manager);
        {
            PxgSapSourceMembersNative sap(allocators, 32u, 16u);
            constexpr std::array<physx::Bp::ShapeHandle, 1> created{0u};
            const std::array<physx::PxBounds3, 1> bounds{
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(0.0f, 0.0f, 0.0f),
                    physx::PxVec3(1.0f, 2.0f, 3.0f))};
            const std::array<physx::Bp::FilterGroup::Enum, 1> groups{
                physx::Bp::getFilterGroup_Dynamics(0u, false)};
            constexpr std::array<physx::PxReal, 1> distances{0.125f};
            const physx::Bp::BpFilter filter(false, false);
            const physx::Bp::BroadPhaseUpdateData initial(
                created.data(), 1u, nullptr, 0u, nullptr, 0u,
                bounds.data(), groups.data(), distances.data(),
                1u, filter, true, true);

            pre_broadphase_native(initial, sap.pre_broadphase_buffers(), stream);
            auto frame = begin_sap_update_native(
                initial, sap.dma_up_members(nullptr, mappedRoots),
                stream, sap.force_update());
            require(frame.execute_gpu_stages &&
                    frame.counts.previous_boxes == 0u &&
                    frame.counts.current_boxes == 1u &&
                    frame.counts.created == 1u &&
                    frame.counts.projections == 2u &&
                    sap.num_boxes() == 1u && sap.force_update(),
                    "Original SAP created-frame DMA admission changed");

            // Inspect the real original broadphase descriptor uploaded by
            // gpuDMAUp, not a synthetic host-side descriptor substitute.
            const auto descriptorGpu = sap.dma_up_members(nullptr, mappedRoots)
                .descriptor_gpu_buffers.broadphase.getDevicePtr();
            const auto descriptorView = addresses.resolve(
                descriptorGpu, sizeof(physx::PxgBroadPhaseDesc));
            stream.wait_at_fetch();
            physx::PxgBroadPhaseDesc uploaded{};
            context.download(descriptorView.buffer, descriptorView.offset,
                             &uploaded, sizeof(uploaded));
            require(uploaded.numCreatedHandles == 1u &&
                    uploaded.numRemovedHandles == 0u &&
                    uploaded.numPreviousHandles == 0u &&
                    uploaded.max_found_lost_pairs == 32u &&
                    uploaded.max_found_lost_agg_pairs == 16u &&
                    uploaded.updateData_createdHandles != nullptr &&
                    uploaded.updateData_fpBounds != nullptr,
                    "Original SAP uploaded descriptor lost source fields");

            const auto createdGpu = reinterpret_cast<std::uintptr_t>(
                uploaded.updateData_createdHandles);
            const auto createdView = addresses.resolve(
                createdGpu, sizeof(physx::PxU32));
            physx::PxU32 observedHandle = ~physx::PxU32{0};
            context.download(createdView.buffer, createdView.offset,
                             &observedHandle, sizeof(observedHandle));
            require(observedHandle == created.front(),
                    "Original SAP created handle was not uploaded on source stream");

            const auto boundsGpu = reinterpret_cast<std::uintptr_t>(
                uploaded.updateData_fpBounds);
            const auto boundsView = addresses.resolve(
                boundsGpu, sizeof(physx::PxBounds3));
            physx::PxBounds3 observedBounds;
            context.download(boundsView.buffer, boundsView.offset,
                             &observedBounds, sizeof(observedBounds));
            require(observedBounds.minimum.x == -1.0f &&
                    observedBounds.minimum.y == -2.0f &&
                    observedBounds.maximum.z == 3.0f,
                    "Original SAP bounds were not transferred by preBroadPhase");

            // Source update() runs gpuDMAUp even when only the forced
            // convergence pass is pending, then once more on an idle pass.
            const physx::Bp::BroadPhaseUpdateData quiet(
                nullptr, 0u, nullptr, 0u, nullptr, 0u,
                bounds.data(), groups.data(), distances.data(),
                1u, filter, false, false);
            for (unsigned step = 0; step != 2u; ++step) {
                pre_broadphase_native(
                    quiet, sap.pre_broadphase_buffers(), stream);
                const auto before = stream.submission_stats().upload_batches;
                frame = begin_sap_update_native(
                    quiet, sap.dma_up_members(nullptr, mappedRoots),
                    stream, sap.force_update());
                require(frame.counts.current_boxes == 1u &&
                        frame.counts.created == 0u &&
                        frame.counts.projections == 2u &&
                        frame.execute_gpu_stages == (step == 0u) &&
                        !sap.force_update() &&
                        stream.submission_stats().upload_batches > before,
                        "Original SAP quiet DMA/forced-update ordering changed");
                stream.wait_at_fetch();
                manager.flushDeferredDeallocs();
            }
            physx::PxgBroadPhaseDesc quietDescriptor{};
            context.download(descriptorView.buffer, descriptorView.offset,
                             &quietDescriptor, sizeof(quietDescriptor));
            require(quietDescriptor.numCreatedHandles == 0u &&
                    quietDescriptor.numRemovedHandles == 0u &&
                    quietDescriptor.numPreviousHandles == 1u &&
                    quietDescriptor.max_found_lost_pairs == 32u &&
                    quietDescriptor.updateData_fpBounds ==
                        uploaded.updateData_fpBounds,
                    "Original SAP quiet DMA descriptor lost persistent history");
        }
        manager.flushDeferredDeallocs();
        require(manager.getDeviceHeapStats().stats[group] == 0u,
                "Original SAP DMA frame teardown left live source buddy blocks");
    }
    deviceRoots.close(&stream);
    require(deviceRoots.live_roots() == 0u &&
            pinnedRoots.live_roots() == 0u &&
            mappedRoots.live_roots() == 0u &&
            addresses.live_allocations() == 0u &&
            !aborted && !foundation.errors.reported,
            "Original SAP DMA-up leaked native resources or reported an error");
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
        std::cout << "Original PhysX SAP DMA-up and update admission PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original PhysX SAP DMA-up and update admission FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
