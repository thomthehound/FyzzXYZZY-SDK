// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_translate_aabbs_native.h"
#include "physx_gpu/source_sap_created_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ShaderBinary load_shader(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Missing original SAP GPU shader");
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    ShaderBinary shader;
#if defined(FYZZXYZZY_TEST_D3D12)
    shader.format = ShaderFormat::Dxil;
#else
    shader.format = ShaderFormat::Spirv;
#endif
    shader.bytes.assign(bytes.begin(), bytes.end());
    return shader;
}

struct SourceErrors final : physx::PxErrorCallback {
    bool reported = false;
    void reportError(physx::PxErrorCode::Enum, const char* message,
                     const char*, int) override {
        reported = true;
        std::cerr << "Original SAP source runtime: " << message << '\n';
    }
};
struct Foundation {
    physx::PxDefaultAllocator allocator;
    SourceErrors errors;
    physx::PxFoundation* instance;
    Foundation() : instance(PxCreateFoundation(
        PX_PHYSICS_VERSION, allocator, errors)) {
        require(instance != nullptr, "Original PhysX Foundation creation failed");
    }
    ~Foundation() { if (instance) instance->release(); }
};

void execute(Context& context, ShaderBinary translateShader,
             ShaderBinary createdShader) {
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
    const auto read_words = [&](CUdeviceptr address, std::size_t count) {
        std::vector<std::uint32_t> result(count);
        const auto view = addresses.resolve(address, count * sizeof(std::uint32_t));
        context.download(view.buffer, view.offset,
                         result.data(), count * sizeof(std::uint32_t));
        return result;
    };
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            65536u, deviceCallback, pinnedRoots, mappedRoots, abort, &aborted);
        auto allocators = physx::PxgCreateAllocatorDesc(manager);
        {
            PxgSapSourceMembersNative sap(allocators, 32u, 16u);
            AabbEncodingStage encoding(context, std::move(translateShader));
            CudaAabbLaunchLowering lowering(addresses, encoding);
            SourceSapCreatedNativeStage createdStage(
                context, std::move(createdShader));

            constexpr std::array<physx::Bp::ShapeHandle, 1> created{0u};
            const std::array<physx::PxBounds3, 1> bounds{
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(0.0f, 0.0f, 0.0f),
                    physx::PxVec3(1.0f, 2.0f, 3.0f))};
            const std::array<physx::Bp::FilterGroup::Enum, 1> groups{
                physx::Bp::getFilterGroup_Dynamics(0u, false)};
            constexpr std::array<physx::PxReal, 1> distances{0.125f};
            const physx::Bp::BpFilter filter(false, false);
            const physx::Bp::BroadPhaseUpdateData update(
                created.data(), 1u, nullptr, 0u, nullptr, 0u,
                bounds.data(), groups.data(), distances.data(),
                1u, filter, true, true);

            pre_broadphase_native(update, sap.pre_broadphase_buffers(), stream);
            auto dma = sap.dma_up_members(nullptr, mappedRoots);
            const auto admission = begin_sap_update_native(
                update, dma, stream, sap.force_update());
            require(admission.execute_gpu_stages &&
                    admission.counts.previous_boxes == 0u &&
                    admission.counts.current_boxes == 1u &&
                    admission.counts.padded_projections == 4u &&
                    sap.force_update() &&
                    stream.commands().empty(),
                    "Original SAP DMA-up did not admit the created frame");

            const auto& descriptor = sap.descriptor();
            const auto gpu_address = [](auto ptr) {
                return static_cast<CudaDeviceAddressMap::Address>(
                    reinterpret_cast<std::uintptr_t>(ptr));
            };
            const SourceSapTranslateAabbsNativeInput translation{
                gpu_address(descriptor.updateData_fpBounds),
                gpu_address(descriptor.newIntegerBounds),
                gpu_address(descriptor.updateData_contactDistances),
                0u,
                {1u, 4u, 4u, 4u, 0u, 0u, 0u}
            };
            auto& commands = stream.commands();
            append_source_sap_translate_aabbs_native(
                commands, lowering, translation, true);

            // Original markCreatedPairsKernel consumes the integer bounds
            // written by the preceding translateAABBsKernel, without an
            // intervening CPU download or an independent staging buffer.
            createdStage.append(commands, descriptor, addresses, {1u, 4u});
            require(commands.size() == 2u &&
                    std::get<DispatchCommand>(commands[0]).buffers.size() == 4u &&
                    std::get<DispatchCommand>(commands[1]).groups.x == 32u &&
                    std::get<DispatchCommand>(commands[1]).buffers.size() == 9u,
                    "Original SAP created-frame kernel dispatches changed");

            // The source stream submits DMA and both GPU stages in order.
            // Its last submission fence establishes one fetch-time boundary.
            stream.wait_at_fetch();
            manager.flushDeferredDeallocs();

            const auto integerBounds = read_words(
                sap.new_integer_bounds().getDevicePtr(), 6u);
            const auto oldBounds = read_words(
                sap.old_integer_bounds().getDevicePtr(), 6u);
            const auto expected_ordinal = [](float value, bool upper) {
                std::uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                const auto ordered = (bits & 0x80000000u)
                    ? ~bits : (bits | 0x80000000u);
                return ((ordered >> 4u) & ~std::uint32_t{1u}) |
                    (upper ? 1u : 0u);
            };
            const std::array<float, 3> extents{1.0f, 2.0f, 3.0f};
            for (std::size_t axis = 0; axis < 3u; ++axis) {
                require(integerBounds[axis] ==
                            expected_ordinal(-extents[axis] - 0.125f, false) &&
                        integerBounds[axis + 3u] ==
                            expected_ordinal(extents[axis] + 0.125f, true) &&
                        oldBounds[axis] == 0xff7fffffu &&
                        oldBounds[axis + 3u] == 0x00800000u,
                        "Original SAP GPU translation/created bounds mismatch");

                const auto projection = read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxProjections[axis]), 4u);
                const auto handles = read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxHandles[0][axis]), 4u);
                require(projection[0] == integerBounds[axis] &&
                        projection[1] == integerBounds[axis + 3u] &&
                        projection[2] == 0xffffffffu &&
                        projection[3] == 0xffffffffu &&
                        handles[0] == 3u && handles[1] == 2u &&
                        handles[2] == 0x3fffffffu &&
                        handles[3] == 0x3fffffffu,
                        "Original SAP GPU created endpoints/rank padding mismatch");
            }

            require(manager.getDeviceHeapStats()
                        .stats[physx::PxsHeapStats::eBROADPHASE] > 0u,
                    "Created frame bypassed the original PhysX GPU buddy heap");
        }
        manager.flushDeferredDeallocs();
        require(manager.getDeviceHeapStats()
                    .stats[physx::PxsHeapStats::eBROADPHASE] == 0u,
                "Source SAP created frame left live buddy blocks");
    }
    deviceRoots.close(&stream);
    require(deviceRoots.live_roots() == 0u &&
            pinnedRoots.live_roots() == 0u &&
            mappedRoots.live_roots() == 0u &&
            addresses.live_allocations() == 0u &&
            !aborted && !foundation.errors.reported,
            "Created SAP GPU frame leaked native resources");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("Expected translated-AABB and created-pairs shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        execute(context, load_shader(argv[1]), load_shader(argv[2]));
        std::cout << "Original PhysX SAP native GPU created-frame PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original PhysX SAP native GPU created-frame FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
