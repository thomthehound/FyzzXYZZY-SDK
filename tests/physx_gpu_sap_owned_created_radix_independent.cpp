// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgSapDmaUpFrameNative.h"
#include "physx_port/lowlevelaabb/include/BpBroadPhaseUpdate.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_translate_aabbs_native.h"
#include "physx_gpu/source_sap_created_native.h"
#include "physx_gpu/source_sap_radix_native.h"
#include "physx_gpu/source_sapbox_indices_native.h"
#include "physx_gpu/source_sap_endpoint_histograms_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
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
             ShaderBinary createdShader, ShaderBinary initShader,
             ShaderBinary countShader, ShaderBinary scatterShader,
             ShaderBinary gatherShader, ShaderBinary indicesShader,
             ShaderBinary firstHistogramShader,
             ShaderBinary secondHistogramShader) {
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
        // Exercise multiple source allocations sharing one physical GPU
        // page; D3D12 must not require separate SRV/UAV resources.
        physx::PxgHeapMemoryAllocatorManager manager(
            1u << 20u, deviceCallback, pinnedRoots, mappedRoots, abort, &aborted);
        auto allocators = physx::PxgCreateAllocatorDesc(manager);
        {
            PxgSapSourceMembersNative sap(allocators, 32u, 16u);
            AabbEncodingStage encoding(context, std::move(translateShader));
            CudaAabbLaunchLowering lowering(addresses, encoding);
            SourceSapCreatedNativeStage createdStage(
                context, std::move(createdShader));
            SourceSapRadixNativeStage radixStage(
                context, std::move(initShader), std::move(countShader),
                std::move(scatterShader), std::move(gatherShader));
            SourceSapBoxIndicesNativeStage indicesStage(
                context, std::move(indicesShader));
            SourceSapEndpointHistogramsNativeStage histogramsStage(
                context, std::move(firstHistogramShader),
                std::move(secondHistogramShader));

            constexpr std::array<physx::Bp::ShapeHandle, 3> created{0u,1u,2u};
            const std::array<physx::PxBounds3, 3> bounds{
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(6.0f, 0.0f, -3.0f),
                    physx::PxVec3(1.0f, 1.0f, 1.0f)),
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(-4.0f, 0.0f, 2.0f),
                    physx::PxVec3(1.0f, 1.0f, 1.0f)),
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(1.0f, 0.0f, -1.0f),
                    physx::PxVec3(2.0f, 1.0f, 1.0f))};
            const std::array<physx::Bp::FilterGroup::Enum, 3> groups{
                physx::Bp::getFilterGroup_Dynamics(0u, false),
                physx::Bp::getFilterGroup_Dynamics(1u, false),
                physx::Bp::getFilterGroup_Dynamics(2u, false)};
            constexpr std::array<physx::PxReal, 3> distances{
                0.125f,0.25f,0.125f};
            const physx::Bp::BpFilter filter(false, false);
            const physx::Bp::BroadPhaseUpdateData update(
                created.data(), 3u, nullptr, 0u, nullptr, 0u,
                bounds.data(), groups.data(), distances.data(),
                3u, filter, true, true);

            pre_broadphase_native(update, sap.pre_broadphase_buffers(), stream);
            auto dma = sap.dma_up_members(nullptr, mappedRoots);
            const auto admission = begin_sap_update_native(
                update, dma, stream, sap.force_update());
            require(admission.execute_gpu_stages &&
                    admission.counts.previous_boxes == 0u &&
                    admission.counts.current_boxes == 3u &&
                    admission.counts.created == 3u &&
                    admission.counts.projections == 6u &&
                    admission.counts.padded_projections == 8u &&
                    sap.force_update() &&
                    stream.commands().empty(),
                    "Original SAP DMA-up did not admit the created frame");

            const auto& descriptor = sap.descriptor();
            require(descriptor.numPreviousHandles==0u &&
                    descriptor.numCreatedHandles==3u &&
                    descriptor.numRemovedHandles==0u,
                    "Original PhysX GPU descriptor did not retain created-only frame roles");
            const auto gpu_address = [](auto ptr) {
                return static_cast<CudaDeviceAddressMap::Address>(
                    reinterpret_cast<std::uintptr_t>(ptr));
            };
            const SourceSapTranslateAabbsNativeInput translation{
                gpu_address(descriptor.updateData_fpBounds),
                gpu_address(descriptor.newIntegerBounds),
                gpu_address(descriptor.updateData_contactDistances),
                0u,
                {3u, 4u, 4u, 4u, 0u, 0u, 0u}
            };
            const auto boundsView = addresses.resolve(
                translation.fp_bounds, 3u*sizeof(physx::PxBounds3));
            const auto distanceView = addresses.resolve(
                translation.contact_distances, 3u*sizeof(physx::PxReal));
            const auto outputView = addresses.resolve(
                translation.new_integer_bounds, 18u * sizeof(std::uint32_t));
            require(boundsView.buffer == outputView.buffer &&
                    distanceView.buffer == outputView.buffer &&
                    boundsView.offset != outputView.offset &&
                    distanceView.offset != outputView.offset,
                    "Source buddy heap did not exercise D3D12 shared-root bindings");

            auto& commands = stream.commands();
            append_source_sap_translate_aabbs_native(
                commands, lowering, translation, true);

            // Original markCreatedPairsKernel consumes the integer bounds
            // written by the preceding translateAABBsKernel, without an
            // intervening CPU download or an independent staging buffer.
            createdStage.append(commands, descriptor, addresses, {3u, 8u});
            require(commands.size() == 2u &&
                    std::get<DispatchCommand>(commands[0]).buffers.size() == 4u &&
                    std::get<DispatchCommand>(commands[1]).groups.x == 32u &&
                    std::get<DispatchCommand>(commands[1]).buffers.size() == 9u,
                    "Original SAP created-frame kernel dispatches changed");

            // The original GPU WOR path consumes these just-created
            // projections and encoded handles through its six current
            // source radix descriptors. All 16 count/scatter dispatches
            // are appended without another allocator, upload or queue.
            std::array<physx::PxgRadixSortDesc,6> radixDescriptors{};
            std::copy_n(dma.radix_wor_descriptors,6u,radixDescriptors.begin());
            for(std::uint32_t axis=0;axis<3u;++axis) {
                require(radixDescriptors[axis].inputKeys ==
                            descriptor.boxProjections[axis] &&
                        radixDescriptors[axis].inputRanks ==
                            descriptor.boxHandles[0][axis],
                        "WOR radix descriptors do not borrow original SAP histories");
            }
            auto invalidRadix=radixDescriptors;
            invalidRadix[5].outputRanks=reinterpret_cast<physx::PxU32*>(
                static_cast<std::uintptr_t>(0x1234u));
            bool rejected=false;
            try {
                radixStage.append_wor(
                    commands,invalidRadix,addresses,0u,3u);
            } catch (const std::exception&) { rejected=true; }
            require(rejected && commands.size()==2u,
                    "Invalid source radix tail partially appended GPU work");
            radixStage.append_wor(
                commands,radixDescriptors,addresses,0u,3u);
            require(commands.size()==18u &&
                    std::get<DispatchCommand>(commands[2]).groups.x==32u &&
                    std::get<DispatchCommand>(commands[2]).groups.y==3u,
                    "Original WOR eight-pass GPU radix dispatch was not appended");

            // After the source WOR sort, PhysX calculates current-frame
            // endpoint histograms before final initializeSapBoxKernel.
            // Do not move the index rebuild ahead of those histories.
            auto invalidHistograms=descriptor;
            invalidHistograms.blockTotalEndPtHistogram[2u]=
                reinterpret_cast<physx::PxU32*>(
                    static_cast<std::uintptr_t>(0x1234u));
            rejected=false;
            try {
                histogramsStage.append(commands,invalidHistograms,
                                       addresses,{8u,3u},false);
            } catch (const std::exception&) { rejected=true; }
            require(rejected && commands.size()==18u,
                    "Invalid endpoint history appended partial GPU work");
            histogramsStage.append(commands,descriptor,addresses,
                                   {8u,3u},false);

            auto invalidIndices=descriptor;
            invalidIndices.boxSapBox1D[2u]=
                reinterpret_cast<physx::PxgSapBox1D*>(
                    static_cast<std::uintptr_t>(0x1234u));
            rejected=false;
            try {
                indicesStage.append(commands,invalidIndices,addresses,
                                    {3u,8u},3u,false);
            } catch (const std::exception&) { rejected=true; }
            require(rejected && commands.size()==20u,
                    "Invalid source SAP box history appended partial work");
            indicesStage.append(commands,descriptor,addresses,{3u,8u},3u,false);
            require(commands.size()==21u &&
                    std::get<DispatchCommand>(commands[18]).buffers.size()==48u &&
                    std::get<DispatchCommand>(commands[19]).buffers.size()==48u &&
                    std::get<DispatchCommand>(commands[20]).buffers.size()==6u &&
                    std::get<DispatchCommand>(commands[19]).groups.x==32u &&
                    std::get<DispatchCommand>(commands[19]).groups.y==3u,
                    "Original source SAP endpoint/index execution shape changed");

            // Original DMA, created-pair, WOR, histogram and SAP index work
            // submits together, retiring only at the source fetch boundary.
            stream.wait_at_fetch();
            require(stream.submission_stats().command_batches==1u,
                    "Source-owned SAP GPU histories split command submissions");
            manager.flushDeferredDeallocs();

            const auto integerBounds = read_words(
                sap.new_integer_bounds().getDevicePtr(), 18u);
            const auto oldBounds = read_words(
                sap.old_integer_bounds().getDevicePtr(), 18u);
            const auto expected_ordinal = [](float value, bool upper) {
                std::uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                const auto ordered = (bits & 0x80000000u)
                    ? ~bits : (bits | 0x80000000u);
                return ((ordered >> 4u) & ~std::uint32_t{1u}) |
                    (upper ? 1u : 0u);
            };
            for (std::size_t handle=0u;handle<created.size();++handle) {
                const auto& b=bounds[handle];
                const std::array<float,6> coordinates{
                    b.minimum.x,b.minimum.y,b.minimum.z,
                    b.maximum.x,b.maximum.y,b.maximum.z};
                for (std::size_t coordinate=0u;coordinate<6u;++coordinate) {
                    const auto upper=coordinate>=3u;
                    const auto expected=expected_ordinal(
                        coordinates[coordinate] +
                            (upper?distances[handle]:-distances[handle]),
                        upper);
                    require(integerBounds[handle*6u+coordinate]==expected &&
                            oldBounds[handle*6u+coordinate]==
                                (upper?0x00800000u:0xff7fffffu),
                        "Original SAP translated/created GPU bound mismatch");
                }
            }

            // Read the ACTUAL persistent PhysX projections and endpoint
            // handles after source WOR GPU sorting. Tied Y endpoints must
            // retain their original stable created-handle order.
            for (std::uint32_t axis=0u;axis<3u;++axis) {
                std::vector<std::pair<std::uint32_t,std::uint32_t>> expected;
                expected.reserve(8u);
                for (std::uint32_t handle=0u;handle<created.size();++handle) {
                    expected.emplace_back(integerBounds[handle*6u+axis],
                                          (handle<<3u)|3u);
                    expected.emplace_back(integerBounds[handle*6u+axis+3u],
                                          (handle<<3u)|2u);
                }
                expected.emplace_back(0xffffffffu,0x3fffffffu);
                expected.emplace_back(0xffffffffu,0x3fffffffu);
                std::stable_sort(expected.begin(),expected.end(),
                    [](const auto& left,const auto& right){
                        return left.first<right.first;
                    });
                const auto projections=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxProjections[axis]),8u);
                const auto handles=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxHandles[0][axis]),8u);
                for(std::size_t slot=0u;slot<expected.size();++slot)
                    require(projections[slot]==expected[slot].first &&
                            handles[slot]==expected[slot].second,
                            "Original SAP GPU WOR radix order or handle history differs");

                std::array<std::uint32_t,6> expectedIndices{};
                std::array<std::uint32_t,6> endPrefix{},startPrefix{};
                std::array<std::uint32_t,3> expectedEnds{},expectedStarts{};
                std::uint32_t endCount=0u,startCount=0u;
                for(std::size_t slot=0u;slot<6u;++slot) {
                    const auto tagged=expected[slot].second;
                    const auto handle=tagged>>3u;
                    const bool start=(tagged&1u)!=0u;
                    expectedIndices[handle*2u+(start?0u:1u)]=
                        static_cast<std::uint32_t>(slot);
                    endPrefix[slot]=endCount;
                    startPrefix[slot]=startCount;
                    if(start)expectedStarts[startCount++]=handle;
                    else expectedEnds[endCount++]=handle;
                }
                require(startCount==3u&&endCount==3u,
                        "Created SAP population did not retain three endpoints of each kind");
                const auto indexHistory=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxSapBox1D[axis]),6u);
                const auto endpoints=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.endPtHistogram[0u][axis]),6u);
                const auto starts=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.startPtHistogram[0u][axis]),6u);
                const auto totalEnds=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.totalEndPtHistogram[axis]),6u);
                // Histories must use the exact original descriptor allocations,
                // not a copied endpoint cache or rearranged test-owned buffer.
                const auto mappedIndices=addresses.resolve(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxSapBox1D[axis]),6u*sizeof(std::uint32_t));
                const auto mappedEnds=addresses.resolve(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.endPtHistogram[0u][axis]),6u*sizeof(std::uint32_t));
                const auto mappedStarts=addresses.resolve(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.startPtHistogram[0u][axis]),6u*sizeof(std::uint32_t));
                require(mappedIndices.buffer==mappedEnds.buffer &&
                        mappedEnds.buffer==mappedStarts.buffer &&
                        mappedIndices.offset!=mappedEnds.offset &&
                        mappedEnds.offset!=mappedStarts.offset,
                        "Original SAP current histories were not aliased to the source heap root");
                const auto orderedEnds=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.endPointHandles[0u][axis]),3u);
                const auto orderedStarts=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.startPointHandles[0u][axis]),3u);
                for(std::size_t slot=0u;slot<6u;++slot) {
                    require(indexHistory[slot]==expectedIndices[slot] &&
                            endpoints[slot]==endPrefix[slot] &&
                            starts[slot]==startPrefix[slot] &&
                            totalEnds[slot]==endPrefix[slot],
                            "Source-owned GPU SAP box indices or endpoint prefix differs");
                }
                for(std::size_t slot=0u;slot<3u;++slot)
                    require(orderedEnds[slot]==expectedEnds[slot] &&
                            orderedStarts[slot]==expectedStarts[slot],
                            "Source-owned endpoint ordered handles differ");
                const auto blockEnd=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.blockEndPtHistogram[0u][axis]),32u);
                const auto blockStart=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.blockStartPtHistogram[0u][axis]),32u);
                const auto blockTotal=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.blockTotalEndPtHistogram[axis]),32u);
                for(std::size_t block=0u;block<32u;++block)
                    require(blockEnd[block]==(block?0u:3u) &&
                            blockStart[block]==(block?0u:3u) &&
                            blockTotal[block]==(block?0u:3u),
                            "Original SAP GPU endpoint block histogram differs");
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
        if (argc != 10)
            throw std::invalid_argument("Expected translated-AABB, created, radix, box indices and endpoint histogram shaders");
        Context context(
#if defined(FYZZXYZZY_TEST_D3D12)
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        execute(context, load_shader(argv[1]), load_shader(argv[2]),
                load_shader(argv[3]), load_shader(argv[4]),
                load_shader(argv[5]), load_shader(argv[6]),
                load_shader(argv[7]), load_shader(argv[8]),
                load_shader(argv[9]));
        std::cout << "Original PhysX SAP native GPU created endpoint histories PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original PhysX SAP native GPU created endpoint histories FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
