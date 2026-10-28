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
#include "physx_gpu/source_fill_d32_native.h"
#include "physx_gpu/source_region_create_native.h"
#include "physx_gpu/source_region_histograms_native.h"
#include "physx_gpu/source_region_newbounds_chain_native.h"
#include "physx_gpu/source_sap_report_multi_root.h"
#include "physx_gpu/source_sap_fetch_core_native.h"
#include "physx_gpu/source_sap_scene_reports.h"
#include "physx_gpu/source_sap_removed_native.h"
#include "physx_gpu/source_sap_updated_native.h"
#include "physx_gpu/source_sap_incremental_counts_native.h"
#include "physx_gpu/source_sap_incremental_pairs_native.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_gpu/source_sap_aabb_report_retire_native.h"
#include "physx_gpu/source_sap_volume_actor_lifetime_native.h"

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"

#include <algorithm>
#include <array>
#include <cstddef>
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
static_assert(sizeof(physx::Bp::VolumeData)==16u);
static_assert(sizeof(physx::PxgBroadPhasePair)==8u);
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
             ShaderBinary secondHistogramShader,
             ShaderBinary fillShader, ShaderBinary regionCreateShader,
             ShaderBinary regionFirstShader, ShaderBinary regionWaveShader,
             ShaderBinary regionSecondShader,
             ShaderBinary activeWriteoutShader, ShaderBinary activeWaveShader,
             ShaderBinary activePortableShader, ShaderBinary activeOutputShader,
             ShaderBinary overlapWorkShader, ShaderBinary overlapScanFirstShader,
             ShaderBinary overlapScanSecondShader, ShaderBinary foundPairsShader,
             ShaderBinary reportFirstShader, ShaderBinary reportSecondShader,
             ShaderBinary reportCopyShader, ShaderBinary reportWaveFirstShader,
             ShaderBinary reportWaveSecondShader,
             ShaderBinary removedMarkShader,
             ShaderBinary removedProjectionsShader,
             ShaderBinary updatedShader,
             ShaderBinary comparisonFirstShader,
             ShaderBinary comparisonSecondShader,
             ShaderBinary incrementalPairsShader) {
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
            PxgAabbSourceMembersNative aabb(allocators,3u);
            std::array<std::uint32_t,3> sourceElements{17u,3u,9u};
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
            SourceFillD32NativeStage fillStage(
                context, std::move(fillShader));
            SourceRegionCreateNativeStage regionStage(
                context, std::move(regionCreateShader), fillStage);
            SourceRegionHistogramsNativeStage regionHistograms(
                context, std::move(regionFirstShader),
                std::move(regionWaveShader), std::move(regionSecondShader));
            SourceActiveRegionListsNativeStage activeStage(
                context, std::move(activeWriteoutShader),
                std::move(activeWaveShader), std::move(activePortableShader),
                std::move(activeOutputShader));
            SourceOverlapWorkNativeStage overlapWork(
                context, std::move(overlapWorkShader));
            SourceOverlapPrefixNativeStage overlapScan(
                context, std::move(overlapScanFirstShader),
                std::move(overlapScanSecondShader));
            SourceNewBoundsPairsNativeStage foundPairs(
                context, std::move(foundPairsShader));
            SourceNewBoundsSequenceNativeStage newBounds(
                overlapWork, overlapScan, foundPairs);
            SourceRegionNewBoundsNativeStage sourceRegions(
                regionStage, regionHistograms, activeStage, newBounds);
            SourceSapReportNativeStage reportStage(
                context, std::move(reportFirstShader),
                std::move(reportSecondShader), std::move(reportCopyShader),
                std::move(reportWaveFirstShader),
                std::move(reportWaveSecondShader));
            SourceSapRemovedNativeStage removedStage(
                context, std::move(removedMarkShader),
                std::move(removedProjectionsShader));
            SourceSapUpdatedNativeStage updatedStage(
                context,std::move(updatedShader));
            SourceSapIncrementalCountsNativeStage comparisonsStage(
                context,std::move(comparisonFirstShader),
                std::move(comparisonSecondShader));
            SourceSapIncrementalPairsNativeStage incrementalStage(
                context,std::move(incrementalPairsShader));

            constexpr std::array<physx::Bp::ShapeHandle, 3> created{0u,1u,2u};
            const std::array<physx::PxBounds3, 3> bounds{
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(0.0f, 0.0f, 0.0f),
                    physx::PxVec3(2.0f, 1.0f, 2.0f)),
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(1.0f, 0.0f, 0.5f),
                    physx::PxVec3(1.0f, 1.0f, 1.5f)),
                physx::PxBounds3::centerExtents(
                    physx::PxVec3(8.0f, 0.0f, -5.0f),
                    physx::PxVec3(1.0f, 1.0f, 1.0f))};
            const std::array<physx::Bp::FilterGroup::Enum, 3> groups{
                physx::Bp::getFilterGroup_Dynamics(0u, false),
                physx::Bp::getFilterGroup_Dynamics(1u, false),
                physx::Bp::getFilterGroup_Dynamics(2u, false)};
            constexpr std::array<physx::PxReal, 3> distances{
                0.125f,0.125f,0.125f};
            const physx::Bp::BpFilter filter(false, false);
            const std::array<fyzz::native::GpuBroadphaseBox,3> liveBoxes{{
                {17u,{0.0f,0.0f,0.0f},{2.0f,1.0f,2.0f},{},0.125f,
                 fyzz::native::GpuBroadphaseBox::Motion::Dynamic},
                {3u,{1.0f,0.0f,0.5f},{1.0f,1.0f,1.5f},{},0.125f,
                 fyzz::native::GpuBroadphaseBox::Motion::Dynamic},
                {9u,{8.0f,0.0f,-5.0f},{1.0f,1.0f,1.0f},{},0.125f,
                 fyzz::native::GpuBroadphaseBox::Motion::Dynamic}
            }};
            const std::array<fyzz::BodyId,3> actorIds{17u,3u,9u};
            for(std::uint32_t i=0u;i<3u;++i)
                aabb.admit(i,actorIds[i],distances[i],groups[i],
                           &sourceElements[i]);
            const auto scActors=aabb.volume_to_actor();
            const auto update=aabb.prepare(liveBoxes,created,filter);
            const auto sceneBounds=aabb.scene_bounds();
            require(aabb.capacity()==4u &&
                    sceneBounds.bounds.data()==aabb.source_bounds().begin() &&
                    sceneBounds.groups.size()==aabb.capacity() &&
                    sceneBounds.contact_distances.size()==aabb.capacity(),
                    "Original AABB source owner changed bounds capacity or aliases");
            const auto aabbPre=aabb.pre_update(sap.pre_broadphase_buffers());
            const auto actorOwner=aabb.actor_lifetime();
            validate_source_sap_owned_aabb_aliases_native(
                aabb,aabbPre,actorOwner.volume_to_actor);
            std::vector<fyzz::BodyId> foreignActors(
                actorOwner.volume_to_actor.begin(),
                actorOwner.volume_to_actor.end());
            bool rejectedForeignOwner=false;
            try {
                validate_source_sap_owned_aabb_aliases_native(
                    aabb,aabbPre,foreignActors);
            }catch(const std::invalid_argument&) {
                rejectedForeignOwner=true;
            }
            require(rejectedForeignOwner,
                    "Original source scene frame admitted a copied actor registry");
            pre_update_source_sap_aabb_manager_native(update,aabbPre,stream);
            const auto aabbView=source_sap_aabb_descriptor_native(aabbPre);
            auto dma=sap.dma_up_members(&aabbView,mappedRoots);
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

            // Run the original region, active-list and found-pair sequence
            // against the same source-owned three-axis histories. Every
            // kernel appends to the one existing mStream command batch.
            const auto gpuDescriptor=
                dma.descriptor_gpu_buffers.broadphase.getDevicePtr();
            const SourceRegionNewBoundsNativeLayout regionLayout{
                {3u,8u},
                {gpuDescriptor,8u},
                {gpuDescriptor,3u,8u,6u*64u},
                {gpuDescriptor,3u,3u,6u,3u*64u}
            };
            auto invalidRegion=descriptor;
            invalidRegion.foundPairReport=nullptr;
            bool rejectedChain=false;
            try {
                sourceRegions.append(commands,invalidRegion,addresses,regionLayout);
            } catch(const std::exception&) { rejectedChain=true; }
            require(rejectedChain&&commands.size()==21u,
                    "Invalid source found-pair report appended partial GPU work");
            sourceRegions.append(commands,descriptor,addresses,regionLayout);
            require(commands.size()==35u &&
                    std::get<DispatchCommand>(commands[23]).buffers.size()==10u &&
                    std::get<DispatchCommand>(commands[27]).buffers.size()==3u &&
                    std::get<DispatchCommand>(commands[28]).buffers.size()==6u &&
                    std::get<DispatchCommand>(commands[29]).groups.x==64u &&
                    std::get<DispatchCommand>(commands[30]).buffers.size()==9u &&
                    std::get<DispatchCommand>(commands[31]).buffers.size()==9u &&
                    std::get<DispatchCommand>(commands[34]).groups.x==64u &&
                    std::get<DispatchCommand>(commands[34]).buffers.size()==16u,
                    "Original source region/active/pair GPU dispatch order changed");

            // Original hostMappedAlloc destinations are the only allowed
            // GPU report aliases. They are not test-created staging buffers.
            const auto mappedBytes=
                sap.max_found_lost_pairs()*sizeof(physx::PxgBroadPhasePair);
            require(mappedRoots.device_address(
                        sap.mapped_found_actor_pairs(),mappedBytes)==
                        static_cast<CudaDeviceAddressMap::Address>(
                            reinterpret_cast<std::uintptr_t>(
                                descriptor.foundPairReportMap)) &&
                    mappedRoots.device_address(
                        sap.mapped_lost_actor_pairs(),mappedBytes)==
                        static_cast<CudaDeviceAddressMap::Address>(
                            reinterpret_cast<std::uintptr_t>(
                                descriptor.lostPairReportMap)),
                    "Original PhysX source report DMA host aliases changed");

            // The original three report kernels classify raw GPU pairs,
            // copy actor pairs into PhysX host-mapped destination allocations,
            // and publish both count families in the resident source desc.
            auto malformedReport=descriptor;
            malformedReport.lostPairReportMap=nullptr;
            rejected=false;
            try {
                reportStage.append(commands,malformedReport,addresses,
                                   {gpuDescriptor,aabb.capacity(),
                                    sizeof(physx::Bp::VolumeData),8u});
            } catch(const std::exception&) { rejected=true; }
            require(rejected && commands.size()==35u,
                    "Invalid original report destination appended partial work");
            reportStage.append(commands,descriptor,addresses,
                               {gpuDescriptor,aabb.capacity(),
                                sizeof(physx::Bp::VolumeData),8u});
            require(commands.size()==38u &&
                    std::get<DispatchCommand>(commands[35]).buffers.size()==8u &&
                    std::get<DispatchCommand>(commands[36]).buffers.size()==13u &&
                    std::get<DispatchCommand>(commands[37]).buffers.size()==7u,
                    "Original PhysX source GPU report classification launches changed");

            // Original update() exchanges integer AABB owners AFTER
            // pair generation but BEFORE fetch/history rotation. The GPU
            // commands already borrowed their original source addresses.
            swap_sap_integer_bounds_after_pairs_native(
                sap.new_integer_bounds(),sap.old_integer_bounds());

            // No download or cross-queue wait until original source fetch.
            stream.wait_at_fetch();
            require(stream.submission_stats().command_batches==1u,
                    "Source-owned SAP GPU histories split command submissions");
            manager.flushDeferredDeallocs();

            const auto integerBounds = read_words(
                sap.old_integer_bounds().getDevicePtr(), 18u);
            const auto oldBounds = read_words(
                sap.new_integer_bounds().getDevicePtr(), 18u);
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

            // Independently reconstruct the region-range and exclusive
            // prefix contracts from verified sorted endpoint histories.
            constexpr std::uint32_t currentBoxes=3u;
            constexpr std::uint32_t projections=6u;
            std::array<std::array<std::uint32_t,3>,3> first{},last{};
            for(std::uint32_t axis=0;axis<3u;++axis) {
                const auto sapIndices=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.boxSapBox1D[axis]),currentBoxes*2u);
                const auto totalEnds=read_words(
                    reinterpret_cast<CUdeviceptr>(
                        descriptor.totalEndPtHistogram[axis]),projections);
                const auto perRegion=(currentBoxes+3u)/4u;
                for(std::uint32_t h=0;h<currentBoxes;++h) {
                    require(sapIndices[h*2u]<projections &&
                            sapIndices[h*2u+1u]<projections,
                            "Original SAP region range references invalid endpoint");
                    first[h][axis]=totalEnds[sapIndices[h*2u]]/perRegion;
                    last[h][axis]=totalEnds[sapIndices[h*2u+1u]]/perRegion;
                }
            }
            const auto regionRanges=read_words(
                reinterpret_cast<CUdeviceptr>(descriptor.regionRange),
                currentBoxes*8u);
            std::array<std::uint32_t,projections> startContribution{},
                regionContribution{};
            const auto sortedX=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.boxHandles[0][0]),projections);
            const auto xIndices=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.boxSapBox1D[0]),currentBoxes*2u);
            for(std::uint32_t h=0;h<currentBoxes;++h) {
                const auto base=h*8u;
                const auto yz=(last[h][1]-first[h][1]+1u)*
                              (last[h][2]-first[h][2]+1u);
                for(std::uint32_t axis=0;axis<3u;++axis)
                    require(regionRanges[base+axis]==first[h][axis] &&
                            regionRanges[base+4u+axis]==last[h][axis],
                            "Original PhysX GPU region spatial bounds differ");
                require(regionRanges[base+3u]==0u &&
                        regionRanges[base+7u]==0u,
                        "Original region buffer padding was overwritten");
                startContribution[xIndices[h*2u]]=yz;
                regionContribution[xIndices[h*2u]]=
                    (last[h][0]-first[h][0]+1u)*yz;
            }
            std::array<std::uint32_t,projections> expectedStarts{},expectedRegions{};
            std::uint32_t startTotal=0u,regionTotal=0u;
            for(std::uint32_t i=0u;i<projections;++i) {
                require((sortedX[i]>>3u)<currentBoxes,
                        "Region producer consumed non-source endpoint");
                expectedStarts[i]=startTotal;
                expectedRegions[i]=regionTotal;
                startTotal+=startContribution[i];
                regionTotal+=regionContribution[i];
            }
            const auto observedStarts=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.startRegionAccum),projections);
            const auto observedRegions=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.regionAccum),projections);
            for(std::uint32_t i=0u;i<projections;++i)
                require(observedStarts[i]==expectedStarts[i] &&
                        observedRegions[i]==expectedRegions[i],
                        "Source region GPU prefix did not consume live created-frame bounds");
            const auto blockStarts=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.blockStartRegionAccum),32u);
            const auto blockRegions=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.blockRegionAccum),32u);
            for(std::uint32_t block=0u;block<32u;++block)
                require(blockStarts[block]==(block?0u:startTotal) &&
                        blockRegions[block]==(block?0u:regionTotal),
                        "Original source region GPU block totals differ");
            const auto descView=addresses.resolve(
                gpuDescriptor,sizeof(physx::PxgBroadPhaseDesc));
            physx::PxgBroadPhaseDesc residentDescriptor{};
            context.download(descView.buffer,descView.offset,
                             &residentDescriptor,sizeof(residentDescriptor));
            require(residentDescriptor.startRegionAccumTotal==startTotal &&
                    residentDescriptor.regionAccumTotal==regionTotal &&
                    regionTotal>=startTotal &&
                    startTotal!=0u,
                    "Original source GPU descriptor did not receive region totals");

            // The active-region lists are derived from the original GPU
            // sorted X handles and spatial ranges. Validate the complete
            // per-region prefix tables before consuming the GPU pair report.
            const auto activePrefix=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.activeRegionsHistogram),64u*projections);
            const auto startPrefix=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.startRegionsHistogram),64u*projections);
            const auto activeOrder=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.orderedActiveRegionHandles),64u*projections);
            const auto startOrder=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.orderedStartRegionHandles),64u*projections);
            for(std::uint32_t region=0u;region<64u;++region) {
                const auto rx=region%4u;
                const auto ry=(region/4u)%4u;
                const auto rz=region/16u;
                std::uint32_t running=0u;
                for(std::uint32_t slot=0u;slot<projections;++slot) {
                    const auto handle=sortedX[slot]>>3u;
                    const bool start=(sortedX[slot]&1u)!=0u;
                    require(startPrefix[region*projections+slot]==running &&
                            activePrefix[region*projections+slot]==running,
                            "Original source active-region GPU prefix mismatch");
                    if(start && first[handle][0]==rx &&
                       first[handle][1]<=ry && ry<=last[handle][1] &&
                       first[handle][2]<=rz && rz<=last[handle][2]) {
                        require(startOrder[region*projections+running]==handle &&
                                activeOrder[region*projections+running]==handle,
                                "Original source GPU ordered active/start handles differ");
                        ++running;
                    }
                }
            }

            // This created-only scene has one physical overlap: the first
            // two volumes. The distant third volume must not be reported.
            // Check the GPU-produced found-pair counter and exact report,
            // not a separate CPU broadphase or a synthetic pair cache.
            require(residentDescriptor.sharedFoundPairIndex==1u &&
                    residentDescriptor.overlapChecksTotalRegion>0u,
                    "Original GPU new-bounds chain did not produce one found pair");
            const auto reports=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.foundPairReport),2u);
            require(reports[0]==0u&&reports[1]==1u,
                    "Original GPU found-pair report disagrees with source AABBs");
            const auto overlapCounts=read_words(
                reinterpret_cast<CUdeviceptr>(
                    descriptor.overlapChecksRegion),3u*64u*2u);
            std::uint64_t previousOverlap=0u;
            require(residentDescriptor.regionAccumTotal<=3u*64u,
                    "Source overlap work exceeds original region allocation");
            for(std::uint32_t slot=0u;
                slot<residentDescriptor.regionAccumTotal;++slot) {
                const auto lo=overlapCounts[slot*2u];
                const auto hi=overlapCounts[slot*2u+1u];
                const auto prefix=(std::uint64_t(hi)<<32u)|lo;
                require(prefix>=previousOverlap,
                        "Original source GPU overlap work prefix is not monotonic");
                previousOverlap=prefix;
            }
            require(residentDescriptor.overlapChecksTotalRegion>=previousOverlap,
                    "Original source GPU overlap count exceeds resident total");

            // Both actor streams and the seven endpoint histories are owned
            // by the real PhysX SAP object, not a test-side staging layout.
            const auto histories=sap.fetch_histories();
            const std::array<physx::PxgCudaBufferN<6>*,7> historyFamilies{
                &histories.projection_handles,
                &histories.block_end_histograms,
                &histories.end_histograms,
                &histories.end_handles,
                &histories.block_start_histograms,
                &histories.start_histograms,
                &histories.start_handles
            };
            std::array<std::array<CUdeviceptr,2>,7> historyAddresses{};
            for(std::size_t i=0;i<historyFamilies.size();++i) {
                historyAddresses[i][0]=(*historyFamilies[i])[0].getDevicePtr();
                historyAddresses[i][1]=(*historyFamilies[i])[3].getDevicePtr();
            }
            const auto histories_match=[&](bool swapped) {
                for(std::size_t i=0;i<historyFamilies.size();++i) {
                    if((*historyFamilies[i])[0].getDevicePtr()!=
                            historyAddresses[i][swapped?1u:0u] ||
                       (*historyFamilies[i])[3].getDevicePtr()!=
                            historyAddresses[i][swapped?0u:1u])
                        return false;
                }
                return true;
            };
            physx::PxgBroadPhaseDesc fetchedDescriptor{};
            std::vector<fyzz::native::Pair> scFound,scLost;
            bool rotated=false,published=false;
            auto publish=[&](const SourceSapFetchedCounts& counts) {
                require(!counts.overflow &&
                        counts.found_actors==1u &&
                        counts.lost_actors==0u,
                        "PhysX GPU classified report counts changed before Sc fetch");
                lower_source_sap_actor_reports_to_sc(
                    {sap.mapped_found_actor_pairs(),counts.found_actors},
                    {sap.mapped_lost_actor_pairs(),counts.lost_actors},
                    scActors,scFound,scLost);
                published=true;
            };
            bool rejectedFetch=false;
            try {
                (void)fetch_source_sap_results_native(
                    context,addresses,gpuDescriptor,sap.descriptor(),
                    {sap.mapped_found_actor_pairs(),0u},
                    {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
                    false,[&]() noexcept {rotated=true;},
                    &mappedRoots,publish);
            } catch(const std::out_of_range&) { rejectedFetch=true; }
            require(rejectedFetch && !rotated && !published &&
                    histories_match(false),
                    "Rejected GPU pair fetch advanced original PhysX history");

            // Sc may reject the classified report after GPU completion.
            // The original fetch must leave both descriptor publication
            // and all seven double-buffer history families untouched.
            bool rejectedAdmission=false;
            try {
                (void)fetch_source_sap_results_native(
                    context,addresses,gpuDescriptor,sap.descriptor(),
                    {sap.mapped_found_actor_pairs(),sap.max_found_lost_pairs()},
                    {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
                    false,[&]() noexcept {rotated=true;},
                    &mappedRoots,[&](const SourceSapFetchedCounts&) {
                        throw std::invalid_argument(
                            "Rejected Sc actor identity at source fetch");
                    });
            }catch(const std::invalid_argument&){rejectedAdmission=true;}
            require(rejectedAdmission&&!rotated&&!published&&
                    histories_match(false)&&
                    sap.descriptor().sharedFoundPairIndex==0u,
                    "Rejected Sc admission committed original source fetch");

            const auto fetched=fetch_source_sap_results_native(
                context,addresses,gpuDescriptor,sap.descriptor(),
                {sap.mapped_found_actor_pairs(),sap.max_found_lost_pairs()},
                {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
                false,[&histories,&rotated]() noexcept {
                    swap_sap_endpoint_histories_after_fetch_native(histories);
                    rotated=true;
                },&mappedRoots,publish);
            fetchedDescriptor=sap.descriptor();
            require(published&&rotated&&fetched.found_actors==1u &&
                    fetched.lost_actors==0u&&fetched.pairs_needed==1u &&
                    fetchedDescriptor.sharedFoundPairIndex==1u &&
                    fetchedDescriptor.sharedFoundAggPairIndex==0u &&
                    fetchedDescriptor.sharedLostPairIndex==0u &&
                    !fetched.overflow &&
                    scFound==std::vector<fyzz::native::Pair>{{3u,17u}} &&
                    scLost.empty() && histories_match(true),
                    "GPU classified reports did not publish through PhysX mapped fetch and rotate histories");

            // Source AABB report processing retires the original added
            // and removed GPU/host bitmaps only AFTER Sc has consumed the
            // first frame. The source stream preserves their GPU ordering
            // into the subsequent motion update.
            retire_source_sap_aabb_report_bitmaps_native(
                stream,fillStage,addresses,
                aabb.added_report_bitmap(),aabb.removed_report_bitmap());
            aabb.complete_fetch(update);

            // Frame two separates two still-live volumes. The source
            // changed bitmap and persistent VolumeData remain original
            // PhysX member types; no independent GPU pair registry.
            auto movedBoxes=liveBoxes;
            movedBoxes[1].position.x=20.0f;
            aabb.mark_changed(1u,distances[1]);
            const auto moved=aabb.prepare(movedBoxes,created,filter);
            const auto nextPre=aabb.pre_update(sap.pre_broadphase_buffers());
            validate_source_sap_owned_aabb_aliases_native(
                aabb,nextPre,aabb.actor_lifetime().volume_to_actor);
            pre_update_source_sap_aabb_manager_native(moved,nextPre,stream);
            const auto nextAabbView=source_sap_aabb_descriptor_native(nextPre);
            auto nextDma=sap.dma_up_members(&nextAabbView,mappedRoots);
            const auto admitted=begin_sap_update_native(
                moved,nextDma,stream,sap.force_update());
            require(admitted.execute_gpu_stages&&
                    admitted.counts.previous_boxes==3u&&
                    admitted.counts.current_boxes==3u&&
                    admitted.counts.created==0u&&
                    admitted.counts.removed==0u,
                    "Second PhysX update did not preserve the three live shapes");
            const auto& nextDesc=sap.descriptor();
            const auto nextGpu=nextDma.descriptor_gpu_buffers.broadphase.getDevicePtr();
            require(nextDesc.aabbMngr_changedHandleBitMapWordCounts==1u&&
                    gpu_address(nextDesc.aabbMngr_changedHandleMap)==
                        nextAabbView.changed_handles&&
                    gpu_address(nextDesc.aabbMngr_volumeData)==
                        nextAabbView.volume_data&&
                    nextDesc.boxHandles[1][0]==
                        reinterpret_cast<physx::PxU32*>(
                            historyAddresses[0][0]),
                    "Motion frame lost original changed-bit or fetched handle ownership");

            auto& pending=stream.commands();
            append_source_sap_translate_aabbs_native(
                pending,lowering,{
                    gpu_address(nextDesc.updateData_fpBounds),
                    gpu_address(nextDesc.newIntegerBounds),
                    gpu_address(nextDesc.updateData_contactDistances),0u,
                    {3u,4u,4u,4u,0u,0u,0u}},true);
            updatedStage.append(pending,nextDesc,addresses,
                                {admitted.counts.capacity,
                                 admitted.counts.padded_projections});
            radixStage.append_wr(
                pending,nextDesc,
                std::span<const physx::PxgRadixSortDesc,6>(
                    nextDma.radix_descriptors,6u),
                addresses,admitted.counts.previous_boxes);
            histogramsStage.append(
                pending,nextDesc,addresses,
                {admitted.counts.padded_projections,3u},true);
            comparisonsStage.append(
                pending,nextDesc,nextGpu,addresses,
                {admitted.counts.padded_projections});
            incrementalStage.append(
                pending,nextDesc,nextGpu,addresses,
                {admitted.counts.capacity,admitted.counts.padded_projections,
                 admitted.counts.previous_boxes,
                 sap.max_found_lost_pairs()});
            indicesStage.append(pending,nextDesc,addresses,
                                {admitted.counts.capacity,
                                 admitted.counts.padded_projections},3u,false);
            reportStage.append(pending,nextDesc,addresses,
                               {nextGpu,aabb.capacity(),
                                sizeof(physx::Bp::VolumeData),8u});
            require(pending.size()>24u,
                    "Source GPU moving frame lost WR/incremental/report stages");
            swap_sap_integer_bounds_after_pairs_native(
                sap.new_integer_bounds(),sap.old_integer_bounds());
            stream.wait_at_fetch();

            const auto secondGpu=read_words(
                reinterpret_cast<CUdeviceptr>(nextDesc.lostPairReport),2u);
            require(secondGpu[0]==0u&&secondGpu[1]==1u,
                    "Original GPU incremental SAP did not report the separated pair");

            bool secondPublished=false;
            const auto lostCounts=fetch_source_sap_results_native(
                context,addresses,nextGpu,sap.descriptor(),
                {sap.mapped_found_actor_pairs(),sap.max_found_lost_pairs()},
                {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
                false,[&histories]() noexcept {
                    swap_sap_endpoint_histories_after_fetch_native(histories);
                },&mappedRoots,[&](const SourceSapFetchedCounts& counts) {
                    require(!counts.overflow&&counts.found_actors==0u&&
                            counts.lost_actors==1u,
                            "GPU motion did not classify one original lost actor pair");
                    lower_source_sap_actor_reports_to_sc(
                        {sap.mapped_found_actor_pairs(),counts.found_actors},
                        {sap.mapped_lost_actor_pairs(),counts.lost_actors},
                        scActors,scFound,scLost);
                    secondPublished=true;
                });
            require(secondPublished&&lostCounts.found_actors==0u&&
                    lostCounts.lost_actors==1u&&
                    sap.descriptor().sharedLostPairIndex==1u&&
                    sap.descriptor().sharedLostAggPairIndex==0u&&
                    scFound.empty()&&
                    scLost==std::vector<fyzz::native::Pair>{{3u,17u}}&&
                    histories_match(false),
                    "GPU separated pair was not published through original Sc fetch");
            retire_source_sap_aabb_report_bitmaps_native(
                stream,fillStage,addresses,
                aabb.added_report_bitmap(),aabb.removed_report_bitmap());
            aabb.complete_fetch(moved);
            stream.wait_at_fetch();
            require(aabb.actor_lifetime().volume_data[1].isSingleActor()&&
                    aabb.actor_lifetime().volume_data[1].getUserData()==&sourceElements[1]&&
                    scActors[1u]==3u,
                    "GPU separation retired a live source Sc identity");

            // Frame three deletes a shape that was live but separated
            // in frame two. Original PxgAABBManager removes VolumeData
            // immediately, yet the Sc identity survives through GPU
            // fetch and source report processing.
            const std::array<physx::Bp::ShapeHandle,1> retiredHandles{1u};
            require(!aabb.remove(1u),
                    "Published source volume incorrectly canceled its removal");
            const auto lifetimes=aabb.actor_lifetime();
            bool rejectedReuse=false;
            try {
                admit_source_sap_volume_actor_identity_native(
                    lifetimes,retiredHandles[0],27u);
            }catch(const std::logic_error&){rejectedReuse=true;}
            require(rejectedReuse&&scActors[1]==3u,
                    "Removed Sc identity recycled before source GPU fetch");
            const std::array<fyzz::native::GpuBroadphaseBox,2> remainingBoxes{
                movedBoxes[0],movedBoxes[2]};
            const std::array<physx::Bp::ShapeHandle,2> remainingHandles{0u,2u};
            const auto retirement=aabb.prepare(remainingBoxes,remainingHandles,filter);
            const auto retirePre=aabb.pre_update(sap.pre_broadphase_buffers());
            validate_source_sap_owned_aabb_aliases_native(
                aabb,retirePre,aabb.actor_lifetime().volume_to_actor);
            pre_update_source_sap_aabb_manager_native(
                retirement,retirePre,stream);
            const auto retiredAabbView=source_sap_aabb_descriptor_native(retirePre);
            auto retireDma=sap.dma_up_members(
                &retiredAabbView,mappedRoots);
            const auto admittedRemoval=begin_sap_update_native(
                retirement,retireDma,stream,sap.force_update());
            require(admittedRemoval.execute_gpu_stages&&
                    admittedRemoval.counts.previous_boxes==3u&&
                    admittedRemoval.counts.current_boxes==2u&&
                    admittedRemoval.counts.created==0u&&
                    admittedRemoval.counts.removed==1u,
                    "Source volume removal did not retain prior SAP frame counts");
            const auto& removalDesc=sap.descriptor();
            const auto removalGpu=
                retireDma.descriptor_gpu_buffers.broadphase.getDevicePtr();
            auto& removePending=stream.commands();
            append_source_sap_translate_aabbs_native(
                removePending,lowering,{
                    gpu_address(removalDesc.updateData_fpBounds),
                    gpu_address(removalDesc.newIntegerBounds),
                    gpu_address(removalDesc.updateData_contactDistances),0u,
                    {3u,4u,4u,4u,0u,0u,0u}},true);
            removedStage.append_mark_removed(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.capacity,
                 admittedRemoval.counts.padded_projections});
            radixStage.append_wr(
                removePending,removalDesc,
                std::span<const physx::PxgRadixSortDesc,6>(
                    retireDma.radix_descriptors,6u),
                addresses,3u);
            histogramsStage.append(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.padded_projections,3u},true);
            indicesStage.append(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.capacity,
                 admittedRemoval.counts.padded_projections},3u,false);
            removedStage.append_remove_projections(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.capacity,
                 admittedRemoval.counts.padded_projections});
            radixStage.append_wor(
                removePending,
                std::span<const physx::PxgRadixSortDesc,6>(
                    retireDma.radix_wor_descriptors,6u),
                addresses,3u,0u);
            histogramsStage.append(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.padded_projections,3u},false);
            indicesStage.append(
                removePending,removalDesc,addresses,
                {admittedRemoval.counts.capacity,
                 admittedRemoval.counts.padded_projections},2u,false);
            reportStage.append(removePending,removalDesc,addresses,
                               {removalGpu,static_cast<std::uint32_t>(
                                   aabb.capacity()),
                                sizeof(physx::Bp::VolumeData),8u});
            swap_sap_integer_bounds_after_pairs_native(
                sap.new_integer_bounds(),sap.old_integer_bounds());
            stream.wait_at_fetch();
            bool removalPublished=false;
            const auto removalCounts=fetch_source_sap_results_native(
                context,addresses,removalGpu,sap.descriptor(),
                {sap.mapped_found_actor_pairs(),sap.max_found_lost_pairs()},
                {sap.mapped_lost_actor_pairs(),sap.max_found_lost_pairs()},
                false,[&histories]() noexcept {
                    swap_sap_endpoint_histories_after_fetch_native(histories);
                },&mappedRoots,[&](const SourceSapFetchedCounts& counts) {
                    require(!counts.overflow&&counts.found_actors==0u&&
                            counts.lost_actors==0u&&scActors[1]==3u,
                            "Removed source volume fabricated a lost pair or retired Sc identity early");
                    lower_source_sap_actor_reports_to_sc(
                        {sap.mapped_found_actor_pairs(),counts.found_actors},
                        {sap.mapped_lost_actor_pairs(),counts.lost_actors},
                        scActors,scFound,scLost);
                    removalPublished=true;
                });
            require(removalPublished&&removalCounts.found_actors==0u&&
                    removalCounts.lost_actors==0u&&
                    scFound.empty()&&scLost.empty()&&
                    histories_match(true)&&scActors[1]==3u,
                    "Source removed frame did not publish and rotate original histories");
            retire_source_sap_aabb_report_bitmaps_native(
                stream,fillStage,addresses,
                aabb.added_report_bitmap(),aabb.removed_report_bitmap());
            aabb.complete_fetch(retirement);
            stream.wait_at_fetch();
            require(scActors[1]==std::numeric_limits<fyzz::BodyId>::max()&&
                    scActors[0]==17u&&scActors[2]==9u,
                    "Original AABB source did not retire exactly the removed Sc identity");
            aabb.admit(1u,27u,distances[1],groups[1],
                       &sourceElements[1]);
            require(scActors[1]==27u,
                    "Original volume handle remained unavailable after GPU fetch");

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
        if (argc != 34)
            throw std::invalid_argument("Expected source AABB, created, radix, indices, endpoint and region shader paths");
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
                load_shader(argv[9]), load_shader(argv[10]),
                load_shader(argv[11]), load_shader(argv[12]),
                load_shader(argv[13]), load_shader(argv[14]),
                load_shader(argv[15]), load_shader(argv[16]),
                load_shader(argv[17]), load_shader(argv[18]),
                load_shader(argv[19]), load_shader(argv[20]),
                load_shader(argv[21]), load_shader(argv[22]),
                load_shader(argv[23]), load_shader(argv[24]),
                load_shader(argv[25]), load_shader(argv[26]),
                load_shader(argv[27]), load_shader(argv[28]),
                load_shader(argv[29]), load_shader(argv[30]),
                load_shader(argv[31]), load_shader(argv[32]),
                load_shader(argv[33]));
        std::cout << "Original PhysX SAP native GPU AABB owner three-frame lifecycle PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Original PhysX SAP native GPU AABB owner three-frame lifecycle FAILED: "
                  << e.what() << '\n';
        return 1;
    }
}
