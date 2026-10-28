// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_physx_heap_roots.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_physx_mapped_report_roots.h"
#include "physx_gpu/source_sap_fetch_native.h"
#include "physx_gpu/source_sap_scene_reports.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include "physx_port/gpucommon/PxgCudaBuffer.h"
#include "physx_port/gpubroadphase/PxgSapPreBroadPhaseNative.h"
#include "physx_port/gpubroadphase/PxgSapSourceMembersNative.h"
#include "physx_port/gpubroadphase/PxgAabbManagerPreUpdateNative.h"
#include "physx_port/gpubroadphase/PxgAabbManagerVolumeLifecycleNative.h"
#include "physx_port/gpubroadphase/PxgAabbSourceMembersNative.h"
#include "physx_gpu/source_sc_element_lifecycle_native.h"
#include "physx_gpu/source_sap_single_volume_scene_native.h"
#include "physx_gpu/source_sap_scene_frame_native.h"
#include "physx_port/gpubroadphase/PxgSapGpuDMAUpHandlesNative.h"
#include "physx_port/gpubroadphase/PxgSapHistoryGrowthNative.h"
#include "physx_port/gpubroadphase/PxgSapPostAxisBuffersNative.h"
#include "physx_port/gpubroadphase/PxgSapRadixDescriptorsNative.h"
#include "physx_port/gpubroadphase/PxgSapUpdateFetchNative.h"
#include "physx_port/gpucommon/include/PxgRadixSortKernelIndices.h"
#include "BpBroadPhaseUpdate.h"
#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using fyzz::physx_gpu::CudaDeviceAddressMap;

namespace {
// PhysX source buddy-heap bookkeeping uses PxReflectionAllocator/PxPool,
// which call PxGetBroadcastAllocator(). The original PhysX foundation must
// therefore be alive BEFORE constructing this heap; without it, source
// allocation bookkeeping dereferences an uninitialized foundation pointer.
struct SourcePhysxFoundation {
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback errors;
    physx::PxFoundation* foundation = nullptr;
    SourcePhysxFoundation() {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        if (!foundation)
            throw std::runtime_error("Unable to initialize original PhysX allocation foundation");
    }
    ~SourcePhysxFoundation() {
        if (foundation) foundation->release();
    }
    SourcePhysxFoundation(const SourcePhysxFoundation&) = delete;
    SourcePhysxFoundation& operator=(const SourcePhysxFoundation&) = delete;
};
// Model only PxgCudaBuffer's source-visible pointer/size fields; PhysX
// remains the owner of the real allocation and device context.
struct SourceBufferState {
    CudaDeviceAddressMap::Address raw = 0u;
    std::size_t size = 0u;
    CudaDeviceAddressMap::Address device() const {
        return (raw + 127u) & ~CudaDeviceAddressMap::Address{127u};
    }
};

void exercise_reused_source_submission_scratch(Context& context) {
    constexpr std::size_t bytes=256u;
    const auto gpu=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    CudaDeviceAddressMap addresses;
    const auto base=addresses.register_allocation(gpu,bytes);
    fyzz::physx_gpu::SourceSapDmaNative stream(context,addresses);
    std::array<std::uint32_t,8> first{},second{},third{},observed{};
    for(std::uint32_t i=0u;i<8u;++i) {
        first[i]=i+11u;
        second[i]=i+101u;
        third[i]=i+201u;
    }

    // Each source group must snapshot host bytes before its retained
    // vector is cleared for the next group, with one submission per group.
    auto& first_group=stream.upload_scratch();
    first_group.push_back({base,first.data(),sizeof(first)});
    first_group.push_back({base+64u,second.data(),sizeof(second)});
    stream.queue_uploads(first_group);
    const auto retained_capacity=first_group.capacity();
    first.fill(0u);
    second.fill(0u);
    auto& second_group=stream.upload_scratch();
    if(&first_group!=&second_group ||
       second_group.capacity()<retained_capacity ||
       !second_group.empty())
        throw std::runtime_error("Source upload group scratch was not retained");
    second_group.push_back({base+128u,third.data(),sizeof(third)});
    stream.queue_uploads(second_group);
    third.fill(0u);

    const auto before=stream.submission_stats();
    auto& invalid_group=stream.upload_scratch();
    invalid_group.push_back({base+160u,third.data(),sizeof(third)});
    invalid_group.push_back({base+bytes,third.data(),sizeof(third)});
    bool rejected=false;
    try {stream.queue_uploads(invalid_group);}
    catch(const std::out_of_range&){rejected=true;}
    if(!rejected || stream.submission_stats().upload_batches!=
            before.upload_batches)
        throw std::runtime_error(
            "Invalid reused source upload group partially submitted");

    // Overlapping copies within a single batch require the original
    // ordered transfer semantics even when both target one heap root.
    std::array<std::uint32_t,8> earlier{},later{};
    for(std::uint32_t i=0u;i<8u;++i) {
        earlier[i]=i+301u;
        later[i]=i+401u;
    }
    auto& overlapping=stream.upload_scratch();
    overlapping.push_back({base+192u,earlier.data(),sizeof(earlier)});
    overlapping.push_back({base+208u,later.data(),sizeof(later)});
    stream.queue_uploads(overlapping);
    earlier.fill(0u);
    later.fill(0u);
    if(stream.submission_stats().upload_batches!=before.upload_batches+1u)
        throw std::runtime_error(
            "Overlapping source upload changed GPU submission count");
    stream.wait_at_fetch();
    context.download(gpu,0u,observed.data(),sizeof(observed));
    for(std::uint32_t i=0u;i<observed.size();++i)
        if(observed[i]!=i+11u)
            throw std::runtime_error(
                "Reused source upload staging lost first host snapshot");
    context.download(gpu,64u,observed.data(),sizeof(observed));
    for(std::uint32_t i=0u;i<observed.size();++i)
        if(observed[i]!=i+101u)
            throw std::runtime_error(
                "Reused source upload staging lost second host snapshot");
    context.download(gpu,128u,observed.data(),sizeof(observed));
    for(std::uint32_t i=0u;i<observed.size();++i)
        if(observed[i]!=i+201u)
            throw std::runtime_error(
                "Reused source upload staging lost later host snapshot");

    std::array<std::uint32_t,12> overlap_result{};
    context.download(gpu,192u,overlap_result.data(),sizeof(overlap_result));
    for(std::uint32_t i=0u;i<overlap_result.size();++i) {
        const auto expected=i<4u?i+301u:i-4u+401u;
        if(overlap_result[i]!=expected)
            throw std::runtime_error(
                "Overlapping Vulkan/D3D12 source upload lost final writer");
    }

    auto& staged=stream.update_scratch();
    staged.emplace_back(CopyCommand{gpu,0u,gpu,192u,32u});
    const auto staged_capacity=staged.capacity();
    auto& retry=stream.update_scratch();
    if(!retry.empty() || retry.capacity()<staged_capacity ||
       !stream.commands().empty())
        throw std::runtime_error(
            "Discarded source update commands contaminated the GPU stream");
    addresses.unregister_allocation(base);
    context.destroy_buffer(gpu);
}

void exercise_source_raw_buffer_growth(Context& context) {
    // PhysX buddy heaps keep a root resource while their child raw
    // allocator keys move/swap. The two raw addresses deliberately have
    // DIFFERENT 128-byte residues to expose incorrect aligned-copy paths.
    constexpr std::size_t rootBytes=4096u;
    constexpr std::size_t oldPhysical=260u, newPhysical=784u, standbyPhysical=2048u;
    constexpr std::size_t oldSize=256u, newSize=512u;
    std::vector<std::uint8_t> initial(rootBytes,0xcdu),expected;
    for(std::size_t i=0u;i<oldSize;++i)
        initial[oldPhysical+i]=static_cast<std::uint8_t>((i*73u+17u)&255u);
    expected=initial;
    const auto root=context.create_buffer(rootBytes,MemoryPlacement::DeviceLocal);
    CudaDeviceAddressMap pointers;
    SourceBufferState growing{
        pointers.register_source_slice(root,oldPhysical,oldSize,4u),oldSize};
    SourceBufferState standby{
        pointers.register_source_slice(root,standbyPhysical,256u,0u),256u};
    if(growing.raw%128u!=4u || growing.device()-growing.raw!=124u)
        throw std::runtime_error("PhysX raw source pointer alignment was lost");

    context.upload(root,0u,initial.data(),rootBytes);
    const SourceBufferState old=growing;
    // Exact source allocateCopyOldDataAsync:
    // new size = max(requested, oldSize*2); DtoD is from RAW mPtr,
    // NOT getDevicePtr(); old pointer is deferred until completion.
    growing.size=newSize;
    growing.raw=pointers.register_source_slice(
        root,newPhysical,growing.size,16u);
    const auto src=pointers.resolve(old.raw,old.size);
    const auto dst=pointers.resolve(growing.raw,old.size);
    // The same-resource native operation is valid only for disjoint
    // ranges; an overlapping copy is not the source buffer-growth
    // contract and must be rejected before any GPU work is submitted.
    bool overlappingRejected=false;
    try {
        (void)context.submit_copy_buffer(
            root, oldPhysical, root, oldPhysical+4u, 128u);
    } catch(const std::invalid_argument&) {
        overlappingRejected=true;
    }
    if(!overlappingRejected)
        throw std::runtime_error("Overlapping source native GPU copy was accepted");
    overlappingRejected=false;
    try {
        (void)context.submit_batch({
            CopyCommand{root,oldPhysical,root,oldPhysical+4u,128u}});
    } catch(const std::invalid_argument&) {
        overlappingRejected=true;
    }
    if(!overlappingRejected)
        throw std::runtime_error("Overlapping batched native GPU copy was accepted");
    const auto copy=context.submit_copy_buffer(
        src.buffer,src.offset,dst.buffer,dst.offset,old.size);
    // Original deferred-free list must not permit physical block reuse
    // while this queued command still reads it.
    if(pointers.resolve(old.raw,old.size).offset!=oldPhysical)
        throw std::runtime_error("Deferred raw allocation retired before GPU copy");
    context.wait_submission(copy);
    std::copy_n(initial.begin()+oldPhysical,oldSize,expected.begin()+newPhysical);
    std::vector<std::uint8_t> observed(rootBytes);
    context.download(root,0u,observed.data(),rootBytes);
    if(observed!=expected)
        throw std::runtime_error("PhysX raw DtoD growth copy or surrounding canaries changed");

    // Same-root GPU copies must also compose inside the caller's native
    // source-ordered batch; standalone submit_copy_buffer is not the
    // only path a future PxCudaContext stream adapter can exercise.
    // The second copy consumes bytes written by the first without
    // inserting a host wait between the two.
    constexpr std::size_t chainedDestination=3072u;
    const std::vector<BatchCommand> nativeCopies{
        CopyCommand{root,newPhysical,root,standbyPhysical,oldSize},
        CopyCommand{root,standbyPhysical,root,chainedDestination,oldSize}};
    context.wait_submission(context.submit_batch(nativeCopies));
    std::copy_n(initial.begin()+oldPhysical,oldSize,
                expected.begin()+standbyPhysical);
    std::copy_n(initial.begin()+oldPhysical,oldSize,
                expected.begin()+chainedDestination);
    context.download(root,0u,observed.data(),rootBytes);
    if(observed!=expected)
        throw std::runtime_error(
            "Ordered same-root GPU copy batch corrupted source data or canaries");

    if(growing.size!=old.size*2u ||
       pointers.resolve(growing.device(),growing.size-
           static_cast<std::size_t>(growing.device()-growing.raw)).offset!=
           newPhysical+112u)
        throw std::runtime_error("PhysX device-visible alignment lost after raw growth");

    // Flush source's deferred allocation only after completion. The old
    // virtual address becomes stale; the root remains owned by PhysX.
    const auto freed=pointers.unregister_slice(old.raw);
    if(freed.buffer!=root || freed.offset!=oldPhysical ||
       freed.remaining!=oldSize)
        throw std::runtime_error("Deferred free lost allocator's RAW key");
    bool rejected=false;
    try{(void)pointers.resolve(old.raw,1u);}
    catch(const std::out_of_range&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("Stale raw pointer resurrected after deferred free");

    // Source PxgCudaBuffer::swapBuffer exchanges RAW identities and sizes.
    std::swap(growing.raw,standby.raw);
    std::swap(growing.size,standby.size);
    if(pointers.resolve(growing.raw,1u).offset!=standbyPhysical ||
       pointers.resolve(standby.raw,1u).offset!=newPhysical)
        throw std::runtime_error("Buffer swap exchanged aligned pointers instead of raw keys");
    // Source assign() releases its current RAW key, takes rhs ownership,
    // and nulls rhs (the underlying heap root itself remains live).
    if(pointers.unregister_slice(growing.raw).offset!=standbyPhysical)
        throw std::runtime_error("Assign failed to deallocate raw destination");
    growing=standby;
    standby={};
    if(!growing.raw || !growing.size ||
       standby.raw || standby.size ||
       pointers.resolve(growing.raw,1u).offset!=newPhysical)
        throw std::runtime_error("PxgCudaBuffer assign lost source raw allocation identity");
    if(pointers.unregister_slice(growing.raw).offset!=newPhysical ||
       pointers.live_allocations()!=0u)
        throw std::runtime_error("Source buffer assign/free left a logical allocation live");
    context.destroy_buffer(root);
}

void exercise_native_physx_heap_roots(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourceSapDmaNative;
    CudaDeviceAddressMap addresses;
    SourcePhysxHeapRoots roots(context,addresses);

    // PhysX's PxgHeapMemoryAllocator asks its lower-level allocator for
    // pages and occasional exceptional allocations, then suballocates
    // within each ROOT using raw pointer arithmetic. No second buddy
    // allocator, per-block registration or fake pointer alignment.
    constexpr std::size_t oldBytes=4096u, newBytes=8192u, exceptionBytes=2048u;
    const auto oldRoot=roots.allocate_root(oldBytes);
    const auto newRoot=roots.allocate_root(newBytes);
    const auto exceptionalRoot=roots.allocate_root(exceptionBytes);
    if(roots.live_roots()!=3u || addresses.live_allocations()!=3u ||
       oldRoot%256u || newRoot%256u || exceptionalRoot%256u)
        throw std::runtime_error("Original PhysX heap root identities are invalid");

    const auto oldRaw=oldRoot+256u;
    const auto newRaw=newRoot+1024u;
    const auto exceptionalRaw=exceptionalRoot+128u;
    const auto oldView=addresses.resolve(oldRaw,512u);
    const auto newView=addresses.resolve(newRaw,512u);
    const auto exceptionalView=addresses.resolve(exceptionalRaw,512u);
    if(oldView.offset!=256u || newView.offset!=1024u ||
       exceptionalView.offset!=128u ||
       oldView.buffer==newView.buffer ||
       exceptionalView.buffer==newView.buffer)
        throw std::runtime_error("PhysX multi-root buddy/exceptional address lowering changed");

    bool rejected=false;
    try {roots.release_root(oldRaw);}
    catch(const std::out_of_range&) {rejected=true;}
    if(!rejected || roots.live_roots()!=3u)
        throw std::runtime_error("Original PhysX buddy block mistaken for root free");
    rejected=false;
    try {(void)roots.allocate_root(0u);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected || roots.live_roots()!=3u)
        throw std::runtime_error("Invalid PhysX root allocation changed owner state");

    std::array<std::uint8_t,512> payload{},observed{};
    for(std::size_t i=0;i<payload.size();++i)
        payload[i]=static_cast<std::uint8_t>((i*73u+19u)%251u);
    // PhysX's original preBroadPhase upload is ordered before growth copies.
    SourceSapDmaNative source(context,addresses);
    source.queue_uploads({{oldRaw,payload.data(),payload.size()}});
    // Original PxgCudaBuffer::allocateCopyOldDataAsync copies exactly the
    // OLD RAW allocation bytes from old to new; the downstream copy reads
    // those bytes on the same GPU stream, with no intermediate CPU wait.
    source.queue_device_copy(newRaw,oldRaw,payload.size());
    source.queue_device_copy(exceptionalRaw,newRaw,payload.size());
    // The original heap callback may release the OLD ROOT once its owning
    // allocations have been deferred. Submit pending GPU work first, never
    // wait; the native Context retains in-flight storage until completion.
    CudaDeviceAddressMap unrelatedAddresses;
    SourceSapDmaNative unrelatedStream(context,unrelatedAddresses);
    rejected=false;
    try {roots.release_root(oldRoot,&unrelatedStream);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected || roots.live_roots()!=3u ||
       addresses.resolve(oldRaw,1u).offset!=256u ||
       source.commands().size()!=2u)
        throw std::runtime_error(
            "Foreign scene stream retired original PhysX GPU allocation");

    roots.release_root(oldRoot,&source);
    rejected=false;
    try {(void)addresses.resolve(oldRaw,1u);}
    catch(const std::out_of_range&) {rejected=true;}
    if(!rejected || addresses.live_allocations()!=2u ||
       roots.live_roots()!=2u)
        throw std::runtime_error("Retired source heap root remains GPU-addressable");

    source.wait_at_fetch();
    context.download(exceptionalView.buffer,exceptionalView.offset,
                     observed.data(),observed.size());
    if(observed!=payload)
        throw std::runtime_error("Original PhysX root retirement raced ordered GPU copy");
    context.download(newView.buffer,newView.offset,
                     observed.data(),observed.size());
    if(observed!=payload)
        throw std::runtime_error("Original PhysX source raw growth lost new buffer contents");

    roots.release_root(exceptionalRoot);
    roots.release_root(newRoot);
    if(roots.live_roots()!=0u || addresses.live_allocations()!=0u)
        throw std::runtime_error("Original PhysX native heap roots leaked");
    const auto again=roots.allocate_root(1024u);
    if(again==oldRoot || again==newRoot || again==exceptionalRoot)
        throw std::runtime_error("Freed PhysX source CUDA root address was reused");
    roots.release_root(again);
}


void exercise_native_heap_raii_teardown(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourceSapDmaNative;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative source(context, addresses);
    constexpr std::size_t bytes=4096u, offset=512u, length=256u;
    std::array<std::uint8_t,length> pattern{};
    for (std::size_t i=0u;i<pattern.size();++i)
        pattern[i]=static_cast<std::uint8_t>((i*29u+7u)%251u);
    CudaDeviceAddressMap::Address oldRoot=0u, newRoot=0u;
    BufferId oldNative=0u, newNative=0u;
    {
        SourcePhysxHeapRoots roots(context,addresses);
        oldRoot=roots.allocate_root(bytes);
        newRoot=roots.allocate_root(bytes);
        oldNative=addresses.resolve(oldRoot,1u).buffer;
        newNative=addresses.resolve(newRoot,1u).buffer;
        source.queue_uploads({{oldRoot+offset,pattern.data(),length}});
        source.queue_device_copy(newRoot+offset,oldRoot+offset,length);
        source.submit(); // source caller must submit before unwinding owner
        // Neither root is explicitly freed. Destructor must retire both,
        // invalidate mappings and preserve submitted upload/copy order.
    }
    if(addresses.live_allocations()!=0u)
        throw std::runtime_error("Implicit PhysX heap shutdown left CUDA root mappings");
    for(const auto pointer:{oldRoot,newRoot}) {
        bool invalid=false;
        try {(void)addresses.resolve(pointer,1u);}
        catch(const std::out_of_range&){invalid=true;}
        if(!invalid)
            throw std::runtime_error("RAII-retired PhysX root address remained resolvable");
    }
    source.wait_at_fetch();
    for(const auto buffer:{oldNative,newNative}) {
        bool invalid=false;
        try {context.destroy_buffer(buffer);}
        catch(const std::out_of_range&){invalid=true;}
        catch(const std::invalid_argument&){invalid=true;}
        if(!invalid)
            throw std::runtime_error("RAII PhysX heap owner left backing compute buffer live");
    }

    // Scene initialization failure is also a normal RAII cleanup path.
    // Address map and GPU context intentionally outlive the owner.
    CudaDeviceAddressMap::Address exceptionalRoot=0u;
    BufferId exceptionalNative=0u;
    bool threw=false;
    try {
        SourcePhysxHeapRoots roots(context,addresses);
        exceptionalRoot=roots.allocate_root(1024u);
        exceptionalNative=addresses.resolve(exceptionalRoot,1u).buffer;
        (void)roots.allocate_root(2048u);
        throw std::runtime_error("Original PhysX scene setup failed");
    } catch(const std::runtime_error&) {threw=true;}
    if(!threw||addresses.live_allocations()!=0u)
        throw std::runtime_error("Scene-setup exception leaked native PhysX heap allocations");
    bool stale=false;
    try {(void)addresses.resolve(exceptionalRoot,1u);}
    catch(const std::out_of_range&){stale=true;}
    if(!stale)
        throw std::runtime_error("Exceptional scene teardown left a stale CUDA root");
    stale=false;
    try {context.destroy_buffer(exceptionalNative);}
    catch(const std::out_of_range&){stale=true;}
    catch(const std::invalid_argument&){stale=true;}
    if(!stale)
        throw std::runtime_error("Exceptional scene teardown leaked an owned GPU buffer");

    // Normal close must also perform source-stream submission exactly once,
    // reject alien streams without changing ownership, and be idempotent.
    CudaDeviceAddressMap foreignAddresses;
    SourceSapDmaNative foreign(context,foreignAddresses);
    CudaDeviceAddressMap::Address normalRoot=0u;
    {
        SourcePhysxHeapRoots roots(context,addresses);
        normalRoot=roots.allocate_root(bytes);
        source.queue_uploads({{normalRoot+offset,pattern.data(),length}});
        bool rejected=false;
        try {roots.close(&foreign);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected || roots.live_roots()!=1u ||
           addresses.resolve(normalRoot,1u).offset!=0u)
            throw std::runtime_error("Alien PhysX close mutated live heap ownership");
        roots.close(&source);
        roots.close(&source);
        if(roots.live_roots()!=0u||addresses.live_allocations()!=0u)
            throw std::runtime_error("Explicit native PhysX heap shutdown left roots live");
    }
    source.wait_at_fetch();
}


void exercise_source_virtual_device_allocator(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    CudaDeviceAddressMap addresses;
    SourcePhysxHeapRoots roots(context,addresses);
    SourceSapDmaNative source(context,addresses);
    SourcePhysxDeviceRootCallback native(roots,&source);
    // Use the EXACT source Cm::VirtualAllocatorCallback interface, not a
    // lookalike function table or an invented CUDA context API.
    physx::Cm::VirtualAllocatorCallback& callback=native;
    if(callback.allocate(0u,1,"source-heap",7)!=nullptr)
        throw std::runtime_error("Original PhysX zero-byte root request returned a pointer");
    void* old=callback.allocate(4096u,1,"source-heap",17);
    void* successor=callback.allocate(8192u,2,"source-heap",19);
    void* exceptional=callback.allocate(2048u,3,"source-heap",21);
    const auto address=[](void* p) {
        return static_cast<CudaDeviceAddressMap::Address>(
            reinterpret_cast<std::uintptr_t>(p));
    };
    const auto oldAddress=address(old);
    const auto newAddress=address(successor);
    const auto exceptionalAddress=address(exceptional);
    if(!old || !successor || !exceptional ||
       addresses.live_allocations()!=3u || roots.live_roots()!=3u ||
       oldAddress==newAddress || newAddress==exceptionalAddress)
        throw std::runtime_error("Source virtual heap callback lost root allocation identity");
    const auto oldView=addresses.resolve(oldAddress+256u,512u);
    const auto newView=addresses.resolve(newAddress+1024u,512u);
    const auto exceptionalView=addresses.resolve(exceptionalAddress+128u,512u);
    if(oldView.offset!=256u || newView.offset!=1024u ||
       exceptionalView.offset!=128u ||
       oldView.buffer==newView.buffer || newView.buffer==exceptionalView.buffer)
        throw std::runtime_error("Source virtual callback lost buddy/exceptional root arithmetic");
    bool invalid=false;
    try {
        callback.deallocate(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(oldAddress+256u)));
    } catch(const std::out_of_range&) {invalid=true;}
    if(!invalid || roots.live_roots()!=3u)
        throw std::runtime_error("Source callback accepted an interior buddy block as a root");

    std::array<std::uint8_t,512u> initial{},actual{};
    for(std::size_t i=0u;i<initial.size();++i)
        initial[i]=static_cast<std::uint8_t>((i*43u+29u)%251u);
    source.queue_uploads({{oldAddress+256u,initial.data(),initial.size()}});
    source.queue_device_copy(newAddress+1024u,
                             oldAddress+256u,initial.size());
    source.queue_device_copy(exceptionalAddress+128u,
                             newAddress+1024u,initial.size());
    // Original PxgCudaDeviceMemoryAllocatorCallback::deallocate(ptr)
    // receives the EXACT heap root from PxgHeapMemoryAllocator.
    // This submits queued stream commands and retires the old native
    // allocation, without blocking the CPU at a scene setup boundary.
    callback.deallocate(old);
    invalid=false;
    try {(void)addresses.resolve(oldAddress+256u,1u);}
    catch(const std::out_of_range&) {invalid=true;}
    if(!invalid || roots.live_roots()!=2u)
        throw std::runtime_error("Source virtual root free retained stale GPU address");
    source.wait_at_fetch();
    context.download(newView.buffer,newView.offset,actual.data(),actual.size());
    if(actual!=initial)
        throw std::runtime_error("Source allocator callback interrupted raw growth copy");
    context.download(exceptionalView.buffer,exceptionalView.offset,actual.data(),
                     actual.size());
    if(actual!=initial)
        throw std::runtime_error("Source allocator callback corrupted exceptional root copy");

    callback.deallocate(exceptional);
    callback.deallocate(successor);
    callback.deallocate(nullptr);
    if(roots.live_roots()!=0u || addresses.live_allocations()!=0u)
        throw std::runtime_error("Original PhysX virtual allocator callback leaked roots");
    roots.close(&source);
}


void exercise_original_source_physx_buddy_heap(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    CudaDeviceAddressMap addresses;
    SourcePhysxHeapRoots roots(context, addresses);
    SourceSapDmaNative source(context, addresses);
    SourcePhysxDeviceRootCallback native(roots, &source);
    bool aborted=false;
    const auto abort=[](void* ctx) noexcept {
        *static_cast<bool*>(ctx)=true;
    };
    std::array<std::uint8_t,256u> payload{}, observed{};
    for(std::size_t i=0u;i<payload.size();++i)
        payload[i]=static_cast<std::uint8_t>((i*59u+31u)%251u);
    {
        // This IS the original source's Block/PxgHeapMemoryAllocator
        // implementation, using its original buddy lists, mHashMap,
        // exceptional allocation policy and deferred free path. Only
        // native root callback and abort notification are injected.
        physx::PxgHeapMemoryAllocator heap(4096u,native,abort,&aborted);
        void* a=heap.allocate(256u,0,__FILE__,__LINE__);
        void* b=heap.allocate(384u,0,__FILE__,__LINE__);
        void* exception=heap.allocate(3072u,0,__FILE__,__LINE__);
        if(!a || !b || !exception || aborted)
            throw std::runtime_error("Original PhysX buddy heap failed native allocation");
        const auto addr=[](void* pointer) {
            return static_cast<CudaDeviceAddressMap::Address>(
                reinterpret_cast<std::uintptr_t>(pointer));
        };
        const auto aRaw=addr(a),bRaw=addr(b),exceptionRaw=addr(exception);
        const auto ar=addresses.resolve(aRaw,256u);
        const auto br=addresses.resolve(bRaw,256u);
        const auto er=addresses.resolve(exceptionRaw,256u);
        if(ar.buffer!=br.buffer || ar.buffer==er.buffer ||
           ar.offset==br.offset || ar.offset+256u>4096u ||
           br.offset+384u>4096u || er.offset!=0u ||
           roots.live_roots()!=2u)
            throw std::runtime_error("Native original PhysX buddy splitting or exceptional root mismatch");
        source.queue_uploads({{aRaw,payload.data(),payload.size()}});
        source.queue_device_copy(exceptionRaw,aRaw,payload.size());
        source.queue_device_copy(bRaw,exceptionRaw,payload.size());
        // Source exceptional free is a ROOT release through the exact
        // Cm::VirtualAllocatorCallback; submits the original GPU stream
        // but never waits on the CPU.
        heap.deallocate(exception);
        if(roots.live_roots()!=1u)
            throw std::runtime_error("Original PhysX exceptional free did not retire its root");
        source.wait_at_fetch();
        context.download(br.buffer,br.offset,observed.data(),observed.size());
        if(observed!=payload)
            throw std::runtime_error("Original PhysX buddy heap interrupted native GPU copies");
        heap.deallocateDeferred(b);
        // The source heap (not the root address mapper) owns buddy
        // suballocation lifetimes. Assert its actual accounting here;
        // deterministic reuse is tested separately below.
        if(heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=640u)
            throw std::runtime_error("Source deferred free changed live allocation stats");
        heap.flushDeferredDeallocs();
        if(heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=256u)
            throw std::runtime_error("Source deferred flush did not release block accounting");
        heap.deallocate(a);
        // Same source heap can reuse a released block; the root stays alive
        // until the original heap's destructor, not a per-block native free.
        void* reused=heap.allocate(256u,0,__FILE__,__LINE__);
        if(!reused || roots.live_roots()!=1u)
            throw std::runtime_error("Original PhysX buddy heap lost its free-list reuse contract");
        heap.deallocate(reused);
    }

    // A full two-block heap lets us prove the actual deferred FREE LIST
    // contract independently of the root address map. Without flushing,
    // allocating an identically-sized block must take a new heap root;
    // after flushing, once that new root is full, the next block MUST be
    // the original deferred address. No allocator heuristics are guessed.
    {
        physx::PxgHeapMemoryAllocator heap(1024u,native,abort,&aborted);
        void* a=heap.allocate(512u,0,__FILE__,__LINE__);
        void* b=heap.allocate(512u,0,__FILE__,__LINE__);
        if(!a||!b||roots.live_roots()!=1u)
            throw std::runtime_error("Source deferred free fixture did not fill one root");
        heap.deallocateDeferred(b);
        void* early=heap.allocate(512u,0,__FILE__,__LINE__);
        if(!early || early==b || roots.live_roots()!=2u)
            throw std::runtime_error("Source buddy block reused BEFORE deferred flush");
        void* fill=heap.allocate(512u,0,__FILE__,__LINE__);
        if(!fill || fill==b || roots.live_roots()!=2u)
            throw std::runtime_error("Source second root not filled before deferred flush");
        heap.flushDeferredDeallocs();
        void* recycled=heap.allocate(512u,0,__FILE__,__LINE__);
        if(recycled!=b || roots.live_roots()!=2u)
            throw std::runtime_error("Source buddy block not reusable AFTER deferred flush");
        heap.deallocate(recycled);
        heap.deallocate(fill);
        heap.deallocate(early);
        heap.deallocate(a);
        if(heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=0u)
            throw std::runtime_error("Source deferred free fixture leaked heap statistics");
    }

    // An actual GPU root allocator can return null on device OOM or throw
    // during allocation. Neither path may account bytes that do not exist.
    // Inject both failures AFTER constructor/root setup, in ordinary-root
    // and exceptional-allocation paths.
    struct FailingRoots final : physx::Cm::VirtualAllocatorCallback {
        physx::Cm::VirtualAllocatorCallback& native;
        int failure = 0;
        explicit FailingRoots(physx::Cm::VirtualAllocatorCallback& allocator)
            : native(allocator) {}
        void* allocate(std::size_t bytes,int group,const char* file,int line) override {
            if(failure==1) return nullptr;
            if(failure==2) throw std::bad_alloc();
            return native.allocate(bytes,group,file,line);
        }
        void deallocate(void* address) override {native.deallocate(address);}
    } controlled(native);
    {
        physx::PxgHeapMemoryAllocator heap(1024u,controlled,abort,&aborted);
        void* a=heap.allocate(512u,0,__FILE__,__LINE__);
        void* b=heap.allocate(512u,0,__FILE__,__LINE__);
        if(!a||!b||heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=1024u)
            throw std::runtime_error("Source OOM fixture has incorrect baseline allocations");
        for(int mode:{1,2}) {
            controlled.failure=mode;
            for(std::size_t bytes:{512u,2048u}) {
                bool rejected=false;
                try {rejected=heap.allocate(bytes,0,__FILE__,__LINE__)==nullptr;}
                catch(const std::bad_alloc&){rejected=true;}
                if(!rejected ||
                   heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=1024u)
                    throw std::runtime_error("Original PhysX heap charged a failed GPU allocation");
            }
        }
        controlled.failure=0;
        heap.deallocate(b);
        heap.deallocate(a);
        if(heap.getHeapStats().stats[physx::PxsHeapStats::eOTHER]!=0u)
            throw std::runtime_error("Original PhysX GPU heap OOM fixture leaked statistics");
    }

    if(aborted || roots.live_roots()!=0u ||
       addresses.live_allocations()!=0u)
        throw std::runtime_error("Original PhysX heap root lifetime or abort signaling changed");
}

void exercise_original_native_physx_cuda_buffer(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted = false;
    const auto abort=[](void* value) noexcept {
        *static_cast<bool*>(value) = true;
    };
    std::array<std::uint8_t,256u> bytes{}, received{};
    for(std::size_t i=0; i<bytes.size(); ++i)
        bytes[i] = static_cast<std::uint8_t>((i * 37u + 19u) % 251u);

    {
        physx::PxgHeapMemoryAllocator heap(8192u, callback, abort, &aborted);
        {
            // Instantiate the real PhysX buffer object, not a fixture
            // reproducing its pointer and size fields.
            physx::PxgCudaBuffer buffer(heap, physx::PxsHeapStats::eBROADPHASE);
            buffer.allocate(256u, __FILE__, __LINE__);
            const auto old = buffer.getDevicePtr();
            if (!old || buffer.getSize() != 256u)
                throw std::runtime_error("PhysX source buffer failed initial device allocation");
            stream.queue_uploads({{old, bytes.data(), bytes.size()}});
            buffer.allocateCopyOldDataAsync(513u, stream, __FILE__, __LINE__);
            const auto newer = buffer.getDevicePtr();
            if (!newer || newer == old || buffer.getSize() != 513u)
                throw std::runtime_error("Original PhysX buffer growth lost raw identity/capacity");
            if (addresses.resolve(old, bytes.size()).buffer !=
                    addresses.resolve(newer, bytes.size()).buffer)
                throw std::runtime_error("PhysX buffer fixture did not exercise a shared native heap root");

            // The old raw block must remain allocated until the original
            // heap's deferred-flush boundary, including after GPU submission.
            physx::PxgCudaBufferN<2> history(heap, physx::PxsHeapStats::eBROADPHASE);
            history[0].allocate(256u, __FILE__, __LINE__);
            if (history[0].getDevicePtr() == old)
                throw std::runtime_error("PhysX deferred buffer was reused before flush");
            history[1].allocate(128u, __FILE__, __LINE__);
            const auto first = history[0].getDevicePtr();
            const auto second = history[1].getDevicePtr();
            physx::PxgCudaBuffer::swapBuffer(history[0], history[1]);
            if (history[0].getDevicePtr() != second ||
                history[1].getDevicePtr() != first)
                throw std::runtime_error("Original PhysX history swap lost raw allocation identity");

            stream.wait_at_fetch();
            const auto dst = addresses.resolve(newer, bytes.size());
            context.download(dst.buffer, dst.offset, received.data(), received.size());
            if (received != bytes)
                throw std::runtime_error("Original PxgCudaBuffer native raw growth copy changed payload");
            heap.flushDeferredDeallocs();

            // assign() releases the destination and transfers the source
            // allocation, leaving exactly one owner of the raw heap key.
            history[0].assign(history[1]);
            if (history[0].getDevicePtr() != first ||
                history[1].getDevicePtr() || history[1].getSize())
                throw std::runtime_error("Original PhysX buffer assign changed ownership");
            buffer.deallocateDeferred();
            if (buffer.getDevicePtr() || buffer.getSize())
                throw std::runtime_error("Deferred source buffer free retained ownership");
            heap.flushDeferredDeallocs();
        }
        if (heap.getHeapStats().stats[physx::PxsHeapStats::eBROADPHASE] != 0u)
            throw std::runtime_error("Original PhysX buffer destructors leaked GPU heap accounting");
    }
    if (aborted || roots.live_roots() || addresses.live_allocations())
        throw std::runtime_error("Original PhysX CUDA buffer left live GPU root allocations");
}

void exercise_source_single_volume_bitmap_lifecycle_native(Context& context) {
    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback deviceCallback(roots,&stream);
    SourcePhysxMappedReportRoots pinned(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* data) noexcept {
        *static_cast<bool*>(data)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,deviceCallback,abort,&aborted);
        constexpr auto stat=physx::PxsHeapStats::eBROADPHASE;
        // The source manager owns its original bitmap and PinnableArray
        // members. Admission and first-pass lowering borrow these directly.
        physx::Cm::PinnableBitMap added(pinned,stat),removed(pinned,stat),
            changed(pinned,stat),aggregated(pinned,stat);
        physx::Cm::PinnableArray<physx::Bp::ShapeHandle> created(pinned,stat),
            retired(pinned,stat),updated(pinned,stat);
        std::array<physx::Bp::VolumeData,40u> volumes{};
        for(auto& v:volumes)v.reset();
        std::array<physx::Bp::FilterGroup::Enum,40u> groups{};
        groups.fill(physx::Bp::FilterGroup::eINVALID);
        std::array<physx::PxReal,40u> offsets{};
        std::array<physx::PxBounds3,40u> bounds{};
        bounds.fill(physx::PxBounds3::centerExtents(
            physx::PxVec3(0.f),physx::PxVec3(1.f)));
        bool persistent=false,gpu=false;
        SourceSapSingleVolumeMembersNative manager{
            added,removed,changed,aggregated,volumes,groups,offsets,
            persistent,gpu};
        std::array<fyzz::BodyId,40u> sourceActorIds{};
        sourceActorIds.fill(std::numeric_limits<fyzz::BodyId>::max());
        const SourceSapVolumeActorLifetimeNative identity{
            sourceActorIds,volumes};
        const auto first_pass=[&]() {
            collect_source_sap_single_volume_first_pass_native(
                manager,{created,retired,updated});
        };
        // Original AABB handles need not equal actor IDs or arrive sorted.
        alignas(4) int actorA=1,actorB=2;
        add_source_sap_single_volume_actor_native(
            manager,identity,36u,0.04f,
            physx::Bp::FilterGroup::eSTATICS,&actorA,36u);
        add_source_sap_single_volume_actor_native(
            manager,identity,2u,0.02f,
            physx::Bp::getFilterGroup_Dynamics(99u,false),&actorB,2u);
        add_source_sap_single_volume_actor_native(
            manager,identity,7u,0.03f,
            physx::Bp::FilterGroup::eSTATICS,&actorA,7u);
        if(!remove_source_sap_single_volume_actor_native(manager,identity,7u) ||
           sourceActorIds[7]!=std::numeric_limits<fyzz::BodyId>::max())
            throw std::runtime_error(
                "Canceled source volume retained a GPU-unpublished actor identity");
        bool rejected=false;
        try {add_source_sap_single_volume_actor_native(
            manager,identity,2u,0.04f,
            physx::Bp::FilterGroup::eSTATICS,&actorA,2u);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected || !volumes[2].isSingleActor() ||
           groups[2]!=physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error("Duplicate source volume admission changed the owner");
        first_pass();
        if(created.size()!=2u || created[0]!=2u || created[1]!=36u ||
           retired.size()!=0u || updated.size()!=0u || gpu || !persistent ||
           added.getWordCount()!=2u || !added.boundedTest(36u) ||
           added.boundedTest(7u))
            throw std::runtime_error(
                "PhysX single-volume source first pass lost sorted/cancelled handles");

        physx::PxgCudaBuffer addedGpu(heap,stat),removedGpu(heap,stat),
            aggregatedGpu(heap,stat),changedGpu(heap,stat),volumeGpu(heap,stat);
        physx::PxgTypedCudaBuffer<physx::PxBounds3> boundsGpu(heap,stat);
        physx::PxgTypedCudaBuffer<physx::PxReal> distancesGpu(heap,stat);
        physx::PxgTypedCudaBuffer<physx::PxU32> groupsGpu(heap,stat),
            environmentGpu(heap,stat);
        physx::PxU32 boxCapacity=0u;
        const auto manager_view=[&]() {
            return SourceSapAabbManagerPreUpdateNative{
                {boundsGpu,distancesGpu,groupsGpu,environmentGpu,boxCapacity},
                source_sap_aabb_bitmap_native(addedGpu,added),
                source_sap_aabb_bitmap_native(removedGpu,removed),
                source_sap_aabb_bitmap_native(aggregatedGpu,aggregated),
                source_sap_aabb_bitmap_native(changedGpu,changed),
                volumeGpu,volumes,persistent,gpu,false};
        };
        const physx::Bp::BpFilter filter(false,false);
        using Box=fyzz::native::GpuBroadphaseBox;
        std::array<Box,2u> live_boxes{{
            {36u,{3.f,0.f,0.f},{1.f,1.f,1.f},{},0.04f,
                Box::Motion::Static},
            {2u,{0.f,0.f,0.f},{1.f,1.f,1.f},{},0.02f,
                Box::Motion::Dynamic}
        }};
        const std::array<physx::Bp::ShapeHandle,2u> live_handles{36u,2u};
        SourceSapSceneBoundsScratch sceneScratch;
        const SourceSapSceneBounds sceneBounds{bounds,offsets,groups};
        const std::array<physx::Bp::ShapeHandle,2u> unadmitted{36u,7u};
        bool rejectedUnadmitted=false;
        try {
            (void)prepare_source_sap_single_volume_scene_native(
                manager,{created,retired,updated},live_boxes,unadmitted,
                sceneBounds,filter,true,sceneScratch);
        }catch(const std::invalid_argument&){rejectedUnadmitted=true;}
        if(!rejectedUnadmitted ||
           groups[7]!=physx::Bp::FilterGroup::eINVALID ||
           groups[36]!=physx::Bp::FilterGroup::eSTATICS)
            throw std::runtime_error(
                "Unadmitted source volume was resurrected by Sc geometry");

        const auto frame=prepare_source_sap_single_volume_scene_native(
            manager,{created,retired,updated},live_boxes,live_handles,
            sceneBounds,filter,true,sceneScratch);
        validate_source_sap_volume_actor_lifetime_native(identity,frame);
        if(groups[2]!=physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error(
                "GPU Sc bounds overwrote admitted PhysX rigid actor filter group");
        if(frame.getNumCreatedHandles()!=2u ||
           frame.getNumUpdatedHandles()!=0u ||
           frame.getNumRemovedHandles()!=0u ||
           !frame.getStateChanged() || frame.getGpuStateChanged() ||
           frame.getCreatedHandles()!=created.begin())
            throw std::runtime_error(
                "Source single-volume first pass did not construct original Bp frame");
        pre_update_source_sap_aabb_manager_native(
            frame,manager_view(),stream);
        stream.wait_at_fetch();
        const auto gpuWords=addresses.resolve(
            addedGpu.getDevicePtr(),added.getWordCount()*sizeof(physx::PxU32));
        std::array<physx::PxU32,2> observed{};
        context.download(gpuWords.buffer,gpuWords.offset,
                         observed.data(),sizeof(observed));
        if(observed[0]!=(1u<<2u) || observed[1]!=(1u<<4u))
            throw std::runtime_error(
                "Native GPU AABB bitmap differs from original source admission");

        // Source processFoundPairs/processLostPairs retire these host bits
        // after successful fetch; the next frame reuses the same allocations.
        clear_source_sap_single_volume_report_bitmaps_native(manager);
        persistent=false;gpu=false;
        if(remove_source_sap_single_volume_actor_native(manager,identity,2u))
            throw std::runtime_error("Previously admitted volume was cancelled");
        add_source_sap_single_volume_actor_native(
            manager,identity,2u,0.05f,
            physx::Bp::getFilterGroup_Dynamics(99u,false),&actorB,2u);
        if(remove_source_sap_single_volume_actor_native(manager,identity,36u))
            throw std::runtime_error("Previously published volume was not retired");
        if(sourceActorIds[36]!=36u)
            throw std::runtime_error(
                "Retired source volume lost identity before GPU lost-pair fetch");
        bool blockedReuse=false;
        try {add_source_sap_single_volume_actor_native(
            manager,identity,36u,0.04f,
            physx::Bp::FilterGroup::eSTATICS,&actorA,99u);}
        catch(const std::logic_error&){blockedReuse=true;}
        if(!blockedReuse || groups[36]!=physx::Bp::FilterGroup::eINVALID)
            throw std::runtime_error(
                "Source volume was recycled before its lost report completed");
        mark_source_sap_single_volume_changed_native(manager,2u,0.125f);
        first_pass();
        if(created.size()!=0u || retired.size()!=1u || retired[0]!=36u ||
           updated.size()!=0u || !gpu || !persistent ||
           groups[36]!=physx::Bp::FilterGroup::eINVALID ||
           volumes[36].getUserData()!=nullptr ||
           offsets[2]!=0.125f)
            throw std::runtime_error(
                "Original source remove/re-add and changed bitmap semantics diverged");
        std::array<Box,1u> remaining{{
            {2u,{6.f,0.f,0.f},{1.f,1.f,1.f},{},0.125f,
                Box::Motion::Dynamic}
        }};
        const std::array<physx::Bp::ShapeHandle,1u> retained_handle{2u};
        const auto next=prepare_source_sap_single_volume_scene_native(
            manager,{created,retired,updated},remaining,retained_handle,
            sceneBounds,filter,true,sceneScratch);
        if(groups[2]!=physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error(
                "Moving source volume replaced original Sc actor filter group");
        if(std::fabs(bounds[2].minimum.x-5.f)>1.0e-6f ||
           std::fabs(bounds[36].minimum.x-2.f)>1.0e-6f)
            throw std::runtime_error(
                "GPU-only changed volume did not advance original AABB bounds");
        validate_source_sap_volume_actor_lifetime_native(identity,next);
        if(next.getNumCreatedHandles()!=0u ||
           next.getNumUpdatedHandles()!=0u ||
           next.getNumRemovedHandles()!=1u ||
           !next.getStateChanged() || !next.getGpuStateChanged() ||
           next.getRemovedHandles()!=retired.begin())
            throw std::runtime_error(
                "Source removed/changed bitmap frame omitted original GPU dirty flags");
        pre_update_source_sap_aabb_manager_native(next,manager_view(),stream);
        stream.wait_at_fetch();
        const auto removedGpuWords=addresses.resolve(
            removedGpu.getDevicePtr(),removed.getWordCount()*sizeof(physx::PxU32));
        observed={};
        context.download(removedGpuWords.buffer,removedGpuWords.offset,
                         observed.data(),sizeof(observed));
        if(observed[0]!=0u || observed[1]!=(1u<<4u))
            throw std::runtime_error(
                "Original source GPU removed bitmap lost retired volume");
        const auto changedGpuWords=addresses.resolve(
            changedGpu.getDevicePtr(),changed.getWordCount()*sizeof(physx::PxU32));
        physx::PxU32 observedChanged=0;
        context.download(changedGpuWords.buffer,changedGpuWords.offset,
                         &observedChanged,sizeof(observedChanged));
        if(!(observedChanged & (1u<<2u)))
            throw std::runtime_error(
                "Original source changed-handle bitmap not uploaded to native GPU");

        // GPU report identities are retained across source VolumeData reset.
        // Found reports must reject the retired source volume, while the
        // same original pair remains valid as a lost report until fetch.
        const std::array<physx::PxgBroadPhasePair,1u> formerPair{
            physx::PxgBroadPhasePair{2u,36u}};
        bool staleFound=false;
        try {validate_source_sap_found_volume_lifetime_native(
            formerPair,identity);}
        catch(const std::invalid_argument&){staleFound=true;}
        std::vector<fyzz::native::Pair> foundPairs,lostPairs;
        lower_source_sap_actor_reports_to_sc(
            {},formerPair,sourceActorIds,foundPairs,lostPairs);
        if(!staleFound || lostPairs.size()!=1u ||
           lostPairs.front()!=fyzz::native::Pair{2u,36u})
            throw std::runtime_error(
                "Original source retired handle lost valid GPU lost-pair identity");
        // Source processFoundPairs/processLostPairs clear the original
        // bitmap deltas before a previously removed handle is reused.
        clear_source_sap_single_volume_report_bitmaps_native(manager);
        retire_source_sap_volume_actor_lifetime_native(identity,next);
        if(sourceActorIds[36]!=std::numeric_limits<fyzz::BodyId>::max() ||
           sourceActorIds[2]!=2u)
            throw std::runtime_error(
                "Source GPU fetch did not release only completed removed handles");
        add_source_sap_single_volume_actor_native(
            manager,identity,36u,0.04f,
            physx::Bp::FilterGroup::eSTATICS,&actorA,99u);
        if(sourceActorIds[36]!=99u || !added.boundedTest(36u))
            throw std::runtime_error(
                "Source AABB manager did not admit recycled handle after fetch");
    }
    stream.wait_at_fetch();
    if(aborted || roots.live_roots() || addresses.live_allocations() ||
       pinned.live_roots())
        throw std::runtime_error("Source AABB volume bitmap lifecycle leaked GPU roots");
}

void exercise_persistent_source_aabb_manager_native(Context& context) {
    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback callback(roots,&stream);
    SourcePhysxMappedReportRoots mapped(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* state) noexcept {
        *static_cast<bool*>(state)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        physx::PxgAllocatorDesc allocators(heap,mapped,mapped);
        PxgAabbSourceMembersNative owner(allocators,3u);
        PxgSapSourceMembersNative sap(allocators,8u,2u);
        // The full GPU frame must borrow its actual PhysX AABB owner,
        // not a foreign Sc actor mapping with coincidentally equal size.
        const auto preflight=owner.pre_update(sap.pre_broadphase_buffers());
        const auto originalIdentity=owner.actor_lifetime().volume_to_actor;
        validate_source_sap_owned_aabb_aliases_native(
            owner,preflight,originalIdentity);
        std::vector<fyzz::BodyId> alienIdentity(
            owner.capacity(),std::numeric_limits<fyzz::BodyId>::max());
        bool rejectedAlien=false;
        try {
            validate_source_sap_owned_aabb_aliases_native(
                owner,preflight,alienIdentity);
        }catch(const std::invalid_argument&){rejectedAlien=true;}
        if(!rejectedAlien || stream.submission_stats().upload_batches!=0u)
            throw std::runtime_error(
                "Source GPU frame accepted non-owning Sc identity before DMA");
        if(owner.capacity()<3u ||
           owner.volume_to_actor().size()!=owner.capacity())
            throw std::runtime_error(
                "Source PxgAABBManager original member storage was not initialized");
        alignas(4) physx::PxU32 shapeA=100u,shapeB=200u,shapeC=300u;
        owner.admit(36u,36u,0.04f,physx::Bp::FilterGroup::eSTATICS,&shapeA);
        owner.admit(2u,2u,0.02f,
            physx::Bp::getFilterGroup_Dynamics(99u,false),&shapeB);
        owner.admit(7u,7u,0.03f,physx::Bp::FilterGroup::eSTATICS,&shapeC);
        if(!owner.remove(7u) ||
           owner.volume_to_actor()[7u]!=
               std::numeric_limits<fyzz::BodyId>::max())
            throw std::runtime_error(
                "Canceled source volume retained actor identity in persistent manager");

        using Box=fyzz::native::GpuBroadphaseBox;
        std::array<Box,2u> boxes{{
            {36u,{3.f,0.f,0.f},{1.f,1.f,1.f},{},0.04f,
                Box::Motion::Static},
            {2u,{0.f,0.f,0.f},{1.f,1.f,1.f},{},0.015f,
                Box::Motion::Dynamic}
        }};
        const std::array<physx::Bp::ShapeHandle,2u> handles{36u,2u};
        const physx::Bp::BpFilter filter(false,false);
        const auto first=owner.prepare(boxes,handles,filter);
        if(std::fabs(owner.single_volumes().contact_distances[2]-0.02f)>1.e-6f)
            throw std::runtime_error(
                "Original Sc contact distance was overwritten by stale native admission snapshot");
        if(owner.single_volumes().groups[2]!=
           physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error(
                "Persistent GPU AABB source writer overwrote Sc filter group");
        // ScShapeSimBase::getBPGroup permits DYNAMIC filtering for a
        // kinematic body with forced kinematic notifications.
        auto forcedKinematic=boxes;
        forcedKinematic[1].motion=Box::Motion::Kinematic;
        (void)owner.prepare(forcedKinematic,handles,filter);
        if(owner.single_volumes().groups[2]!=
           physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error(
                "Forced kinematic Sc filtering was overwritten by native motion");
        auto incorrectMotion=boxes;
        incorrectMotion[1].motion=Box::Motion::Static;
        bool mismatchedMotionRejected=false;
        try {
            (void)owner.prepare(incorrectMotion,handles,filter);
        }catch(const std::invalid_argument&){mismatchedMotionRejected=true;}
        if(!mismatchedMotionRejected ||
           owner.single_volumes().groups[2]!=
               physx::Bp::getFilterGroup_Dynamics(99u,false))
            throw std::runtime_error(
                "Source Sc motion changed without original filter readmission");
        const auto originalMinimum=
            owner.source_bounds().getBounds(36u).minimum.x;
        auto invalidActors=boxes;
        invalidActors[0].id=99u;
        invalidActors[0].position.x=123.f;
        bool wrongActorRejected=false;
        try {
            (void)owner.prepare(invalidActors,handles,filter);
        }catch(const std::invalid_argument&){wrongActorRejected=true;}
        if(!wrongActorRejected ||
           owner.source_bounds().getBounds(36u).minimum.x!=originalMinimum ||
           owner.volume_to_actor()[36u]!=36u)
            throw std::runtime_error(
                "Persistent GPU AABB owner accepted wrong Sc actor for source handle");
        if(first.getNumCreatedHandles()!=2u ||
           first.getNumUpdatedHandles()!=0u ||
           first.getNumRemovedHandles()!=0u ||
           !first.getStateChanged() || first.getGpuStateChanged() ||
           std::fabs(owner.source_bounds().getBounds(36u).minimum.x-2.f)>1.e-6f)
            throw std::runtime_error(
                "Persistent PhysX AABB manager did not publish original first frame");
        const auto originalPre=owner.pre_update(sap.pre_broadphase_buffers());
        pre_update_source_sap_aabb_manager_native(first,originalPre,stream);
        stream.wait_at_fetch();
        // The GPU bitmaps may be allocated on this very first upload.
        // Owner-backed fetch must use the POST-allocation device aliases.
        if(!owner.added_report_bitmap().device ||
           !owner.removed_report_bitmap().device ||
           originalPre.added.device.getDevicePtr()!=
               owner.added_report_bitmap().device ||
           originalPre.removed.device.getDevicePtr()!=
               owner.removed_report_bitmap().device)
            throw std::runtime_error(
                "Source GPU frame used stale AABB report bitmap pointers");
        const auto gpu_distances=sap.pre_broadphase_buffers().contact_distances.getDevicePtr();
        const auto admitted_offset=addresses.resolve(
            gpu_distances+2u*sizeof(physx::PxReal),sizeof(physx::PxReal));
        physx::PxReal gpu_offset=0.f;
        context.download(admitted_offset.buffer,admitted_offset.offset,
                         &gpu_offset,sizeof(gpu_offset));
        if(std::fabs(gpu_offset-0.02f)>1.e-6f)
            throw std::runtime_error(
                "Native GPU contact distance disagrees with source AABB admission");
        const auto initial=owner.added_report_bitmap();
        const auto firstWords=addresses.resolve(
            initial.device,2u*sizeof(physx::PxU32));
        std::array<physx::PxU32,2> observed{};
        context.download(firstWords.buffer,firstWords.offset,
                         observed.data(),sizeof(observed));
        if(observed!=std::array<physx::PxU32,2>{1u<<2u,1u<<4u})
            throw std::runtime_error(
                "Persistent source AABB admission bitmap diverged on GPU");

        owner.complete_fetch(first);
        if(owner.source_added().hasAnyBitSet() ||
           owner.source_removed().hasAnyBitSet())
            throw std::runtime_error(
                "Source AABB manager did not retire host report bitmaps");
        owner.mark_changed(2u,0.125f);
        if(owner.remove(36u) ||
           owner.volume_to_actor()[36u]!=36u)
            throw std::runtime_error(
                "Published source AABB removal lost pre-fetch actor identity");
        bool prematureReuse=false;
        try {
            owner.admit(36u,99u,0.04f,
                physx::Bp::FilterGroup::eSTATICS,&shapeC);
        }catch(const std::logic_error&){prematureReuse=true;}
        if(!prematureReuse)
            throw std::runtime_error(
                "Persistent source AABB manager allowed premature handle reuse");
        const std::array<Box,1u> remaining{{
            {2u,{6.f,0.f,0.f},{1.f,1.f,1.f},{},0.02f,
                Box::Motion::Dynamic}
        }};
        const std::array<physx::Bp::ShapeHandle,1u> remainingHandles{2u};
        const auto second=owner.prepare(remaining,remainingHandles,filter);
        if(std::fabs(owner.single_volumes().contact_distances[2]-0.125f)>1.e-6f)
            throw std::runtime_error(
                "Source contact-distance update was undone by stale native motion snapshot");
        if(second.getNumRemovedHandles()!=1u ||
           second.getRemovedHandles()[0]!=36u ||
           second.getNumUpdatedHandles()!=0u ||
           !second.getGpuStateChanged() ||
           std::fabs(owner.source_bounds().getBounds(2u).minimum.x-5.f)>1.e-6f ||
           owner.volume_to_actor()[36u]!=36u)
            throw std::runtime_error(
                "Persistent source manager lost original multi-frame deltas");
        pre_update_source_sap_aabb_manager_native(
            second,owner.pre_update(sap.pre_broadphase_buffers()),stream);
        stream.wait_at_fetch();
        const auto updated_offset=addresses.resolve(
            gpu_distances+2u*sizeof(physx::PxReal),sizeof(physx::PxReal));
        gpu_offset=0.f;
        context.download(updated_offset.buffer,updated_offset.offset,
                         &gpu_offset,sizeof(gpu_offset));
        if(std::fabs(gpu_offset-0.125f)>1.e-6f)
            throw std::runtime_error(
                "Native GPU uploaded stale Sc contact offset after source change");
        const auto removed=owner.removed_report_bitmap();
        const auto secondWords=addresses.resolve(
            removed.device,2u*sizeof(physx::PxU32));
        observed={};
        context.download(secondWords.buffer,secondWords.offset,
                         observed.data(),sizeof(observed));
        if(observed!=std::array<physx::PxU32,2>{0u,1u<<4u})
            throw std::runtime_error(
                "Persistent source GPU removed bitmap lost second-frame retirement");
        owner.complete_fetch(second);
        if(owner.volume_to_actor()[36u]!=
               std::numeric_limits<fyzz::BodyId>::max() ||
           owner.volume_to_actor()[2u]!=2u)
            throw std::runtime_error(
                "Persistent source fetch retired wrong actor identity");
        owner.admit(36u,99u,0.04f,
            physx::Bp::FilterGroup::eSTATICS,&shapeC);
        if(owner.volume_to_actor()[36u]!=99u ||
           !owner.source_added().boundedTest(36u))
            throw std::runtime_error(
                "Persistent source manager could not reuse fetched handle");
    }
    stream.wait_at_fetch();
    if(aborted || roots.live_roots() || addresses.live_allocations() ||
       mapped.live_roots())
        throw std::runtime_error(
            "Persistent source AABB manager leaked original PhysX GPU heap roots");
}

void exercise_source_sc_element_id_lifecycle_native(Context& context) {
    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback callback(roots,&stream);
    SourcePhysxMappedReportRoots mapped(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* state) noexcept {
        *static_cast<bool*>(state)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        physx::PxgAllocatorDesc allocators(heap,mapped,mapped);
        PxgAabbSourceMembersNative manager(allocators);
        PxgSapSourceMembersNative sap(allocators,8u,2u);
        // Sc::Scene owns DIFFERENT original trackers for rigid actor IDs
        // (filtering) and ElementSim IDs (AABB volume handles).
        physx::Sc::ObjectIDTracker scActors,scElements;
        const auto rigidA=scActors.createID();
        const auto rigidB=scActors.createID();
        const auto rigidC=scActors.createID();
        const auto rigidD=scActors.createID();
        const auto rigidE=scActors.createID();

        using Box=fyzz::native::GpuBroadphaseBox;
        // These are Sc actor IDs, NOT Sc::ElementSim IDs. The original
        // ObjectIDTracker alone selects the volume indices (0, 1, 2 ...).
        alignas(4) physx::PxU32 elementA=17u,elementB=18u,
            elementCanceled=19u,elementD=20u,elementE=21u;
        const auto shapeA=SourceScElementLifecycleNative::admit(
            scElements,manager,300u,rigidA,0.04f,
            Box::Motion::Static,&elementA);
        const auto shapeB=SourceScElementLifecycleNative::admit(
            scElements,manager,900u,rigidB,0.02f,
            Box::Motion::Dynamic,&elementB);
        const auto canceled=SourceScElementLifecycleNative::admit(
            scElements,manager,450u,rigidC,0.03f,
            Box::Motion::Static,&elementCanceled);
        if(manager.single_volumes().groups[shapeB]!=
               physx::Bp::getFilterGroup(false,rigidB,false) ||
           manager.single_volumes().groups[shapeB]==
               physx::Bp::getFilterGroup(false,900u,false) ||
           manager.single_volumes().groups[shapeA]!=
               physx::Bp::getFilterGroup(true,rigidA,false))
            throw std::runtime_error(
                "Source Sc actor admission omitted original Bp filter encoding");
        if(shapeA!=0u || shapeB!=1u || canceled!=2u ||
           shapeA==300u || shapeB==900u ||
           !SourceScElementLifecycleNative::remove(
               scElements,manager,canceled) ||
           !scElements.isDeletedID(canceled) ||
           scElements.getDeletedIDCount()!=1u)
            throw std::runtime_error(
                "Source Sc element IDPool did not defer canceled admission");

        std::array<Box,2u> boxes{{
            {300u,{3.f,0.f,0.f},{1.f,1.f,1.f},{},0.04f,
                Box::Motion::Static},
            {900u,{0.f,0.f,0.f},{1.f,1.f,1.f},{},0.02f,
                Box::Motion::Dynamic}
        }};
        const std::array<physx::Bp::ShapeHandle,2u> handles{
            shapeA,shapeB};
        const physx::Bp::BpFilter filter(false,false);
        const auto first=manager.prepare(boxes,handles,filter);
        if(manager.single_volumes().groups[shapeB]!=
               physx::Bp::getFilterGroup(false,rigidB,false) ||
           manager.single_volumes().groups[shapeB]==
               physx::Bp::getFilterGroup(false,900u,false))
            throw std::runtime_error(
                "Sc element volume bounds overwrote the original rigid actor filter group");
        if(first.getNumCreatedHandles()!=2u ||
           first.getCreatedHandles()[0]!=shapeA ||
           first.getCreatedHandles()[1]!=shapeB ||
           manager.volume_to_actor()[shapeA]!=300u ||
           manager.volume_to_actor()[shapeB]!=900u)
            throw std::runtime_error(
                "Sc ElementIDPool IDs were confused with original source actor IDs");
        pre_update_source_sap_aabb_manager_native(
            first,manager.pre_update(sap.pre_broadphase_buffers()),stream);
        stream.wait_at_fetch();
        manager.complete_fetch(first);
        SourceScElementLifecycleNative::post_reports_cleanup(scElements);
        if(scElements.isDeletedID(canceled) ||
           scElements.getDeletedIDCount()!=0u)
            throw std::runtime_error(
                "Sc postReportsCleanup did not release canceled element ID");

        // Original Sc element teardown releases its ID before GPU fetch.
        // ObjectIDTracker defers its recycling through postReportsCleanup.
        if(SourceScElementLifecycleNative::remove(
               scElements,manager,shapeB) ||
           !scElements.isDeletedID(shapeB) ||
           manager.volume_to_actor()[shapeB]!=900u)
            throw std::runtime_error(
                "Published Sc element removal lost pre-fetch volume identity");
        const auto shapeD=SourceScElementLifecycleNative::admit(
            scElements,manager,650u,rigidD,0.035f,
            Box::Motion::Static,&elementD);
        if(shapeD==shapeB || manager.volume_to_actor()[shapeD]!=650u ||
           manager.volume_to_actor()[shapeB]!=900u)
            throw std::runtime_error(
                "Original ObjectIDTracker recycled a pending GPU lost volume");
        manager.mark_changed(shapeA,0.125f);
        std::array<Box,2u> moved{{
            {300u,{6.f,0.f,0.f},{1.f,1.f,1.f},{},0.125f,
                Box::Motion::Static},
            {650u,{9.f,0.f,0.f},{1.f,1.f,1.f},{},0.035f,
                Box::Motion::Static}
        }};
        const std::array<physx::Bp::ShapeHandle,2u> movedHandles{
            shapeA,shapeD};
        const auto second=manager.prepare(moved,movedHandles,filter);
        if(second.getNumRemovedHandles()!=1u ||
           second.getRemovedHandles()[0]!=shapeB ||
           second.getNumCreatedHandles()!=1u ||
           second.getCreatedHandles()[0]!=shapeD ||
           !second.getGpuStateChanged() ||
           std::fabs(manager.source_bounds().getBounds(shapeA).minimum.x-5.f)>1.e-6f)
            throw std::runtime_error(
                "Original Sc shape IDs did not produce expected GPU AABB frame deltas");
        pre_update_source_sap_aabb_manager_native(
            second,manager.pre_update(sap.pre_broadphase_buffers()),stream);
        stream.wait_at_fetch();
        const auto removed=manager.removed_report_bitmap();
        const auto view=addresses.resolve(removed.device,sizeof(physx::PxU32));
        physx::PxU32 gpu_removed=0;
        context.download(view.buffer,view.offset,&gpu_removed,sizeof(gpu_removed));
        if((gpu_removed&(1u<<shapeB))==0u)
            throw std::runtime_error(
                "Source Sc element removal was not uploaded to real GPU bitmap");

        // The fetched lost pair still resolves BOTH actors, even though
        // Sc already released the element ID and VolumeData has reset.
        const std::array<physx::PxgBroadPhasePair,1u> sourceLost{
            physx::PxgBroadPhasePair{shapeA,shapeB}};
        std::vector<fyzz::native::Pair> found,lost;
        lower_source_sap_actor_reports_to_sc(
            {},sourceLost,manager.volume_to_actor(),found,lost);
        if(lost!=std::vector<fyzz::native::Pair>{
               fyzz::native::Pair{300u,900u}})
            throw std::runtime_error(
                "Source GPU lost pair failed with original Sc deferred element ID");
        manager.complete_fetch(second);
        SourceScElementLifecycleNative::post_reports_cleanup(scElements);
        if(manager.volume_to_actor()[shapeB]!=
               std::numeric_limits<fyzz::BodyId>::max())
            throw std::runtime_error(
                "Original Sc element identity was not retired after GPU fetch");

        // Cm::IDPool now owns the released ID and may recycle it. The
        // source AABB manager admits it for a DIFFERENT actor only after
        // the fetched lost report and postReportsCleanup completed.
        const auto shapeE=SourceScElementLifecycleNative::admit(
            scElements,manager,1200u,rigidE,0.05f,
            Box::Motion::Dynamic,&elementE);
        if(shapeE!=shapeB ||
           manager.volume_to_actor()[shapeE]!=1200u ||
           manager.volume_to_actor()[shapeA]!=300u)
            throw std::runtime_error(
                "Original Sc ObjectIDTracker did not recycle fetched source volume");
    }
    stream.wait_at_fetch();
    if(aborted || roots.live_roots() || addresses.live_allocations() ||
       mapped.live_roots())
        throw std::runtime_error(
            "Source Sc element ID lifetime leaked PhysX GPU buddy-heap roots");
}

void exercise_original_sap_source_members_native(Context& context) {
    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback callback(roots,&stream);
    SourcePhysxMappedReportRoots mapped(
        context,addresses,stream,SourcePhysxMappedReportRoots::Mode::ForceStaged);
    bool aborted=false;
    const auto abort=[](void* state) noexcept {
        *static_cast<bool*>(state)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        physx::PxgAllocatorDesc sourceAlloc(heap,mapped,mapped);
        PxgSapSourceMembersNative source(sourceAlloc,4u,2u);
        if(!source.force_update() || source.num_boxes()!=0u)
            throw std::runtime_error(
                "Original PhysX SAP constructor did not initialize source frame ownership");
        const auto foundAddress=mapped.device_address(
            source.mapped_found_actor_pairs(),
            4u*sizeof(physx::PxgBroadPhasePair));
        const auto lostAddress=mapped.device_address(
            source.mapped_lost_actor_pairs(),
            4u*sizeof(physx::PxgBroadPhasePair));
        auto members=source.dma_up_members(nullptr,mapped);
        // Source GPU radix count buffers are allocated once at the original
        // 32 x 16 grid capacity, not per frame or by a parallel allocator.
        const auto count_bytes=sizeof(physx::PxU32)*
            physx::PxgRadixSortKernelGridDim::RADIX_SORT*16u;
        for(physx::PxU32 axis=0u;axis<3u;++axis) {
            const auto& count=members.radix_buffers.radix_counts[axis];
            if(count.getSize()!=count_bytes || !count.getDevicePtr())
                throw std::runtime_error(
                    "Original SAP radix source heap size or pointer changed");
        }
        auto pre=source.pre_broadphase_buffers();
        if(&pre.bounds!=&members.broadphase_buffers.fp_bounds ||
           &members.broadphase_buffers.projection_handles!=
                &source.fetch_histories().projection_handles ||
           members.descriptor_gpu_buffers.broadphase.getDevicePtr()==0u ||
           members.descriptor_gpu_buffers.radix.getSize()<
                6u*sizeof(physx::PxgRadixSortDesc) ||
           members.descriptor_gpu_buffers.radix_without_ranks.getSize()<
                6u*sizeof(physx::PxgRadixSortDesc))
            throw std::runtime_error(
                "Original SAP host views reference different GPU allocations");
        physx::Bp::BpFilter filter(false,false);
        std::array<physx::PxBounds3,2u> bounds{
            physx::PxBounds3{physx::PxVec3{0.f},physx::PxVec3{1.f}},
            physx::PxBounds3{physx::PxVec3{3.f},physx::PxVec3{4.f}}};
        std::array<physx::PxReal,2u> distances{0.02f,0.03f};
        std::array<physx::Bp::FilterGroup::Enum,2u> groups{
            physx::Bp::FilterGroup::eDYNAMICS_BASE,
            physx::Bp::FilterGroup::eSTATICS};
        const std::array<physx::PxU32,2u> created{0u,1u};
        const physx::Bp::BroadPhaseUpdateData first(
            created.data(),2u,nullptr,0u,nullptr,0u,
            bounds.data(),groups.data(),distances.data(),
            nullptr,2u,filter,true,true);
        // Actual PxgAABBManager preBpUpdate_GPU source-owned bitmap and
        // volume buffers, independently indexed by original source handles.
        physx::PxgCudaBuffer addedMap(heap,physx::PxsHeapStats::eBROADPHASE);
        physx::PxgCudaBuffer removedMap(heap,physx::PxsHeapStats::eBROADPHASE);
        physx::PxgCudaBuffer aggregatedMap(heap,physx::PxsHeapStats::eBROADPHASE);
        physx::PxgCudaBuffer changedMap(heap,physx::PxsHeapStats::eBROADPHASE);
        physx::PxgCudaBuffer volumeGpu(heap,physx::PxsHeapStats::eBROADPHASE);
        std::array<physx::PxU32,1u> addedBits{3u},removedBits{0u},
            aggregateBits{0u},changedBits{3u};
        // The original manager may reserve more VolumeData slots than
        // its current broadphase capacity; only the frame's first two
        // records belong to this GPU upload.
        std::array<physx::Bp::VolumeData,3u> volumeData{};
        for(auto& v:volumeData)v.reset();
        const auto manager_view=[&](bool persistent,bool gpu) {
            return SourceSapAabbManagerPreUpdateNative{
                source.pre_broadphase_buffers(),
                {addedMap,addedBits},{removedMap,removedBits},
                {aggregatedMap,aggregateBits},{changedMap,changedBits},
                volumeGpu,volumeData,persistent,gpu,false};
        };
        auto manager=manager_view(true,true);
        // Malformed source volume ownership must fail before ANY of the
        // AABB manager GPU bitmap uploads or heap allocations are queued.
        auto missingVolumes=manager;
        missingVolumes.volume_data={};
        bool invalidPreupdate=false;
        try {pre_update_source_sap_aabb_manager_native(
            first,missingVolumes,stream);}
        catch(const std::invalid_argument&){invalidPreupdate=true;}
        if(!invalidPreupdate || stream.submission_stats().upload_batches!=0u ||
           addedMap.getDevicePtr()!=0u)
            throw std::runtime_error(
                "Invalid source AABB manager frame partially queued GPU work");
        const physx::Bp::BroadPhaseUpdateData missingGroups(
            created.data(),2u,nullptr,0u,nullptr,0u,
            bounds.data(),nullptr,distances.data(),
            nullptr,2u,filter,true,true);
        bool invalidGroups=false;
        try {pre_update_source_sap_aabb_manager_native(
            missingGroups,manager,stream);}
        catch(const std::invalid_argument&){invalidGroups=true;}
        if(!invalidGroups || stream.submission_stats().upload_batches!=0u ||
           addedMap.getDevicePtr()!=0u)
            throw std::runtime_error(
                "Malformed preBroadPhase group array partially queued manager bitmaps");

        pre_update_source_sap_aabb_manager_native(first,manager,stream);
        if(stream.submission_stats().upload_batches!=4u)
            throw std::runtime_error(
                "Original source AABB first-pass GPU transfer ordering changed");
        const auto gpu_manager=source_sap_aabb_descriptor_native(manager);
        auto managedMembers=source.dma_up_members(&gpu_manager,mapped);
        const auto counts=gpu_dma_up_sap_frame_native(first,managedMembers,stream);
        if(counts.current_boxes!=2u || source.num_boxes()!=2u ||
           source.descriptor().numCreatedHandles!=2u ||
           source.descriptor().max_found_lost_pairs!=4u ||
           source.descriptor().foundPairReportMap!=
                reinterpret_cast<physx::PxgBroadPhasePair*>(
                    static_cast<std::uintptr_t>(foundAddress)) ||
           source.descriptor().aabbMngr_addedHandleMap!=
                reinterpret_cast<physx::PxU32*>(addedMap.getDevicePtr()) ||
           source.descriptor().aabbMngr_changedHandleBitMapWordCounts!=1u ||
           source.descriptor().aabbMngr_volumeData!=
                reinterpret_cast<physx::Bp::VolumeData*>(
                    volumeGpu.getDevicePtr()))
            throw std::runtime_error(
                "First original source member frame lost count or mapped report ownership");
        stream.wait_at_fetch();
        const auto observe=[&](physx::PxgCudaBuffer& buffer,auto& data) {
            const auto view=addresses.resolve(buffer.getDevicePtr(),sizeof(data));
            std::remove_cvref_t<decltype(data)> copied{};
            context.download(view.buffer,view.offset,&copied,sizeof(copied));
            if(std::memcmp(&copied,&data,sizeof(data))!=0)
                throw std::runtime_error(
                    "Original AABB manager GPU bitmap/volume upload changed payload");
        };
        observe(addedMap,addedBits);
        observe(removedMap,removedBits);
        observe(aggregatedMap,aggregateBits);
        observe(changedMap,changedBits);
        const auto admittedBytes=sizeof(physx::Bp::VolumeData)*2u;
        if(volumeGpu.getSize()!=admittedBytes)
            throw std::runtime_error(
                "Source AABB volume-data upload included reserved inactive entries");
        const auto gpuVolume=addresses.resolve(
            volumeGpu.getDevicePtr(),admittedBytes);
        std::array<std::byte,sizeof(physx::Bp::VolumeData)*2u> gpuVolumeBytes{};
        context.download(gpuVolume.buffer,gpuVolume.offset,
                         gpuVolumeBytes.data(),admittedBytes);
        if(std::memcmp(gpuVolumeBytes.data(),volumeData.data(),admittedBytes))
            throw std::runtime_error(
                "Source AABB manager volume-data transfer changed original bytes");
        const auto originalOld=source.old_integer_bounds().getDevicePtr();
        const auto originalHistory=source.fetch_histories().projection_handles[0].getDevicePtr();
        const physx::Bp::BroadPhaseUpdateData second(
            nullptr,0u,nullptr,0u,nullptr,0u,
            bounds.data(),groups.data(),distances.data(),
            nullptr,2u,filter,false,false);
        const auto previousUploads=stream.submission_stats().upload_batches;
        auto quietManager=manager_view(false,false);
        pre_update_source_sap_aabb_manager_native(second,quietManager,stream);
        if(stream.submission_stats().upload_batches!=previousUploads)
            throw std::runtime_error(
                "Quiet AABB manager first pass uploaded stale source bitmap history");
        const auto quiet=gpu_dma_up_sap_frame_native(
            second,source.dma_up_members(&gpu_manager,mapped),
            stream);
        stream.wait_at_fetch();
        if(quiet.current_boxes!=2u || quiet.created!=0u ||
           source.old_integer_bounds().getDevicePtr()!=originalOld ||
           source.fetch_histories().projection_handles[0].getDevicePtr()!=originalHistory)
            throw std::runtime_error(
                "Quiet original SAP frame reallocated stable GPU history");
        swap_sap_endpoint_histories_after_fetch_native(source.fetch_histories());
        swap_sap_endpoint_histories_after_fetch_native(source.fetch_histories());
        if(source.fetch_histories().projection_handles[0].getDevicePtr()!=originalHistory)
            throw std::runtime_error(
                "Original owner did not retain fetch-time history identity");

        // A GPU-only update uploads changed bitmap words without replaying
        // the source CPU-bound/group/volume transfers or creation records.
        changedBits[0]=1u;
        auto gpuOnly=manager_view(false,true);
        const auto beforeGpuOnly=stream.submission_stats();
        pre_update_source_sap_aabb_manager_native(second,gpuOnly,stream);
        if(stream.submission_stats().upload_batches!=
               beforeGpuOnly.upload_batches+1u ||
           stream.submission_stats().upload_spans!=
               beforeGpuOnly.upload_spans+1u)
            throw std::runtime_error(
                "GPU-only source AABB update replayed persistent CPU transfers");
        stream.wait_at_fetch();
        observe(changedMap,changedBits);

        auto unsupportedAggregate=manager_view(true,true);
        unsupportedAggregate.has_aggregates=true;
        const auto beforeRejected=stream.submission_stats().upload_batches;
        bool rejectedAggregate=false;
        try {pre_update_source_sap_aabb_manager_native(
            first,unsupportedAggregate,stream);}
        catch(const std::logic_error&){rejectedAggregate=true;}
        if(!rejectedAggregate ||
           stream.submission_stats().upload_batches!=beforeRejected)
            throw std::runtime_error(
                "Unsupported GPU aggregate source state silently entered SAP");
    }
    stream.wait_at_fetch();
    if(aborted || roots.live_roots() || addresses.live_allocations() ||
       mapped.live_roots())
        throw std::runtime_error(
            "Original source SAP member owner leaked native PhysX heap allocations");
}

void exercise_original_sap_update_fetch_lifecycle(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapFetchHistoryBuffers;
    using fyzz::physx_gpu::should_execute_sap_update_after_dma_up_native;
    using fyzz::physx_gpu::swap_sap_integer_bounds_after_pairs_native;
    using fyzz::physx_gpu::swap_sap_endpoint_histories_after_fetch_native;

    // Source mForceUpdate is initialized TRUE by its constructor. This
    // policy is evaluated AFTER gpuDMAUp, including on empty frames.
    physx::Bp::BpFilter filter(false,false);
    bool force=true;
    const physx::Bp::BroadPhaseUpdateData idle(
        nullptr,0u,nullptr,0u,nullptr,0u,
        nullptr,nullptr,nullptr,8u,filter,false,false);
    if(!should_execute_sap_update_after_dma_up_native(idle,force) || force)
        throw std::runtime_error("Source SAP first idle frame must force final GPU pass");
    if(should_execute_sap_update_after_dma_up_native(idle,force) || force)
        throw std::runtime_error("Source SAP second idle frame must skip GPU kernels");

    const std::array<physx::PxU32,1u> one{1u};
    const physx::Bp::BroadPhaseUpdateData created(
        one.data(),1u,nullptr,0u,nullptr,0u,
        nullptr,nullptr,nullptr,8u,filter,false,false);
    if(!should_execute_sap_update_after_dma_up_native(created,force)||!force)
        throw std::runtime_error("Source SAP created handle did not arm forced quiet pass");
    if(!should_execute_sap_update_after_dma_up_native(idle,force)||force ||
       should_execute_sap_update_after_dma_up_native(idle,force))
        throw std::runtime_error("Source SAP final quiet pass was not exactly once");

    const physx::Bp::BroadPhaseUpdateData gpuState(
        nullptr,0u,nullptr,0u,nullptr,0u,
        nullptr,nullptr,nullptr,8u,filter,false,true);
    if(!should_execute_sap_update_after_dma_up_native(gpuState,force)||!force)
        throw std::runtime_error("Source SAP GPU-only change incorrectly skipped kernels");
    const physx::Bp::BroadPhaseUpdateData cpuState(
        nullptr,0u,nullptr,0u,nullptr,0u,
        nullptr,nullptr,nullptr,8u,filter,true,false);
    if(!should_execute_sap_update_after_dma_up_native(cpuState,force)||!force)
        throw std::runtime_error("Source SAP CPU state change incorrectly skipped kernels");
    force=false;
    const physx::Bp::BroadPhaseUpdateData removed(
        nullptr,0u,nullptr,0u,one.data(),1u,
        nullptr,nullptr,nullptr,8u,filter,false,false);
    if(!should_execute_sap_update_after_dma_up_native(removed,force)||!force)
        throw std::runtime_error("Source SAP removed handle incorrectly skipped kernels");
    force=false;
    const physx::Bp::BroadPhaseUpdateData onlyUpdated(
        nullptr,0u,one.data(),1u,nullptr,0u,
        nullptr,nullptr,nullptr,8u,filter,false,false);
    // Upstream uses the created/removed counts and state flags to
    // decide; a bare updated list does not change this policy.
    if(should_execute_sap_update_after_dma_up_native(onlyUpdated,force)||force)
        throw std::runtime_error("Source SAP altered the original updated-only gate");

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        constexpr auto group=physx::PxsHeapStats::eBROADPHASE;
        physx::PxgCudaBuffer oldBounds(heap,group),newBounds(heap,group);
        oldBounds.allocate(64u,PX_FL);
        newBounds.allocate(256u,PX_FL);
        const auto oldAddress=oldBounds.getDevicePtr();
        const auto newAddress=newBounds.getDevicePtr();
        const auto oldSize=oldBounds.getSize();
        const auto newSize=newBounds.getSize();
        std::array<physx::PxU32,4u> oldPattern{13u,29u,41u,57u};
        std::array<physx::PxU32,4u> newPattern{71u,89u,97u,113u};
        stream.queue_uploads({
            {oldAddress,oldPattern.data(),sizeof(oldPattern)},
            {newAddress,newPattern.data(),sizeof(newPattern)}
        });
        stream.wait_at_fetch(); // Fixture's prior completed GPU work.
        swap_sap_integer_bounds_after_pairs_native(newBounds,oldBounds);
        if(newBounds.getDevicePtr()!=oldAddress||newBounds.getSize()!=oldSize ||
           oldBounds.getDevicePtr()!=newAddress||oldBounds.getSize()!=newSize)
            throw std::runtime_error("Source update() integer-bounds swap lost GPU ownership");
        const auto check=[&](const physx::PxgCudaBuffer& owner,
                             const auto& expected) {
            const auto ptr=addresses.resolve(owner.getDevicePtr(),sizeof(expected));
            std::array<physx::PxU32,4u> observed{};
            context.download(ptr.buffer,ptr.offset,observed.data(),sizeof(observed));
            if(observed!=expected)
                throw std::runtime_error("Source integer bounds swap lost existing GPU bytes");
        };
        check(newBounds,oldPattern);
        check(oldBounds,newPattern);

        physx::PxgCudaBufferN<6> handles(heap,group),blockEnd(heap,group);
        physx::PxgCudaBufferN<6> endHist(heap,group),endHandles(heap,group);
        physx::PxgCudaBufferN<6> blockStart(heap,group),startHist(heap,group);
        physx::PxgCudaBufferN<6> startHandles(heap,group);
        const std::array<physx::PxgCudaBufferN<6>*,7> owners{
            &handles,&blockEnd,&endHist,&endHandles,
            &blockStart,&startHist,&startHandles};
        std::array<std::array<CUdeviceptr,6>,7> initialPointers{};
        std::array<std::array<physx::PxU64,6>,7> initialSizes{};
        for(std::size_t kind=0;kind<owners.size();++kind)
            for(physx::PxU32 i=0;i<6u;++i) {
                auto& buffer=(*owners[kind])[i];
                buffer.allocate(32u + 24u*i + 8u*kind,PX_FL);
                initialPointers[kind][i]=buffer.getDevicePtr();
                initialSizes[kind][i]=buffer.getSize();
            }
        SourceSapFetchHistoryBuffers source{
            handles,blockEnd,endHist,endHandles,blockStart,startHist,startHandles};
        // Completion/readback belongs to gpuDMABack; fetch swaps only
        // the 7x3 original buffer ownership pairs, NEVER copies memory.
        swap_sap_endpoint_histories_after_fetch_native(source);
        for(std::size_t kind=0;kind<owners.size();++kind)
            for(physx::PxU32 i=0;i<6u;++i) {
                const physx::PxU32 opposite=i<3u ? i+3u : i-3u;
                const auto& buffer=(*owners[kind])[i];
                if(buffer.getDevicePtr()!=initialPointers[kind][opposite] ||
                   buffer.getSize()!=initialSizes[kind][opposite])
                    throw std::runtime_error("Source fetch endpoint/history swap mapping changed");
            }
        swap_sap_endpoint_histories_after_fetch_native(source);
        for(std::size_t kind=0;kind<owners.size();++kind)
            for(physx::PxU32 i=0;i<6u;++i)
                if((*owners[kind])[i].getDevicePtr()!=initialPointers[kind][i] ||
                   (*owners[kind])[i].getSize()!=initialSizes[kind][i])
                    throw std::runtime_error("Repeated source fetch swap did not restore owners");
        // Multi-frame fetch against the real seven PhysX owners, not
        // a separate BufferId role table. Each of the 42 heap allocations
        // contains its own marker and size across frame rotations.
        physx::PxgCudaBuffer gpuDescriptor(heap,group);
        physx::PxgCudaBuffer mappedFound(heap,group),mappedLost(heap,group);
        gpuDescriptor.allocate(sizeof(physx::PxgBroadPhaseDesc),PX_FL);
        mappedFound.allocate(2u*sizeof(physx::PxgBroadPhasePair),PX_FL);
        mappedLost.allocate(2u*sizeof(physx::PxgBroadPhasePair),PX_FL);
        const auto descriptorAddress=gpuDescriptor.getDevicePtr();
        physx::PxgBroadPhaseDesc gpu{};
        gpu.max_found_lost_pairs=2u;
        gpu.sharedFoundPairIndex=1u;
        gpu.sharedLostPairIndex=1u;
        gpu.foundPairReportMap=reinterpret_cast<physx::PxgBroadPhasePair*>(
            static_cast<std::uintptr_t>(mappedFound.getDevicePtr()));
        gpu.lostPairReportMap=reinterpret_cast<physx::PxgBroadPhasePair*>(
            static_cast<std::uintptr_t>(mappedLost.getDevicePtr()));
        std::array<physx::PxgBroadPhasePair,2> foundGPU{
            physx::PxgBroadPhasePair{3u,8u},
            physx::PxgBroadPhasePair{11u,19u}};
        std::array<physx::PxgBroadPhasePair,2> lostGPU{
            physx::PxgBroadPhasePair{2u,7u},
            physx::PxgBroadPhasePair{23u,29u}};
        std::array<physx::PxgBroadPhasePair,2> foundCPU{},lostCPU{};
        physx::PxgBroadPhaseDesc host{};
        std::array<std::array<physx::PxU32,6>,7> markers{};
        std::vector<fyzz::physx_gpu::SourceSapUpload> uploads;
        uploads.reserve(45u);
        for(std::size_t kind=0u;kind<owners.size();++kind)
            for(physx::PxU32 i=0;i<6u;++i) {
                markers[kind][i]=
                    static_cast<physx::PxU32>(100000u+kind*101u+i*17u);
                uploads.push_back({initialPointers[kind][i],
                                   &markers[kind][i],sizeof(physx::PxU32)});
            }
        uploads.push_back({descriptorAddress,&gpu,sizeof(gpu)});
        uploads.push_back({mappedFound.getDevicePtr(),
                           foundGPU.data(),sizeof(foundGPU)});
        uploads.push_back({mappedLost.getDevicePtr(),
                           lostGPU.data(),sizeof(lostGPU)});
        stream.queue_uploads(uploads);
        stream.wait_at_fetch();
        const auto check_histories=[&](bool rotated) {
            for(std::size_t kind=0u;kind<owners.size();++kind)
                for(physx::PxU32 i=0;i<6u;++i) {
                    const auto origin=rotated
                        ? (i<3u?i+3u:i-3u):i;
                    const auto& owner=(*owners[kind])[i];
                    if(owner.getDevicePtr()!=initialPointers[kind][origin] ||
                       owner.getSize()!=initialSizes[kind][origin])
                        throw std::runtime_error(
                            "Original source SAP multi-frame fetch lost PhysX heap owner identity");
                    const auto view=addresses.resolve(owner.getDevicePtr(),4u);
                    physx::PxU32 value=0u;
                    context.download(view.buffer,view.offset,&value,sizeof(value));
                    if(value!=markers[kind][origin])
                        throw std::runtime_error(
                            "Original source SAP multi-frame fetch corrupted GPU history");
                }
        };
        // First frame publishes the source descriptor, then rotates
        // ORIGINAL owners before the AABB-manager found/lost callbacks.
        const auto first=fyzz::physx_gpu::fetch_source_sap_results(
            context,addresses,descriptorAddress,host,
            {foundCPU.data(),foundCPU.size()},
            {lostCPU.data(),lostCPU.size()},false,source);
        if(first.found_actors!=1u||first.lost_actors!=1u ||
           first.pairs_needed!=1u||
           host.max_found_lost_pairs!=gpu.max_found_lost_pairs ||
           foundCPU[0].mVolA!=3u||foundCPU[0].mVolB!=8u ||
           lostCPU[0].mVolA!=2u||lostCPU[0].mVolB!=7u)
            throw std::runtime_error(
                "Original SAP owner fetch failed GPU actor-pair publication");
        check_histories(true);

        // A malformed second-frame descriptor must not mutate the
        // previous successfully published descriptor OR any GPU owner.
        auto corrupt=gpu;
        corrupt.sharedFoundPairIndex=1u;
        corrupt.sharedFoundAggPairIndex=2u;
        stream.queue_uploads({{descriptorAddress,&corrupt,sizeof(corrupt)}});
        stream.wait_at_fetch();
        bool invalidDescriptor=false;
        try {
            (void)fyzz::physx_gpu::fetch_source_sap_results(
                context,addresses,descriptorAddress,host,
                {foundCPU.data(),foundCPU.size()},
                {lostCPU.data(),lostCPU.size()},false,source);
        }catch(const std::invalid_argument&){invalidDescriptor=true;}
        if(!invalidDescriptor || host.sharedFoundPairIndex!=1u ||
           foundCPU[0].mVolA!=3u || lostCPU[0].mVolA!=2u)
            throw std::runtime_error(
                "Failed source SAP fetch partially published descriptor");
        check_histories(true);

        // Retrying that frame after its original GPU result is repaired
        // must rotate the seven histories once, not twice.
        stream.queue_uploads({{descriptorAddress,&gpu,sizeof(gpu)}});
        stream.wait_at_fetch();
        (void)fyzz::physx_gpu::fetch_source_sap_results(
            context,addresses,descriptorAddress,host,
            {foundCPU.data(),foundCPU.size()},
            {lostCPU.data(),lostCPU.size()},false,source);
        check_histories(false);

        // PxgCudaBroadPhaseSap::fetchBroadPhaseResults also rotates
        // histories on early-return/quiet frames, even with no GPU work.
        // Corrupted late lost mapping must not publish found reports,
        // the host descriptor, or swap any of the original buffers.
        auto invalidLost=gpu;
        invalidLost.lostPairReportMap=reinterpret_cast<physx::PxgBroadPhasePair*>(
            static_cast<std::uintptr_t>(0x1234u));
        stream.queue_uploads({{descriptorAddress,&invalidLost,sizeof(invalidLost)}});
        stream.wait_at_fetch();
        foundCPU[0]=physx::PxgBroadPhasePair{31u,37u};
        bool invalidPairSource=false;
        try {
            (void)fyzz::physx_gpu::fetch_source_sap_results(
                context,addresses,descriptorAddress,host,
                {foundCPU.data(),foundCPU.size()},
                {lostCPU.data(),lostCPU.size()},false,source);
        }catch(const std::out_of_range&){invalidPairSource=true;}
        if(!invalidPairSource || foundCPU[0].mVolA!=31u ||
           host.sharedFoundPairIndex!=1u)
            throw std::runtime_error(
                "Late corrupt source lost-pair mapping partially published found pairs");
        check_histories(false);

        // Sc admission must precede source fetch history rotation.
        // A valid GPU report with a retired handle is not a valid Sc pair.
        // Keep using the ACTUAL original mapped pair arrays and all 42
        // PhysX heap allocations; no test-only replacement frame buffers.
        std::array<fyzz::BodyId,24u> volumeToActor{};
        volumeToActor.fill(std::numeric_limits<fyzz::BodyId>::max());
        volumeToActor[3u]=12u;
        volumeToActor[8u]=4u;
        volumeToActor[2u]=8u;
        volumeToActor[7u]=5u;
        auto invalidMap=volumeToActor;
        invalidMap[8u]=std::numeric_limits<fyzz::BodyId>::max();
        std::vector<fyzz::native::Pair> scFound{{1u,2u}};
        std::vector<fyzz::native::Pair> scLost{{3u,4u}};
        stream.queue_uploads({{descriptorAddress,&gpu,sizeof(gpu)}});
        stream.wait_at_fetch();
        const auto publish_to_sc=[&](
            std::span<const fyzz::BodyId> map) {
            return [&,map](const fyzz::physx_gpu::SourceSapFetchedCounts& counts) {
                fyzz::physx_gpu::lower_source_sap_actor_reports_to_sc(
                    std::span<const physx::PxgBroadPhasePair>(
                        foundCPU.data(),counts.found_actors),
                    std::span<const physx::PxgBroadPhasePair>(
                        lostCPU.data(),counts.lost_actors),
                    map,scFound,scLost);
            };
        };
        bool rejectedActor=false;
        try {
            (void)fyzz::physx_gpu::fetch_source_sap_results_native(
                context,addresses,descriptorAddress,host,
                {foundCPU.data(),foundCPU.size()},
                {lostCPU.data(),lostCPU.size()},false,
                [source]() noexcept {
                    fyzz::physx_gpu::swap_sap_endpoint_histories_after_fetch_native(source);
                },nullptr,publish_to_sc(invalidMap));
        }catch(const std::out_of_range&){rejectedActor=true;}
        if(!rejectedActor || host.sharedFoundPairIndex!=1u ||
           scFound!=std::vector<fyzz::native::Pair>{{1u,2u}} ||
           scLost!=std::vector<fyzz::native::Pair>{{3u,4u}})
            throw std::runtime_error(
                "Rejected Sc volume handle partially committed a PhysX SAP frame");
        check_histories(false);

        // Repair the original volume-to-actor mapping and retry the SAME
        // GPU frame. Only this successful admission rotates source owners.
        const auto admitted=fyzz::physx_gpu::fetch_source_sap_results_native(
            context,addresses,descriptorAddress,host,
            {foundCPU.data(),foundCPU.size()},
            {lostCPU.data(),lostCPU.size()},false,
            [source]() noexcept {
                fyzz::physx_gpu::swap_sap_endpoint_histories_after_fetch_native(source);
            },nullptr,publish_to_sc(volumeToActor));
        if(admitted.found_actors!=1u || admitted.lost_actors!=1u ||
           scFound!=std::vector<fyzz::native::Pair>{{4u,12u}} ||
           scLost!=std::vector<fyzz::native::Pair>{{5u,8u}})
            throw std::runtime_error(
                "Original GPU fetch did not publish source volume reports to Sc");
        check_histories(true);

        auto quietGpu=gpu;
        quietGpu.sharedFoundPairIndex=0u;
        quietGpu.sharedLostPairIndex=0u;
        stream.queue_uploads({{descriptorAddress,&quietGpu,sizeof(quietGpu)}});
        stream.wait_at_fetch();
        const auto quiet=fyzz::physx_gpu::fetch_source_sap_results(
            context,addresses,descriptorAddress,host,{},{},false,source);
        if(quiet.found_actors||quiet.lost_actors)
            throw std::runtime_error("Quiet source fetch published stale actor-pair reports");
        check_histories(false);
        (void)fyzz::physx_gpu::fetch_source_sap_results(
            context,addresses,descriptorAddress,host,{},{},true,source);
        check_histories(true);
        if(!stream.commands().empty())
            throw std::runtime_error("Source SAP update/fetch ownership swaps queued GPU commands");
    }
    if(aborted||roots.live_roots()||addresses.live_allocations())
        throw std::runtime_error("PhysX SAP update/fetch source GPU heap leak");
}

void exercise_original_sap_radix_descriptors_native(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapRadixDescriptorBuffers;
    using fyzz::physx_gpu::prepare_sap_radix_descriptors_native;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxHeapRoots roots(context,addresses);
    SourcePhysxDeviceRootCallback callback(roots,&stream);
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        constexpr auto group=physx::PxsHeapStats::eBROADPHASE;
        physx::PxgCudaBufferN<3> projections(heap,group),ranks(heap,group);
        physx::PxgCudaBufferN<3> tempKeys(heap,group),tempHandles(heap,group);
        physx::PxgCudaBufferN<3> radixCounts(heap,group);
        physx::PxgCudaBufferN<6> boxHandles(heap,group);
        physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> gpuNormal(heap,group);
        physx::PxgTypedCudaBuffer<physx::PxgRadixSortDesc> gpuWOR(heap,group);
        gpuNormal.allocateElements(6u,PX_FL);
        gpuWOR.allocateElements(6u,PX_FL);

        SourceSapRadixDescriptorBuffers source{
            projections,ranks,tempKeys,tempHandles,radixCounts,boxHandles};
        std::array<physx::PxgRadixSortDesc,6u> normal{},withoutRanks{};
        for (std::size_t i=0;i<6u;++i) {
            normal[i].count=static_cast<physx::PxU32>(100u+i);
            withoutRanks[i].count=static_cast<physx::PxU32>(200u+i);
        }
        bool rejected=false;
        try {prepare_sap_radix_descriptors_native(nullptr,withoutRanks.data(),source);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected)
            throw std::runtime_error("Original SAP radix descriptor allowed null input");

        // Original constructor allocates radix-count buffers once; the
        // history arrays come from the source gpuDMAUp three-axis loop.
        for(physx::PxU32 i=0;i<3u;++i) {
            projections[i].allocate(16u,PX_FL);
            ranks[i].allocate(16u,PX_FL);
            tempKeys[i].allocate(16u,PX_FL);
            tempHandles[i].allocate(16u,PX_FL);
            radixCounts[i].allocate(physx::PxgRadixSortKernelGridDim::RADIX_SORT * 16u * sizeof(physx::PxU32),PX_FL);
        }
        for(physx::PxU32 i=0;i<6u;++i)
            boxHandles[i].allocate(16u,PX_FL);

        auto check=[&]() {
            for (physx::PxU32 i=0;i<3u;++i) {
                const auto k=reinterpret_cast<physx::PxU32*>(projections[i].getDevicePtr());
                const auto r=reinterpret_cast<physx::PxU32*>(ranks[i].getDevicePtr());
                const auto tk=reinterpret_cast<physx::PxU32*>(tempKeys[i].getDevicePtr());
                const auto tr=reinterpret_cast<physx::PxU32*>(tempHandles[i].getDevicePtr());
                const auto counts=reinterpret_cast<physx::PxU32*>(radixCounts[i].getDevicePtr());
                const auto h=reinterpret_cast<physx::PxU32*>(boxHandles[i].getDevicePtr());
                const auto j=i+3u;
                if(normal[i].inputKeys!=k || normal[i].inputRanks!=r ||
                   normal[i].outputKeys!=tk || normal[i].outputRanks!=tr ||
                   normal[i].radixBlockCounts!=counts ||
                   normal[j].inputKeys!=tk || normal[j].inputRanks!=tr ||
                   normal[j].outputKeys!=k || normal[j].outputRanks!=r ||
                   normal[j].radixBlockCounts!=counts ||
                   withoutRanks[i].inputKeys!=k || withoutRanks[i].inputRanks!=h ||
                   withoutRanks[i].outputKeys!=tk || withoutRanks[i].outputRanks!=tr ||
                   withoutRanks[i].radixBlockCounts!=counts ||
                   withoutRanks[j].inputKeys!=tk || withoutRanks[j].inputRanks!=tr ||
                   withoutRanks[j].outputKeys!=k || withoutRanks[j].outputRanks!=h ||
                   withoutRanks[j].radixBlockCounts!=counts)
                    throw std::runtime_error("Original SAP radix forward/reverse/WOR pointer wiring changed");
                if(normal[i].count!=100u+i || normal[j].count!=100u+j ||
                   withoutRanks[i].count!=200u+i ||
                   withoutRanks[j].count!=200u+j)
                    throw std::runtime_error("Source radix descriptor count was unexpectedly overwritten");
            }
        };

        prepare_sap_radix_descriptors_native(normal.data(),withoutRanks.data(),source);
        check();
        // Source expects source GPU pointer rebinding after history buffer
        // growth; the allocator is allowed to reuse raw pointers.
        for(physx::PxU32 i=0;i<3u;++i) {
            projections[i].allocate(128u,PX_FL);
            ranks[i].allocate(128u,PX_FL);
            tempKeys[i].allocate(128u,PX_FL);
            tempHandles[i].allocate(128u,PX_FL);
            boxHandles[i].allocate(128u,PX_FL);
        }
        prepare_sap_radix_descriptors_native(normal.data(),withoutRanks.data(),source);
        check();

        // These two GPU descriptors are source bitwise records, not a
        // reinterpretation of host-side Context/Vulkan/D3D12 objects.
        // The actual gpuDMAUp upload order additionally requires the
        // broadphase descriptor FIRST; here we test only raw GPU ABI.
        stream.queue_uploads({
            {gpuNormal.getDevicePtr(),normal.data(),sizeof(normal)},
            {gpuWOR.getDevicePtr(),withoutRanks.data(),sizeof(withoutRanks)}
        });
        // Upload batches snapshot source host bytes at enqueue.
        const auto savedNormal=normal;
        const auto savedWOR=withoutRanks;
        for(auto& x:normal)x.inputKeys=nullptr;
        for(auto& x:withoutRanks)x.inputRanks=nullptr;
        stream.wait_at_fetch();
        std::array<physx::PxgRadixSortDesc,6u> actualNormal{},actualWOR{};
        const auto read=[&](const physx::PxgCudaBuffer& owner,auto& output) {
            const auto resolved=addresses.resolve(owner.getDevicePtr(),sizeof(output));
            context.download(resolved.buffer,resolved.offset,output.data(),sizeof(output));
        };
        read(gpuNormal,actualNormal);
        read(gpuWOR,actualWOR);
        for(std::size_t i=0;i<6u;++i) {
            const auto assertMatch=[&](const physx::PxgRadixSortDesc& expected,
                                      const physx::PxgRadixSortDesc& actual){
                if(expected.inputKeys!=actual.inputKeys ||
                   expected.inputRanks!=actual.inputRanks ||
                   expected.outputKeys!=actual.outputKeys ||
                   expected.outputRanks!=actual.outputRanks ||
                   expected.radixBlockCounts!=actual.radixBlockCounts ||
                   expected.count!=actual.count)
                    throw std::runtime_error("Source radix GPU descriptor upload changed ABI or host snapshot");
            };
            assertMatch(savedNormal[i],actualNormal[i]);
            assertMatch(savedWOR[i],actualWOR[i]);
        }
    }
    if(aborted||roots.live_roots()||addresses.live_allocations())
        throw std::runtime_error("Original PhysX radix descriptor test leaked GPU heap roots");
}

void exercise_original_sap_post_axis_buffers_native(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapPostAxisBuffers;
    using fyzz::physx_gpu::prepare_gpu_dma_up_counts_native;
    using fyzz::physx_gpu::allocate_sap_post_axis_buffers_native;
    using fyzz::physx_gpu::allocate_sap_new_integer_bounds_native;
    using fyzz::physx_gpu::allocate_gpu_dma_up_handles_native;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    physx::Bp::BpFilter filter(false,false);
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        constexpr auto group=physx::PxsHeapStats::eBROADPHASE;
        physx::PxgCudaBufferN<2> aggregateBlocks(heap,group),actorBlocks(heap,group);
        physx::PxgCudaBuffer active(heap,group),start(heap,group);
        physx::PxgCudaBuffer orderedActive(heap,group),orderedStart(heap,group);
        physx::PxgCudaBuffer overlap(heap,group),blockOverlap(heap,group);
        physx::PxgCudaBuffer handleRegions(heap,group),regionRange(heap,group);
        physx::PxgCudaBuffer startAccum(heap,group),blockStartAccum(heap,group);
        physx::PxgCudaBuffer regionAccum(heap,group),blockRegionAccum(heap,group);
        physx::PxgCudaBuffer newBounds(heap,group);
        physx::PxgTypedCudaBuffer<physx::PxU32> created(
            heap,physx::PxsHeapStats::eBROADPHASE);
        physx::PxgTypedCudaBuffer<physx::PxU32> removed(
            heap,physx::PxsHeapStats::eBROADPHASE);
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        physx::PxgTypedCudaBuffer<physx::PxU32> updated(
            heap,physx::PxsHeapStats::eBROADPHASE);
#endif
        SourceSapPostAxisBuffers buffers{
            aggregateBlocks,actorBlocks,
            active,start,orderedActive,orderedStart,
            overlap,blockOverlap,handleRegions,regionRange,
            startAccum,blockStartAccum,regionAccum,blockRegionAccum};
        const std::array<physx::PxgCudaBuffer*,12> regions{
            &active,&start,&orderedActive,&orderedStart,&overlap,
            &blockOverlap,&handleRegions,&regionRange,
            &startAccum,&blockStartAccum,&regionAccum,&blockRegionAccum};

        physx::PxU32 numBoxes=0,capacity=0,createdSize=0,removedSize=0;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        physx::PxU32 updatedSize=0;
        fyzz::physx_gpu::SourceSapGpuDMAUpCountMembers state{
            numBoxes,capacity,createdSize,removedSize,updatedSize};
        fyzz::physx_gpu::SourceSapGpuDMAUpHandleBuffers handles{created,removed,updated};
#else
        fyzz::physx_gpu::SourceSapGpuDMAUpCountMembers state{
            numBoxes,capacity,createdSize,removedSize};
        fyzz::physx_gpu::SourceSapGpuDMAUpHandleBuffers handles{created,removed};
#endif
        const auto required=[&](const fyzz::physx_gpu::SourceSapGpuDMAUpCounts& c) {
            const std::uint64_t regionSlots=
                (std::uint64_t(c.projections)*64u+3u)&~std::uint64_t{3u};
            return std::array<std::uint64_t,12>{
                regionSlots*sizeof(int), regionSlots*sizeof(int),
                regionSlots*sizeof(int), regionSlots*sizeof(int),
                64u*std::uint64_t(c.current_boxes)*sizeof(physx::PxU64),
                32u*sizeof(physx::PxU64),
                64u*std::uint64_t(c.current_boxes)*8u,
                std::uint64_t(c.capacity)*32u,
                std::uint64_t(c.projections)*sizeof(int),
                32u*sizeof(int),
                std::uint64_t(c.projections)*sizeof(int),
                32u*sizeof(int)};
        };
        const auto verify=[&](const fyzz::physx_gpu::SourceSapGpuDMAUpCounts& c) {
            const auto req=required(c);
            for(std::size_t i=0;i<regions.size();++i)
                if(regions[i]->getSize()<req[i])
                    throw std::runtime_error("Original SAP post-axis GPU buffer too small");
            for(physx::PxU32 i=0;i<2u;++i)
                if(aggregateBlocks[i].getSize()<32u*sizeof(physx::PxU32) ||
                   actorBlocks[i].getSize()<32u*sizeof(physx::PxU32))
                    throw std::runtime_error("Source GPU report block size changed");
            if(newBounds.getSize()<std::uint64_t(c.capacity)*24u)
                throw std::runtime_error("Original new integer AABB buffer too small");
        };
        const auto allocate=[&](const fyzz::physx_gpu::SourceSapGpuDMAUpCounts& c) {
            // The original sequence after ALL three completed axes:
            // report and region scratch, changed-handle buffers, and only
            // then new integer bounds; descriptor construction comes next.
            allocate_sap_post_axis_buffers_native(c,buffers);
            allocate_gpu_dma_up_handles_native(c,handles);
            allocate_sap_new_integer_bounds_native(c,newBounds);
            verify(c);
        };
        const std::array<physx::PxU32,2> initial{0u,1u};
        const physx::Bp::BroadPhaseUpdateData first(
            initial.data(),2u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,2u,filter,true,true);
        const auto a=prepare_gpu_dma_up_counts_native(first,state);
        allocate(a);
        if(a.current_boxes!=2u || a.projections!=4u ||
           a.padded_projections!=4u || numBoxes!=2u)
            throw std::runtime_error("Original SAP post-axis first frame sizing wrong");
        std::array<CUdeviceptr,12> priorPointers{};
        std::array<physx::PxU64,12> priorCapacities{};
        for(std::size_t i=0;i<regions.size();++i) {
            priorPointers[i]=regions[i]->getDevicePtr();
            priorCapacities[i]=regions[i]->getSize();
        }
        const auto firstReport=aggregateBlocks[0].getDevicePtr();
        const auto oldBoundsSize=newBounds.getSize();

        // A second frame grows selected source buffers, not the fixed-grid
        // report/block scratch. Source non-copying allocate() releases the
        // old allocation BEFORE requesting the larger one, so the buddy
        // heap is permitted to reuse the very same raw device address.
        // Test source capacity growth, NOT a fictitious relocation rule.
        const std::array<physx::PxU32,3> more{2u,3u,4u};
        const physx::Bp::BroadPhaseUpdateData second(
            more.data(),3u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,6u,filter,true,true);
        const auto b=prepare_gpu_dma_up_counts_native(second,state);
        allocate(b);
        if(b.previous_boxes!=2u||b.current_boxes!=5u||
           b.projections!=10u||b.padded_projections!=12u||
           aggregateBlocks[0].getDevicePtr()!=firstReport ||
           newBounds.getSize()!=std::max(
               std::uint64_t(b.capacity)*24u, std::uint64_t(oldBoundsSize)*2u))
            throw std::runtime_error("Original SAP post-axis growth policy changed");
        const auto after=required(b);
        std::size_t grew=0, stayed=0;
        for(std::size_t i=0;i<regions.size();++i) {
            const auto previous=priorCapacities[i];
            const bool grows=after[i]>previous;
            const auto expectedCapacity=grows
                ? std::max(after[i], std::uint64_t(previous)*2u)
                : std::uint64_t(previous);
            if(regions[i]->getSize()!=expectedCapacity)
                throw std::runtime_error("SAP post-axis native buffer capacity growth changed");
            // A non-growing buffer MUST retain its address. When growth
            // occurs, reallocating in-place or at a new address are both
            // legal allocator outcomes; only its new capacity matters.
            if(!grows && regions[i]->getDevicePtr()!=priorPointers[i])
                throw std::runtime_error("No-growth PhysX post-axis buffer relocated");
            grows ? ++grew : ++stayed;
        }
        if(!grew || !stayed)
            throw std::runtime_error("SAP post-axis test missed growing or fixed buffers");

        // Source's zero-change frame preserves all allocation identities;
        // it does not recreate region scratch on every frame.
        const physx::Bp::BroadPhaseUpdateData stable(
            nullptr,0u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,6u,filter,false,false);
        const auto c=prepare_gpu_dma_up_counts_native(stable,state);
        std::array<CUdeviceptr,12> stablePointers{};
        for(std::size_t i=0;i<regions.size();++i)
            stablePointers[i]=regions[i]->getDevicePtr();
        const auto stableNew=newBounds.getDevicePtr();
        allocate(c);
        if(newBounds.getDevicePtr()!=stableNew ||
           aggregateBlocks[0].getDevicePtr()!=firstReport)
            throw std::runtime_error("Unchanged SAP post-axis allocation moved device storage");
        for(std::size_t i=0;i<regions.size();++i)
            if(regions[i]->getDevicePtr()!=stablePointers[i])
                throw std::runtime_error("Unchanged SAP region buffer was needlessly reallocated");

        // Region count computation is stored in PxU32 in source PhysX:
        // reject an impossible projection count BEFORE touching the GPU.
        auto overflow=c;
        overflow.projections=std::numeric_limits<physx::PxU32>::max();
        bool rejected=false;
        try {allocate_sap_post_axis_buffers_native(overflow,buffers);}
        catch(const std::overflow_error&) {rejected=true;}
        if(!rejected || active.getDevicePtr()!=stablePointers[0] ||
           !stream.commands().empty())
            throw std::runtime_error("Overflowed SAP region sizing modified GPU heap or stream");
    }
    if(aborted||roots.live_roots()||addresses.live_allocations())
        throw std::runtime_error("Source post-axis GPU allocations leaked original PhysX roots");
}

void exercise_original_sap_history_growth_native(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapHistoryAxisBuffers;
    using fyzz::physx_gpu::SourceSapHistoryAxisRemainderBuffers;
    using fyzz::physx_gpu::prepare_gpu_dma_up_counts_native;
    using fyzz::physx_gpu::grow_sap_old_bounds_native;
    using fyzz::physx_gpu::grow_sap_history_axis_prefix_native;
    using fyzz::physx_gpu::grow_sap_history_axis_remainder_native;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    physx::Bp::BpFilter filter(false,false);
    {
        physx::PxgHeapMemoryAllocator heap(16384u,callback,abort,&aborted);
        constexpr int group=physx::PxsHeapStats::eBROADPHASE;
        physx::PxgCudaBuffer oldBounds(heap,group);
        physx::PxgCudaBufferN<3> sap(heap,group),newSap(heap,group);
        physx::PxgCudaBufferN<3> projections(heap,group),ranks(heap,group);
        physx::PxgCudaBufferN<3> tempKeys(heap,group),tempRanks(heap,group);
        physx::PxgCudaBufferN<6> endpointHandles(heap,group),endHistogram(heap,group);
        physx::PxgCudaBufferN<6> blockEndHistogram(heap,group),endHandles(heap,group);
        physx::PxgCudaBufferN<6> startHistogram(heap,group),blockStartHistogram(heap,group);
        physx::PxgCudaBufferN<6> startHandles(heap,group);
        physx::PxgCudaBufferN<3> incremental(heap,group),blockIncremental(heap,group);
        // Original PxgCudaBroadPhaseSap.h declares N<6>, though only
        // the first three indices are used for x/y/z per-axis totals.
        physx::PxgCudaBufferN<6> totalEndHistogram(heap,group),blockTotalEndHistogram(heap,group);
        SourceSapHistoryAxisRemainderBuffers rest{
            endpointHandles,endHistogram,blockEndHistogram,endHandles,
            startHistogram,blockStartHistogram,startHandles,
            incremental,blockIncremental,totalEndHistogram,blockTotalEndHistogram};
        const std::array<physx::PxgCudaBufferN<6>*,7u> histories{
            &endpointHandles,&endHistogram,&blockEndHistogram,&endHandles,
            &startHistogram,&blockStartHistogram,&startHandles};

        physx::PxU32 numBoxes=0,capacity=0,createdSize=0,removedSize=0;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        physx::PxU32 updatedSize=0;
        fyzz::physx_gpu::SourceSapGpuDMAUpCountMembers members{
            numBoxes,capacity,createdSize,removedSize,updatedSize};
#else
        fyzz::physx_gpu::SourceSapGpuDMAUpCountMembers members{
            numBoxes,capacity,createdSize,removedSize};
#endif
        auto axis=[&](physx::PxU32 i) -> SourceSapHistoryAxisBuffers {
            return {sap[i],newSap[i],projections[i],
                    ranks[i],tempKeys[i],tempRanks[i]};
        };
        auto upload=[&](const physx::PxgCudaBuffer& owner,
                        const void* data,std::size_t bytes) {
            stream.queue_uploads({{owner.getDevicePtr(),data,bytes}});
        };
        auto check=[&](const physx::PxgCudaBuffer& owner,
                       const void* expected,std::size_t bytes,
                       const char* error) {
            if(owner.getSize()<bytes)
                throw std::runtime_error("PhysX SAP native history capacity too small");
            const auto r=addresses.resolve(owner.getDevicePtr(),bytes);
            std::vector<std::uint8_t> actual(bytes);
            context.download(r.buffer,r.offset,actual.data(),bytes);
            if(std::memcmp(actual.data(),expected,bytes))
                throw std::runtime_error(error);
        };

        // Initial frame: 2 boxes, four already padded projections. Pattern
        // writes are submitted on the SAME original source stream as the
        // later DtoD growth, without first waiting for the GPU.
        const std::array<physx::PxU32,2> firstCreated{1u,3u};
        const physx::Bp::BroadPhaseUpdateData first(
            firstCreated.data(),2u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,2u,filter,true,true);
        const auto a=prepare_gpu_dma_up_counts_native(first,members);
        if(a.capacity!=2u||a.padded_projections!=4u)
            throw std::runtime_error("Original SAP initial source sizing changed");
        grow_sap_old_bounds_native(a,oldBounds,stream);
        for(physx::PxU32 i=0;i<3u;++i) {
            grow_sap_history_axis_prefix_native(a,axis(i),stream);
            grow_sap_history_axis_remainder_native(a,i,rest,stream);
        }

        std::array<std::uint8_t,48u> boundsPattern{};
        std::array<std::array<std::uint8_t,16u>,3u> sapPattern{},projectionPattern{};
        for(std::size_t k=0;k<boundsPattern.size();++k)
            boundsPattern[k]=static_cast<std::uint8_t>((k*7u+51u)%251u);
        upload(oldBounds,boundsPattern.data(),boundsPattern.size());
        for(physx::PxU32 i=0;i<3u;++i) {
            for(std::size_t k=0;k<16u;++k) {
                sapPattern[i][k]=static_cast<std::uint8_t>((k*13u+i*11u+5u)%251u);
                projectionPattern[i][k]=static_cast<std::uint8_t>((k*17u+i*19u+7u)%251u);
            }
            upload(sap[i],sapPattern[i].data(),sapPattern[i].size());
            upload(projections[i],projectionPattern[i].data(),projectionPattern[i].size());
        }
        const auto priorOld=oldBounds.getDevicePtr();
        const auto priorSap=sap[1].getDevicePtr();
        const auto priorProjection=projections[2].getDevicePtr();
        const auto sourceBounds=boundsPattern;
        const auto sourceSap=sapPattern;
        const auto sourceProjection=projectionPattern;

        // Populate all fourteen copy-preserving arrays per spatial axis
        // (seven buffers for each j=0,1), including fixed-size block
        // histograms. Uploads stay queued before the next frame's DtoD growth.
        std::array<std::array<std::array<std::uint8_t,8>,6>,7> historyPatterns{};
        std::array<std::array<CUdeviceptr,6>,7> originalPointers{};
        std::array<std::array<physx::PxU64,6>,7> originalCapacities{};
        for (std::size_t type=0;type<histories.size();++type)
            for (physx::PxU32 index=0;index<6u;++index) {
                auto& original=(*histories[type])[index];
                if (original.getSize()<8u)
                    throw std::runtime_error("Source initial endpoint history buffer undersized");
                auto& pattern=historyPatterns[type][index];
                for (std::size_t k=0;k<pattern.size();++k)
                    pattern[k]=static_cast<std::uint8_t>((type*23u+index*17u+k*29u+3u)%251u);
                upload(original,pattern.data(),pattern.size());
                originalPointers[type][index]=original.getDevicePtr();
                originalCapacities[type][index]=original.getSize();
            }
        const auto savedHistory=historyPatterns;

        // Second frame grows to 6 slots and 12 padded projections. DtoD
        // must copy original old raw allocation bytes, including any
        // preceding HtoD data still queued on the same source GPU stream.
        const std::array<physx::PxU32,3> moreCreated{0u,2u,4u};
        const physx::Bp::BroadPhaseUpdateData second(
            moreCreated.data(),3u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,6u,filter,true,true);
        const auto b=prepare_gpu_dma_up_counts_native(second,members);
        if(b.previous_boxes!=2u || b.current_boxes!=5u ||
           b.projections!=10u || b.padded_projections!=12u)
            throw std::runtime_error("Source SAP growth sizes lost pre-removal history");
        grow_sap_old_bounds_native(b,oldBounds,stream);
        for(physx::PxU32 i=0;i<3u;++i) {
            grow_sap_history_axis_prefix_native(b,axis(i),stream);
            grow_sap_history_axis_remainder_native(b,i,rest,stream);
        }
        // The copied contents are snapshot at the source queue enqueue,
        // not when the host input is later reused.
        boundsPattern.fill(0xaa);
        for(auto& v:sapPattern)v.fill(0xbb);
        for(auto& v:projectionPattern)v.fill(0xcc);
        for(auto& kind:historyPatterns)
            for(auto& pattern:kind) pattern.fill(0xdd);
        stream.wait_at_fetch();
        if(oldBounds.getDevicePtr()==priorOld ||
           sap[1].getDevicePtr()==priorSap ||
           projections[2].getDevicePtr()==priorProjection)
            throw std::runtime_error("Original PhysX SAP history did not grow");
        check(oldBounds,sourceBounds.data(),sourceBounds.size(),
              "Original old integer AABB contents lost across GPU DtoD growth");
        for(physx::PxU32 i=0;i<3u;++i){
            check(sap[i],sourceSap[i].data(),sourceSap[i].size(),
                  "Source SAP per-axis 1D history lost during growth");
            check(projections[i],sourceProjection[i].data(),sourceProjection[i].size(),
                  "Source SAP per-axis projection history lost during growth");
            if(newSap[i].getSize()<6u*8u ||
               ranks[i].getSize()<12u*sizeof(int) ||
               tempKeys[i].getSize()<12u*sizeof(int) ||
               tempRanks[i].getSize()<12u*sizeof(int))
                throw std::runtime_error("Original scratch/swap buffer sizing changed");
            if(incremental[i].getSize()<b.projections*sizeof(int) ||
               blockIncremental[i].getSize()<32u*sizeof(int) ||
               totalEndHistogram[i].getSize()<b.projections*sizeof(physx::PxU32) ||
               blockTotalEndHistogram[i].getSize()<32u*sizeof(physx::PxU32))
                throw std::runtime_error("Source per-axis histogram scratch sizing changed");
        }

        const std::array<physx::PxU64,7> required{
            physx::PxU64(b.padded_projections)*sizeof(int),
            physx::PxU64(b.projections)*sizeof(int),
            32u*sizeof(int), physx::PxU64(b.current_boxes)*sizeof(int),
            physx::PxU64(b.projections)*sizeof(int),
            32u*sizeof(int), physx::PxU64(b.current_boxes)*sizeof(int)};
        for (std::size_t type=0;type<histories.size();++type)
            for (physx::PxU32 index=0;index<6u;++index) {
                const auto& owner=(*histories[type])[index];
                const bool hadToGrow=required[type]>originalCapacities[type][index];
                if(owner.getSize()<required[type] ||
                   (owner.getDevicePtr()!=originalPointers[type][index])!=hadToGrow)
                    throw std::runtime_error("Source endpoint buffer growth condition changed");
                check(owner,savedHistory[type][index].data(),8u,
                      "Source histogram/endpoint history lost across ordered GPU growth");
            }

        // No growth on an unchanged capacity: stable raw device pointers
        // and the original source's retained history remain unchanged.
        const physx::Bp::BroadPhaseUpdateData third(
            nullptr,0u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,6u,filter,false,false);
        const auto c=prepare_gpu_dma_up_counts_native(third,members);
        const auto keepOld=oldBounds.getDevicePtr();
        const auto keepProj=projections[0].getDevicePtr();
        grow_sap_old_bounds_native(c,oldBounds,stream);
        for(physx::PxU32 i=0;i<3u;++i) {
            grow_sap_history_axis_prefix_native(c,axis(i),stream);
            grow_sap_history_axis_remainder_native(c,i,rest,stream);
        }
        if(oldBounds.getDevicePtr()!=keepOld ||
           projections[0].getDevicePtr()!=keepProj ||
           !stream.commands().empty())
            throw std::runtime_error("Source SAP history was reallocated on a no-growth frame");
        check(oldBounds,sourceBounds.data(),sourceBounds.size(),
              "No-growth source SAP frame destroyed persistent history");
        for (std::size_t type=0;type<histories.size();++type)
            for (physx::PxU32 index=0;index<6u;++index)
                check((*histories[type])[index],savedHistory[type][index].data(),8u,
                      "No-growth frame destroyed endpoint history");
        // Reject invalid axis before touching source GPU allocations.
        bool invalidAxisRejected=false;
        try {grow_sap_history_axis_remainder_native(c,3u,rest,stream);}
        catch (const std::out_of_range&) {invalidAxisRejected=true;}
        if (!invalidAxisRejected || !stream.commands().empty())
            throw std::runtime_error("Invalid SAP axis accepted or mutated native GPU stream");
    }
    if(aborted||roots.live_roots()||addresses.live_allocations())
        throw std::runtime_error("Source SAP native history growth leaked PhysX GPU roots");
}

void exercise_original_gpu_dma_up_handles_native(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapGpuDMAUpCountMembers;
    using fyzz::physx_gpu::SourceSapGpuDMAUpHandleBuffers;
    using fyzz::physx_gpu::prepare_gpu_dma_up_counts_native;
    using fyzz::physx_gpu::allocate_gpu_dma_up_handles_native;
    using fyzz::physx_gpu::queue_gpu_dma_up_handles_native;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    physx::Bp::BpFilter filter(false, false);
    {
        physx::PxgHeapMemoryAllocator heap(16384u, callback, abort, &aborted);
        physx::PxgTypedCudaBuffer<physx::PxU32> created(
            heap, physx::PxsHeapStats::eBROADPHASE);
        physx::PxgTypedCudaBuffer<physx::PxU32> removed(
            heap, physx::PxsHeapStats::eBROADPHASE);
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        physx::PxgTypedCudaBuffer<physx::PxU32> updated(
            heap, physx::PxsHeapStats::eBROADPHASE);
#endif
        physx::PxU32 numBoxes=0, capacity=0,createdSize=0,removedSize=0;
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        physx::PxU32 updatedSize=0;
        SourceSapGpuDMAUpCountMembers state{
            numBoxes,capacity,createdSize,removedSize,updatedSize};
        SourceSapGpuDMAUpHandleBuffers arrays{created,removed,updated};
#else
        // Default source GPU/AABB-manager layout has no updated-handle
        // counter or allocation. A helper requiring either cannot bind it.
        SourceSapGpuDMAUpCountMembers state{
            numBoxes,capacity,createdSize,removedSize};
        SourceSapGpuDMAUpHandleBuffers arrays{created,removed};
#endif

        auto verify=[&](const physx::PxgCudaBuffer& buffer,
                        const physx::PxU32* expected, std::size_t n,
                        const char* diagnostic) {
            if(!n) return;
            const auto bytes=n*sizeof(physx::PxU32);
            if(buffer.getSize()<bytes)
                throw std::runtime_error("Original PhysX GPU changed-handle capacity is too small");
            const auto physical=addresses.resolve(buffer.getDevicePtr(),bytes);
            std::vector<physx::PxU32> observed(n);
            context.download(physical.buffer,physical.offset,observed.data(),bytes);
            if(std::memcmp(observed.data(),expected,bytes))
                throw std::runtime_error(diagnostic);
        };

        const std::array<physx::PxU32,2> firstCreated{1u,3u};
        const physx::Bp::BroadPhaseUpdateData first(
            firstCreated.data(),2u, nullptr,0u, nullptr,0u,
            nullptr,nullptr,nullptr,4u,filter,true,true);
        auto firstCounts=prepare_gpu_dma_up_counts_native(first,state);
        if(firstCounts.previous_boxes!=0u || numBoxes!=2u ||
           firstCounts.projections!=4u || firstCounts.padded_projections!=4u ||
           createdSize!=2u || removedSize
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
           || updatedSize
#endif
           )
            throw std::runtime_error("PhysX first gpuDMAUp count/size state changed");
        allocate_gpu_dma_up_handles_native(firstCounts,arrays);
        queue_gpu_dma_up_handles_native(first,firstCounts,arrays,stream);
        stream.wait_at_fetch();
        verify(created,firstCreated.data(),firstCreated.size(),
               "First PhysX source created-handle upload failed");
        const auto rawCreated=created.getDevicePtr();

        // Mixed insertion, removal and update: source projection history
        // includes previous+created, NOT post-removal live boxes.
        std::array<physx::PxU32,2> secondCreated{0u,2u};
        std::array<physx::PxU32,1> secondRemoved{3u};
        std::array<physx::PxU32,1> secondUpdated{1u};
        const auto expectedCreated=secondCreated;
        const auto expectedRemoved=secondRemoved;
        const auto expectedUpdated=secondUpdated;
        const physx::Bp::BroadPhaseUpdateData second(
            secondCreated.data(),2u,secondUpdated.data(),1u,
            secondRemoved.data(),1u,nullptr,nullptr,nullptr,4u,
            filter,true,true);
        auto secondCounts=prepare_gpu_dma_up_counts_native(second,state);
        if(secondCounts.previous_boxes!=2u || numBoxes!=3u ||
           secondCounts.projections!=8u || secondCounts.padded_projections!=8u ||
           createdSize!=2u || removedSize!=1u
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
           || updatedSize!=1u
#endif
           )
            throw std::runtime_error("Mixed source gpuDMAUp sizing/handles lost");
        allocate_gpu_dma_up_handles_native(secondCounts,arrays);
        queue_gpu_dma_up_handles_native(second,secondCounts,arrays,stream);
        // The source update arrays are borrowed and may be reused after HtoD.
        secondCreated[0]=99u;
        secondRemoved[0]=99u;
        secondUpdated[0]=99u;
        stream.wait_at_fetch();
        verify(created,expectedCreated.data(),2u,
               "PhysX source created handles lost input snapshot");
        verify(removed,expectedRemoved.data(),1u,
               "PhysX source removed handles lost input snapshot");
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        verify(updated,expectedUpdated.data(),1u,
               "PhysX source updated handles lost input snapshot");
#else
        // Original default gpuDMAUp does not enqueue updated handle uploads:
        // the GPU AABB manager supplies its changed-handle bitmap instead.
        (void)expectedUpdated;
#endif
        if(created.getDevicePtr()!=rawCreated)
            throw std::runtime_error("Original GPU created-handle allocation grew unnecessarily");

        // Downsize without releasing source buffers or silently shrinking
        // the original grow-only PxgCudaBuffer allocation.
        const std::array<physx::PxU32,1> thirdCreated{3u};
        const std::array<physx::PxU32,2> thirdRemoved{0u,2u};
        const std::array<physx::PxU32,1> thirdUpdated{1u};
        const physx::Bp::BroadPhaseUpdateData third(
            thirdCreated.data(),1u,thirdUpdated.data(),1u,
            thirdRemoved.data(),2u,nullptr,nullptr,nullptr,4u,filter,true,true);
        const auto thirdCounts=prepare_gpu_dma_up_counts_native(third,state);
        if(numBoxes!=2u || thirdCounts.projections!=8u ||
           thirdCounts.padded_projections!=8u)
            throw std::runtime_error("Source gpuDMAUp projection capacity ignored removed boxes");
        allocate_gpu_dma_up_handles_native(thirdCounts,arrays);
        queue_gpu_dma_up_handles_native(third,thirdCounts,arrays,stream);
        stream.wait_at_fetch();
        verify(created,thirdCreated.data(),1u,"Original downsize created handles incorrect");
        verify(removed,thirdRemoved.data(),2u,"Original downsize removed handles incorrect");
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        verify(updated,thirdUpdated.data(),1u,"Original downsize updated handles incorrect");
#endif
        if(created.getDevicePtr()!=rawCreated)
            throw std::runtime_error("Source changed handles shrank device allocation");

        // Empty unchanged source frame: no GPU transfer, original device
        // buffers and count state persist.
        const physx::Bp::BroadPhaseUpdateData empty(
            nullptr,0u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,4u,filter,false,false);
        const auto emptyCounts=prepare_gpu_dma_up_counts_native(empty,state);
        allocate_gpu_dma_up_handles_native(emptyCounts,arrays);
        queue_gpu_dma_up_handles_native(empty,emptyCounts,arrays,stream);
        if(numBoxes!=2u || createdSize || removedSize ||
#ifdef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
           updatedSize ||
#endif
           emptyCounts.projections!=4u || emptyCounts.padded_projections!=4u ||
           !stream.commands().empty() || created.getDevicePtr()!=rawCreated)
            throw std::runtime_error("Empty source frame mutated GPU handles or state");

        // Invalid source inputs must fail before touching counters, GPU
        // memory or the native stream. Source Bp ownership stays unchanged.
        const std::array<physx::PxU32,3> excessive{0u,1u,2u};
        const physx::Bp::BroadPhaseUpdateData invalid(
            nullptr,0u,nullptr,0u,excessive.data(),3u,
            nullptr,nullptr,nullptr,4u,filter,false,false);
        bool rejected=false;
        try {(void)prepare_gpu_dma_up_counts_native(invalid,state);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || numBoxes!=2u || removedSize || capacity!=4u)
            throw std::runtime_error("Invalid PhysX source remove corrupted live box count");
        const physx::Bp::BroadPhaseUpdateData noData(
            nullptr,1u,nullptr,0u,nullptr,0u,
            nullptr,nullptr,nullptr,4u,filter,false,false);
        rejected=false;
        try {(void)prepare_gpu_dma_up_counts_native(noData,state);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || numBoxes!=2u || createdSize)
            throw std::runtime_error("Null PhysX created list corrupted live box count");

        // Stale frame sizes may not enqueue a partial upload.
        rejected=false;
        try {queue_gpu_dma_up_handles_native(third,emptyCounts,arrays,stream);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || !stream.commands().empty())
            throw std::runtime_error("Mismatched PhysX source frame enqueued GPU writes");

#ifndef SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
        // A malformed optional updated list must not become a dependency
        // of default source GPU staging (which never reads this list).
        const physx::Bp::BroadPhaseUpdateData noUpdatedArray(
            nullptr,0u,nullptr,1u,nullptr,0u,
            nullptr,nullptr,nullptr,4u,filter,false,false);
        const auto unchanged=prepare_gpu_dma_up_counts_native(noUpdatedArray,state);
        if(unchanged.current_boxes!=numBoxes || unchanged.created || unchanged.removed)
            throw std::runtime_error("Default GPU build incorrectly requires optional updated handles");
#endif
    }
    if(aborted || roots.live_roots() || addresses.live_allocations())
        throw std::runtime_error("Original gpuDMAUp changed-handle buffers leaked GPU heap roots");
}

void exercise_original_prebroadphase_native(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;
    using fyzz::physx_gpu::SourceSapPreBroadPhaseBuffers;
    using fyzz::physx_gpu::pre_broadphase_native;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context, addresses);
    SourcePhysxHeapRoots roots(context, addresses);
    SourcePhysxDeviceRootCallback callback(roots, &stream);
    bool aborted = false;
    const auto abort=[](void* value) noexcept {
        *static_cast<bool*>(value) = true;
    };
    physx::Bp::BpFilter filter(false, false);

    // The same four original PhysX buffers are reused across scene frames.
    // Source capacity and dirty flags come from real BroadPhaseUpdateData.
    {
        physx::PxgHeapMemoryAllocator heap(8192u, callback, abort, &aborted);
        physx::PxgTypedCudaBuffer<physx::PxBounds3> boundsBuf(
            heap, physx::PxsHeapStats::eBROADPHASE);
        physx::PxgTypedCudaBuffer<physx::PxReal> distancesBuf(
            heap, physx::PxsHeapStats::eBROADPHASE);
        physx::PxgTypedCudaBuffer<physx::PxU32> groupsBuf(
            heap, physx::PxsHeapStats::eBROADPHASE);
        physx::PxgTypedCudaBuffer<physx::PxU32> envBuf(
            heap, physx::PxsHeapStats::eBROADPHASE);
        physx::PxU32 sourceBoxesCapacity=0u;
        SourceSapPreBroadPhaseBuffers owners{
            boundsBuf, distancesBuf, groupsBuf, envBuf, sourceBoxesCapacity};

        std::array<physx::PxBounds3,4u> bounds{};
        std::array<physx::PxReal,4u> distances{0.1f,0.2f,0.3f,0.4f};
        std::array<physx::Bp::FilterGroup::Enum,4u> groups{
            physx::Bp::FilterGroup::eSTATICS,
            physx::Bp::FilterGroup::eDYNAMICS_BASE,
            physx::Bp::FilterGroup::eSTATICS,
            physx::Bp::FilterGroup::eDYNAMICS_BASE};
        std::array<physx::PxU32,4u> env{2u,5u,7u,11u};
        for(std::size_t i=0u;i<bounds.size();++i)
            bounds[i] = physx::PxBounds3(
                physx::PxVec3(static_cast<float>(i),0.0f,-1.0f),
                physx::PxVec3(static_cast<float>(i)+1.0f,1.0f,1.0f));

        auto run=[&](physx::PxU32 cap, bool changed, bool environment) {
            const physx::Bp::BroadPhaseUpdateData update(
                nullptr,0u, nullptr,0u, nullptr,0u,
                bounds.data(), groups.data(), distances.data(),
                environment ? env.data() : nullptr, cap,
                filter, changed, false);
            pre_broadphase_native(update, owners, stream);
        };
        auto verify=[&](const physx::PxgCudaBuffer& buffer,
                        const void* expected, std::size_t length,
                        const char* failure) {
            if (!buffer.getDevicePtr() || buffer.getSize() < length)
                throw std::runtime_error("Original source preBroadPhase buffer missing");
            const auto destination = addresses.resolve(buffer.getDevicePtr(), length);
            std::vector<std::uint8_t> actual(length);
            context.download(destination.buffer, destination.offset,
                             actual.data(), length);
            if (std::memcmp(actual.data(), expected, length))
                throw std::runtime_error(failure);
        };
        run(2u, true, true);
        stream.wait_at_fetch();
        if(sourceBoxesCapacity!=2u)
            throw std::runtime_error("Original preBroadPhase capacity field not initialized");
        verify(boundsBuf,bounds.data(),2u*sizeof(bounds[0]),
               "Original source initial FP bounds were not uploaded");
        verify(distancesBuf,distances.data(),2u*sizeof(distances[0]),
               "Original source initial contact distances were not uploaded");
        verify(groupsBuf,groups.data(),2u*sizeof(groups[0]),
               "Original source initial group IDs were not uploaded");
        verify(envBuf,env.data(),2u*sizeof(env[0]),
               "Original source initial environment IDs were not uploaded");
        const auto preservedBounds = boundsBuf.getDevicePtr();
        const auto preservedEnv = envBuf.getDevicePtr();
        const auto firstBounds = bounds;

        // A non-dirty frame must neither upload nor reallocate FP bounds.
        // Contact distances and groups are still updated; absent env IDs
        // must not touch the existing GPU environment array.
        distances[0]=0.75f;
        groups[0]=physx::Bp::FilterGroup::eDYNAMICS_BASE;
        env[0]=99u;
        bounds[0]=physx::PxBounds3(
            physx::PxVec3(50.0f),physx::PxVec3(60.0f));
        run(2u, false, false);
        stream.wait_at_fetch();
        if(sourceBoxesCapacity!=2u ||
           boundsBuf.getDevicePtr()!=preservedBounds ||
           envBuf.getDevicePtr()!=preservedEnv)
            throw std::runtime_error("Non-dirty source frame reallocated inactive data");
        verify(boundsBuf,firstBounds.data(),2u*sizeof(bounds[0]),
               "Non-dirty source preBroadPhase overwrote persistent FP bounds");
        verify(distancesBuf,distances.data(),2u*sizeof(distances[0]),
               "Non-dirty source frame skipped contact distances");
        verify(groupsBuf,groups.data(),2u*sizeof(groups[0]),
               "Non-dirty source frame skipped groups");
        const std::array<physx::PxU32,2> priorEnv{2u,5u};
        verify(envBuf,priorEnv.data(),sizeof(priorEnv),
               "Absent env IDs changed persistent source GPU data");

        // A larger dirty frame grows original PhysX buffers, still using
        // the original buddy allocator and ordered native upload stream.
        const auto uploadedBounds=bounds;
        const auto uploadedDistances=distances;
        const auto uploadedGroups=groups;
        const auto uploadedEnv=env;
        run(4u, true, true);
        // PhysX update input arrays can be reused by CPU tasks after the
        // nonblocking HtoD submission; native uploads own a staging snapshot.
        bounds[1]=physx::PxBounds3(
            physx::PxVec3(100.0f),physx::PxVec3(101.0f));
        distances[1]=-77.0f;
        groups[1]=physx::Bp::FilterGroup::eINVALID;
        env[1]=123456u;
        stream.wait_at_fetch();
        if(sourceBoxesCapacity!=4u)
            throw std::runtime_error("Resized source preBroadPhase lost box capacity");
        verify(boundsBuf,uploadedBounds.data(),sizeof(bounds),
               "Resized PhysX FP bounds upload lost input snapshot");
        verify(distancesBuf,uploadedDistances.data(),sizeof(distances),
               "Resized PhysX contact-distance upload lost input snapshot");
        verify(groupsBuf,uploadedGroups.data(),sizeof(groups),
               "Resized PhysX filter-group upload lost input snapshot");
        verify(envBuf,uploadedEnv.data(),sizeof(env),
               "Resized PhysX environment-ID upload lost input snapshot");

        // Invalid source input must not enqueue any native GPU command or
        // change buffer identities.
        const auto existing=distancesBuf.getDevicePtr();
        const physx::Bp::BroadPhaseUpdateData invalid(
            nullptr,0u, nullptr,0u, nullptr,0u,
            nullptr, groups.data(), distances.data(), env.data(), 4u,
            filter, true, false);
        bool rejected=false;
        try {pre_broadphase_native(invalid, owners, stream);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || sourceBoxesCapacity!=4u ||
           distancesBuf.getDevicePtr()!=existing ||
           !stream.commands().empty())
            throw std::runtime_error("Invalid PhysX source upload mutated native state");
    }
    if(aborted || roots.live_roots() || addresses.live_allocations())
        throw std::runtime_error("Source preBroadPhase GPU heap roots leaked");
}

void exercise_source_gpu_heap_manager(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceSapDmaNative;

    struct HostRoots final : physx::Cm::VirtualAllocatorCallback {
        physx::PxDefaultAllocator backing;
        std::size_t allocated=0,freed=0;
        bool fail=false;
        void* allocate(std::size_t bytes,int,const char* file,int line) override {
            if(fail) throw std::bad_alloc();
            void* p=backing.allocate(bytes,"SourcePinnedHostFixture",file,line);
            if(p) ++allocated;
            return p;
        }
        void deallocate(void* p) override {
            if(!p) return;
            ++freed;
            backing.deallocate(p);
        }
    };

    CudaDeviceAddressMap addresses;
    SourcePhysxHeapRoots roots(context,addresses);
    SourceSapDmaNative stream(context,addresses);
    SourcePhysxDeviceRootCallback native(roots,&stream);
    HostRoots plainHost,mappedHost;
    bool aborted=false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };

    // Original non-debug constructor exposes TWO views of the SAME mapped
    // pinned heap; the plain-host callback must never be selected.
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            2048u,native,plainHost,mappedHost,abort,&aborted);
        if(manager.mPinnedHostMemoryAllocator!=
               manager.mPinnedHostMappedMemoryAllocator ||
           plainHost.allocated!=0u || mappedHost.allocated!=1u ||
           roots.live_roots()!=1u || manager.getDeviceMemorySize()!=2048u)
            throw std::runtime_error("Source normal GPU heap manager changed mapped-host aliasing");
        const auto desc=physx::PxgCreateAllocatorDesc(manager);
        if(&desc.deviceAlloc!=manager.mDeviceMemoryAllocator ||
           &desc.hostAlloc!=manager.mPinnedHostMemoryAllocator ||
           &desc.hostMappedAlloc!=manager.mPinnedHostMappedMemoryAllocator)
            throw std::runtime_error("Original PxgAllocatorDesc changed normal heap references");
        auto* host=manager.mPinnedHostMappedMemoryAllocator;
        void* hostBlock=host->allocate(256u,0,__FILE__,__LINE__);
        if(!hostBlock || plainHost.allocated!=0u || mappedHost.allocated!=1u)
            throw std::runtime_error("Source heap manager routed normal pinned memory incorrectly");
        host->deallocate(hostBlock);
        auto* device=manager.mDeviceMemoryAllocator;
        void* b=device->allocate(512u,physx::PxsHeapStats::eBROADPHASE,
                                 __FILE__,__LINE__);
        if(!b || manager.getDeviceHeapStats().stats[
                     physx::PxsHeapStats::eBROADPHASE]!=512u)
            throw std::runtime_error("Original GPU manager discarded device heap accounting");
        device->deallocateDeferred(b);
        if(manager.getDeviceHeapStats().stats[
               physx::PxsHeapStats::eBROADPHASE]!=512u)
            throw std::runtime_error("GPU heap manager freed a deferred block early");
        manager.flushDeferredDeallocs();
        if(manager.getDeviceHeapStats().stats[
               physx::PxsHeapStats::eBROADPHASE]!=0u)
            throw std::runtime_error("GPU manager did not flush original deferred free");
    }
    if(roots.live_roots()!=0u || addresses.live_allocations()!=0u ||
       mappedHost.allocated!=mappedHost.freed || plainHost.allocated!=0u)
        throw std::runtime_error("Normal PhysX heap manager leaked or double-freed roots");

    // Source heapCapacity==0 debug policy maintains separate plain and
    // GPU-mapped host heap identities, and allocates exceptional roots.
    {
        physx::PxgHeapMemoryAllocatorManager manager(
            0u,native,plainHost,mappedHost,abort,&aborted);
        if(manager.mPinnedHostMemoryAllocator==
               manager.mPinnedHostMappedMemoryAllocator ||
           manager.getDeviceMemorySize()!=0u)
            throw std::runtime_error("Original debug PhysX heap host identity changed");
        const auto desc=physx::PxgCreateAllocatorDesc(manager);
        if(&desc.hostAlloc==&desc.hostMappedAlloc ||
           &desc.deviceAlloc!=manager.mDeviceMemoryAllocator)
            throw std::runtime_error("Source GPU debug allocator descriptor aliased distinct hosts");
        void* plain=manager.mPinnedHostMemoryAllocator->
            allocate(64u,0,__FILE__,__LINE__);
        void* mapped=manager.mPinnedHostMappedMemoryAllocator->
            allocate(64u,0,__FILE__,__LINE__);
        void* device=manager.mDeviceMemoryAllocator->
            allocate(256u,physx::PxsHeapStats::eSOLVER,__FILE__,__LINE__);
        if(!plain || !mapped || !device ||
           plainHost.allocated!=1u || mappedHost.allocated!=2u ||
           manager.getDeviceMemorySize()!=256u ||
           roots.live_roots()!=1u)
            throw std::runtime_error("Debug PhysX heap manager did not allocate independent roots");
    }
    if(roots.live_roots()!=0u || addresses.live_allocations()!=0u ||
       plainHost.allocated!=plainHost.freed ||
       mappedHost.allocated!=mappedHost.freed || aborted)
        throw std::runtime_error("Debug PhysX heap manager failed destruction ownership");

    // If one of the original host pinned allocators fails during source
    // manager construction, the device root already allocated must be
    // cleaned up without stealing the callback's ownership.
    mappedHost.fail=true;
    bool failed=false;
    try {
        physx::PxgHeapMemoryAllocatorManager impossible(
            2048u,native,plainHost,mappedHost,abort,&aborted);
    } catch(const std::bad_alloc&) {failed=true;}
    mappedHost.fail=false;
    if(!failed || roots.live_roots()!=0u ||
       addresses.live_allocations()!=0u ||
       mappedHost.allocated!=mappedHost.freed || aborted)
        throw std::runtime_error("Failed PhysX GPU heap manager construction leaked roots");
}


void exercise_source_mapped_pair_reports(Context& context, bool force_staged) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourcePhysxMappedReportRoots;
    using fyzz::physx_gpu::SourceSapDmaNative;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative source(context, addresses);
    SourcePhysxHeapRoots deviceRoots(context, addresses);
    SourcePhysxDeviceRootCallback device(deviceRoots, &source);
    SourcePhysxMappedReportRoots mapped(
        context, addresses, source,
        force_staged ? SourcePhysxMappedReportRoots::Mode::ForceStaged
                     : SourcePhysxMappedReportRoots::Mode::PreferDirect);
    bool aborted = false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    constexpr std::size_t bytes=256u;
    std::array<std::uint8_t,bytes> created{},removed{},hostInput{},inputObserved{};
    for(std::size_t i=0;i<bytes;++i) {
        created[i]=static_cast<std::uint8_t>((i*19u+13u)%251u);
        removed[i]=static_cast<std::uint8_t>((i*41u+7u)%251u);
        hostInput[i]=static_cast<std::uint8_t>((i*23u+113u)%251u);
    }
    CudaDeviceAddressMap::Address foundVirtual=0u;
    {
        // These are the ORIGINAL PhysX GPU manager and buddy heaps, not
        // special fixture-managed actor pair buffers. The ordinary-mode
        // source host and mapped-host slots alias the same host heap.
        physx::PxgHeapMemoryAllocatorManager manager(
            4096u,device,mapped,mapped,abort,&aborted);
        if(manager.mPinnedHostMemoryAllocator !=
           manager.mPinnedHostMappedMemoryAllocator)
            throw std::runtime_error("Original PhysX host-mapped report heap alias changed");
        // The manager exposes the source generic callback ABI to consumers.
        // Deferred frees belong to its actual original buddy heap, not to
        // Cm::VirtualAllocatorCallback itself.
        auto* hostHeap=static_cast<physx::PxgHeapMemoryAllocator*>(
            manager.mPinnedHostMappedMemoryAllocator);
        void* found=hostHeap->allocate(bytes,physx::PxsHeapStats::eBROADPHASE,
                                       __FILE__,__LINE__);
        void* lost=hostHeap->allocate(bytes,physx::PxsHeapStats::eBROADPHASE,
                                      __FILE__,__LINE__);
        void* cpuSource=hostHeap->allocate(
            bytes,physx::PxsHeapStats::eBROADPHASE,__FILE__,__LINE__);
        if(!found||!lost||!cpuSource||mapped.live_roots()!=1u)
            throw std::runtime_error("Original PhysX host report heap did not allocate root");
        const bool direct=mapped.is_direct(found,bytes);
        if(direct != mapped.is_direct(lost,bytes) ||
           direct != mapped.is_direct(cpuSource,bytes) ||
           (direct && (force_staged ||
                       !context.device_info().supports_gpu_writable_host_visible)))
            throw std::runtime_error("Source mapped reports chose the wrong native memory path");
        if(!force_staged &&
           context.device_info().supports_gpu_writable_host_visible &&
           !direct) {
            // The general Vulkan memory-type capability may lack a
            // compatible storage-buffer memoryTypeBits intersection.
            // An unsupported-placement error must be the ONLY cause
            // for this permitted fallback; don't hide OOM/driver errors.
            bool incompatible=false;
            try {
                const auto probe=context.create_buffer(
                    4096u+255u,MemoryPlacement::HostVisible);
                context.destroy_buffer(probe);
            } catch (const UnsupportedMemoryPlacement&) {
                incompatible=true;
            }
            if(!incompatible)
                throw std::runtime_error("PhysX mapped path fell back without an incompatible memory type");
        }
        if((reinterpret_cast<std::uintptr_t>(found)&255u)!=0u ||
           (reinterpret_cast<std::uintptr_t>(lost)&255u)!=0u)
            throw std::runtime_error("Source mapped report host roots lost buddy alignment");
        // In PhysX's normal mode, hostAlloc and hostMappedAlloc alias.
        // The third buddy block is a CPU-written pinned source descriptor,
        // which must remain GPU-readable even on the direct report path.
        std::memcpy(cpuSource,hostInput.data(),bytes);
        if(!direct) {
            // Only staged CPU report memory may be inspected BEFORE the GPU fence.
            // Direct zero-copy is GPU-written and may not be read early.
            std::memset(found,0,bytes);
            std::memset(lost,0,bytes);
        }
        const auto foundGPU=mapped.device_address(found,bytes);
        const auto lostGPU=mapped.device_address(lost,bytes);
        foundVirtual=foundGPU;
        if(foundGPU==lostGPU || addresses.resolve(foundGPU,bytes).buffer !=
                                addresses.resolve(lostGPU,bytes).buffer)
            // Both actor reports are slices of one original pinned heap
            // ROOT: distinct virtual subranges of the SAME native buffer.
            throw std::runtime_error("PhysX mapped host buddy regions did not retain source root ownership");
        auto foundView=addresses.resolve(foundGPU,bytes);
        auto lostView=addresses.resolve(lostGPU,bytes);
        if(foundView.buffer!=lostView.buffer ||
           foundView.offset==lostView.offset ||
           context.buffer_is_host_visible(foundView.buffer)!=direct)
            throw std::runtime_error("PhysX mapped host report native root mapping or placement mismatch");
        auto rawCreated=deviceRoots.allocate_root(bytes);
        auto rawLost=deviceRoots.allocate_root(bytes);
        auto rawCPUInput=deviceRoots.allocate_root(bytes);
        source.queue_uploads({
            {rawCreated,created.data(),bytes},
            {rawLost,removed.data(),bytes}
        });
        source.queue_device_copy(foundGPU,rawCreated,bytes);
        source.queue_device_copy(lostGPU,rawLost,bytes);
        if(direct) {
            // An actual GPU operation must read the SAME CPU-authored
            // pinned-host source bytes without an upload/staging step.
            source.queue_device_copy(rawCPUInput,
                                     mapped.device_address(cpuSource,bytes),bytes);
        } else {
            // Original staged pinned-host DMA supplies the source bytes.
            source.queue_uploads({{rawCPUInput,cpuSource,bytes}});
        }
        source.submit();

        if(!direct &&
           (static_cast<const std::uint8_t*>(found)[0]!=0u ||
            static_cast<const std::uint8_t*>(lost)[0]!=0u))
            throw std::runtime_error("GPU report staged copy reached CPU before fetch");
        // No CPU read is permitted in the direct path until the original
        // source completion fence is reaped by publish_at_fetch().
        mapped.publish_at_fetch(found,bytes);
        mapped.publish_at_fetch(lost,bytes);
        if(std::memcmp(found,created.data(),bytes)!=0 ||
           std::memcmp(lost,removed.data(),bytes)!=0)
            throw std::runtime_error("Source mapped actor reports not published at native fetch");
        const auto inputView=addresses.resolve(rawCPUInput,bytes);
        context.download(inputView.buffer,inputView.offset,
                         inputObserved.data(),bytes);
        if(inputObserved!=hostInput)
            throw std::runtime_error("Original PhysX pinned hostAlloc source not GPU-readable");

        bool rejected=false;
        try {
            (void)mapped.device_address(
                static_cast<const std::uint8_t*>(found)+bytes-8u,16u);
        } catch(const std::out_of_range&) {rejected=true;}
        // This range is inside the original 4096-byte HOST ROOT even though
        // it crosses a 256-byte buddy allocation: only the source heap
        // decides buddy-block access validity, never the root mapper.
        if(rejected)
            throw std::runtime_error("Native source host root map conflated buddy and root lifetime");
        rejected=false;
        try {
            (void)mapped.device_address(reinterpret_cast<const void*>(0x1234u),8u);
        } catch(const std::out_of_range&) {rejected=true;}
        if(!rejected)
            throw std::runtime_error("Foreign host pointer accepted as GPU-mapped source alias");
        hostHeap->deallocateDeferred(found);
        hostHeap->deallocateDeferred(lost);
        hostHeap->deallocateDeferred(cpuSource);
        manager.flushDeferredDeallocs(); // device heap ONLY, per source!
        // The host heap retains its own deferred frees until its own flush.
        if(mapped.live_roots()!=1u)
            throw std::runtime_error("PhysX device-only deferred flush retired mapped host root");
        hostHeap->flushDeferredDeallocs();
        deviceRoots.release_root(rawCreated,&source);
        deviceRoots.release_root(rawLost,&source);
        deviceRoots.release_root(rawCPUInput,&source);
    }
    if(mapped.live_roots()!=0u || deviceRoots.live_roots()!=0u ||
       addresses.live_allocations()!=0u || aborted)
        throw std::runtime_error("Original PhysX mapped host report roots leaked");
    bool stale=false;
    try {(void)addresses.resolve(foundVirtual,1u);}
    catch(const std::out_of_range&) {stale=true;}
    if(!stale)
        throw std::runtime_error("Source mapped host report GPU alias survived root retirement");
}


void exercise_true_native_host_visible(Context& context) {
    // PhysX actor report semantics: GPU writes a storage buffer, the host
    // reads ONLY AFTER the existing source submission fence completes.
    constexpr std::size_t bytes=1024u;
    std::array<std::uint8_t,bytes> input{},observed{};
    for(std::size_t i=0;i<bytes;++i)
        input[i]=static_cast<std::uint8_t>((i*37u+11u)%251u);
    const auto device=context.create_buffer(bytes,MemoryPlacement::DeviceLocal);
    if(context.buffer_is_host_visible(device))
        throw std::runtime_error("DeviceLocal buffer unexpectedly claims direct CPU mapping");
    bool rejected=false;
    try {(void)context.map_buffer(device);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("DeviceLocal GPU buffer allowed direct CPU mapping");

    if(!context.device_info().supports_gpu_writable_host_visible) {
        rejected=false;
        try {
            const auto bad=context.create_buffer(bytes,MemoryPlacement::HostVisible);
            context.destroy_buffer(bad);
        } catch(const std::runtime_error&){rejected=true;}
        if(!rejected)
            throw std::runtime_error("Unsupported host-visible UAV allocated silently");
        context.destroy_buffer(device);
        return; // Discrete GPU/WARP without mapped UAV capability uses staged fallback.
    }
    BufferId mapped=0;
    try {
        mapped=context.create_buffer(bytes,MemoryPlacement::HostVisible);
    } catch(const UnsupportedMemoryPlacement&) {
        // Vulkan's per-buffer memoryTypeBits can rule out a host-coherent
        // type despite a general physical-device memory-type capability.
        context.destroy_buffer(device);
        return;
    }
    if(!context.buffer_is_host_visible(mapped))
        throw std::runtime_error("Explicit HostVisible allocation silently used default heap");
    auto* first=static_cast<std::uint8_t*>(context.map_buffer(mapped,HostMapAccess::Write));
    if(!first) throw std::runtime_error("Native GPU-writable mapped buffer returned null");
    std::memset(first,0,bytes);
    rejected=false;
    try {(void)context.map_buffer(mapped);}
    catch(const std::logic_error&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("GPU host storage accepted simultaneous CPU maps");
    rejected=false;
    try {context.retire_buffer(mapped);}
    catch(const std::logic_error&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("Mapped GPU host storage retired before unmap");
    context.unmap_buffer(mapped);
    context.upload(device,0u,input.data(),bytes);
    const auto fence=context.submit_copy_buffer(device,0u,mapped,0u,bytes);
    // GPU completion uses existing backend fences (D3D12 OS event / Vulkan
    // vkWaitForFences); never spin on a mapped CUDA completion flag.
    context.wait_submission(fence);
    const auto* result=static_cast<const std::uint8_t*>(context.map_buffer(mapped,HostMapAccess::Read));
    std::memcpy(observed.data(),result,bytes);
    if(observed!=input)
        throw std::runtime_error("GPU-written zero-copy host mapped reports not visible after fence");
    rejected=false;
    try {context.destroy_buffer(mapped);}
    catch(const std::logic_error&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("Mapped GPU host storage destroyed while held by CPU");
    context.unmap_buffer(mapped);
    // Also verify the reverse direction: CPU writes shared storage, then
    // a native GPU command reads it directly (no intervening host upload).
    auto* cpuWrite=static_cast<std::uint8_t*>(
        context.map_buffer(mapped,HostMapAccess::Write));
    for(std::size_t i=0;i<bytes;++i)
        cpuWrite[i]=static_cast<std::uint8_t>((i*17u+71u)%251u);
    context.unmap_buffer(mapped);
    context.wait_submission(context.submit_copy_buffer(mapped,0u,device,0u,bytes));
    context.download(device,0u,observed.data(),bytes);
    for(std::size_t i=0;i<bytes;++i)
        if(observed[i]!=static_cast<std::uint8_t>((i*17u+71u)%251u))
            throw std::runtime_error("CPU-written shared GPU memory not visible to native GPU copy");
    rejected=false;
    try {context.unmap_buffer(mapped);}
    catch(const std::logic_error&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("GPU host storage unmap without map accepted");
    context.retire_buffer(mapped);
    rejected=false;
    try {(void)context.buffer_is_host_visible(mapped);}
    catch(const std::exception&){rejected=true;}
    if(!rejected)
        throw std::runtime_error("Retired GPU mapped buffer stayed addressable");
    context.destroy_buffer(device);
}

} // namespace

int main() {
    try {
        SourcePhysxFoundation originalPhysxRuntime;
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        CudaDeviceAddressMap pointers;
        constexpr std::size_t bytes = 4096u;
        const auto source = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
        const auto destination = context.create_buffer(bytes, MemoryPlacement::DeviceLocal);
        const auto srcAddress = pointers.register_allocation(source, bytes);
        const auto dstAddress = pointers.register_allocation(destination, bytes);

        std::array<std::uint32_t, bytes / sizeof(std::uint32_t)> initial{};
        context.upload(source, 0u, initial.data(), bytes);
        context.upload(destination, 0u, initial.data(), bytes);

        // Original PhysX GPU host code passes a CUdeviceptr+offset, NOT
        // a buffer ID. Translate it only at this native transfer boundary.
        std::array<std::uint32_t, 64u> patch{};
        for (std::size_t i = 0; i < patch.size(); ++i)
            patch[i] = static_cast<std::uint32_t>(i * 239u + 11u);
        const auto src = pointers.resolve(srcAddress + 512u, sizeof(patch));
        const auto dst = pointers.resolve(dstAddress + 1024u, sizeof(patch));

        const SubmissionId upload = context.submit_upload_buffer(
            src.buffer, src.offset, patch.data(), sizeof(patch));
        const SubmissionId copy = context.submit_copy_buffer(
            src.buffer, src.offset, dst.buffer, dst.offset, sizeof(patch));

        // PhysX may free/reallocate a source immediately after enqueue.
        // Unmapping must invalidate the host pointer without freeing GPU
        // memory still used by earlier submissions.
        if (pointers.unregister_allocation(srcAddress) != source)
            throw std::runtime_error("Source mapping lost compute handle");
        context.retire_buffer(source);
        bool rejected = false;
        try { (void)pointers.resolve(srcAddress + 512u, sizeof(patch)); }
        catch (const std::out_of_range&) { rejected = true; }
        if (!rejected)
            throw std::runtime_error("Retired CUDA virtual pointer still resolves");

        context.wait_submission(copy);
        context.wait_submission(upload);
        std::array<std::uint32_t, bytes / sizeof(std::uint32_t)> result{};
        context.download(destination, 0u, result.data(), bytes);
        for (std::size_t i = 0; i < result.size(); ++i) {
            const std::uint32_t expected =
                i >= 256u && i < 256u + patch.size() ?
                patch[i - 256u] : 0u;
            if (result[i] != expected)
                throw std::runtime_error(
                    "CUDA device-address/native-buffer GPU transfer mismatch");
        }

        if (pointers.unregister_allocation(dstAddress) != destination)
            throw std::runtime_error("Destination mapping lost compute handle");
        context.destroy_buffer(destination);
        if (pointers.live_allocations() != 0u)
            throw std::runtime_error("GPU transfer mapping leak");
        exercise_source_raw_buffer_growth(context);
        exercise_reused_source_submission_scratch(context);
        exercise_native_physx_heap_roots(context);
        exercise_native_heap_raii_teardown(context);
        exercise_source_virtual_device_allocator(context);
        exercise_original_source_physx_buddy_heap(context);
        exercise_original_native_physx_cuda_buffer(context);
        exercise_original_prebroadphase_native(context);
        exercise_original_gpu_dma_up_handles_native(context);
        exercise_original_sap_history_growth_native(context);
        exercise_original_sap_post_axis_buffers_native(context);
        exercise_original_sap_radix_descriptors_native(context);
        exercise_source_single_volume_bitmap_lifecycle_native(context);
        exercise_persistent_source_aabb_manager_native(context);
        exercise_source_sc_element_id_lifecycle_native(context);
        exercise_original_sap_source_members_native(context);
        exercise_original_sap_update_fetch_lifecycle(context);
        exercise_source_gpu_heap_manager(context);
        exercise_source_mapped_pair_reports(context,true);
        exercise_source_mapped_pair_reports(context,false);
        exercise_true_native_host_visible(context);

        std::cout << "PhysX CUDA-style host pointer to native GPU transfer PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX CUDA host transfer FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
