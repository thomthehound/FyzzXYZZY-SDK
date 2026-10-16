// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_physx_heap_roots.h"
#include "physx_gpu/source_physx_root_callback.h"
#include "physx_gpu/source_staged_host_report_roots.h"
#include "physx_port/gpucommon/PxgHeapMemAllocator.h"
#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
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


void exercise_source_staged_mapped_pair_reports(Context& context) {
    using fyzz::physx_gpu::SourcePhysxHeapRoots;
    using fyzz::physx_gpu::SourcePhysxDeviceRootCallback;
    using fyzz::physx_gpu::SourceStagedHostReportRoots;
    using fyzz::physx_gpu::SourceSapDmaNative;

    CudaDeviceAddressMap addresses;
    SourceSapDmaNative source(context, addresses);
    SourcePhysxHeapRoots deviceRoots(context, addresses);
    SourcePhysxDeviceRootCallback device(deviceRoots, &source);
    SourceStagedHostReportRoots mapped(context, addresses, source);
    bool aborted = false;
    const auto abort=[](void* user) noexcept {
        *static_cast<bool*>(user)=true;
    };
    constexpr std::size_t bytes=256u;
    std::array<std::uint8_t,bytes> created{},removed{};
    for(std::size_t i=0;i<bytes;++i) {
        created[i]=static_cast<std::uint8_t>((i*19u+13u)%251u);
        removed[i]=static_cast<std::uint8_t>((i*41u+7u)%251u);
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
        if(!found||!lost||mapped.live_roots()!=1u)
            throw std::runtime_error("Original PhysX host report heap failed native staging");
        std::memset(found,0,bytes);
        std::memset(lost,0,bytes);
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
           foundView.offset==lostView.offset)
            throw std::runtime_error("PhysX mapped host report slices lost shared native heap root");
        auto rawCreated=deviceRoots.allocate_root(bytes);
        auto rawLost=deviceRoots.allocate_root(bytes);
        source.queue_uploads({
            {rawCreated,created.data(),bytes},
            {rawLost,removed.data(),bytes}
        });
        source.queue_device_copy(foundGPU,rawCreated,bytes);
        source.queue_device_copy(lostGPU,rawLost,bytes);
        source.submit();

        // CPU memory is NOT directly coherent on our staged path and MUST
        // not be interpreted as an up-to-date CUDA-mapped report until
        // source fetch. This is intentionally observable and tested.
        if(static_cast<const std::uint8_t*>(found)[0]!=0u ||
           static_cast<const std::uint8_t*>(lost)[0]!=0u)
            throw std::runtime_error("GPU report staged copy reached CPU before fetch");
        mapped.publish_at_fetch(found,bytes);
        mapped.publish_at_fetch(lost,bytes);
        if(std::memcmp(found,created.data(),bytes)!=0 ||
           std::memcmp(lost,removed.data(),bytes)!=0)
            throw std::runtime_error("Source mapped actor reports not published at native fetch");

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
        manager.flushDeferredDeallocs(); // device heap ONLY, per source!
        // The host heap retains its own deferred frees until its own flush.
        if(mapped.live_roots()!=1u)
            throw std::runtime_error("PhysX device-only deferred flush retired mapped host root");
        hostHeap->flushDeferredDeallocs();
        deviceRoots.release_root(rawCreated,&source);
        deviceRoots.release_root(rawLost,&source);
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
    const auto mapped=context.create_buffer(bytes,MemoryPlacement::HostVisible);
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
        exercise_native_physx_heap_roots(context);
        exercise_native_heap_raii_teardown(context);
        exercise_source_virtual_device_allocator(context);
        exercise_original_source_physx_buddy_heap(context);
        exercise_source_gpu_heap_manager(context);
        exercise_source_staged_mapped_pair_reports(context);
        exercise_true_native_host_visible(context);

        std::cout << "PhysX CUDA-style host pointer to native GPU transfer PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX CUDA host transfer FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
