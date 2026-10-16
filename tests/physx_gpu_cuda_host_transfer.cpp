// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_physx_heap_roots.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fyzz::compute;
using fyzz::physx_gpu::CudaDeviceAddressMap;

namespace {
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

} // namespace

int main() {
    try {
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

        std::cout << "PhysX CUDA-style host pointer to native GPU transfer PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PhysX CUDA host transfer FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
