// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fyzzxyzzy/compute.h"

#include <cstddef>
#include <cstdint>
#include <map>

// Internal bridge for the CUdeviceptr arithmetic used by original PhysX
// GPU host code. Not a public CUDA ABI and not a GPU shader dereference
// scheme: the lowering layer must resolve addresses and bind the resulting
// storage buffers before executing translated shaders.
namespace fyzz::physx_gpu {

class CudaDeviceAddressMap final {
public:
    using Address = std::uint64_t;

    struct Resolved {
        compute::BufferId buffer = 0;
        std::size_t offset = 0;
        std::size_t remaining = 0;
    };

    // Model a CUDA allocation using a stable, nonzero virtual address.
    // The caller allocates/manages the real compute buffer separately.
    // Addresses are 256-byte aligned, monotonically assigned, and never
    // reused within this map's lifetime (stale aliases cannot hit a new
    // allocation). Registration does not perform a GPU operation.
    Address register_allocation(compute::BufferId buffer,
                                std::size_t allocation_bytes);

    // Original PxgCudaBuffer/heap allocation can be a *slice* of one
    // persistent native storage resource. Slices receive distinct stable
    // CUDA virtual addresses even if they share the same backing buffer.
    // Their owner (not this address table) owns and retires that buffer.
    Address register_slice(compute::BufferId buffer, std::size_t backing_offset,
                           std::size_t allocation_bytes);

    // A source-ordered GPU copy must already have been enqueued before a
    // live slice moves to a grown arena. This changes future host lowering
    // without invalidating prior submitted GPU commands or virtual pointers.
    void relocate_slice(Address base, compute::BufferId buffer,
                        std::size_t backing_offset);

    // Logical free only: removes the address mapping immediately, but leaves
    // physical retirement and safe reuse to the original host allocator.
    Resolved unregister_slice(Address base);

    // A CUdeviceptr may be base+offset, including offsets > UINT32_MAX.
    // Rejects addresses outside a live allocation, crossing its end,
    // or a zero-length access. Does not query the device or synchronize.
    Resolved resolve(Address address, std::size_t access_bytes) const;

    // Removes an EXACT allocation base immediately from the host lookup.
    // Caller must retire_buffer() on the returned compute handle so work
    // submitted before removal can complete before physical reclamation.
    compute::BufferId unregister_allocation(Address base);

    std::size_t live_allocations() const noexcept;

private:
    struct Allocation {
        compute::BufferId buffer;
        std::size_t bytes;
        std::size_t backing_offset = 0u;
        bool owned = true; // dedicated allocation; slices never retire backing.
    };
    static constexpr Address kAlignment = 256;
    Address next_address_ = 0x100000000ull; // all generated pointers >32-bit
    std::map<Address, Allocation> allocations_;
    std::map<compute::BufferId, Address> by_buffer_; // uniquely owned, retireable
    std::map<compute::BufferId, std::size_t> slice_counts_; // non-owning references
};

} // namespace fyzz::physx_gpu
