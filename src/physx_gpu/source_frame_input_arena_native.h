// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/frame_working_set.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fyzz::physx_gpu {

// CPU-authored, GPU-read frame inputs use a single physical allocation
// on unified-memory devices. Device numerical state is allocated separately.
class SourceFrameInputArenaNative final {
public:
    SourceFrameInputArenaNative(
        compute::Context& context,CudaDeviceAddressMap& addresses) noexcept
        : context_(context),addresses_(addresses) {
        direct_=context_.device_info().supports_shared_frame_input;
    }

    SourceFrameInputArenaNative(const SourceFrameInputArenaNative&)=delete;
    SourceFrameInputArenaNative& operator=(const SourceFrameInputArenaNative&)=delete;
    ~SourceFrameInputArenaNative() noexcept {
        try { close(); } catch(...) {}
    }

    bool direct() const noexcept { return direct_; }
    std::size_t capacity() const noexcept { return capacity_; }
    CudaDeviceAddressMap::Address address() const noexcept { return address_; }

    CudaDeviceAddressMap::Address write(const void* data,std::size_t bytes) {
        return write_pair(data,bytes,nullptr,0u,0u);
    }

    CudaDeviceAddressMap::Address write_pair(
            const void* first,std::size_t first_bytes,
            const void* second,std::size_t second_bytes,
            std::size_t second_offset) {
        if(!direct_ || (!first_bytes && !second_bytes) ||
           (first_bytes && !first) || (second_bytes && !second) ||
           first_bytes%4u || second_bytes%4u || second_offset%4u ||
           (second_bytes && second_offset<first_bytes))
            throw std::invalid_argument("Invalid mapped GPU frame input");
        if(second_bytes>std::numeric_limits<std::size_t>::max()-second_offset)
            throw std::out_of_range("Frame input extent overflow");
        const auto bytes=std::max(
            first_bytes,second_bytes?second_offset+second_bytes:0u);
        if(bytes>context_.device_info().max_storage_buffer_range)
            throw std::out_of_range("Frame input exceeds GPU storage view limit");
        const auto write_parts=[&](compute::BufferId buffer) {
            auto* mapped=static_cast<std::byte*>(
                context_.map_buffer(buffer,compute::HostMapAccess::Write));
            if(first_bytes)std::memcpy(mapped,first,first_bytes);
            if(second_bytes)
                std::memcpy(mapped+second_offset,second,second_bytes);
            context_.unmap_buffer(buffer);
        };
        const bool contract=working_set_.should_contract(bytes,capacity_);
        if(bytes>capacity_ || contract) {
            const auto limit=static_cast<std::size_t>(
                context_.device_info().max_storage_buffer_range);
            const auto growth=capacity_<=limit/2u
                ? capacity_*2u : limit;
            const auto next_capacity=contract?bytes:std::max(bytes,growth);
            const auto next=context_.create_buffer(
                next_capacity,compute::MemoryPlacement::SharedLocal);
            CudaDeviceAddressMap::Address next_address=0;
            try {
                next_address=addresses_.register_allocation(next,next_capacity);
                write_parts(next);
                // Recorded consumers own the retired allocation until their fence.
                if(buffer_) {
                    context_.retire_buffer(buffer_);
                    addresses_.unregister_allocation(address_);
                }
            }catch(...) {
                if(next_address)addresses_.unregister_allocation(next_address);
                context_.retire_buffer(next);
                throw;
            }
            buffer_=next;
            address_=next_address;
            capacity_=next_capacity;
            working_set_.reset();
        }else write_parts(buffer_);
        return address_;
    }

    void close() {
        if(!buffer_)return;
        context_.retire_buffer(buffer_);
        addresses_.unregister_allocation(address_);
        buffer_=0;
        address_=0;
        capacity_=0;
        working_set_.reset();
    }

private:
    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    bool direct_=false;
    compute::BufferId buffer_=0;
    CudaDeviceAddressMap::Address address_=0;
    std::size_t capacity_=0;
    FrameWorkingSet working_set_;
};
} // namespace fyzz::physx_gpu
