// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_gpu/cuda_device_address_map.h"
#include "physx_gpu/source_sap_dma_native.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace fyzz::physx_gpu {

// CPU-authored, GPU-read frame inputs use a single physical allocation
// on unified-memory devices. Device numerical state is allocated separately.
class SourceFrameInputArenaNative final {
public:
    SourceFrameInputArenaNative(
        compute::Context& context,CudaDeviceAddressMap& addresses,
        SourceSapDmaNative& stream) noexcept
        : context_(context),addresses_(addresses),stream_(stream) {
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
        if(!direct_ || !data || !bytes || bytes%4u)
            throw std::invalid_argument("Invalid mapped GPU frame input");
        if(bytes>context_.device_info().max_storage_buffer_range)
            throw std::out_of_range("Frame input exceeds GPU storage view limit");
        if(bytes>capacity_) {
            const auto limit=static_cast<std::size_t>(
                context_.device_info().max_storage_buffer_range);
            const auto growth=capacity_<=limit/2u
                ? capacity_*2u : limit;
            const auto next_capacity=std::max(bytes,growth);
            const auto next=context_.create_buffer(
                next_capacity,compute::MemoryPlacement::SharedLocal);
            CudaDeviceAddressMap::Address next_address=0;
            try {
                next_address=addresses_.register_allocation(next,next_capacity);
                write_bytes(next,data,bytes);
                // Previously recorded readers retain the old resource
                // until their source submission reaches its GPU fence.
                if(buffer_)stream_.submit();
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
        } else {
            write_bytes(buffer_,data,bytes);
        }
        return address_;
    }

    void close() {
        if(!buffer_)return;
        context_.retire_buffer(buffer_);
        addresses_.unregister_allocation(address_);
        buffer_=0;
        address_=0;
        capacity_=0;
    }

private:
    void write_bytes(compute::BufferId buffer,const void* data,
                     std::size_t bytes) {
        void* mapped=context_.map_buffer(buffer,compute::HostMapAccess::Write);
        std::memcpy(mapped,data,bytes);
        context_.unmap_buffer(buffer);
    }

    compute::Context& context_;
    CudaDeviceAddressMap& addresses_;
    SourceSapDmaNative& stream_;
    bool direct_=false;
    compute::BufferId buffer_=0;
    CudaDeviceAddressMap::Address address_=0;
    std::size_t capacity_=0;
};
} // namespace fyzz::physx_gpu
