// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_aggregate_member_history_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
ShaderBinary load(const char* path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Missing source aggregate member-history shader");
    ShaderBinary result;
#if defined(FYZZXYZZY_TEST_D3D12)
    result.format=ShaderFormat::Dxil;
#else
    result.format=ShaderFormat::Spirv;
#endif
    result.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    require(!result.bytes.empty(),"Empty aggregate member-history shader");
    return result;
}
void verify(Context& context,ShaderBinary shader) {
    // Two native workgroups, with an added and removed member beyond 128.
    // Original PxgIntegerAABB is precisely six 32-bit encoded endpoints.
    constexpr std::uint32_t capacity=136u;
    std::array<std::uint32_t,capacity*6u> current{},previous{};
    for(std::uint32_t i=0;i<capacity;++i)
        for(std::uint32_t axis=0;axis<6u;++axis) {
            current[i*6u+axis]=0x80000010u+i*32u+axis;
            previous[i*6u+axis]=0x80000050u+i*32u+axis;
        }
    const auto original_current=current,original_previous=previous;
    constexpr std::array<std::uint32_t,3> added{0u,128u,135u};
    constexpr std::array<std::uint32_t,3> removed{2u,129u,134u};

    // The independent original-root allocations may be unaligned slices
    // in the same buddy-heap block. Keep explicit canaries around them.
    constexpr std::size_t current_offset=4u;
    constexpr std::size_t previous_offset=current_offset+sizeof(current)+4u;
    constexpr std::size_t added_offset=previous_offset+sizeof(previous)+4u;
    constexpr std::size_t removed_offset=added_offset+sizeof(added)+4u;
    constexpr std::size_t total=removed_offset+sizeof(removed)+4u;
    std::array<std::uint8_t,total> initial{};
    initial.fill(0xD7u);
    const auto buffer=context.create_buffer(total);
    context.upload(buffer,0,initial.data(),initial.size());
    context.upload(buffer,current_offset,current.data(),sizeof(current));
    context.upload(buffer,previous_offset,previous.data(),sizeof(previous));
    context.upload(buffer,added_offset,added.data(),sizeof(added));
    context.upload(buffer,removed_offset,removed.data(),sizeof(removed));

    CudaDeviceAddressMap addresses;
    const auto current_address=addresses.register_slice(
        buffer,current_offset,sizeof(current));
    const auto previous_address=addresses.register_slice(
        buffer,previous_offset,sizeof(previous));
    const auto added_address=addresses.register_slice(
        buffer,added_offset,sizeof(added));
    const auto removed_address=addresses.register_slice(
        buffer,removed_offset,sizeof(removed));
    {
        SourceSapDmaNative stream(context,addresses);
        SourceAggregateMemberHistoryNativeStage stage(
            context,std::move(shader));
        const SourceAggregateMemberHistoryNativeBuffers buffers{
            current_address,previous_address,added_address,
            removed_address,capacity,
            static_cast<std::uint32_t>(added.size()),
            static_cast<std::uint32_t>(removed.size())};
        stage.append(stream,addresses,{});
        require(stream.commands().empty(),
                "Empty original aggregate source history unexpectedly dispatched");
        bool rejected=false;
        try {
            auto invalid=buffers;
            invalid.old_integer_bounds=0;
            stage.append(stream,addresses,invalid);
        }catch(const std::invalid_argument&) {
            rejected=true;
        }
        require(rejected && stream.commands().empty(),
                "Invalid aggregate history mutated the source command stream");
        rejected=false;
        try {
            auto aliased=buffers;
            aliased.old_integer_bounds=aliased.new_integer_bounds;
            stage.append(stream,addresses,aliased);
        }catch(const std::invalid_argument&) {
            rejected=true;
        }
        require(rejected && stream.commands().empty(),
                "Aliased current and previous aggregate GPU history was admitted");
        stage.append(stream,addresses,buffers);
        require(stream.commands().size()==1u,
                "Original source aggregate member history is not one GPU stage");
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Aggregate history did not use the existing source fetch fence");

        // One-sided quiet lists must never require a non-existent buffer.
        context.upload(buffer,current_offset,original_current.data(),sizeof(current));
        context.upload(buffer,previous_offset,original_previous.data(),sizeof(previous));
        auto just_added=buffers;
        just_added.num_removed=0u;
        just_added.removed_members=0u;
        stage.append(stream,addresses,just_added);
        stream.wait_at_fetch();
        context.download(buffer,current_offset,current.data(),sizeof(current));
        context.download(buffer,previous_offset,previous.data(),sizeof(previous));
        require(current==original_current,
                "New aggregate member incorrectly invalidated current bounds");
        for(std::uint32_t i=0;i<capacity;++i)
            for(std::uint32_t axis=0;axis<6u;++axis) {
                const bool is_added=std::find(added.begin(),added.end(),i)!=added.end();
                const auto expected=is_added
                    ? (axis<3u?0xff7fffffu:0x00800000u)
                    : original_previous[i*6u+axis];
                require(previous[i*6u+axis]==expected,
                        "Source added-member previous AABB invalidation differs");
            }
        context.upload(buffer,current_offset,original_current.data(),sizeof(current));
        context.upload(buffer,previous_offset,original_previous.data(),sizeof(previous));
        auto just_removed=buffers;
        just_removed.num_added=0u;
        just_removed.added_members=0u;
        stage.append(stream,addresses,just_removed);
        stream.wait_at_fetch();
        context.download(buffer,current_offset,current.data(),sizeof(current));
        context.download(buffer,previous_offset,previous.data(),sizeof(previous));
        for(std::uint32_t i=0;i<capacity;++i)
            for(std::uint32_t axis=0;axis<6u;++axis) {
                const bool is_removed=std::find(
                    removed.begin(),removed.end(),i)!=removed.end();
                const auto word=std::size_t(i)*6u+axis;
                const auto empty=axis<3u?0xff7fffffu:0x00800000u;
                require(current[word]==(is_removed?empty:original_current[word]) &&
                        previous[word]==(is_removed?empty:original_previous[word]),
                        "Source removed-only member history invalidation differs");
            }

        context.upload(buffer,current_offset,original_current.data(),sizeof(current));
        context.upload(buffer,previous_offset,original_previous.data(),sizeof(previous));
        stage.append(stream,addresses,buffers);
        stream.wait_at_fetch();
    }

    context.download(buffer,current_offset,current.data(),sizeof(current));
    context.download(buffer,previous_offset,previous.data(),sizeof(previous));
    for(std::uint32_t i=0;i<capacity;++i)
        for(std::uint32_t axis=0;axis<6u;++axis) {
            const bool is_added=std::find(added.begin(),added.end(),i)!=added.end();
            const bool is_removed=std::find(removed.begin(),removed.end(),i)!=removed.end();
            const auto empty=axis<3u?0xff7fffffu:0x00800000u;
            require(current[i*6u+axis]==(is_removed?empty:original_current[i*6u+axis]),
                    "Source removed-member current AABB differs");
            require(previous[i*6u+axis]==(
                is_added || is_removed?empty:original_previous[i*6u+axis]),
                    "Source added/removed member previous AABB differs");
        }
    std::array<std::uint8_t,total> final_bytes{};
    context.download(buffer,0,final_bytes.data(),final_bytes.size());
    for(const std::size_t index:{
            std::size_t{0},previous_offset-1,added_offset-1,
            removed_offset-1,total-1})
        require(final_bytes[index]==0xD7u,
                "Source aggregate history overwrote neighboring heap memory");
    addresses.unregister_slice(removed_address);
    addresses.unregister_slice(added_address);
    addresses.unregister_slice(previous_address);
    addresses.unregister_slice(current_address);
    require(addresses.live_allocations()==0u,"Aggregate member history address map leaked");
    context.destroy_buffer(buffer);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected compiled aggregate history shader");
#if defined(FYZZXYZZY_TEST_D3D12)
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,load(argv[1]));
        std::cout<<"Original aggregate member AABB history GPU stage PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original aggregate member AABB history GPU stage FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
