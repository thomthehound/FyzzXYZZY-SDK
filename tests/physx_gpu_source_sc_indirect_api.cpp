// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_sc_device_islands_native.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}

ShaderBinary load_shader(const char* filename) {
    std::ifstream input(filename,std::ios::binary);
    require(bool(input),"Source Sc indirect shader unavailable");
    ShaderBinary shader{};
    shader.format=ShaderFormat::Dxil;
    shader.bytes.assign(std::istreambuf_iterator<char>{input},
                        std::istreambuf_iterator<char>{});
    require(!shader.bytes.empty(),"Source Sc indirect shader is empty");
    return shader;
}

void verify(const char* source_shader,const char* consumer_shader) {
    Context context(Backend::D3D12);
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourceScDeviceIslandsNativeStage stage(
        context,load_shader(source_shader));

    constexpr std::uint32_t node_count=2u;
    const auto layout=SourceScDeviceIslandLayout::for_counts(node_count,0u);
    const auto node_buffer=context.create_buffer(
        node_count*sizeof(std::uint32_t),MemoryPlacement::DeviceLocal);
    const auto work_buffer=context.create_buffer(
        layout.bytes,MemoryPlacement::DeviceLocal);
    const auto indirect_buffer=context.create_buffer(
        sizeof(DispatchSize),MemoryPlacement::DeviceLocal);
    const auto result_buffer=context.create_buffer(
        node_count*4u*sizeof(std::uint32_t),MemoryPlacement::DeviceLocal);
    const auto node_address=addresses.register_slice(
        node_buffer,0u,node_count*sizeof(std::uint32_t));
    const auto work_address=addresses.register_slice(
        work_buffer,0u,layout.bytes);
    const auto indirect_address=addresses.register_slice(
        indirect_buffer,0u,sizeof(DispatchSize));

    const std::array<std::uint32_t,node_count> nodes{{
        4u|(1u<<8u)|source_sc_device_island_active_bit,
        5u|(2u<<8u)|source_sc_device_island_active_bit
    }};
    context.upload(node_buffer,0u,nodes.data(),sizeof(nodes));
    std::array<std::uint32_t,node_count*4u> output{};
    output.fill(0xffffffffu);
    context.upload(result_buffer,0u,output.data(),sizeof(output));

    stage.append(stream,addresses,0u,0u,node_address,work_address,
                 node_count,0u,0u,SourceScDeviceIslandSources{});
    stream.queue_device_copy(
        indirect_address,
        work_address+offsetof(SourceScDeviceIslandHeader,dispatch_x),
        sizeof(DispatchSize));

    ComputePipelineDesc desc{};
    desc.shader=load_shader(consumer_shader);
    desc.storage_buffer_binding_count=2u;
    desc.storage_buffer_views={
        StorageBufferViewKind::RawRead,
        StorageBufferViewKind::RawReadWrite};
    desc.push_constant_bytes=sizeof(std::uint32_t);
    const auto consumer=context.create_compute_pipeline(desc);
    DispatchCommand dispatch{};
    dispatch.pipeline=consumer;
    dispatch.groups={1u,1u,1u};
    dispatch.indirect=DispatchIndirectArgs{indirect_buffer,0u};
    dispatch.buffers={
        {0u,work_buffer,0u,layout.bytes,BufferAccess::Read},
        {1u,result_buffer,0u,sizeof(output),BufferAccess::Write}};
    const auto task_offset=static_cast<std::uint32_t>(layout.tasks);
    dispatch.push_constants.resize(sizeof(task_offset));
    std::memcpy(dispatch.push_constants.data(),&task_offset,
                sizeof(task_offset));
    stream.commands().emplace_back(std::move(dispatch));
    stream.wait_at_fetch();

    SourceScDeviceIslandHeader header{};
    context.download(work_buffer,0u,&header,sizeof(header));
    context.download(result_buffer,0u,output.data(),sizeof(output));
    require(header.errors==0u &&
            header.runnable_components==node_count &&
            header.responding_nodes==node_count &&
            header.admitted_edges==0u &&
            header.dispatch_x==node_count &&
            header.dispatch_y==1u && header.dispatch_z==1u,
            "Device Sc failed to publish the physical indirect task count");
    const std::array<std::uint32_t,node_count*4u> expected{{
        0u,1u,4u,1u,
        1u,2u,5u,2u
    }};
    require(output==expected,
            "D3D12 indirect dispatch did not consume every device-authored task");

    context.destroy_compute_pipeline(consumer);
    addresses.unregister_slice(indirect_address);
    addresses.unregister_slice(work_address);
    addresses.unregister_slice(node_address);
    require(addresses.live_allocations()==0u,
            "Source Sc indirect fixture leaked virtual allocations");
    context.destroy_buffer(result_buffer);
    context.destroy_buffer(indirect_buffer);
    context.destroy_buffer(work_buffer);
    context.destroy_buffer(node_buffer);
}
}

int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected Sc producer and indirect consumer shaders");
        verify(argv[1],argv[2]);
        std::cout<<"Source Sc device indirect ABI PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Source Sc device indirect ABI FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
