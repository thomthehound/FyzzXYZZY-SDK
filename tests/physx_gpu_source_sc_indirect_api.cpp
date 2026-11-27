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
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

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

    ComputePipelineDesc desc{};
    desc.shader=load_shader(consumer_shader);
    desc.storage_buffer_binding_count=2u;
    desc.storage_buffer_views={
        StorageBufferViewKind::RawRead,
        StorageBufferViewKind::RawReadWrite};
    desc.push_constant_bytes=sizeof(std::uint32_t);
    const auto consumer=context.create_compute_pipeline(desc);

    const auto exercise=[&](std::uint32_t node_count) {
        const auto layout=SourceScDeviceIslandLayout::for_counts(node_count,0u,0u);
        const auto node_bytes=std::size_t(node_count)*sizeof(std::uint32_t);
        const auto result_bytes=std::size_t(node_count)*4u*sizeof(std::uint32_t);
        const auto node_buffer=context.create_buffer(
            node_bytes,MemoryPlacement::DeviceLocal);
        const auto work_buffer=context.create_buffer(
            layout.bytes,MemoryPlacement::DeviceLocal);
        const auto indirect_buffer=context.create_buffer(
            sizeof(DispatchSize),MemoryPlacement::DeviceLocal);
        const auto result_buffer=context.create_buffer(
            result_bytes,MemoryPlacement::DeviceLocal);
        const auto node_address=addresses.register_slice(
            node_buffer,0u,node_bytes);
        const auto work_address=addresses.register_slice(
            work_buffer,0u,layout.bytes);
        const auto indirect_address=addresses.register_slice(
            indirect_buffer,0u,sizeof(DispatchSize));

        std::vector<std::uint32_t> nodes(node_count);
        for(std::uint32_t i=0u;i<node_count;++i)
            nodes[i]=(4u+(i&1u))|((1u+(i&1u))<<8u)|
                source_sc_device_island_active_bit;
        context.upload(node_buffer,0u,nodes.data(),node_bytes);
        std::vector<std::uint32_t> output(std::size_t(node_count)*4u,
                                          0xffffffffu);
        context.upload(result_buffer,0u,output.data(),result_bytes);

        stage.append(stream,addresses,0u,0u,node_address,work_address,
                     node_count,0u,0u,SourceScDeviceIslandSources{});
        stream.queue_device_copy(
            indirect_address,
            work_address+offsetof(SourceScDeviceIslandHeader,dispatch_x),
            sizeof(DispatchSize));

        DispatchCommand dispatch{};
        dispatch.pipeline=consumer;
        dispatch.groups={1u,1u,1u};
        dispatch.indirect=DispatchIndirectArgs{indirect_buffer,0u};
        dispatch.buffers={
            {0u,work_buffer,0u,layout.bytes,BufferAccess::Read},
            {1u,result_buffer,0u,result_bytes,BufferAccess::Write}};
        const auto task_offset=static_cast<std::uint32_t>(layout.tasks);
        dispatch.push_constants.resize(sizeof(task_offset));
        std::memcpy(dispatch.push_constants.data(),&task_offset,
                    sizeof(task_offset));
        stream.commands().emplace_back(dispatch);
        stream.wait_at_fetch();

        SourceScDeviceIslandHeader header{};
        context.download(work_buffer,0u,&header,sizeof(header));
        context.download(result_buffer,0u,output.data(),result_bytes);
        const auto limits=source_sc_indirect_group_limits(context.device_info());
        require(header.errors==0u &&
                header.runnable_components==node_count &&
                header.responding_nodes==node_count &&
                header.admitted_edges==0u &&
                header.dispatch_x>0u && header.dispatch_x<=limits[0] &&
                header.dispatch_y>0u && header.dispatch_y<=limits[1] &&
                header.dispatch_z==1u &&
                std::uint64_t(header.dispatch_x)*header.dispatch_y>=node_count &&
                (node_count<=limits[0] || header.dispatch_y>1u),
                "Device Sc failed to publish the physical indirect task grid");
        for(std::uint32_t i=0u;i<node_count;++i) {
            const std::size_t first=std::size_t(i)*4u;
            require(output[first]==i && output[first+1u]==i+1u &&
                    output[first+2u]==4u+(i&1u) &&
                    output[first+3u]==1u+(i&1u),
                    "Indirect consumer lost a device-authored task ordinal");
        }

        if(node_count==2u) {
            std::vector<std::uint32_t> untouched(output.size(),0xffffffffu);
            context.upload(result_buffer,0u,untouched.data(),result_bytes);
            for(unsigned defect=0u;defect<5u;++defect) {
                auto invalid=dispatch;
                switch(defect) {
                    case 0u: invalid.buffers[0].access=BufferAccess::Write; break;
                    case 1u: invalid.buffers[1].access_extent=BufferAccessExtent{0u,0u}; break;
                    case 2u: invalid.buffers[1].slot=0u; break;
                    case 3u: invalid.groups.x=2u; break;
                    case 4u: invalid.indirect=DispatchIndirectArgs{result_buffer,0u}; break;
                }
                bool rejected=false;
                try {context.execute_batch({dispatch,invalid});}
                catch(const std::invalid_argument&) {rejected=true;}
                require(rejected,"Reused D3D12 table accepted malformed dispatch");
            }
            std::vector<std::uint32_t> observed(output.size());
            context.download(result_buffer,0u,observed.data(),result_bytes);
            require(observed==untouched,"Rejected D3D12 batch executed a valid prefix");
            auto shared=dispatch;
            shared.share_resources();
            auto shared_repeat=shared;
            context.execute_batch({shared,shared_repeat});
            context.download(result_buffer,0u,observed.data(),result_bytes);
            require(observed==output,"Reused D3D12 table lost indirect results");
        }

        addresses.unregister_slice(indirect_address);
        addresses.unregister_slice(work_address);
        addresses.unregister_slice(node_address);
        context.destroy_buffer(result_buffer);
        context.destroy_buffer(indirect_buffer);
        context.destroy_buffer(work_buffer);
        context.destroy_buffer(node_buffer);
    };

    exercise(2u);
    const auto& device=context.device_info();
    if(!device.software_adapter &&
       device.max_workgroup_count[0]<std::numeric_limits<std::uint32_t>::max() &&
       source_sc_indirect_task_limit(device)>device.max_workgroup_count[0])
        exercise(device.max_workgroup_count[0]+1u);

    context.destroy_compute_pipeline(consumer);
    require(addresses.live_allocations()==0u,
            "Source Sc indirect fixture leaked virtual allocations");
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
