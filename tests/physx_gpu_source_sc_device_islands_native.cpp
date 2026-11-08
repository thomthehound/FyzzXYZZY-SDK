// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_sc_device_islands_native.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;
constexpr std::uint32_t invalid=0xffffffffu;

void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
ShaderBinary load_shader(const char* filename) {
    std::ifstream file(filename,std::ios::binary);
    require(bool(file),"Missing device Sc island shader");
    ShaderBinary shader{};
#ifdef FYZZXYZZY_TEST_D3D12
    shader.format=ShaderFormat::Dxil;
#else
    shader.format=ShaderFormat::Spirv;
#endif
    shader.bytes.assign(std::istreambuf_iterator<char>{file},
                        std::istreambuf_iterator<char>{});
    require(!shader.bytes.empty(),"Empty device Sc island shader");
    return shader;
}

template<class T>
T load(std::span<const std::byte> bytes,std::size_t offset) {
    require(offset<=bytes.size() && sizeof(T)<=bytes.size()-offset,
            "Device Sc island result span truncated");
    T result{};
    std::memcpy(&result,bytes.data()+offset,sizeof(result));
    return result;
}

void verify(Context& context,const char* filename) {
    SourceScDeviceIslandsNativeStage stage(context,load_shader(filename));
    const auto evaluate=[&](std::uint32_t node_count,
                            const std::vector<SourceScDeviceIslandEdge>& edges,
                            std::vector<std::uint32_t> touches) {
        constexpr std::size_t record_bytes=240u;
        require(!touches.empty(),"Device Sc test requires a touch arena");
        const auto layout=SourceScDeviceIslandLayout::for_counts(
            node_count,static_cast<std::uint32_t>(edges.size()));
        const auto input_bytes=std::max<std::size_t>(
            sizeof(SourceScDeviceIslandEdge),
            edges.size()*sizeof(SourceScDeviceIslandEdge));
        const auto edge_buffer=context.create_buffer(
            input_bytes,MemoryPlacement::DeviceLocal);
        const auto touch_buffer=context.create_buffer(
            touches.size()*record_bytes,MemoryPlacement::DeviceLocal);
        const auto output_buffer=context.create_buffer(
            layout.bytes,MemoryPlacement::DeviceLocal);
        CudaDeviceAddressMap addresses{};
        const auto edge_addr=addresses.register_slice(
            edge_buffer,0u,input_bytes);
        const auto touch_addr=addresses.register_slice(
            touch_buffer,0u,touches.size()*record_bytes);
        const auto output_addr=addresses.register_slice(
            output_buffer,0u,layout.bytes);
        if(!edges.empty())
            context.upload(edge_buffer,0u,edges.data(),
                           edges.size()*sizeof(edges[0]));
        std::vector<std::uint32_t> history(touches.size()*60u,0u);
        for(std::size_t i=0u;i<touches.size();++i)
            history[i*60u+18u]=touches[i];
        context.upload(touch_buffer,0u,history.data(),
                       history.size()*sizeof(history[0]));
        SourceSapDmaNative stream(context,addresses);
        stage.append(stream,addresses,edge_addr,touch_addr,output_addr,
                     node_count,static_cast<std::uint32_t>(edges.size()),
                     static_cast<std::uint32_t>(touches.size()));
        stream.wait_at_fetch();
        std::vector<std::byte> bytes(layout.bytes);
        context.download(output_buffer,0u,bytes.data(),bytes.size());
        addresses.unregister_slice(output_addr);
        addresses.unregister_slice(touch_addr);
        addresses.unregister_slice(edge_addr);
        require(addresses.live_allocations()==0u,
                "Device Sc island stage leaked virtual allocations");
        context.destroy_buffer(output_buffer);
        context.destroy_buffer(touch_buffer);
        context.destroy_buffer(edge_buffer);
        return std::pair{layout,std::move(bytes)};
    };
    const auto root=[&](const auto& result,std::size_t node) {
        return load<std::uint32_t>(result.second,
            result.first.roots+node*4u);
    };
    const auto count=[&](const auto& result,std::size_t node) {
        return load<SourceScDeviceIslandCount>(result.second,
            result.first.counts+node*sizeof(SourceScDeviceIslandCount));
    };
    const auto admitted=[&](const auto& result,std::size_t edge) {
        return load<SourceScDeviceIslandAdmission>(result.second,
            result.first.admissions+
            edge*sizeof(SourceScDeviceIslandAdmission));
    };
    const auto no_errors=[&](const auto& result) {
        require(load<std::uint32_t>(result.second,0u)==0u,
                "Device accurate island rejected valid source topology");
    };

    std::vector<SourceScDeviceIslandEdge> edges{
        {0u,0u,1u,10u,0u}, // Contact: found
        {0u,1u,2u,11u,1u}, // Joint: always connected
        {1u,2u,3u,12u,0u}, // Contact: separated
        {2u,4u,5u,13u,0u}, // Independent touching island
        {0u,3u,invalid,14u,1u}, // Joint to static body
        {3u,invalid,6u,15u,0u}, // Contact with kinematic body
        {0u,invalid,invalid,16u,1u}, // Nonresponding pair
        {4u,0u,2u,17u,0u}, // Inactive chord
        {5u,2u,0u,18u,0u}  // Active redundant chord
    };
    auto first=evaluate(7u,edges,{1u,0u,1u,1u,0u,1u});
    no_errors(first);
    require(root(first,0u)==0u && root(first,1u)==0u &&
            root(first,2u)==0u && root(first,3u)==3u &&
            root(first,4u)==4u && root(first,5u)==4u &&
            root(first,6u)==6u,
            "Device Sc merged unrelated accurate islands");
    const auto connected=count(first,0u);
    const auto independent=count(first,4u);
    require(connected.dynamic_nodes==3u &&
            connected.admitted_edges==3u &&
            independent.dynamic_nodes==2u &&
            independent.admitted_edges==1u &&
            count(first,3u).admitted_edges==1u &&
            count(first,6u).admitted_edges==1u,
            "Device Sc accurate island admission counts are incorrect");
    for(std::size_t i=0u;i<edges.size();++i) {
        const auto value=admitted(first,i);
        require(value.source_index==edges[i].source_index &&
                value.edge_index==edges[i].edge_index &&
                value.node_a==edges[i].node_a &&
                value.node_b==edges[i].node_b &&
                value.kind==edges[i].kind,
                "Device Sc reordered or changed an authored manager");
    }
    require(admitted(first,0u).island_root==0u &&
            admitted(first,1u).island_root==0u &&
            admitted(first,2u).island_root==invalid &&
            admitted(first,3u).island_root==4u &&
            admitted(first,4u).island_root==3u &&
            admitted(first,5u).island_root==6u &&
            admitted(first,6u).island_root==invalid &&
            admitted(first,7u).island_root==invalid &&
            admitted(first,8u).island_root==0u,
            "Device Sc admitted an inactive touch or discarded a live edge");

    // A lost bridge and chord split the previously connected island.
    auto split=evaluate(7u,edges,{0u,0u,1u,1u,0u,0u});
    no_errors(split);
    require(root(split,0u)==0u && root(split,1u)==1u &&
            root(split,2u)==1u && root(split,3u)==3u &&
            count(split,0u).admitted_edges==0u &&
            count(split,1u).admitted_edges==1u &&
            admitted(split,0u).island_root==invalid &&
            admitted(split,8u).island_root==invalid,
            "Lost source touches did not split accurate device connectivity");
    // A new touch merges the live joint island with its neighbor.
    auto merged=evaluate(7u,edges,{0u,1u,1u,1u,0u,0u});
    no_errors(merged);
    require(root(merged,1u)==root(merged,2u) &&
            root(merged,2u)==root(merged,3u) &&
            root(merged,0u)!=root(merged,3u) &&
            count(merged,1u).admitted_edges==3u &&
            admitted(merged,2u).island_root==1u,
            "Found contact did not merge exactly its accurate components");

    // An adversarial source order forces multiple simultaneous root
    // changes. Concurrent CAS unions must resolve one canonical root,
    // even for a deep graph with reversed edge encounter order.
    constexpr std::uint32_t chain_nodes=257u;
    std::vector<SourceScDeviceIslandEdge> chain{};
    chain.reserve(chain_nodes-1u);
    for(std::uint32_t i=chain_nodes-1u;i>0u;--i)
        chain.push_back({
            0u,i,i-1u,chain_nodes-1u-i,1u});
    auto long_chain=evaluate(chain_nodes,chain,{0u});
    no_errors(long_chain);
    for(std::uint32_t i=0u;i<chain_nodes;++i)
        require(root(long_chain,i)==0u,
                "Device accurate connectivity left a deep graph disconnected");
    require(count(long_chain,0u).dynamic_nodes==chain_nodes &&
            count(long_chain,0u).admitted_edges==chain_nodes-1u,
            "Device island admission lost joint constraints under contention");
    for(std::size_t i=0u;i<chain.size();++i)
        require(admitted(long_chain,i).island_root==0u &&
                admitted(long_chain,i).source_index==chain[i].source_index,
                "Device island joint admission lost source ordering");

    // Malformed generation/edge/node handles must not silently join
    // unrelated solver bodies or access unallocated GPU touch history.
    auto bad_edges=edges;
    bad_edges[0].edge_index=128u;
    bad_edges[1].node_b=1024u;
    const auto rejected=evaluate(7u,bad_edges,{1u,0u,1u,1u,0u,1u});
    const auto error=load<std::uint32_t>(rejected.second,0u);
    require((error&3u)==3u &&
            admitted(rejected,0u).island_root==invalid &&
            admitted(rejected,1u).island_root==invalid,
            "Device Sc accepted invalid touch or responding endpoint");
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected device Sc accurate island shader");
#ifdef FYZZXYZZY_TEST_D3D12
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1]);
        std::cout<<"Source Sc device accurate island admission PASS\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"Source Sc device accurate island admission FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
