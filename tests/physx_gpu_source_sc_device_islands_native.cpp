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
#include <span>
#include <utility>
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
    struct DeviceResult {
        SourceScDeviceIslandLayout first{};
        std::vector<std::byte> second{};
        std::uint32_t node_count=0u;
    };
    SourceScDeviceIslandsNativeStage stage(context,load_shader(filename));
    const auto evaluate=[&](std::uint32_t node_count,
                            const std::vector<SourceScDeviceIslandEdge>& edges,
                            std::vector<std::uint32_t> touches,
                            std::vector<std::uint32_t> node_flags={}) {
        constexpr std::size_t record_bytes=240u;
        // Joint-only scenes have no physical-touch arena. Active seeds
        // and authored iteration limits remain separate from connectivity.
        if(node_flags.empty()) {
            node_flags.resize(node_count);
            for(std::uint32_t node=0u;node<node_count;++node)
                node_flags[node]=(4u+node%3u)|
                    ((1u+node%2u)<<8u)|
                    (node%4u==0u?source_sc_device_island_active_bit:0u);
        }
        require(node_flags.size()==node_count,
                "Device Sc test node metadata does not cover its graph");
        const auto layout=SourceScDeviceIslandLayout::for_counts(
            node_count,static_cast<std::uint32_t>(edges.size()));
        const auto input_bytes=std::max<std::size_t>(
            sizeof(SourceScDeviceIslandEdge),
            edges.size()*sizeof(SourceScDeviceIslandEdge));
        const auto edge_buffer=context.create_buffer(
            input_bytes,MemoryPlacement::DeviceLocal);
        const auto touch_bytes=std::max<std::size_t>(
            record_bytes,touches.size()*record_bytes);
        const auto touch_buffer=context.create_buffer(
            touch_bytes,MemoryPlacement::DeviceLocal);
        const auto node_bytes=std::max<std::size_t>(
            sizeof(std::uint32_t),node_flags.size()*sizeof(node_flags[0]));
        const auto node_buffer=context.create_buffer(
            node_bytes,MemoryPlacement::DeviceLocal);
        const auto output_buffer=context.create_buffer(
            layout.bytes,MemoryPlacement::DeviceLocal);
        CudaDeviceAddressMap addresses{};
        const auto edge_addr=addresses.register_slice(
            edge_buffer,0u,input_bytes);
        const auto touch_addr=addresses.register_slice(
            touch_buffer,0u,touch_bytes);
        const auto node_addr=addresses.register_slice(
            node_buffer,0u,node_bytes);
        const auto output_addr=addresses.register_slice(
            output_buffer,0u,layout.bytes);
        if(!edges.empty())
            context.upload(edge_buffer,0u,edges.data(),
                           edges.size()*sizeof(edges[0]));
        if(!node_flags.empty())
            context.upload(node_buffer,0u,node_flags.data(),
                           node_flags.size()*sizeof(node_flags[0]));
        if(!node_count) {
            constexpr std::array<std::uint32_t,4> prior{
                0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu};
            context.upload(output_buffer,0u,prior.data(),sizeof(prior));
        }
        std::vector<std::uint32_t> history(touches.size()*60u,0u);
        for(std::size_t i=0u;i<touches.size();++i)
            history[i*60u+18u]=touches[i];
        if(!history.empty())
            context.upload(touch_buffer,0u,history.data(),
                           history.size()*sizeof(history[0]));
        SourceSapDmaNative stream(context,addresses);
        stage.append(stream,addresses,edge_addr,touch_addr,node_addr,output_addr,
                     node_count,static_cast<std::uint32_t>(edges.size()),
                     static_cast<std::uint32_t>(touches.size()));
        stream.wait_at_fetch();
        std::vector<std::byte> bytes(layout.bytes);
        context.download(output_buffer,0u,bytes.data(),bytes.size());
        addresses.unregister_slice(output_addr);
        addresses.unregister_slice(node_addr);
        addresses.unregister_slice(touch_addr);
        addresses.unregister_slice(edge_addr);
        require(addresses.live_allocations()==0u,
                "Device Sc island stage leaked virtual allocations");
        context.destroy_buffer(output_buffer);
        context.destroy_buffer(node_buffer);
        context.destroy_buffer(touch_buffer);
        context.destroy_buffer(edge_buffer);
        return DeviceResult{layout,std::move(bytes),node_count};
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
    const auto header=[&](const auto& result) {
        return load<SourceScDeviceIslandHeader>(result.second,0u);
    };
    const auto task=[&](const auto& result,std::uint32_t root_id) {
        const auto summary=header(result);
        for(std::uint32_t i=0u;i<summary.runnable_components;++i) {
            const auto item=load<SourceScDeviceIslandTask>(result.second,
                result.first.tasks+i*sizeof(SourceScDeviceIslandTask));
            if(item.root==root_id)return item;
        }
        throw std::runtime_error("Missing source device island task");
    };
    const auto check_task_mapping=[&](const auto& result) {
        const auto summary=header(result);
        std::vector<std::uint8_t> visited(summary.runnable_components,0u);
        std::uint32_t nodes=0u,edges=0u;
        for(std::size_t root_id=0u;
            root_id<result.node_count;++root_id) {
            // Task roots need not be numerically ordered or contiguous.
            const auto record=count(result,root_id);
            if(record.task_index==invalid)continue;
            require(record.task_index<summary.runnable_components &&
                    !visited[record.task_index],
                    "Device Sc task ordinal aliases another active component");
            visited[record.task_index]=1u;
            const auto entry=load<SourceScDeviceIslandTask>(result.second,
                result.first.tasks+
                record.task_index*sizeof(SourceScDeviceIslandTask));
            require(entry.root==root_id &&
                    entry.responding_nodes==record.responding_nodes &&
                    entry.admitted_edges==record.admitted_edges &&
                    entry.position_iterations==record.position_iterations &&
                    entry.velocity_iterations==record.velocity_iterations,
                    "Device Sc root-to-task mapping changes source task inputs");
            nodes+=record.responding_nodes;
            edges+=record.admitted_edges;
        }
        require(nodes==summary.responding_nodes &&
                edges==summary.admitted_edges &&
                std::all_of(visited.begin(),visited.end(),
                            [](std::uint8_t v){return v!=0u;}),
                "Device Sc runnable task mapping has a missing component");
    };
    const auto no_errors=[&](const auto& result) {
        require(load<std::uint32_t>(result.second,0u)==0u,
                "Device accurate island rejected valid source topology");
    };

    std::vector<SourceScDeviceIslandEdge> edges{
        {0u,0u,1u,10u,0u}, // Contact: found
        {31u,1u,2u,11u,1u}, // Joint: always connected
        {1u,2u,3u,12u,0u}, // Contact: separated
        {2u,4u,5u,13u,0u}, // Independent touching island
        {32u,3u,invalid,14u,1u}, // Joint to static body
        {3u,invalid,6u,15u,0u}, // Contact with kinematic body
        {33u,invalid,invalid,16u,1u}, // Nonresponding pair
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
    const auto initial=header(first);
    require(initial.runnable_components==2u &&
            initial.dispatch_x==2u &&
            initial.dispatch_y==1u && initial.dispatch_z==1u &&
            initial.responding_nodes==5u &&
            initial.admitted_edges==4u &&
            task(first,0u).responding_nodes==3u &&
            task(first,0u).admitted_edges==3u &&
            task(first,0u).position_iterations==6u &&
            task(first,0u).velocity_iterations==2u &&
            task(first,4u).responding_nodes==2u &&
            task(first,4u).admitted_edges==1u,
            "Device Sc runnable task stream omitted or mixed components");
    require(connected.responding_nodes==3u &&
            connected.admitted_edges==3u &&
            connected.active_seeds==1u &&
            connected.position_iterations==6u &&
            connected.velocity_iterations==2u &&
            independent.responding_nodes==2u &&
            independent.admitted_edges==1u &&
            independent.active_seeds==1u &&
            independent.position_iterations==6u &&
            independent.velocity_iterations==2u &&
            count(first,3u).admitted_edges==1u &&
            count(first,3u).active_seeds==0u &&
            count(first,6u).admitted_edges==1u,
            "Device Sc accurate island admission counts are incorrect");
    check_task_mapping(first);
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

    // Removing one edge of a cycle must not split the component while
    // its surviving route remains connected.
    auto surviving_route=evaluate(7u,edges,{0u,0u,1u,1u,0u,1u});
    no_errors(surviving_route);
    require(root(surviving_route,0u)==0u &&
            root(surviving_route,1u)==0u &&
            root(surviving_route,2u)==0u &&
            count(surviving_route,0u).admitted_edges==2u &&
            admitted(surviving_route,0u).island_root==invalid &&
            admitted(surviving_route,8u).island_root==0u,
            "Lost chord split an island with a surviving accurate route");

    // Accurate connectedness is independent of Dy row admission.
    // A zero-impulse contact or a response-disabled constraint remains
    // a graph edge but must not consume a solver descriptor or task slot.
    auto nonresponding=edges;
    nonresponding[0].kind=2u;
    nonresponding[1].kind=3u;
    auto no_response=evaluate(7u,nonresponding,{1u,0u,1u,1u,0u,1u});
    no_errors(no_response);
    require(root(no_response,0u)==0u &&
            root(no_response,1u)==0u &&
            root(no_response,2u)==0u &&
            count(no_response,0u).responding_nodes==3u &&
            count(no_response,0u).admitted_edges==1u &&
            count(no_response,0u).active_seeds==1u &&
            count(no_response,0u).position_iterations==6u &&
            task(no_response,0u).admitted_edges==1u &&
            admitted(no_response,0u).island_root==0u &&
            admitted(no_response,0u).kind==2u &&
            admitted(no_response,1u).island_root==0u &&
            admitted(no_response,1u).kind==3u &&
            admitted(no_response,8u).island_root==0u,
            "Device Sc conflated topology with Dy response admission");

    // A lost bridge and chord split the previously connected island.
    auto split=evaluate(7u,edges,{0u,0u,1u,1u,0u,0u});
    no_errors(split);
    check_task_mapping(split);
    require(root(split,0u)==0u && root(split,1u)==1u &&
            root(split,2u)==1u && root(split,3u)==3u &&
            count(split,0u).admitted_edges==0u &&
            count(split,1u).admitted_edges==1u &&
            count(split,1u).active_seeds==0u &&
            count(split,1u).position_iterations==6u &&
            header(split).runnable_components==2u &&
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

    // A newly touching edge joins the previously independent awake
    // island into one device task. Losing that edge immediately
    // restores two accurate tasks with separate active seeds.
    auto bridge=edges;
    bridge.push_back({6u,2u,4u,19u,0u});
    auto joined=evaluate(7u,bridge,{1u,0u,1u,1u,0u,1u,1u});
    no_errors(joined);
    check_task_mapping(joined);
    require(root(joined,0u)==root(joined,4u) &&
            root(joined,5u)==0u &&
            count(joined,0u).responding_nodes==5u &&
            count(joined,0u).admitted_edges==5u &&
            count(joined,0u).active_seeds==2u &&
            count(joined,0u).position_iterations==6u &&
            count(joined,0u).velocity_iterations==2u &&
            admitted(joined,9u).island_root==0u &&
            header(joined).runnable_components==1u &&
            header(joined).responding_nodes==5u &&
            header(joined).admitted_edges==5u &&
            task(joined,0u).admitted_edges==5u,
            "A new contact failed to propagate active Dy admission");
    auto detached=evaluate(7u,bridge,{1u,0u,1u,1u,0u,1u,0u});
    no_errors(detached);
    require(root(detached,0u)==0u &&
            root(detached,4u)==4u &&
            count(detached,0u).responding_nodes==3u &&
            count(detached,0u).active_seeds==1u &&
            count(detached,4u).responding_nodes==2u &&
            count(detached,4u).active_seeds==1u &&
            admitted(detached,9u).island_root==invalid &&
            header(detached).runnable_components==2u &&
            header(detached).responding_nodes==5u &&
            header(detached).admitted_edges==4u &&
            task(detached,4u).admitted_edges==1u,
            "Lost contact retained a cross-island Dy task dependency");

    // A no-edge frame leaves each responding node as a singleton;
    // no dormant previous-frame component may survive.
    auto empty=evaluate(7u,{}, {0u});
    no_errors(empty);
    for(std::uint32_t i=0u;i<7u;++i)
        require(root(empty,i)==i &&
                count(empty,i).responding_nodes==1u &&
                count(empty,i).admitted_edges==0u,
                "Quiet zero-edge device graph retained an old union");

    // Joint-only admission works without any contact-history allocation.
    auto joints_only=evaluate(3u,{
        {71u,0u,1u,0u,1u},
        {72u,1u,2u,1u,1u},
    },{});
    no_errors(joints_only);
    require(root(joints_only,0u)==0u &&
            root(joints_only,1u)==0u &&
            root(joints_only,2u)==0u &&
            count(joints_only,0u).admitted_edges==2u &&
            count(joints_only,0u).active_seeds==1u &&
            count(joints_only,0u).position_iterations==6u &&
            count(joints_only,0u).velocity_iterations==2u &&
            header(joints_only).runnable_components==1u &&
            header(joints_only).responding_nodes==3u &&
            task(joints_only,0u).admitted_edges==2u,
            "Joint-only islands require nonexistent PCM touch storage");

    // Dormant components retain their exact connectivity but enqueue
    // no numerical Dy task until an authored or interacting node wakes.
    auto dormant_flags=std::vector<std::uint32_t>{
        4u|0x100u,5u|0x200u,6u|0x100u,4u|0x200u,
        5u|0x100u,6u|0x200u,4u|0x100u};
    auto dormant=evaluate(7u,edges,{1u,0u,1u,1u,0u,1u},
                          dormant_flags);
    no_errors(dormant);
    require(root(dormant,1u)==0u &&
            root(dormant,5u)==4u &&
            count(dormant,0u).responding_nodes==3u &&
            count(dormant,0u).admitted_edges==3u &&
            header(dormant).runnable_components==0u &&
            header(dormant).dispatch_x==0u &&
            header(dormant).dispatch_y==1u &&
            header(dormant).dispatch_z==1u &&
            header(dormant).responding_nodes==0u &&
            header(dormant).admitted_edges==0u,
            "Dormant source islands generated redundant GPU Dy tasks");

    // No responding nodes may coexist with live static/kinematic
    // managers. The task header must be zeroed even when last frame's
    // output allocation contains a nonzero runnable task count.
    auto no_dynamic=evaluate(0u,{
        {0u,invalid,invalid,91u,0u},
        {42u,invalid,invalid,92u,1u},
    },{1u});
    no_errors(no_dynamic);
    require(header(no_dynamic).runnable_components==0u &&
            header(no_dynamic).responding_nodes==0u &&
            header(no_dynamic).admitted_edges==0u &&
            header(no_dynamic).dispatch_x==0u &&
            admitted(no_dynamic,0u).source_index==91u &&
            admitted(no_dynamic,0u).island_root==invalid &&
            admitted(no_dynamic,1u).source_index==92u &&
            admitted(no_dynamic,1u).island_root==invalid,
            "Nonresponding-only frame replayed stale Dy task admission");
    auto empty_scene=evaluate(0u,{}, {});
    no_errors(empty_scene);
    require(header(empty_scene).runnable_components==0u &&
            header(empty_scene).responding_nodes==0u &&
            header(empty_scene).admitted_edges==0u &&
            header(empty_scene).dispatch_x==0u &&
            header(empty_scene).dispatch_y==1u &&
            header(empty_scene).dispatch_z==1u,
            "Empty frame retained a previous device task header");

    // An adversarial source order forces multiple simultaneous root
    // changes. Concurrent CAS unions must resolve one canonical root,
    // even for a deep graph with reversed edge encounter order.
    constexpr std::uint32_t chain_nodes=257u;
    std::vector<SourceScDeviceIslandEdge> chain{};
    chain.reserve(chain_nodes-1u);
    for(std::uint32_t i=chain_nodes-1u;i>0u;--i)
        chain.push_back({
            100u+i,i,i-1u,chain_nodes-1u-i,1u});
    auto long_chain=evaluate(chain_nodes,chain,{0u});
    no_errors(long_chain);
    for(std::uint32_t i=0u;i<chain_nodes;++i)
        require(root(long_chain,i)==0u,
                "Device accurate connectivity left a deep graph disconnected");
    require(header(long_chain).runnable_components==1u &&
            header(long_chain).responding_nodes==chain_nodes &&
            header(long_chain).admitted_edges==chain_nodes-1u &&
            task(long_chain,0u).admitted_edges==chain_nodes-1u &&
            count(long_chain,0u).responding_nodes==chain_nodes &&
            count(long_chain,0u).admitted_edges==chain_nodes-1u &&
            count(long_chain,0u).active_seeds==65u &&
            count(long_chain,0u).position_iterations==6u &&
            count(long_chain,0u).velocity_iterations==2u,
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
    bad_edges[2].kind=4u;
    std::vector<std::uint32_t> malformed_flags{
        4u|0x10000u|0x100u,5u|0x200u,6u|0x100u,
        4u|0x200u,0u,6u|0x200u,4u|0x100u};
    const auto rejected=evaluate(7u,bad_edges,{1u,0u,1u,1u,0u,1u},
                                 malformed_flags);
    const auto error=load<std::uint32_t>(rejected.second,0u);
    require((error&19u)==19u &&
            header(rejected).runnable_components==0u &&
            header(rejected).dispatch_x==0u &&
            admitted(rejected,0u).island_root==invalid &&
            admitted(rejected,1u).island_root==invalid &&
            admitted(rejected,2u).island_root==invalid,
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
