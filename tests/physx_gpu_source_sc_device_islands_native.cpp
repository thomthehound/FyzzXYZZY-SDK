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
#include <optional>
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
        std::uint32_t node_count=0u,edge_count=0u;
    };
    SourceScDeviceIslandsNativeStage stage(context,load_shader(filename));
    const auto evaluate=[&](std::uint32_t node_count,
                            const std::vector<SourceScDeviceIslandEdge>& edges,
                            std::vector<std::uint32_t> touches,
                            std::vector<std::uint32_t> node_flags={},
                            std::vector<std::uint32_t> prior_touches={},
                            std::optional<SourceScDeviceIslandSources> limits=std::nullopt) {
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
        if(prior_touches.empty())prior_touches=touches;
        require(prior_touches.size()==touches.size(),
                "Device Sc test touch snapshots have different capacities");
        std::vector<std::uint32_t> history(touches.size()*60u,0u);
        for(std::size_t i=0u;i<touches.size();++i) {
            history[i*60u+18u]=touches[i];
            history[i*60u+19u]=prior_touches[i];
        }
        if(!history.empty())
            context.upload(touch_buffer,0u,history.data(),
                           history.size()*sizeof(history[0]));
        // Default fixture limits represent actual typed Dy source
        // tables; targeted tests can narrow one independently.
        SourceScDeviceIslandSources sources{};
        for(const auto& edge:edges) {
            if((edge.kind&source_sc_device_island_no_response_bit)!=0u ||
               edge.source_index>=0x40000000u)continue;
            auto& count=(edge.kind&source_sc_device_island_joint_bit)==0u
                ?sources.contact_managers:
                (edge.kind&source_sc_device_island_device_joint_bit)!=0u
                    ?sources.joint_rows:sources.full_descriptors;
            count=std::max(count,edge.source_index+1u);
        }
        if(limits)sources=*limits;
        SourceSapDmaNative stream(context,addresses);
        stage.append(stream,addresses,edge_addr,touch_addr,node_addr,output_addr,
                     node_count,static_cast<std::uint32_t>(edges.size()),
                     static_cast<std::uint32_t>(touches.size()),sources);
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
        return DeviceResult{layout,std::move(bytes),node_count,
                            static_cast<std::uint32_t>(edges.size())};
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
        if(root_id>=result.node_count)
            throw std::runtime_error("Device Sc task root exceeds node view");
        const auto index=count(result,root_id).task_index;
        if(index==invalid || index>=header(result).runnable_components)
            throw std::runtime_error("Missing source device island task");
        return load<SourceScDeviceIslandTask>(result.second,
            result.first.tasks+index*sizeof(SourceScDeviceIslandTask));
    };
    const auto task_nodes=[&](const auto& result,std::uint32_t root_id) {
        const auto entry=task(result,root_id);
        return entry.body_end-entry.body_begin;
    };
    const auto task_edges=[&](const auto& result,std::uint32_t root_id) {
        const auto entry=task(result,root_id);
        return entry.descriptor_end-entry.descriptor_begin;
    };
    const auto check_task_mapping=[&](const auto& result) {
        const auto summary=header(result);
        require(summary.source_edges==result.edge_count,
                "Device Sc header omitted source admission capacity");
        std::vector<std::uint8_t> visited(summary.runnable_components,0u);
        std::uint32_t nodes=0u,edges=0u;
        std::uint32_t next_task=0u,next_response=0u,next_body=0u;
        for(std::size_t root_id=0u;
            root_id<result.node_count;++root_id) {
            // Task roots need not be numerically ordered or contiguous.
            const auto record=count(result,root_id);
            if(record.task_index==invalid)continue;
            require(record.task_index==next_task++ &&
                    record.task_index<summary.runnable_components &&
                    !visited[record.task_index],
                    "Device Sc task ordinal aliases another active component");
            visited[record.task_index]=1u;
            const auto entry=load<SourceScDeviceIslandTask>(result.second,
                result.first.tasks+
                record.task_index*sizeof(SourceScDeviceIslandTask));
            require(entry.descriptor_begin==next_response &&
                    entry.descriptor_end==next_response+record.admitted_edges &&
                    entry.partition_begin==entry.descriptor_begin &&
                    entry.partition_end==entry.descriptor_end &&
                    entry.body_begin==next_body &&
                    entry.body_end==next_body+record.responding_nodes &&
                    entry.position_iterations==record.position_iterations &&
                    entry.velocity_iterations==record.velocity_iterations &&
                    entry.error==0u,
                    "Device Sc root-to-task mapping or response prefix is invalid");
            next_response+=record.admitted_edges;
            next_body+=record.responding_nodes;
            nodes+=record.responding_nodes;
            edges+=record.admitted_edges;
        }
        require(nodes==summary.responding_nodes &&
                edges==summary.admitted_edges &&
                next_response==summary.admitted_edges &&
                next_body==summary.responding_nodes &&
                std::all_of(visited.begin(),visited.end(),
                            [](std::uint8_t v){return v!=0u;}),
                "Device Sc runnable task mapping has a missing component");
        std::vector<std::pair<std::uint32_t,std::uint32_t>> expected_bodies{};
        for(std::uint32_t node=0u;node<result.node_count;++node) {
            const auto owner=root(result,node);
            if(owner<result.node_count &&
               count(result,owner).task_index!=invalid)
                expected_bodies.emplace_back(owner,node);
        }
        std::stable_sort(expected_bodies.begin(),expected_bodies.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        require(expected_bodies.size()==summary.responding_nodes,
                "Device Sc body prefix disagrees with responding nodes");
        for(std::size_t i=0u;i<expected_bodies.size();++i)
            require(load<std::uint32_t>(result.second,
                        result.first.body_order+i*4u)==expected_bodies[i].second,
                    "Device Sc body scatter reordered a solver node");
        std::vector<std::pair<std::uint32_t,std::uint32_t>> expected{};
        for(std::uint32_t edge=0u;edge<result.edge_count;++edge) {
            const auto record=admitted(result,edge);
            if(record.island_root==invalid ||
               (record.kind&source_sc_device_island_no_response_bit)!=0u ||
               count(result,record.island_root).task_index==invalid)
                continue;
            expected.emplace_back(record.island_root,edge);
        }
        std::stable_sort(expected.begin(),expected.end(),
            [&](const auto& a,const auto& b) {
                if(a.first!=b.first)return a.first<b.first;
                const bool jointA=(admitted(result,a.second).kind&
                    source_sc_device_island_joint_bit)!=0u;
                const bool jointB=(admitted(result,b.second).kind&
                    source_sc_device_island_joint_bit)!=0u;
                return jointA && !jointB;
            });
        require(expected.size()==summary.admitted_edges,
                "Device Sc response count disagrees with source admissions");
        for(std::size_t i=0u;i<expected.size();++i) {
            const auto edge=expected[i].second;
            const auto record=admitted(result,edge);
            const auto tag=(record.kind&source_sc_device_island_joint_bit)==0u
                ?0xc0000000u:
                ((record.kind&source_sc_device_island_device_joint_bit)!=0u
                    ?0x80000000u:0u);
            const auto observed=load<std::uint32_t>(
                result.second,result.first.response_order+i*4u);
            const auto token=load<std::uint32_t>(
                result.second,result.first.response_tokens+i*4u);
            if(observed!=edge || token!=(tag|record.source_index))
                throw std::runtime_error(
                    "Device Sc response mismatch at " +
                    std::to_string(i) + ": edge=" +
                    std::to_string(observed) + ", expected=" +
                    std::to_string(edge) + ", token=" +
                    std::to_string(token));
        }
    };
    const auto no_errors=[&](const auto& result) {
        const auto errors=load<std::uint32_t>(result.second,0u);
        if(errors!=0u)
            throw std::runtime_error(
                "Device accurate island rejected valid source topology: error=" +
                    std::to_string(errors));
    };

    std::vector<SourceScDeviceIslandEdge> edges{
        {0u,0u,1u,10u,0u}, // Contact: found
        {31u,1u,2u,11u,source_sc_device_island_joint_bit|
            source_sc_device_island_device_joint_bit}, // Device 1D row
        {1u,2u,3u,12u,0u}, // Contact: separated
        {2u,4u,5u,13u,0u}, // Independent touching island
        {32u,3u,invalid,14u,1u}, // Joint to static body
        {3u,invalid,6u,15u,0u}, // Contact with kinematic body
        {33u,invalid,invalid,16u,1u}, // Nonresponding pair
        {4u,0u,2u,17u,0u}, // Inactive chord
        {5u,2u,0u,18u,0u}  // Active redundant chord
    };
    // Device numeric order must place source 1D descriptors before
    // contacts even when Sc's input graph interleaves both kinds.
    const std::vector<SourceScDeviceIslandEdge> interleaved{
        {0u,0u,1u,40u,0u},
        {31u,0u,1u,20u,source_sc_device_island_joint_bit},
        {1u,1u,0u,41u,0u},
        {32u,1u,0u,19u,source_sc_device_island_joint_bit|
                          source_sc_device_island_device_joint_bit}
    };
    auto ordered=evaluate(2u,interleaved,{1u,1u}, {0x10000u|0x104u,0x104u});
    no_errors(ordered);
    check_task_mapping(ordered);
    for(std::uint32_t i=0u;i<4u;++i)
        require(load<std::uint32_t>(ordered.second,
                    ordered.first.response_order+i*4u)==
                    std::array<std::uint32_t,4>{1u,3u,0u,2u}[i],
                "Device Sc numerical admission did not place joints first");

    // In the small direct-scatter path, nonzero component roots must
    // decode the same three-way key as radix-sorted components.
    const std::vector<SourceScDeviceIslandEdge> split_types{
        {0u,0u,1u,3u,0u},
        {31u,0u,1u,2u,source_sc_device_island_joint_bit},
        {1u,2u,3u,7u,0u},
        {32u,3u,2u,6u,source_sc_device_island_joint_bit}
    };
    auto direct_split=evaluate(4u,split_types,{1u,1u},
        {0x10104u,0x104u,0x10104u,0x104u});
    no_errors(direct_split);
    check_task_mapping(direct_split);
    require(header(direct_split).runnable_components==2u &&
            task(direct_split,0u).descriptor_begin==0u &&
            task(direct_split,2u).descriptor_begin==2u &&
            load<std::uint32_t>(direct_split.second,
                direct_split.first.response_order+0u)==1u &&
            load<std::uint32_t>(direct_split.second,
                direct_split.first.response_order+4u)==0u &&
            load<std::uint32_t>(direct_split.second,
                direct_split.first.response_order+8u)==3u &&
            load<std::uint32_t>(direct_split.second,
                direct_split.first.response_order+12u)==2u,
        "Device Sc direct scatter misdecoded a nonzero island root");

    // An otherwise valid connected graph must not launch Dy with a
    // token that exceeds its *own* numerical table. Contact managers,
    // device-authored rows and full joint descriptors have independent
    // ordinal spaces even when they occur in the same island.
    const std::vector<std::uint32_t> typed_flags{
        0x10104u,0x104u
    };
    const auto bad_manager=evaluate(2u,interleaved,{1u,1u},
        typed_flags,{},SourceScDeviceIslandSources{41u,20u,21u});
    const auto bad_joint_row=evaluate(2u,interleaved,{1u,1u},
        typed_flags,{},SourceScDeviceIslandSources{42u,19u,21u});
    const auto bad_full_joint=evaluate(2u,interleaved,{1u,1u},
        typed_flags,{},SourceScDeviceIslandSources{42u,20u,20u});
    for(const auto* result:{&bad_manager,&bad_joint_row,&bad_full_joint})
        require((header(*result).errors&64u)!=0u &&
                header(*result).runnable_components==0u &&
                header(*result).dispatch_x==0u,
                "Device Sc accepted an out-of-range typed Dy source token");

    // A broadphase overlap without physical touch is not a solver
    // constraint. It has no Dy source record and must not invalidate
    // otherwise runnable independent bodies. The same source identity
    // becomes mandatory as soon as the contact physically touches.
    const std::vector<SourceScDeviceIslandEdge> separated_contact{
        {0u,0u,1u,invalid,0u}
    };
    const std::vector<std::uint32_t> separate_flags{0x10104u,0x104u};
    auto no_touch_source=evaluate(2u,separated_contact,{0u},
        separate_flags,{},SourceScDeviceIslandSources{});
    no_errors(no_touch_source);
    check_task_mapping(no_touch_source);
    require(root(no_touch_source,0u)!=root(no_touch_source,1u) &&
            header(no_touch_source).runnable_components==1u &&
            header(no_touch_source).admitted_edges==0u &&
            admitted(no_touch_source,0u).island_root==invalid,
            "Inactive contact incorrectly required a Dy numerical record");
    auto touching_missing_source=evaluate(2u,separated_contact,{1u},
        separate_flags,{0u},SourceScDeviceIslandSources{});
    require((header(touching_missing_source).errors&64u)!=0u &&
            header(touching_missing_source).dispatch_x==0u,
            "Touching contact admitted a missing numerical source record");

    // Response-disabled physical connectivity still admits every
    // responding body, even without a numerical source-table entry.
    // Its token is never consumed by Dy.
    auto connectivity_only=evaluate(2u,{
        {0u,0u,1u,invalid,source_sc_device_island_no_response_bit},
    },{1u},{0x10104u,0x104u},{0u},
        SourceScDeviceIslandSources{});
    no_errors(connectivity_only);
    check_task_mapping(connectivity_only);
    require(root(connectivity_only,0u)==root(connectivity_only,1u) &&
            count(connectivity_only,0u).responding_nodes==2u &&
            count(connectivity_only,0u).admitted_edges==0u &&
            header(connectivity_only).runnable_components==1u &&
            header(connectivity_only).responding_nodes==2u &&
            header(connectivity_only).admitted_edges==0u,
            "Response-disabled contact lost accurate connectivity without a Dy source");

    auto first=evaluate(7u,edges,{1u,0u,1u,1u,0u,1u});
    no_errors(first);
    require(root(first,0u)==0u && root(first,1u)==0u &&
            root(first,2u)==0u && root(first,3u)==3u &&
            root(first,4u)==4u && root(first,5u)==4u &&
            root(first,6u)==6u,
            "Device Sc merged unrelated accurate islands");
    check_task_mapping(first);
    const auto connected=count(first,0u);
    const auto independent=count(first,4u);
    const auto initial=header(first);
    require(initial.runnable_components==2u &&
            initial.dispatch_x==2u &&
            initial.dispatch_y==1u && initial.dispatch_z==1u &&
            initial.responding_nodes==5u &&
            initial.admitted_edges==4u &&
            task_nodes(first,0u)==3u &&
            task_edges(first,0u)==3u &&
            task(first,0u).position_iterations==6u &&
            task(first,0u).velocity_iterations==2u &&
            task_nodes(first,4u)==2u &&
            task_edges(first,4u)==1u &&
            task(first,0u).descriptor_begin==0u &&
            task(first,4u).descriptor_begin==3u,
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
            task_edges(no_response,0u)==1u &&
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
            task_edges(joined,0u)==5u,
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
            task_edges(detached,4u)==1u,
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

    // CPU Gu and device PCM publish into the same EdgeIndex touch
    // history. Physical connectivity cannot be asserted by a host bit.
    auto mixed_gu=edges;
    mixed_gu.push_back({70u,3u,4u,22u,0u});
    std::vector<std::uint32_t> gu_touches(71u,0u);
    constexpr std::array<std::uint32_t,6> pcm_touches{1u,0u,1u,1u,0u,1u};
    std::copy(pcm_touches.begin(),pcm_touches.end(),gu_touches.begin());
    gu_touches[70u]=1u;
    auto new_gu_prior=gu_touches;
    new_gu_prior[70u]=0u;
    auto host_connected=evaluate(7u,mixed_gu,gu_touches);
    no_errors(host_connected);
    check_task_mapping(host_connected);
    require(root(host_connected,3u)==root(host_connected,4u) &&
            root(host_connected,3u)!=root(host_connected,0u) &&
            count(host_connected,3u).responding_nodes==3u &&
            count(host_connected,3u).admitted_edges==3u &&
            admitted(host_connected,9u).source_index==22u &&
            admitted(host_connected,9u).island_root==3u,
            "Published CPU Gu touch lost device accurate connectivity");
    // The same published contact is new only when the journaled prior
    // touch is zero. A source-side kind flag cannot manufacture a wake.
    auto cpu_gu_new=mixed_gu;
    const std::vector<std::uint32_t> dormant_gu_flags{
        4u|0x100u,5u|0x200u,6u|0x100u,4u|0x200u,
        5u|0x100u,6u|0x200u,4u|0x100u};
    auto host_found=evaluate(7u,cpu_gu_new,gu_touches,
                             dormant_gu_flags,new_gu_prior);
    no_errors(host_found);
    check_task_mapping(host_found);
    require(root(host_found,3u)==root(host_found,4u) &&
            count(host_found,3u).active_seeds==1u &&
            header(host_found).runnable_components==1u &&
            header(host_found).responding_nodes==3u &&
            header(host_found).admitted_edges==3u,
            "New CPU Gu contact did not wake its accurate device island");
    auto host_persists=evaluate(
        7u,mixed_gu,gu_touches,dormant_gu_flags);
    no_errors(host_persists);
    require(header(host_persists).runnable_components==0u &&
            count(host_persists,3u).active_seeds==0u,
            "Persistent CPU Gu touch repeatedly woke a dormant component");
    auto lost_gu_touches=gu_touches;
    lost_gu_touches[70u]=0u;
    auto disconnected_gu=evaluate(7u,mixed_gu,lost_gu_touches);
    no_errors(disconnected_gu);
    require(root(disconnected_gu,3u)!=root(disconnected_gu,4u) &&
            admitted(disconnected_gu,9u).island_root==invalid,
            "CPU Gu contact bypassed its device physical-touch state");

    auto malformed_host=mixed_gu;
    malformed_host.back().kind=4u;
    auto rejected_host=evaluate(7u,malformed_host,gu_touches);
    require((header(rejected_host).errors&1u)!=0u &&
            header(rejected_host).runnable_components==0u,
            "Source Sc accepted a host physical-touch override");
    auto invalid_host_joint=mixed_gu;
    invalid_host_joint.back().kind=source_sc_device_island_device_joint_bit;
    auto rejected_host_joint=evaluate(
        7u,invalid_host_joint,gu_touches);
    require((header(rejected_host_joint).errors&1u)!=0u &&
            header(rejected_host_joint).runnable_components==0u,
            "Device Sc accepted device-joint metadata on a contact edge");

    auto only_host=evaluate(1u,{
        {0u,0u,invalid,23u,0u},
    },{1u},{4u|0x100u},{0u});
    no_errors(only_host);
    check_task_mapping(only_host);
    require(header(only_host).runnable_components==1u &&
            header(only_host).responding_nodes==1u &&
            header(only_host).admitted_edges==1u,
            "CPU Gu-only scene lost device EdgeIndex history admission");
    auto invalid_handle=evaluate(1u,{
        {invalid,0u,invalid,23u,0u},
    },{1u},{
        4u|0x100u|source_sc_device_island_active_bit});
    require((header(invalid_handle).errors&1u)!=0u &&
            header(invalid_handle).runnable_components==0u &&
            admitted(invalid_handle,0u).island_root==invalid,
            "Invalid Sc EdgeIndex admitted a CPU Gu device task");

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
            task_edges(joints_only,0u)==2u,
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

    // A physically new contact wakes an otherwise dormant dynamic
    // component before Sc has consumed its host Gu witness. A persistent
    // touch does not repeatedly wake it; losing that touch does not
    // synthesize an active numerical task.
    const std::vector<SourceScDeviceIslandEdge> wake_edges{
        {0u,0u,1u,20u,0u},
        {1u,1u,2u,21u,0u},
    };
    const std::vector<std::uint32_t> sleeping_flags{
        4u|0x100u,5u|0x200u,6u|0x100u};
    auto found_wake=evaluate(3u,wake_edges,{1u,1u},
                             sleeping_flags,{0u,1u});
    no_errors(found_wake);
    check_task_mapping(found_wake);
    require(root(found_wake,0u)==0u &&
            root(found_wake,2u)==0u &&
            count(found_wake,0u).active_seeds==1u &&
            header(found_wake).runnable_components==1u &&
            header(found_wake).responding_nodes==3u &&
            header(found_wake).admitted_edges==2u &&
            task(found_wake,0u).position_iterations==6u,
            "New physical touch failed to wake GPU accurate component");
    auto continued=evaluate(3u,wake_edges,{1u,1u},
                            sleeping_flags,{1u,1u});
    no_errors(continued);
    check_task_mapping(continued);
    require(root(continued,0u)==0u &&
            header(continued).runnable_components==0u &&
            count(continued,0u).active_seeds==0u,
            "Persistent physical touch repeatedly woke a dormant island");
    auto removed=evaluate(3u,wake_edges,{0u,1u},
                          sleeping_flags,{1u,1u});
    no_errors(removed);
    check_task_mapping(removed);
    require(root(removed,0u)==0u &&
            root(removed,1u)==1u &&
            header(removed).runnable_components==0u &&
            count(removed,1u).active_seeds==0u,
            "Lost physical touch emitted a spurious device wake task");

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

    // The prefix crosses 128-block boundaries without nondeterministic
    // task scatter or a CPU island count. Almost all nodes are dormant;
    // sparse active roots exercise the first, last and middle blocks.
    constexpr std::uint32_t prefix_nodes=16385u;
    std::vector<std::uint32_t> prefix_flags(prefix_nodes,4u|0x100u);
    for(const std::uint32_t root_id:{0u,127u,128u,16383u,16384u})
        prefix_flags[root_id]|=source_sc_device_island_active_bit;
    const std::vector<SourceScDeviceIslandEdge> prefix_edges{
        {6u,16384u,invalid,42u,source_sc_device_island_joint_bit},
        {4u,128u,invalid,43u,source_sc_device_island_joint_bit},
        {2u,0u,invalid,44u,source_sc_device_island_joint_bit},
        {3u,0u,invalid,45u,source_sc_device_island_joint_bit|
                                source_sc_device_island_no_response_bit},
        {5u,16383u,invalid,46u,source_sc_device_island_joint_bit},
    };
    auto prefix=evaluate(prefix_nodes,prefix_edges,{},prefix_flags);
    no_errors(prefix);
    check_task_mapping(prefix);
    require(header(prefix).runnable_components==5u &&
            header(prefix).dispatch_x==5u &&
            header(prefix).admitted_edges==4u &&
            count(prefix,0u).task_index==0u &&
            count(prefix,127u).task_index==1u &&
            count(prefix,128u).task_index==2u &&
            count(prefix,16383u).task_index==3u &&
            count(prefix,16384u).task_index==4u &&
            task(prefix,0u).descriptor_begin==0u &&
            task(prefix,127u).descriptor_begin==1u &&
            task(prefix,128u).descriptor_begin==1u &&
            task(prefix,16383u).descriptor_begin==2u &&
            task(prefix,16384u).descriptor_begin==3u &&
            count(prefix,16384u).task_index==4u,
            "Device Sc prefix reordered roots or cross-block response offsets");

    // Interleaved response streams span multiple radix tiles and root
    // bytes; dormant and no-response edges sort past runnable work.
    constexpr std::uint32_t radix_nodes=257u;
    std::vector<std::uint32_t> radix_flags(radix_nodes,4u|0x100u);
    for(const auto id:{0u,127u,256u})
        radix_flags[id]|=source_sc_device_island_active_bit;
    std::vector<SourceScDeviceIslandEdge> radix_edges;
    for(std::uint32_t i=0u;i<517u;++i) {
        const auto owner=i%4u==0u?0u:i%4u==1u?256u:
                         i%4u==2u?127u:1u;
        radix_edges.push_back({i,owner,invalid,517u-i,
            source_sc_device_island_joint_bit|
            (i%17u==0u?source_sc_device_island_no_response_bit:0u)});
    }
    auto radix=evaluate(radix_nodes,radix_edges,{},radix_flags);
    no_errors(radix);
    check_task_mapping(radix);
    require(header(radix).runnable_components==3u &&
            header(radix).admitted_edges>300u,
            "Device Sc multi-tile response scatter missed runnable components");

    // An odd number of radix passes must publish the scratch output
    // as the final response stream.
    std::vector<std::uint32_t> short_flags(5u,4u|0x100u);
    short_flags[0u]|=source_sc_device_island_active_bit;
    short_flags[4u]|=source_sc_device_island_active_bit;
    std::vector<SourceScDeviceIslandEdge> short_edges;
    for(std::uint32_t i=0u;i<385u;++i)
        short_edges.push_back({i,i%3u==0u?0u:i%3u==1u?4u:3u,
            invalid,385u-i,source_sc_device_island_joint_bit|
            (i%13u==0u?source_sc_device_island_no_response_bit:0u)});
    auto short_radix=evaluate(5u,short_edges,{},short_flags);
    no_errors(short_radix);
    check_task_mapping(short_radix);
    require(header(short_radix).runnable_components==2u &&
            short_radix.first.response_order!=
                short_radix.first.response_scratch,
            "Device Sc single-pass radix output did not select scratch");

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
            task_edges(long_chain,0u)==chain_nodes-1u &&
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

    // Two moving rigid bodies attached to one kinematic node must not
    // merge through that nonresponding endpoint. The kinematic neither
    // acquires a solver-body ordinal nor shifts the authored body order.
    const std::vector<std::uint32_t> kinematic_flags{
        4u|0x100u|source_sc_device_island_active_bit,
        source_sc_device_island_nonresponding_bit,
        5u|0x200u|source_sc_device_island_active_bit};
    const std::vector<SourceScDeviceIslandEdge> kinematic_joints{
        {0u,0u,1u,0u,source_sc_device_island_joint_bit|
             source_sc_device_island_no_response_bit},
        {1u,2u,1u,0u,source_sc_device_island_joint_bit|
             source_sc_device_island_no_response_bit}};
    const auto independent_dynamic=evaluate(
        3u,kinematic_joints,{},kinematic_flags);
    no_errors(independent_dynamic);
    check_task_mapping(independent_dynamic);
    require(root(independent_dynamic,0u)!=root(independent_dynamic,2u) &&
            count(independent_dynamic,1u).responding_nodes==0u &&
            header(independent_dynamic).runnable_components==2u &&
            header(independent_dynamic).responding_nodes==2u &&
            load<std::uint32_t>(independent_dynamic.second,
                independent_dynamic.first.body_order)==0u &&
            load<std::uint32_t>(independent_dynamic.second,
                independent_dynamic.first.body_order+4u)==2u,
            "Nonresponding kinematic incorrectly united dynamic islands");

    // A touching contact with that endpoint still admits its dynamic
    // response token and body, without integrating the kinematic row.
    const auto dynamic_kinematic_contact=evaluate(
        3u,{{0u,0u,1u,0u,0u}},{1u},kinematic_flags,{0u},
        SourceScDeviceIslandSources{1u,0u,0u});
    no_errors(dynamic_kinematic_contact);
    check_task_mapping(dynamic_kinematic_contact);
    require(header(dynamic_kinematic_contact).runnable_components==2u &&
            header(dynamic_kinematic_contact).responding_nodes==2u &&
            header(dynamic_kinematic_contact).admitted_edges==1u &&
            count(dynamic_kinematic_contact,0u).admitted_edges==1u,
            "Kinematic contact lost its dynamic numerical admission");

    // A source descriptor index needs 30 bits; larger identifiers
    // must invalidate indirect numerical launch, not alias its tag.
    auto oversized_source=edges;
    oversized_source[0].source_index=0x40000000u;
    const auto invalid_token=evaluate(7u,oversized_source,
                                      {1u,0u,1u,1u,0u,1u});
    require((header(invalid_token).errors&64u)!=0u &&
            header(invalid_token).dispatch_x==0u,
            "Device Sc source token overflow could launch the wrong constraint");

    // Malformed generation/edge/node handles must not silently join
    // unrelated solver bodies or access unallocated GPU touch history.
    auto bad_edges=edges;
    bad_edges[0].edge_index=128u;
    bad_edges[1].node_b=1024u;
    bad_edges[2].kind=8u;
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
