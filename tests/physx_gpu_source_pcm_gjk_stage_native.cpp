// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_pcm_box_candidates_native.h"
#include "physx_gpu/source_pcm_epa_native.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace fyzz::compute;
using namespace fyzz::physx_gpu;

void require(bool condition, const char* message) {
    if(!condition)throw std::runtime_error(message);
}
ShaderBinary load_shader(const char* path) {
    std::ifstream input(path,std::ios::binary);
    require(bool(input),"Missing GJK stage DXIL");
    ShaderBinary shader;
    shader.format=ShaderFormat::Dxil;
    shader.bytes.assign(std::istreambuf_iterator<char>{input},
                        std::istreambuf_iterator<char>{});
    require(!shader.bytes.empty(),"Empty GJK stage DXIL");
    return shader;
}
struct Transform {
    float orientation[4];
    float position[3];
    std::uint32_t flags=0u;
};
struct Geometry {
    float half_extents[3];
    float contact_distance=0.f;
    std::uint32_t other[12]{};
};
struct GjkStateSample {
    std::uint32_t count=0u;
    float closest[3]{};
    float previous=0.f;
    float weights[4]{};
    float distance_squared=0.f;
    std::uint32_t mask=0u,phase=0u,outcome=0u;
    float projection=0.f,contact_limit=0.f;
    std::uint32_t point_core=0u;
};
static_assert(sizeof(Transform)==32u);
static_assert(sizeof(Geometry)==64u);
static_assert(sizeof(GjkStateSample)==64u);

void verify(Context& context,const char* advance_dxil,
            const char* reduce_dxil,bool run_reduction) {
    // These two unit boxes penetrate by 0.5 along X. Their first support
    // is (+0.5, 0, 0) in the second box's frame.
    const std::array<Transform,2> transforms{{
        {{0.f,0.f,0.f,1.f},{0.f,0.f,0.f}},
        {{0.f,0.f,0.f,1.f},{1.5f,0.f,0.f}}
    }};
    const std::array<Geometry,2> geometry{{
        {{1.f,1.f,1.f},0.02f},
        {{1.f,1.f,1.f},0.02f}
    }};
    const SourcePcmBoxPairNative pair{0u,1u,0u,0u,1u};
    SourcePcmBoxPairResultNative result{};
    result.edge_handle=0u;
    result.status=3u;
    SourcePcmScEdgeCacheNative edge{};
    edge.initialized=1u;
    edge.needs_gjk=1u;
    constexpr auto aligned=[](std::size_t size) {
        return (size+15u)&~std::size_t{15u};
    };
    constexpr std::size_t transforms_offset=0u;
    constexpr std::size_t geometry_offset=aligned(transforms_offset+sizeof(transforms));
    constexpr std::size_t pair_offset=aligned(geometry_offset+sizeof(geometry));
    constexpr std::size_t result_offset=aligned(pair_offset+sizeof(pair));
    constexpr std::size_t candidate_offset=aligned(result_offset+sizeof(result));
    constexpr std::size_t edge_offset=aligned(candidate_offset+40u);
    constexpr std::size_t work_offset=aligned(edge_offset+sizeof(edge));
    constexpr std::size_t output_offset=aligned(work_offset+4u);
    constexpr std::size_t total_bytes=output_offset+84u;
    static_assert(total_bytes%4u==0u);
    std::array<std::uint8_t,total_bytes> data{};
    std::memcpy(data.data()+transforms_offset,transforms.data(),sizeof(transforms));
    std::memcpy(data.data()+geometry_offset,geometry.data(),sizeof(geometry));
    std::memcpy(data.data()+pair_offset,&pair,sizeof(pair));
    std::memcpy(data.data()+result_offset,&result,sizeof(result));
    std::memcpy(data.data()+edge_offset,&edge,sizeof(edge));

    const auto root=context.create_buffer(data.size(),MemoryPlacement::DeviceLocal);
    context.upload(root,0u,data.data(),data.size());
    std::clog<<"GJK stage: source state uploaded\n"<<std::flush;
    CudaDeviceAddressMap addresses;
    const auto transforms_address=addresses.register_slice(
        root,transforms_offset,sizeof(transforms));
    const auto geometry_address=addresses.register_slice(
        root,geometry_offset,sizeof(geometry));
    const auto pair_address=addresses.register_slice(root,pair_offset,sizeof(pair));
    const auto result_address=addresses.register_slice(root,result_offset,sizeof(result));
    const auto candidate_address=addresses.register_slice(root,candidate_offset,40u);
    const auto edge_address=addresses.register_slice(root,edge_offset,sizeof(edge));
    const auto work_address=addresses.register_slice(root,work_offset,4u);
    const auto output_address=addresses.register_slice(root,output_offset,84u);
    {
        SourcePcmEpaNativeStage advance(context,load_shader(advance_dxil),false,true);
        std::clog<<"GJK stage: advance pipeline initialized\n"<<std::flush;
        const auto execute=[&](const SourcePcmEpaNativeStage& stage,
                               const char* name) {
            SourceSapDmaNative stream(context,addresses);
            stage.append(stream,addresses,transforms_address,geometry_address,
                         pair_address,result_address,candidate_address,
                         edge_address,work_address,0u,output_address,
                         transforms_address,0u,1u,2u,1u,1u);
            std::clog<<"GJK stage: submitting "<<name<<"\n"<<std::flush;
            stream.submit();
            std::clog<<"GJK stage: waiting for "<<name<<"\n"<<std::flush;
            stream.wait_at_fetch();
            std::clog<<"GJK stage: completed "<<name<<"\n"<<std::flush;
        };
        const auto sample=[&] {
            GjkStateSample state{};
            context.download(root,edge_offset+160u,&state,sizeof(state));
            require(state.count==1u && std::isfinite(state.closest[0]) &&
                    std::isfinite(state.closest[1]) &&
                    std::isfinite(state.closest[2]),
                    "GJK stage wrote an invalid simplex");
            return state;
        };
        execute(advance,"advance");
        const auto after_advance=sample();
        require(after_advance.phase==1u &&
                std::abs(after_advance.closest[0]+1.5f)<1.e-5f,
                "GJK advance did not admit the penetrating simplex");
        if(run_reduction) {
            SourcePcmEpaNativeStage reduce(context,load_shader(reduce_dxil),false,true);
            std::clog<<"GJK stage: reducer pipeline initialized\n"<<std::flush;
            execute(reduce,"reduction");
            const auto after_reduce=sample();
            require(after_reduce.phase==0u &&
                    after_reduce.mask==1u &&
                    std::abs(after_reduce.closest[0]-0.5f)<1.e-5f &&
                    std::abs(after_reduce.distance_squared-0.25f)<1.e-5f &&
                    std::abs(after_reduce.weights[0]-1.f)<1.e-5f,
                    "GJK reduction differs from the analytic support witness");
        }
    }
    addresses.unregister_slice(output_address);
    addresses.unregister_slice(work_address);
    addresses.unregister_slice(edge_address);
    addresses.unregister_slice(candidate_address);
    addresses.unregister_slice(result_address);
    addresses.unregister_slice(pair_address);
    addresses.unregister_slice(geometry_address);
    addresses.unregister_slice(transforms_address);
    require(addresses.live_allocations()==0u,"GJK fixture leaked device mappings");
    context.destroy_buffer(root);
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==4,"Expected advance shader, reducer shader, and stage");
        const std::string stage=argv[3];
        require(stage=="advance" || stage=="reduction","Unknown GJK stage");
        Context context(Backend::D3D12);
        verify(context,argv[1],argv[2],stage=="reduction");
        std::cout<<"Source GJK stage validation PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Source GJK stage validation FAIL: "<<e.what()<<"\n";
        return 1;
    }
}
