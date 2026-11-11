// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/compute.h"
#include "physx_gpu/source_sc_box_transform_bounds_native.h"
#include "physx_gpu/source_frame_input_arena_native.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace fyzz::compute;
using fyzz::physx_gpu::SourceScBoxTransformInputNative;

struct Bounds {
    float minimum[3];
    float maximum[3];
};
struct CachedTransform {
    float orientation[4];
    float position[3];
    std::uint32_t flags;
};
static_assert(sizeof(Bounds)==24u);
static_assert(sizeof(CachedTransform)==32u);

void require(bool condition,const char* error) {
    if(!condition)throw std::runtime_error(error);
}
ShaderBinary load(const char* path) {
    std::ifstream stream(path,std::ios::binary);
    require(bool(stream),"Source Sc GPU box transform shader missing");
    ShaderBinary binary;
#ifdef FYZZXYZZY_TEST_D3D12
    binary.format=ShaderFormat::Dxil;
#else
    binary.format=ShaderFormat::Spirv;
#endif
    binary.bytes.assign(std::istreambuf_iterator<char>{stream},
                        std::istreambuf_iterator<char>{});
    require(!binary.bytes.empty(),"Source Sc box transform shader empty");
    return binary;
}

void verify(Context& context,const char* shader) {
    using namespace fyzz::physx_gpu;
    constexpr float c=0.7071067811865476f;
    std::array<SourceScBoxTransformInputNative,4> shapes{{
        {1u,{2.f,3.f,4.f},{0.f,0.f,0.f,1.f},
         {1.f,2.f,3.f},0.25f},
        {6u,{4.f,-2.f,0.f},{0.f,0.f,c,c},
         {2.f,1.f,3.f},0.5f},
        {31u,{0.f,10.f,0.f},{c,0.f,0.f,c},
         {1.f,2.f,3.f},0.25f},
        {12u,{1.5f,-2.2f,0.75f},
         {0.09539871f,0.19079742f,0.28619614f,0.93412376f},
         {0.5f,1.2f,2.3f},1.25f}
    }};
    shapes[0].local_position[0]=0.75f;
    shapes[0].local_position[1]=-0.25f;
    shapes[0].local_orientation[2]=c;shapes[0].local_orientation[3]=c;
    shapes[0].next_element=6u;
    constexpr std::uint32_t capacity=32u;
    constexpr Bounds unchanged{{-88.f,-77.f,-66.f},{88.f,77.f,66.f}};
    std::array<Bounds,capacity> bounds;
    bounds.fill(unchanged);
    CachedTransform invalid{};
    invalid.flags=0xdeadbeefu;
    std::array<CachedTransform,capacity> transforms;
    transforms.fill(invalid);
    const std::size_t input_offset=4u;
    const std::size_t bounds_offset=input_offset+sizeof(shapes)+12u;
    const std::size_t cache_offset=bounds_offset+sizeof(bounds)+4u;
    const std::size_t geometry_offset=cache_offset+sizeof(transforms)+4u;
    std::array<SourceScBoxGeometryNative,capacity> geometry{};
    for(auto& value:geometry)value.contact_distance=-1.f;
    const auto root=context.create_buffer(geometry_offset+sizeof(geometry)+16u);
    context.upload(root,input_offset,shapes.data(),sizeof(shapes));
    context.upload(root,bounds_offset,bounds.data(),sizeof(bounds));
    context.upload(root,cache_offset,transforms.data(),sizeof(transforms));
    context.upload(root,geometry_offset,geometry.data(),sizeof(geometry));

    CudaDeviceAddressMap addresses;
    const auto input=addresses.register_slice(
        root,input_offset,sizeof(shapes));
    const auto output=addresses.register_slice(
        root,bounds_offset,sizeof(bounds));
    const auto cache=addresses.register_slice(
        root,cache_offset,sizeof(transforms));
    const auto shape_geometry=addresses.register_slice(
        root,geometry_offset,sizeof(geometry));
    SourceSapDmaNative stream(context,addresses);
    {
        SourceScBoxTransformBoundsNativeStage stage(context,load(shader));
        stage.append(stream,addresses,input,output,cache,shape_geometry,
                     input,0u,
                     static_cast<std::uint32_t>(shapes.size()),capacity);
        require(stream.commands().size()==1u,
                "Sc box transform did not enter source GPU stream");
        stream.wait_at_fetch();
        require(stream.submission_stats().command_batches==1u,
                "Sc box transform introduced an extra GPU submission");
        context.download(root,bounds_offset,bounds.data(),sizeof(bounds));
        context.download(root,cache_offset,transforms.data(),sizeof(transforms));
        context.download(root,geometry_offset,geometry.data(),sizeof(geometry));

        const auto near=[](float a,float b) {
            return std::abs(a-b)<0.0001f;
        };
        const auto check_bounds=[&](std::uint32_t i,
                                    std::array<float,3> low,
                                    std::array<float,3> high) {
            for(unsigned axis=0;axis<3u;++axis)
                require(near(bounds[i].minimum[axis],low[axis]) &&
                        near(bounds[i].maximum[axis],high[axis]),
                        "Sc source GPU box bounds differ from PxBounds3::poseExtent");
        };
        check_bounds(1u,{1.f,1.f,1.f},{3.f,5.f,7.f});
        check_bounds(6u,{3.f,-4.f,-3.f},{5.f,0.f,3.f});
        check_bounds(31u,{-1.f,7.f,-2.f},{1.f,13.f,2.f});
        check_bounds(12u,{-0.42506928f,-3.62586260f,-1.83653189f},
                         {3.42506928f,-0.77413740f,3.33653189f});
        for(std::size_t shape=0;shape<shapes.size();++shape) {
            const auto index=shapes[shape].element_index;
            for(unsigned i=0;i<4u;++i)
                require(near(transforms[index].orientation[i],
                             shapes[shape].orientation[i]),
                        "GPU source transform cache lost Sc quaternion");
            for(unsigned i=0;i<3u;++i)
                require(near(transforms[index].position[i],
                             shapes[shape].position[i]),
                        "GPU source transform cache lost Sc translation");
            require(geometry[index].next_element==shapes[shape].next_element &&
                    geometry[index].geometry_kind==shapes[shape].geometry_kind,
                    "GPU shape geometry lost its attachment link or type");
            for(unsigned axis=0;axis<3u;++axis)
                require(near(geometry[index].local_position[axis],shapes[shape].local_position[axis]),
                        "GPU shape geometry lost its local translation");
            for(unsigned axis=0;axis<4u;++axis)
                require(near(geometry[index].local_orientation[axis],shapes[shape].local_orientation[axis]),
                        "GPU shape geometry lost its local orientation");
            require(transforms[index].flags==0u,
                    "GPU source transform cache flags not initialized");
            for(unsigned axis=0;axis<3u;++axis)
                require(near(geometry[index].half_extents[axis],
                             shapes[shape].half_extents[axis]),
                        "GPU shape geometry cache lost Sc extents");
            require(near(geometry[index].contact_distance,
                         shapes[shape].contact_distance),
                    "GPU shape geometry cache lost source contact distance");
        }
        require(bounds[8u].minimum[0]==unchanged.minimum[0] &&
                transforms[8u].flags==invalid.flags &&
                geometry[8u].contact_distance==-1.f,
                "Unchanged source element cache or AABB was overwritten");

        shapes[1].position[0]=7.f;
        shapes[1].contact_distance=0.625f;
        context.upload(root,input_offset+sizeof(shapes[0]),
                       &shapes[1],sizeof(shapes[1]));
        stage.append(stream,addresses,input+sizeof(shapes[0]),
                     output,cache,shape_geometry,input,0u,1u,capacity);
        stream.wait_at_fetch();
        context.download(root,bounds_offset,bounds.data(),sizeof(bounds));
        context.download(root,cache_offset,transforms.data(),sizeof(transforms));
        context.download(root,geometry_offset,geometry.data(),sizeof(geometry));
        check_bounds(6u,{6.f,-4.f,-3.f},{8.f,0.f,3.f});
        require(near(transforms[6u].position[0],7.f) &&
                near(geometry[6u].contact_distance,0.625f) &&
                near(geometry[1u].contact_distance,0.25f) &&
                bounds[1u].minimum[0]==1.f &&
                bounds[31u].minimum[1]==7.f,
                "Sparse device transform update corrupted quiet source shapes");
        bool rejected=false;
        try {
            stage.append(stream,addresses,input,output,cache,shape_geometry,input,0u,33u,capacity);
        }catch(const std::invalid_argument&) {rejected=true;}
        require(rejected && stream.commands().empty(),
                "Invalid source transform capacity partially enqueued GPU work");
    }
    addresses.unregister_slice(shape_geometry);
    addresses.unregister_slice(cache);
    addresses.unregister_slice(output);
    addresses.unregister_slice(input);
    require(addresses.live_allocations()==0u,
            "Source box transform test leaked GPU address mappings");
    context.destroy_buffer(root);
}
void verify_convex_bounds(Context& context,const char* shader) {
    using namespace fyzz::physx_gpu;
    constexpr float c=0.7071067811865476f;
    SourceScBoxTransformInputNative shape{};
    shape.element_index=0u;
    shape.position[0]=5.f;
    shape.position[1]=2.f;
    shape.position[2]=-3.f;
    shape.orientation[1]=c;
    shape.orientation[3]=c;
    shape.half_extents[0]=1.f;
    shape.half_extents[1]=1.f;
    shape.half_extents[2]=1.f;
    shape.geometry_kind=2u;
    shape.reserved[0]=0u;
    shape.reserved[1]=4u;
    constexpr std::array<fyzz::Vec3,4> vertices{{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1}}};
    constexpr std::size_t inputOffset=4u;
    constexpr std::size_t boundsOffset=inputOffset+sizeof(shape)+16u;
    constexpr std::size_t cacheOffset=boundsOffset+sizeof(Bounds)+16u;
    constexpr std::size_t geometryOffset=cacheOffset+sizeof(CachedTransform)+16u;
    constexpr std::size_t verticesOffset=
        geometryOffset+sizeof(SourceScBoxGeometryNative)+16u;
    constexpr std::size_t allocation=verticesOffset+sizeof(vertices)+16u;
    const auto buffer=context.create_buffer(allocation);
    context.upload(buffer,inputOffset,&shape,sizeof(shape));
    context.upload(buffer,verticesOffset,vertices.data(),sizeof(vertices));
    CudaDeviceAddressMap addresses;
    const auto input=addresses.register_slice(buffer,inputOffset,sizeof(shape));
    const auto bounds=addresses.register_slice(buffer,boundsOffset,sizeof(Bounds));
    const auto cache=addresses.register_slice(buffer,cacheOffset,sizeof(CachedTransform));
    const auto geometry=addresses.register_slice(
        buffer,geometryOffset,sizeof(SourceScBoxGeometryNative));
    const auto hull=addresses.register_slice(
        buffer,verticesOffset,sizeof(vertices));
    SourceScBoxTransformBoundsNativeStage stage(context,load(shader));
    SourceSapDmaNative stream(context,addresses);
    stage.append(stream,addresses,input,bounds,cache,geometry,
                 hull,static_cast<std::uint32_t>(vertices.size()),1u,1u);
    stream.wait_at_fetch();
    Bounds result{};
    SourceScBoxGeometryNative stored{};
    context.download(buffer,boundsOffset,&result,sizeof(result));
    context.download(buffer,geometryOffset,&stored,sizeof(stored));
    const auto near=[](float a,float b){return std::abs(a-b)<0.0001f;};
    const std::array<float,3> expectedMin{5.f,2.f,-4.f};
    const std::array<float,3> expectedMax{6.f,3.f,-3.f};
    for(unsigned axis=0u;axis<3u;++axis)
        require(near(result.minimum[axis],expectedMin[axis]) &&
                near(result.maximum[axis],expectedMax[axis]),
            "Rotated convex hull bounds differ from actual transformed vertices");
    require(stored.geometry_kind==2u &&
            stored.reserved[0]==0u && stored.reserved[1]==vertices.size(),
        "Sc geometry lost convex resident vertex span");
    addresses.unregister_slice(hull);
    addresses.unregister_slice(geometry);
    addresses.unregister_slice(cache);
    addresses.unregister_slice(bounds);
    addresses.unregister_slice(input);
    context.destroy_buffer(buffer);
}

void verify_direct_frame_input(Context& context,const char* shader) {
    using namespace fyzz::physx_gpu;
    CudaDeviceAddressMap addresses;
    SourceSapDmaNative stream(context,addresses);
    SourceFrameInputArenaNative input(context,addresses,stream);
    if(!input.direct())return;

    constexpr std::uint32_t count=8u;
    const auto bounds_buffer=context.create_buffer(
        sizeof(Bounds)*count,MemoryPlacement::DeviceLocal);
    const auto cache_buffer=context.create_buffer(
        sizeof(CachedTransform)*count,MemoryPlacement::DeviceLocal);
    const auto geometry_buffer=context.create_buffer(
        sizeof(SourceScBoxGeometryNative)*count,MemoryPlacement::DeviceLocal);
    const auto bounds=addresses.register_allocation(bounds_buffer,sizeof(Bounds)*count);
    const auto cache=addresses.register_allocation(cache_buffer,sizeof(CachedTransform)*count);
    const auto geometry=addresses.register_allocation(
        geometry_buffer,sizeof(SourceScBoxGeometryNative)*count);

    {
        SourceScBoxTransformBoundsNativeStage stage(context,load(shader));
        std::array<SourceScBoxTransformInputNative,count> shapes{};
        for(std::uint32_t i=0u;i<count;++i) {
            shapes[i].element_index=i;
            shapes[i].position[0]=static_cast<float>(i*3u);
            shapes[i].orientation[3]=1.f;
            shapes[i].half_extents[0]=1.f;
            shapes[i].half_extents[1]=2.f;
            shapes[i].half_extents[2]=3.f;
        }

        const auto first=input.write(shapes.data(),sizeof(shapes[0]));
        stage.append(stream,addresses,first,bounds,cache,geometry,first,
                     0u,1u,count);
        require(stream.submission_stats().upload_bytes==0u,
                "Mapped input triggered a source transfer");
        stream.wait_at_fetch();
        Bounds result{};
        context.download(bounds_buffer,0u,&result,sizeof(result));
        require(std::abs(result.minimum[0]+1.f)<1.0e-5f &&
                std::abs(result.maximum[0]-1.f)<1.0e-5f,
                "Mapped frame input failed first GPU transform");

        const auto grown=input.write(shapes.data(),sizeof(shapes));
        require(grown!=first && input.capacity()>=sizeof(shapes),
                "Frame input growth reused an invalid previous device address");
        stage.append(stream,addresses,grown,bounds,cache,geometry,grown,
                     0u,count,count);
        stream.wait_at_fetch();
        std::array<Bounds,count> results{};
        context.download(bounds_buffer,0u,results.data(),sizeof(results));
        for(std::uint32_t i=0u;i<count;++i)
            require(std::abs(results[i].minimum[0]-(float(i*3u)-1.f))<1.0e-5f &&
                    std::abs(results[i].maximum[0]-(float(i*3u)+1.f))<1.0e-5f,
                    "Mapped frame input lost source order during growth");

        shapes[0].position[0]=42.f;
        const auto reused=input.write(shapes.data(),sizeof(shapes));
        require(reused==grown,"Stable mapped frame input changed its allocation");
        stage.append(stream,addresses,reused,bounds,cache,geometry,reused,
                     0u,count,count);
        stream.wait_at_fetch();
        context.download(bounds_buffer,0u,&result,sizeof(result));
        require(std::abs(result.minimum[0]-41.f)<1.0e-5f &&
                std::abs(result.maximum[0]-43.f)<1.0e-5f &&
                stream.submission_stats().upload_bytes==0u,
                "Mapped frame input did not update without GPU transfer");
    }
    input.close();
    addresses.unregister_allocation(geometry);
    addresses.unregister_allocation(cache);
    addresses.unregister_allocation(bounds);
    context.destroy_buffer(geometry_buffer);
    context.destroy_buffer(cache_buffer);
    context.destroy_buffer(bounds_buffer);
}

} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected compiled Sc box transform shader");
#ifdef FYZZXYZZY_TEST_D3D12
        Context context(Backend::D3D12);
#else
        Context context(Backend::Vulkan);
#endif
        verify(context,argv[1]);
        verify_convex_bounds(context,argv[1]);
        verify_direct_frame_input(context,argv[1]);
        std::cout<<"Source GPU box/convex transform/cache/bounds PASS\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<"Source GPU box transform/cache/bounds FAILED: "
                 <<e.what()<<'\n';
        return 1;
    }
}
