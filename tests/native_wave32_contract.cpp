// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    try {
        using namespace fyzz::compute;
        if(argc!=2)throw std::invalid_argument("Expected the compiled wave32 shader");
        std::ifstream file(argv[1],std::ios::binary);
        if(!file)throw std::runtime_error("Wave32 shader missing");
        const std::vector<char> bytes(std::istreambuf_iterator<char>{file},
                                      std::istreambuf_iterator<char>{});
        Context context(
#ifdef FYZZXYZZY_TEST_D3D12
            Backend::D3D12
#else
            Backend::Vulkan
#endif
        );
        const auto& info=context.device_info();
        std::cout << "Wave capability device=\"" << info.name
                  << "\" default=" << info.subgroup_size
                  << " min=" << info.min_subgroup_size
                  << " max=" << info.max_subgroup_size
                  << " selectable=" << info.supports_required_subgroup_size
                  << '\n';
        ComputePipelineDesc desc;
#ifdef FYZZXYZZY_TEST_D3D12
        desc.shader.format=ShaderFormat::Dxil;
#else
        desc.shader.format=ShaderFormat::Spirv;
#endif
        desc.shader.bytes.assign(bytes.begin(),bytes.end());
        desc.storage_buffer_binding_count=1u;
        desc.push_constant_bytes=16u;

        desc.required_subgroup_size=3u;
        bool rejected=false;
        try {(void)context.create_compute_pipeline(desc);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Non-power-of-two wave width admitted");

        desc.required_subgroup_size=32u;
        const bool supported=info.subgroup_size==32u ||
            (info.supports_required_subgroup_size &&
             info.min_subgroup_size<=32u &&
             info.max_subgroup_size>=32u);
        if(!supported) {
            rejected=false;
            try {(void)context.create_compute_pipeline(desc);}
            catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error(
                "Wave32 shader accepted by unsupported compute adapter");
            std::cout << "Wave32 correctly rejected by this adapter\n";
            return 0;
        }

        const auto pipeline=context.create_compute_pipeline(desc);
        std::array<std::uint32_t,4> output{0u,0u,0u,0u};
        const auto buffer=context.create_buffer(sizeof(output),MemoryPlacement::DeviceLocal);
        context.upload(buffer,0u,output.data(),sizeof(output));
        const std::array<std::uint32_t,4> params{4u,0u,0u,0u};
        context.dispatch(pipeline,{1u,1u,1u},
                         {{0u,buffer,0u,sizeof(output),BufferAccess::Write}},
                         params.data(),sizeof(params));
        context.download(buffer,0u,output.data(),sizeof(output));
        if(output[0]!=0u || output[1]!=32u ||
           output[2]!=0u || output[3]!=0u)
            throw std::runtime_error("Compute shader did not execute at wave32");
        context.destroy_buffer(buffer);
        context.destroy_compute_pipeline(pipeline);
        std::cout << "Native wave32 pipeline admission and execution PASS\n";
        return 0;
    }catch(const std::exception& ex){
        std::cerr << "Native wave32 contract FAILED: " << ex.what() << '\n';
        return 1;
    }
}
