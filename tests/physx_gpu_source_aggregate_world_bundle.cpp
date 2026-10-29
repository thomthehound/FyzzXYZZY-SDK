// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#include "fyzzxyzzy/native_gpu_world.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void require(bool pass,const char* message) {
    if(!pass)throw std::runtime_error(message);
}
void run(fyzz::compute::Backend backend,
         const char* sap_manifest,const char* aggregate_manifest) {
    bool rejected=false;
    try {
        (void)fyzz::load_native_gpu_aggregate_shaders(
            backend,sap_manifest);
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected,
        "Original SAP manifest was accepted as the aggregate source bundle");
    auto sap=fyzz::load_native_gpu_sap_shaders(backend,sap_manifest);
    auto aggregate=fyzz::load_native_gpu_aggregate_shaders(
        backend,aggregate_manifest);
    require(aggregate.size()==12u,
        "Original aggregate GPU bundle lost a required source stage");

    // The shader validation is performed before creating any original
    // PhysX GPU heap, source stream, or Sc scene.
    auto invalid=aggregate;
    invalid.back().bytes.clear();
    rejected=false;
    try {
        (void)fyzz::create_native_gpu_world(
            backend,sap,std::move(invalid));
    }catch(const std::invalid_argument&) {
        rejected=true;
    }
    require(rejected,
        "Incomplete aggregate GPU shader set reached source Sc admission");

    // This constructs all twelve real backend pipelines in the same
    // Context as the thirty-four source SAP kernels. The ordinary
    // actor-only scene continues to use exactly the same Sc/PCM/Dy path.
    auto world=fyzz::create_native_gpu_world(
        backend,std::move(sap),std::move(aggregate));
    const auto body=world.add_dynamic_box(
        {0.25f,0.25f,0.25f},{0.0f,2.0f,0.0f},1.0f);
    world.step(1.0f/120.0f);
    const auto state=world.body_state(body);
    require(std::isfinite(state.position.y) && state.position.y<2.0f,
        "Aggregate-enabled source GPU World changed ordinary Sc body integration");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)
            throw std::invalid_argument(
                "Expected source SAP and aggregate GPU shader manifests");
#if defined(FYZZXYZZY_TEST_D3D12)
        run(fyzz::compute::Backend::D3D12,argv[1],argv[2]);
#else
        run(fyzz::compute::Backend::Vulkan,argv[1],argv[2]);
#endif
        std::cout<<"Original source aggregate GPU World stage ownership PASS\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<"Original source aggregate GPU World stage ownership FAILED: "
                 <<error.what()<<'\n';
        return 1;
    }
}
