// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <iostream>
#include <string_view>
#include <utility>

namespace fyzz::tests {
template<class F>
void measure_gpu_case(std::string_view suite,std::string_view name,F&& run) {
    const auto start=std::chrono::steady_clock::now();
    try {
        std::forward<F>(run)();
    }catch(...) {
        std::cerr<<"GPU regression failed: suite="<<suite
                 <<" case="<<name<<'\n';
        throw;
    }
    const auto elapsed=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    std::cout<<"FYZZXYZZY_METRIC suite="<<suite<<" case="<<name
             <<" host_wall_ms="<<elapsed<<'\n';
}
} // namespace fyzz::tests
