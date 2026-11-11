// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "../src/hardware_profile.h"

#if !defined(FYZZXYZZY_PROFILE_GPU)
#error Profiling stream regression requires FYZZXYZZY_PROFILE_GPU
#endif

#include <cstdint>
#include <iostream>

template<class Event,class MakeEvent>
bool verify_stream(std::uint64_t count,MakeEvent make) {
    fyzz::hardware_profile::EventStream<Event> stream;
    for(std::uint64_t i=0u;i<count;++i) {
        if(!stream.append(make(i)))return false;
    }
    if(stream.size()!=count)return false;
    for(std::uint64_t i=0u;i<count;++i) {
        if(stream[i].trace_id!=i+1u)return false;
    }
    stream.clear();
    return stream.size()==0u && stream.append(make(count)) &&
           stream[0].trace_id==count+1u;
}

int main() {
    using namespace fyzz::hardware_profile;
    const bool gpu=verify_stream<GpuEvent>(131072u,[](std::uint64_t i) {
        GpuEvent event{};
        event.trace_id=i+1u;
        event.op=i;
        return event;
    });
    const bool buffers=verify_stream<BufferEvent>(65536u,[](std::uint64_t i) {
        BufferEvent event{};
        event.trace_id=i+1u;
        event.buffer=i;
        return event;
    });
    const bool cpu=verify_stream<CpuEvent>(32768u,[](std::uint64_t i) {
        CpuEvent event{};
        event.trace_id=i+1u;
        event.begin=i;
        return event;
    });
    if(!gpu || !buffers || !cpu) {
        std::cerr<<"Profiling event stream did not preserve its records\n";
        return 1;
    }
    std::cout<<"Profiling event stream PASS\n";
    return 0;
}
