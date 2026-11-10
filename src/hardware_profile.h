// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(FYZZXYZZY_PROFILE_GPU)
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace fyzz::hardware_profile {

using Clock=std::chrono::steady_clock;

inline std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

struct Event {
    const char* label=nullptr;
    std::uint64_t begin=0u;
    std::uint64_t end=0u;
};
struct Counter {
    const char* label=nullptr;
    std::uint64_t value=0u;
};
struct Frame {
    std::array<Event,512> events{};
    std::array<Counter,64> counters{};
    std::size_t event_count=0u;
    std::size_t counter_count=0u;
    std::uint64_t begin=0u;
    std::uint64_t trace_id=0u;
    bool active=false;
};
inline thread_local Frame current;
inline std::atomic<std::uint64_t> next_trace_id{1u};
inline std::uint64_t current_trace_id() noexcept {
    return current.active?current.trace_id:0u;
}

inline void begin_frame() noexcept {
    current.event_count=0u;
    current.counter_count=0u;
    current.begin=now_ns();
    current.trace_id=next_trace_id.fetch_add(1u,std::memory_order_relaxed);
    current.active=true;
}
struct Scope {
    const char* label;
    std::uint64_t begin;
    explicit Scope(const char* name) noexcept
        : label(name),begin(current.active?now_ns():0u) {}
    ~Scope() noexcept {
        if(!begin || !current.active ||
           current.event_count==current.events.size())return;
        current.events[current.event_count++]={label,begin,now_ns()};
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
inline void counter(const char* label,std::uint64_t value) noexcept {
    if(current.active && current.counter_count<current.counters.size())
        current.counters[current.counter_count++]={label,value};
}
inline void end_frame(std::uint64_t frame,bool gpu_dy,bool success) noexcept {
    if(!current.active)return;
    const auto end=now_ns();
    current.active=false;
    const auto origin=current.begin;
    std::printf(
        "FYZZXYZZY_TRACE kind=frame trace_id=%llu frame=%llu gpu_dy=%u success=%u start_ns=%llu end_ns=%llu\n",
        static_cast<unsigned long long>(current.trace_id),
        static_cast<unsigned long long>(frame),unsigned(gpu_dy),unsigned(success),
        static_cast<unsigned long long>(origin),
        static_cast<unsigned long long>(end));
    for(std::size_t i=0u;i<current.event_count;++i) {
        const auto& event=current.events[i];
        std::printf(
            "FYZZXYZZY_TRACE kind=cpu trace_id=%llu frame=%llu stage=%s start_ns=%llu end_ns=%llu duration_ns=%llu\n",
            static_cast<unsigned long long>(current.trace_id),
            static_cast<unsigned long long>(frame),event.label,
            static_cast<unsigned long long>(event.begin-origin),
            static_cast<unsigned long long>(event.end-origin),
            static_cast<unsigned long long>(event.end-event.begin));
    }
    for(std::size_t i=0u;i<current.counter_count;++i) {
        const auto& metric=current.counters[i];
        std::printf("FYZZXYZZY_TRACE kind=counter trace_id=%llu frame=%llu name=%s value=%llu\n",
                    static_cast<unsigned long long>(current.trace_id),
                    static_cast<unsigned long long>(frame),metric.label,
                    static_cast<unsigned long long>(metric.value));
    }
}

} // namespace fyzz::hardware_profile

#define FYZZ_PROFILE_JOIN_(a,b) a##b
#define FYZZ_PROFILE_JOIN(a,b) FYZZ_PROFILE_JOIN_(a,b)
#define FYZZ_PROFILE_SCOPE(name) \
    ::fyzz::hardware_profile::Scope FYZZ_PROFILE_JOIN(fyzz_profile_scope_,__LINE__){name}
#define FYZZ_PROFILE_BEGIN_FRAME() ::fyzz::hardware_profile::begin_frame()
#define FYZZ_PROFILE_END_FRAME(frame,gpu,success) \
    ::fyzz::hardware_profile::end_frame(frame,gpu,success)
#define FYZZ_PROFILE_COUNTER(label,value) \
    ::fyzz::hardware_profile::counter(label,static_cast<std::uint64_t>(value))
#else
#define FYZZ_PROFILE_SCOPE(name)
#define FYZZ_PROFILE_BEGIN_FRAME()
#define FYZZ_PROFILE_END_FRAME(frame,gpu,success)
#define FYZZ_PROFILE_COUNTER(label,value)
#endif
