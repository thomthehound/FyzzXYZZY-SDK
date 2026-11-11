// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(FYZZXYZZY_PROFILE_GPU)
#include <memory>
#include <utility>
#include <vector>
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

struct CpuEvent {
    const char* label=nullptr;
    std::uint64_t trace_id=0u,frame=0u,begin=0u,end=0u;
};
struct Counter {
    const char* label=nullptr;
    std::uint64_t trace_id=0u,frame=0u,value=0u;
};
struct FrameEvent {
    std::uint64_t trace_id=0u,frame=0u,begin=0u,end=0u;
    bool gpu_dy=false,success=false;
};
struct GpuEvent {
    const char* backend=nullptr;
    const char* type=nullptr;
    std::uint64_t context=0u,trace_id=0u,submission=0u,op=0u,pipeline=0u;
    std::uint64_t bytes=0u;
    std::uint32_t x=0u,y=0u,z=0u;
    bool indirect=false;
    double duration_ns=0.0;
};
struct BufferEvent {
    const char* backend=nullptr;
    const char* action=nullptr;
    std::uint64_t context=0u,trace_id=0u,buffer=0u,bytes=0u;
    unsigned placement=0u;
};
struct PipelineEvent {
    const char* backend=nullptr;
    const char* label=nullptr;
    std::uint64_t context=0u,pipeline=0u;
};

// Segmented streams preserve all records without relocating existing events.
// Trace output is deferred until after the workload completes.
template<class T,std::size_t SegmentSize=4096u>
class EventStream {
public:
    bool append(const T& event) noexcept {
        if(count_==segments_.size()*SegmentSize) {
            try {
                auto segment=std::make_unique<T[]>(SegmentSize);
                segments_.push_back(std::move(segment));
            } catch(...) {
                return false;
            }
        }
        (*this)[count_++]=event;
        return true;
    }
    T& operator[](std::size_t index) noexcept {
        return segments_[index/SegmentSize][index%SegmentSize];
    }
    const T& operator[](std::size_t index) const noexcept {
        return segments_[index/SegmentSize][index%SegmentSize];
    }
    std::size_t size() const noexcept {return count_;}
    void clear() noexcept {
        count_=0u;
        segments_.clear();
    }
private:
    std::vector<std::unique_ptr<T[]>> segments_;
    std::size_t count_=0u;
};

struct Archive {
    EventStream<FrameEvent,512u> frames;
    EventStream<CpuEvent> cpu;
    EventStream<Counter> counters;
    EventStream<GpuEvent> gpu;
    EventStream<BufferEvent> buffers;
    EventStream<PipelineEvent,256u> pipelines;
    std::size_t dropped=0u;
    ~Archive() noexcept {flush();}

    void flush() noexcept {
        for(std::size_t i=0u;i<frames.size();++i) {
            const auto& v=frames[i];
            std::printf(
                "FYZZXYZZY_TRACE kind=frame trace_id=%llu frame=%llu gpu_dy=%u success=%u start_ns=%llu end_ns=%llu\n",
                static_cast<unsigned long long>(v.trace_id),
                static_cast<unsigned long long>(v.frame),
                unsigned(v.gpu_dy),unsigned(v.success),
                static_cast<unsigned long long>(v.begin),
                static_cast<unsigned long long>(v.end));
        }
        for(std::size_t i=0u;i<cpu.size();++i) {
            const auto& v=cpu[i];
            std::printf(
                "FYZZXYZZY_TRACE kind=cpu trace_id=%llu frame=%llu stage=%s start_ns=%llu end_ns=%llu duration_ns=%llu\n",
                static_cast<unsigned long long>(v.trace_id),
                static_cast<unsigned long long>(v.frame),v.label,
                static_cast<unsigned long long>(v.begin),
                static_cast<unsigned long long>(v.end),
                static_cast<unsigned long long>(v.end-v.begin));
        }
        for(std::size_t i=0u;i<counters.size();++i) {
            const auto& v=counters[i];
            std::printf(
                "FYZZXYZZY_TRACE kind=counter trace_id=%llu frame=%llu name=%s value=%llu\n",
                static_cast<unsigned long long>(v.trace_id),
                static_cast<unsigned long long>(v.frame),v.label,
                static_cast<unsigned long long>(v.value));
        }
        for(std::size_t i=0u;i<gpu.size();++i) {
            const auto& v=gpu[i];
            std::printf(
                "FYZZXYZZY_TRACE kind=gpu backend=%s context=%llu trace_id=%llu submission=%llu op=%llu type=%s pipeline=%llu indirect=%u x=%u y=%u z=%u copy_bytes=%llu duration_ns=%.0f\n",
                v.backend,static_cast<unsigned long long>(v.context),
                static_cast<unsigned long long>(v.trace_id),
                static_cast<unsigned long long>(v.submission),
                static_cast<unsigned long long>(v.op),v.type,
                static_cast<unsigned long long>(v.pipeline),
                unsigned(v.indirect),v.x,v.y,v.z,
                static_cast<unsigned long long>(v.bytes),v.duration_ns);
        }
        for(std::size_t i=0u;i<buffers.size();++i) {
            const auto& v=buffers[i];
            if(v.action[0]=='c')
                std::printf(
                    "FYZZXYZZY_TRACE kind=buffer backend=%s context=%llu trace_id=%llu action=%s buffer=%llu bytes=%llu placement=%u\n",
                    v.backend,static_cast<unsigned long long>(v.context),
                    static_cast<unsigned long long>(v.trace_id),v.action,
                    static_cast<unsigned long long>(v.buffer),
                    static_cast<unsigned long long>(v.bytes),v.placement);
            else
                std::printf(
                    "FYZZXYZZY_TRACE kind=buffer backend=%s context=%llu trace_id=%llu action=%s buffer=%llu bytes=%llu\n",
                    v.backend,static_cast<unsigned long long>(v.context),
                    static_cast<unsigned long long>(v.trace_id),v.action,
                    static_cast<unsigned long long>(v.buffer),
                    static_cast<unsigned long long>(v.bytes));
        }
        for(std::size_t i=0u;i<pipelines.size();++i) {
            const auto& v=pipelines[i];
            std::printf(
                "FYZZXYZZY_TRACE kind=pipeline backend=%s context=%llu pipeline=%llu label=%s\n",
                v.backend,static_cast<unsigned long long>(v.context),
                static_cast<unsigned long long>(v.pipeline),v.label);
        }
        if(dropped)
            std::printf("FYZZXYZZY_TRACE kind=overflow dropped=%llu\n",
                static_cast<unsigned long long>(dropped));
        frames.clear();
        cpu.clear();
        counters.clear();
        gpu.clear();
        buffers.clear();
        pipelines.clear();
        dropped=0u;
    }
};

struct Frame {
    std::uint64_t begin=0u,trace_id=0u;
    bool active=false;
};
inline thread_local Frame current;
inline thread_local Archive archive;
inline std::atomic<std::uint64_t> next_trace_id{1u};
inline std::atomic<std::uint64_t> next_context_id{1u};

inline std::uint64_t allocate_context_id() noexcept {
    return next_context_id.fetch_add(1u,std::memory_order_relaxed);
}
inline std::uint64_t current_trace_id() noexcept {
    return current.active?current.trace_id:0u;
}
inline std::uint64_t current_frame_begin() noexcept {
    return current.active?current.begin:0u;
}
inline void begin_worker(std::uint64_t trace_id,std::uint64_t origin) noexcept {
    current.begin=origin;
    current.trace_id=trace_id;
    current.active=trace_id!=0u;
}
inline void begin_frame() noexcept {
    current.begin=now_ns();
    current.trace_id=next_trace_id.fetch_add(1u,std::memory_order_relaxed);
    current.active=true;
}
struct Scope {
    const char* label;
    std::uint64_t begin;
    explicit Scope(const char* name) noexcept
        :label(name),begin(current.active?now_ns():0u){}
    ~Scope() noexcept {
        if(!begin || !current.active)return;
        const auto end=now_ns();
        if(!archive.cpu.append({label,current.trace_id,0u,begin,end}))
            ++archive.dropped;
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
inline void counter(const char* label,std::uint64_t value) noexcept {
    if(!current.active)return;
    if(!archive.counters.append({label,current.trace_id,0u,value}))
        ++archive.dropped;
}
inline void gpu_event(GpuEvent event) noexcept {
    if(!archive.gpu.append(event))++archive.dropped;
}
inline void buffer_event(BufferEvent event) noexcept {
    if(!archive.buffers.append(event))++archive.dropped;
}
inline void pipeline_event(PipelineEvent event) noexcept {
    if(!archive.pipelines.append(event))++archive.dropped;
}
inline void complete_thread_trace(std::uint64_t frame) noexcept {
    // Thread-local spans are correlated by the shared submission trace ID.
    for(std::size_t i=archive.cpu.size();i>0u;--i) {
        auto& v=archive.cpu[i-1u];
        if(v.trace_id!=current.trace_id)break;
        v.frame=frame;
        v.begin-=current.begin;
        v.end-=current.begin;
    }
    for(std::size_t i=archive.counters.size();i>0u;--i) {
        auto& v=archive.counters[i-1u];
        if(v.trace_id!=current.trace_id)break;
        v.frame=frame;
    }
}
inline void end_worker(std::uint64_t frame) noexcept {
    if(!current.active)return;
    current.active=false;
    complete_thread_trace(frame);
}
inline void end_frame(std::uint64_t frame,bool gpu_dy,bool success) noexcept {
    if(!current.active)return;
    const auto end=now_ns();
    current.active=false;
    if(!archive.frames.append({
        current.trace_id,frame,current.begin,end,gpu_dy,success}))
        ++archive.dropped;
    complete_thread_trace(frame);
}
inline void flush() noexcept {
    archive.flush();
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
