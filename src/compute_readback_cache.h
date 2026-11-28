// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace fyzz::compute {

// Only completed private staging allocations enter this cache. Age is
// measured in subsequent readback requests, not elapsed wall time.
template<class Resource>
class CompletedReadbackCache {
public:
    static constexpr std::size_t slot_count=4u;
    static constexpr std::size_t reuse_window=2u*slot_count;
    struct Entry {
        Resource resource{};
        std::size_t capacity=0u;
        std::size_t remaining=0u;
    };

    template<class Release>
    Entry acquire(std::size_t bytes,Release release) noexcept {
        std::size_t selected=slot_count;
        for(std::size_t i=0u;i<slot_count;++i) {
            auto& entry=entries_[i];
            if(!entry.capacity)continue;
            if(--entry.remaining==0u) {
                release(std::move(entry.resource));
                entry={};
                continue;
            }
            // Near-sized reuse avoids pinning an exceptional peak with
            // a stream of much smaller requests.
            if(entry.capacity>=bytes && entry.capacity-bytes<=bytes &&
               (selected==slot_count || entry.capacity<entries_[selected].capacity))
                selected=i;
        }
        if(selected==slot_count)return {};
        return std::exchange(entries_[selected],Entry{});
    }

    template<class Release>
    void recycle(Resource resource,std::size_t capacity,Release release) noexcept {
        std::size_t selected=0u;
        for(std::size_t i=1u;i<slot_count;++i)
            if(entries_[i].remaining<entries_[selected].remaining)selected=i;
        auto& entry=entries_[selected];
        if(entry.capacity)release(std::move(entry.resource));
        entry={std::move(resource),capacity,reuse_window};
    }

private:
    std::array<Entry,slot_count> entries_{};
};

} // namespace fyzz::compute
