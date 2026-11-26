// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fyzz::compute {

struct RangeAccessScope {
    std::uint32_t stages=0u,access=0u;
    bool operator==(const RangeAccessScope&) const=default;
    void include(RangeAccessScope other) noexcept {
        stages|=other.stages;
        access|=other.access;
    }
    bool covers(RangeAccessScope other) const noexcept {
        return (stages&other.stages)==other.stages &&
               (access&other.access)==other.access;
    }
};

// Each interval retains its last producer and subsequent reader scopes.
// A read's visibility dependency does not replace the producer.
class BufferRangeHazards {
    struct State {
        RangeAccessScope writer{},readers{};
        bool operator==(const State&) const=default;
    };
    struct Interval {
        std::size_t begin,end;
        State state;
    };
    std::vector<Interval> intervals_;
public:
    // The caller validates ranges and merges overlapping uses of one operation.
    template<class Emit>
    void use(std::size_t begin,std::size_t end,RangeAccessScope scope,
             bool writes,Emit emit) {
        auto i=static_cast<std::size_t>(std::lower_bound(
            intervals_.begin(),intervals_.end(),begin,
            [](const Interval& interval,std::size_t offset) {
                return interval.end<=offset;
            })-intervals_.begin());
        const auto first=i;
        while(begin<end) {
            if(i==intervals_.size() || intervals_[i].begin>begin) {
                const auto stop=i==intervals_.size()
                    ?end:std::min(end,intervals_[i].begin);
                intervals_.insert(intervals_.begin()+i,{begin,stop,{}});
            }else if(intervals_[i].begin<begin) {
                auto right=intervals_[i];
                right.begin=begin;
                intervals_[i].end=begin;
                ++i;
                intervals_.insert(intervals_.begin()+i,right);
            }
            if(intervals_[i].end>end) {
                auto right=intervals_[i];
                right.begin=end;
                intervals_[i].end=end;
                intervals_.insert(intervals_.begin()+i+1u,right);
            }
            auto& interval=intervals_[i];
            auto& state=interval.state;
            RangeAccessScope dependency{};
            if(writes) {
                dependency=state.writer;
                dependency.stages|=state.readers.stages;
                state={scope,{}};
            }else {
                if(!state.readers.covers(scope))
                    dependency=state.writer;
                state.readers.include(scope);
            }
            if(dependency.stages)
                emit(interval.begin,interval.end,dependency,scope);
            begin=interval.end;
            ++i;
        }
        // Coalesce equal states without revisiting unrelated intervals.
        auto j=first?first-1u:0u;
        while(j+1u<intervals_.size() && j<i) {
            auto& left=intervals_[j];
            const auto& right=intervals_[j+1u];
            if(left.end==right.begin && left.state==right.state) {
                left.end=right.end;
                intervals_.erase(intervals_.begin()+j+1u);
                --i;
            }else ++j;
        }
    }
};

// A D3D12 UAV barrier orders every pending access to the resource.
// Track only ranges accessed since its last transition or UAV barrier.
class PendingUavRanges {
    struct Range {
        std::size_t begin,end;
    };
    class RangeSet {
        std::vector<Range> ranges_;
    public:
        bool overlaps(std::size_t begin,std::size_t end) const noexcept {
            const auto it=std::lower_bound(ranges_.begin(),ranges_.end(),begin,
                [](const Range& range,std::size_t offset) {
                    return range.end<=offset;
                });
            return it!=ranges_.end() && it->begin<end;
        }
        void add(std::size_t begin,std::size_t end) {
            const auto first=std::lower_bound(ranges_.begin(),ranges_.end(),begin,
                [](const Range& range,std::size_t offset) {
                    return range.end<offset;
                });
            auto last=first;
            while(last!=ranges_.end() && last->begin<=end) {
                begin=std::min(begin,last->begin);
                end=std::max(end,last->end);
                ++last;
            }
            if(first==last)
                ranges_.insert(first,{begin,end});
            else {
                *first={begin,end};
                ranges_.erase(first+1,last);
            }
        }
        void clear() noexcept { ranges_.clear(); }
    };
    RangeSet readers_,writers_;
public:
    bool conflicts(std::size_t begin,std::size_t end,bool writes) const noexcept {
        return writers_.overlaps(begin,end) ||
               (writes && readers_.overlaps(begin,end));
    }
    void record(std::size_t begin,std::size_t end,bool writes) {
        (writes?writers_:readers_).add(begin,end);
    }
    void clear() noexcept {
        readers_.clear();
        writers_.clear();
    }
};

} // namespace fyzz::compute
