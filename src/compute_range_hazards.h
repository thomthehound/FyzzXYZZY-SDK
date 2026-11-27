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

struct OperationRangeUse {
    std::uint64_t buffer;
    std::size_t begin,end;
    RangeAccessScope scope;
    bool writes;
    bool operator==(const OperationRangeUse&) const=default;
};

// Normalize simultaneous accesses before updating queue history. Only
// overlapping bytes inherit the combined access scope.
inline void normalize_operation_ranges(std::vector<OperationRangeUse>& uses,
                                       std::vector<OperationRangeUse>& result) {
    std::sort(uses.begin(),uses.end(),[](const auto& a,const auto& b) {
        if(a.buffer!=b.buffer)return a.buffer<b.buffer;
        if(a.begin!=b.begin)return a.begin<b.begin;
        return a.end<b.end;
    });
    result.clear();
    for(auto use:uses) {
        if(result.empty() || result.back().buffer!=use.buffer ||
           result.back().end<=use.begin) {
            result.push_back(use);
            continue;
        }
        auto i=static_cast<std::size_t>(std::lower_bound(
            result.begin(),result.end(),use,
            [](const auto& prior,const auto& next) {
                return prior.buffer<next.buffer ||
                    (prior.buffer==next.buffer && prior.end<=next.begin);
            })-result.begin());
        while(use.begin<use.end) {
            if(i==result.size()) {
                result.push_back(use);
                break;
            }
            // Earlier inputs start no later than this input, so the
            // overlapping suffix is contiguous within this buffer.
            if(result[i].begin<use.begin) {
                auto right=result[i];
                right.begin=use.begin;
                result[i].end=use.begin;
                result.insert(result.begin()+ ++i,right);
            }
            if(result[i].end>use.end) {
                auto right=result[i];
                right.begin=use.end;
                result[i].end=use.end;
                result.insert(result.begin()+i+1u,right);
            }
            auto& current=result[i];
            current.scope.include(use.scope);
            current.writes|=use.writes;
            use.begin=current.end;
            ++i;
        }
    }
    std::size_t count=0u;
    for(const auto& use:result) {
        if(count && result[count-1u].buffer==use.buffer &&
           result[count-1u].end==use.begin &&
           result[count-1u].scope==use.scope &&
           result[count-1u].writes==use.writes)
            result[count-1u].end=use.end;
        else result[count++]=use;
    }
    result.resize(count);
}

// Each interval retains its last producer and subsequent reader scopes.
// A read's visibility dependency does not replace the producer.
class BufferRangeHazards {
    struct State {
        RangeAccessScope writer{},readers{};
        bool host_write_pending=false;
        bool operator==(const State&) const=default;
    };
    struct Interval {
        std::size_t begin,end;
        State state;
    };
    std::vector<Interval> intervals_;
public:
    // The caller supplies validated, nonoverlapping operation intervals.
    template<class Emit>
    void use(std::size_t begin,std::size_t end,RangeAccessScope scope,
             bool writes,Emit emit,bool publish_to_host=false) {
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
                state={scope,{},publish_to_host};
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

    // Publish only current-batch writes to mappable storage. Host
    // observation is fence-owned, not a GPU reader dependency.
    template<class Emit>
    void publish_host_writes(Emit emit) {
        std::size_t count=0u;
        for(auto& interval:intervals_) {
            if(interval.state.host_write_pending) {
                emit(interval.begin,interval.end,interval.state.writer);
                interval.state.host_write_pending=false;
            }
            if(count && intervals_[count-1u].end==interval.begin &&
               intervals_[count-1u].state==interval.state)
                intervals_[count-1u].end=interval.end;
            else intervals_[count++]=interval;
        }
        intervals_.resize(count);
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
