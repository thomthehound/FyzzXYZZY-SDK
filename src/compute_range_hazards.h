// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace fyzz::compute {

struct RangeAccessScope {
    std::uint64_t stages=0u,access=0u;
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

inline bool read_only_view(StorageBufferViewKind kind) noexcept {
    return kind==StorageBufferViewKind::RawRead ||
           kind==StorageBufferViewKind::Structured32Read;
}

inline void validate_binding_access(const BufferBinding& binding,bool read_only) {
    if(binding.access!=BufferAccess::Unspecified &&
       binding.access!=BufferAccess::Read &&
       binding.access!=BufferAccess::Write &&
       binding.access!=BufferAccess::ReadWrite)
        throw std::invalid_argument("Invalid compute buffer access intent");
    if(read_only && binding.access!=BufferAccess::Unspecified &&
       binding.access!=BufferAccess::Read)
        throw std::invalid_argument("Read-only compute view declared writable");
    const auto extent=binding.accessed_extent();
    if(!extent.bytes || extent.offset>binding.bytes ||
       extent.bytes>binding.bytes-extent.offset)
        throw std::invalid_argument("Compute access extent exceeds descriptor");
}

// A descriptor table can serve several distinct logical access declarations.
inline bool same_dispatch_accesses(const DispatchCommand& a,
                                   const DispatchCommand& b) noexcept {
    if(a.pipeline!=b.pipeline || a.buffers.size()!=b.buffers.size() ||
       a.accesses.size()!=b.accesses.size() ||
       a.indirect.has_value()!=b.indirect.has_value())
        return false;
    if(a.indirect && (a.indirect->buffer!=b.indirect->buffer ||
                      a.indirect->offset!=b.indirect->offset))
        return false;
    for(std::size_t i=0u;i<a.buffers.size();++i) {
        const auto& x=a.buffers[i];
        const auto& y=b.buffers[i];
        const auto xe=x.accessed_extent(),ye=y.accessed_extent();
        if(x.slot!=y.slot || x.buffer!=y.buffer || x.offset!=y.offset ||
           x.bytes!=y.bytes || x.access!=y.access ||
           xe.offset!=ye.offset || xe.bytes!=ye.bytes)
            return false;
    }
    for(std::size_t i=0u;i<a.accesses.size();++i) {
        const auto& x=a.accesses[i];
        const auto& y=b.accesses[i];
        if(x.slot!=y.slot || x.access!=y.access ||
           x.extent.offset!=y.extent.offset || x.extent.bytes!=y.extent.bytes)
            return false;
    }
    return true;
}

// Batch-local immutable ranges; queue dependency history is not cached.
class DispatchRangePlans {
    static constexpr auto missing=std::numeric_limits<std::size_t>::max();
    struct Plan {
        const DispatchCommand* command;
        std::size_t begin,count;
    };
    std::vector<Plan> plans_;
    std::vector<std::size_t> latest_;
    std::vector<OperationRangeUse> ranges_,scratch_;
public:
    template<class Prepare>
    std::size_t prepare(std::size_t table,const DispatchCommand& command,
                        Prepare prepare_ranges) {
        if(table>=latest_.size())latest_.resize(table+1u,missing);
        auto& latest=latest_[table];
        if(latest!=missing && same_dispatch_accesses(*plans_[latest].command,command))
            return latest;
        scratch_.clear();
        prepare_ranges(scratch_);
        const auto begin=ranges_.size();
        ranges_.insert(ranges_.end(),scratch_.begin(),scratch_.end());
        latest=plans_.size();
        plans_.push_back({&command,begin,scratch_.size()});
        return latest;
    }
    std::span<const OperationRangeUse> ranges(std::size_t plan) const {
        const auto& p=plans_[plan];
        return std::span<const OperationRangeUse>(ranges_).subspan(p.begin,p.count);
    }
    std::size_t size() const noexcept { return plans_.size(); }
};

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
        // A contained access that leaves the state unchanged needs no
        // interval split, insertion or coalescing.
        if(i<intervals_.size() && intervals_[i].begin<=begin &&
           intervals_[i].end>=end) {
            const auto& state=intervals_[i].state;
            if(!writes && state.readers.covers(scope))return;
            if(writes && state==State{scope,{},publish_to_host}) {
                if(state.writer.stages)emit(begin,end,state.writer,scope);
                return;
            }
        }
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

// Enhanced buffer barriers cover the entire resource. Readers can
// overlap one disjoint writer, but sequential writers require ordering.
class PendingBufferRanges {
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
            if(first!=ranges_.end() && first->begin<=begin && first->end>=end)
                return;
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
        bool empty() const noexcept { return ranges_.empty(); }
        void clear() noexcept { ranges_.clear(); }
    };
    RangeSet readers_,writers_;
public:
    bool conflicts(std::size_t begin,std::size_t end,bool writes) const noexcept {
        if(writes)
            return !writers_.empty() || readers_.overlaps(begin,end);
        return writers_.overlaps(begin,end);
    }
    void record(std::size_t begin,std::size_t end,bool writes) {
        // Once a writer exists, every subsequent write ends the epoch;
        // only its written intervals can conflict with later readers.
        if(writes) {
            if(writers_.empty())readers_.clear();
            writers_.add(begin,end);
        }else if(writers_.empty())readers_.add(begin,end);
    }
    void clear() noexcept {
        readers_.clear();
        writers_.clear();
    }
};

} // namespace fyzz::compute
