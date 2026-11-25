// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace fyzz::physx_gpu {

// Contract transient capacity only after sustained demand below half
// its allocation. A normal-sized request cancels the contraction.
class FrameWorkingSet {
public:
    bool should_contract(std::size_t required,std::size_t capacity) noexcept {
        if(required>=capacity/2u) {
            small_frames_=0u;
            return false;
        }
        if(small_frames_<contraction_frames)++small_frames_;
        return small_frames_==contraction_frames;
    }

    void reset() noexcept { small_frames_=0u; }
    static constexpr unsigned contraction_frames=8u;

private:
    unsigned small_frames_=0u;
};

} // namespace fyzz::physx_gpu
