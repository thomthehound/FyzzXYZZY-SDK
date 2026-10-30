// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "physx_port/common/src/CmPinnableArray.h"
#include "foundation/PxBitUtils.h"
#include "foundation/PxBounds3.h"

#include <new>

namespace physx::Bp {

// Original Bp::BoundsArray's pinned host array and capacity contract,
// extracted for source-owned single-volume GPU broadphase. Gu geometry
// updates are performed by the existing Sc bounds preparation stage;
// this class owns exactly one original PinnableArray and its dirty flag.
class SourceBoundsArrayNative final {
public:
    SourceBoundsArrayNative(
        Cm::VirtualAllocatorCallback& allocator,PxI32 group=0)
        :bounds_(allocator,group) {}

    bool initEntry(PxU32 index) {
        ++index; // source SIMD-readable trailing entry
        if(index>=bounds_.size()) {
            const auto capacity=PxNextPowerOfTwo(index);
            if(!bounds_.reserve(capacity))return false;
            bounds_.forceSize_Unsafe(capacity);
        }
        return true;
    }
    PxBounds3* begin() noexcept {return bounds_.begin();}
    const PxBounds3* begin() const noexcept {return bounds_.begin();}
    PxU32 size() const noexcept {return bounds_.size();}
    const PxBounds3* getBounds() const noexcept {return bounds_.begin();}
    PxBounds3* getBounds() noexcept {return bounds_.begin();}
    const PxBounds3& getBounds(PxU32 index) const {return bounds_[index];}
    bool hasChanged() const noexcept {return changed_;}
    void resetChangedState() noexcept {changed_=false;}
    void setChangedState() noexcept {changed_=true;}
    void setBounds(const PxBounds3& bounds,PxU32 index) {
        bounds_[index]=bounds;
        changed_=true;
    }
private:
    Cm::PinnableArray<PxBounds3> bounds_;
    bool changed_=true;
};

} // namespace physx::Bp
