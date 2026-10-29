// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "extensions/PxDefaultAllocator.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"

#include <memory>

namespace fyzz::physx_runtime {

// PxCreateFoundation is process-global. Both the reference CPU runtime and
// source-owned native GPU heap acquire this same lease. Physics and source
// heap objects are released before their last Foundation reference.
class Foundation final {
public:
    Foundation();
    ~Foundation();

    Foundation(const Foundation&) = delete;
    Foundation& operator=(const Foundation&) = delete;

    physx::PxFoundation& get() const noexcept { return *instance_; }

private:
    struct Errors final : physx::PxErrorCallback {
        void reportError(physx::PxErrorCode::Enum,
                         const char* message,const char* file,int line) override;
    };

    physx::PxDefaultAllocator allocator_;
    Errors errors_;
    physx::PxFoundation* instance_;
};

std::shared_ptr<Foundation> acquire_foundation();

} // namespace fyzz::physx_runtime
