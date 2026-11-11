// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fyzzxyzzy/compute.h"

#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

namespace fyzz::physx_gpu::test {

// Source command inspections exclude CPU input transfers while retaining
// the original GPU operation order and command-list dependencies.
class SourceCommandView final {
public:
    explicit SourceCommandView(
        const std::vector<compute::BatchCommand>& commands) noexcept
        : commands_(commands) {}

    std::size_t size() const noexcept {
        std::size_t count=0;
        for(const auto& operation:commands_)
            if(!std::holds_alternative<compute::UploadCommand>(operation))
                ++count;
        return count;
    }
    bool empty() const noexcept { return size()==0u; }

    const compute::BatchCommand& operator[](std::size_t index) const {
        for(const auto& operation:commands_)
            if(!std::holds_alternative<compute::UploadCommand>(operation)) {
                if(!index)return operation;
                --index;
            }
        throw std::out_of_range("GPU command ordinal exceeds staged source work");
    }

private:
    const std::vector<compute::BatchCommand>& commands_;
};
} // namespace fyzz::physx_gpu::test
