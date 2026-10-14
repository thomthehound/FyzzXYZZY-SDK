// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "physx_gpu/broadphase_completion.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace fyzz::compute;
using fyzz::physx_gpu::BroadPhaseCompletionGate;

struct PairTask final : physx::PxBaseTask {
    explicit PairTask(const char* label, std::vector<std::string>& events)
        : label_(label), events_(events) {}

    void run() override {}
    const char* getName() const override { return label_; }
    void addReference() override { ++refs_; }
    void removeReference() override {
        if (refs_ != 1)
            throw std::runtime_error("PhysX task released twice");
        --refs_;
        ++releases_;
        events_.emplace_back(label_);
    }
    int32_t getReference() const override { return refs_; }
    void release() override {}

    int releases() const { return releases_; }

    const char* label_;
    std::vector<std::string>& events_;
    int refs_ = 1;
    int releases_ = 0;
};

template<class F>
void require_throws(F&& fn, const char* what) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error(what);
}

void check_event_order(const std::vector<std::string>& events) {
    if (events != std::vector<std::string>{"publish", "found", "lost"})
        throw std::runtime_error("PhysX pair tasks ran before result publication or out of order");
}

void exercise(Backend backend) {
    Context context(backend);
    BroadPhaseCompletionGate gate(context);
    std::vector<std::string> events;

    std::array<std::uint32_t, 73> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0xca110000u + static_cast<std::uint32_t>(i);
    const auto buffer = context.create_buffer(sizeof(input), MemoryPlacement::DeviceLocal);

    // A normal frame: GPU work finishes at the fetch boundary. The CPU
    // result-publication callback verifies the copied data, and pair tasks
    // become runnable only after that operation has succeeded.
    auto submission = context.submit_upload_buffer(buffer, 0, input.data(), sizeof(input));
    gate.arm(submission);
    require_throws([&] { gate.arm(submission); },
                   "Overlapping broadphase GPU frames were accepted");
    PairTask found("found", events), lost("lost", events);
    if (found.releases() || lost.releases())
        throw std::runtime_error("PhysX task released before GPU completion");
    gate.fetch_and_release([&] {
        std::array<std::uint32_t, 73> observed{};
        context.download(buffer, 0, observed.data(), sizeof(observed));
        if (observed != input)
            throw std::runtime_error("PhysX broadphase result visibility failed");
        events.emplace_back("publish");
    }, found, lost);
    check_event_order(events);
    if (gate.armed() || found.releases() != 1 || lost.releases() != 1)
        throw std::runtime_error("PhysX broadphase publication was not one-shot");
    require_throws([&] { gate.fetch_and_release([] {}, found, lost); },
                   "Duplicate pair-task release was permitted");

    // Publication failure after GPU completion must not release tasks.
    // The caller can apply PhysX's own abort/empty-pairs policy and retry.
    events.clear();
    gate.arm(context.submit_upload_buffer(buffer, 0, input.data(), sizeof(input)));
    PairTask fail_found("found", events), fail_lost("lost", events);
    require_throws([&] {
        gate.fetch_and_release([&] { throw std::runtime_error("Invalid result count"); },
                               fail_found, fail_lost);
    }, "Publication failure was swallowed");
    if (!gate.armed() || fail_found.releases() || fail_lost.releases() ||
        !gate.gpu_complete())
        throw std::runtime_error("Tasks released against failed PhysX publication");
    gate.fetch_and_release([&] { events.emplace_back("publish"); },
                           fail_found, fail_lost);
    check_event_order(events);

    // Upstream no-state-change fast path may have no new GPU command, but
    // the original postBroadPhase still fetches and releases pair tasks.
    events.clear();
    gate.arm(0);
    PairTask empty_found("found", events), empty_lost("lost", events);
    gate.fetch_and_release([&] { events.emplace_back("publish"); },
                           empty_found, empty_lost);
    check_event_order(events);

    // No callback, aliased task inputs or unarmed state may never silently
    // release a reference.
    gate.arm(0);
    PairTask invalid_found("found", events);
    require_throws([&] {
        gate.fetch_and_release({}, invalid_found, invalid_found);
    }, "Null callback and alias task were accepted");
    require_throws([&] {
        gate.fetch_and_release([] {}, invalid_found, invalid_found);
    }, "Aliased PhysX pair tasks were accepted");
    if (invalid_found.releases())
        throw std::runtime_error("Malformed PhysX publication released a task");
    PairTask valid_lost("lost", events);
    gate.fetch_and_release([] {}, invalid_found, valid_lost);
    context.destroy_buffer(buffer);
}

} // namespace

int main() {
    try {
#if defined(FYZZXYZZY_TEST_D3D12)
        exercise(Backend::D3D12);
#else
        exercise(Backend::Vulkan);
#endif
        return 0;
    } catch (const std::exception& e) {
        (void)e;
        return 1;
    }
}
