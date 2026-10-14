// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {

struct Parameters {
    std::uint32_t count;
    std::uint32_t multiplier;
    std::uint32_t addend;
};

std::vector<std::uint8_t> read_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Compiled compute shader was not found");
    }
    std::vector<char> input(std::istreambuf_iterator<char>{stream},
                            std::istreambuf_iterator<char>{});
    return {input.begin(), input.end()};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected the compiled shader file path");
        }
        using namespace fyzz::compute;
        constexpr std::uint32_t size = 259;
        constexpr std::uint32_t array_size = 320;
        constexpr std::uint32_t sentinel = 0xFEEDACEDu;
        std::array<std::uint32_t, array_size> input{};
        std::array<std::uint32_t, array_size> output{};
        for (std::uint32_t i = 0; i < array_size; ++i) {
            input[i] = i * 37u + 11u;
            output[i] = sentinel;
        }

        Context context(Backend::Vulkan, 0);
        const BufferId src = context.create_buffer(sizeof(input), MemoryPlacement::DeviceLocal);
        const BufferId dst = context.create_buffer(sizeof(output), MemoryPlacement::DeviceLocal);
        context.upload(src, 0, input.data(), sizeof(input));
        context.upload(dst, 0, output.data(), sizeof(output));

        ComputePipelineDesc description;
        description.shader.format = ShaderFormat::Spirv;
        description.shader.bytes = read_shader(argv[1]);
        description.shader.entry_point = "main";
        description.storage_buffer_binding_count = 2;
        description.push_constant_bytes = sizeof(Parameters);
        const PipelineId pipeline = context.create_compute_pipeline(description);

        Parameters parameters{size, 3, 7};
        const std::vector<BufferBinding> bindings = {
            {1, dst, 0, sizeof(output)},
            {0, src, 0, sizeof(input)}
        };
        context.dispatch(pipeline, {(size + 63u) / 64u, 1, 1},
                         bindings, &parameters, sizeof(parameters));
        context.download(dst, 0, output.data(), sizeof(output));

        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected =
                i < size ? input[i] * parameters.multiplier + parameters.addend : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Compute shader produced incorrect output at " +
                                         std::to_string(i));
            }
        }

        bool rejected = false;
        try {
            context.dispatch(pipeline, {0, 1, 1}, bindings, &parameters, sizeof(parameters));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Zero-workgroup dispatch was accepted");
        }
        rejected = false;
        try {
            context.dispatch(pipeline, {1, 1, 1},
                             {{0, src, 0, sizeof(input)}, {0, dst, 0, sizeof(output)}},
                             &parameters, sizeof(parameters));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Duplicate descriptor slot was accepted");
        }
        rejected = false;
        try {
            context.dispatch(pipeline, {1, 1, 1}, bindings, &parameters, 0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("Mismatched push constant size was accepted");
        }

        // A compute dispatch and its dependent transfer are both enqueued
        // before waiting. The shader writes must be visible to the copy.
        const BufferId mirror = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        parameters.multiplier = 7;
        parameters.addend = 19;
        const SubmissionId dispatched = context.submit_dispatch(
            pipeline, {(size + 63u) / 64u, 1, 1},
            bindings, &parameters, sizeof(parameters));
        const SubmissionId copied = context.submit_copy_buffer(
            dst, 0, mirror, 0, sizeof(output));
        context.wait_submission(copied);
        if (!context.submission_complete(dispatched)) {
            throw std::runtime_error("Dependent compute submission remains incomplete");
        }
        context.wait_submission(dispatched);
        context.download(mirror, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected =
                i < size ? input[i] * parameters.multiplier + parameters.addend : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Dependent asynchronous dispatch/copy result incorrect");
            }
        }
        // A three-command batch has two dependent dispatches and a copy.
        // It must execute in order under a single completion fence.
        Parameters first_pass{size, 2, 11};
        Parameters second_pass{size, 5, 2};
        auto push_bytes = [](const Parameters& value) {
            std::vector<std::uint8_t> bytes(sizeof(value));
            std::memcpy(bytes.data(), &value, sizeof(value));
            return bytes;
        };
        const DispatchSize groups{(size + 63u) / 64u, 1, 1};
        const std::vector<BatchCommand> batch = {
            DispatchCommand{pipeline, groups, bindings, push_bytes(first_pass)},
            CopyCommand{dst, 0, mirror, 0, sizeof(output)},
            DispatchCommand{pipeline, groups,
                            {{0, mirror, 0, sizeof(output)}, {1, dst, 0, sizeof(output)}},
                            push_bytes(second_pass)}
        };
        const SubmissionId batch_id = context.submit_batch(batch);
        context.wait_submission(batch_id);
        context.download(dst, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? (input[i] * first_pass.multiplier + first_pass.addend) *
                       second_pass.multiplier + second_pass.addend
                : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Ordered Vulkan compute batch result incorrect");
            }
        }

        bool rejected_batch = false;
        try {
            context.submit_batch({});
        } catch (const std::invalid_argument&) {
            rejected_batch = true;
        }
        if (!rejected_batch) {
            throw std::runtime_error("Empty Vulkan compute batch was accepted");
        }

        // All input validation must precede submission. No earlier operation
        // may execute if a later command has an invalid buffer range.
        rejected_batch = false;
        try {
            context.submit_batch({
                CopyCommand{src, 0, dst, 0, sizeof(input)},
                CopyCommand{src, 0, dst, 0, sizeof(input) + 4}
            });
        } catch (const std::out_of_range&) {
            rejected_batch = true;
        }
        if (!rejected_batch) {
            throw std::runtime_error("Invalid second batch command was accepted");
        }
        context.download(dst, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? (input[i] * first_pass.multiplier + first_pass.addend) *
                       second_pass.multiplier + second_pass.addend
                : sentinel;
            if (output[i] != expected) {
                throw std::runtime_error("Rejected batch unexpectedly changed GPU data");
            }
        }
        // Unrelated buffers must not force global inter-operation memory
        // barriers. Their commands remain in the same ordered batch, while
        // a dependent copy and second dispatch still require RAW barriers.
        const BufferId independent_input = context.create_buffer(
            sizeof(input), MemoryPlacement::DeviceLocal);
        const BufferId independent_output = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        const BufferId independent_copy = context.create_buffer(
            sizeof(output), MemoryPlacement::DeviceLocal);
        context.upload(independent_input, 0, input.data(), sizeof(input));
        std::array<std::uint32_t, array_size> initial{};
        initial.fill(sentinel);
        context.upload(independent_output, 0, initial.data(), sizeof(initial));
        Parameters alternate{size, 9, 13};
        std::vector<BatchCommand> independent_batch = {
            DispatchCommand{pipeline, groups, bindings, push_bytes(first_pass)},
            DispatchCommand{pipeline, groups,
                {{0, independent_input, 0, sizeof(input)},
                 {1, independent_output, 0, sizeof(output)}}, push_bytes(alternate)},
            CopyCommand{independent_output, 0, independent_copy, 0, sizeof(output)}
        };
        for (int iteration = 0; iteration < 12; ++iteration) {
            const SubmissionId id = context.submit_batch(independent_batch);
            context.wait_submission(id);
        }
        context.download(independent_copy, 0, output.data(), sizeof(output));
        for (std::uint32_t i = 0; i < array_size; ++i) {
            const std::uint32_t expected = i < size
                ? input[i] * alternate.multiplier + alternate.addend : sentinel;
            if (output[i] != expected)
                throw std::runtime_error("Independent dispatch/copy or recycled Vulkan resources failed");
        }
        // Simultaneously in-flight submissions must not reuse one another's
        // command buffer or descriptor pool before the corresponding fence.
        const SubmissionId first_in_flight = context.submit_batch(independent_batch);
        const SubmissionId second_in_flight = context.submit_batch(independent_batch);
        context.wait_submission(second_in_flight);
        if (!context.submission_complete(first_in_flight))
            throw std::runtime_error("Earlier same-queue submission not completed");
        context.wait_submission(first_in_flight);
        context.destroy_buffer(independent_copy);
        context.destroy_buffer(independent_output);
        context.destroy_buffer(independent_input);

        context.destroy_buffer(mirror);
        context.destroy_compute_pipeline(pipeline);
        context.destroy_buffer(dst);
        context.destroy_buffer(src);
        std::cout << "Portable compute shader acceptance PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Portable compute shader acceptance FAILED: " << e.what() << '\n';
        return 1;
    }
}
