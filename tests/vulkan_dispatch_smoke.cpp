// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fizzx/compute.h"

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
        using namespace fizzx::compute;
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
        const BufferId src = context.create_buffer(sizeof(input));
        const BufferId dst = context.create_buffer(sizeof(output));
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
