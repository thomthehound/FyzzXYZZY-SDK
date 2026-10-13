// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

#include "fyzzxyzzy/compute.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        using namespace fy_::compute;
        const auto devices = enumerate_devices(Backend::Vulkan);
        if (devices.empty()) {
            throw std::runtime_error("No Vulkan device exposing a compute queue");
        }
        Context context(Backend::Vulkan, 0);
        const auto& info = context.device_info();
        std::cout << "Vulkan compute device: " << info.name
                  << ", vendor: " << info.vendor_id << '\n';

        std::array<std::uint32_t, 64> source{};
        std::array<std::uint32_t, 64> target{};
        std::array<std::uint32_t, 64> result{};
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<std::uint32_t>(i * 17 + 3);
            target[i] = 0xC0FFEEu;
        }

        const auto src = context.create_buffer(sizeof(source));
        const auto dst = context.create_buffer(sizeof(target));
        context.upload(src, 0, source.data(), sizeof(source));
        context.upload(dst, 0, target.data(), sizeof(target));

        constexpr std::size_t kStart = 8;
        constexpr std::size_t kCount = 32;
        context.copy_buffer(src, kStart * sizeof(std::uint32_t),
                            dst, 16 * sizeof(std::uint32_t),
                            kCount * sizeof(std::uint32_t));
        context.download(dst, 0, result.data(), sizeof(result));

        for (std::size_t i = 0; i < result.size(); ++i) {
            const std::uint32_t expected =
                i >= 16 && i < 16 + kCount ? source[kStart + i - 16] : 0xC0FFEEu;
            if (result[i] != expected) {
                throw std::runtime_error("Vulkan buffer-copy result incorrect");
            }
        }

        bool checked_bounds = false;
        try {
            context.copy_buffer(src, sizeof(source) - 4, dst, 0, 8);
        } catch (const std::out_of_range&) {
            checked_bounds = true;
        }
        if (!checked_bounds) {
            throw std::runtime_error("Compute buffer bounds check missing");
        }
        context.destroy_buffer(dst);
        context.destroy_buffer(src);
        std::cout << "Portable compute memory acceptance PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Portable compute memory acceptance FAILED: " << e.what() << '\n';
        return 1;
    }
}
