// SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
// SPDX-License-Identifier: Apache-2.0

// Build contract: a compiler must really select C++23, rather than
// accepting the CMake request and silently compiling in an older mode.
#include <expected>
#include <string>

#if defined(_MSVC_LANG)
static_assert(_MSVC_LANG > 202002L, "FyzzXYZZY requires C++23 compiler mode");
#else
static_assert(__cplusplus > 202002L, "FyzzXYZZY requires C++23 compiler mode");
#endif

#ifndef __cpp_lib_expected
#error FyzzXYZZY requires the C++23 std::expected library facility
#endif

constexpr int source_mode_probe(int x) {
    if consteval {
        return x + 23;
    } else {
        return x + 1;
    }
}
static_assert(source_mode_probe(1) == 24,
              "C++23 if consteval must be enabled");

int main() {
    // Check a C++23 standard-library contract as well as the language mode,
    // so a misconfigured standard library is caught before SDK compilation.
    const std::expected<int, std::string> value(23);
    const std::expected<int, std::string> error(
        std::unexpected<std::string>("no value"));
    return value.has_value() && *value == 23 &&
           !error.has_value() && error.error() == "no value" &&
           source_mode_probe(1) == 2 ? 0 : 1;
}
