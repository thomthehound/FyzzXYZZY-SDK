#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Standard hosted images may already have usable APT package indexes.
# Avoid unconditional mirror refreshes: they have delayed CI by 5-10 min.
# Refresh only if an install from the current indexes fails.
packages=(libvulkan-dev mesa-vulkan-drivers glslang-tools)
apt_options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

if sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y -qq --no-install-recommends "${packages[@]}"; then
    exit 0
fi

echo 'Cached APT indexes could not satisfy Vulkan dependencies; refreshing repositories'
sudo apt-get "${apt_options[@]}" update
sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y -qq --no-install-recommends "${packages[@]}"
