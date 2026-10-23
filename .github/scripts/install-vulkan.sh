#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Reuse complete Ubuntu .deb archives between CI jobs. APT itself selects
# package versions and verifies the repository metadata; do not unpack
# cached files by hand or bypass dpkg's dependency validation.
# GitHub caches are best-effort; a cold or stale archive falls back to APT.
archives="${GITHUB_WORKSPACE:-$PWD}/.ci-cache/vulkan-apt/archives"
mkdir -p "$archives/partial"
# APT's downloader runs as _apt; the workspace directory is runner-owned.
# Give _apt the scratch directory while keeping completed archives readable.
sudo chown _apt:root "$archives/partial"
sudo chmod 0700 "$archives/partial"
packages=(libvulkan-dev mesa-vulkan-drivers glslang-tools)
apt_options=(
    -o "Dir::Cache::archives=$archives"
    -o "APT::Keep-Downloaded-Packages=true"
    -o Acquire::Retries=2
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

apt_install() {
    sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install \
        -y -qq --no-install-recommends "$@" "${packages[@]}"
}

if find "$archives" -maxdepth 1 -type f -name '*.deb' -print -quit | grep -q .; then
    # A cached archive may be stale after an Ubuntu security/package update.
    # Do not download anything or wait for mirrors on the normal hot path.
    if apt_install --no-download; then
        echo 'Installed Vulkan packages entirely from the existing APT archives'
    else
        echo 'Cached packages no longer match APT indexes; downloading missing updates'
        apt_install || {
            echo 'APT install failed with current indexes; refreshing repositories'
            sudo apt-get "${apt_options[@]}" update
            apt_install
        }
    fi
else
    # No archive yet: populate one in the ordinary, authenticated APT path.
    apt_install || {
        echo 'APT indexes could not satisfy Vulkan dependencies; refreshing repositories'
        sudo apt-get "${apt_options[@]}" update
        apt_install
    }
fi

for package in "${packages[@]}"; do
    dpkg-query -W -f='${db:Status-Abbrev}' "$package" | grep -qx 'ii '
done
test -f /usr/include/vulkan/vulkan.h
command -v glslangValidator >/dev/null
# APT leaves a root-only lock and _apt-owned scratch directory. Restore the
# runner ownership before actions/cache archives the completed .deb files.
sudo chown -R "$(id -u):$(id -g)" "$archives"
find "$archives" -maxdepth 1 -type f -name '*.deb' -print
echo "APT archive count: $(find "$archives" -maxdepth 1 -type f -name '*.deb' | wc -l)"
