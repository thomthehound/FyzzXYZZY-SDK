#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Key the immutable Actions cache to what APT would install, including
# currently selected transitive upgrades. This local query performs no
# downloads; changed package candidates create a new writable cache key.
if [[ "${1:-}" == '--cache-revision' ]]; then
    {
        apt-cache policy libvulkan-dev mesa-vulkan-drivers glslang-tools |
            awk '/^[^ ]/ { package=$1 } /^  Candidate:/ { print package, $2 }'
        apt-get -s install --no-install-recommends \
            libvulkan-dev mesa-vulkan-drivers glslang-tools 2>/dev/null |
            grep '^Inst ' || true
    } | sha256sum | cut -c 1-20
    exit 0
fi

# APT must download into its normal /var/cache/apt/archives directory so
# the unprivileged _apt sandbox can verify and fetch packages as designed.
# Only completed, authenticated .deb files move through the Actions cache.
# A stale or absent Actions cache simply falls back to normal APT.
cache_dir="${GITHUB_WORKSPACE:-$PWD}/.ci-cache/vulkan-apt/archives"
mkdir -p "$cache_dir"
shopt -s nullglob

packages=(libvulkan-dev mesa-vulkan-drivers glslang-tools)
apt_options=(
    -o "APT::Keep-Downloaded-Packages=true"
    -o Acquire::Retries=2
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)
apt_install() {
    sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install \
        -y -qq --no-install-recommends "$@" "${packages[@]}"
}

cached=("$cache_dir/"*.deb)
if (( ${#cached[@]} )); then
    # Ask APT to use its own cache, never install .deb files directly.
    sudo cp -n -- "${cached[@]}" /var/cache/apt/archives/
    if apt_install --no-download; then
        echo 'Installed Vulkan packages without network downloads'
    else
        echo 'Cached packages no longer satisfy current APT indexes'
        apt_install || {
            echo 'APT indexes are stale; refreshing the repositories'
            sudo apt-get "${apt_options[@]}" update
            apt_install
        }
    fi
else
    apt_install || {
        echo 'APT indexes are stale; refreshing the repositories'
        sudo apt-get "${apt_options[@]}" update
        apt_install
    }
fi

for package in "${packages[@]}"; do
    dpkg-query -W -f='${db:Status-Abbrev}' "$package" | grep -qx 'ii '
done
test -f /usr/include/vulkan/vulkan.h
command -v glslangValidator >/dev/null

# apt-get keeps its completed archives; the copied files are ordinary
# runner-readable cache inputs (no root-owned lock or partial directory).
completed=(/var/cache/apt/archives/*.deb)
if (( ${#completed[@]} )); then
    cp -- "${completed[@]}" "$cache_dir/"
fi
echo "APT archive count: $(find "$cache_dir" -maxdepth 1 -type f -name '*.deb' | wc -l)"
