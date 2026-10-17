#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0
"""Enforce SDK-wide architecture review sections for non-draft code PRs.

Reads the GitHub event payload, requires no third-party Python package, and
does not execute code supplied by the PR body. Push/dispatch builds skip.
"""
import json
import os
import re
import sys
from pathlib import Path

SECTIONS = (
    "Owner and source contract",
    "Runtime cost and scheduling",
    "Integration evidence",
    "Redundancy and retirement",
)

def validate(body: str) -> list[str]:
    errors: list[str] = []
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    sections = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[match.end():next_start]
    for name in SECTIONS:
        value = sections.get(name, "")
        # HTML comments are scaffolding, not a review explanation.
        value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL).strip()
        if len(value) < 28:
            errors.append(f"Missing substantive '## {name}' section")
    return errors

def main() -> int:
    path = os.getenv("GITHUB_EVENT_PATH")
    if not path:
        print("Local check: supply GITHUB_EVENT_PATH to evaluate a PR.")
        return 0
    event = json.loads(Path(path).read_text(encoding="utf-8"))
    pr = event.get("pull_request")
    if not pr:
        print("Architecture review: not a pull request.")
        return 0
    if pr.get("draft"):
        print("Architecture review: draft, not a merge gate yet.")
        return 0
    errors = validate(pr.get("body") or "")
    if errors:
        for item in errors:
            print(f"::error::{item}")
        print("Use .github/pull_request_template.md and docs/ARCHITECTURE_CONTRACT.md.")
        return 1
    print("SDK architecture PR review sections present; semantic review still required.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
