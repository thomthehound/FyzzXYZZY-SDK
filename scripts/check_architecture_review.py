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
from urllib.request import Request, urlopen

SECTIONS = (
    "Owner and source contract",
    "Runtime cost and scheduling",
    "Integration evidence",
    "Redundancy and retirement",
    "Source fidelity and deviations",
)

CLASSIFICATIONS = {
    "source-translation",
    "semantics-preserving-lowering",
    "non-production-only",
    "not-applicable",
}

def production_physics_changed(files: list[dict]) -> bool:
    """Source-fidelity declarations fail closed for native production physics."""
    roots = (
        "src/native/",
        "src/physx_port/",
        "src/physx_gpu/",
        "src/kernels/",
        "src/floor_contact.cpp",
        "src/world_vulkan.cpp",
    )
    return any(
        any(item.get(key, "").startswith(roots)
            for key in ("filename", "previous_filename"))
        for item in files
    )

def documentation_only(files: list[dict]) -> bool:
    """An all-docs PR still produces a green status on a protected branch."""
    def doc_path(path: str) -> bool:
        return path.startswith("docs/") or path.endswith(".md")
    return bool(files) and all(
        doc_path(item.get("filename", "")) and
        doc_path(item.get("previous_filename", item.get("filename", "")))
        for item in files
    )


def complete_changed_file_listing(files: list[dict], changed_files: object) -> bool:
    """Require the entire PR file list before classifying source fidelity."""
    return (isinstance(changed_files, int) and
            not isinstance(changed_files, bool) and
            len(files) == changed_files)


def complete_documentation_only(files: list[dict], changed_files: object) -> bool:
    """Never exempt a possibly truncated GitHub REST file listing.

    GitHub lists at most 3,000 PR files. 'changed_files' from the PR
    payload reflects the full PR, so a mismatch fails closed into the
    ordinary substantive architecture review.
    """
    return complete_changed_file_listing(files, changed_files) and documentation_only(files)


def github_pr_files(repository: str, number: int, token: str) -> list[dict]:
    """Use GitHub's PR file list, not fragile git diff/fetch-depth guessing."""
    if not repository or not repository.replace("-", "").replace("_", "").replace("/", "").isalnum():
        raise ValueError("Invalid repository name")
    files: list[dict] = []
    for page in range(1, 101):
        uri = (f"https://api.github.com/repos/{repository}/pulls/{number}/files"
               f"?per_page=100&page={page}")
        request = Request(uri, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fyzzxyzzy-architecture-review",
        })
        with urlopen(request, timeout=10) as result:
            batch = json.load(result)
        if not isinstance(batch, list):
            raise ValueError("Unexpected pull request files response")
        files.extend(batch)
        if len(batch) < 100:
            return files
    raise ValueError("Pull request file list exceeds policy pagination range")


def validate(body: str, files: list[dict] | None = None) -> list[str]:
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
    fidelity = re.sub(r"<!--.*?-->", "",
                      sections.get("Source fidelity and deviations", ""),
                      flags=re.DOTALL)
    found = re.findall(r"(?mi)^Fidelity classification:\s*([\w-]+)\s*$",
                       fidelity)
    if len(found) != 1 or found[0] not in CLASSIFICATIONS:
        errors.append("Source fidelity: declare one valid 'Fidelity classification:'")
    else:
        classification = found[0]
        if files is not None and production_physics_changed(files) and (
                classification not in {"source-translation",
                                       "semantics-preserving-lowering"}):
            errors.append(
                "Production physics requires source-translation or "
                "semantics-preserving-lowering, never a provisional substitute")
        if classification in {"source-translation", "semantics-preserving-lowering"}:
            for field in ("PhysX source", "Departures", "Parity evidence"):
                match = re.search(rf"(?mi)^{re.escape(field)}:\s*(.+)$",
                                  fidelity)
                if not match or len(match.group(1).strip()) < 12:
                    errors.append(f"Source fidelity: substantive '{field}:' required")
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
    # Documentation-only changes should not need GPU lifetime analysis, but
    # the required workflow MUST still report an explicit successful check.
    # API errors fail closed into normal body validation, never silent bypass.
    files: list[dict] | None = None
    try:
        token = os.getenv("GITHUB_TOKEN", "")
        repo = os.getenv("GITHUB_REPOSITORY", "")
        number = pr.get("number")
        if token and repo and isinstance(number, int):
            files = github_pr_files(repo, number, token)
            if complete_documentation_only(files, pr.get("changed_files")):
                print("Architecture review: docs-only PR, explicit successful check.")
                return 0
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Architecture review: docs-only classification unavailable ({exc}); "
              "requiring normal substantive review.")
    errors = validate(pr.get("body") or "", files)
    if files is None or not complete_changed_file_listing(
            files, pr.get("changed_files")):
        errors.append("Source fidelity: incomplete or unavailable PR file list; "
                      "cannot verify all changed production paths")
    if errors:
        for item in errors:
            print(f"::error::{item}")
        print("Use .github/pull_request_template.md and docs/ARCHITECTURE_CONTRACT.md.")
        return 1
    print("SDK architecture PR review sections present; semantic review still required.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
