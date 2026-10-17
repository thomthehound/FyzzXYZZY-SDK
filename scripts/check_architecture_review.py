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
    # Documentation-only changes should not need GPU lifetime analysis, but
    # the required workflow MUST still report an explicit successful check.
    # API errors fail closed into normal body validation, never silent bypass.
    try:
        token = os.getenv("GITHUB_TOKEN", "")
        repo = os.getenv("GITHUB_REPOSITORY", "")
        number = pr.get("number")
        if token and repo and isinstance(number, int):
            if documentation_only(github_pr_files(repo, number, token)):
                print("Architecture review: docs-only PR, explicit successful check.")
                return 0
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Architecture review: docs-only classification unavailable ({exc}); "
              "requiring normal substantive review.")
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
