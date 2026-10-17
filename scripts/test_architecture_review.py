# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0
"""Covers the review gate's nontrivial PR-body parsing and empty templates."""
import unittest

from check_architecture_review import SECTIONS, documentation_only, complete_documentation_only, validate


class ArchitectureReviewTest(unittest.TestCase):
    def test_complete_source_grounded_review(self):
        body = "\n\n".join(
            f"## {name}\nProduction owner and source lifetime are documented; "
            f"native Linux and Windows tests verify this particular contract."
            for name in SECTIONS
        )
        self.assertEqual([], validate(body))

    def test_unfilled_template_is_not_evidence(self):
        body = "\n\n".join(
            f"## {name}\n<!-- Placeholder words that are not evidence of any kind -->"
            for name in SECTIONS
        )
        self.assertEqual(len(SECTIONS), len(validate(body)))

    def test_missing_or_too_short_section_blocks_merge(self):
        body = "\n\n".join(
            f"## {name}\nA meaningful analysis of actual source behavior and tests."
            for name in SECTIONS[:-1]
        )
        body += f"\n\n## {SECTIONS[-1]}\nNone."
        self.assertEqual(1, len(validate(body)))
        self.assertIn(SECTIONS[-1], validate(body)[0])

    def test_docs_only_gets_explicit_green_status(self):
        self.assertTrue(documentation_only([
            {"filename": "README.md", "status": "modified"},
            {"filename": "docs/architecture.md", "status": "modified"},
        ]))
        self.assertFalse(documentation_only([]))

    def test_truncated_all_docs_list_never_exempts_code(self):
        # GitHub's files endpoint caps its return at 3,000 entries even
        # when the PR's changed_files count is larger.
        docs = [{"filename": f"docs/page-{i}.md"} for i in range(3000)]
        self.assertTrue(documentation_only(docs))
        self.assertFalse(complete_documentation_only(docs, 3001))
        self.assertFalse(complete_documentation_only(docs, None))
        self.assertFalse(complete_documentation_only(docs, True))
        self.assertTrue(complete_documentation_only(docs, 3000))

    def test_partial_page_mismatch_rejects_docs_exemption(self):
        docs = [{"filename": "docs/a.md"}, {"filename": "README.md"}]
        self.assertFalse(complete_documentation_only(docs, 3))
        self.assertTrue(complete_documentation_only(docs, 2))
        self.assertFalse(complete_documentation_only([], 0))

    def test_mixed_code_or_renamed_code_needs_review(self):
        self.assertFalse(documentation_only([
            {"filename": "docs/notes.md"},
            {"filename": "src/world.cpp"},
        ]))
        self.assertFalse(documentation_only([
            {"filename": "docs/former_code.md",
             "previous_filename": "src/world.cpp", "status": "renamed"},
        ]))

    def test_unrelated_headings_do_not_satisfy_contract(self):
        body = "\n\n".join(
            f"### {name}\nA meaningful but incorrectly nested source review paragraph."
            for name in SECTIONS
        )
        self.assertEqual(len(SECTIONS), len(validate(body)))


if __name__ == "__main__":
    unittest.main()
