# SPDX-FileCopyrightText: Copyright (c) 2026 thomthehound
# SPDX-License-Identifier: Apache-2.0
"""Covers the review gate's nontrivial PR-body parsing and empty templates."""
import unittest

from check_architecture_review import SECTIONS, validate


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

    def test_unrelated_headings_do_not_satisfy_contract(self):
        body = "\n\n".join(
            f"### {name}\nA meaningful but incorrectly nested source review paragraph."
            for name in SECTIONS
        )
        self.assertEqual(len(SECTIONS), len(validate(body)))


if __name__ == "__main__":
    unittest.main()
