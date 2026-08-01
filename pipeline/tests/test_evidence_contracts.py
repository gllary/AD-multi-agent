"""Tests for explicit missingness and field-grounded evidence contracts."""

from __future__ import annotations

import unittest

import pandas as pd

from llm_tool_multi_agent.curated_evidence import (
    curated_evidence_from_row,
    evidence_contract_from_text,
)


class EvidenceContractTest(unittest.TestCase):
    def test_missing_fields_are_rendered_as_unknown(self) -> None:
        row = pd.Series(
            {
                "Age": 61.0,
                "Sex": float("nan"),
                "history__sudden_onset_pain": 1.0,
                "text_suggests_ad": 1.0,
            }
        )
        text = curated_evidence_from_row(
            row,
            include_columns=(
                "Age",
                "Sex",
                "history__sudden_onset_pain",
                "history__severe_pain",
                "text_suggests_ad",
            ),
        )

        self.assertIn("- Age: 61", text)
        self.assertIn("- Sex: unknown", text)
        self.assertIn("- history__severe_pain: unknown", text)
        self.assertNotIn("text_suggests_ad", text)

        references, missing = evidence_contract_from_text(text)
        self.assertEqual(
            references,
            ["Age=61", "history__sudden_onset_pain=1"],
        )
        self.assertEqual(missing, ["Sex", "history__severe_pain"])


if __name__ == "__main__":
    unittest.main()
