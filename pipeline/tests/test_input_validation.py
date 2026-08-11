"""Tests for score-table, label, and probability-domain validation."""

from __future__ import annotations

import unittest

import pandas as pd

from llm_tool_multi_agent.quantitative_tools import (
    risk_level,
    validate_binary_series,
    validate_score_table,
)


def valid_score_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["D000001", "D000002"],
            "label": [0, 1],
            "CP1": [0.0, 1.0],
            "CP2": [0.1, 0.9],
            "CP3": [0.2, 0.8],
            "CP4": [0.3, 0.7],
        }
    )


class ScoreTableValidationTest(unittest.TestCase):
    def test_accepts_and_normalizes_valid_table(self) -> None:
        frame = valid_score_table()
        frame["CP1"] = frame["CP1"].astype(str)

        validated = validate_score_table(frame)

        self.assertEqual(validated["label"].tolist(), [0, 1])
        self.assertEqual(validated["CP1"].tolist(), [0.0, 1.0])

    def test_rejects_nonbinary_label(self) -> None:
        frame = valid_score_table()
        frame.loc[1, "label"] = 2
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            validate_score_table(frame)

    def test_rejects_nonfinite_or_out_of_range_scores(self) -> None:
        for bad_value in (float("nan"), float("inf"), -0.1, 1.1, "not-a-score"):
            with self.subTest(bad_value=bad_value):
                frame = valid_score_table()
                frame["CP2"] = frame["CP2"].astype(object)
                frame.loc[0, "CP2"] = bad_value
                with self.assertRaisesRegex(ValueError, r"finite values in \[0, 1\]"):
                    validate_score_table(frame)

    def test_allocation_summary_rejects_nonbinary_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reference-standard labels"):
            validate_binary_series(pd.Series([0, 2]), "Reference-standard labels")
        with self.assertRaisesRegex(ValueError, "Assigned-escalation indicators"):
            validate_binary_series(pd.Series([0, -1]), "Assigned-escalation indicators")


class RiskLevelValidationTest(unittest.TestCase):
    def test_rejects_invalid_score_domain(self) -> None:
        for bad_value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, r"finite value in \[0, 1\]"):
                    risk_level(bad_value, 0.1, 0.5)

    def test_rejects_inverted_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than or equal"):
            risk_level(0.3, 0.6, 0.5)


if __name__ == "__main__":
    unittest.main()
