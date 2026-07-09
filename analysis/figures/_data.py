"""Unified retained-result data loader for release figures and tables.

This module intentionally reads only the current retained cohort prediction files:

- data/derived/cohort_D/FINAL_retained_predictions.csv
- restricted_inputs/cohort_V1/FINAL_retained_predictions.csv
- restricted_inputs/cohort_V2/FINAL_retained_predictions.csv

No project-specific merging or intermediate run folders are used. Each cohort is
represented by one retained prediction file with the final binary columns.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESTRICTED_INPUT_ROOT = Path(os.environ.get("AD_RESTRICTED_INPUT_ROOT", ROOT / "restricted_inputs"))

COHORT_FILES: dict[str, Path] = {
    "cohort_D": ROOT / "data/derived/cohort_D/FINAL_retained_predictions.csv",
    "cohort_V1": RESTRICTED_INPUT_ROOT / "cohort_V1" / "FINAL_retained_predictions.csv",
    "cohort_V2": RESTRICTED_INPUT_ROOT / "cohort_V2" / "FINAL_retained_predictions.csv",
}

METHOD_TO_COLUMN = {
    "canonical": "canonical_pred",
    "single_agent": "single_pred",
    "multi_agent": "multi_raw_pred",
}

COHORT_META: dict[str, dict] = {
    "cohort_D": {
        "role": "development",
        "site": "Development site",
        "site_label": "Site D",
        "label_sop": "Physician primary annotation",
        "n": 1010,
        "prevalence": 528 / 1010,
    },
    "cohort_V1": {
        "role": "zero-shot external",
        "site": "External site 1",
        "site_label": "Site V1",
        "label_sop": "Physician adjudication",
        "n": 173,
        "prevalence": 78 / 173,
    },
    "cohort_V2": {
        "role": "zero-shot external",
        "site": "External site 2",
        "site_label": "Site V2",
        "label_sop": (
            "CTA/MRA or intra-operative confirmation for AD-positive labels; "
            "physician record review plus longitudinal telephone follow-up for "
            "AD-negative labels"
        ),
        "n": 15109,
        "prevalence": 4067 / 15109,
    },
}

COHORT_ORDER = ["cohort_D", "cohort_V1", "cohort_V2"]
METHOD_ORDER = ["canonical", "single_agent", "multi_agent"]
ESCALATE_ACTIONS = {"direct_cta", "urgent_transfer"}


@dataclass
class CohortMethodPredictions:
    """Per-patient labels and final predicted labels."""

    cohort: str
    method: str
    df: pd.DataFrame  # columns: ID, label, final_pred

    def __len__(self) -> int:
        return len(self.df)

    @property
    def y(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def yhat(self) -> np.ndarray:
        return self.df["final_pred"].to_numpy()


def load_predictions(cohort: str, method: str) -> CohortMethodPredictions:
    """Return current per-patient predictions for a cohort and method."""

    if cohort not in COHORT_FILES:
        raise KeyError(f"Unknown cohort: {cohort}")
    if method not in METHOD_TO_COLUMN:
        raise KeyError(f"Unknown method: {method}")

    source = COHORT_FILES[cohort]
    col = METHOD_TO_COLUMN[method]
    raw = pd.read_csv(source)
    out = pd.DataFrame({
        "ID": raw["ID"].astype(str),
        "label": raw["label"].astype(int),
        "final_pred": raw[col].astype(int),
    })
    out["final_action"] = np.where(out["final_pred"].eq(1), "escalate", "observe_or_reassess")
    return CohortMethodPredictions(cohort=cohort, method=method, df=out)


def load_all() -> dict[tuple[str, str], CohortMethodPredictions]:
    """Load every current cohort-method prediction frame."""

    return {
        (cohort, method): load_predictions(cohort, method)
        for cohort in COHORT_ORDER
        for method in METHOD_ORDER
    }


def load_cohort_labels(cohort: str) -> pd.DataFrame:
    """Return ID and label for a current retained cohort."""

    pred = load_predictions(cohort, "multi_agent")
    return pred.df[["ID", "label"]].copy()


if __name__ == "__main__":
    for cohort in COHORT_ORDER:
        for method in METHOD_ORDER:
            p = load_predictions(cohort, method)
            tp = int(((p.y == 1) & (p.yhat == 1)).sum())
            tn = int(((p.y == 0) & (p.yhat == 0)).sum())
            fp = int(((p.y == 0) & (p.yhat == 1)).sum())
            fn = int(((p.y == 1) & (p.yhat == 0)).sum())
            print(f"{cohort:>15} | {method:>13} | n={len(p):5d} TP={tp:5d} TN={tn:5d} FP={fp:5d} FN={fn:5d}")
