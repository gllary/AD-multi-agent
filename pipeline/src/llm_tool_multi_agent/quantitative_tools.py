"""Policy and evidence helpers for CP1-CP4 scores."""

from __future__ import annotations

import json
import math

import pandas as pd

from .config import (
    CP3_FEATURES_CSV,
    CP_CSV,
    CP_SRC_DIR,
    POLICY_JSON,
)


PATHWAY_STAGES = ("CP1", "CP2", "CP3", "CP4")
REQUIRED_SCORE_COLUMNS = {"ID", "label", *PATHWAY_STAGES}


def _sample_ids(frame: pd.DataFrame, invalid: pd.Series) -> list[str]:
    return frame.loc[invalid, "ID"].astype(str).head(5).tolist()


def validate_binary_series(values: pd.Series, name: str) -> pd.Series:
    """Return a normalized integer series after enforcing a binary contract."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return numeric.astype(int)


def validate_score_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the patient-level pathway input contract."""
    missing = REQUIRED_SCORE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Score table missing columns: {sorted(missing)}")

    validated = frame.copy()
    invalid_ids = (
        validated["ID"].isna()
        | validated["ID"].astype(str).str.strip().eq("")
    )
    if invalid_ids.any():
        raise ValueError("Score table contains missing or blank patient IDs")
    validated["ID"] = validated["ID"].astype(str)
    if validated["ID"].duplicated().any():
        raise ValueError("Score table contains duplicate patient IDs")

    labels = pd.to_numeric(validated["label"], errors="coerce")
    invalid_labels = labels.isna() | ~labels.isin((0, 1))
    if invalid_labels.any():
        raise ValueError(
            "label must contain only binary 0/1 values; invalid patient IDs: "
            f"{_sample_ids(validated, invalid_labels)}"
        )
    validated["label"] = labels.astype(int)

    for stage in PATHWAY_STAGES:
        values = pd.to_numeric(validated[stage], errors="coerce")
        finite = values.map(
            lambda value: bool(pd.notna(value) and math.isfinite(float(value)))
        )
        invalid_scores = ~finite | ~values.between(0.0, 1.0, inclusive="both")
        if invalid_scores.any():
            raise ValueError(
                f"{stage} scores must be finite values in [0, 1]; invalid patient IDs: "
                f"{_sample_ids(validated, invalid_scores)}"
            )
        validated[stage] = values.astype(float)

    return validated


def load_policy() -> dict:
    return json.loads(POLICY_JSON.read_text(encoding="utf-8"))


def risk_level(score: float, continue_threshold: float, action_threshold: float) -> str:
    score = float(score)
    continue_threshold = float(continue_threshold)
    action_threshold = float(action_threshold)
    for name, value in (
        ("score", score),
        ("continue_threshold", continue_threshold),
        ("action_threshold", action_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be a finite value in [0, 1]; got {value!r}"
            )
    if continue_threshold > action_threshold:
        raise ValueError(
            "continue_threshold must be less than or equal to action_threshold"
        )
    if score < continue_threshold:
        return "low"
    if score >= action_threshold:
        return "high"
    return "intermediate"


def load_cp_row(patient_id: str, stage: str) -> pd.Series:
    if stage == "CP3":
        source = CP3_FEATURES_CSV
    else:
        source = CP_SRC_DIR / CP_CSV[stage]
    if not source.exists():
        raise FileNotFoundError(f"Stage-bounded evidence table not found: {source}")
    frame = pd.read_csv(source)
    frame["ID"] = frame["ID"].astype(str)
    matched = frame[frame["ID"] == str(patient_id)]
    if matched.empty:
        raise KeyError(f"Patient ID {patient_id} not found in {source}")
    return matched.iloc[0]
