"""Policy and evidence helpers for CP1-CP4 scores."""

from __future__ import annotations

import json

import pandas as pd

from .config import (
    CP3_FEATURES_CSV,
    CP_CSV,
    CP_SRC_DIR,
    POLICY_JSON,
)


def load_policy() -> dict:
    return json.loads(POLICY_JSON.read_text(encoding="utf-8"))


def risk_level(score: float, continue_threshold: float, action_threshold: float) -> str:
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
