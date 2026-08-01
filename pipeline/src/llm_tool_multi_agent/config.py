"""Paths and frozen runtime settings for the public method release."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
REPOSITORY_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RELEASE_DATA_DIR = REPOSITORY_ROOT / "data"

DEVELOPMENT_RAW_DIR = RELEASE_DATA_DIR / "raw_data" / "development_cohort"

POLICY_JSON = ARTIFACTS_DIR / "policy" / "frozen_policy_thresholds.json"

CP_SRC_DIR = Path(
    os.environ.get("AD_EVIDENCE_DIR", str(DEVELOPMENT_RAW_DIR))
).expanduser()
CP_CSV = {
    "CP1": os.environ.get(
        "AD_CP1_EVIDENCE_FILE", "development_CP1_demo_history_exam.csv"
    ),
    "CP2": os.environ.get(
        "AD_CP2_EVIDENCE_FILE", "development_CP2_demo_history_exam_lab.csv"
    ),
    "CP4": os.environ.get(
        "AD_CP4_EVIDENCE_FILE", "development_CP4_demo_history_exam_lab_echo.csv"
    ),
}
# Controlled-access evaluations provide one row per patient containing only
# concept-coded ECG fields and structured measurements.
CP3_FEATURES_CSV = Path(
    os.environ.get(
        "AD_CP3_EVIDENCE_FILE",
        str(DATA_DIR / "features" / "ecg_structured_features.csv"),
    )
).expanduser()

DEFAULT_MODEL = "Qwen3-235B-A22B"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_ATTEMPTS = 3

OUTPUT_DIR = PROJECT_ROOT / "outputs"
