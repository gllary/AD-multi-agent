# -*- coding: utf-8 -*-
"""Path and artifact configuration for the release package.

Expected package layout under ``pipeline/``:

    data/
      ids/          optional cohort ID lists
      cp/           CP1/CP2/CP2E input tables
      features/     optional ECG or text-derived feature files
    artifacts/
      policy/       frozen policy thresholds
      precomputed/  retained score tables for released cohorts
      models/lgbm/  CP1, CP2 and CP2E fold boosters
      models/cnn/   optional CP3 waveform models
    outputs/        generated outputs
    src/llm_tool_multi_agent/
"""

from __future__ import annotations

from pathlib import Path


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def project_root() -> Path:
    """``llm_tool_multi_agent`` 仓库根（含 data、artifacts、src）。"""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MAIN_PROJECT_ROOT = PROJECT_ROOT.parent
RELEASE_DATA_DIR = MAIN_PROJECT_ROOT / "data"
RELEASE_COHORT_D_RAW_DIR = RELEASE_DATA_DIR / "raw_data" / "cohort_D"
RELEASE_COHORT_D_DERIVED_DIR = RELEASE_DATA_DIR / "derived" / "cohort_D"

# Frozen policy and retained precomputed score tables.
POLICY_JSON = ARTIFACTS_DIR / "policy" / "best_policy_thresholds.json"
OOF_SCORE_TABLE = _first_existing(
    ARTIFACTS_DIR / "precomputed" / "cohort_D" / "oof_score_table.csv",
    RELEASE_COHORT_D_DERIVED_DIR / "oof_score_table.csv",
)
OOF_SCORE_TABLE_CP3_TEXT = _first_existing(
    ARTIFACTS_DIR / "precomputed" / "cohort_D" / "oof_score_table_cp3_text.csv",
    RELEASE_COHORT_D_DERIVED_DIR / "oof_score_table_cp3_text.csv",
)

# Clinical pathway input tables. Release files use cohort_D_* names; users can
# also provide generic dataset_* names under pipeline/data/cp/.
CP_SRC_DIR = _first_existing(DATA_DIR / "cp", RELEASE_COHORT_D_RAW_DIR)
if (CP_SRC_DIR / "cohort_D_CP1_demo_history_exam.csv").exists():
    CP_CSV = {
        "CP1": "cohort_D_CP1_demo_history_exam.csv",
        "CP2": "cohort_D_CP2_demo_history_exam_lab.csv",
        "CP2E": "cohort_D_CP2E_demo_history_exam_lab_echo.csv",
    }
else:
    CP_CSV = {
        "CP1": "dataset_CP1_demo_history_exam.csv",
        "CP2": "dataset_CP2_demo_history_exam_lab.csv",
        "CP2E": "dataset_CP2E_demo_history_exam_lab_echo.csv",
    }

# Column-alignment reference. In release use, the inference CP files are the
# reference schema.
CP_TRAIN_REF_DIR = CP_SRC_DIR

COHORT_D_IDS = _first_existing(
    DATA_DIR / "ids" / "cohort_D_ids.csv",
    RELEASE_COHORT_D_DERIVED_DIR / "RETAINED_sample_ids.csv",
)
COHORT_V1_IDS = DATA_DIR / "ids" / "cohort_V1_ids.csv"

ECG_NPY = DATA_DIR / "features" / "ecg_signals.npy"
ECG_IDS = DATA_DIR / "features" / "ecg_signal_ids.csv"
ECG_TEXT_CSV = _first_existing(
    DATA_DIR / "features" / "ecg_raw_text_from_xml_cleaned.csv",
)
ECG_TEXT_JSON = _first_existing(
    DATA_DIR / "features" / "ecg_raw_text_from_xml_cleaned.json",
)
ECG_MEASUREMENTS_CSV = _first_existing(
    DATA_DIR / "features" / "ecg_measurements_from_xml_cleaned.csv",
)

MULTI_AGENT_DIR = ARTIFACTS_DIR / "models" / "lgbm"
S1B_DIR = ARTIFACTS_DIR / "models" / "cnn"

PRECOMPUTED_COHORT_V1_DIR = ARTIFACTS_DIR / "precomputed" / "cohort_V1"
PRECOMPUTED_COHORT_V1_CP3_TEXT = PRECOMPUTED_COHORT_V1_DIR / "holdout_score_table_cp3_text.csv"

DEFAULT_MODEL = "provider-model-name"
DEFAULT_API_BASE = "https://api.openai.com/v1"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
