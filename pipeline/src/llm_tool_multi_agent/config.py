# -*- coding: utf-8 -*-
"""
路径约定：本文件位于 ``<项目根>/src/llm_tool_multi_agent/config.py``。

项目根目录 ``PROJECT_ROOT`` 下应有::

  data/
    ids/           — 队列 ID 列表
    cp/            — CP1/CP2/CP2E 宽表 CSV（与主研究同 schema）
    features/      — ECG：ecg_signals.npy + ecg_signal_ids.csv（可选）
  artifacts/
    policy/        — best_policy_thresholds.json
    precomputed/   — OOF 分数表、datasetB_v6 预计算概率等
    models/lgbm/   — CP1, CP2, CP2E 子目录，各含 fold_models/fold*_booster.txt
    models/cnn/    — fold_models/fold*_best.pt（可选）
  outputs/         — 运行输出
  src/llm_tool_multi_agent/  — 本包
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
MAIN_INTERIM_FEATURE_DIR = MAIN_PROJECT_ROOT / "data" / "interim" / "features"

# 策略与预计算分数（随仓库提供小文件；重训后可覆盖）
POLICY_JSON = ARTIFACTS_DIR / "policy" / "best_policy_thresholds.json"
OOF_SCORE_TABLE = ARTIFACTS_DIR / "precomputed" / "datasetA" / "oof_score_table.csv"
OOF_SCORE_TABLE_CP3_TEXT = ARTIFACTS_DIR / "precomputed" / "datasetA" / "oof_score_table_cp3_text.csv"

# 临床路径宽表（用户将 CSV 放入 data/cp/）
CP_SRC_DIR = DATA_DIR / "cp"
CP_CSV = {
    "CP1": "dataset_CP1_demo_history_exam.csv",
    "CP2": "dataset_CP2_demo_history_exam_lab.csv",
    "CP2E": "dataset_CP2E_demo_history_exam_lab_echo.csv",
}

# 列对齐参考：与推断使用同一套 cp 文件即可（通常为全队列表）
CP_TRAIN_REF_DIR = CP_SRC_DIR

DATASET_A_IDS = DATA_DIR / "ids" / "datasetA_ids.csv"
DATASET_B_V6_IDS = DATA_DIR / "ids" / "datasetB_v6_ids.csv"

ECG_NPY = DATA_DIR / "features" / "ecg_signals.npy"
ECG_IDS = DATA_DIR / "features" / "ecg_signal_ids.csv"
ECG_TEXT_CSV = _first_existing(
    DATA_DIR / "features" / "ecg_raw_text_from_xml_cleaned.csv",
    MAIN_INTERIM_FEATURE_DIR / "ecg_raw_text_from_xml_cleaned.csv",
)
ECG_TEXT_JSON = _first_existing(
    DATA_DIR / "features" / "ecg_raw_text_from_xml_cleaned.json",
    MAIN_INTERIM_FEATURE_DIR / "ecg_raw_text_from_xml_cleaned.json",
)
ECG_MEASUREMENTS_CSV = _first_existing(
    DATA_DIR / "features" / "ecg_measurements_from_xml_cleaned.csv",
    MAIN_INTERIM_FEATURE_DIR / "ecg_measurements_from_xml_cleaned.csv",
)

MULTI_AGENT_DIR = ARTIFACTS_DIR / "models" / "lgbm"
S1B_DIR = ARTIFACTS_DIR / "models" / "cnn"

PRECOMPUTED_V6_DIR = ARTIFACTS_DIR / "precomputed" / "datasetB_v6"
PRECOMPUTED_V6_CP3_TEXT = PRECOMPUTED_V6_DIR / "holdout_score_table_cp3_text.csv"

DEFAULT_MODEL = "Qwen3.5-235B-A22B"
DEFAULT_API_BASE = "https://api.openai.com/v1"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
