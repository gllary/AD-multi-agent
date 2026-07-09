"""Per-cohort baseline-feature loader.

Returns a unified per-patient feature DataFrame for each cohort, restricted
to the patients actually used by the current multi-agent paper
(1,010 / 173 / 630 / 14,748).

Feature schema is the union of CP1 / CP2 / CP2E + selected echo:
    Age, Sex,
    history__{sudden_onset, severe, tearing, migrating}_pain,
    history__{trauma_related, marfan_or_ctd, aortic_disease_history},
    exam__{pulse_deficit, bp_difference, new_aortic_regurgitation_murmur,
           neurologic_deficit, hypotension_or_shock,
           text_suggests_ais, text_suggests_aos},
    troponin_abnormal, D_D_abnormal,
    D_D_log, NT_proBNP_log, Mb_log, CK_MB_log,
    echo__{ascending_aorta_dilated, aortic_valve_disease, pericardial_effusion,
           suspected_intimal_flap, suggest_ad_on_echo},
    AD  # binary reference label column, interpreted as AD-positive/AD-negative

Cohort sources:
    datasetA       — shared/multi_agent_data/cp_features/dataset_CP2E*.csv
                     filtered to shared/multi_agent_data/ids/datasetA_ids.csv
    datasetB_v6    — same shared file, filtered to datasetB_v6_ids.csv
    xiangya_720    — phase1_qwen_720_bundle/cp_inputs/dataset_CP2E*.csv
    xiangya_16218  — phase1_qwen_16218_bundle/cp_inputs/dataset_CP2E*.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[3]))

CP2E_PATHS = {
    "datasetA":      ROOT / "shared/multi_agent_data/cp_features/dataset_CP2E_demo_history_exam_lab_echo.csv",
    "datasetB_v6":   ROOT / "shared/multi_agent_data/cp_features/dataset_CP2E_demo_history_exam_lab_echo.csv",
    "xiangya_720":   ROOT / "phase1_qwen_720_bundle/cp_inputs/dataset_CP2E_demo_history_exam_lab_echo.csv",
    "xiangya_16218": ROOT / "phase1_qwen_16218_bundle/cp_inputs/dataset_CP2E_demo_history_exam_lab_echo.csv",
}

ID_FILTERS = {
    "datasetA":    ROOT / "shared/multi_agent_data/ids/datasetA_ids.csv",
    "datasetB_v6": ROOT / "shared/multi_agent_data/ids/datasetB_v6_ids.csv",
}

# Continuous lab columns (already log-transformed in source)
CONT_FEATURES = ["Age", "D_D_log", "NT_proBNP_log", "Mb_log", "CK_MB_log"]

# Binary / categorical features (string '0'/'1'/'unknown' encoded)
BIN_FEATURES = [
    "history__sudden_onset_pain", "history__severe_pain",
    "history__tearing_pain", "history__migrating_pain",
    "history__trauma_related", "history__marfan_or_ctd",
    "history__aortic_disease_history",
    "exam__pulse_deficit", "exam__bp_difference",
    "exam__new_aortic_regurgitation_murmur",
    "exam__neurologic_deficit", "exam__hypotension_or_shock",
    "troponin_abnormal", "D_D_abnormal",
    "echo__ascending_aorta_dilated", "echo__aortic_valve_disease",
    "echo__pericardial_effusion", "echo__suspected_intimal_flap",
    "echo__suggest_ad_on_echo",
]


def _to_bool01(series: pd.Series) -> pd.Series:
    """Return 1 / 0 / NaN ('unknown' or blank). NaN-as-unknown is the convention."""
    s = series.astype(str).str.strip()
    out = pd.Series(np.where(s == "1", 1.0, np.where(s == "0", 0.0, np.nan)), index=series.index)
    return out


def load_cohort_features(cohort: str) -> pd.DataFrame:
    if cohort not in CP2E_PATHS:
        raise KeyError(cohort)
    df = pd.read_csv(CP2E_PATHS[cohort])
    if cohort in ID_FILTERS:
        ids = pd.read_csv(ID_FILTERS[cohort])["ID"]
        df = df[df["ID"].isin(ids)].copy()
    out = df.copy()
    # Normalise binary columns to 1/0/NaN
    for col in BIN_FEATURES:
        if col in out.columns:
            out[col] = _to_bool01(out[col])
        else:
            out[col] = np.nan
    # Sex column: '男'/'女' or 'M'/'F'
    if "Sex" in out.columns:
        sex_str = out["Sex"].astype(str).str.strip()
        out["sex_male"] = np.where(sex_str.isin(["男", "M", "Male", "male", "1"]), 1.0,
                            np.where(sex_str.isin(["女", "F", "Female", "female", "0"]), 0.0, np.nan))
    # Age numeric
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    # Ensure the binary AD reference label column exists.
    if "AD" not in out.columns:
        raise RuntimeError(f"AD column missing in {cohort}")
    out["AD"] = pd.to_numeric(out["AD"], errors="coerce").astype("Int64")
    return out


if __name__ == "__main__":
    for c in CP2E_PATHS:
        df = load_cohort_features(c)
        pos = int((df["AD"] == 1).sum())
        neg = int((df["AD"] == 0).sum())
        print(f"{c:>15} | n={len(df):5d} pos={pos:5d} neg={neg:5d}")
