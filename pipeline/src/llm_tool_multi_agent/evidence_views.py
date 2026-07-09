# -*- coding: utf-8 -*-
"""Role-bounded evidence views for specialist agents."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .curated_evidence import curated_evidence_from_row
from .config import ECG_MEASUREMENTS_CSV, ECG_TEXT_CSV
from .quantitative_tools import load_cp_row

HISTORY_COLUMNS = (
    "Age",
    "Sex",
    "history__sudden_onset_pain",
    "history__severe_pain",
    "history__tearing_pain",
    "history__migrating_pain",
    "history__trauma_related",
    "history__marfan_or_ctd",
    "history__aortic_disease_history",
)

EXAM_COLUMNS = (
    "Age",
    "Sex",
    "exam__pulse_deficit",
    "exam__bp_difference",
    "exam__new_aortic_regurgitation_murmur",
    "exam__neurologic_deficit",
    "exam__hypotension_or_shock",
    "exam__text_suggests_ais",
    "exam__text_suggests_aos",
)

HISTORY_EXAM_COLUMNS = HISTORY_COLUMNS + EXAM_COLUMNS[2:]

LAB_COLUMNS = HISTORY_EXAM_COLUMNS + (
    "troponin_abnormal",
    "D_D_abnormal",
    "D_D_log",
    "NT_proBNP_log",
    "Mb_log",
    "CK_MB_log",
)

LAB_CONTEXT_COLUMNS = HISTORY_EXAM_COLUMNS

LAB_BIOMARKER_COLUMNS = (
    "Age",
    "Sex",
    "troponin_abnormal",
    "D_D_abnormal",
    "D_D_log",
    "NT_proBNP_log",
    "Mb_log",
    "CK_MB_log",
)

ECHO_COLUMNS = LAB_COLUMNS + (
    "echo__ascending_aorta_dilated",
    "echo__aortic_valve_disease",
    "echo__pericardial_effusion",
    "echo__suspected_intimal_flap",
    "echo__suggest_ad_on_echo",
)

ECG_MEASUREMENT_COLUMNS = (
    "ecg_ventricularrate",
    "ecg_atrialrate",
    "ecg_printerval",
    "ecg_qrsduration",
    "ecg_qtinterval",
    "ecg_qtcbazett",
    "ecg_paxis",
    "ecg_raxis",
    "ecg_taxis",
    "ecg_sv1",
    "ecg_rv5",
    "ecg_sv1rv5",
    "ecg_rrinterval",
)


@lru_cache(maxsize=1)
def _load_ecg_text_map() -> dict[str, str]:
    if not ECG_TEXT_CSV.exists():
        return {}
    df = pd.read_csv(ECG_TEXT_CSV)
    if "ID" not in df.columns or "ecg_diagnosis_text" not in df.columns:
        return {}
    df["ID"] = df["ID"].astype(str).str.strip()
    return {
        pid: str(text).strip()
        for pid, text in zip(df["ID"], df["ecg_diagnosis_text"])
        if pd.notna(text) and str(text).strip()
    }


@lru_cache(maxsize=1)
def _load_ecg_measurement_map() -> dict[str, dict[str, object]]:
    if not ECG_MEASUREMENTS_CSV.exists():
        return {}
    df = pd.read_csv(ECG_MEASUREMENTS_CSV)
    if "ID" not in df.columns:
        return {}
    df["ID"] = df["ID"].astype(str).str.strip()
    cols = ["ID"] + [c for c in ECG_MEASUREMENT_COLUMNS if c in df.columns]
    slim = df[cols].copy()
    out: dict[str, dict[str, object]] = {}
    for _, row in slim.iterrows():
        pid = str(row["ID"]).strip()
        out[pid] = row.to_dict()
    return out


def build_history_view(patient_id: str, max_items: int = 24) -> str:
    row = load_cp_row(patient_id, "CP1")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=HISTORY_COLUMNS)


def build_examination_view(patient_id: str, max_items: int = 24) -> str:
    row = load_cp_row(patient_id, "CP1")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=EXAM_COLUMNS)


def build_history_exam_view(patient_id: str, max_items: int = 48) -> str:
    row = load_cp_row(patient_id, "CP1")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=HISTORY_EXAM_COLUMNS)


def build_lab_view(patient_id: str, max_items: int = 48) -> str:
    row = load_cp_row(patient_id, "CP2")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=LAB_COLUMNS)


def build_lab_context_view(patient_id: str, max_items: int = 32) -> str:
    row = load_cp_row(patient_id, "CP2")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=LAB_CONTEXT_COLUMNS)


def build_lab_biomarker_view(patient_id: str, max_items: int = 24) -> str:
    row = load_cp_row(patient_id, "CP2")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=LAB_BIOMARKER_COLUMNS)


def build_echo_view(patient_id: str, max_items: int = 48) -> str:
    row = load_cp_row(patient_id, "CP4")
    return curated_evidence_from_row(row, max_items=max_items, include_columns=ECHO_COLUMNS)


def build_ecg_view(patient_id: str, max_items: int = 24) -> str:
    row = load_cp_row(patient_id, "CP1")
    history_exam = curated_evidence_from_row(row, max_items=max_items, include_columns=HISTORY_EXAM_COLUMNS)

    text_map = _load_ecg_text_map()
    ecg_text = text_map.get(str(patient_id).strip(), "")

    measurement_map = _load_ecg_measurement_map()
    measurement_row = measurement_map.get(str(patient_id).strip())
    measurement_block = "(no ECG measurement fields available)"
    if measurement_row:
        measurement_series = pd.Series(measurement_row)
        measurement_block = curated_evidence_from_row(
            measurement_series,
            max_items=16,
            include_columns=ECG_MEASUREMENT_COLUMNS,
        )

    parts = [
        "ECG diagnosis text:",
        ecg_text if ecg_text else "(no ECG diagnosis text available)",
        "",
        "ECG measurements:",
        measurement_block,
        "",
        "Minimal history/exam context:",
        history_exam,
    ]
    return "\n".join(parts)


def build_specialist_view(patient_id: str, stage: str, role: str | None = None) -> str:
    if stage == "CP1" and role == "history":
        return build_history_view(patient_id)
    if stage == "CP1" and role == "examination":
        return build_examination_view(patient_id)
    if stage == "CP1":
        return build_history_exam_view(patient_id)
    if stage == "CP2" and role == "lab_context":
        return build_lab_context_view(patient_id)
    if stage == "CP2" and role == "lab_biomarker":
        return build_lab_biomarker_view(patient_id)
    if stage == "CP2":
        return build_lab_view(patient_id)
    if stage == "CP3":
        return build_ecg_view(patient_id)
    if stage == "CP4":
        return build_echo_view(patient_id)
    raise ValueError(f"Unknown stage: {stage}")


def build_single_agent_view(patient_id: str, stage: str) -> str:
    if stage == "CP1":
        return build_history_exam_view(patient_id)
    if stage == "CP2":
        return build_lab_view(patient_id)
    if stage == "CP3":
        return build_ecg_view(patient_id)
    if stage == "CP4":
        return build_echo_view(patient_id)
    raise ValueError(f"Unknown stage: {stage}")
