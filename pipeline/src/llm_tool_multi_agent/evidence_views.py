# -*- coding: utf-8 -*-
"""Role-bounded evidence views for specialist agents."""

from __future__ import annotations

from .curated_evidence import curated_evidence_from_row
from .quantitative_tools import load_cp_row

# These allowlists mirror the released feature dictionary. Unsupported general
# vital-sign, timestamp, and malperfusion fields are not inferred, and
# diagnostic-summary extractor outputs are intentionally absent.
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

ECHO_COLUMNS = (
    "echo__ascending_aorta_dilated",
    "echo__aortic_valve_disease",
    "echo__pericardial_effusion",
    "echo__suspected_intimal_flap",
)

CP4_ALL_COLUMNS = LAB_COLUMNS + ECHO_COLUMNS

ECG_CONCEPT_COLUMNS = (
    "ecg_text_st_elevation",
    "ecg_text_st_depression",
    "ecg_text_arrhythmia",
    "ecg_text_acs_like_ecg",
    "ecg_pattern_risk_context",
    "kw_abnormal_ecg",
    "kw_arrhythmia",
    "kw_bradycardia",
    "kw_lvh_or_rvh",
    "kw_qt_prolonged",
    "kw_qwave_abnormal",
    "kw_st_depression",
    "kw_st_elevation",
    "kw_stmt_count",
    "kw_tachycardia",
    "kw_text_len",
    "kw_twave_change",
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


def build_cp4_full_view(patient_id: str, max_items: int = 64) -> str:
    row = load_cp_row(patient_id, "CP4")
    return curated_evidence_from_row(
        row,
        max_items=max_items,
        include_columns=CP4_ALL_COLUMNS,
    )


def build_ecg_view(patient_id: str, max_items: int = 64) -> str:
    row = load_cp_row(patient_id, "CP3")
    return curated_evidence_from_row(
        row,
        max_items=max_items,
        include_columns=ECG_CONCEPT_COLUMNS + ECG_MEASUREMENT_COLUMNS,
    )


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
        return "\n".join(
            [
                "Prior structured evidence:",
                build_lab_view(patient_id),
                "",
                "CP3 ECG evidence:",
                build_ecg_view(patient_id),
            ]
        )
    if stage == "CP4":
        return build_cp4_full_view(patient_id)
    raise ValueError(f"Unknown stage: {stage}")
