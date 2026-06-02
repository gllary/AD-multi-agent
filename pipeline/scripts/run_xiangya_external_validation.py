#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Xiangya third-center external validation with ECG text-only CP3."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tool_multi_agent.canonical_engine import run_canonical_for_patient
from llm_tool_multi_agent.config import OUTPUT_DIR
from llm_tool_multi_agent.deliberation import summarize_coordinator_history, summarize_specialist_history
from llm_tool_multi_agent.evidence_views import _load_ecg_measurement_map, _load_ecg_text_map
from llm_tool_multi_agent.llm_client import LLMClient, _stub_coordinator, _stub_single_agent, _stub_specialist
from llm_tool_multi_agent.pathway_engine import AGENT_NAMES, STAGE_SPECIALISTS, run_pathway_for_patient
from llm_tool_multi_agent.lgbm_preprocess import add_missing_indicators, drop_leakage_cols, encode_llm_string_columns
from llm_tool_multi_agent.quantitative_tools import load_policy
from llm_tool_multi_agent.safety_layer import POSITIVE_ACTIONS, action_to_next_stage, allowed_actions_for_stage, validate_coordinator_proposal
from llm_tool_multi_agent.single_agent_engine import run_single_agent_for_patient
from run_pathway import _metrics, _path_metrics


MAIN_PROJECT_ROOT = PROJECT_ROOT.parent
XIANGYA_FINAL_CSV = MAIN_PROJECT_ROOT / "data" / "interim" / "xiangya_external_refined" / "xiangya_main_cohort_model_input_labeled.csv"
OUT_ROOT = OUTPUT_DIR / "xiangya_external_validation"
CP_DIR = OUT_ROOT / "cp_inputs"
FEATURE_DIR = OUT_ROOT / "features"
SCORE_DIR = OUT_ROOT / "scores"
REPORT_PATH = OUT_ROOT / "xiangya_external_validation_report_zh.md"

CP3_TEXT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "cp3_text"
LGBM_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "lgbm"


def configure_live_env_from_qwen() -> None:
    if not os.environ.get("LLM_API_KEY") and os.environ.get("QWEN_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["QWEN_API_KEY"]
    if not os.environ.get("LLM_API_BASE") and os.environ.get("QWEN_BASE_URL"):
        os.environ["LLM_API_BASE"] = os.environ["QWEN_BASE_URL"]
    if not os.environ.get("LLM_MODEL") and os.environ.get("QWEN_MODEL_NAME"):
        os.environ["LLM_MODEL"] = os.environ["QWEN_MODEL_NAME"]


def _s(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _contains(text: str, patterns: list[str]) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in patterns)


def _binary_from_text(text: str, positive_patterns: list[str]) -> str:
    if not text:
        return "unknown"
    return "1" if _contains(text, positive_patterns) else "0"


def _suggest_level(text: str, high_patterns: list[str], medium_patterns: list[str]) -> str:
    if not text:
        return "unknown"
    if _contains(text, high_patterns):
        return "high"
    if _contains(text, medium_patterns):
        return "medium"
    return "low"


def _sex_map(series: pd.Series) -> pd.Series:
    return series.map({1: 1, 2: 0, 1.0: 1, 2.0: 0, "1": 1, "2": 0, "男": 1, "女": 0})


def _clean_num(series: pd.Series) -> pd.Series:
    def parse_one(value: object) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"([<>]?)\s*([-+]?\d+(?:\.\d+)?)", str(value).replace("＜", "<").replace("＞", ">"))
        return float(m.group(2)) if m else np.nan

    return series.map(parse_one)


def _log1p(series: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(series, errors="coerce"))


def _parse_hypotension(text: str) -> str:
    if not text:
        return "unknown"
    if _contains(text, [r"休克", r"低血压"]):
        return "1"
    m = re.search(r"血压[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg", text, flags=re.IGNORECASE)
    if m:
        sbp = int(m.group(1))
        return "1" if sbp < 90 else "0"
    return "0"


def _history_exam_text(row: pd.Series) -> tuple[str, str]:
    history = " ".join([
        _s(row.get("主诉")),
        _s(row.get("现病史")),
        _s(row.get("其他相关病史（包括既往史、个人史、月经史、婚育史、家族史）")),
    ]).strip()
    exam = _s(row.get("体格检查"))
    return history, exam


def build_cp_tables(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = final_df.copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Sex"] = _sex_map(df["Sex"])

    history_text = []
    exam_text = []
    echo_text = []
    for _, row in df.iterrows():
        h, e = _history_exam_text(row)
        history_text.append(h)
        exam_text.append(e)
        echo_text.append(" ".join([_s(row.get("心脏彩超首次检测结论")), _s(row.get("心脏彩超首次检查所见"))]).strip())
    df["_history_text"] = history_text
    df["_exam_text"] = exam_text
    df["_echo_text"] = echo_text
    df["_ecg_text"] = df["ECG诊断结论"].map(_s)

    high_aas_history = [
        r"主动脉夹层", r"撕裂样", r"刀割样", r"胸背痛", r"胸腹痛", r"壁间血肿", r"穿透性溃疡",
    ]
    medium_aas_history = [
        r"突发", r"突然", r"急性起病", r"放射痛", r"迁移", r"主动脉瘤", r"马凡", r"结缔组织",
    ]
    high_aas_exam = [r"双上肢血压.*差", r"脉搏.*不对称", r"主动脉瓣.*返流", r"休克", r"低血压"]
    medium_aas_exam = [r"杂音", r"神经功能缺失", r"偏瘫", r"晕厥"]
    high_aas_echo = [r"主动脉夹层", r"内膜片", r"双腔", r"真假腔", r"壁间血肿"]
    medium_aas_echo = [r"升主动脉.*增宽", r"主动脉内径增宽", r"升主动脉扩张", r"主动脉窦部高值", r"心包积液"]

    cp1 = pd.DataFrame(
        {
            "ID": df["ID"],
            "Age": df["Age"],
            "Sex": df["Sex"],
            "history__sudden_onset_pain": df["_history_text"].map(lambda x: _binary_from_text(x, [r"突发", r"突然", r"骤然", r"急性起病"])),
            "history__severe_pain": df["_history_text"].map(lambda x: _binary_from_text(x, [r"剧痛", r"疼痛明显", r"难以忍受", r"持续数小时", r"疼痛剧烈"])),
            "history__tearing_pain": df["_history_text"].map(lambda x: _binary_from_text(x, [r"撕裂样", r"刀割样"])),
            "history__migrating_pain": df["_history_text"].map(lambda x: _binary_from_text(x, [r"迁移", r"放射痛", r"放射至", r"波及背", r"波及腹", r"肩胛区"])),
            "history__trauma_related": df["_history_text"].map(lambda x: _binary_from_text(x, [r"外伤", r"车祸", r"撞伤", r"摔伤"])),
            "history__marfan_or_ctd": df["_history_text"].map(lambda x: _binary_from_text(x, [r"马凡", r"marfan", r"结缔组织", r"风湿免疫"])),
            "history__aortic_disease_history": df["_history_text"].map(lambda x: _binary_from_text(x, [r"主动脉夹层", r"主动脉瘤", r"壁间血肿", r"穿透性溃疡", r"支架植入术后", r"bentall", r"孙氏术后"])),
            "exam__pulse_deficit": df["_exam_text"].map(lambda x: _binary_from_text(x, [r"脉搏.*消失", r"脉搏.*减弱", r"双侧.*脉搏.*不对称", r"桡动脉搏动.*弱"])),
            "exam__bp_difference": df["_exam_text"].map(lambda x: _binary_from_text(x, [r"双上肢血压.*差", r"左右上肢血压.*不一致", r"血压差"])),
            "exam__new_aortic_regurgitation_murmur": df["_exam_text"].map(lambda x: _binary_from_text(x, [r"主动脉瓣.*返流", r"舒张期杂音", r"主动脉瓣关闭不全"])),
            "exam__neurologic_deficit": df["_exam_text"].map(lambda x: _binary_from_text(x, [r"偏瘫", r"失语", r"意识障碍", r"神经功能缺失", r"肢体无力"])),
            "exam__hypotension_or_shock": df["_exam_text"].map(_parse_hypotension),
            "exam__text_suggests_ais": np.nan,
            "exam__text_suggests_aos": np.nan,
            "AAS": df["AAS"].astype(int),
        }
    )

    tni = _clean_num(df["TnI"])
    tnt = _clean_num(df["TnT"])
    dd = _clean_num(df["D-D"])
    ntprobnp = _clean_num(df["NT-proBNP"])
    mb = _clean_num(df["Mb"])
    ckmb = _clean_num(df["CK-MB"])

    troponin_flag = np.where((tni > 0.10) | (tnt > 14.0), 1.0, np.where((~tni.isna()) | (~tnt.isna()), 0.0, np.nan))
    cp2 = cp1.copy()
    cp2["troponin_abnormal"] = troponin_flag
    cp2["D_D_abnormal"] = np.where(dd.notna(), (dd > 0.5).astype(float), np.nan)
    cp2["D_D_log"] = _log1p(dd)
    cp2["NT_proBNP_log"] = _log1p(ntprobnp)
    cp2["Mb_log"] = _log1p(mb)
    cp2["CK_MB_log"] = _log1p(ckmb)

    cp2e = cp2.copy()
    cp2e["echo__ascending_aorta_dilated"] = df["_echo_text"].map(lambda x: _binary_from_text(x, [r"升主动脉.*增宽", r"主动脉内径增宽", r"升主动脉扩张", r"主动脉窦部高值"]))
    cp2e["echo__aortic_valve_disease"] = df["_echo_text"].map(lambda x: _binary_from_text(x, [r"主动脉瓣.*返流", r"主动脉瓣.*狭窄", r"主动脉瓣钙化"]))
    cp2e["echo__pericardial_effusion"] = df["_echo_text"].map(lambda x: _binary_from_text(x, [r"心包积液", r"心包少量积液", r"心包腔.*液性暗区"]))
    cp2e["echo__suspected_intimal_flap"] = df["_echo_text"].map(lambda x: _binary_from_text(x, [r"内膜片", r"双腔", r"真假腔", r"主动脉夹层", r"壁间血肿"]))
    cp2e["echo__suggest_aas_on_echo"] = df["_echo_text"].map(lambda x: _binary_from_text(x, high_aas_echo + medium_aas_echo))

    return cp1, cp2, cp2e


def build_ecg_text_features(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = final_df[["ID", "AAS", "ECG诊断结论"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["ecg_diagnosis_text"] = df["ECG诊断结论"].map(_s)
    text = df["ecg_diagnosis_text"].fillna("").astype(str).str.lower()

    def kw(col_patterns: tuple[str, ...]) -> pd.Series:
        return text.apply(lambda s: int(any(k in s for k in col_patterns)))

    df["ecg_text_st_elevation"] = kw(("st段上抬", "st段抬高", "st抬高"))
    df["ecg_text_st_depression"] = kw(("st段下移", "st段压低", "st压低"))
    df["ecg_text_arrhythmia"] = kw(("房颤", "房扑", "早搏", "心律失常", "室上速", "室速", "传导阻滞"))
    df["ecg_text_acs_like_ecg"] = kw(("st段上抬", "st段抬高", "st段下移", "st段压低", "t波改变", "st-t改变", "心肌缺血"))
    df["ecg_text_text_suggests_aas"] = [
        "high" if ("st段上抬" in s or "st段抬高" in s or "房颤" in s)
        else "medium" if ("st段下移" in s or "t波改变" in s or "心律失常" in s)
        else "low" if s else "unknown"
        for s in text
    ]
    df["kw_st_elevation"] = kw(("st段上抬", "st段抬高", "st抬高"))
    df["kw_st_depression"] = kw(("st段下移", "st段压低", "st压低"))
    df["kw_twave_change"] = kw(("t波改变", "st-t改变", "st-t异常"))
    df["kw_qwave_abnormal"] = kw(("q波异常",))
    df["kw_arrhythmia"] = kw(("房颤", "房扑", "早搏", "心律失常", "室上速", "室速", "传导阻滞"))
    df["kw_bradycardia"] = kw(("心动过缓",))
    df["kw_tachycardia"] = kw(("心动过速",))
    df["kw_qt_prolonged"] = kw(("qt间期延长", "qtc延长"))
    df["kw_lvh_or_rvh"] = kw(("左心室肥大", "右室面电压偏高", "高电压", "左室面高电压"))
    df["kw_abnormal_ecg"] = kw(("异常心电图", "临界 ecg", "临界ecg"))
    df["kw_text_len"] = text.str.len().astype(float)
    df["kw_stmt_count"] = text.apply(lambda s: float(s.count("；") + 1) if s else 0.0)
    return (
        df[["ID", "ecg_diagnosis_text"]].copy(),
        df[
            [
                "ID",
                "AAS",
                "ecg_diagnosis_text",
                "ecg_text_st_elevation",
                "ecg_text_st_depression",
                "ecg_text_arrhythmia",
                "ecg_text_acs_like_ecg",
                "ecg_text_text_suggests_aas",
                "kw_st_elevation",
                "kw_st_depression",
                "kw_twave_change",
                "kw_qwave_abnormal",
                "kw_arrhythmia",
                "kw_bradycardia",
                "kw_tachycardia",
                "kw_qt_prolonged",
                "kw_lvh_or_rvh",
                "kw_abnormal_ecg",
                "kw_text_len",
                "kw_stmt_count",
            ]
        ].copy(),
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _load_cp3_text_spec() -> dict:
    return json.loads((CP3_TEXT_MODEL_DIR / "feature_spec.json").read_text(encoding="utf-8"))


def _make_cp3_matrix(df: pd.DataFrame, spec: dict) -> np.ndarray:
    df = df.copy()
    category_maps: dict[str, list[str]] = spec["category_maps"]
    numeric_cols: list[str] = spec["numeric_cols"]
    feature_names: list[str] = spec["feature_names"]

    for col in category_maps:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    blocks: list[np.ndarray] = []
    for col, cats in category_maps.items():
        arr = np.zeros((len(df), len(cats)), dtype=float)
        cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        values = df[col].fillna("").astype(str).tolist()
        for i, value in enumerate(values):
            arr[i, cat_to_idx.get(value, 0)] = 1.0
        blocks.append(arr)
    if numeric_cols:
        blocks.append(df[numeric_cols].astype(float).values)
    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=float)
    if X.shape[1] != len(feature_names):
        raise ValueError(f"CP3 text design mismatch: matrix={X.shape[1]} spec={len(feature_names)}")
    return X


def predict_cp3_text(ecg_feat_df: pd.DataFrame) -> pd.DataFrame:
    spec = _load_cp3_text_spec()
    X = _make_cp3_matrix(ecg_feat_df, spec)
    preds = np.zeros(len(ecg_feat_df), dtype=float)
    weight_files = sorted((CP3_TEXT_MODEL_DIR / "fold_models").glob("fold*_weights.npz"))
    for wf in weight_files:
        payload = np.load(wf, allow_pickle=True)
        w = payload["weights"]
        b = float(payload["bias"][0])
        medians = payload["medians"]
        means = payload["means"]
        stds = payload["stds"]
        X2 = X.copy()
        start = int(spec["n_categorical_expanded"])
        if start < X2.shape[1]:
            num = X2[:, start:]
            inds = np.where(np.isnan(num))
            if len(inds[0]) > 0:
                num[inds] = medians[inds[1]]
            num = (num - means) / stds
            X2[:, start:] = num
        preds += _sigmoid(X2 @ w + b)
    preds /= max(len(weight_files), 1)
    return pd.DataFrame({"ID": ecg_feat_df["ID"].astype(str), "prob": preds, "label": ecg_feat_df["AAS"].astype(int)})


def predict_lgbm_stage(cp_csv: Path, stage_key: str) -> pd.DataFrame:
    import lightgbm as lgb

    df = pd.read_csv(cp_csv)
    df["ID"] = df["ID"].astype(str)
    y = df["AAS"].astype(int).values
    X = df.drop(columns=["ID", "AAS"])
    X = drop_leakage_cols(X)
    X = encode_llm_string_columns(X)
    X = add_missing_indicators(X)

    model_files = sorted((LGBM_MODEL_DIR / stage_key / "fold_models").glob("fold*_booster.txt"))
    if not model_files:
        raise FileNotFoundError(f"No fold models found for {stage_key}")

    preds = np.zeros(len(df), dtype=float)
    for mf in model_files:
        booster = lgb.Booster(model_file=str(mf))
        feat_names = booster.feature_name()
        X_use = X.copy()
        for col in feat_names:
            if col not in X_use.columns:
                X_use[col] = np.nan
        X_use = X_use[feat_names]
        preds += booster.predict(X_use.values)
    preds /= len(model_files)
    return pd.DataFrame({"ID": df["ID"], "prob": preds, "label": y})


def _stage_metric_df(score_tbl: pd.DataFrame, score_col: str, threshold: float = 0.5) -> dict:
    y_true = score_tbl["label"].astype(int)
    scores = score_tbl[score_col].astype(float)
    y_pred = (scores >= threshold).astype(int)
    out = _metrics(y_true, y_pred, scores)
    out["stage"] = score_col
    out["threshold"] = threshold
    return out


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run_baseline(score_tbl: pd.DataFrame, policy: dict, llm: LLMClient, mode: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    traces = []
    total = len(score_tbl)
    _log(f"[baseline:{mode}] start n={total} llm_live={llm.is_live}")
    for idx, (_, r) in enumerate(score_tbl.iterrows(), start=1):
        pid = str(r["ID"])
        label = int(r["label"])
        scores = {"CP1": float(r["CP1"]), "CP2": float(r["CP2"]), "CP3": float(r["CP3"]), "CP4": float(r["CP4"])}
        if mode == "multi_agent":
            res = _run_multi_agent_fast(pid, label, scores, policy) if not llm.is_live else run_pathway_for_patient(pid, label, scores, policy, llm)
        elif mode == "single_agent":
            res = _run_single_agent_fast(pid, label, scores, policy) if not llm.is_live else run_single_agent_for_patient(pid, label, scores, policy, llm)
        elif mode == "canonical":
            res = run_canonical_for_patient(pid, label, scores, policy)
        else:
            raise ValueError(mode)
        rows.append(
            {
                "ID": res.patient_id,
                "label": res.label,
                "visited_stages": res.visited_stages,
                "final_stage": res.final_stage,
                "final_score": res.final_score,
                "final_action": res.final_action,
                "final_pred": res.final_pred,
            }
        )
        traces.extend(res.trace)
        if idx == 1 or idx % 10 == 0 or idx == total:
            _log(f"[baseline:{mode}] progress {idx}/{total} last_patient={pid}")

    final_df = pd.DataFrame(rows)
    final_df["error_group"] = final_df.apply(
        lambda x: "TP"
        if x["label"] == 1 and x["final_pred"] == 1
        else "TN"
        if x["label"] == 0 and x["final_pred"] == 0
        else "FP"
        if x["label"] == 0 and x["final_pred"] == 1
        else "FN",
        axis=1,
    )
    metrics = _metrics(final_df["label"], final_df["final_pred"], final_df["final_score"])
    metrics.update(_path_metrics(final_df))
    trace_df = pd.DataFrame(traces)
    _log(f"[baseline:{mode}] done auroc={metrics.get('AUROC')} auprc={metrics.get('AUPRC')}")
    return final_df, metrics, trace_df


def _run_multi_agent_fast(patient_id: str, label: int, scores: dict[str, float], policy: dict) -> object:
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]
    current = "CP1"
    visited, trace, specialists, coordinators = [], [], [], []

    while current is not None:
        score = float(scores.get(current, 0.5))
        c_thr = float(c_thrs.get(current, 0.5))
        a_thr = float(a_thrs.get(current, 0.5))
        from llm_tool_multi_agent.quantitative_tools import risk_level

        rl_tool = risk_level(score, c_thr, a_thr)
        allowed_actions = allowed_actions_for_stage(current)
        specialist_history = [{"stage": x["stage"], "output": x["output"]} for x in specialists]
        coordinator_history = [{"stage": x["stage"], "output": x["output"]} for x in coordinators]
        specialist_summary = summarize_specialist_history(specialist_history)
        coordinator_summary = summarize_coordinator_history(coordinator_history)

        current_stage_specialists = []
        for role_key in STAGE_SPECIALISTS[current]:
            spec_out = _stub_specialist(role_key, current, score, rl_tool)
            current_specialist = {"stage": current, "output": spec_out}
            specialists.append(current_specialist)
            current_stage_specialists.append(current_specialist)

        coord_user = {
            "patient_id": patient_id,
            "label": label,
            "current_stage": current,
            "current_quantitative_state": {
                "risk_score": score,
                "risk_level": rl_tool,
                "continue_threshold": c_thr,
                "action_threshold": a_thr,
            },
            "allowed_actions": allowed_actions,
            "current_specialist_outputs": [x["output"] for x in current_stage_specialists],
            "specialist_summary": specialist_summary,
            "specialist_history": specialist_history,
            "coordinator_summary": coordinator_summary,
            "coordinator_history": coordinator_history,
        }
        coord_raw = _stub_coordinator(current, current_stage_specialists, coord_user)
        coordinators.append({"stage": current, "output": coord_raw})
        dec = validate_coordinator_proposal(current, score, policy, coord_raw)
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "label": label,
                "stage": current,
                "agent": "coordinator",
                "specialist_agents": [AGENT_NAMES[x["output"]["agent_role"]] for x in current_stage_specialists],
                "risk_score": score,
                "risk_level": rl_tool,
                "evidence_boundary": current,
                "allowed_actions": allowed_actions,
                "specialist_summary": specialist_summary,
                "coordinator_summary": coordinator_summary,
                "specialist_jsons": [x["output"] for x in current_stage_specialists],
                "coordinator_json": coord_raw,
                "consensus_state": coord_raw.get("consensus_state"),
                "key_conflicts": coord_raw.get("key_conflicts"),
                "information_gap": coord_raw.get("information_gap"),
                "coordinator_proposed_action": dec.coordinator_action,
                "canonical_action": dec.canonical_action,
                "canonical_next_stage": dec.canonical_next_stage,
                "is_action_allowed": dec.is_action_allowed,
                "final_action": dec.final_action,
                "final_next_stage": dec.final_next_stage,
                "override_reason": dec.override_reason,
                "safety_risk_level": dec.safety_risk_level,
                "policy_basis": dec.policy_basis,
            }
        )
        nxt = dec.final_next_stage
        if nxt is None:
            final_action = dec.final_action
            final_pred = 1 if final_action in POSITIVE_ACTIONS else 0
            from llm_tool_multi_agent.pathway_engine import PathwayResult

            return PathwayResult(
                patient_id=patient_id,
                label=label,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=final_action,
                final_pred=final_pred,
                trace=trace,
                specialist_outputs=specialists,
                coordinator_outputs=coordinators,
            )
        resolved_next = action_to_next_stage(dec.final_action)
        if resolved_next is None:
            raise RuntimeError(f"Non-terminal action without next stage mapping: {dec.final_action}")
        current = resolved_next
    raise RuntimeError("pathway fell through without terminal")


def _run_single_agent_fast(patient_id: str, label: int, scores: dict[str, float], policy: dict) -> object:
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]
    current = "CP1"
    visited, trace, controllers = [], [], []

    while current is not None:
        score = float(scores.get(current, 0.5))
        c_thr = float(c_thrs.get(current, 0.5))
        a_thr = float(a_thrs.get(current, 0.5))
        from llm_tool_multi_agent.quantitative_tools import risk_level

        rl_tool = risk_level(score, c_thr, a_thr)
        allowed_actions = allowed_actions_for_stage(current)
        ctrl_out = _stub_single_agent(
            current,
            {
                "current_quantitative_state": {
                    "risk_score": score,
                    "risk_level": rl_tool,
                    "continue_threshold": c_thr,
                    "action_threshold": a_thr,
                }
            },
        )
        controllers.append({"stage": current, "output": ctrl_out})
        dec = validate_coordinator_proposal(current, score, policy, ctrl_out)
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "label": label,
                "stage": current,
                "agent": "single_agent_controller",
                "risk_score": score,
                "risk_level": rl_tool,
                "allowed_actions": allowed_actions,
                "evidence_boundary": current,
                "controller_json": ctrl_out,
                "consensus_state": ctrl_out.get("consensus_state"),
                "key_conflicts": ctrl_out.get("key_conflicts"),
                "information_gap": ctrl_out.get("information_gap"),
                "controller_proposed_action": dec.coordinator_action,
                "canonical_action": dec.canonical_action,
                "canonical_next_stage": dec.canonical_next_stage,
                "is_action_allowed": dec.is_action_allowed,
                "final_action": dec.final_action,
                "final_next_stage": dec.final_next_stage,
                "override_reason": dec.override_reason,
                "safety_risk_level": dec.safety_risk_level,
                "policy_basis": dec.policy_basis,
            }
        )
        nxt = dec.final_next_stage
        if nxt is None:
            final_action = dec.final_action
            final_pred = 1 if final_action in POSITIVE_ACTIONS else 0
            from llm_tool_multi_agent.single_agent_engine import SingleAgentResult

            return SingleAgentResult(
                patient_id=patient_id,
                label=label,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=final_action,
                final_pred=final_pred,
                trace=trace,
                controller_outputs=controllers,
            )
        resolved_next = action_to_next_stage(dec.final_action)
        if resolved_next is None:
            raise RuntimeError(f"Non-terminal action without next stage mapping: {dec.final_action}")
        current = resolved_next
    raise RuntimeError("single-agent pathway fell through without terminal")


def write_report(
    n: int,
    n_pos: int,
    stage_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    llm_live: bool,
) -> None:
    metric_cols = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", "MCC"]
    lines = [
        "# 湘雅医院第三中心外部验证结果",
        "",
        f"- 输入文件：`{XIANGYA_FINAL_CSV}`",
        f"- 样本量：`{n}`",
        f"- AAS 阳性：`{n_pos}`",
        f"- AAS 阴性：`{n - n_pos}`",
        f"- ECG 分支：`文本版 ECG诊断结论（不使用原始波形）`",
        f"- LLM 模式：`{'live' if llm_live else 'stub'}`",
        "",
        "## 单阶段定量分支",
        "",
        stage_metrics.reindex(columns=["stage"] + metric_cols).to_string(index=False),
        "",
        "## 路径级对比",
        "",
        baseline_metrics.reindex(
            columns=["baseline"] + metric_cols + ["MeanStagesVisited", "CP3UtilizationRate", "CP4UtilizationRate", "PositiveActionRate"]
        ).to_string(index=False),
        "",
        "## 说明",
        "",
        "- CP1 / CP2 / CP4 使用 datasetA 训练好的 LightGBM fold ensemble 在湘雅外部队列上零样本推理。",
        "- CP3 使用已训练的 ECG 文本模型（text-only route）进行外推，不使用 ECG 原始波形。",
        "- 动态 multi-agent、single-agent 和 canonical baseline 共享同一套 policy thresholds。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT_ROOT, CP_DIR, FEATURE_DIR, SCORE_DIR, REPORT_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max patients to run (0=all)")
    ap.add_argument("--offset", type=int, default=0, help="row offset before applying limit")
    ap.add_argument("--run-name", type=str, default="", help="optional output subdirectory name")
    ap.add_argument(
        "--baselines",
        type=str,
        default="multi_agent,single_agent,canonical",
        help="comma-separated baseline modes to run",
    )
    args = ap.parse_args()

    configure_live_env_from_qwen()
    if args.run_name:
        OUT_ROOT = OUTPUT_DIR / args.run_name
        CP_DIR = OUT_ROOT / "cp_inputs"
        FEATURE_DIR = OUT_ROOT / "features"
        SCORE_DIR = OUT_ROOT / "scores"
        REPORT_PATH = OUT_ROOT / "xiangya_external_validation_report_zh.md"
    _log("[main] configured LLM environment from QWEN_* variables")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"[main] output root: {OUT_ROOT}")

    _log(f"[main] loading final dataset: {XIANGYA_FINAL_CSV}")
    final_df = pd.read_csv(XIANGYA_FINAL_CSV)
    final_df["ID"] = final_df["ID"].astype(str).str.strip()
    final_df["AAS"] = final_df["AAS"].astype(int)
    if args.offset and args.offset > 0:
        final_df = final_df.iloc[args.offset:].copy()
    if args.limit and args.limit > 0:
        final_df = final_df.head(args.limit).copy()
    baseline_modes = [x.strip() for x in args.baselines.split(",") if x.strip()]
    allowed_modes = {"multi_agent", "single_agent", "canonical"}
    if not baseline_modes or any(x not in allowed_modes for x in baseline_modes):
        raise ValueError(f"Invalid --baselines: {args.baselines}")
    _log(f"[main] loaded n={len(final_df)} pos={int(final_df['AAS'].sum())} neg={int(len(final_df) - final_df['AAS'].sum())}")

    _log("[main] building CP1/CP2/CP2E input tables")
    cp1, cp2, cp2e = build_cp_tables(final_df)
    cp1_path = CP_DIR / "dataset_CP1_demo_history_exam.csv"
    cp2_path = CP_DIR / "dataset_CP2_demo_history_exam_lab.csv"
    cp2e_path = CP_DIR / "dataset_CP2E_demo_history_exam_lab_echo.csv"
    cp1.to_csv(cp1_path, index=False)
    cp2.to_csv(cp2_path, index=False)
    cp2e.to_csv(cp2e_path, index=False)
    _log("[main] wrote CP input tables")

    _log("[main] building ECG text features")
    ecg_text_csv, ecg_feat_df = build_ecg_text_features(final_df)
    ecg_text_path = FEATURE_DIR / "ecg_raw_text_from_xml_cleaned.csv"
    ecg_meas_path = FEATURE_DIR / "ecg_measurements_from_xml_cleaned.csv"
    ecg_text_csv.to_csv(ecg_text_path, index=False)
    pd.DataFrame({"ID": final_df["ID"]}).to_csv(ecg_meas_path, index=False)
    _log("[main] wrote ECG text feature inputs")

    _log("[main] predicting CP3 text-only probabilities")
    cp3_text = predict_cp3_text(ecg_feat_df)
    cp3_text.to_csv(SCORE_DIR / "holdout_CP3_text_probs.csv", index=False)
    _log("[main] wrote CP3 text-only probabilities")

    import llm_tool_multi_agent.config as cfg
    import llm_tool_multi_agent.quantitative_tools as qt
    import llm_tool_multi_agent.evidence_views as ev

    orig_cp_src = qt.CP_SRC_DIR
    orig_ecg_text = cfg.ECG_TEXT_CSV
    orig_ecg_meas = cfg.ECG_MEASUREMENTS_CSV
    qt.CP_SRC_DIR = CP_DIR
    ev._load_ecg_text_map.cache_clear()
    ev._load_ecg_measurement_map.cache_clear()
    cfg.ECG_TEXT_CSV = ecg_text_path
    cfg.ECG_MEASUREMENTS_CSV = ecg_meas_path
    ev.ECG_TEXT_CSV = ecg_text_path
    ev.ECG_MEASUREMENTS_CSV = ecg_meas_path
    try:
        _log("[main] predicting CP1 probabilities")
        d1 = predict_lgbm_stage(cp1_path, "CP1").rename(columns={"prob": "CP1"})
        _log("[main] predicting CP2 probabilities")
        d2 = predict_lgbm_stage(cp2_path, "CP2").rename(columns={"prob": "CP2"})
        _log("[main] predicting CP4 probabilities")
        d4 = predict_lgbm_stage(cp2e_path, "CP2E").rename(columns={"prob": "CP4"})
        score_tbl = d1[["ID", "label", "CP1"]].merge(d2[["ID", "CP2"]], on="ID", how="left")
        score_tbl = score_tbl.merge(cp3_text.rename(columns={"prob": "CP3"})[["ID", "CP3"]], on="ID", how="left")
        score_tbl = score_tbl.merge(d4[["ID", "CP4"]], on="ID", how="left")
        score_tbl = score_tbl.fillna({"CP1": 0.5, "CP2": 0.5, "CP3": 0.5, "CP4": 0.5})
        score_tbl.to_csv(SCORE_DIR / "xiangya_score_table_cp3_text.csv", index=False)
        d1.to_csv(SCORE_DIR / "holdout_CP1_probs.csv", index=False)
        d2.to_csv(SCORE_DIR / "holdout_CP2_probs.csv", index=False)
        d4.to_csv(SCORE_DIR / "holdout_CP2E_probs.csv", index=False)
        _log("[main] wrote stage score tables")

        stage_metrics = pd.DataFrame([_stage_metric_df(score_tbl, stage) for stage in ["CP1", "CP2", "CP3", "CP4"]])
        stage_metrics.to_csv(OUT_ROOT / "single_stage_metrics.csv", index=False)
        _log("[main] wrote single-stage metrics")

        policy = load_policy()
        llm = LLMClient()
        _log(f"[main] policy loaded; llm_live={llm.is_live}")
        baseline_rows = []
        for mode in baseline_modes:
            final_outcomes, metrics, trace_df = _run_baseline(score_tbl, policy, llm, mode)
            subdir = OUT_ROOT / mode
            subdir.mkdir(parents=True, exist_ok=True)
            final_outcomes.to_csv(subdir / "pathway_final_outcomes.csv", index=False)
            trace_df.to_csv(subdir / "pathway_decision_trace.csv", index=False)
            pd.DataFrame(
                final_outcomes["visited_stages"].value_counts().rename_axis("path").reset_index(name="count")
            ).to_csv(subdir / "pathway_path_summary.csv", index=False)
            (subdir / "pathway_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            metrics["baseline"] = mode
            baseline_rows.append(metrics)
            _log(f"[main] wrote outputs for baseline={mode}")

        baseline_metrics = pd.DataFrame(baseline_rows)
        baseline_metrics.to_csv(OUT_ROOT / "baseline_metrics.csv", index=False)
        _log("[main] wrote baseline metrics")

        write_report(
            n=len(score_tbl),
            n_pos=int(score_tbl["label"].sum()),
            stage_metrics=stage_metrics,
            baseline_metrics=baseline_metrics,
            llm_live=llm.is_live,
        )
        _log(f"[main] wrote report: {REPORT_PATH}")

        summary = {
            "n": int(len(score_tbl)),
            "n_pos": int(score_tbl["label"].sum()),
            "n_neg": int(len(score_tbl) - score_tbl["label"].sum()),
            "llm_live": llm.is_live,
            "outputs": str(OUT_ROOT),
            "offset": int(args.offset),
            "limit": int(args.limit),
            "baselines": baseline_modes,
        }
        (OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        qt.CP_SRC_DIR = orig_cp_src
        cfg.ECG_TEXT_CSV = orig_ecg_text
        cfg.ECG_MEASUREMENTS_CSV = orig_ecg_meas
        ev.ECG_TEXT_CSV = orig_ecg_text
        ev.ECG_MEASUREMENTS_CSV = orig_ecg_meas
        ev._load_ecg_text_map.cache_clear()
        ev._load_ecg_measurement_map.cache_clear()
        _log("[main] restored module globals and cleared caches")


if __name__ == "__main__":
    main()
