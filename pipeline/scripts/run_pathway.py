#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行工具增强 LLM 多智能体路径。

在 **本仓库根目录**（即包含 ``data/``、``artifacts/``、``src/`` 的 ``llm_tool_multi_agent`` 文件夹）下执行::

  python scripts/run_pathway.py --cohort cohort_D --limit 50
  python scripts/run_pathway.py --cohort cohort_V1

环境变量 ``LLM_API_KEY`` / ``LLM_API_BASE`` / ``LLM_MODEL`` 见 README。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tool_multi_agent.config import (
    COHORT_D_IDS,
    COHORT_V1_IDS,
    OOF_SCORE_TABLE,
    OOF_SCORE_TABLE_CP3_TEXT,
    OUTPUT_DIR,
    POLICY_JSON,
)
from llm_tool_multi_agent.llm_client import LLMClient
from llm_tool_multi_agent.pathway_engine import PathwayResult, run_pathway_for_patient
from llm_tool_multi_agent.quantitative_tools import build_score_table_for_ids, load_policy


def _roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auprc_manual(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = y_true.astype(int)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for p, r, is_pos in zip(precision, recall, y_sorted == 1):
        if is_pos:
            ap += float(p) * float(r - prev_recall)
            prev_recall = float(r)
    return float(ap)


def _ece(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(scores)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(scores[mask].mean())
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def _metrics(y_true, y_pred, scores):
    y_true = y_true.astype(int).values
    y_pred = y_pred.astype(int).values
    scores = scores.astype(float).values
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    brier = float(np.mean((scores - y_true) ** 2)) if len(scores) else 0.0
    out = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": accuracy,
        "FPR": fpr,
        "FNR": fnr,
        "F1": f1,
        "BalancedAccuracy": balanced_accuracy,
        "MCC": mcc,
        "Prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "BrierScore": brier,
        "ECE": _ece(y_true, scores) if len(scores) else 0.0,
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(y_true.tolist())) == 2:
            out["AUROC"] = float(roc_auc_score(y_true, scores))
            out["AUPRC"] = float(average_precision_score(y_true, scores))
    except ImportError:
        out["AUROC"] = _roc_auc_manual(y_true, scores)
        out["AUPRC"] = _auprc_manual(y_true, scores)
    return out


def _count_stages(path_str: str) -> int:
    if not isinstance(path_str, str) or not path_str.strip():
        return 0
    return len([x for x in path_str.split("->") if x.strip()])


def _path_metrics(final_df: pd.DataFrame) -> dict[str, float | int]:
    df = final_df.copy()
    df["n_stages_visited"] = df["visited_stages"].apply(_count_stages)
    out: dict[str, float | int] = {
        "MeanStagesVisited": float(df["n_stages_visited"].mean()) if len(df) else 0.0,
        "MedianStagesVisited": float(df["n_stages_visited"].median()) if len(df) else 0.0,
        "MeanStagesVisited_Positive": float(df.loc[df["label"] == 1, "n_stages_visited"].mean()) if (df["label"] == 1).any() else 0.0,
        "MeanStagesVisited_Negative": float(df.loc[df["label"] == 0, "n_stages_visited"].mean()) if (df["label"] == 0).any() else 0.0,
        "EarlyStopRate_CP1": float((df["final_stage"] == "CP1").mean()) if len(df) else 0.0,
        "EarlyStopRate_CP2": float((df["final_stage"] == "CP2").mean()) if len(df) else 0.0,
        "CP3UtilizationRate": float(df["visited_stages"].fillna("").astype(str).str.contains("CP3").mean()) if len(df) else 0.0,
        "CP4UtilizationRate": float(df["visited_stages"].fillna("").astype(str).str.contains("CP4").mean()) if len(df) else 0.0,
        "DirectCTARate": float((df["final_action"] == "direct_cta").mean()) if len(df) else 0.0,
        "UrgentTransferRate": float((df["final_action"] == "urgent_transfer").mean()) if len(df) else 0.0,
        "ObserveRate": float((df["final_action"] == "observe_or_reassess").mean()) if len(df) else 0.0,
        "PositiveActionRate": float(df["final_action"].isin({"direct_cta", "urgent_transfer"}).mean()) if len(df) else 0.0,
        "UniquePathCount": int(df["visited_stages"].nunique()) if len(df) else 0,
    }
    return out


def load_ids(cohort: str, limit: int | None) -> pd.DataFrame:
    if cohort == "cohort_D":
        df = pd.read_csv(COHORT_D_IDS)
    elif cohort == "cohort_V1":
        df = pd.read_csv(COHORT_V1_IDS)
    else:
        raise ValueError(cohort)
    df["ID"] = df["ID"].astype(str)
    if limit:
        df = df.head(limit).copy()
    return df


def load_score_table(cohort: str, ids: list[str], cp3_source: str = "cnn") -> pd.DataFrame:
    if cohort == "cohort_D":
        score_path = OOF_SCORE_TABLE_CP3_TEXT if cp3_source == "text" and OOF_SCORE_TABLE_CP3_TEXT.exists() else OOF_SCORE_TABLE
        st = pd.read_csv(score_path)
        st["ID"] = st["ID"].astype(str)
        st = st[st["ID"].isin(set(ids))].copy()
        return st
    if cohort == "cohort_V1":
        try:
            if cp3_source == "text":
                from llm_tool_multi_agent.precomputed_cohort_v1 import load_cohort_v1_precomputed_score_table

                return load_cohort_v1_precomputed_score_table(ids, cp3_source="text")
            return build_score_table_for_ids(ids)
        except (ImportError, FileNotFoundError, OSError):
            from llm_tool_multi_agent.precomputed_cohort_v1 import load_cohort_v1_precomputed_score_table

            return load_cohort_v1_precomputed_score_table(ids, cp3_source=cp3_source)
    raise ValueError(cohort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["cohort_D", "cohort_V1"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="max patients (0=all)")
    ap.add_argument("--out", type=str, default="", help="output directory override")
    ap.add_argument("--cp3-source", choices=["cnn", "text"], default="text", help="source of CP3 risk score")
    args = ap.parse_args()
    limit = args.limit if args.limit > 0 else None

    ids_df = load_ids(args.cohort, limit)
    ids = ids_df["ID"].astype(str).tolist()
    score_tbl = load_score_table(args.cohort, ids, cp3_source=args.cp3_source)
    if score_tbl.empty:
        print("No score rows; check data/cp CSVs, artifacts/models, and ids.", file=sys.stderr)
        sys.exit(1)

    policy = load_policy()
    llm = LLMClient()
    out_root = Path(args.out) if args.out else OUTPUT_DIR / args.cohort
    out_root.mkdir(parents=True, exist_ok=True)
    audit_dir = out_root / "audit_cases"
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows: list[PathwayResult] = []
    traces: list[dict] = []

    for _, r in score_tbl.iterrows():
        pid = str(r["ID"])
        label = int(r["label"])
        scores = {"CP1": float(r["CP1"]), "CP2": float(r["CP2"]), "CP3": float(r["CP3"]), "CP4": float(r["CP4"])}
        res = run_pathway_for_patient(pid, label, scores, policy, llm)
        rows.append(res)
        for t in res.trace:
            traces.append(t)
        audit_payload = {
            "patient_id": res.patient_id,
            "label": res.label,
            "visited_stages": res.visited_stages,
            "final_stage": res.final_stage,
            "final_score": res.final_score,
            "final_action": res.final_action,
            "final_pred": res.final_pred,
            "specialist_outputs": res.specialist_outputs,
            "coordinator_outputs": res.coordinator_outputs,
            "trace": res.trace,
        }
        (audit_dir / f"{pid}.json").write_text(
            json.dumps(audit_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    final_df = pd.DataFrame(
        [
            {
                "ID": r.patient_id,
                "label": r.label,
                "visited_stages": r.visited_stages,
                "final_stage": r.final_stage,
                "final_score": r.final_score,
                "final_action": r.final_action,
                "final_pred": r.final_pred,
            }
            for r in rows
        ]
    )
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
    final_df.to_csv(out_root / "pathway_final_outcomes.csv", index=False)
    pd.DataFrame(traces).to_csv(out_root / "pathway_decision_trace.csv", index=False)

    m = _metrics(final_df["label"], final_df["final_pred"], final_df["final_score"])
    m.update(_path_metrics(final_df))
    (out_root / "pathway_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    path_sum = final_df["visited_stages"].value_counts().rename_axis("path").reset_index(name="count")
    path_sum.to_csv(out_root / "pathway_path_summary.csv", index=False)

    if args.cohort == "cohort_D":
        score_src = "oof_score_table_cp3_text" if args.cp3_source == "text" else "oof_score_table"
    else:
        score_src = "holdout_precomputed_cp3_text" if args.cp3_source == "text" else "fold_ensemble_zero_shot"
    meta = {
        "cohort": args.cohort,
        "n": len(final_df),
        "llm_live": llm.is_live,
        "llm_model": llm.model,
        "policy": str(POLICY_JSON),
        "score_source": score_src,
        "project_root": str(PROJECT_ROOT),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(m, indent=2))
    print(f"Wrote: {out_root}")


if __name__ == "__main__":
    main()
