"""Build the 2026-05-30 figure set with the merged external V2 cohort.

This wrapper reuses the 2026-05-29 figure grammar, but treats the two retained
Xiangya Hospital extracts as one non-overlapping external validation cohort
named Cohort V2 at the result level. Residual-safety panels use full-cohort
error counts and structured fields available for every retained patient.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

import build_figures_0529 as base


ROOT = Path(os.environ.get("AAS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
FIGURE_SET = os.environ.get("AAS_FIGURE_SET", "figures_0530")
OUT = ROOT / "paper_figures" / FIGURE_SET
AUDIT_OUT = ROOT / "paper_figures" / f"{FIGURE_SET}_audit"
SOURCE_OUT = ROOT / "paper_figures" / "figures_0529"
FROZEN_ID_FILE = os.environ.get("AAS_COHORT_V2_FROZEN_IDS")

MERGED_KEY = "V2"
MERGED_COHORTS = ["V2", "V3"]
ORIG_PREDS: dict[str, pd.DataFrame] = {}
ORIGINAL_COHORTS = list(base.COHORTS)
ORIGINAL_READ_DATA = base.read_data
ORIGINAL_RETAINED_IDS = base.retained_ids
ORIGINAL_CURRENT_FEATURES = base.current_features
ORIGINAL_FINAL_ACTIONS = base.final_actions
ORIGINAL_TRACE_ROWS = base.trace_rows
ORIGINAL_SCORE_FRAME = base.score_frame


def configure_base() -> None:
    base.OUT = OUT
    base.AUDIT_OUT = AUDIT_OUT
    base.COHORTS = [
        base.CohortSpec("D", "Cohort D", "cohort_d_datasetA_raw_exploratory", "Xiangya Second Hospital"),
        base.CohortSpec("V1", "Cohort V1", "cohort_v1_b_v6_raw_exploratory", "Changsha Central Hospital"),
        base.CohortSpec(MERGED_KEY, "Cohort V2", "merged_v2", "Xiangya Hospital"),
    ]
    base.COHORT_COLORS = {
        "D": "#4E79A7",
        "V1": "#59A89C",
        MERGED_KEY: "#D96C5F",
    }


def _with_prefixed_id(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    out["source_cohort"] = cohort
    out["ID"] = cohort + "::" + out["ID"].astype(str)
    return out


def _metric_from_predictions(
    cohort: str,
    method: str,
    pred_df: pd.DataFrame,
    score_keep_ids: set[str] | None = None,
) -> dict[str, float | int | str]:
    pred_col = f"{method}_pred"
    y = pred_df["label"].astype(int).to_numpy()
    yhat = pred_df[pred_col].astype(int).to_numpy()
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    n = int(len(pred_df))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / max(n, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_denom) if mcc_denom else 0.0
    pe = (((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)) if n else 0.0
    kappa = (acc - pe) / (1 - pe) if pe != 1 else 0.0

    score_parts = []
    for source in MERGED_COHORTS:
        sf = ORIGINAL_SCORE_FRAME(source, method, ORIG_PREDS)
        score_parts.append(_with_prefixed_id(sf, source)[["ID", "label", "final_score"]])
    scores = pd.concat(score_parts, ignore_index=True)
    if score_keep_ids is not None:
        scores = scores[scores["ID"].astype(str).isin(score_keep_ids)].copy()
    score = scores["final_score"].astype(float).to_numpy()
    y_score = scores["label"].astype(int).to_numpy()

    return {
        "method": method,
        "n": n,
        "prevalence": float((tp + fn) / max(n, 1)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sens": sens,
        "Spec": spec,
        "PPV": ppv,
        "NPV": npv,
        "Acc": acc,
        "BalAcc": (sens + spec) / 2,
        "F1": f1,
        "MCC": float(mcc),
        "Kappa": float(kappa),
        "AUROC": base.auc_np(*base.roc_curve_np(y_score, score)),
        "AUPRC": base.average_precision_np(y_score, score),
        "cohort": cohort,
        "cohort_label": "Cohort V2",
        "method_label": base.METHOD_LABEL[method],
    }


def read_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    global ORIG_PREDS
    current_cohorts = base.COHORTS
    base.COHORTS = ORIGINAL_COHORTS
    old_metrics, old_preds = ORIGINAL_READ_DATA()
    base.COHORTS = current_cohorts
    ORIG_PREDS = old_preds

    metrics = old_metrics[old_metrics["cohort"].isin(["D", "V1"])].copy()
    merged_preds = pd.concat([
        _with_prefixed_id(old_preds[c], c) for c in MERGED_COHORTS
    ], ignore_index=True)
    if FROZEN_ID_FILE:
        frozen = pd.read_csv(Path(FROZEN_ID_FILE))
        frozen["ID"] = frozen["ID"].astype(str)
        expected = set(merged_preds["ID"].astype(str))
        missing = sorted(set(frozen["ID"]) - expected)
        if missing:
            raise ValueError(f"Frozen Cohort V2 ID file contains {len(missing)} IDs not present in retained predictions")
        merged_preds = frozen.merge(
            merged_preds.drop(columns=[c for c in frozen.columns if c != "ID"], errors="ignore"),
            on="ID",
            how="left",
        )
    keep_ids = set(merged_preds["ID"].astype(str))
    merged_rows = [_metric_from_predictions(MERGED_KEY, m, merged_preds, keep_ids) for m in base.METHODS]
    metrics = pd.concat([metrics, pd.DataFrame(merged_rows)], ignore_index=True)

    preds = {
        "D": old_preds["D"].copy(),
        "V1": old_preds["V1"].copy(),
        MERGED_KEY: merged_preds.copy(),
    }
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    preds[MERGED_KEY].to_csv(AUDIT_OUT / f"{FIGURE_SET}_cohort_v2_retained_predictions.csv", index=False)
    return metrics, preds


def retained_ids(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.Series:
    if cohort == MERGED_KEY:
        return preds[cohort]["ID"].astype(str)
    return ORIGINAL_RETAINED_IDS(cohort, preds)


def current_features(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cohort != MERGED_KEY:
        return ORIGINAL_CURRENT_FEATURES(cohort, preds)
    frames = []
    for source in MERGED_COHORTS:
        frame = ORIGINAL_CURRENT_FEATURES(source, ORIG_PREDS)
        frame = _with_prefixed_id(frame, source)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    keep_ids = set(preds[MERGED_KEY]["ID"].astype(str))
    return out[out["ID"].astype(str).isin(keep_ids)].copy()


def final_actions(cohort: str, method: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cohort != MERGED_KEY:
        return ORIGINAL_FINAL_ACTIONS(cohort, method, preds)
    frames = []
    for source in MERGED_COHORTS:
        frame = ORIGINAL_FINAL_ACTIONS(source, method, ORIG_PREDS)
        frames.append(_with_prefixed_id(frame, source))
    out = pd.concat(frames, ignore_index=True)
    keep_ids = set(preds[MERGED_KEY]["ID"].astype(str))
    return out[out["ID"].astype(str).isin(keep_ids)].copy()


def trace_rows(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cohort != MERGED_KEY:
        return ORIGINAL_TRACE_ROWS(cohort, preds)
    frames = []
    for source in MERGED_COHORTS:
        frame = ORIGINAL_TRACE_ROWS(source, ORIG_PREDS)
        frames.append(_with_prefixed_id(frame, source))
    out = pd.concat(frames, ignore_index=True)
    keep_ids = set(preds[MERGED_KEY]["ID"].astype(str))
    return out[out["ID"].astype(str).isin(keep_ids)].copy()


def score_frame(cohort: str, method: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cohort != MERGED_KEY:
        return ORIGINAL_SCORE_FRAME(cohort, method, preds)
    frames = []
    for source in MERGED_COHORTS:
        frame = ORIGINAL_SCORE_FRAME(source, method, ORIG_PREDS)
        frames.append(_with_prefixed_id(frame, source))
    out = pd.concat(frames, ignore_index=True)
    keep_ids = set(preds[MERGED_KEY]["ID"].astype(str))
    return out[out["ID"].astype(str).isin(keep_ids)].copy()


def fig_study_design(metrics: pd.DataFrame) -> None:
    summary = base.current_summary(metrics).set_index("cohort")
    fig, ax = plt.subplots(figsize=(9.6, 6.7))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.025)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ink = "#111111"
    muted = "#4F5B63"
    line = "#7D858C"
    pale_blue = "#EEF6FA"
    pale_green = "#EFF8F2"
    pale_coral = "#FBF1ED"
    pale_grey = "#F7F7F7"
    header = "#EECBAD"
    risk = "#4E79A7"
    agents = "#009E73"
    guard = "#D96C5F"
    observe = "#8A96A0"

    def rounded(x, y, w, h, fc, ec=line, lw=0.8, radius=0.012, alpha=1.0, z=1, ls="-"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, linestyle=ls,
        )
        ax.add_patch(patch)
        return patch

    def panel(x, y, w, h, label, title=None):
        rounded(x, y, w, h, "#FFFFFF", ec="#555555", lw=1.0, radius=0.022, ls=(0, (6, 3)))
        ax.text(x - 0.008, y + h - 0.010, label, ha="right", va="top",
                fontsize=15, fontweight="bold", color=ink)
        if title:
            ax.add_patch(Rectangle((x + 0.016, y + h - 0.055), w - 0.032, 0.035,
                                   fc=header, ec="none", zorder=2))
            ax.text(x + w / 2, y + h - 0.038, title, ha="center", va="center",
                    fontsize=9.2, fontweight="bold", color=ink, zorder=3)

    def arrow(x1, y1, x2, y2, color=muted, lw=1.0, style="-|>"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0, shrinkB=0),
            zorder=4,
        )

    def small_box(x, y, w, h, title, subtitle="", fc="#FFFFFF", ec=line, color=ink, fs=7.0):
        rounded(x, y, w, h, fc, ec=ec, lw=0.8, radius=0.010)
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                    fontsize=fs - 1.0, color=muted, linespacing=0.95)

    def stack_icon(x, y, color):
        for dx, dy in [(0.010, 0.010), (0.005, 0.005), (0, 0)]:
            ax.add_patch(Rectangle((x + dx, y + dy), 0.030, 0.038,
                                   fc="#FFFFFF", ec=color, lw=0.8, zorder=3))
        ax.plot([x + 0.006, x + 0.024], [y + 0.026, y + 0.026], color=color, lw=0.8)
        ax.plot([x + 0.006, x + 0.021], [y + 0.016, y + 0.016], color=color, lw=0.8)

    def shield_icon(cx, cy, color):
        xs = [cx, cx + 0.022, cx + 0.018, cx, cx - 0.018, cx - 0.022]
        ys = [cy + 0.025, cy + 0.013, cy - 0.012, cy - 0.027, cy - 0.012, cy + 0.013]
        ax.fill(xs, ys, color=color, alpha=0.18, ec=color, lw=1.0, zorder=3)
        ax.plot([cx, cx], [cy - 0.012, cy + 0.012], color=color, lw=0.9, zorder=4)

    # A. Whole-system overview.
    panel(0.020, 0.705, 0.960, 0.270, "a", "Safety-governed multi-agent pathway control for suspected acute aortic syndrome")
    x_nodes = [0.085, 0.250, 0.425, 0.615, 0.795]
    overview = [
        ("Stage-bounded\nclinical evidence", "history, labs,\nECG, echo", pale_blue, risk),
        ("Quantitative\nrisk substrate", "score + risk band\nat each checkpoint", "#F3F5F7", risk),
        ("Role-restricted\nLLM specialists", "whitelisted views\nstructured JSON", pale_green, agents),
        ("Deterministic\nsafety governance", "legal actions\nguardrails", pale_coral, guard),
        ("Terminal\naction", "observe/reassess\nCTA or transfer", "#FFFFFF", ink),
    ]
    for i, (x, (title, sub, fc, ec)) in enumerate(zip(x_nodes, overview)):
        small_box(x - 0.060, 0.790, 0.120, 0.100, title, sub, fc=fc, ec=ec, color=ec, fs=7.2)
        if i == 0:
            stack_icon(x - 0.017, 0.727, risk)
        elif i == 1:
            ax.plot([x - 0.030, x - 0.018, x - 0.006, x + 0.006, x + 0.018, x + 0.030],
                    [0.740, 0.747, 0.735, 0.756, 0.728, 0.744], color=risk, lw=1.2)
        elif i == 2:
            for j in range(5):
                ax.add_patch(Circle((x - 0.030 + j * 0.015, 0.744 + (j % 2) * 0.010),
                                    0.010, fc=agents, ec="white", lw=0.4, alpha=0.85))
        elif i == 3:
            shield_icon(x, 0.744, guard)
        else:
            for j, (lab, col) in enumerate([("O", observe), ("C", risk), ("T", guard)]):
                ax.add_patch(Circle((x - 0.030 + j * 0.030, 0.744), 0.014, fc=col, ec="none"))
                ax.text(x - 0.030 + j * 0.030, 0.744, lab, ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")
        if i < len(x_nodes) - 1:
            arrow(x + 0.066, 0.840, x_nodes[i + 1] - 0.066, 0.840, color=ink, lw=0.9)
    ax.text(0.500, 0.720,
            "The LLM layer proposes actions only within a frozen policy; final authority remains with deterministic governance.",
            ha="center", va="center", fontsize=7.0, color=muted)

    # B. Frozen development.
    panel(0.020, 0.390, 0.455, 0.285, "b", "Development and freezing on Cohort D")
    dev_steps = [
        ("Cohort D", f"n={int(summary.loc['D', 'n']):,}\nAAS+ {summary.loc['D', 'prevalence']:.1%}", base.COHORT_COLORS["D"]),
        ("Stage models", "LightGBM\nCP1/2/4\nLR ECG concepts", risk),
        ("Prompt system", "specialist roles\ncoordinator\nJSON schema", agents),
        ("Policy freeze", "thresholds\nlegal actions\nfallbacks", guard),
    ]
    for i, (title, sub, col) in enumerate(dev_steps):
        x = 0.055 + i * 0.100
        small_box(x, 0.500, 0.080, 0.105, title, sub, fc="#FFFFFF", ec=col, color=col, fs=7.1)
        if i < len(dev_steps) - 1:
            arrow(x + 0.083, 0.552, x + 0.097, 0.552, color=ink, lw=0.9)
    ax.add_patch(Rectangle((0.060, 0.430), 0.360, 0.035, fc="#F0F0F0", ec=line, lw=0.7))
    ax.text(0.240, 0.448, "No external-cohort exposure before release", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=ink)
    ax.plot([0.060, 0.420], [0.415, 0.415], color=guard, lw=1.1)
    ax.text(0.240, 0.405, "risk models + prompts + thresholds + safety rules frozen", ha="center",
            va="center", fontsize=6.3, color=muted)

    # C. Runtime controller.
    panel(0.505, 0.390, 0.475, 0.285, "c", "Runtime checkpoint controller")
    cps = [("CP1", "history\nexam"), ("CP2", "labs"), ("CP3", "ECG\ncontext"), ("CP4", "echo")]
    xs = [0.585, 0.675, 0.765, 0.855]
    for i, (x, (cp, sub)) in enumerate(zip(xs, cps)):
        small_box(x - 0.034, 0.585, 0.068, 0.055, cp, sub, fc="#FFFFFF", ec=line, fs=6.8)
        for y, col, fc in [(0.535, risk, pale_blue), (0.485, agents, pale_green), (0.435, guard, pale_coral)]:
            rounded(x - 0.030, y - 0.020, 0.060, 0.036, fc, ec=col, lw=0.8, radius=0.008)
            ax.add_patch(Circle((x, y - 0.002), 0.008, fc=col, ec="white", lw=0.3, zorder=4))
        if i < 3:
            arrow(x + 0.036, 0.612, xs[i + 1] - 0.038, 0.612, color=ink, lw=0.8)
            arrow(x + 0.033, 0.485, xs[i + 1] - 0.035, 0.485, color=line, lw=0.8)
    layer_labels = [("risk", risk, 0.535), ("agent", agents, 0.485), ("gate", guard, 0.435)]
    for lab, col, y in layer_labels:
        ax.text(0.522, y, lab, ha="left", va="center", fontsize=6.0, fontweight="bold", color=col)
    arrow(0.890, 0.485, 0.935, 0.485, color=ink, lw=0.9)
    small_box(0.925, 0.425, 0.040, 0.125, "Action", "observe\nCTA\ntransfer", fc="#FFFFFF", ec=ink, fs=6.4)

    # D. Validation and action-derived evaluation.
    panel(0.020, 0.050, 0.960, 0.315, "d", "No-refit external validation and action-derived endpoints")
    cohort_rows = [
        ("Cohort D", "Development", "Xiangya Second Hospital", "D", base.COHORT_COLORS["D"], 0.065),
        ("Cohort V1", "No-refit external", "Changsha Central Hospital", "V1", base.COHORT_COLORS["V1"], 0.265),
        ("Cohort V2", "No-refit external", "Xiangya Hospital", "V2", base.COHORT_COLORS[MERGED_KEY], 0.465),
    ]
    for name, role, site, key, col, x in cohort_rows:
        rounded(x, 0.190, 0.155, 0.095, "#FFFFFF", ec=col, lw=0.9, radius=0.010)
        ax.add_patch(Circle((x + 0.028, 0.238), 0.020, fc=col, ec="none"))
        ax.text(x + 0.028, 0.238, key, ha="center", va="center", fontsize=7.0,
                color="white", fontweight="bold")
        ax.text(x + 0.060, 0.257, name, ha="left", va="center", fontsize=7.0, fontweight="bold")
        ax.text(x + 0.060, 0.237, role, ha="left", va="center", fontsize=6.0, color=muted)
        ax.text(x + 0.060, 0.216, f"n={int(summary.loc[key if key != 'V2' else MERGED_KEY, 'n']):,}",
                ha="left", va="center", fontsize=6.2, color=muted)
        ax.text(x + 0.078, 0.175, site, ha="center", va="top", fontsize=5.8, color=muted)
    arrow(0.225, 0.238, 0.265, 0.238, color=ink, lw=0.9)
    arrow(0.425, 0.238, 0.465, 0.238, color=ink, lw=0.9)
    metrics_box_x = 0.665
    small_box(metrics_box_x, 0.210, 0.115, 0.080, "Primary endpoint", "terminal CTA /\nurgent escalation", fc="#FFFFFF", ec=ink, fs=6.7)
    arrow(metrics_box_x + 0.120, 0.250, metrics_box_x + 0.160, 0.250, color=ink, lw=0.9)
    small_box(metrics_box_x + 0.165, 0.210, 0.105, 0.080, "Action metrics", "sensitivity\nspecificity, F1", fc="#FFFFFF", ec=ink, fs=6.7)
    rounded(0.690, 0.090, 0.225, 0.075, "#F5F8FA", ec="#C8D2D8", lw=0.8, radius=0.010)
    ax.text(0.803, 0.139, "Cohort V2 headline", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=ink)
    ax.text(0.803, 0.112,
            "sensitivity 0.944  |  specificity 0.724  |  F1 0.706",
            ha="center", va="center", fontsize=6.2, color=muted)
    base.save(fig, "F1_study_design_redesigned")


def fig_operational_burden(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    rows = []
    for method in base.METHODS:
        final = final_actions(MERGED_KEY, method, preds)
        y = final["label"].to_numpy()
        escalate = final[f"{method}_pred"].astype(int).to_numpy() == 1
        observe = ~escalate
        trace = trace_rows(MERGED_KEY, preds)
        rows.append({
            "method": method,
            "Positive escalation": int(escalate.sum()),
            "False-positive assigned escalation": int(((y == 0) & escalate).sum()),
            "Observe/reassess\n(clinician oversight)": int(observe.sum()),
            "Mean visited checkpoints": float(trace.groupby("ID").size().mean()),
        })
    df = pd.DataFrame(rows).set_index("method").loc[base.METHODS].reset_index()
    method_labels = [base.METHOD_LABEL[m] for m in df["method"]]

    fig = plt.figure(figsize=(9.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.90], wspace=0.36,
                          left=0.10, right=0.97, bottom=0.24, top=0.76)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    burden_metrics = [
        ("Positive escalation", "#5B8CC0"),
        ("False-positive assigned escalation", "#D77A72"),
        ("Observe/reassess\n(clinician oversight)", base.GREY),
    ]
    ybase = np.arange(len(df))[::-1]
    offsets = np.array([0.24, 0.0, -0.24])
    for mi, (metric, color) in enumerate(burden_metrics):
        vals = df[metric].to_numpy()
        ypos = ybase + offsets[mi]
        ax_a.barh(ypos, vals, height=0.20, color=color, edgecolor="white", linewidth=0.6,
                  label=metric, alpha=0.95)
        for yv, val in zip(ypos, vals):
            ax_a.text(val + 220, yv, f"{val:,}", va="center", ha="left", fontsize=7.5, color=base.INK)
    ax_a.set_yticks(ybase)
    ax_a.set_yticklabels(method_labels)
    ax_a.set_xlabel("Patients in Cohort V2")
    ax_a.set_title("Terminal action burden")
    ax_a.set_xlim(0, max(df["Observe/reassess\n(clinician oversight)"].max(), df["Positive escalation"].max()) * 1.18)
    base.panel(ax_a, "a")

    x = np.arange(len(df))
    ax_b.bar(x, df["Mean visited checkpoints"], color=[base.METHOD_COLORS[m] for m in df["method"]],
             edgecolor="white", linewidth=0.6, width=0.62)
    for xi, val in zip(x, df["Mean visited checkpoints"]):
        ax_b.text(xi, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8.0, color=base.INK)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Canonical", "Single\nagent", "Multi\nagent"])
    ax_b.set_ylabel("Mean visited checkpoints")
    ax_b.set_ylim(0, 3.05)
    ax_b.set_title("Pathway length")
    base.panel(ax_b, "b")
    fig.legend(handles=[Patch(facecolor=color, edgecolor="white", label=metric) for metric, color in burden_metrics],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035), fontsize=8.2)
    fig.suptitle("Assigned escalation burden in Cohort V2", y=0.965, fontsize=11, color=base.INK)
    base.save(fig, "F_S6_operational_burden_redesigned")


def _merged_feature_predictions(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    features = current_features(MERGED_KEY, preds)
    if "label" not in features.columns and "AAS" in features.columns:
        features["label"] = features["AAS"].astype(int)
    pred = preds[MERGED_KEY].copy()
    df = pred.merge(features, on=["ID", "label"], how="left")
    df["age_bin"] = pd.cut(
        df["Age"],
        bins=[0, 39, 54, 64, 74, 200],
        labels=["<40", "40-54", "55-64", "65-74", ">=75"],
        include_lowest=True,
    ).astype(str)
    pain_cols = [
        "history__sudden_onset_pain",
        "history__severe_pain",
        "history__tearing_pain",
        "history__migrating_pain",
    ]
    df["classic_pain"] = df[pain_cols].fillna(0).astype(float).max(axis=1).astype(int)
    df["prior_aortic_disease"] = df["history__aortic_disease_history"].fillna(0).astype(float).astype(int)
    df["any_exam_signal"] = df[[
        "exam__neurologic_deficit",
        "exam__hypotension_or_shock",
        "exam__pulse_deficit",
        "exam__new_aortic_regurgitation_murmur",
    ]].fillna(0).astype(float).max(axis=1).astype(int)
    return df


def fig_fn_safety(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    full = _merged_feature_predictions(preds)
    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.15], hspace=0.48, wspace=0.58,
                          top=0.86, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    rows = []
    for method in base.METHODS:
        r = metrics[(metrics["cohort"] == MERGED_KEY) & (metrics["method"] == method)].iloc[0]
        rows.append({
            "method": method,
            "False positives": int(r["FP"]),
            "False negatives": int(r["FN"]),
            "Sensitivity": float(r["Sens"]),
            "Specificity": float(r["Spec"]),
            "PPV": float(r["PPV"]),
            "NPV": float(r["NPV"]),
        })
    err = pd.DataFrame(rows)
    x = np.arange(len(err))
    width = 0.36
    ax_a.bar(x - width / 2, err["False positives"], width=width, color="#D77A72", label="False positives")
    ax_a.bar(x + width / 2, err["False negatives"], width=width, color=base.WINE, label="False negatives")
    for xi, fp, fn in zip(x, err["False positives"], err["False negatives"]):
        ax_a.text(xi - width / 2, fp + 90, f"{fp:,}", ha="center", va="bottom", fontsize=7.2)
        ax_a.text(xi + width / 2, fn + 90, f"{fn:,}", ha="center", va="bottom", fontsize=7.2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([base.METHOD_LABEL[m] for m in err["method"]], rotation=15, ha="right")
    ax_a.set_ylabel("Patients")
    ax_a.set_title(f"Residual error counts in Cohort V2 (n={len(full):,})")
    ax_a.legend(loc="upper right", fontsize=6.8)

    multi = metrics[(metrics["cohort"] == MERGED_KEY) & (metrics["method"] == "multi_raw")].iloc[0]
    outcome_rows = [
        ("AAS+ assigned escalation", int(multi["TP"]), base.GREEN),
        ("AAS+ assigned obs/reassess", int(multi["FN"]), base.WINE),
        ("AAS- assigned obs/reassess", int(multi["TN"]), base.BLUE),
        ("AAS- assigned escalation", int(multi["FP"]), "#D77A72"),
    ]
    y = np.arange(len(outcome_rows))[::-1]
    maxv = max(v for _, v, _ in outcome_rows)
    for yi, (label, val, color) in zip(y, outcome_rows):
        ax_b.barh(yi, val, color=color, edgecolor="white", height=0.55)
        ax_b.text(val + maxv * 0.025, yi, f"{val:,}", va="center", fontsize=7.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([label for label, _, _ in outcome_rows])
    ax_b.set_xlim(0, maxv * 1.20)
    ax_b.set_xlabel("Patients")
    ax_b.set_title("Multi-agent terminal-action outcome")

    positives = full[full["label"].astype(int) == 1].copy()
    positives["multi_error"] = np.where(positives["multi_raw_pred"].astype(int).eq(0), "FN", "TP")
    subgroup_specs = [
        ("Age", "age_bin", ["<40", "40-54", "55-64", "65-74", ">=75"]),
        ("Pain profile", "classic_pain", [0, 1]),
        ("History", "prior_aortic_disease", [0, 1]),
        ("Exam", "any_exam_signal", [0, 1]),
    ]
    label_map = {
        ("Pain profile", 0): "no documented classic pain",
        ("Pain profile", 1): "documented classic pain",
        ("History", 0): "no documented prior aortic disease",
        ("History", 1): "documented prior aortic disease",
        ("Exam", 0): "no high-yield exam signal",
        ("Exam", 1): "any high-yield exam signal",
    }
    domain_colors = {"Age": base.GREY, "Pain profile": base.WINE, "History": base.GOLD, "Exam": base.CYAN}
    subgroup_rows = []
    for domain, col, cats in subgroup_specs:
        for cat in cats:
            sub = positives[positives[col].astype(str) == str(cat)]
            if len(sub):
                subgroup_rows.append({
                    "domain": domain,
                    "label": label_map.get((domain, cat), str(cat)),
                    "rate": (sub["multi_error"] == "FN").mean() * 100,
                    "n": len(sub),
                    "fn": int((sub["multi_error"] == "FN").sum()),
                })
    top = pd.DataFrame(subgroup_rows).sort_values("rate", ascending=False).head(10).sort_values("rate")
    yy = np.arange(len(top))
    colors = [domain_colors[d] for d in top["domain"]]
    ax_c.hlines(yy, 0, top["rate"], color=colors, lw=3.2, alpha=0.80)
    ax_c.plot(top["rate"], yy, "o", ms=5.0, color=base.WINE, mec=base.INK, mew=0.25)
    for yv, (_, r) in zip(yy, top.iterrows()):
        ax_c.text(r["rate"] + 0.45, yv, f"{r['rate']:.1f}% ({int(r['fn'])}/{int(r['n'])})",
                  va="center", fontsize=6.7, color=base.INK)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels(top["label"], fontsize=6.9)
    ax_c.set_xlim(0, max(top["rate"].max() * 1.35, 10))
    ax_c.set_xlabel("False-negative rate among AAS-positive patients in subgroup (%)")
    ax_c.set_title("Full-cohort residual false-negative concentration")
    ax_c.grid(axis="x", color=base.GRID)
    ax_c.grid(axis="y", visible=False)
    ax_c.legend(handles=[Patch(color=color, label=domain) for domain, color in domain_colors.items()],
                loc="lower right", fontsize=6.4, frameon=False, ncol=2)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        base.panel(ax, lab)
    base.save(fig, "F6_fn_safety_redesigned")


def fig_fn_safety_supplement(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    full = _merged_feature_predictions(preds)
    positives = full[full["label"].astype(int) == 1].copy()
    positives["multi_error"] = np.where(positives["multi_raw_pred"].astype(int).eq(0), "FN", "TP")

    fig = plt.figure(figsize=(9.4, 4.9))
    gs = fig.add_gridspec(1, 2, wspace=0.34, left=0.12, right=0.96, bottom=0.19, top=0.84)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    age_order = ["<40", "40-54", "55-64", "65-74", ">=75"]
    age_tab = positives.groupby(["age_bin", "multi_error"], observed=False).size().unstack(fill_value=0).reindex(age_order).fillna(0)
    fn_rate = age_tab.get("FN", 0) / age_tab.sum(axis=1).replace(0, np.nan) * 100
    age_plot = pd.DataFrame({
        "age": age_order,
        "rate": fn_rate.to_numpy(dtype=float),
        "fn": age_tab.get("FN", 0).to_numpy(dtype=int),
        "n": age_tab.sum(axis=1).to_numpy(dtype=int),
    }).iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(age_plot))
    ax_a.hlines(yy, 0, age_plot["rate"], color=base.GREY, lw=3.2, alpha=0.85)
    ax_a.plot(age_plot["rate"], yy, "o", ms=5.2, color=base.WINE, mec=base.INK, mew=0.25)
    for yv, (_, r) in zip(yy, age_plot.iterrows()):
        val = r["rate"]
        if not np.isnan(val):
            ax_a.text(val + 0.45, yv, f"{val:.1f}% ({int(r['fn'])}/{int(r['n'])})",
                      va="center", fontsize=6.9, color=base.INK)
    ax_a.set_yticks(yy)
    ax_a.set_yticklabels(age_plot["age"])
    ax_a.set_xlim(0, max(age_plot["rate"].max() * 1.35, 18))
    ax_a.set_xlabel("False-negative rate among AAS-positive patients (%)")
    ax_a.set_title("Age-stratified residual miss rate")
    ax_a.grid(axis="x", color=base.GRID)
    ax_a.grid(axis="y", visible=False)

    variables = [
        ("No documented classic pain", 1 - positives["classic_pain"]),
        ("Documented prior aortic disease", positives["prior_aortic_disease"]),
        ("High-yield exam signal", positives["any_exam_signal"]),
    ]
    vals_fn, vals_tp = [], []
    for _, series in variables:
        vals_fn.append(series[positives["multi_error"] == "FN"].mean() * 100)
        vals_tp.append(series[positives["multi_error"] == "TP"].mean() * 100)
    contrast = pd.DataFrame({
        "feature": [name for name, _ in variables],
        "FN": vals_fn,
        "TP": vals_tp,
    }).iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(contrast))
    for yv, (_, r) in zip(yy, contrast.iterrows()):
        ax_b.hlines(yv, min(r["FN"], r["TP"]), max(r["FN"], r["TP"]),
                    color=base.MUTED, lw=1.2, alpha=0.45)
        ax_b.plot(r["TP"], yv, "o", ms=5.2, color=base.CYAN, mec="white", mew=0.5)
        ax_b.plot(r["FN"], yv, "o", ms=5.8, color=base.WINE, mec="white", mew=0.5)
        fn_x = min(r["FN"] + 2.0, 101.5)
        tp_x = min(r["TP"] + 2.0, 101.5)
        ax_b.text(fn_x, yv + 0.14, f"FN {r['FN']:.0f}%", color=base.WINE,
                  fontsize=6.9, ha="left", va="bottom")
        ax_b.text(tp_x, yv - 0.14, f"TP {r['TP']:.0f}%", color=base.CYAN,
                  fontsize=6.9, ha="left", va="top")
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(contrast["feature"])
    ax_b.set_xlim(0, 108)
    ax_b.set_ylim(-0.35, len(contrast) - 0.65)
    ax_b.set_xlabel("Prevalence among AAS-positive patients (%)")
    ax_b.set_title("FN versus TP feature profile")
    ax_b.grid(axis="x", color=base.GRID)
    ax_b.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], marker="o", color=base.CYAN, linestyle="None", markersize=5.2, label="True positive"),
        Line2D([], [], marker="o", color=base.WINE, linestyle="None", markersize=5.8, label="False negative"),
    ]
    ax_b.legend(handles=handles, loc="lower right")

    for ax, lab in zip([ax_a, ax_b], "ab"):
        base.panel(ax, lab)
    base.save(fig, "F6_fn_safety_supplement_redesigned")


def fig_action_confusion(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    actions = ["observe_or_reassess", "direct_cta", "urgent_transfer"]
    canon = final_actions(MERGED_KEY, "canonical", preds).set_index("ID")["final_action_current"].astype(str)
    multi = final_actions(MERGED_KEY, "multi_raw", preds).set_index("ID")["final_action_current"].astype(str)
    common = canon.index.intersection(multi.index)
    mat = np.zeros((3, 3), dtype=int)
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            mat[i, j] = int(((canon.loc[common] == a) & (multi.loc[common] == b)).sum())
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    agree = np.trace(mat) / mat.sum()

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{mat[i, j]:,}\n{norm[i, j] * 100:.0f}%",
                ha="center", va="center", fontsize=8.4,
                color="white" if norm[i, j] > 0.52 else base.INK,
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels([base.ACTION_LABELS[a] for a in actions], rotation=28, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels([base.ACTION_LABELS[a] for a in actions])
    ax.set_xlabel("Multi-agent terminal action")
    ax.set_ylabel("Canonical terminal action")
    ax.set_title(f"Cohort V2 terminal-action rerouting (agreement {agree * 100:.1f}%)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised share")
    base.save(fig, "F10_action_confusion_redesigned")


def copy_static_shap() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in [
        "F7_shap_CP1_redesigned",
        "F7_shap_CP2_redesigned",
        "F7_shap_CP3_text_redesigned",
        "F7_shap_CP4_redesigned",
    ]:
        for ext in ("png", "pdf"):
            shutil.copy2(SOURCE_OUT / f"{stem}.{ext}", OUT / f"{stem}.{ext}")


def write_audit(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    summary = base.current_summary(metrics)
    audit_suffix = FIGURE_SET.replace("figures_", "")
    metrics.to_csv(AUDIT_OUT / f"metrics_{audit_suffix}.csv", index=False)
    summary.to_csv(AUDIT_OUT / f"cohort_summary_{audit_suffix}.csv", index=False)

    def md_table(df: pd.DataFrame) -> str:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda x: f"{x:.4f}")
        headers = [str(c) for c in out.columns]
        rows = [[str(v) for v in row] for row in out.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    overlap = set(ORIG_PREDS["V2"]["ID"].astype(str)) & set(ORIG_PREDS["V3"]["ID"].astype(str))
    lines = [
        f"# Figure audit for {FIGURE_SET}",
        "",
        "Inputs: D and V1 retained results plus the merged Cohort V2 retained results.",
        "",
        f"Frozen Cohort V2 ID file: {FROZEN_ID_FILE or 'not used'}.",
        "",
        f"Raw ID overlap between the two merged source extracts before source-prefixing: {len(overlap):,}.",
        "",
        "All cohort sample sizes and denominators use the retained current analysis cohorts.",
        "",
        "Residual-safety panels use the full retained Cohort V2 denominator and structured fields available for every retained patient.",
        "",
        "## Current cohort summary",
        "",
        md_table(summary),
        "",
        "## Headline metrics",
        "",
        md_table(metrics),
        "",
    ]
    (AUDIT_OUT / f"FIGURE_AUDIT_{audit_suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    operational = []
    for method in base.METHODS:
        final = final_actions(MERGED_KEY, method, preds)
        y = final["label"].to_numpy()
        escalate = final[f"{method}_pred"].astype(int).to_numpy() == 1
        operational.append({
            "method": method,
            "positive_escalation": int(escalate.sum()),
            "false_positive_escalation": int(((y == 0) & escalate).sum()),
            "observe_or_reassess": int((~escalate).sum()),
        })
    pd.DataFrame(operational).to_csv(AUDIT_OUT / f"operational_counts_{audit_suffix}.csv", index=False)


def main() -> None:
    configure_base()
    base.style()
    base.retained_ids = retained_ids
    base.current_features = current_features
    base.final_actions = final_actions
    base.trace_rows = trace_rows
    base.score_frame = score_frame
    metrics, preds = read_data()
    fig_study_design(metrics)
    base.fig_baseline(metrics, preds)
    base.fig_discrimination(metrics, preds)
    base.fig_score_quality(metrics, preds)
    base.fig_calibration_decision(metrics, preds)
    base.fig_mechanism(metrics, preds)
    fig_fn_safety(metrics, preds)
    fig_fn_safety_supplement(metrics, preds)
    fig_action_confusion(metrics, preds)
    base.fig_error_tradeoff(metrics)
    fig_operational_burden(metrics, preds)
    base.fig_table(metrics)
    base.fig_tau_audit(metrics, preds)
    base.fig_literature(metrics)
    copy_static_shap()
    base.fig_shap_surrogates(metrics)
    write_audit(metrics, preds)
    print(f"Wrote merged Cohort V2 figures and audit files to {OUT}")


if __name__ == "__main__":
    main()
