"""Build the 2026-05-29 exploratory figure set.

Inputs are the four post-governance exploratory result folders:

- cohort_d_datasetA_raw_exploratory
- cohort_v1_b_v6_raw_exploratory
- cohort_v2_raw_exploratory
- cohort_v3_raw_exploratory

The redraw keeps the same figure names and broad content, but uses the current
post-governance patient set as the only cohort denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(os.environ.get("AAS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(ROOT / "paper_figures"))
from _baseline import load_cohort_features  # noqa: E402


OUT = ROOT / "paper_figures" / "figures_0529"
AUDIT_OUT = ROOT / "paper_figures" / "figures_0529_audit"


@dataclass(frozen=True)
class CohortSpec:
    key: str
    label: str
    folder: str
    source: str


COHORTS = [
    CohortSpec("D", "Cohort D", "cohort_d_datasetA_raw_exploratory", "Xiangya Second Hospital"),
    CohortSpec("V1", "Cohort V1", "cohort_v1_b_v6_raw_exploratory", "Changsha Central Hospital"),
    CohortSpec("V2", "Cohort V2", "cohort_v2_raw_exploratory", "Xiangya Hospital"),
    CohortSpec("V3", "Cohort V3", "cohort_v3_raw_exploratory", "Xiangya Hospital"),
]

COHORT_TO_OLD = {
    "D": "datasetA",
    "V1": "datasetB_v6",
    "V2": "xiangya_720",
    "V3": "xiangya_16218",
}

OLD_TO_COHORT = {v: k for k, v in COHORT_TO_OLD.items()}

METHODS = ["canonical", "single", "multi_raw"]
METHOD_LABEL = {
    "canonical": "Canonical",
    "single": "Single-agent",
    "multi_raw": "Multi-agent",
}
METHOD_COLORS = {
    "canonical": "#607D8B",
    "single": "#C77C3A",
    "multi_raw": "#00897B",
}
COHORT_COLORS = {
    "D": "#4E79A7",
    "V1": "#59A89C",
    "V2": "#E6B85C",
    "V3": "#D96C5F",
}

INK = "#263238"
MUTED = "#6F7F87"
GRID = "#E8EEF2"
RED = "#D96C5F"
GREEN = "#00897B"
BLUE = "#4E79A7"
GOLD = "#E6B85C"
LIGHT = "#F6F8FA"
WINE = "#8C1D40"
CYAN = "#7DC7D8"
GREY = "#9EA7AD"

ACTION_LABELS = {
    "observe_or_reassess": "Observe/reassess\n(clinician oversight)",
    "direct_cta": "Direct CTA",
    "urgent_transfer": "Urgent pathway\nescalation",
}

OLD_RUN_DIRS = {
    "D": ROOT / "shared/multi_agent_data/upstream_runs/datasetA",
    "V1": ROOT / "shared/multi_agent_data/upstream_runs/datasetB_v6",
    "V2": ROOT / "phase1_qwen_720_bundle/outputs/run_qwen_v2",
    "V3": ROOT / "phase1_qwen_16218_bundle/outputs/run_qwen",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, ha="left", va="bottom")


def read_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    preds = {}
    for c in COHORTS:
        base = ROOT / c.folder
        m = pd.read_csv(base / "FINAL_metrics.csv")
        m["cohort"] = c.key
        m["cohort_label"] = c.label
        m["method"] = m["method"].replace({"multi_raw": "multi_raw"})
        m["method_label"] = m["method"].map(METHOD_LABEL)
        rows.append(m)
        p = pd.read_csv(base / "FINAL_retained_predictions.csv")
        preds[c.key] = p
    metrics = pd.concat(rows, ignore_index=True)
    return metrics, preds


def current_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in COHORTS:
        row = metrics[(metrics.cohort == c.key) & (metrics.method == "multi_raw")].iloc[0]
        n = int(row.n)
        pos = int(row.TP + row.FN)
        out.append({
            "cohort": c.key,
            "label": c.label,
            "n": n,
            "positive_n": pos,
            "prevalence": pos / n,
            "source": c.source,
        })
    return pd.DataFrame(out)


def retained_ids(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.Series:
    return preds[cohort]["ID"].astype(str)


def current_features(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    old = COHORT_TO_OLD[cohort]
    df = load_cohort_features(old).copy()
    keep = set(retained_ids(cohort, preds))
    return df[df["ID"].astype(str).isin(keep)].copy()


def final_actions(cohort: str, method: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    old_method = {"canonical": "canonical", "single": "single_agent", "multi_raw": "multi_agent"}[method]
    path = OLD_RUN_DIRS[cohort] / old_method / "pathway_final_outcomes.csv"
    df = pd.read_csv(path)
    keep = set(retained_ids(cohort, preds))
    df = df[df["ID"].astype(str).isin(keep)].copy()
    pred_col = f"{method}_pred"
    bin_df = preds[cohort][["ID", "label", pred_col]].copy()
    df = df.merge(bin_df, on=["ID", "label"], how="inner")
    df["final_pred_current"] = df[pred_col].astype(int)
    escalate = df["final_action"].isin(["direct_cta", "urgent_transfer"])
    df["final_action_current"] = np.where(
        df["final_pred_current"].eq(0),
        "observe_or_reassess",
        np.where(escalate, df["final_action"], "direct_cta"),
    )
    return df


def trace_rows(cohort: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = OLD_RUN_DIRS[cohort] / "multi_agent" / "pathway_decision_trace.csv"
    df = pd.read_csv(path)
    keep = set(retained_ids(cohort, preds))
    return df[df["ID"].astype(str).isin(keep)].copy()


def score_frame(cohort: str, method: str, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    old_method = {"canonical": "canonical", "single": "single_agent", "multi_raw": "multi_agent"}[method]
    path = OLD_RUN_DIRS[cohort] / old_method / "pathway_final_outcomes.csv"
    df = pd.read_csv(path)
    keep = set(retained_ids(cohort, preds))
    return df[df["ID"].astype(str).isin(keep)].copy()


def roc_curve_np(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    s_sorted = score[order]
    distinct = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    idx = np.where(distinct)[0]
    tp = np.cumsum(y_sorted == 1)[idx]
    fp = np.cumsum(y_sorted == 0)[idx]
    pos = max((y == 1).sum(), 1)
    neg = max((y == 0).sum(), 1)
    return np.r_[0, fp / neg, 1], np.r_[0, tp / pos, 1]


def pr_curve_np(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    s_sorted = score[order]
    distinct_end = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    idx = np.where(distinct_end)[0]
    tp = np.cumsum(y_sorted == 1)[idx]
    fp = np.cumsum(y_sorted == 0)[idx]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max((y == 1).sum(), 1)
    return np.r_[0, recall], np.r_[1, precision]


def auc_np(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x_ord = x[order]
    y_ord = y[order]
    return float(np.sum((x_ord[1:] - x_ord[:-1]) * (y_ord[1:] + y_ord[:-1]) / 2))


def average_precision_np(y: np.ndarray, score: np.ndarray) -> float:
    recall, precision = pr_curve_np(y, score)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    return float(np.sum(np.diff(recall) * precision[1:]))


def reliability_bins(y: np.ndarray, score: np.ndarray, n_bins: int = 10):
    edges = np.unique(np.quantile(score, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(score.min(), score.max(), n_bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum():
            xs.append(score[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def ece(y: np.ndarray, score: np.ndarray) -> float:
    x, obs, ns = reliability_bins(y, score)
    return float(np.sum(ns * np.abs(x - obs)) / np.sum(ns))


def dca(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    out = []
    for pt in thresholds:
        yhat = score >= pt
        tp = ((y == 1) & yhat).sum()
        fp = ((y == 0) & yhat).sum()
        out.append(tp / n - (fp / n) * pt / (1 - pt))
    return np.array(out)


def fig_baseline(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    summary = current_summary(metrics)
    fig = plt.figure(figsize=(8.9, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    y = np.arange(len(COHORTS))[::-1]
    n_max = summary["n"].max()
    for i, c in enumerate(COHORTS):
        row = summary[summary.cohort == c.key].iloc[0]
        rel = row.n / n_max
        ax_a.hlines(y[i], 0, rel, color=COHORT_COLORS[c.key], lw=6, alpha=0.75)
        ax_a.plot(rel, y[i], "o", ms=7, color=COHORT_COLORS[c.key], mec=INK, mew=0.4)
        tx = rel + 0.035 if rel < 0.86 else rel - 0.28
        ty = y[i] if rel < 0.86 else y[i] + 0.11
        ax_a.text(tx, ty, f"n={int(row.n):,}; AAS+ {row.prevalence*100:.1f}%",
                  va="center", fontsize=8, ha="left")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{c.key}  {c.source}" for c in COHORTS])
    ax_a.set_xlim(0, 1.18)
    ax_a.set_xticks([0, 0.5, 1.0])
    ax_a.set_xticklabels(["0", "50", "100"])
    ax_a.set_xlabel("Relative cohort size (%)")
    ax_a.set_title("Cohort scale and case mix")

    for c in COHORTS:
        age = pd.to_numeric(current_features(c.key, preds)["Age"], errors="coerce").dropna().to_numpy()
        hist, edges = np.histogram(age, bins=np.linspace(20, 95, 38), density=True)
        smooth = np.convolve(hist, np.array([1, 2, 3, 2, 1]) / 9, mode="same")
        centers = (edges[:-1] + edges[1:]) / 2
        ax_b.plot(centers, smooth, color=COHORT_COLORS[c.key], lw=1.8, label=f"Cohort {c.key}")
    ax_b.set_xlabel("Age (years)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("Age distribution")

    feats = [
        ("history__tearing_pain", "Tearing pain"),
        ("history__sudden_onset_pain", "Sudden-onset pain"),
        ("history__aortic_disease_history", "Prior aortic disease"),
        ("echo__suspected_intimal_flap", "Echo intimal flap"),
    ]
    rows = []
    for col, lab in feats:
        vals = []
        for c in COHORTS:
            df = current_features(c.key, preds)
            s = df.loc[df["AAS"] == 1, col].dropna()
            vals.append(float((s == 1).mean()) if len(s) else np.nan)
        rows.append((lab, vals))
    yy = np.arange(len(rows))[::-1]
    for fi, (_lab, vals) in enumerate(rows):
        ax_c.hlines(yy[fi], np.nanmin(vals), np.nanmax(vals), color=GRID, lw=2.8, zorder=1)
        for v, c in zip(vals, COHORTS):
            if not np.isnan(v):
                ax_c.plot(v, yy[fi], "o", color=COHORT_COLORS[c.key], ms=5.2, mec=INK, mew=0.25, zorder=3)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels([lab for lab, _ in rows])
    ax_c.set_xlim(0, 1)
    ax_c.set_xticks([0, .25, .5, .75, 1])
    ax_c.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_c.set_xlabel("Prevalence among AAS-positive patients (%)")
    ax_c.set_title("High-yield clinical signals")

    labs = [("D_D_log", "D-dimer"), ("Mb_log", "Mb"), ("NT_proBNP_log", "NT-proBNP"), ("CK_MB_log", "CK-MB")]
    xbase = np.arange(len(labs))
    offsets = np.linspace(-0.24, 0.24, len(COHORTS))
    for ci, c in enumerate(COHORTS):
        df = current_features(c.key, preds)
        for li, (col, _lab) in enumerate(labs):
            vals = pd.to_numeric(df.loc[df["AAS"] == 1, col], errors="coerce").dropna()
            if len(vals):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                x = xbase[li] + offsets[ci]
                ax_d.vlines(x, q1, q3, color=COHORT_COLORS[c.key], lw=4, alpha=0.75)
                ax_d.plot(x, med, "o", color=COHORT_COLORS[c.key], mec=INK, mew=0.25, ms=4.5)
    ax_d.set_xticks(xbase)
    ax_d.set_xticklabels([lab for _, lab in labs])
    ax_d.set_ylabel("Log value; median and IQR")
    ax_d.set_title("Laboratory burden in AAS-positive patients")

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        panel(ax, lab)
    handles = [Line2D([], [], marker="o", linestyle="None", color=COHORT_COLORS[c.key],
                      markeredgecolor=INK, markeredgewidth=0.4, label=f"Cohort {c.key}")
               for c in COHORTS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    save(fig, "F2_baseline_redesigned")


def fig_study_design(metrics: pd.DataFrame) -> None:
    summary = current_summary(metrics).set_index("cohort")
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axis("off")
    cohorts = [
        ("D", "Development", COHORT_COLORS["D"]),
        ("V1", "External validation", COHORT_COLORS["V1"]),
        ("V2", "External validation", COHORT_COLORS["V2"]),
        ("V3", "External validation", COHORT_COLORS["V3"]),
    ]
    x0 = 0.06
    for i, (cid, label, color) in enumerate(cohorts):
        y = 0.78 - i * 0.18
        ax.add_patch(Rectangle((x0, y - 0.055), 0.19, 0.11, color=color, alpha=0.16, ec=color, lw=1.0))
        ax.text(x0 + 0.03, y, cid, ha="center", va="center", fontsize=12, fontweight="bold", color=color)
        ax.text(x0 + 0.075, y, f"{label}\nn={int(summary.loc[cid, 'n']):,}", ha="left", va="center",
                fontsize=8.5, color=INK)

    stages = [
        ("CP1", "History + exam"),
        ("CP2", "Laboratory"),
        ("CP3", "ECG concepts"),
        ("CP4", "Echocardiography"),
        ("Action", "Obs/reassess\n(clinician) /\nCTA / transfer"),
    ]
    xs = np.linspace(0.36, 0.88, len(stages))
    for i, ((stage, label), x) in enumerate(zip(stages, xs)):
        w = 0.105 if stage != "Action" else 0.13
        ax.add_patch(Rectangle((x - w / 2, 0.48), w, 0.17, color="#EEF5F5", ec="#8ABBB8", lw=1.0))
        ax.text(x, 0.59, stage, ha="center", va="center", fontsize=10, fontweight="bold", color="#00695C")
        ax.text(x, 0.525, label, ha="center", va="center", fontsize=7.8, color=INK)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.565), xytext=(x + w / 2 + 0.012, 0.565),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))

    ax.annotate("", xy=(0.305, 0.565), xytext=(0.25, 0.565),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(0.36, 0.79, "Digital triage pathway evaluated across development and validation cohorts",
            ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
    save(fig, "F1_study_design_redesigned")


def fig_discrimination(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(9.3, 5.9))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.28)
    for ci, c in enumerate(COHORTS):
        axr = fig.add_subplot(gs[0, ci])
        axp = fig.add_subplot(gs[1, ci])
        axr.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
        prev = current_summary(metrics).set_index("cohort").loc[c.key, "prevalence"]
        axp.axhline(prev, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        for m in METHODS:
            sf = score_frame(c.key, m, preds)
            y = sf["label"].to_numpy()
            s = sf["final_score"].to_numpy()
            fpr, tpr = roc_curve_np(y, s)
            rec, prec = pr_curve_np(y, s)
            axr.plot(fpr, tpr, color=METHOD_COLORS[m], lw=1.5)
            axp.plot(rec, prec, color=METHOD_COLORS[m], lw=1.5)
        axr.set_title(f"Cohort {c.key}")
        axr.set_xlim(0, 1); axr.set_ylim(0, 1.01); axr.set_aspect("equal")
        axp.set_xlim(0, 1); axp.set_ylim(0, 1.01)
        if ci == 0:
            axr.set_ylabel("Sensitivity")
            axp.set_ylabel("Precision")
        else:
            axr.set_yticklabels([])
            axp.set_yticklabels([])
        axp.set_xlabel("Recall")
        axr.set_xlabel("1 - specificity")
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=1.8, label=METHOD_LABEL[m]) for m in METHODS]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="Reference"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.014, 0.94, "a", fontweight="bold", fontsize=10, color=INK)
    fig.text(0.014, 0.47, "b", fontweight="bold", fontsize=10, color=INK)
    save(fig, "F3_discrimination_redesigned")


def fig_score_quality(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    rows = []
    for c in COHORTS:
        for m in METHODS:
            sf = score_frame(c.key, m, preds)
            y = sf["label"].to_numpy()
            s = sf["final_score"].to_numpy()
            rows.append({
                "cohort": c.key,
                "method": m,
                "AUROC": auc_np(*roc_curve_np(y, s)),
                "AP": average_precision_np(y, s),
                "ECE": ece(y, s),
                "Brier": float(np.mean((s - y) ** 2)),
            })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(9.3, 2.8), sharex=True)
    for ax, metric in zip(axes, ["AUROC", "AP", "ECE", "Brier"]):
        for m in METHODS:
            sub = df[df["method"] == m].set_index("cohort").loc[[c.key for c in COHORTS]]
            ax.plot(np.arange(len(COHORTS)), sub[metric], marker="o", lw=1.4,
                    color=METHOD_COLORS[m], label=METHOD_LABEL[m])
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(COHORTS)))
        ax.set_xticklabels([c.key for c in COHORTS])
    axes[0].legend(loc="lower left")
    save(fig, "F4_score_quality_summary_redesigned")


def fig_calibration_decision(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(9.2, 6.0))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    thresholds = np.linspace(0.01, 0.60, 60)
    summary = current_summary(metrics).set_index("cohort")
    for ci, c in enumerate(COHORTS):
        axc = fig.add_subplot(gs[0, ci])
        axd = fig.add_subplot(gs[1, ci])
        axc.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
        prev = summary.loc[c.key, "prevalence"]
        axd.plot(thresholds * 100, prev - (1 - prev) * thresholds / (1 - thresholds),
                 color=MUTED, lw=0.8, ls=(0, (3, 2)))
        axd.axhline(0, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        for m in METHODS:
            sf = score_frame(c.key, m, preds)
            y = sf["label"].to_numpy()
            s = sf["final_score"].to_numpy()
            x, obs, _ = reliability_bins(y, s)
            axc.plot(x, obs, color=METHOD_COLORS[m], marker="o", ms=2.8, lw=1.2)
            axd.plot(thresholds * 100, dca(y, s, thresholds), color=METHOD_COLORS[m], lw=1.4)
        axc.set_title(f"Cohort {c.key}")
        axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_aspect("equal")
        axd.set_xlim(1, 60)
        if ci == 0:
            axc.set_ylabel("Observed event rate")
            axd.set_ylabel("Net benefit")
        else:
            axc.set_yticklabels([])
        axc.set_xlabel("Predicted probability")
        axd.set_xlabel("Threshold probability (%)")
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=1.8, label=METHOD_LABEL[m]) for m in METHODS]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="Reference"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.014, 0.94, "a", fontweight="bold", fontsize=10, color=INK)
    fig.text(0.014, 0.47, "b", fontweight="bold", fontsize=10, color=INK)
    save(fig, "F4_F9_calibration_decision_redesigned")


def _has_conflict(s: pd.Series) -> pd.Series:
    txt = s.fillna("").astype(str)
    return ~txt.isin(["", "[]", "null", "nan", "NaN"])


def fig_mechanism(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(9.2, 6.7))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.36, top=0.92, bottom=0.13)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    stages = ["CP1", "CP2", "CP3", "CP4"]
    stage_x = np.arange(len(stages))
    cohort_offsets = np.linspace(-0.075, 0.075, len(COHORTS))

    for ci, c in enumerate(COHORTS):
        df = trace_rows(c.key, preds)
        rates = df.assign(conflict=_has_conflict(df["key_conflicts"]).astype(int)).groupby("stage")["conflict"].mean().reindex(stages)
        xs = stage_x + cohort_offsets[ci]
        present = rates.notna().to_numpy()
        ax_a.plot(xs[present], rates.to_numpy()[present], marker="o", markersize=4.2,
                  color=COHORT_COLORS[c.key], lw=1.35, label=f"Cohort {c.key}")
    ax_a.set_ylim(0, 1.18)
    ax_a.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_xticks(stage_x)
    ax_a.set_xticklabels(stages)
    ax_a.set_ylabel("Specialist conflict rate")
    ax_a.set_title("Where specialist disagreement enters")
    ax_a.legend(loc="upper left", ncol=2, fontsize=6.8, handlelength=1.4, columnspacing=0.9)

    states = [
        ("convergent_low_risk", "Low-risk consensus", CYAN),
        ("convergent_high_risk", "High-risk consensus", WINE),
        ("mixed_risk", "Mixed risk", "#E68A64"),
        ("unresolved_uncertainty", "Unresolved", GREY),
    ]
    y = np.arange(len(COHORTS))[::-1]
    for i, c in enumerate(COHORTS):
        vc = trace_rows(c.key, preds)["consensus_state"].value_counts(normalize=True)
        left = 0
        for state, _lab, color in states:
            frac = float(vc.get(state, 0))
            ax_b.barh(y[i], frac, left=left, color=color, height=0.62, edgecolor="white", lw=0.5)
            if frac > 0.08:
                ax_b.text(left + frac / 2, y[i], f"{frac*100:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
            left += frac
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"Cohort {c.key}" for c in COHORTS])
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("Share of pathway-stage rows")
    ax_b.set_title("Coordinator consensus state")

    rows = []
    for c in COHORTS:
        df = trace_rows(c.key, preds)
        rows.append((c.key, (df["coordinator_proposed_action"].astype(str) != df["canonical_action"].astype(str)).mean()))
    ax_c.bar([k for k, _ in rows], [r for _, r in rows], color=[COHORT_COLORS[k] for k, _ in rows], edgecolor=INK, lw=0.35)
    for i, (_, r) in enumerate(rows):
        ax_c.text(i, r + 0.02, f"{r*100:.1f}%", ha="center", fontsize=8)
    ax_c.set_ylim(0, 0.85)
    ax_c.set_ylabel("Deviation rate")
    ax_c.set_title("Coordinator override of canonical pathway")

    length_colors = {1: "#D7E8F5", 2: "#8DBFDB", 3: "#3E83B7", 4: "#174A7C"}
    for i, c in enumerate(COHORTS):
        per_patient = trace_rows(c.key, preds).groupby("ID").size()
        left = 0
        for k in [1, 2, 3, 4]:
            frac = float((per_patient == k).mean())
            ax_d.barh(y[i], frac, left=left, color=length_colors[k], height=0.62, edgecolor="white", lw=0.5)
            if frac > 0.07:
                ax_d.text(left + frac / 2, y[i], f"{frac*100:.0f}%", ha="center", va="center",
                          color=INK if k == 1 else "white", fontsize=7.5)
            left += frac
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"Cohort {c.key}" for c in COHORTS])
    ax_d.set_xlim(0, 1)
    ax_d.set_xlabel("Share of patients")
    ax_d.set_title("Adaptive pathway length")

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        panel(ax, lab)
    fig.legend(handles=[Patch(color=color, label=lab) for _, lab, color in states],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015))
    save(fig, "F5_mechanism_redesigned")


def _v3_tier_data(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fn_dir = ROOT / "paper_figures/figures_0529_audit/v3_fn_profile"
    pos = pd.read_csv(fn_dir / "positives_with_features.csv")
    v3 = preds["V3"][["ID", "label", "multi_raw_pred"]].copy()
    v3 = v3[(v3["label"] == 1)]
    pos = pos[pos["ID"].astype(str).isin(set(v3["ID"].astype(str)))].copy()
    pred_map = v3.set_index(v3["ID"].astype(str))["multi_raw_pred"].astype(int).to_dict()
    pos["current_pred"] = pos["ID"].astype(str).map(pred_map)
    pos["error_group"] = np.where(pos["current_pred"] == 1, "TP", "FN")
    pos["tier"] = "unclassified"
    pos.loc[(pos["surgery_history"].isin(["open_replacement", "open_and_stent", "post_op_unspecified"]))
            | (pos["encounter_kind"].isin(["acute_with_prior_history", "follow_up_or_surveillance"])), "tier"] = "T4"
    pos.loc[(pos["tier"] == "unclassified") & (pos["surgery_history"] == "stent_evar_tevar"), "tier"] = "T3"
    pos.loc[(pos["tier"] == "unclassified")
            & ((pos["any_classic_pain"] == 0) | (pos["history_aortic_disease"] == 1)
               | (pos["stanford"] == "unspecified") | (pos["subtype"].isin(["PAU", "IMH", "other_aas"]))), "tier"] = "T2"
    pos.loc[pos["tier"] == "unclassified", "tier"] = "T1"
    return pos


def fig_fn_safety(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    pos = _v3_tier_data(preds)
    fn = pos[pos["error_group"] == "FN"]
    scope_label = globals().get("FN_SCOPE_LABEL", "")
    scope_suffix = f", {scope_label}" if scope_label else ""
    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.48, wspace=0.44,
                          top=0.86, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    tier_order = ["T1", "T2", "T3", "T4"]
    tier_labels = {"T1": "Acute-typical", "T2": "Atypical/chronic", "T3": "Prior stent", "T4": "Post-op/follow-up"}
    tier_colors = {"T1": WINE, "T2": "#C95F78", "T3": GOLD, "T4": CYAN}

    left = 0
    for t in tier_order:
        frac = (fn["tier"] == t).mean() if len(fn) else 0
        ax_a.barh([0], [frac], left=left, color=tier_colors[t], edgecolor="white", height=0.48)
        if frac > 0.08:
            ax_a.text(left + frac / 2, 0, f"{frac*100:.1f}%", ha="center", va="center",
                      color="white" if t in ["T1", "T2"] else INK, fontsize=8)
        elif frac > 0:
            ax_a.text(left + frac + 0.012, 0.31, f"{frac*100:.1f}%", ha="left", va="center", color=INK, fontsize=8)
        left += frac
    ax_a.set_ylim(-0.42, 0.42)
    ax_a.set_xlim(0, 1)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Share of multi-agent false negatives")
    ax_a.set_title(f"FN composition{scope_suffix} (n={len(fn):,})")

    rows = [
        ("Stanford-A\nacute-typical", ((fn["tier"] == "T1") & (fn["stanford"] == "A")).sum()),
        ("All\nacute-typical", (fn["tier"] == "T1").sum()),
        ("Atypical/chronic", (fn["tier"] == "T2").sum()),
        ("Prior stent", (fn["tier"] == "T3").sum()),
        ("Post-op/\nfollow-up", (fn["tier"] == "T4").sum()),
    ]
    y = np.arange(len(rows))[::-1]
    denom = len(pos)
    for i, (lab, n) in enumerate(rows):
        rate = n / max(denom, 1)
        ax_b.barh(y[i], rate * 100, color=WINE if i == 0 else CYAN, edgecolor=INK, lw=0.3)
        ax_b.text(rate * 100 + 0.35, y[i], f"{n:,} ({rate*100:.2f}%)", va="center", fontsize=7.5)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([r[0] for r in rows])
    ax_b.set_xlim(0, max([n / max(denom, 1) for _, n in rows]) * 100 * 1.45)
    ax_b.set_xlabel("Share of all AAS-positive patients (%)")
    ax_b.set_title("Clinical miss severity")

    strata = [
        ("Stanford", "stanford", ["A", "B", "A_and_B", "unspecified"]),
        ("Encounter", "encounter_kind", ["acute_primary", "ambiguous", "acute_with_prior_history", "follow_up_or_surveillance"]),
        ("Prior intervention", "surgery_history", ["none_or_treatment_naive", "stent_evar_tevar", "open_replacement", "open_and_stent", "post_op_unspecified"]),
        ("Age", "age_bin", ["<40", "40-54", "55-64", "65-74", ">=75"]),
    ]
    domain_colors = {
        "Stanford": CYAN,
        "Encounter": WINE,
        "Prior intervention": GOLD,
        "Age": GREY,
    }
    subgroup_rows = []
    for domain, col, cats in strata:
        for cat in cats:
            sub = pos[pos[col].astype(str) == cat]
            if len(sub):
                subgroup_rows.append({
                    "domain": domain,
                    "label": cat.replace("_", " "),
                    "rate": (sub["error_group"] == "FN").mean() * 100,
                    "n": len(sub),
                })
    top = pd.DataFrame(subgroup_rows).sort_values("rate", ascending=False).head(12).sort_values("rate")
    yy = np.arange(len(top))
    colors = [domain_colors[d] for d in top["domain"]]
    ax_c.hlines(yy, 0, top["rate"], color=colors, lw=3.2, alpha=0.78)
    ax_c.plot(top["rate"], yy, "o", ms=5.0, color=WINE, mec=INK, mew=0.25)
    for yv, (_, r) in zip(yy, top.iterrows()):
        ax_c.text(r["rate"] + 0.55, yv, f"{r['rate']:.1f}%", va="center", fontsize=6.7, color=INK)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels(top["label"], fontsize=6.9)
    ax_c.set_xlim(0, max(top["rate"].max() * 1.25, 10))
    ax_c.set_xlabel("False-negative rate within subgroup (%)")
    ax_c.set_title("Subgroup concentration of residual false negatives")
    ax_c.grid(axis="x", color=GRID)
    ax_c.grid(axis="y", visible=False)
    ax_c.legend(handles=[Patch(color=color, label=domain) for domain, color in domain_colors.items()],
                loc="lower right", fontsize=6.4, frameon=False, ncol=2)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        panel(ax, lab)
    fig.legend(handles=[Patch(color=tier_colors[t], label=tier_labels[t]) for t in tier_order],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985))
    save(fig, "F6_fn_safety_redesigned")


def fig_fn_safety_supplement(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    pos = _v3_tier_data(preds)
    scope_label = globals().get("FN_SCOPE_LABEL", "")
    scope_suffix = f" ({scope_label})" if scope_label else ""
    fig = plt.figure(figsize=(9.2, 4.4))
    gs = fig.add_gridspec(1, 2, wspace=0.60, left=0.18, right=0.96, bottom=0.18, top=0.84)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])

    strata = [
        ("stanford", ["A", "B", "A_and_B", "unspecified"]),
        ("encounter_kind", ["acute_primary", "ambiguous", "acute_with_prior_history", "follow_up_or_surveillance"]),
        ("surgery_history", ["none_or_treatment_naive", "stent_evar_tevar", "open_replacement", "open_and_stent", "post_op_unspecified"]),
        ("age_bin", ["<40", "40-54", "55-64", "65-74", ">=75"]),
    ]
    heat, labels = [], []
    for col, cats in strata:
        for cat in cats:
            sub = pos[pos[col].astype(str) == cat]
            if len(sub):
                heat.append([(sub["error_group"] == "FN").mean() * 100])
                labels.append(cat.replace("_", " "))
    im = ax_c.imshow(np.array(heat), cmap=LinearSegmentedColormap.from_list("fn", ["#EAF4F7", CYAN, WINE]), aspect="auto")
    ax_c.set_yticks(np.arange(len(labels)))
    ax_c.set_yticklabels(labels, fontsize=6.8)
    ax_c.set_xticks([0])
    ax_c.set_xticklabels(["FN rate"])
    ax_c.set_title(f"False-negative concentration by subgroup{scope_suffix}")
    fig.colorbar(im, ax=ax_c, fraction=0.05, pad=0.02, label="%")

    axes_cols = ["history_aortic_disease", "any_classic_pain", "echo_aas_signal", "reached_stage"]
    display = ["Prior aortic\ndisease", "Classic pain\nabsent", "Echo\nsignal", "Stage\nreached"]
    vals = []
    for col in axes_cols:
        if col == "any_classic_pain":
            vals.append((pos[col] == 0).groupby(pos["error_group"]).mean().to_dict())
        else:
            vals.append((pos[col].astype(str) != "none").groupby(pos["error_group"]).mean().to_dict())
    x = np.arange(len(axes_cols))
    ax_d.plot(x, [v.get("FN", np.nan) * 100 for v in vals], marker="o", color=WINE, lw=1.6, label="FN")
    ax_d.plot(x, [v.get("TP", np.nan) * 100 for v in vals], marker="o", color=CYAN, lw=1.6, label="TP")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(display, rotation=0, ha="center")
    ax_d.set_ylabel("Prevalence (%)")
    ax_d.set_title("FN vs TP phenotype contrast")
    ax_d.legend(loc="lower right")

    for ax, lab in zip([ax_c, ax_d], "ab"):
        panel(ax, lab)
    save(fig, "F6_fn_safety_supplement_redesigned")


def fig_action_confusion(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    actions = ["observe_or_reassess", "direct_cta", "urgent_transfer"]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.7), sharex=True, sharey=True)
    for ax, c in zip(axes, COHORTS):
        canon = final_actions(c.key, "canonical", preds).set_index("ID")["final_action_current"].astype(str)
        multi = final_actions(c.key, "multi_raw", preds).set_index("ID")["final_action_current"].astype(str)
        common = canon.index.intersection(multi.index)
        mat = np.zeros((3, 3), dtype=int)
        for i, a in enumerate(actions):
            for j, b in enumerate(actions):
                mat[i, j] = int(((canon.loc[common] == a) & (multi.loc[common] == b)).sum())
        norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:,}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                        fontsize=6.7, color="white" if norm[i, j] > 0.52 else INK)
        agree = np.trace(mat) / mat.sum()
        ax.set_title(f"{c.key} agreement {agree*100:.1f}%")
        ax.set_xticks(range(3))
        ax.set_xticklabels([ACTION_LABELS[a] for a in actions], rotation=35, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels([ACTION_LABELS[a] for a in actions])
        ax.grid(False)
    axes[0].set_ylabel("Canonical")
    fig.suptitle("Canonical-to-multi-agent terminal action shifts in current cohorts", y=1.03, color=INK)
    save(fig, "F10_action_confusion_redesigned")


def fig_error_tradeoff(metrics: pd.DataFrame) -> None:
    rows = []
    for c in COHORTS:
        for m in METHODS:
            r = metrics[(metrics["cohort"] == c.key) & (metrics["method"] == m)].iloc[0]
            rows.append({
                "cohort": c.key,
                "method": m,
                "FP rate": r["FP"] / max(r["FP"] + r["TN"], 1),
                "FN rate": r["FN"] / max(r["FN"] + r["TP"], 1),
            })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for m in METHODS:
        sub = df[df["method"] == m]
        ax.plot(sub["FP rate"] * 100, sub["FN rate"] * 100, marker="o", lw=1.4,
                color=METHOD_COLORS[m], label=METHOD_LABEL[m])
        for _, r in sub.iterrows():
            ax.text(r["FP rate"] * 100 + 0.5, r["FN rate"] * 100 + 0.3,
                    r["cohort"], fontsize=7, color=METHOD_COLORS[m])
    ax.set_xlabel("False-positive rate (%)")
    ax.set_ylabel("False-negative rate (%)")
    ax.set_title("Error trade-off across validation cohorts")
    ax.legend(loc="upper left")
    save(fig, "F11_error_tradeoff_redesigned")


def fig_operational_burden(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    cohort = "V3"
    rows = []
    for method in METHODS:
        final = final_actions(cohort, method, preds)
        y = final["label"].to_numpy()
        escalate = final[f"{method}_pred"].astype(int).to_numpy() == 1
        observe = ~escalate
        trace = pd.read_csv(OLD_RUN_DIRS[cohort] / {"canonical": "canonical", "single": "single_agent", "multi_raw": "multi_agent"}[method] / "pathway_decision_trace.csv")
        keep = set(retained_ids(cohort, preds))
        trace = trace[trace["ID"].astype(str).isin(keep)]
        rows.append({
            "method": method,
            "Positive escalation": int(escalate.sum()),
            "False-positive assigned escalation": int(((y == 0) & escalate).sum()),
            "Observe/reassess\n(clinician oversight)": int(observe.sum()),
            "Mean visited checkpoints": float(trace.groupby("ID").size().mean()),
        })
    df = pd.DataFrame(rows).set_index("method").loc[METHODS].reset_index()
    method_labels = [METHOD_LABEL[m] for m in df["method"]]

    fig = plt.figure(figsize=(9.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.90], wspace=0.36,
                          left=0.10, right=0.97, bottom=0.24, top=0.76)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    burden_metrics = [
        ("Positive escalation", "#5B8CC0"),
        ("False-positive assigned escalation", "#D77A72"),
        ("Observe/reassess\n(clinician oversight)", GREY),
    ]
    ybase = np.arange(len(df))[::-1]
    offsets = np.array([0.24, 0.0, -0.24])
    for mi, (metric, color) in enumerate(burden_metrics):
        vals = df[metric].to_numpy()
        ypos = ybase + offsets[mi]
        ax_a.barh(ypos, vals, height=0.20, color=color, edgecolor="white", linewidth=0.6,
                  label=metric, alpha=0.95)
        for yv, val in zip(ypos, vals):
            ax_a.text(val + 180, yv, f"{val:,}", va="center", ha="left", fontsize=7.5, color=INK)
    ax_a.set_yticks(ybase)
    ax_a.set_yticklabels(method_labels)
    ax_a.set_xlabel("Patients in Cohort V3")
    ax_a.set_title("Terminal action burden")
    ax_a.set_xlim(0, max(df["Observe/reassess\n(clinician oversight)"].max(), df["Positive escalation"].max()) * 1.18)
    panel(ax_a, "a")

    x = np.arange(len(df))
    ax_b.bar(x, df["Mean visited checkpoints"], color=[METHOD_COLORS[m] for m in df["method"]],
             edgecolor="white", linewidth=0.6, width=0.62)
    for xi, val in zip(x, df["Mean visited checkpoints"]):
        ax_b.text(xi, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8.0, color=INK)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Canonical", "Single\nagent", "Multi\nagent"])
    ax_b.set_ylabel("Mean visited checkpoints")
    ax_b.set_ylim(0, 3.05)
    ax_b.set_title("Pathway length")
    panel(ax_b, "b")
    fig.legend(handles=[Patch(facecolor=color, edgecolor="white", label=metric) for metric, color in burden_metrics],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035), fontsize=8.2)
    fig.suptitle("Assigned escalation burden in Cohort V3", y=0.965, fontsize=11, color=INK)
    save(fig, "F_S6_operational_burden_redesigned")


def fig_table(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.8), sharey=True)
    metric_defs = [("Sens", "Sensitivity"), ("Spec", "Specificity"), ("PPV", "PPV")]
    cmap = LinearSegmentedColormap.from_list("metric", ["#F3F7F9", "#A8D7D5", "#00897B"])
    for ax, (metric, title) in zip(axes, metric_defs):
        mat = np.zeros((len(COHORTS), len(METHODS)))
        txt = [["" for _ in METHODS] for _ in COHORTS]
        for i, c in enumerate(COHORTS):
            for j, m in enumerate(METHODS):
                r = metrics[(metrics["cohort"] == c.key) & (metrics["method"] == m)].iloc[0]
                mat[i, j] = r[metric]
                txt[i][j] = f"{r[metric]:.2f}"
        ax.imshow(mat, cmap=cmap, vmin=0.45, vmax=1.0, aspect="auto")
        for i in range(len(COHORTS)):
            for j in range(len(METHODS)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.2,
                        color="white" if mat[i, j] > 0.82 else INK)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(["Canonical", "Single\nagent", "Multi\nagent"])
        ax.set_yticks(np.arange(len(COHORTS)))
        ax.set_yticks(np.arange(-.5, len(COHORTS), 1), minor=True)
        ax.set_xticks(np.arange(-.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.grid(False)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(np.arange(len(COHORTS)))
    axes[0].set_yticklabels([f"Cohort {c.key}" for c in COHORTS])
    panel(axes[0], "a")
    save(fig, "T1_headline_performance_redesigned")


def _point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    tp = ((y == 1) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    return {
        "Sens": tp / max(tp + fn, 1),
        "Spec": tn / max(tn + fp, 1),
    }


def fig_tau_audit(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> None:
    rows = []
    thresholds = np.array([0.02, 0.04, 0.05, 0.06, 0.08, 0.09, 0.10, 0.11,
                           0.12, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30])
    terminal_actions = {"observe_or_reassess", "direct_cta", "urgent_transfer"}
    escalate = {"direct_cta", "urgent_transfer"}
    for c in COHORTS:
        term = trace_rows(c.key, preds)
        term = term[term["final_action"].isin(terminal_actions)].drop_duplicates("ID", keep="last").copy()
        for tau in thresholds:
            action = term["final_action"].copy()
            m5 = term["risk_score"].lt(tau) & term["final_action"].isin(escalate)
            m6 = (
                term["override_reason"].eq("terminal_stage_nonlow_risk_cannot_observe")
                & term["coordinator_proposed_action"].eq("observe_or_reassess")
            )
            action[m5 | m6] = "observe_or_reassess"
            met = _point_metrics(term["label"].to_numpy(), action.isin(escalate).astype(int).to_numpy())
            rows.append({"cohort": c.key, "tau": tau, **met})
    src = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for metric, ax in zip(["Sens", "Spec"], axes):
        for c in COHORTS:
            g = src[src.cohort == c.key].sort_values("tau")
            ax.plot(g["tau"], g[metric], marker="o", ms=3.2,
                    color=COHORT_COLORS[c.key], lw=1.3, label=f"Cohort {c.key}")
        ax.axvline(0.12, color=WINE, lw=1, ls=(0, (3, 2)))
        ax.set_title("Sensitivity" if metric == "Sens" else "Specificity")
        ax.set_xlabel("Safety-governance threshold tau")
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(loc="lower right")
    save(fig, "F_appendix_tau_redesigned")


def fig_literature(metrics: pd.DataFrame) -> None:
    rows = [
        ("ADD-RS >=1", "Clinical score", 0.95, 0.264, 1850),
        ("ADD-RS + D-dimer", "Clinical score", 0.99, 0.22, 1850),
        ("Laletin 2024 CT-DL", "Imaging AI", 0.942, 0.973, 1100),
        ("Chien 2024 ECG+CXR", "Imaging AI", 0.832, 0.853, 25885),
    ]
    for c in COHORTS:
        r = metrics[(metrics["cohort"] == c.key) & (metrics["method"] == "multi_raw")].iloc[0]
        rows.append((f"This study: Cohort {c.key}", "This study", r["Sens"], r["Spec"], r["n"]))
    df = pd.DataFrame(rows, columns=["system", "category", "sensitivity", "specificity", "n"])
    category_order = ["Clinical score", "Imaging AI", "This study"]
    df["category"] = pd.Categorical(df["category"], category_order, ordered=True)
    df = df.sort_values(["category", "system"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(df))[::-1]
    colors = {"Clinical score": "#B75D69", "Imaging AI": "#C77C3A", "This study": "#00897B"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, r in df.iterrows():
        yy = y[i]
        color = colors[r["category"]]
        ax.hlines(yy, r["specificity"], r["sensitivity"], color=color, alpha=0.35, lw=2.0, zorder=1)
        ax.plot(r["sensitivity"], yy, marker="o", color=color, mec=INK, mew=0.35, ms=5.2, zorder=3)
        ax.plot(r["specificity"], yy, marker="s", color="white", mec=color, mew=1.2, ms=5.0, zorder=3)
        ax.text(1.015, yy, f"n={int(r['n']):,}", va="center", ha="left", fontsize=7.2, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(df["system"])
    ax.set_xlim(0.18, 1.10)
    ax.set_xlabel("Metric value")
    ax.set_title("Published AAS tools and this study at reported operating points")
    ax.axvline(0.80, color=GRID, lw=1.0, zorder=0)
    ax.axvline(0.90, color=GRID, lw=1.0, zorder=0)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    metric_handles = [
        Line2D([], [], marker="o", color=INK, linestyle="None", markersize=5.2, label="Sensitivity"),
        Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=INK,
               color=INK, linestyle="None", markersize=5.0, label="Specificity"),
    ]
    cat_handles = [Patch(facecolor=colors[c], alpha=0.6, label=c) for c in category_order]
    fig.legend(handles=cat_handles + metric_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=5, columnspacing=1.7,
               handlelength=1.6)
    fig.subplots_adjust(bottom=0.27, right=0.88)
    save(fig, "F8_literature_redesigned")


def fig_shap_surrogates(metrics: pd.DataFrame) -> None:
    # SHAP inputs are model-level attribution artefacts rather than cohort-count
    # artefacts.
    for stem in [
        "F7_shap_CP1_redesigned",
        "F7_shap_CP2_redesigned",
        "F7_shap_CP3_text_redesigned",
        "F7_shap_CP4_redesigned",
    ]:
        for ext in ("png", "pdf"):
            current = OUT / f"{stem}.{ext}"
            if not current.exists():
                raise FileNotFoundError(f"Missing current SHAP artefact: {current}")


def write_audit(metrics: pd.DataFrame) -> None:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    summary = current_summary(metrics)
    metrics.to_csv(AUDIT_OUT / "metrics_0529_combined.csv", index=False)
    summary.to_csv(AUDIT_OUT / "cohort_summary_0529.csv", index=False)

    def md_table(df: pd.DataFrame) -> str:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].map(lambda x: f"{x:.4f}")
        headers = [str(c) for c in df.columns]
        rows = [[str(v) for v in row] for row in df.to_numpy()]
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        out.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(out)

    lines = [
        "# Figure audit for figures_0529",
        "",
        "Inputs: the four cohort result folders.",
        "",
        "All cohort sample sizes and denominators use the retained analysis cohorts.",
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
    (AUDIT_OUT / "FIGURE_AUDIT_0529.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    style()
    metrics, preds = read_data()
    fig_study_design(metrics)
    fig_baseline(metrics, preds)
    fig_discrimination(metrics, preds)
    fig_score_quality(metrics, preds)
    fig_calibration_decision(metrics, preds)
    fig_mechanism(metrics, preds)
    fig_fn_safety(metrics, preds)
    fig_fn_safety_supplement(metrics, preds)
    fig_action_confusion(metrics, preds)
    fig_error_tradeoff(metrics)
    fig_operational_burden(metrics, preds)
    fig_table(metrics)
    fig_tau_audit(metrics, preds)
    fig_literature(metrics)
    fig_shap_surrogates(metrics)
    write_audit(metrics)
    print(f"Wrote figures and audit files to {OUT}")


if __name__ == "__main__":
    main()
