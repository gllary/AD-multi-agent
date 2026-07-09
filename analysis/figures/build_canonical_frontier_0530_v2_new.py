"""Redraw the Cohort V2 canonical threshold-frontier supplement figure.

This script replays the deterministic canonical route on the frozen merged
Cohort V2 set. It jointly sweeps the continuation-threshold scale and the
common action threshold, then plots the best achievable canonical sensitivity
at each or lower false-positive rate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = ROOT / "paper_figures" / "figures_0530_v2_new"
AUDIT_OUT = ROOT / "paper_figures" / "figures_0530_v2_new_audit"
FROZEN_IDS = ROOT / "sn-article-template" / "current_v2_dataset_ids0605.csv"
POLICY_JSON = ROOT / "phase1_qwen_16218_bundle" / "artifacts" / "policy" / "best_policy_thresholds.json"

SCORE_SOURCES = [
    ("V2", ROOT / "phase1_qwen_720_bundle" / "outputs" / "run_qwen" / "scores" / "xiangya_score_table_cp3_text.csv"),
    ("V3", ROOT / "phase1_qwen_16218_bundle" / "outputs" / "run_qwen" / "scores" / "xiangya_score_table_cp3_text.csv"),
]

INK = "#263238"
MUTED = "#6F7F87"
GRID = "#E8EEF2"
CANONICAL = "#607D8B"
SINGLE = "#C77C3A"
MULTI = "#00897B"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "grid.alpha": 1,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_scores() -> pd.DataFrame:
    frozen = pd.read_csv(FROZEN_IDS)
    frozen["ID"] = frozen["ID"].astype(str)
    frames = []
    for prefix, path in SCORE_SOURCES:
        frame = pd.read_csv(path)
        frame["ID"] = prefix + "::" + frame["ID"].astype(str)
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    out = frozen[["ID"]].merge(scores, on="ID", how="left")
    missing = out[["label", "CP1", "CP2", "CP3", "CP4"]].isna().any(axis=1)
    if missing.any():
        examples = ", ".join(out.loc[missing, "ID"].head(5).astype(str))
        raise ValueError(f"Missing stage scores for {int(missing.sum())} frozen IDs, examples: {examples}")
    return out


def canonical_predictions(
    scores: pd.DataFrame,
    policy: dict,
    continuation_scale: float,
    action_threshold: float,
) -> np.ndarray:
    s1 = scores["CP1"].astype(float).to_numpy()
    s2 = scores["CP2"].astype(float).to_numpy()
    s3 = scores["CP3"].astype(float).to_numpy()
    s4 = scores["CP4"].astype(float).to_numpy()
    c = {
        stage: float(policy["continue_thresholds"][stage]) * continuation_scale
        for stage in ("CP1", "CP2", "CP3", "CP4")
    }

    positive = np.zeros(len(scores), dtype=bool)

    low1 = s1 < c["CP1"]
    high1 = (~low1) & (s1 >= action_threshold)
    inter1 = (~low1) & (~high1)
    positive |= high1 & (s1 >= 0.9)
    to_cp4 = high1 & (s1 < 0.9)
    to_cp2 = inter1

    low2 = to_cp2 & (s2 < c["CP2"])
    high2 = to_cp2 & (~low2) & (s2 >= action_threshold)
    inter2 = to_cp2 & (~low2) & (~high2)
    positive |= high2 & (s2 >= 0.9)
    to_cp4 |= high2 & (s2 < 0.9)
    to_cp4 |= inter2 & (s2 >= 0.15)
    to_cp3 = inter2 & (s2 < 0.15)

    low3 = to_cp3 & (s3 < c["CP3"])
    not_low3 = to_cp3 & (~low3)
    positive |= not_low3 & (s3 >= 0.75)
    to_cp4 |= not_low3 & (s3 < 0.75)

    low4 = to_cp4 & (s4 < c["CP4"])
    positive |= to_cp4 & (~low4)
    return positive.astype(int)


def confusion_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "fpr": fp / max(tn + fp, 1),
    }


def build_frontier(scores: pd.DataFrame, policy: dict) -> pd.DataFrame:
    y = scores["label"].astype(int).to_numpy()
    rows = []
    continuation_scales = np.r_[np.linspace(0.25, 1.50, 51), np.linspace(1.55, 5.00, 70)]
    action_thresholds = np.linspace(0.20, 0.95, 151)
    for continuation_scale in continuation_scales:
        for action_threshold in action_thresholds:
            pred = canonical_predictions(scores, policy, float(continuation_scale), float(action_threshold))
            row = confusion_metrics(y, pred)
            row["continuation_scale"] = float(continuation_scale)
            row["action_threshold"] = float(action_threshold)
            rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["fpr", "sensitivity"]).reset_index(drop=True)
    grid["frontier_sensitivity"] = grid["sensitivity"].cummax()
    is_frontier = grid["frontier_sensitivity"].gt(grid["frontier_sensitivity"].shift(fill_value=-1))
    frontier = grid.loc[is_frontier, [
        "fpr", "frontier_sensitivity", "sensitivity", "specificity",
        "continuation_scale", "action_threshold", "TP", "TN", "FP", "FN",
    ]].rename(columns={"frontier_sensitivity": "best_sensitivity"})
    return grid, frontier


def best_at_or_below(frontier: pd.DataFrame, fpr: float) -> float:
    eligible = frontier[frontier["fpr"] <= fpr]
    if eligible.empty:
        return float("nan")
    return float(eligible["best_sensitivity"].max())


def plot(frontier: pd.DataFrame) -> None:
    operating = pd.DataFrame([
        {"method": "Canonical", "fpr": 0.3781017931534142, "sensitivity": 0.9518072289156626, "color": CANONICAL},
        {"method": "Single-agent", "fpr": 0.30447382720521643, "sensitivity": 0.9429554954511938, "color": SINGLE},
        {"method": "Multi-agent", "fpr": 0.27639920297047636, "sensitivity": 0.9697565761003196, "color": MULTI},
    ])
    multi = operating[operating["method"] == "Multi-agent"].iloc[0]
    frontier_at_multi = best_at_or_below(frontier, float(multi["fpr"]))
    delta = float(multi["sensitivity"]) - frontier_at_multi

    style()
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    ax.plot(frontier["fpr"], frontier["best_sensitivity"], color="#8B969C", lw=2.0,
            label="Canonical threshold frontier")
    ax.scatter(operating["fpr"], operating["sensitivity"], s=[54, 54, 70],
               c=operating["color"], edgecolor="white", linewidth=0.8, zorder=5)

    offsets = {
        "Canonical": (0.012, -0.010, "left"),
        "Single-agent": (0.010, 0.007, "left"),
        "Multi-agent": (-0.012, 0.007, "right"),
    }
    for _, row in operating.iterrows():
        dx, dy, ha = offsets[row["method"]]
        ax.text(row["fpr"] + dx, row["sensitivity"] + dy, row["method"],
                ha=ha, va="center", color=row["color"], fontweight="bold")

    ax.vlines(float(multi["fpr"]), frontier_at_multi, float(multi["sensitivity"]),
              color=MULTI, lw=1.1, linestyles=(0, (3, 2)))
    ax.annotate(
        f"+{delta:.3f} sensitivity\nabove frontier\nat same or lower FPR",
        xy=(float(multi["fpr"]), (frontier_at_multi + float(multi["sensitivity"])) / 2),
        xytext=(0.315, 0.938),
        arrowprops=dict(arrowstyle="-|>", color=MULTI, lw=0.9),
        ha="left",
        va="center",
        color=MULTI,
        fontsize=8.4,
    )

    ax.set_xlim(0.16, 0.49)
    ax.set_ylim(0.84, 0.985)
    ax.set_xlabel("False-positive rate (1 - specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Canonical threshold frontier on frozen Cohort V2")
    ax.text(0.162, 0.848, "Grid replay: continuation-threshold scale x common action threshold",
            color=MUTED, fontsize=7.6, ha="left", va="bottom")
    ax.legend(loc="lower right", handlelength=2.3)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F_S10_canonical_frontier_redesigned.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scores = load_scores()
    policy = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    grid, frontier = build_frontier(scores, policy)

    y = scores["label"].astype(int).to_numpy()
    default_pred = canonical_predictions(scores, policy, 1.0, 0.5)
    default_metrics = confusion_metrics(y, default_pred)
    if default_metrics["TP"] != 3871 or default_metrics["TN"] != 6867:
        raise RuntimeError(f"Default replay did not reproduce frozen canonical counts: {default_metrics}")

    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(AUDIT_OUT / "F_S10_canonical_threshold_grid.csv", index=False)
    frontier.to_csv(AUDIT_OUT / "F_S10_canonical_threshold_frontier.csv", index=False)
    plot(frontier)
    multi_fpr = 0.27639920297047636
    frontier_at_multi = best_at_or_below(frontier, multi_fpr)
    print(
        "Wrote F_S10_canonical_frontier_redesigned; "
        f"default TP/TN/FP/FN={default_metrics['TP']}/{default_metrics['TN']}/"
        f"{default_metrics['FP']}/{default_metrics['FN']}; "
        f"frontier sensitivity at multi FPR={frontier_at_multi:.6f}"
    )


if __name__ == "__main__":
    main()
