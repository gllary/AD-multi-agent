"""Build a generic canonical-threshold frontier from one stage-score table.

Expected score-table columns:

    ID, label, CP1, CP2, CP3, CP4

By default the script reads:

    restricted_inputs/cohort_V2/scores/score_table_cp3_text.csv

Override with ``AD_CANONICAL_FRONTIER_SCORE_TABLE`` or ``--score-table``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESTRICTED_INPUT_ROOT = Path(os.environ.get("AD_RESTRICTED_INPUT_ROOT", ROOT / "restricted_inputs"))


def load_scores(score_table: Path, id_file: Path | None = None) -> pd.DataFrame:
    scores = pd.read_csv(score_table)
    required = {"ID", "label", "CP1", "CP2", "CP3", "CP4"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{score_table} missing required columns: {missing}")
    scores["ID"] = scores["ID"].astype(str)
    if id_file and id_file.exists():
        ids = pd.read_csv(id_file)
        if "ID" not in ids.columns:
            raise ValueError(f"{id_file} must contain an ID column")
        ids["ID"] = ids["ID"].astype(str)
        scores = ids[["ID"]].merge(scores, on="ID", how="left")
        missing_scores = scores[["label", "CP1", "CP2", "CP3", "CP4"]].isna().any(axis=1)
        if missing_scores.any():
            examples = ", ".join(scores.loc[missing_scores, "ID"].head(5).astype(str))
            raise ValueError(f"Missing stage scores for {int(missing_scores.sum())} IDs, examples: {examples}")
    return scores


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


def build_frontier(scores: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def plot_frontier(frontier: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(frontier["fpr"], frontier["best_sensitivity"], color="#607D8B", lw=2.0)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Best achievable sensitivity")
    ax.set_title("Canonical threshold frontier")
    ax.grid(True, color="#E8EEF2", linewidth=0.65)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"canonical_threshold_frontier.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a canonical threshold frontier from one score table.")
    parser.add_argument(
        "--score-table",
        type=Path,
        default=Path(os.environ.get(
            "AD_CANONICAL_FRONTIER_SCORE_TABLE",
            RESTRICTED_INPUT_ROOT / "cohort_V2" / "scores" / "score_table_cp3_text.csv",
        )),
    )
    parser.add_argument(
        "--id-file",
        type=Path,
        default=Path(os.environ.get(
            "AD_CANONICAL_FRONTIER_ID_FILE",
            RESTRICTED_INPUT_ROOT / "cohort_V2" / "retained_ids.csv",
        )),
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path(os.environ.get(
            "AD_POLICY_JSON",
            ROOT / "pipeline" / "artifacts" / "policy" / "best_policy_thresholds.json",
        )),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "paper_figures" / "canonical_frontier")
    args = parser.parse_args()

    scores = load_scores(args.score_table, args.id_file if args.id_file.exists() else None)
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    grid, frontier = build_frontier(scores, policy)
    args.out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out / "canonical_threshold_grid.csv", index=False)
    frontier.to_csv(args.out / "canonical_threshold_frontier.csv", index=False)
    plot_frontier(frontier, args.out)
    print(f"Wrote canonical frontier outputs to {args.out}")


if __name__ == "__main__":
    main()
