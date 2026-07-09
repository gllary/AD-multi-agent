"""Build release figures from generic retained prediction files.

Expected input layout:

    data/derived/cohort_D/FINAL_retained_predictions.csv
    restricted_inputs/cohort_V1/FINAL_retained_predictions.csv
    restricted_inputs/cohort_V2/FINAL_retained_predictions.csv

External cohort files are not included in the public repository. Provide them
under ``AD_RESTRICTED_INPUT_ROOT`` with the same relative layout, or override
the cohort list with ``--cohorts`` after editing ``_data.COHORT_FILES``.

Each retained prediction file must contain:

    ID, label, canonical_pred, single_pred, multi_raw_pred

Optional per-method pathway outputs may be placed under each cohort directory:

    run_outputs/{canonical,single_agent,multi_agent}/pathway_final_outcomes.csv
    run_outputs/multi_agent/pathway_decision_trace.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _data import COHORT_FILES, COHORT_META, COHORT_ORDER, METHOD_TO_COLUMN  # noqa: E402
from _style import COLORS, METHOD_LABELS, apply_style, save_fig  # noqa: E402

METHODS = ["canonical", "single_agent", "multi_agent"]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _cohort_label(cohort: str) -> str:
    meta = COHORT_META.get(cohort, {})
    if cohort == "cohort_D":
        return "Cohort D"
    if cohort == "cohort_V1":
        return "Cohort V1"
    if cohort == "cohort_V2":
        return "Cohort V2"
    return str(meta.get("site_label", cohort))


def _read_predictions(cohort: str) -> pd.DataFrame | None:
    path = COHORT_FILES[cohort]
    if not path.exists():
        print(f"[build_figures] skip {cohort}: missing {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    required = {"ID", "label", *METHOD_TO_COLUMN.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df.copy()
    out["ID"] = out["ID"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def _confusion(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    yv = y.astype(int).to_numpy()
    pv = pred.astype(int).to_numpy()
    tp = int(((yv == 1) & (pv == 1)).sum())
    tn = int(((yv == 0) & (pv == 0)).sum())
    fp = int(((yv == 0) & (pv == 1)).sum())
    fn = int(((yv == 1) & (pv == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "n": int(len(yv)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
    }


def load_results(cohorts: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for cohort in cohorts:
        pred = _read_predictions(cohort)
        if pred is None:
            continue
        frames[cohort] = pred
        for method in METHODS:
            col = METHOD_TO_COLUMN[method]
            row = _confusion(pred["label"], pred[col])
            row.update({
                "cohort": cohort,
                "cohort_label": _cohort_label(cohort),
                "method": method,
                "method_label": _method_label(method),
            })
            rows.append(row)
    if not rows:
        raise RuntimeError("No cohort prediction files were available")
    return pd.DataFrame(rows), frames


def write_tables(metrics: pd.DataFrame, preds: dict[str, pd.DataFrame], out: Path) -> None:
    audit = out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(audit / "headline_metrics.csv", index=False)
    summary_rows = []
    for cohort, df in preds.items():
        n = len(df)
        pos = int(df["label"].sum())
        summary_rows.append({
            "cohort": cohort,
            "cohort_label": _cohort_label(cohort),
            "n": n,
            "ad_positive": pos,
            "ad_negative": n - pos,
            "ad_prevalence": pos / max(n, 1),
        })
    pd.DataFrame(summary_rows).to_csv(audit / "cohort_summary.csv", index=False)


def plot_cohort_summary(preds: dict[str, pd.DataFrame], out: Path) -> None:
    rows = []
    for cohort, df in preds.items():
        pos = int(df["label"].sum())
        rows.append({
            "cohort": _cohort_label(cohort),
            "ad_positive": pos,
            "ad_negative": len(df) - pos,
            "prevalence": pos / max(len(df), 1),
        })
    data = pd.DataFrame(rows)
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(data))
    ax.bar(x, data["ad_negative"], color="#B8C1C8", label="AD-negative")
    ax.bar(x, data["ad_positive"], bottom=data["ad_negative"], color=COLORS["cohort_V2"], label="AD-positive")
    for i, row in data.iterrows():
        ax.text(i, row["ad_negative"] + row["ad_positive"], f"{row['prevalence']*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["cohort"])
    ax.set_ylabel("Patients")
    ax.set_title("Retained cohort composition")
    ax.legend(loc="upper left")
    save_fig(fig, out / "cohort_composition")
    plt.close(fig)


def plot_performance(metrics: pd.DataFrame, out: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    method_colors = {
        "canonical": COLORS["canonical"],
        "single_agent": COLORS["single_agent"],
        "multi_agent": COLORS["multi_agent"],
    }
    xlabels = [_cohort_label(c) for c in metrics["cohort"].drop_duplicates()]
    x = np.arange(len(xlabels))
    width = 0.23
    for offset, method in zip([-width, 0, width], METHODS):
        sub = metrics[metrics["method"].eq(method)]
        axes[0].bar(x + offset, sub["sensitivity"], width, color=method_colors[method], label=_method_label(method))
        axes[1].bar(x + offset, sub["specificity"], width, color=method_colors[method], label=_method_label(method))
    for ax, title in zip(axes, ["AD-positive assigned escalation", "AD-negative non-escalation"]):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
    axes[0].set_ylabel("Proportion")
    axes[1].legend(loc="lower right")
    save_fig(fig, out / "headline_performance")
    plt.close(fig)


def plot_error_counts(metrics: pd.DataFrame, out: Path) -> None:
    apply_style()
    multi = metrics[metrics["method"].eq("multi_agent")].copy()
    labels = multi["cohort"].map(_cohort_label).tolist()
    x = np.arange(len(multi))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x, multi["FP"], color=COLORS["fp"], label="AD-negative assigned escalation")
    ax.bar(x, multi["FN"], bottom=multi["FP"], color=COLORS["fn"], label="AD-positive reassessment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patients")
    ax.set_title("Residual error counts for the multi-agent pathway")
    ax.legend(loc="upper left")
    save_fig(fig, out / "multi_agent_error_counts")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic release figures from retained prediction files.")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_figures" / "release_figures")
    parser.add_argument("--cohorts", nargs="*", default=COHORT_ORDER, choices=list(COHORT_FILES))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metrics, preds = load_results(args.cohorts)
    write_tables(metrics, preds, args.out)
    plot_cohort_summary(preds, args.out)
    plot_performance(metrics, args.out)
    plot_error_counts(metrics, args.out)
    print(f"Wrote release figures and audit tables to {args.out}")


if __name__ == "__main__":
    main()
