"""Redraw Supplement Fig. S9 clinical-score operating points for frozen V2."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AAS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUT = ROOT / "paper_figures" / "figures_0530_v2_new"
AUDIT_OUT = ROOT / "paper_figures" / "figures_0530_v2_new_audit"
METRICS = AUDIT_OUT / "metrics_0530_v2_new.csv"

INK = "#263238"
MUTED = "#6F7F87"
GRID = "#E8EEF2"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
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


def build_points() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    v2 = metrics[metrics["cohort"].eq("V2")].copy()
    label_map = {
        "canonical": "Canonical",
        "single": "Single-agent",
        "multi_raw": "Multi-agent",
    }
    rows = []
    for method, label in label_map.items():
        row = v2[v2["method"].eq(method)].iloc[0]
        rows.append({
            "system": label,
            "specificity": float(row["Spec"]),
            "sensitivity": float(row["Sens"]),
            "marker": "o",
            "color": {"Canonical": "#607D8B", "Single-agent": "#C77C3A", "Multi-agent": "#00897B"}[label],
        })
    rows.extend([
        {
            "system": "ADD-RS >=1",
            "specificity": 0.4968302844,
            "sensitivity": 0.9253231211,
            "marker": "s",
            "color": "#E6B85C",
        },
        {
            "system": "ADvISED-style",
            "specificity": 0.3952182576,
            "sensitivity": 0.9104834849,
            "marker": "s",
            "color": "#D96C5F",
        },
    ])
    out = pd.DataFrame(rows)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT_OUT / "F_S9_addrs_advised_operating_points.csv", index=False)
    return out


def main() -> None:
    df = build_points()
    style()
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    for _, row in df.iterrows():
        ax.scatter(row["specificity"], row["sensitivity"], s=80, marker=row["marker"],
                   color=row["color"], edgecolor="white", linewidth=0.8, zorder=3)
        if row["system"] == "Multi-agent":
            dx, dy, ha = 0.012, 0.003, "left"
        elif row["system"] == "Single-agent":
            dx, dy, ha = 0.012, -0.004, "left"
        elif row["system"] == "Canonical":
            dx, dy, ha = -0.012, -0.005, "right"
        elif row["system"] == "ADD-RS >=1":
            dx, dy, ha = 0.012, 0.003, "left"
        else:
            dx, dy, ha = 0.012, -0.004, "left"
        ax.text(row["specificity"] + dx, row["sensitivity"] + dy, row["system"],
                color=row["color"], ha=ha, va="center", fontweight="bold", fontsize=8.6)

    ax.set_xlim(0.34, 0.76)
    ax.set_ylim(0.895, 0.952)
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Same-patient operating points in Cohort V2")
    ax.text(0.342, 0.897,
            "Clinical-score comparators were reconstructed from structured fields\n"
            "and evaluated against the same action-derived escalation outcome.",
            ha="left", va="bottom", color=MUTED, fontsize=7.6)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F_S9_addrs_advised_operating_points_redesigned.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote F_S9_addrs_advised_operating_points_redesigned")


if __name__ == "__main__":
    main()
