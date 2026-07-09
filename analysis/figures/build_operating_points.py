"""Plot operating points from a generic CSV file.

Input CSV columns:

    system, specificity, sensitivity

Optional columns:

    marker, color
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLORS = ["#607D8B", "#C77C3A", "#00897B", "#E6B85C", "#D96C5F"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot same-patient operating points.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with system/specificity/sensitivity columns")
    parser.add_argument("--out", type=Path, default=Path("paper_figures/operating_points"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"system", "specificity", "sensitivity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{args.input} missing required columns: {missing}")
    if "marker" not in df.columns:
        df["marker"] = "o"
    if "color" not in df.columns:
        df["color"] = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(df))]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, row in df.iterrows():
        ax.scatter(row["specificity"], row["sensitivity"], s=78, marker=row["marker"],
                   color=row["color"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(row["specificity"] + 0.01, row["sensitivity"], row["system"],
                color=row["color"], ha="left", va="center", fontweight="bold", fontsize=8.4)
    ax.set_xlim(max(0, df["specificity"].min() - 0.08), min(1, df["specificity"].max() + 0.18))
    ax.set_ylim(max(0, df["sensitivity"].min() - 0.08), min(1, df["sensitivity"].max() + 0.08))
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Same-patient operating points")
    ax.grid(True, color="#E8EEF2", linewidth=0.65)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"operating_points.{ext}", bbox_inches="tight", dpi=320)
    plt.close(fig)
    print(f"Wrote operating-point figure to {args.out}")


if __name__ == "__main__":
    main()
