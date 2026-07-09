"""Build the 2026-05-30 revised figure set with the frozen Cohort V2 IDs."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
os.environ["AD_FIGURE_SET"] = "figures_0530_v2_new"
os.environ["AD_COHORT_V2_FROZEN_IDS"] = str(ROOT / "sn-article-template" / "current_v2_dataset_ids0605.csv")

import build_figures_0530 as build  # noqa: E402


if __name__ == "__main__":
    build.main()
