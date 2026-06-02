"""Build the 2026-05-30 revised figure set with the frozen Cohort V2 IDs."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("AAS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
os.environ["AAS_FIGURE_SET"] = "figures_0530_v2_new"
os.environ["AAS_COHORT_V2_FROZEN_IDS"] = str(ROOT / "sn-article-template" / "cohort_v2_15220_frozen_ids.csv")

import build_figures_0530 as build  # noqa: E402


if __name__ == "__main__":
    build.main()
