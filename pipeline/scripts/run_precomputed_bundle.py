#!/usr/bin/env python3
"""Run external validation from a generic precomputed-input bundle.

Expected bundle layout:

    pipeline/
      inputs/model_input.csv
      cp_inputs/dataset_CP1_demo_history_exam.csv
      cp_inputs/dataset_CP2_demo_history_exam_lab.csv
      cp_inputs/dataset_CP2E_demo_history_exam_lab_echo.csv
      scripts/run_external_validation.py

The script reuses the generic external-validation runner, but replaces its CP
table construction step with a pass-through loader for already prepared CP
tables. It does not assume any study-specific working split or intermediate
run directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

BUNDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = BUNDLE_ROOT / "src"
SCRIPTS = BUNDLE_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _make_precomputed_build_cp_tables(cp_dir: pathlib.Path):
    cp_paths = {
        "CP1": cp_dir / "dataset_CP1_demo_history_exam.csv",
        "CP2": cp_dir / "dataset_CP2_demo_history_exam_lab.csv",
        "CP2E": cp_dir / "dataset_CP2E_demo_history_exam_lab_echo.csv",
    }
    for path in cp_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"precomputed CP table missing: {path}")

    def _passthrough(final_df: pd.DataFrame):
        cp1 = pd.read_csv(cp_paths["CP1"])
        cp2 = pd.read_csv(cp_paths["CP2"])
        cp2e = pd.read_csv(cp_paths["CP2E"])
        for frame in (cp1, cp2, cp2e):
            frame["ID"] = frame["ID"].astype(str).str.strip()
        keep_ids = set(final_df["ID"].astype(str).str.strip())
        order = pd.Series(range(len(final_df)), index=final_df["ID"].astype(str).str.strip()).to_dict()
        out = []
        for frame in (cp1, cp2, cp2e):
            filtered = frame[frame["ID"].isin(keep_ids)].copy()
            filtered["_ord"] = filtered["ID"].map(order)
            filtered.sort_values("_ord", inplace=True)
            filtered.drop(columns=["_ord"], inplace=True)
            filtered.reset_index(drop=True, inplace=True)
            out.append(filtered)
        return tuple(out)

    return _passthrough


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation from precomputed CP input tables.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--baselines",
        type=str,
        default="canonical,single_agent,multi_agent",
        help="comma-separated baselines",
    )
    parser.add_argument("--run-tag", type=str, default="precomputed_bundle", help="subdirectory under outputs/")
    args = parser.parse_args()

    model_input = BUNDLE_ROOT / "inputs" / "model_input.csv"
    cp_inputs = BUNDLE_ROOT / "cp_inputs"
    if not model_input.exists():
        print(f"[run_precomputed_bundle] ERROR: missing {model_input}", file=sys.stderr)
        return 2
    if not cp_inputs.exists():
        print(f"[run_precomputed_bundle] ERROR: missing {cp_inputs}", file=sys.stderr)
        return 2

    out_root = BUNDLE_ROOT / "outputs" / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    import run_external_validation as runner  # type: ignore

    runner.EXTERNAL_VALIDATION_CSV = model_input
    runner.OUT_ROOT = out_root
    runner.CP_DIR = out_root / "cp_inputs"
    runner.FEATURE_DIR = out_root / "features"
    runner.SCORE_DIR = out_root / "scores"
    runner.REPORT_PATH = out_root / "report.md"
    runner.CP_DIR.mkdir(parents=True, exist_ok=True)

    runner.build_cp_tables = _make_precomputed_build_cp_tables(cp_inputs)
    sys.argv = [
        "run_external_validation.py",
        "--limit", str(args.limit),
        "--offset", str(args.offset),
        "--baselines", args.baselines,
    ]

    print(f"[run_precomputed_bundle] bundle_root  = {BUNDLE_ROOT}", flush=True)
    print(f"[run_precomputed_bundle] model_input  = {model_input}", flush=True)
    print(f"[run_precomputed_bundle] cp_inputs    = {cp_inputs}", flush=True)
    print(f"[run_precomputed_bundle] output       = {out_root}", flush=True)
    runner.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
