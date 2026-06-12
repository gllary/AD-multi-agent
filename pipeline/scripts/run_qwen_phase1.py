#!/usr/bin/env python3
"""
Server entrypoint for a Phase 1 Qwen-ETL run.

This file is the runner template, copied into each generated bundle as
`scripts/run_qwen_phase1.py`. It assumes the standard bundle layout:

    <bundle_root>/
      cp_inputs/                          ← precomputed CP1/CP2/CP2E (Qwen ETL)
        dataset_CP1_demo_history_exam.csv
        dataset_CP2_demo_history_exam_lab.csv
        dataset_CP2E_demo_history_exam_lab_echo.csv
      inputs/
        model_input.csv                   ← 19-col schema, required for ECG-text + AAS labels
      artifacts/
        policy/, precomputed/, models/{lgbm, cp3_text}
      src/llm_tool_multi_agent/
      scripts/{run_qwen_phase1.py, run_xiangya_external_validation.py, run_pathway.py}

The difference vs phase1_server_bundle/scripts/run_720.py:
    - skips `build_cp_tables` (= regex ETL) and reads the precomputed Qwen-ETL
      CP1/CP2/CP2E CSVs verbatim
    - still calls `build_ecg_text_features` (= regex on ECG诊断结论 column) to
      feed the CP3 text model — that model's feature_spec.json is built for the
      regex schema, so we keep CP3 on the regex pathway by design
    - all other downstream (LightGBM CP1/CP2/CP2E inference, multi-agent
      pathway, safety layer) is unchanged
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ── Bundle root ────────────────────────────────────────────────────────
BUNDLE_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = BUNDLE_ROOT / "src"
SCRIPTS = BUNDLE_ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── LLM gateway defaults ──────────────────────────────────────────────
# QWEN_API_KEY must be supplied by the runtime environment.
os.environ.setdefault("QWEN_BASE_URL", "http://127.0.0.1:8003/v1/")
os.environ.setdefault("QWEN_MODEL_NAME", "Qwen3-235B-A22B-Instruct")
os.environ.setdefault("QWEN_TIMEOUT", "120")
os.environ.setdefault("MPLCONFIGDIR", str(BUNDLE_ROOT / ".mplconfig"))
pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _make_parallel_run_baseline(workers: int):
    """Drop-in replacement for runner._run_baseline that issues per-patient
    pathway calls in a ThreadPoolExecutor of size `workers`. Same as the one
    shipped in phase1_server_bundle/scripts/run_720.py."""

    def _parallel_run_baseline(score_tbl, policy, llm, mode):  # type: ignore[no-untyped-def]
        import run_xiangya_external_validation as runner  # type: ignore
        from llm_tool_multi_agent.pathway_engine import run_pathway_for_patient
        from llm_tool_multi_agent.single_agent_engine import run_single_agent_for_patient
        from llm_tool_multi_agent.canonical_engine import run_canonical_for_patient

        total = len(score_tbl)
        runner._log(f"[baseline:{mode}] start n={total} llm_live={llm.is_live} workers={workers}")

        work: list[tuple[int, str, int, dict[str, float]]] = []
        for idx, (_, r) in enumerate(score_tbl.iterrows(), start=1):
            pid = str(r["ID"])
            label = int(r["label"])
            scores = {
                "CP1": float(r["CP1"]),
                "CP2": float(r["CP2"]),
                "CP3": float(r["CP3"]),
                "CP4": float(r["CP4"]),
            }
            work.append((idx, pid, label, scores))

        def process_one(item):
            idx, pid, label, scores = item
            if mode == "multi_agent":
                res = (
                    runner._run_multi_agent_fast(pid, label, scores, policy)
                    if not llm.is_live
                    else run_pathway_for_patient(pid, label, scores, policy, llm)
                )
            elif mode == "single_agent":
                res = (
                    runner._run_single_agent_fast(pid, label, scores, policy)
                    if not llm.is_live
                    else run_single_agent_for_patient(pid, label, scores, policy, llm)
                )
            elif mode == "canonical":
                res = run_canonical_for_patient(pid, label, scores, policy)
            else:
                raise ValueError(mode)
            row = {
                "ID": res.patient_id, "label": res.label,
                "visited_stages": res.visited_stages, "final_stage": res.final_stage,
                "final_score": res.final_score, "final_action": res.final_action,
                "final_pred": res.final_pred,
            }
            return idx, row, list(res.trace)

        rows: list[dict | None] = [None] * total
        traces: list = []
        traces_lock = threading.Lock()
        done = 0
        done_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"pw-{mode}") as ex:
            futures = [ex.submit(process_one, item) for item in work]
            for fut in as_completed(futures):
                idx, row, t = fut.result()
                rows[idx - 1] = row
                with traces_lock:
                    traces.extend(t)
                with done_lock:
                    done += 1
                    if done == 1 or done % 10 == 0 or done == total:
                        runner._log(f"[baseline:{mode}] progress {done}/{total} last_patient={row['ID']}")

        rows = [r for r in rows if r is not None]
        final_df = pd.DataFrame(rows)
        final_df["error_group"] = final_df.apply(
            lambda x: "TP" if x["label"] == 1 and x["final_pred"] == 1
            else "TN" if x["label"] == 0 and x["final_pred"] == 0
            else "FP" if x["label"] == 0 and x["final_pred"] == 1
            else "FN", axis=1,
        )
        metrics = runner._metrics(final_df["label"], final_df["final_pred"], final_df["final_score"])
        metrics.update(runner._path_metrics(final_df))
        trace_df = pd.DataFrame(traces)
        runner._log(f"[baseline:{mode}] done auroc={metrics.get('AUROC')} auprc={metrics.get('AUPRC')}")
        return final_df, metrics, trace_df

    return _parallel_run_baseline


def _make_qwen_build_ecg_text_features(qwen_ecg_json_path: pathlib.Path):
    """Replacement for runner.build_ecg_text_features.

    The CP3 text model's `feature_spec.json` expects categorical columns
    `ecg_text_{st_elevation, st_depression, arrhythmia, acs_like_ecg,
    text_suggests_aas}` whose values at TRAINING time came from the Qwen
    ECG-text JSON (`data/interim/features/ecg_text.json` → loader adds the
    `ecg_text_` prefix; see `train_cp3_text_model.py:_load_structured_text`).

    The default INFERENCE-time `build_ecg_text_features` in
    `run_xiangya_external_validation.py` derives those same column names
    from regex keyword matches. Same column names, different sources → ETL
    mismatch → CP3 collapses to AUROC ≈ random on external Xiangya cohorts.

    Fix: read the Qwen ECG JSON, overwrite the categorical columns. Keep the
    `kw_*` numeric columns from the regex builder (they were also regex at
    training time, see `train_cp3_text_model.py:KEYWORD_RULES`).
    """
    qwen_ecg = json.loads(qwen_ecg_json_path.read_text(encoding="utf-8"))
    print(f"[run_qwen]   loaded Qwen ECG JSON entries: {len(qwen_ecg):,}", flush=True)

    CAT_FIELDS = {
        "ecg_text_st_elevation":      "st_elevation",
        "ecg_text_st_depression":     "st_depression",
        "ecg_text_arrhythmia":        "arrhythmia",
        "ecg_text_acs_like_ecg":      "acs_like_ecg",
        "ecg_text_text_suggests_aas": "text_suggests_aas",
    }

    def _qwen_aware_build_ecg_text_features(final_df: pd.DataFrame):
        # Run the original regex builder to get kw_*, ecg_diagnosis_text, AAS,
        # and a regex-derived stub for ecg_text_*. We'll overwrite the latter.
        import run_xiangya_external_validation as runner  # local import
        ecg_text_csv, ecg_feat_df = runner._orig_build_ecg_text_features(final_df)

        ecg_feat_df = ecg_feat_df.copy()
        ecg_feat_df["ID"] = ecg_feat_df["ID"].astype(str).str.strip()

        # Overwrite the categorical columns from the Qwen ECG JSON.
        for col, qwen_field in CAT_FIELDS.items():
            ecg_feat_df[col] = ecg_feat_df["ID"].map(
                lambda vid: (str(qwen_ecg.get(vid, {}).get(qwen_field, "unknown")).strip() or "unknown")
            )
        return ecg_text_csv, ecg_feat_df

    return _qwen_aware_build_ecg_text_features


def _make_precomputed_build_cp_tables(cp_dir: pathlib.Path):
    """Replacement for runner.build_cp_tables: return the precomputed Qwen-ETL
    CP tables, filtered to the IDs in `final_df` (so --limit / --offset still
    work as expected). The original runner.main() will then `cp1.to_csv(...)`
    (overwriting the same file with the filtered content), and the rest of
    the pipeline proceeds normally.
    """

    cp1_path = cp_dir / "dataset_CP1_demo_history_exam.csv"
    cp2_path = cp_dir / "dataset_CP2_demo_history_exam_lab.csv"
    cp2e_path = cp_dir / "dataset_CP2E_demo_history_exam_lab_echo.csv"
    for p in (cp1_path, cp2_path, cp2e_path):
        if not p.exists():
            raise FileNotFoundError(f"precomputed CP table missing: {p}")

    def _passthrough(final_df: pd.DataFrame):
        cp1 = pd.read_csv(cp1_path)
        cp2 = pd.read_csv(cp2_path)
        cp2e = pd.read_csv(cp2e_path)
        for df in (cp1, cp2, cp2e):
            df["ID"] = df["ID"].astype(str).str.strip()
        # Filter precomputed tables to the IDs main() has retained after its
        # offset/limit. final_df.ID was already stringified in main() (line ~673).
        keep_ids = set(final_df["ID"].astype(str).str.strip())
        cp1  = cp1[cp1["ID"].isin(keep_ids)].copy()
        cp2  = cp2[cp2["ID"].isin(keep_ids)].copy()
        cp2e = cp2e[cp2e["ID"].isin(keep_ids)].copy()
        # Preserve final_df row order so all downstream merges line up.
        order = pd.Series(range(len(final_df)),
                          index=final_df["ID"].astype(str).str.strip()).to_dict()
        for df in (cp1, cp2, cp2e):
            df["_ord"] = df["ID"].map(order)
            df.sort_values("_ord", inplace=True)
            df.drop(columns=["_ord"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return cp1, cp2, cp2e

    return _passthrough


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1 Qwen-ETL runner — reuses precomputed CP1/CP2/CP2E from cp_inputs/"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--baselines",
        type=str,
        default="canonical,single_agent,multi_agent",
        help="comma-separated baselines",
    )
    ap.add_argument(
        "--workers", type=int, default=1,
        help="ThreadPool size; 1 = serial (original behaviour). Try 5.",
    )
    ap.add_argument("--run-tag", type=str, default="run_qwen", help="subdir under outputs/")
    args = ap.parse_args()

    model_input = BUNDLE_ROOT / "inputs" / "model_input.csv"
    cp_inputs = BUNDLE_ROOT / "cp_inputs"
    if not model_input.exists():
        print(f"[run_qwen] ERROR: missing {model_input}", file=sys.stderr); return 2
    if not (cp_inputs / "dataset_CP1_demo_history_exam.csv").exists():
        print(f"[run_qwen] ERROR: missing precomputed cp_inputs/", file=sys.stderr); return 2

    out_root = BUNDLE_ROOT / "outputs" / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Stage the precomputed CP files into the run's cp_inputs dir, since the
    # runner writes its own CP files there mid-run.
    staged_cp = out_root / "cp_inputs"
    staged_cp.mkdir(parents=True, exist_ok=True)
    for name in (
        "dataset_CP1_demo_history_exam.csv",
        "dataset_CP2_demo_history_exam_lab.csv",
        "dataset_CP2E_demo_history_exam_lab_echo.csv",
    ):
        src = cp_inputs / name
        dst = staged_cp / name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            dst.write_bytes(src.read_bytes())

    import run_xiangya_external_validation as runner  # type: ignore

    # Monkey-patch I/O + ETL
    runner.XIANGYA_FINAL_CSV = model_input
    runner.OUT_ROOT = out_root
    runner.CP_DIR = staged_cp                            # ← Qwen-ETL CP tables live here
    runner.FEATURE_DIR = out_root / "features"
    runner.SCORE_DIR = out_root / "scores"
    runner.REPORT_PATH = out_root / "report_zh.md"

    # ★ Swap 1: build_cp_tables → return precomputed Qwen-ETL CP1/CP2/CP2E
    runner.build_cp_tables = _make_precomputed_build_cp_tables(staged_cp)
    print(f"[run_qwen] monkey-patched runner.build_cp_tables -> precomputed loader ({staged_cp})", flush=True)

    # ★ Swap 2: build_ecg_text_features → overwrite ecg_text_* categoricals from Qwen JSON
    qwen_ecg_json = BUNDLE_ROOT / "inputs" / "ecg_text_qwen.json"
    if qwen_ecg_json.exists():
        # Save the original so the patched function can still call kw_*/text-len logic.
        runner._orig_build_ecg_text_features = runner.build_ecg_text_features
        runner.build_ecg_text_features = _make_qwen_build_ecg_text_features(qwen_ecg_json)
        print(f"[run_qwen] monkey-patched runner.build_ecg_text_features "
              f"-> Qwen ECG JSON ({qwen_ecg_json.name})", flush=True)
    else:
        print(f"[run_qwen] WARNING: {qwen_ecg_json} not found — "
              f"CP3 will use REGEX ecg_text_* (train-inference ETL mismatch).", flush=True)

    if args.workers > 1:
        runner._run_baseline = _make_parallel_run_baseline(args.workers)
        print(f"[run_qwen] parallel mode workers={args.workers}", flush=True)

    sys.argv = [
        "run_xiangya_external_validation.py",
        "--limit", str(args.limit),
        "--offset", str(args.offset),
        "--baselines", args.baselines,
    ]

    print(f"[run_qwen] bundle_root  = {BUNDLE_ROOT}", flush=True)
    print(f"[run_qwen] model_input  = {model_input}", flush=True)
    print(f"[run_qwen] cp_inputs    = {cp_inputs}", flush=True)
    print(f"[run_qwen] output       = {out_root}", flush=True)
    print(f"[run_qwen] baselines    = {args.baselines}", flush=True)
    print(f"[run_qwen] limit/offset = {args.limit}/{args.offset}", flush=True)
    print(flush=True)

    runner.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
