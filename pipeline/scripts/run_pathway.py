"""Run pathway allocation from CP1-CP4 risk scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_tool_multi_agent.fixed_threshold_engine import (
    run_fixed_threshold_for_patient,
)
from llm_tool_multi_agent.quantitative_tools import (
    PATHWAY_STAGES,
    load_policy,
    validate_binary_series,
    validate_score_table,
)


def _metrics(labels: pd.Series, predictions: pd.Series) -> dict[str, float | int]:
    labels = validate_binary_series(labels, "Metric labels")
    predictions = validate_binary_series(predictions, "Metric predictions")
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(labels)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pathway using CP1-CP4 risk scores."
    )
    parser.add_argument(
        "--score-table",
        type=Path,
        required=True,
        help="CSV containing ID, label, CP1, CP2, CP3, and CP4 columns.",
    )
    parser.add_argument(
        "--method",
        choices=["fixed-threshold", "single-agent", "multi-agent"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    scores = validate_score_table(pd.read_csv(arguments.score_table))
    if arguments.limit:
        scores = scores.head(arguments.limit)

    policy = load_policy()
    results: list[dict] = []
    traces: list[dict] = []
    for row in scores.itertuples(index=False):
        patient_scores = {
            stage: float(getattr(row, stage))
            for stage in PATHWAY_STAGES
        }
        if arguments.method == "fixed-threshold":
            result = run_fixed_threshold_for_patient(
                str(row.ID),
                patient_scores,
                policy,
            )
        elif arguments.method == "single-agent":
            from llm_tool_multi_agent.single_agent_engine import (
                run_single_agent_for_patient,
            )

            result = run_single_agent_for_patient(
                str(row.ID),
                patient_scores,
                policy,
            )
        else:
            from llm_tool_multi_agent.pathway_engine import run_pathway_for_patient

            result = run_pathway_for_patient(
                str(row.ID),
                patient_scores,
                policy,
            )
        results.append(
            {
                "ID": result.patient_id,
                "label": int(row.label),
                "final_stage": result.final_stage,
                "final_score": result.final_score,
                "final_action": result.final_action,
                "final_pred": result.final_pred,
            }
        )
        traces.extend(result.trace)

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "terminal_actions.csv", index=False)
    (output_dir / "audit_trace.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            _metrics(result_frame["label"], result_frame["final_pred"]),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
