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


def _allocation_summary(
    labels: pd.Series,
    assigned_escalation: pd.Series,
) -> dict[str, float | int]:
    labels = validate_binary_series(labels, "Reference-standard labels")
    assigned_escalation = validate_binary_series(
        assigned_escalation,
        "Assigned-escalation indicators",
    )
    ad_positive_n = int((labels == 1).sum())
    ad_negative_n = int((labels == 0).sum())
    positive_escalation_n = int(((labels == 1) & (assigned_escalation == 1)).sum())
    negative_escalation_n = int(((labels == 0) & (assigned_escalation == 1)).sum())
    return {
        "n": int(len(labels)),
        "ad_positive_n": ad_positive_n,
        "ad_negative_n": ad_negative_n,
        "ad_positive_assigned_escalation_n": positive_escalation_n,
        "ad_positive_assigned_escalation_rate": (
            positive_escalation_n / ad_positive_n if ad_positive_n else float("nan")
        ),
        "ad_negative_assigned_escalation_n": negative_escalation_n,
        "ad_negative_assigned_escalation_rate": (
            negative_escalation_n / ad_negative_n if ad_negative_n else float("nan")
        ),
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
                "assigned_escalation": result.assigned_escalation,
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
            _allocation_summary(
                result_frame["label"],
                result_frame["assigned_escalation"],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
