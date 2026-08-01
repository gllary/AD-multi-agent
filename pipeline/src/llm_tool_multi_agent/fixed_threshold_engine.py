"""Frozen fixed-threshold comparator without LLM deliberation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .quantitative_tools import load_policy, risk_level
from .safety_layer import POSITIVE_ACTIONS, fixed_threshold_route


@dataclass
class FixedThresholdResult:
    patient_id: str
    visited_stages: str
    final_stage: str
    final_score: float
    final_action: str
    final_pred: int
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_fixed_threshold_for_patient(
    patient_id: str,
    scores: dict[str, float],
    policy: dict[str, Any] | None,
) -> FixedThresholdResult:
    policy = policy or load_policy()
    continue_thresholds = policy["continue_thresholds"]
    action_thresholds = policy["action_thresholds"]

    current = "CP1"
    visited: list[str] = []
    trace: list[dict[str, Any]] = []
    while current is not None:
        score = float(scores[current])
        continue_threshold = float(continue_thresholds[current])
        action_threshold = float(action_thresholds[current])
        state = risk_level(score, continue_threshold, action_threshold)
        next_stage, action = fixed_threshold_route(
            current,
            score,
            continue_thresholds,
            action_thresholds,
        )
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "stage": current,
                "agent": "fixed_threshold_router",
                "risk_score": score,
                "risk_level": state,
                "final_action": action,
                "final_next_stage": next_stage,
                "policy_basis": [
                    f"risk_level={state}",
                    "fixed_threshold_pathway",
                ],
            }
        )
        if next_stage is None:
            return FixedThresholdResult(
                patient_id=patient_id,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=action,
                final_pred=int(action in POSITIVE_ACTIONS),
                trace=trace,
            )
        current = next_stage
    raise RuntimeError("Fixed-threshold pathway ended without a terminal action")
