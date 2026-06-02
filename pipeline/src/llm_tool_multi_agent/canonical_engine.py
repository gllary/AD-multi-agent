# -*- coding: utf-8 -*-
"""Pure canonical threshold-route baseline without any LLM or specialist deliberation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .quantitative_tools import load_policy, risk_level
from .safety_layer import POSITIVE_ACTIONS, canonical_route


@dataclass
class CanonicalResult:
    patient_id: str
    label: int
    visited_stages: str
    final_stage: str
    final_score: float
    final_action: str
    final_pred: int
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_canonical_for_patient(
    patient_id: str,
    label: int,
    scores: dict[str, float],
    policy: dict[str, Any] | None,
) -> CanonicalResult:
    policy = policy or load_policy()
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]

    current = "CP1"
    visited: list[str] = []
    trace: list[dict[str, Any]] = []

    while current is not None:
        score = float(scores.get(current, 0.5))
        c_thr = float(c_thrs.get(current, 0.5))
        a_thr = float(a_thrs.get(current, 0.5))
        rl_tool = risk_level(score, c_thr, a_thr)
        nxt, action = canonical_route(current, score, c_thrs, a_thrs)
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "label": label,
                "stage": current,
                "agent": "canonical_router",
                "risk_score": score,
                "risk_level": rl_tool,
                "final_action": action,
                "final_next_stage": nxt,
                "policy_basis": [f"risk_level={rl_tool}", "canonical_threshold_route"],
            }
        )
        if nxt is None:
            final_pred = 1 if action in POSITIVE_ACTIONS else 0
            return CanonicalResult(
                patient_id=patient_id,
                label=label,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=action,
                final_pred=final_pred,
                trace=trace,
            )
        current = nxt

    raise RuntimeError("canonical pathway fell through without terminal")
