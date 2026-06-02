# -*- coding: utf-8 -*-
"""Run role-specialized multi-agent pathway for one patient."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .deliberation import summarize_coordinator_history, summarize_specialist_history
from .evidence_views import build_specialist_view
from .llm_client import LLMClient
from .prompts import COORDINATOR_SYSTEM, SPECIALIST_PROMPTS
from .quantitative_tools import load_policy, risk_level
from .safety_layer import (
    POSITIVE_ACTIONS,
    action_to_next_stage,
    allowed_actions_for_stage,
    validate_coordinator_proposal,
)


STAGE_SPECIALISTS = {
    "CP1": ("history", "examination"),
    "CP2": ("lab_context", "lab_biomarker"),
    "CP3": ("ecg",),
    "CP4": ("echocardiography",),
}

AGENT_NAMES = {
    "history": "history_agent",
    "examination": "examination_agent",
    "lab_context": "lab_context_agent",
    "lab_biomarker": "lab_biomarker_agent",
    "laboratory": "lab_agent",
    "ecg": "ecg_agent",
    "echocardiography": "echo_agent",
}


@dataclass
class PathwayResult:
    patient_id: str
    label: int
    visited_stages: str
    final_stage: str
    final_score: float
    final_action: str
    final_pred: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    specialist_outputs: list[dict[str, Any]] = field(default_factory=list)
    coordinator_outputs: list[dict[str, Any]] = field(default_factory=list)


def _history_summary_blob(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(none)"
    return json.dumps(items, ensure_ascii=False, indent=2)


def run_pathway_for_patient(
    patient_id: str,
    label: int,
    scores: dict[str, float],
    policy: dict[str, Any] | None,
    llm: LLMClient | None = None,
) -> PathwayResult:
    policy = policy or load_policy()
    llm = llm or LLMClient()
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]

    current = "CP1"
    visited: list[str] = []
    trace: list[dict[str, Any]] = []
    specialists: list[dict[str, Any]] = []
    coordinators: list[dict[str, Any]] = []

    while current is not None:
        score = float(scores.get(current, 0.5))
        c_thr = float(c_thrs.get(current, 0.5))
        a_thr = float(a_thrs.get(current, 0.5))
        rl_tool = risk_level(score, c_thr, a_thr)
        allowed_actions = allowed_actions_for_stage(current)

        specialist_history = [{"stage": x["stage"], "output": x["output"]} for x in specialists]
        coordinator_history = [{"stage": x["stage"], "output": x["output"]} for x in coordinators]
        specialist_summary = summarize_specialist_history(specialist_history)
        coordinator_summary = summarize_coordinator_history(coordinator_history)
        current_stage_specialists: list[dict[str, Any]] = []
        for role_key in STAGE_SPECIALISTS[current]:
            evidence = build_specialist_view(patient_id, current, role=role_key)
            user_spec = (
                f"patient_id: {patient_id}\n"
                f"stage: {current}\n"
                f"allowed_actions: {allowed_actions}\n"
                f"tool_risk_score: {score:.6f}\n"
                f"tool_continue_threshold: {c_thr:.6f}\n"
                f"tool_action_threshold: {a_thr:.6f}\n"
                f"tool_discrete_risk_level: {rl_tool}\n"
                f"specialist_role: {role_key}\n"
                f"evidence_boundary: role-bounded structured view only\n"
                f"prior_specialist_summary:\n{json.dumps(specialist_summary, ensure_ascii=False, indent=2)}\n"
                f"prior_specialist_history:\n{_history_summary_blob(specialist_history)}\n"
            )
            if current == "CP3":
                user_spec += (
                    "note: raw 12-lead ECG is not pasted here; this agent sees only the 1D-CNN ECG risk signal "
                    "plus minimal history/exam context, not laboratory or echo findings.\n"
                )
            user_spec += f"curated_structured_evidence:\n{evidence}\n"
            spec_out = llm.specialist_json(
                SPECIALIST_PROMPTS[role_key],
                role_key,
                current,
                user_spec,
                score,
                rl_tool,
            )
            current_specialist = {"stage": current, "output": spec_out}
            specialists.append(current_specialist)
            current_stage_specialists.append(current_specialist)

        coord_user = {
            "patient_id": patient_id,
            "label": label,
            "current_stage": current,
            "current_quantitative_state": {
                "risk_score": score,
                "risk_level": rl_tool,
                "continue_threshold": c_thr,
                "action_threshold": a_thr,
            },
            "allowed_actions": allowed_actions,
            "current_specialist_outputs": [x["output"] for x in current_stage_specialists],
            "specialist_summary": specialist_summary,
            "specialist_history": specialist_history,
            "coordinator_summary": coordinator_summary,
            "coordinator_history": coordinator_history,
        }
        coord_raw = llm.coordinator_json(
            COORDINATOR_SYSTEM,
            current,
            json.dumps(coord_user, ensure_ascii=False, indent=2),
            allowed_actions,
            current_stage_specialists,
        )
        coordinators.append({"stage": current, "output": coord_raw})

        dec = validate_coordinator_proposal(current, score, policy, coord_raw)
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "label": label,
                "stage": current,
                "agent": "coordinator",
                "specialist_agents": [AGENT_NAMES[x["output"]["agent_role"]] for x in current_stage_specialists],
                "risk_score": score,
                "risk_level": rl_tool,
                "evidence_boundary": current,
                "allowed_actions": allowed_actions,
                "specialist_summary": specialist_summary,
                "coordinator_summary": coordinator_summary,
                "specialist_jsons": [x["output"] for x in current_stage_specialists],
                "coordinator_json": coord_raw,
                "consensus_state": coord_raw.get("consensus_state"),
                "key_conflicts": coord_raw.get("key_conflicts"),
                "information_gap": coord_raw.get("information_gap"),
                "coordinator_proposed_action": dec.coordinator_action,
                "canonical_action": dec.canonical_action,
                "canonical_next_stage": dec.canonical_next_stage,
                "is_action_allowed": dec.is_action_allowed,
                "final_action": dec.final_action,
                "final_next_stage": dec.final_next_stage,
                "override_reason": dec.override_reason,
                "safety_risk_level": dec.safety_risk_level,
                "policy_basis": dec.policy_basis,
            }
        )

        nxt = dec.final_next_stage
        if nxt is None:
            final_action = dec.final_action
            final_pred = 1 if final_action in POSITIVE_ACTIONS else 0
            return PathwayResult(
                patient_id=patient_id,
                label=label,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=final_action,
                final_pred=final_pred,
                trace=trace,
                specialist_outputs=specialists,
                coordinator_outputs=coordinators,
            )

        # Defensive check: non-terminal actions must map to a known next stage.
        resolved_next = action_to_next_stage(dec.final_action)
        if resolved_next is None:
            raise RuntimeError(f"Non-terminal action without next stage mapping: {dec.final_action}")
        current = resolved_next

    raise RuntimeError("pathway fell through without terminal")
