# -*- coding: utf-8 -*-
"""Run the single-agent stage-bounded pathway comparator for one patient."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .curated_evidence import evidence_contract_from_text
from .deliberation import summarize_coordinator_history
from .evidence_views import build_single_agent_view
from .llm_client import LLMClient, LLMOutputError, validation_audit_record
from .prompts import SINGLE_AGENT_SYSTEM
from .quantitative_tools import load_policy, risk_level
from .safety_layer import (
    ESCALATION_ACTIONS,
    action_to_next_stage,
    allowed_actions_for_stage,
    validate_coordinator_proposal,
)


@dataclass
class SingleAgentResult:
    patient_id: str
    visited_stages: str
    final_stage: str
    final_score: float
    final_action: str
    assigned_escalation: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    controller_outputs: list[dict[str, Any]] = field(default_factory=list)


def run_single_agent_for_patient(
    patient_id: str,
    scores: dict[str, float],
    policy: dict[str, Any] | None,
    llm: LLMClient | None = None,
) -> SingleAgentResult:
    policy = policy or load_policy()
    llm = llm or LLMClient()
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]

    current = "CP1"
    visited: list[str] = []
    trace: list[dict[str, Any]] = []
    controllers: list[dict[str, Any]] = []

    while current is not None:
        score = float(scores[current])
        c_thr = float(c_thrs.get(current, 0.5))
        a_thr = float(a_thrs.get(current, 0.5))
        rl_tool = risk_level(score, c_thr, a_thr)
        allowed_actions = allowed_actions_for_stage(current)
        evidence = build_single_agent_view(patient_id, current)
        evidence_references, missing_fields = evidence_contract_from_text(evidence)
        prior_controller_summary = summarize_coordinator_history(controllers)

        user_payload = json.dumps(
            {
                "patient_id": patient_id,
                "current_stage": current,
                "current_quantitative_state": {
                    "risk_score": score,
                    "risk_level": rl_tool,
                    "continue_threshold": c_thr,
                    "action_threshold": a_thr,
                },
                "allowed_actions": allowed_actions,
                "stage_bounded_evidence_view": evidence,
                "allowed_evidence_references": evidence_references,
                "allowed_missing_fields": missing_fields,
                "prior_controller_summary": prior_controller_summary,
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            ctrl_out = llm.single_agent_json(
                SINGLE_AGENT_SYSTEM,
                current,
                user_payload,
                allowed_actions,
                expected_risk_score=score,
                expected_risk_level=rl_tool,
                allowed_evidence_references=evidence_references,
                allowed_missing_fields=missing_fields,
            )
            controller_output_valid = True
            controller_validation_audit = validation_audit_record(
                True,
                llm.last_request_audit,
                grounded_output=True,
            )
        except LLMOutputError as exc:
            ctrl_out = {}
            controller_output_valid = False
            controller_validation_audit = validation_audit_record(
                False,
                llm.last_request_audit,
                exc,
                grounded_output=True,
            )
        controllers.append(
            {
                "stage": current,
                "output_valid": controller_output_valid,
                "validation_audit": controller_validation_audit,
                "output": ctrl_out,
            }
        )

        dec = validate_coordinator_proposal(current, score, policy, ctrl_out)
        visited.append(current)
        trace.append(
            {
                "ID": patient_id,
                "stage": current,
                "agent": "single_agent_controller",
                "risk_score": score,
                "risk_level": rl_tool,
                "allowed_actions": allowed_actions,
                "evidence_boundary": current,
                "controller_json": ctrl_out,
                "controller_output_valid": controller_output_valid,
                "controller_validation_audit": controller_validation_audit,
                "consensus_state": ctrl_out.get("consensus_state"),
                "key_conflicts": ctrl_out.get("key_conflicts"),
                "information_gap": ctrl_out.get("information_gap"),
                "controller_proposed_action": dec.coordinator_action,
                "fixed_threshold_action": dec.fixed_threshold_action,
                "fixed_threshold_next_stage": dec.fixed_threshold_next_stage,
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
            assigned_escalation = 1 if final_action in ESCALATION_ACTIONS else 0
            return SingleAgentResult(
                patient_id=patient_id,
                visited_stages=" -> ".join(visited),
                final_stage=current,
                final_score=score,
                final_action=final_action,
                assigned_escalation=assigned_escalation,
                trace=trace,
                controller_outputs=controllers,
            )

        resolved_next = action_to_next_stage(dec.final_action)
        if resolved_next is None:
            raise RuntimeError(f"Non-terminal action without next stage mapping: {dec.final_action}")
        current = resolved_next

    raise RuntimeError("single-agent pathway fell through without terminal")
