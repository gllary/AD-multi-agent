# -*- coding: utf-8 -*-
"""
Safety governance for multi-agent pathway actions.

The coordinator proposes an action, and the safety layer either allows it or overrides it
using explicit pathway constraints and threshold-derived clinical guardrails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .quantitative_tools import risk_level

POSITIVE_ACTIONS = frozenset({"direct_cta", "urgent_transfer"})
TERMINAL_ACTIONS = frozenset({"observe_or_reassess", "direct_cta", "urgent_transfer"})

CLINICAL_ACTIONS = frozenset(
    {
        "observe_or_reassess",
        "call_lab_agent",
        "call_ecg_agent",
        "call_echo_agent",
        "direct_cta",
        "urgent_transfer",
    }
)

STAGE_ALLOWED_ACTIONS = {
    "CP1": frozenset({"observe_or_reassess", "call_lab_agent", "call_echo_agent", "direct_cta"}),
    "CP2": frozenset({"observe_or_reassess", "call_ecg_agent", "call_echo_agent", "direct_cta"}),
    "CP3": frozenset({"observe_or_reassess", "call_echo_agent", "direct_cta"}),
    "CP4": frozenset({"observe_or_reassess", "direct_cta", "urgent_transfer"}),
}

ACTION_TO_NEXT_STAGE = {
    "call_lab_agent": "CP2",
    "call_ecg_agent": "CP3",
    "call_echo_agent": "CP4",
}


def load_policy_dict(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def route_cp1(rl: str, score: float) -> tuple[str | None, str]:
    if rl == "low":
        return None, "observe_or_reassess"
    if rl == "intermediate":
        return "CP2", "call_lab_agent"
    if score >= 0.9:
        return None, "direct_cta"
    return "CP4", "call_echo_agent"


def route_cp2(rl: str, score: float) -> tuple[str | None, str]:
    if rl == "low":
        return None, "observe_or_reassess"
    if rl == "intermediate":
        return ("CP4", "call_echo_agent") if score >= 0.15 else ("CP3", "call_ecg_agent")
    if score >= 0.9:
        return None, "direct_cta"
    return "CP4", "call_echo_agent"


def route_cp3(rl: str, score: float) -> tuple[str | None, str]:
    if rl == "low":
        return None, "observe_or_reassess"
    if score >= 0.75:
        return None, "direct_cta"
    return "CP4", "call_echo_agent"


def route_cp4(rl: str, _score: float) -> tuple[str | None, str]:
    if rl == "low":
        return None, "observe_or_reassess"
    if rl == "intermediate":
        return None, "direct_cta"
    return None, "urgent_transfer"


ROUTERS = {
    "CP1": route_cp1,
    "CP2": route_cp2,
    "CP3": route_cp3,
    "CP4": route_cp4,
}


@dataclass
class SafetyDecision:
    current_stage: str
    coordinator_action: str
    allowed_actions: list[str]
    is_action_allowed: bool
    final_action: str
    final_next_stage: str | None
    override_reason: str
    safety_risk_level: str
    policy_basis: list[str]
    canonical_action: str
    canonical_next_stage: str | None


def canonical_route(
    current: str,
    score: float,
    continue_thresholds: dict[str, float],
    action_thresholds: dict[str, float],
) -> tuple[str | None, str]:
    c_thr = float(continue_thresholds.get(current, 0.5))
    a_thr = float(action_thresholds.get(current, 0.5))
    rl = risk_level(score, c_thr, a_thr)
    return ROUTERS[current](rl, score)


def allowed_actions_for_stage(stage: str) -> list[str]:
    return sorted(STAGE_ALLOWED_ACTIONS[stage])


def action_to_next_stage(action: str) -> str | None:
    return ACTION_TO_NEXT_STAGE.get(action)


def _risk_to_safety_level(rl: str) -> str:
    if rl == "high":
        return "high"
    if rl == "intermediate":
        return "moderate"
    return "low"


def validate_coordinator_proposal(
    current_stage: str,
    risk_score: float,
    policy: dict[str, Any],
    coordinator_json: dict[str, Any] | None,
) -> SafetyDecision:
    c_thrs = policy["continue_thresholds"]
    a_thrs = policy["action_thresholds"]
    canonical_next_stage, canonical_action = canonical_route(current_stage, risk_score, c_thrs, a_thrs)
    c_thr = float(c_thrs.get(current_stage, 0.5))
    a_thr = float(a_thrs.get(current_stage, 0.5))
    rl = risk_level(risk_score, c_thr, a_thr)
    allowed = allowed_actions_for_stage(current_stage)

    coordinator_action = ""
    if coordinator_json and coordinator_json.get("proposed_action"):
        coordinator_action = str(coordinator_json["proposed_action"]).strip()

    if coordinator_action not in CLINICAL_ACTIONS:
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action=canonical_action,
            final_next_stage=canonical_next_stage,
            override_reason="invalid_or_missing_action_reverted_to_canonical_route",
            safety_risk_level=_risk_to_safety_level(rl),
            policy_basis=["invalid_action", f"risk_level={rl}", "canonical_threshold_route"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    if coordinator_action not in STAGE_ALLOWED_ACTIONS[current_stage]:
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action=canonical_action,
            final_next_stage=canonical_next_stage,
            override_reason="stage_illegal_action_reverted_to_canonical_route",
            safety_risk_level=_risk_to_safety_level(rl),
            policy_basis=["illegal_transition", f"risk_level={rl}", "canonical_threshold_route"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    policy_basis = [f"risk_level={rl}"]

    if rl == "high" and coordinator_action in {"observe_or_reassess", "call_lab_agent", "call_ecg_agent"}:
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action=canonical_action,
            final_next_stage=canonical_next_stage,
            override_reason="high_risk_state_blocked_unsafe_deescalation",
            safety_risk_level="high",
            policy_basis=policy_basis + ["high_risk_guardrail", "canonical_threshold_route"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    # Intermediate-risk rescue: when the threshold route still asks for more
    # evidence, specialist consensus alone cannot prematurely terminate the
    # pathway without a high-confidence reassessment state.
    # Without this rescue the safety layer accepted observe as "within bounds"
    # for intermediate risk. We now force a fallback to the canonical
    # continuation action whenever the coordinator wants to terminate early
    # on a non-terminal stage with intermediate risk. The canonical action
    # keeps the patient inside the pathway, preserves recall, and respects
    # the threshold policy fixed on the development cohort.
    if (
        current_stage in {"CP1", "CP2"}
        and rl == "intermediate"
        and coordinator_action == "observe_or_reassess"
        and canonical_action != "observe_or_reassess"
    ):
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action=canonical_action,
            final_next_stage=canonical_next_stage,
            override_reason="intermediate_risk_observe_rescued_to_canonical_continuation",
            safety_risk_level=_risk_to_safety_level(rl),
            policy_basis=policy_basis + ["intermediate_risk_rescue", "canonical_threshold_route"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    if current_stage == "CP4" and rl != "low" and coordinator_action == "observe_or_reassess":
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action=canonical_action,
            final_next_stage=canonical_next_stage,
            override_reason="terminal_stage_nonlow_risk_cannot_observe",
            safety_risk_level=_risk_to_safety_level(rl),
            policy_basis=policy_basis + ["terminal_stage_guardrail", "canonical_threshold_route"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    if current_stage == "CP4" and rl == "high" and coordinator_action != "urgent_transfer":
        return SafetyDecision(
            current_stage=current_stage,
            coordinator_action=coordinator_action,
            allowed_actions=allowed,
            is_action_allowed=False,
            final_action="urgent_transfer",
            final_next_stage=None,
            override_reason="terminal_high_risk_forced_urgent_transfer",
            safety_risk_level="high",
            policy_basis=policy_basis + ["terminal_high_risk_guardrail"],
            canonical_action=canonical_action,
            canonical_next_stage=canonical_next_stage,
        )

    return SafetyDecision(
        current_stage=current_stage,
        coordinator_action=coordinator_action,
        allowed_actions=allowed,
        is_action_allowed=True,
        final_action=coordinator_action,
        final_next_stage=action_to_next_stage(coordinator_action),
        override_reason="",
        safety_risk_level=_risk_to_safety_level(rl),
        policy_basis=policy_basis + ["coordinator_action_within_safety_bounds"],
        canonical_action=canonical_action,
        canonical_next_stage=canonical_next_stage,
    )
