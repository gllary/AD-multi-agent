# -*- coding: utf-8 -*-
"""Strict JSON schemas for specialist, coordinator, and safety records."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


AgentRole = Literal["history", "examination", "lab_context", "lab_biomarker", "ecg", "echocardiography"]
StageName = Literal["CP1", "CP2", "CP3", "CP4"]
RiskLevel = Literal["low", "intermediate", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]
UrgencyLevel = Literal["routine", "urgent", "immediate"]
ConsensusState = Literal[
    "convergent_low_risk",
    "convergent_high_risk",
    "mixed_risk",
    "unresolved_uncertainty",
]
SafetyConcern = Literal["none", "mild", "moderate", "severe"]
SafetyRiskLevel = Literal["low", "moderate", "high"]

ACTION_VALUES = (
    "observe_or_reassess",
    "call_lab_agent",
    "call_ecg_agent",
    "call_echo_agent",
    "direct_cta",
    "urgent_transfer",
)
STAGE_VALUES = ("CP1", "CP2", "CP3", "CP4")
RISK_LEVEL_VALUES = ("low", "intermediate", "high")
CONFIDENCE_VALUES = ("low", "medium", "high")
URGENCY_VALUES = ("routine", "urgent", "immediate")
CONSENSUS_STATE_VALUES = (
    "convergent_low_risk",
    "convergent_high_risk",
    "mixed_risk",
    "unresolved_uncertainty",
)
SAFETY_CONCERN_VALUES = ("none", "mild", "moderate", "severe")
SAFETY_RISK_LEVEL_VALUES = ("low", "moderate", "high")
NEXT_STAGE_VALUES = ("CP2", "CP3", "CP4")


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _next_stage_or_null_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string", "enum": list(NEXT_STAGE_VALUES)},
            {"type": "null"},
        ]
    }


class SpecialistOutput(TypedDict, total=False):
    agent_role: AgentRole
    stage: StageName
    risk_score_tool: float
    risk_level_tool: RiskLevel
    local_assessment: str
    supporting_evidence: list[str]
    counter_evidence: list[str]
    missing_critical_data: list[str]
    recommended_next_action: str
    urgency: UrgencyLevel
    confidence: ConfidenceLevel
    why_not_stop_now: str
    why_not_escalate_now: str
    rationale_summary: str


class CoordinatorOutput(TypedDict, total=False):
    current_stage: StageName
    consensus_state: ConsensusState
    key_conflicts: list[str]
    information_gap: list[str]
    proposed_action: str
    confidence: ConfidenceLevel
    safety_concern: SafetyConcern
    why_this_action_over_alternatives: str
    coordinator_summary: str


class SafetyReview(TypedDict, total=False):
    current_stage: StageName
    coordinator_action: str
    allowed_actions: list[str]
    is_action_allowed: bool
    final_action: str
    final_next_stage: str | None
    override_reason: str
    safety_risk_level: SafetyRiskLevel
    policy_basis: list[str]
    fixed_threshold_action: str
    fixed_threshold_next_stage: str | None


REQUIRED_SPECIALIST_KEYS = (
    "agent_role",
    "stage",
    "risk_score_tool",
    "risk_level_tool",
    "local_assessment",
    "supporting_evidence",
    "counter_evidence",
    "missing_critical_data",
    "recommended_next_action",
    "urgency",
    "confidence",
    "why_not_stop_now",
    "why_not_escalate_now",
    "rationale_summary",
)

REQUIRED_COORDINATOR_KEYS = (
    "current_stage",
    "consensus_state",
    "key_conflicts",
    "information_gap",
    "proposed_action",
    "confidence",
    "safety_concern",
    "why_this_action_over_alternatives",
    "coordinator_summary",
)

REQUIRED_SAFETY_KEYS = (
    "current_stage",
    "coordinator_action",
    "allowed_actions",
    "is_action_allowed",
    "final_action",
    "final_next_stage",
    "override_reason",
    "safety_risk_level",
    "policy_basis",
    "fixed_threshold_action",
    "fixed_threshold_next_stage",
)


def specialist_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_SPECIALIST_KEYS),
        "properties": {
            "agent_role": {
                "type": "string",
                "enum": ["history", "examination", "lab_context", "lab_biomarker", "ecg", "echocardiography"],
            },
            "stage": {"type": "string", "enum": list(STAGE_VALUES)},
            "risk_score_tool": {"type": "number", "minimum": 0, "maximum": 1},
            "risk_level_tool": {
                "type": "string",
                "enum": list(RISK_LEVEL_VALUES),
            },
            "local_assessment": {"type": "string"},
            "supporting_evidence": _string_array_schema(),
            "counter_evidence": _string_array_schema(),
            "missing_critical_data": _string_array_schema(),
            "recommended_next_action": {"type": "string", "enum": list(ACTION_VALUES)},
            "urgency": {"type": "string", "enum": list(URGENCY_VALUES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "why_not_stop_now": {"type": "string"},
            "why_not_escalate_now": {"type": "string"},
            "rationale_summary": {"type": "string"},
        },
    }


def coordinator_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_COORDINATOR_KEYS),
        "properties": {
            "current_stage": {"type": "string", "enum": list(STAGE_VALUES)},
            "consensus_state": {
                "type": "string",
                "enum": list(CONSENSUS_STATE_VALUES),
            },
            "key_conflicts": _string_array_schema(),
            "information_gap": _string_array_schema(),
            "proposed_action": {"type": "string", "enum": list(ACTION_VALUES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "safety_concern": {"type": "string", "enum": list(SAFETY_CONCERN_VALUES)},
            "why_this_action_over_alternatives": {"type": "string"},
            "coordinator_summary": {"type": "string"},
        },
    }


def safety_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_SAFETY_KEYS),
        "properties": {
            "current_stage": {"type": "string", "enum": list(STAGE_VALUES)},
            "coordinator_action": {"type": "string"},
            "allowed_actions": {"type": "array", "items": {"type": "string", "enum": list(ACTION_VALUES)}},
            "is_action_allowed": {"type": "boolean"},
            "final_action": {"type": "string", "enum": list(ACTION_VALUES)},
            "final_next_stage": _next_stage_or_null_schema(),
            "override_reason": {"type": "string"},
            "safety_risk_level": {"type": "string", "enum": list(SAFETY_RISK_LEVEL_VALUES)},
            "policy_basis": _string_array_schema(),
            "fixed_threshold_action": {"type": "string", "enum": list(ACTION_VALUES)},
            "fixed_threshold_next_stage": _next_stage_or_null_schema(),
        },
    }
