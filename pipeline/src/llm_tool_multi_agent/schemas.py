# -*- coding: utf-8 -*-
"""Structured JSON schemas for specialist, coordinator, and safety records."""

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
    override_reason: str
    safety_risk_level: SafetyRiskLevel
    policy_basis: list[str]


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
    "override_reason",
    "safety_risk_level",
    "policy_basis",
)


def specialist_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(REQUIRED_SPECIALIST_KEYS),
        "properties": {
            "agent_role": {
                "type": "string",
                "enum": ["history", "examination", "lab_context", "lab_biomarker", "ecg", "echocardiography"],
            },
            "stage": {"type": "string", "enum": ["CP1", "CP2", "CP3", "CP4"]},
            "risk_score_tool": {"type": "number"},
            "risk_level_tool": {
                "type": "string",
                "enum": ["low", "intermediate", "high"],
            },
            "local_assessment": {"type": "string"},
            "supporting_evidence": {"type": "array", "items": {"type": "string"}},
            "counter_evidence": {"type": "array", "items": {"type": "string"}},
            "missing_critical_data": {"type": "array", "items": {"type": "string"}},
            "recommended_next_action": {"type": "string"},
            "urgency": {"type": "string", "enum": ["routine", "urgent", "immediate"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "why_not_stop_now": {"type": "string"},
            "why_not_escalate_now": {"type": "string"},
            "rationale_summary": {"type": "string"},
        },
    }


def coordinator_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(REQUIRED_COORDINATOR_KEYS),
        "properties": {
            "current_stage": {"type": "string", "enum": ["CP1", "CP2", "CP3", "CP4"]},
            "consensus_state": {
                "type": "string",
                "enum": [
                    "convergent_low_risk",
                    "convergent_high_risk",
                    "mixed_risk",
                    "unresolved_uncertainty",
                ],
            },
            "key_conflicts": {"type": "array", "items": {"type": "string"}},
            "information_gap": {"type": "array", "items": {"type": "string"}},
            "proposed_action": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "safety_concern": {"type": "string", "enum": ["none", "mild", "moderate", "severe"]},
            "why_this_action_over_alternatives": {"type": "string"},
            "coordinator_summary": {"type": "string"},
        },
    }


def safety_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(REQUIRED_SAFETY_KEYS),
        "properties": {
            "current_stage": {"type": "string", "enum": ["CP1", "CP2", "CP3", "CP4"]},
            "coordinator_action": {"type": "string"},
            "allowed_actions": {"type": "array", "items": {"type": "string"}},
            "is_action_allowed": {"type": "boolean"},
            "final_action": {"type": "string"},
            "override_reason": {"type": "string"},
            "safety_risk_level": {"type": "string", "enum": ["low", "moderate", "high"]},
            "policy_basis": {"type": "array", "items": {"type": "string"}},
        },
    }
