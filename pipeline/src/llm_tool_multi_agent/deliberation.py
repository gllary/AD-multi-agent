# -*- coding: utf-8 -*-
"""Helpers for summarizing specialist and coordinator deliberation history."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_specialist_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "n_specialist_steps": 0,
            "recent_recommendations": [],
            "consensus_hint": "no_history",
            "conflict_summary": [],
            "information_gaps": [],
        }

    outputs = [item.get("output", {}) for item in history]
    actions = [str(o.get("recommended_next_action", "")).strip() for o in outputs if o.get("recommended_next_action")]
    confidences = [str(o.get("confidence", "")).strip() for o in outputs if o.get("confidence")]
    urgencies = [str(o.get("urgency", "")).strip() for o in outputs if o.get("urgency")]
    gaps: list[str] = []
    for o in outputs:
        for g in o.get("missing_critical_data", []) or []:
            gs = str(g).strip()
            if gs:
                gaps.append(gs)
    gap_counts = Counter(gaps)

    if not actions:
        consensus = "no_recommendation"
        conflicts = ["No specialist recommendation available."]
    elif len(set(actions)) == 1:
        consensus = "stable_action"
        conflicts = []
    else:
        consensus = "action_conflict"
        conflicts = [f"Specialist actions disagree: {sorted(set(actions))}"]

    if len(set(confidences)) > 1:
        conflicts.append(f"Confidence levels vary across specialists: {sorted(set(confidences))}")
    if len(set(urgencies)) > 1:
        conflicts.append(f"Urgency levels vary across specialists: {sorted(set(urgencies))}")

    return {
        "n_specialist_steps": len(history),
        "recent_recommendations": actions[-4:],
        "consensus_hint": consensus,
        "conflict_summary": conflicts,
        "information_gaps": [g for g, _ in gap_counts.most_common(5)],
    }


def summarize_coordinator_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "n_coordinator_steps": 0,
            "recent_actions": [],
            "recent_consensus_states": [],
        }

    outputs = [item.get("output", {}) for item in history]
    actions = [str(o.get("proposed_action", "")).strip() for o in outputs if o.get("proposed_action")]
    states = [str(o.get("consensus_state", "")).strip() for o in outputs if o.get("consensus_state")]
    return {
        "n_coordinator_steps": len(history),
        "recent_actions": actions[-4:],
        "recent_consensus_states": states[-4:],
    }
