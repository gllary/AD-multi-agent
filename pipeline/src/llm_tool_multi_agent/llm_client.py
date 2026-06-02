# -*- coding: utf-8 -*-
"""
OpenAI-compatible chat completions for specialist and coordinator agents.
Falls back to deterministic structured stubs when no API key is available.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .config import DEFAULT_API_BASE, DEFAULT_MODEL
from .schemas import (
    REQUIRED_COORDINATOR_KEYS,
    REQUIRED_SPECIALIST_KEYS,
    coordinator_json_schema,
    specialist_json_schema,
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_json_payload(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _action_rank(action: str) -> int:
    ranks = {
        "observe_or_reassess": 0,
        "call_lab_agent": 1,
        "call_ecg_agent": 2,
        "call_echo_agent": 3,
        "direct_cta": 4,
        "urgent_transfer": 5,
    }
    return ranks.get(action, -1)


def _specialist_recommendation(role: str, score: float, rlevel: str) -> tuple[str, str, str]:
    rec = {
        "history": "call_lab_agent" if rlevel == "intermediate" else "observe_or_reassess",
        "examination": ("call_lab_agent" if score < 0.05 else "call_echo_agent") if rlevel == "intermediate" else "observe_or_reassess",
        "lab_context": "call_echo_agent" if rlevel == "intermediate" else "observe_or_reassess",
        "lab_biomarker": "call_ecg_agent" if rlevel == "intermediate" and score < 0.15 else "call_echo_agent",
        "laboratory": "call_ecg_agent" if rlevel == "intermediate" and score < 0.15 else "call_echo_agent",
        "ecg": "call_echo_agent",
        "echocardiography": "direct_cta" if rlevel != "low" else "observe_or_reassess",
    }.get(role, "observe_or_reassess")
    urgency = "routine"
    confidence = "medium"
    if rlevel == "high":
        rec = "urgent_transfer" if role == "echocardiography" and score >= 0.75 else "direct_cta"
        urgency = "immediate"
        confidence = "high"
    elif rlevel == "intermediate":
        urgency = "urgent"
    else:
        confidence = "high"
    return rec, urgency, confidence


def _stub_specialist(
    role: str,
    stage: str,
    score: float,
    rlevel: str,
) -> dict[str, Any]:
    rec, urgency, confidence = _specialist_recommendation(role, score, rlevel)
    return {
        "agent_role": role,
        "stage": stage,
        "risk_score_tool": float(score),
        "risk_level_tool": rlevel,
        "local_assessment": f"Deterministic stub assessment for {role}: tool risk is {rlevel} at score {score:.4f}.",
        "supporting_evidence": ["Tool signal supports escalation"] if rlevel != "low" else ["Tool signal remains in low-risk band"],
        "counter_evidence": [] if rlevel == "high" else ["No definitive single-feature diagnosis is available"],
        "missing_critical_data": ["Additional stage-specific correlation may refine certainty"],
        "recommended_next_action": rec,
        "urgency": urgency,
        "confidence": confidence,
        "why_not_stop_now": (
            "Current tool signal is not safely dismissive."
            if rlevel != "low"
            else "Stopping is acceptable because the tool signal is low."
        ),
        "why_not_escalate_now": (
            "Immediate escalation is not mandatory because the tool signal is not extreme."
            if rlevel != "high"
            else "Escalation is warranted now because the tool signal is high."
        ),
        "rationale_summary": f"{role} recommends {rec} with {confidence} confidence.",
    }


def _stub_coordinator(stage: str, specialist_blob: list[dict[str, Any]], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    outputs = [d.get("output", {}) for d in specialist_blob if isinstance(d, dict)]
    if not outputs and payload.get("current_specialist_outputs"):
        outputs = [x for x in payload.get("current_specialist_outputs", []) if isinstance(x, dict)]
    actions = [str(o.get("recommended_next_action", "")).strip() for o in outputs if o.get("recommended_next_action")]
    confidences = [str(o.get("confidence", "medium")) for o in outputs]
    risk_levels = [str(o.get("risk_level_tool", "")) for o in outputs if o.get("risk_level_tool")]
    specialist_summary = payload.get("specialist_summary", {}) if isinstance(payload.get("specialist_summary", {}), dict) else {}
    conflict_summary = specialist_summary.get("conflict_summary", []) or []
    info_gaps = specialist_summary.get("information_gaps", []) or []
    qstate = payload.get("current_quantitative_state", {}) if isinstance(payload.get("current_quantitative_state", {}), dict) else {}
    score = float(qstate.get("risk_score", 0.5) or 0.5)
    tool_level = str(qstate.get("risk_level", "")).strip()

    if not actions:
        proposed = "observe_or_reassess"
        consensus = "unresolved_uncertainty"
        conflicts = ["No specialist action recommendation available."]
        gaps = ["Need at least one valid specialist output."]
    elif len(set(actions)) == 1:
        proposed = actions[-1]
        if risk_levels and set(risk_levels) == {"low"}:
            consensus = "convergent_low_risk"
        elif "high" in risk_levels and len(set(risk_levels)) == 1:
            consensus = "convergent_high_risk"
        else:
            consensus = "unresolved_uncertainty"
        conflicts = list(conflict_summary)
        gaps = list(info_gaps)
    else:
        # Stage-aware deterministic aggregation that reads structured specialist inputs.
        unique_actions = sorted(set(actions), key=_action_rank)
        if "urgent_transfer" in unique_actions:
            proposed = "urgent_transfer"
        elif "direct_cta" in unique_actions:
            proposed = "direct_cta"
        elif stage == "CP1":
            if "call_lab_agent" in unique_actions and "call_echo_agent" in unique_actions:
                proposed = "call_lab_agent" if score < 0.05 else "call_echo_agent"
            else:
                proposed = max(unique_actions, key=_action_rank)
        elif stage == "CP2":
            if "call_echo_agent" in unique_actions and "observe_or_reassess" in unique_actions:
                proposed = "call_echo_agent" if specialist_summary.get("consensus_hint") == "action_conflict" or score >= 0.01 else "observe_or_reassess"
            elif "call_ecg_agent" in unique_actions and "observe_or_reassess" in unique_actions:
                proposed = "call_ecg_agent" if score >= 0.01 else "observe_or_reassess"
            else:
                proposed = max(unique_actions, key=_action_rank)
        elif stage == "CP3":
            proposed = "call_echo_agent" if "call_echo_agent" in unique_actions else max(unique_actions, key=_action_rank)
        else:
            proposed = max(unique_actions, key=_action_rank)
        consensus = "mixed_risk"
        conflicts = list(conflict_summary) or [f"Specialists disagree on next action: {sorted(set(actions))}"]
        gaps = list(info_gaps) or ["Resolve disagreement with safety-conservative coordination."]
    safety_concern = "moderate" if consensus in {"mixed_risk", "unresolved_uncertainty"} else "mild"
    if "high" in risk_levels or tool_level == "high":
        safety_concern = "severe"
    confidence = "medium" if "low" in confidences else "high"
    return {
        "current_stage": stage,
        "consensus_state": consensus,
        "key_conflicts": conflicts,
        "information_gap": gaps,
        "proposed_action": proposed,
        "confidence": confidence,
        "safety_concern": safety_concern,
        "why_this_action_over_alternatives": (
            f"Stub coordinator integrated {len(outputs)} specialist opinions at {stage} and selected {proposed} "
            f"after stage-aware conflict resolution over alternatives {sorted(set(actions)) if actions else []}."
        ),
        "coordinator_summary": f"Stub coordinator reviewed {len(outputs)} specialist outputs at {stage} and proposes {proposed}.",
    }


def _stub_single_agent(stage: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    qstate = payload.get("current_quantitative_state", {}) if isinstance(payload.get("current_quantitative_state", {}), dict) else {}
    score = float(qstate.get("risk_score", 0.5) or 0.5)
    tool_level = str(qstate.get("risk_level", "intermediate")).strip() or "intermediate"
    pseudo_role = {
        "CP1": "history",
        "CP2": "laboratory",
        "CP3": "ecg",
        "CP4": "echocardiography",
    }.get(stage, "history")
    action, _urgency, confidence = _specialist_recommendation(pseudo_role, score, tool_level)
    consensus = (
        "convergent_low_risk" if tool_level == "low"
        else "convergent_high_risk" if tool_level == "high"
        else "unresolved_uncertainty"
    )
    safety_concern = "mild" if tool_level == "low" else "severe" if tool_level == "high" else "moderate"
    return {
        "current_stage": stage,
        "consensus_state": consensus,
        "key_conflicts": [],
        "information_gap": ["Single-agent baseline lacks specialist disagreement modeling."] if tool_level == "intermediate" else [],
        "proposed_action": action,
        "confidence": confidence,
        "safety_concern": safety_concern,
        "why_this_action_over_alternatives": f"Single-agent baseline selected {action} from the stage-bounded evidence view at {stage}.",
        "coordinator_summary": f"Single-agent baseline proposes {action} at {stage}.",
    }


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.api_base = (api_base or os.environ.get("LLM_API_BASE", DEFAULT_API_BASE)).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    @property
    def is_live(self) -> bool:
        return bool(self.api_key)

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        if not self.is_live:
            return ""
        url = f"{self.api_base}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            return f"__ERROR__:{e}"
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return f"__ERROR__:unexpected_response:{payload!r}"

    def specialist_json(
        self,
        system_prompt: str,
        role_key: str,
        stage: str,
        user_payload: str,
        tool_score: float,
        tool_level: str,
    ) -> dict[str, Any]:
        schema = json.dumps(specialist_json_schema(), ensure_ascii=False)
        user = f"{user_payload}\n\nJSON schema: {schema}\n"
        raw = self._chat(system_prompt, user)
        if not raw or raw.startswith("__ERROR__"):
            return _stub_specialist(role_key, stage, tool_score, tool_level)
        parsed = _extract_json_object(raw)
        if not parsed or any(k not in parsed for k in REQUIRED_SPECIALIST_KEYS):
            return _stub_specialist(role_key, stage, tool_score, tool_level)
        return parsed

    def coordinator_json(
        self,
        system_prompt: str,
        stage: str,
        user_payload: str,
        allowed_actions: list[str],
        specialist_blob: list[dict[str, Any]],
    ) -> dict[str, Any]:
        schema = json.dumps(coordinator_json_schema(), ensure_ascii=False)
        user = (
            f"{user_payload}\n\nAllowed proposed_action values: {allowed_actions}\n\nJSON schema: {schema}\n"
        )
        raw = self._chat(system_prompt, user)
        payload = _parse_json_payload(user_payload)
        if not raw or raw.startswith("__ERROR__"):
            return _stub_coordinator(stage, specialist_blob, payload)
        parsed = _extract_json_object(raw)
        if not parsed or any(k not in parsed for k in REQUIRED_COORDINATOR_KEYS):
            return _stub_coordinator(stage, specialist_blob, payload)
        return parsed

    def single_agent_json(
        self,
        system_prompt: str,
        stage: str,
        user_payload: str,
        allowed_actions: list[str],
    ) -> dict[str, Any]:
        schema = json.dumps(coordinator_json_schema(), ensure_ascii=False)
        user = (
            f"{user_payload}\n\nAllowed proposed_action values: {allowed_actions}\n\nJSON schema: {schema}\n"
        )
        raw = self._chat(system_prompt, user)
        payload = _parse_json_payload(user_payload)
        if not raw or raw.startswith("__ERROR__"):
            return _stub_single_agent(stage, payload)
        parsed = _extract_json_object(raw)
        if not parsed or any(k not in parsed for k in REQUIRED_COORDINATOR_KEYS):
            return _stub_single_agent(stage, payload)
        return parsed
