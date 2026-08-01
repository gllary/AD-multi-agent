"""Tests for schema, risk-state, and evidence-grounding validation."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from llm_tool_multi_agent.llm_client import (
    LLMClient,
    LLMOutputError,
    _extract_json_object,
    _validate_specialist,
)


def valid_specialist_payload() -> dict[str, object]:
    return {
        "agent_role": "history",
        "stage": "CP1",
        "risk_score_tool": 0.125,
        "risk_level_tool": "intermediate",
        "local_assessment": "Bounded assessment.",
        "supporting_evidence": ["Age=61"],
        "counter_evidence": [],
        "missing_critical_data": ["Sex"],
        "recommended_next_action": "call_lab_agent",
        "urgency": "urgent",
        "confidence": "medium",
        "why_not_stop_now": "Information remains incomplete.",
        "why_not_escalate_now": "No high-risk field is present.",
        "rationale_summary": "Continue staged evaluation.",
    }


class SequencedClient(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(max_attempts=len(responses))
        self.responses = responses
        self.calls = 0

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        del system, user, temperature
        response = self.responses[self.calls]
        self.calls += 1
        return json.dumps(response)


class LLMValidationTest(unittest.TestCase):
    def validate(self, payload: dict[str, object]) -> dict[str, object]:
        return _validate_specialist(
            payload,
            role="history",
            stage="CP1",
            allowed_actions=["observe_or_reassess", "call_lab_agent", "direct_cta"],
            expected_risk_score=0.125,
            expected_risk_level="intermediate",
            allowed_evidence_references=["Age=61"],
            allowed_missing_fields=["Sex"],
        )

    def test_accepts_grounded_payload(self) -> None:
        payload = valid_specialist_payload()
        self.assertIs(self.validate(payload), payload)

    def test_rejects_tool_risk_mismatch(self) -> None:
        payload = valid_specialist_payload()
        payload["risk_score_tool"] = 0.7
        with self.assertRaisesRegex(LLMOutputError, "risk score"):
            self.validate(payload)

    def test_rejects_unavailable_evidence_reference(self) -> None:
        payload = valid_specialist_payload()
        payload["supporting_evidence"] = ["exam__pulse_deficit=1"]
        with self.assertRaisesRegex(LLMOutputError, "outside the role-bounded packet"):
            self.validate(payload)

    def test_rejects_missing_field_not_marked_unknown(self) -> None:
        payload = valid_specialist_payload()
        payload["missing_critical_data"] = ["D_D_log"]
        with self.assertRaisesRegex(LLMOutputError, "not marked unknown"):
            self.validate(payload)

    def test_rejects_additional_schema_property(self) -> None:
        payload = valid_specialist_payload()
        payload["diagnosis"] = "AD"
        with self.assertRaisesRegex(LLMOutputError, "JSON schema"):
            self.validate(payload)

    def test_requires_one_bare_json_object(self) -> None:
        with self.assertRaisesRegex(LLMOutputError, "exactly one"):
            _extract_json_object('prefix {"current_stage": "CP1"}')

    @patch("llm_tool_multi_agent.llm_client.time.sleep", return_value=None)
    def test_retries_after_semantic_validation_failure(self, _sleep: object) -> None:
        invalid = valid_specialist_payload()
        invalid["risk_score_tool"] = 0.7
        client = SequencedClient([invalid, valid_specialist_payload()])

        result = client.specialist_json(
            "system",
            "history",
            "CP1",
            "payload",
            ["observe_or_reassess", "call_lab_agent", "direct_cta"],
            expected_risk_score=0.125,
            expected_risk_level="intermediate",
            allowed_evidence_references=["Age=61"],
            allowed_missing_fields=["Sex"],
        )

        self.assertEqual(result["risk_score_tool"], 0.125)
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
