"""OpenAI-compatible client with strict structured-output validation."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .config import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
)
from .schemas import (
    coordinator_json_schema,
    single_agent_json_schema,
    specialist_json_schema,
)


class LLMOutputError(RuntimeError):
    """Raised after retries fail or a response violates the frozen contract."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown_error",
        attempts: int | None = None,
        attempt_errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts
        self.attempt_errors = list(attempt_errors or [])


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise LLMOutputError(
        "Response must contain exactly one valid JSON object",
        category="parser_invalid",
    )


def _validate_json_schema(
    payload: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as exc:
        raise LLMOutputError(
            f"{label} output violated the frozen JSON schema: {exc.message}",
            category="schema_invalid",
        ) from exc


def _validate_specialist(
    payload: dict[str, Any],
    role: str,
    stage: str,
    allowed_actions: list[str],
    expected_risk_score: float,
    expected_risk_level: str,
    allowed_evidence_references: list[str],
    allowed_missing_fields: list[str],
) -> dict[str, Any]:
    _validate_json_schema(payload, specialist_json_schema(), "Specialist")
    if payload["agent_role"] != role or payload["stage"] != stage:
        raise LLMOutputError(
            "Specialist role or stage does not match the request",
            category="role_stage_mismatch",
        )
    if payload["recommended_next_action"] not in allowed_actions:
        raise LLMOutputError(
            "Specialist proposed an action outside the stage-legal set",
            category="stage_illegal_action",
        )

    expected_rounded_score = float(f"{expected_risk_score:.6f}")
    if not math.isclose(
        float(payload["risk_score_tool"]),
        expected_rounded_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise LLMOutputError(
            "Specialist risk score does not match the tool-provided score",
            category="risk_score_mismatch",
        )
    if payload["risk_level_tool"] != expected_risk_level:
        raise LLMOutputError(
            "Specialist risk level does not match the tool-provided level",
            category="risk_level_mismatch",
        )

    evidence_contract = set(allowed_evidence_references)
    for key in ("supporting_evidence", "counter_evidence"):
        invalid = [item for item in payload[key] if item not in evidence_contract]
        if invalid:
            raise LLMOutputError(
                f"Specialist {key} contains evidence outside the role-bounded packet: {invalid}",
                category="evidence_reference_mismatch",
            )

    missing_contract = set(allowed_missing_fields)
    invalid_missing = [
        item for item in payload["missing_critical_data"] if item not in missing_contract
    ]
    if invalid_missing:
        raise LLMOutputError(
            "Specialist missing_critical_data contains fields not marked unknown in "
            f"the role-bounded packet: {invalid_missing}",
            category="missing_field_mismatch",
        )
    return payload


def _validate_controller(
    payload: dict[str, Any],
    stage: str,
    allowed_actions: list[str],
) -> dict[str, Any]:
    _validate_json_schema(payload, coordinator_json_schema(), "Controller")
    if payload["current_stage"] != stage:
        raise LLMOutputError(
            "Controller stage does not match the request",
            category="role_stage_mismatch",
        )
    if payload["proposed_action"] not in allowed_actions:
        raise LLMOutputError(
            "Controller proposed an action outside the stage-legal set",
            category="stage_illegal_action",
        )
    return payload


def _validate_single_agent(
    payload: dict[str, Any],
    stage: str,
    allowed_actions: list[str],
    expected_risk_score: float,
    expected_risk_level: str,
    allowed_evidence_references: list[str],
    allowed_missing_fields: list[str],
) -> dict[str, Any]:
    _validate_json_schema(payload, single_agent_json_schema(), "Single-agent")
    if payload["current_stage"] != stage:
        raise LLMOutputError(
            "Single-agent stage does not match the request",
            category="role_stage_mismatch",
        )
    if payload["proposed_action"] not in allowed_actions:
        raise LLMOutputError(
            "Single-agent proposed an action outside the stage-legal set",
            category="stage_illegal_action",
        )

    expected_rounded_score = float(f"{expected_risk_score:.6f}")
    if not math.isclose(
        float(payload["risk_score_tool"]),
        expected_rounded_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise LLMOutputError(
            "Single-agent risk score does not match the tool-provided score",
            category="risk_score_mismatch",
        )
    if payload["risk_level_tool"] != expected_risk_level:
        raise LLMOutputError(
            "Single-agent risk level does not match the tool-provided level",
            category="risk_level_mismatch",
        )

    evidence_contract = set(allowed_evidence_references)
    for key in ("supporting_evidence", "counter_evidence"):
        invalid = [item for item in payload[key] if item not in evidence_contract]
        if invalid:
            raise LLMOutputError(
                f"Single-agent {key} contains evidence outside the stage-bounded packet: {invalid}",
                category="evidence_reference_mismatch",
            )

    missing_contract = set(allowed_missing_fields)
    invalid_missing = [
        item for item in payload["missing_critical_data"] if item not in missing_contract
    ]
    if invalid_missing:
        raise LLMOutputError(
            "Single-agent missing_critical_data contains fields not marked unknown in "
            f"the stage-bounded packet: {invalid_missing}",
            category="missing_field_mismatch",
        )
    return payload


def _exception_category(exc: Exception) -> str:
    if isinstance(exc, LLMOutputError):
        return exc.category
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        return "connection_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "response_json_invalid"
    return "unknown_error"


def validation_audit_record(
    output_valid: bool,
    request_audit: dict[str, Any] | None,
    error: LLMOutputError | None = None,
    *,
    grounded_output: bool,
) -> dict[str, Any]:
    """Build a PHI-free record of the output-contract checks performed."""
    request_audit = request_audit or {}
    category = error.category if error is not None else None
    checks: dict[str, bool | None] = {
        "parser_valid": None,
        "schema_valid": None,
        "role_stage_valid": None,
        "risk_state_valid": None,
        "evidence_references_valid": None,
        "missing_fields_valid": None,
        "proposed_action_valid": None,
    }

    if output_valid:
        checks.update(
            {
                "parser_valid": True,
                "schema_valid": True,
                "role_stage_valid": True,
                "proposed_action_valid": True,
            }
        )
        if grounded_output:
            checks.update(
                {
                    "risk_state_valid": True,
                    "evidence_references_valid": True,
                    "missing_fields_valid": True,
                }
            )
    elif category == "parser_invalid":
        checks["parser_valid"] = False
    elif category == "schema_invalid":
        checks.update({"parser_valid": True, "schema_valid": False})
    elif category == "role_stage_mismatch":
        checks.update(
            {"parser_valid": True, "schema_valid": True, "role_stage_valid": False}
        )
    elif category == "stage_illegal_action":
        checks.update(
            {
                "parser_valid": True,
                "schema_valid": True,
                "role_stage_valid": True,
                "proposed_action_valid": False,
            }
        )
    elif category in {"risk_score_mismatch", "risk_level_mismatch"}:
        checks.update(
            {
                "parser_valid": True,
                "schema_valid": True,
                "role_stage_valid": True,
                "proposed_action_valid": True,
                "risk_state_valid": False,
            }
        )
    elif category == "evidence_reference_mismatch":
        checks.update(
            {
                "parser_valid": True,
                "schema_valid": True,
                "role_stage_valid": True,
                "proposed_action_valid": True,
                "risk_state_valid": True,
                "evidence_references_valid": False,
            }
        )
    elif category == "missing_field_mismatch":
        checks.update(
            {
                "parser_valid": True,
                "schema_valid": True,
                "role_stage_valid": True,
                "proposed_action_valid": True,
                "risk_state_valid": True,
                "evidence_references_valid": True,
                "missing_fields_valid": False,
            }
        )

    return {
        "output_valid": output_valid,
        "attempt_count": request_audit.get("attempt_count", error.attempts if error else None),
        "attempt_failure_categories": request_audit.get(
            "attempt_failure_categories",
            error.attempt_errors if error else [],
        ),
        "final_failure_category": category,
        **checks,
    }


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int | None = None,
        max_attempts: int | None = None,
    ):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.api_base = (
            api_base or os.environ.get("LLM_API_BASE", DEFAULT_API_BASE)
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.timeout_seconds = int(
            timeout_seconds
            or os.environ.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.max_attempts = int(
            max_attempts or os.environ.get("LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
        )
        self.last_request_audit: dict[str, Any] = {}

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMOutputError(
                "Unexpected chat-completion response",
                category="response_contract_invalid",
            ) from exc

    def _request_json(
        self,
        system: str,
        user: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_errors: list[str] = []
        for attempt in range(self.max_attempts):
            try:
                parsed = _extract_json_object(self._chat(system, user))
                result = validator(parsed) if validator is not None else parsed
                self.last_request_audit = {
                    "attempt_count": attempt + 1,
                    "attempt_failure_categories": attempt_errors,
                    "final_failure_category": None,
                }
                return result
            except (
                LLMOutputError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                attempt_errors.append(_exception_category(exc))
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(2**attempt, 4))
        final_category = _exception_category(last_error) if last_error else "unknown_error"
        self.last_request_audit = {
            "attempt_count": self.max_attempts,
            "attempt_failure_categories": attempt_errors,
            "final_failure_category": final_category,
        }
        raise LLMOutputError(
            f"LLM request failed after {self.max_attempts} attempts",
            category=final_category,
            attempts=self.max_attempts,
            attempt_errors=attempt_errors,
        ) from last_error

    def specialist_json(
        self,
        system_prompt: str,
        role_key: str,
        stage: str,
        user_payload: str,
        allowed_actions: list[str],
        expected_risk_score: float,
        expected_risk_level: str,
        allowed_evidence_references: list[str],
        allowed_missing_fields: list[str],
    ) -> dict[str, Any]:
        schema = json.dumps(specialist_json_schema(), ensure_ascii=False)
        return self._request_json(
            system_prompt,
            f"{user_payload}\n\nJSON schema: {schema}\n",
            validator=lambda payload: _validate_specialist(
                payload,
                role_key,
                stage,
                allowed_actions,
                expected_risk_score,
                expected_risk_level,
                allowed_evidence_references,
                allowed_missing_fields,
            ),
        )

    def coordinator_json(
        self,
        system_prompt: str,
        stage: str,
        user_payload: str,
        allowed_actions: list[str],
    ) -> dict[str, Any]:
        schema = json.dumps(coordinator_json_schema(), ensure_ascii=False)
        return self._request_json(
            system_prompt,
            (
                f"{user_payload}\n\nAllowed proposed_action values: "
                f"{allowed_actions}\n\nJSON schema: {schema}\n"
            ),
            validator=lambda payload: _validate_controller(
                payload,
                stage,
                allowed_actions,
            ),
        )

    def single_agent_json(
        self,
        system_prompt: str,
        stage: str,
        user_payload: str,
        allowed_actions: list[str],
        expected_risk_score: float,
        expected_risk_level: str,
        allowed_evidence_references: list[str],
        allowed_missing_fields: list[str],
    ) -> dict[str, Any]:
        schema = json.dumps(single_agent_json_schema(), ensure_ascii=False)
        return self._request_json(
            system_prompt,
            (
                f"{user_payload}\n\nAllowed proposed_action values: "
                f"{allowed_actions}\n\nJSON schema: {schema}\n"
            ),
            validator=lambda payload: _validate_single_agent(
                payload,
                stage,
                allowed_actions,
                expected_risk_score,
                expected_risk_level,
                allowed_evidence_references,
                allowed_missing_fields,
            ),
        )
