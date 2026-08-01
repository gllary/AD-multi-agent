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
    specialist_json_schema,
)


class LLMOutputError(RuntimeError):
    """Raised after retries fail or a response violates the frozen contract."""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise LLMOutputError("Response must contain exactly one valid JSON object")


def _validate_json_schema(
    payload: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as exc:
        raise LLMOutputError(
            f"{label} output violated the frozen JSON schema: {exc.message}"
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
        raise LLMOutputError("Specialist role or stage does not match the request")
    if payload["recommended_next_action"] not in allowed_actions:
        raise LLMOutputError("Specialist proposed an action outside the stage-legal set")

    expected_rounded_score = float(f"{expected_risk_score:.6f}")
    if not math.isclose(
        float(payload["risk_score_tool"]),
        expected_rounded_score,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise LLMOutputError("Specialist risk score does not match the tool-provided score")
    if payload["risk_level_tool"] != expected_risk_level:
        raise LLMOutputError("Specialist risk level does not match the tool-provided level")

    evidence_contract = set(allowed_evidence_references)
    for key in ("supporting_evidence", "counter_evidence"):
        invalid = [item for item in payload[key] if item not in evidence_contract]
        if invalid:
            raise LLMOutputError(
                f"Specialist {key} contains evidence outside the role-bounded packet: {invalid}"
            )

    missing_contract = set(allowed_missing_fields)
    invalid_missing = [
        item for item in payload["missing_critical_data"] if item not in missing_contract
    ]
    if invalid_missing:
        raise LLMOutputError(
            "Specialist missing_critical_data contains fields not marked unknown in "
            f"the role-bounded packet: {invalid_missing}"
        )
    return payload


def _validate_controller(
    payload: dict[str, Any],
    stage: str,
    allowed_actions: list[str],
) -> dict[str, Any]:
    _validate_json_schema(payload, coordinator_json_schema(), "Controller")
    if payload["current_stage"] != stage:
        raise LLMOutputError("Controller stage does not match the request")
    if payload["proposed_action"] not in allowed_actions:
        raise LLMOutputError("Controller proposed an action outside the stage-legal set")
    return payload


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
            raise LLMOutputError("Unexpected chat-completion response") from exc

    def _request_json(
        self,
        system: str,
        user: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                parsed = _extract_json_object(self._chat(system, user))
                return validator(parsed) if validator is not None else parsed
            except (
                LLMOutputError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(2**attempt, 4))
        raise LLMOutputError(
            f"LLM request failed after {self.max_attempts} attempts"
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
    ) -> dict[str, Any]:
        return self.coordinator_json(
            system_prompt,
            stage,
            user_payload,
            allowed_actions,
        )
