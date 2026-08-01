# -*- coding: utf-8 -*-
"""Build a compact curated text block from CP rows for LLM context (no raw table dump)."""

from __future__ import annotations

import re

import pandas as pd


DIAGNOSTIC_SUMMARY_PATTERNS = (
    re.compile(r"(?:^|__)text_suggests_[a-z0-9_]+$"),
    re.compile(r"(?:^|__)suggest_[a-z0-9_]+_on_echo$"),
)
UNKNOWN_VALUE = "unknown"


def _redact_column(name: str) -> bool:
    """Omit fields that encode reference labels or explicit diagnostic wording."""
    cl = str(name).lower()
    return any(pattern.search(cl) for pattern in DIAGNOSTIC_SUMMARY_PATTERNS)


def _fmt_val(v) -> str:
    if pd.isna(v) or v is None:
        return UNKNOWN_VALUE
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "unknown"}:
        return UNKNOWN_VALUE
    return text


def _curated_records_from_row(
    row: pd.Series,
    max_items: int,
    include_columns: list[str] | set[str] | tuple[str, ...] | None,
) -> list[tuple[str, str]]:
    skip = {"ID", "AD", "id"}
    if include_columns is None:
        names = [str(name) for name in row.index]
    else:
        names = [str(name) for name in include_columns]

    records: list[tuple[str, str]] = []
    for name in names:
        if name in skip or _redact_column(name):
            continue
        value = row[name] if name in row.index else None
        records.append((name, _fmt_val(value)))
        if len(records) >= max_items:
            break
    return records


def curated_evidence_from_row(
    row: pd.Series,
    max_items: int = 48,
    include_columns: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str:
    """Render every permitted field, preserving missing values as explicit unknowns."""
    records = _curated_records_from_row(row, max_items, include_columns)
    if not records:
        return "(no structured fields in this view)"
    return "\n".join(f"- {name}: {value}" for name, value in records)


def evidence_contract_from_text(text: str) -> tuple[list[str], list[str]]:
    """Return exact field=value citations and explicitly unknown field names."""
    references: list[str] = []
    missing_fields: list[str] = []
    for line in text.splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        name, value = line[2:].split(": ", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if value.lower() == UNKNOWN_VALUE:
            missing_fields.append(name)
        else:
            references.append(f"{name}={value}")
    return references, missing_fields
