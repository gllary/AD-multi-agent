# -*- coding: utf-8 -*-
"""Build a compact curated text block from CP rows for LLM context (no raw table dump)."""

from __future__ import annotations

import pandas as pd


DIAGNOSTIC_SUMMARY_MARKERS = (
    "text_suggests_ad",
    "text_suggests_aas",
    "text_suggests_ais",
    "text_suggests_aos",
    "suggest_ad_on_echo",
    "suggest_aas_on_echo",
)


def _redact_column(name: str) -> bool:
    """Omit fields that encode reference labels or explicit diagnostic wording."""
    cl = str(name).lower()
    return any(marker in cl for marker in DIAGNOSTIC_SUMMARY_MARKERS)


def _fmt_val(v) -> str:
    if pd.isna(v) or v is None:
        return ""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    return str(v).strip()


def curated_evidence_from_row(
    row: pd.Series,
    max_items: int = 48,
    include_columns: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str:
    """Select non-empty fields as bullet lines; cap count for token budget."""
    lines: list[str] = []
    skip = {"ID", "AD", "id"}
    allowed = set(include_columns) if include_columns is not None else None
    for name, val in row.items():
        if name in skip:
            continue
        if allowed is not None and name not in allowed:
            continue
        if _redact_column(name):
            continue
        s = _fmt_val(val)
        if not s or s.lower() in ("nan", "none", "unknown", ""):
            continue
        lines.append(f"- {name}: {s}")
        if len(lines) >= max_items:
            break
    if not lines:
        return "(no non-empty structured fields in this view)"
    return "\n".join(lines)
