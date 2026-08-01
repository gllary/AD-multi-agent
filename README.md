# AD Multi-Agent Method Release

This repository contains the research code, frozen prompt templates, policy
thresholds, non-sensitive runtime configuration, and de-identified
development-cohort structured inputs permitted for public release for:

**Safety-governed multi-agent clinical decision support for pre-CTA pathway allocation in suspected aortic dissection**

## Public Release Contents

The repository includes:

- The staged fixed-threshold, single-agent, and safety-governed multi-agent
  pathway implementation.
- Six role-restricted specialist prompts, the coordinator prompt, and the
  single-agent comparator prompt.
- PHI-stripped text-extraction prompts and output schemas.
- Structured-output schemas, stage-legal action checks, deterministic safety
  governance, and fixed-threshold fallback behavior.
- Frozen policy thresholds and non-sensitive runtime configuration used by the
  released pathway implementation.
- Feature definitions, coding rules, units, and missingness handling.
- De-identified CP1, CP2, and CP4 development-cohort structured input tables
  permitted for public release.

## Repository Contents

```text
pipeline/
  artifacts/policy/       frozen pathway thresholds
  artifacts/runtime/      non-sensitive LLM runtime configuration
  scripts/run_pathway.py  pathway evaluation entry point
  src/                    prompts, schemas, pathway engines, and governance
data/
  raw_data/development_cohort/
FEATURE_DICTIONARY.md
TEXT_FEATURE_EXTRACTION_PROMPTS.md
MANUSCRIPT_REPOSITORY_MAPPING.md
```

The released development-cohort directory contains three tables for the common
analysis denominator of 1,010 patients:

- `development_CP1_demo_history_exam.csv`
- `development_CP2_demo_history_exam_lab.csv`
- `development_CP4_demo_history_exam_lab_echo.csv`

Release-only pseudonymous identifiers link these tables.

## Information Boundaries

Reference labels are retained as evaluation-only fields. Diagnostic-summary
extraction outputs are documentation-only schema fields. Stage- and
role-specific evidence packets are assembled by whitelists in
`pipeline/src/llm_tool_multi_agent/evidence_views.py` and by the evidence
renderer in `pipeline/src/llm_tool_multi_agent/curated_evidence.py`.
Permitted but unavailable fields are rendered explicitly as `unknown`.
Specialist evidence citations must reproduce an allowed `field=value` reference,
and reported missing fields must come from that role-bounded unknown-field list.
The coordinator receives validated current specialist records plus bounded prior
summaries, rather than raw evidence or unrestricted output history.

The released prompts and feature documentation are:

- `pipeline/src/llm_tool_multi_agent/prompts.py`
- `TEXT_FEATURE_EXTRACTION_PROMPTS.md`
- `pipeline/src/llm_tool_multi_agent/schemas.py`
- `FEATURE_DICTIONARY.md`

A direct manuscript-to-repository crosswalk is provided in
[`MANUSCRIPT_REPOSITORY_MAPPING.md`](MANUSCRIPT_REPOSITORY_MAPPING.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e pipeline
```

`pipeline/scripts/run_pathway.py` accepts a patient-level score table containing
`ID`, `label`, and `CP1`-`CP4` risk-score columns. Single-agent and
multi-agent execution combines these scores with the corresponding
stage-bounded evidence tables through a configured OpenAI-compatible endpoint.

Run a pathway from a score table:

```bash
python pipeline/scripts/run_pathway.py \
  --score-table /path/to/scores.csv \
  --method fixed-threshold \
  --output-dir outputs/fixed-threshold
```

After automatic retries, unavailable output or output that fails strict schema,
role/stage, risk-state, evidence-reference, missing-field, or stage-legal action
validation is passed to the deterministic governance layer, which applies the
frozen fallback.

## Citation

Until a journal citation is available, cite this repository and the associated
manuscript title. Machine-readable metadata are provided in `CITATION.cff`.
