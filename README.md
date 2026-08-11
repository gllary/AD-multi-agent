# AD Multi-Agent Method Release

This repository contains the research code, frozen prompt templates, policy
thresholds, non-sensitive runtime configuration, and de-identified
development-cohort structured inputs permitted for public release for:

**Safety-Governed Multi-Agent Pathway Allocation for Suspected Aortic Dissection**

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

The single-agent and multi-agent LLM pathways use the same current-checkpoint
structured field pool, quantitative risk state, model configuration,
stage-legal actions, and deterministic governance. The multi-agent pathway
partitions that field pool across role-restricted specialists, whereas the
single-agent comparator receives its current-checkpoint union directly. Raw
fields from earlier checkpoints are not repeated; only bounded prior-decision
summaries are carried forward. The fixed-threshold comparator uses the same
frozen quantitative risk signals and thresholds without an LLM call.

The complete model, prompt, routing, and governance configuration was finalized
in December 2025 after development-cohort training and before any validation
run. Qwen text extraction and all fixed-threshold, single-agent, and multi-agent
pathway runs for the four cohorts occurred from January through July 2026.

## Workup Reference-Standard Alignment

The workup-validation cohort contained 15,109 patients, including 11,059
provisional AD-negative patients before outcome adjudication. Retrospective
30-day outcome ascertainment for all 11,059 provisional AD-negative patients
was completed from September 17 through November 1, 2025. An AD diagnosis,
AD-related death, or aortic intervention within 30 days of the index encounter
triggered reclassification; 17 patients were reclassified after subsequent CTA
confirmed AD, leaving 11,042 final AD-negative patients. Two physicians
reviewed deidentified index-hospital records and telephone outcome-ascertainment
materials, with discordant cases adjudicated by a senior physician. These
validation-cohort records and outcome materials are not included in this public
release.

## Clinical Terminology and Output Interpretation

This package performs **pathway allocation**, not autonomous diagnosis,
disposition, or clinical triage. The evaluation-only `label` field stores the
reference-standard aortic dissection status and is never supplied to a pathway.
The terminal output `assigned_escalation` is `1` only when the accepted terminal
action is `direct_cta` or `urgent_transfer`; it is not a diagnostic prediction.

| Code action | Manuscript clinical term | Interpretation |
|---|---|---|
| `observe_or_reassess` | Clinician-overseen reassessment | Continued clinician review; not autonomous discharge or exclusion of aortic dissection |
| `call_lab_agent` | Laboratory continuation | Advance to the laboratory checkpoint |
| `call_ecg_agent` | ECG continuation | Advance to the ECG checkpoint |
| `call_echo_agent` | Echocardiography continuation | Advance to the echocardiography checkpoint |
| `direct_cta` | Direct CTA | Assign computed tomography angiography escalation |
| `urgent_transfer` | Urgent specialist/transfer escalation | Assign urgent specialist review or transfer escalation |

The exact archived prompts retain the phrase `pre-CTA triage` because those
prompts are immutable study artifacts. In this release, that phrase refers to
pre-CTA pathway allocation under clinician oversight and does not denote an
autonomous diagnostic or disposition system.

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
frozen fallback. The audit trace records the attempt count, PHI-free failure
categories, parser/schema/risk/evidence/action validation status, accepted
action, and fallback or override reason.

In the public development-cohort CSV files, unavailable binary concept fields
are encoded as the literal string `unknown`; otherwise, `1` denotes present and
`0` denotes absent.

## Citation

Until a journal citation is available, cite this repository and the associated
manuscript title. Machine-readable metadata are provided in `CITATION.cff`.
