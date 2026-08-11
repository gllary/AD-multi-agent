# Research Pathway Package

This package contains the role-restricted agents, fixed-threshold and
single-agent comparators, policy thresholds, and deterministic safety
governance used in the reported research configuration. It uses patient-level
CP1-CP4 risk scores as quantitative inputs.

The evaluated LLM configuration used the original official
`Qwen/Qwen3-235B-A22B` release (Qwen Team, Alibaba Cloud), without fine-tuning
or quantization, through a private OpenAI-compatible vLLM 0.8.5 endpoint on
four 141-GB NVIDIA H20 GPUs. Text extraction used temperature 0 with thinking
disabled. Pathway-agent calls used temperature 0.2, an 8,192-token server
context limit, a 120-second request timeout, and no more than 3 attempts.

The complete configuration was finalized in December 2025 after
development-cohort training and before any validation run. Text extraction and
all three pathway runs occurred from January through July 2026.

Install from the repository root:

```bash
python -m pip install -r pipeline/requirements.txt
python -m pip install -e pipeline
```

For pathway evaluation, set `AD_EVIDENCE_DIR` to the directory containing
de-identified CP1, CP2, and CP4 evidence tables. Their
filenames can be specified with `AD_CP1_EVIDENCE_FILE`,
`AD_CP2_EVIDENCE_FILE`, and `AD_CP4_EVIDENCE_FILE`. Set
`AD_CP3_EVIDENCE_FILE` to the concept-coded CP3 evidence table. Pathway runs
also require a patient-level score table containing `ID`, `label`, and
`CP1`, `CP2`, `CP3`, and `CP4` risk scores.

The released public data tables support inspection of the CP1, CP2, and CP4
development-cohort evidence views.

Run the fixed-threshold pathway from a CP1-CP4 score table:

```bash
python pipeline/scripts/run_pathway.py \
  --score-table /path/to/scores.csv \
  --method fixed-threshold \
  --output-dir outputs/fixed-threshold
```

Use `--method single-agent` or `--method multi-agent` with a configured LLM
endpoint and the corresponding evidence tables. Pathway runs use the supplied
scores and stage-bounded evidence packets; evaluation labels are retained only
for allocation-summary calculation. Malformed, missing, timed-out, or stage-illegal LLM
output uses the fixed fallback pathway after automatic retries.

`terminal_actions.csv` reports `assigned_escalation=1` only for the accepted
terminal actions `direct_cta` and `urgent_transfer`; otherwise it reports `0`.
This is a pathway-allocation indicator, not a diagnostic prediction. The
evaluation-only `label` column is the reference-standard AD status and is not a
pathway input. The complete code-action to manuscript-clinical-term mapping is
provided in the repository root `README.md`.

At each reached checkpoint, the single-agent comparator receives the union of
the current checkpoint's code-whitelisted fields. The multi-agent pathway
partitions the same field pool across the stage-relevant specialists. Earlier
raw fields are not repeated at CP3 or CP4; only bounded prior-decision
summaries are carried forward. Both LLM pathways validate risk-state agreement,
evidence references, missing-field claims, schema validity, and stage-legal
actions before deterministic governance. Audit traces retain PHI-free failure
categories and validation status after each call sequence.

Permitted but unavailable fields are supplied as explicit `unknown` values. In
the public CSV files, unavailable binary concept fields use the literal string
`unknown`; otherwise, `1` denotes present and `0` denotes absent.
Specialist outputs must reproduce the server-provided risk score and band, cite
only exact `field=value` references from their role-bounded packet, and name
missing fields only when those fields were marked unknown. The coordinator sees
validated current specialist records and bounded prior summaries; it does not
receive raw evidence or unrestricted agent histories.
