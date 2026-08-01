# Research Pathway Package

This package contains the role-restricted agents, fixed-threshold and
single-agent comparators, policy thresholds, and deterministic safety
governance used in the reported research configuration. It uses patient-level
CP1-CP4 risk scores as quantitative inputs.

The evaluated LLM configuration used Qwen3-235B-A22B through a private
OpenAI-compatible vLLM endpoint, temperature 0.2, an 8,192-token server context
limit, and a 120-second request timeout.

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
for metric calculation. Malformed, missing, timed-out, or stage-illegal LLM
output uses the fixed fallback pathway after automatic retries.
