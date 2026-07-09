# AD Multi-Agent Code Release Bundle

This folder is a scoped release copy for the manuscript:

**Safety governed pathway support for suspected acute aortic dissection before computed tomography angiography**

It keeps only materials aligned with the current manuscript narrative:

- Frozen safety-governed Qwen multi-agent, single-agent, and canonical pathway-support code.
- Frozen quantitative model artifacts and policy thresholds.
- Current manuscript figure/table generation scripts.
- PHI-stripped frozen prompt templates.
- De-identified Cohort D retained analysis data.

It excludes restricted validation data, protected health information, credentials,
runtime logs, and non-release working materials.

## Intended Use

This repository is for research transparency, reproducibility review, and
software inspection. It is not clinical-use software, not a medical device, and
not a prospective triage protocol.

The public source tree supports the frozen Cohort D development-boundary
materials and manuscript figure/table scripts. External validation reproduction
requires controlled access to restricted institutional datasets that are not
included in this repository.

## Related Article

This code release is associated with the manuscript:

**Safety governed pathway support for suspected acute aortic dissection before computed tomography angiography**

Manuscript materials included in this bundle:

- `supplementary_file_1_frozen_prompt_templates_phi_stripped.txt`
- `supplementary_file_2_stage_model_development_freeze_details.txt`
- `supplementary_file_3_safety_governance_runtime_audit_details.txt`
- `supplementary_data_1_full_performance_metrics_bootstrap_paired_contrasts.xlsx`
- `supplementary_data_2_runtime_governance_trace_logs.xlsx`
- `supplementary_data_3_stage_models_thresholds_calibration_missingness.xlsx`
- `supplementary_data_4_cohort_v2_negative_strata_metrics.csv`

Current manuscript alignment:

- Cohort V2 final analysis set: n = 15,109.
- Cohort V2 AD-positive cases: n = 4,067.
- Multi-agent residual AD-positive clinician-overseen reassessment cases: n = 123.
- AD-negative labels were assigned after physician record review plus individual telephone follow-up verification that began in December 2025 and was completed in May 2026.

Article status: manuscript/submission draft; journal, DOI, and final citation information are pending. Please update this section after acceptance or public preprint posting.

Suggested citation before DOI assignment:

```text
Authors. Safety governed pathway support for suspected acute aortic dissection
before computed tomography angiography. Manuscript in preparation/submission,
2026. Associated code release.
```

BibTeX placeholder:

```bibtex
@unpublished{ad_multi_agent_triage_2026,
  title  = {Safety governed pathway support for suspected acute aortic dissection before computed tomography angiography},
  author = {Authors},
  year   = {2026},
  note   = {Manuscript in preparation/submission; associated code release}
}
```

## License

Source code in `pipeline/` and `analysis/` is released under the MIT License. See `LICENSE`.

The de-identified Cohort D files in `data/`, source code, and supplementary materials are provided to support research transparency and reproducibility for the associated article. They are not clinical-use software, medical-device materials, or a prospective triage protocol. Use of data and manuscript materials remains subject to the ethics, data-governance, journal, and citation requirements described in the associated article.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements.txt
python -m pip install -e pipeline
```

Run a small Cohort D pathway smoke test:

```bash
cd pipeline
python scripts/run_pathway.py --cohort datasetA --limit 10
```

Live LLM calls require credentials supplied through environment variables. See
`.env.example`; do not commit real credentials.

## Layout

```text
repository-root/
├── pipeline/
│   ├── scripts/                 # frozen pathway/evaluation entry points
│   ├── src/                     # llm_tool_multi_agent package
│   └── artifacts/
│       ├── models/              # LightGBM and CP3 text model artifacts
│       └── policy/              # frozen policy thresholds
├── analysis/
│   ├── figures/                 # current manuscript figure/audit scripts
│   └── tables/                  # current manuscript table scripts
├── data/
│   ├── raw_data/cohort_D/       # retained Cohort D CP1/CP2/CP2E inputs
│   └── derived/cohort_D/        # retained IDs, final predictions, metrics, OOF scores
├── CITATION.cff                 # citation metadata
├── CONTRIBUTING.md              # contribution and frozen-result guidance
├── SECURITY.md                  # security, credential, and data-boundary policy
├── supplementary_file_*.txt      # PHI-stripped prompt and audit supplements
├── supplementary_data_*.xlsx     # bootstrap, runtime trace, and model-freeze workbooks
├── supplementary_data_4_*.csv    # Cohort V2 AD-negative evidence-stratum metrics
└── supplementary_data_4_*.manifest.txt
```

## Cohort D Data Boundary

`data/raw_data/cohort_D/` contains the retained Cohort D analysis denominator only (n = 1,010):

- `cohort_D_CP1_demo_history_exam.csv`
- `cohort_D_CP2_demo_history_exam_lab.csv`
- `cohort_D_CP2E_demo_history_exam_lab_echo.csv`

`data/derived/cohort_D/` contains retained release IDs, final action-level
predictions/metrics, and OOF score tables used to document the frozen
development boundary.

The `ID` column in the public Cohort D CSV files contains release-only
pseudonymous identifiers in the form `D000001`, `D000002`, and so on. These
values are not hospital, encounter, medical-record, or platform identifiers, and
no mapping back to institutional source identifiers is included in this
repository.

The retained Cohort D CSV headers and code paths use `AD` as the binary
reference-label column. In the release documentation and associated manuscript,
that column denotes the AD-positive/AD-negative reference label.

## Exclusions

Raw external validation inputs, V2 model-input tables, patient-level LLM traces,
and unrestricted hospital data are not included because they are governed by
institutional review board and hospital data-governance restrictions. External
validation reproduction therefore requires approved controlled-platform access.
