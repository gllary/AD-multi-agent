# AAS Multi-Agent Code Release Bundle

This folder is a scoped release copy for the manuscript:

**Safety-governed multi-agent pathway control for suspected acute aortic syndrome: a retrospective external evaluation**

It keeps only materials aligned with the current manuscript narrative:

- Frozen safety-governed Qwen multi-agent, single-agent, and canonical pathway-control code.
- Frozen quantitative model artifacts and policy thresholds.
- Current manuscript figure/table generation scripts.
- PHI-stripped frozen prompt templates.
- De-identified Cohort D retained analysis data.

It intentionally excludes historical exploratory runs, obsolete figure sets, unrestricted validation-cohort raw data, LLM call logs, cache files, pycache files, and intermediate output directories.

## Related Article

This code release is associated with the manuscript:

**Safety-governed multi-agent pathway control for suspected acute aortic syndrome: a retrospective external evaluation**

Manuscript materials included in this bundle:

- `supplementary_file_1_frozen_prompt_templates_phi_stripped.txt`
- `supplementary_file_2_stage_model_development_freeze_details.txt`
- `supplementary_file_3_safety_governance_runtime_audit_details.txt`
- `supplementary_data_1_full_performance_metrics_bootstrap_paired_contrasts.xlsx`
- `supplementary_data_2_runtime_governance_trace_logs.xlsx`
- `supplementary_data_3_prompt_schema_config_freeze_inventory.xlsx`

Current manuscript alignment:

- Cohort V2 final analysis set: n = 15,109.
- Cohort V2 AAS-positive cases: n = 4,067.
- Multi-agent residual AAS-positive reassessment/missed cases: n = 123.
- Negative labels were assigned after physician record review plus individual telephone follow-up in May 2026.

Article status: manuscript/submission draft; journal, DOI, and final citation information are pending. Please update this section after acceptance or public preprint posting.

Suggested citation before DOI assignment:

```text
Authors. Safety-governed multi-agent pathway control for suspected acute aortic syndrome:
a retrospective external evaluation. Manuscript in preparation/submission, 2026.
Code release: AAS_multi_agent.
```

BibTeX placeholder:

```bibtex
@unpublished{aas_pathway_control_2026,
  title  = {Safety-governed multi-agent pathway control for suspected acute aortic syndrome: a retrospective external evaluation},
  author = {Authors},
  year   = {2026},
  note   = {Manuscript in preparation/submission; code release: AAS\_multi\_agent}
}
```

## License

Source code in `pipeline/` and `analysis/` is released under the MIT License. See `LICENSE`.

The de-identified Cohort D files in `data/`, source code, and supplementary materials are provided to support research transparency and reproducibility for the associated article. They are not clinical-use software, medical-device materials, or a prospective triage protocol. Use of data and manuscript materials remains subject to the ethics, data-governance, journal, and citation requirements described in the associated article.

## Layout

```text
AAS_multi_agent/
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
├── supplementary_file_*.txt      # PHI-stripped prompt and audit supplements
└── supplementary_data_*.xlsx     # bootstrap, runtime trace, and inventory workbooks
```

## Cohort D Data Boundary

`data/raw_data/cohort_D/` contains the retained Cohort D analysis denominator only (n = 1,010):

- `cohort_D_CP1_demo_history_exam.csv`
- `cohort_D_CP2_demo_history_exam_lab.csv`
- `cohort_D_CP2E_demo_history_exam_lab_echo.csv`

`data/derived/cohort_D/` contains retained IDs, final action-level predictions/metrics, and OOF score tables used to document the frozen development boundary.

## Exclusions

The current manuscript states that Cohort V1 and Xiangya validation data cannot be publicly released because of institutional review board and hospital data-governance restrictions. Accordingly, this bundle does not include raw external validation inputs, V2 model-input tables, LLM output traces, or unrestricted hospital data.

The following local exploratory directories were not copied because they are not part of the clean release narrative:

- `cohort_*_raw_exploratory/run_*.py`
- historical generated `paper_figures/figures_*` variants; `build_figures_0529.py` is retained only because the current `build_figures_0530.py` wrapper imports it as its plotting base
- `phase1_qwen_*_bundle/outputs/`
- `llm_pipeline_v1/outputs/`
- labeling workspaces and annotation logs

## Notes

Some analysis scripts retain references to restricted external validation inputs from the original local project. Those inputs are intentionally absent here; the scripts document the exact current-generation pipeline and use the current Cohort V2 denominator (n = 15,109), while external-cohort reproduction requires approved controlled-platform access.
