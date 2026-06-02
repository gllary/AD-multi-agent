# AAS Code Release Bundle

This folder is a scoped release copy for the manuscript:

**Safety-governed multi-agent pathway control for suspected acute aortic syndrome: a retrospective external evaluation**

It keeps only materials aligned with the current manuscript narrative:

- Frozen safety-governed Qwen multi-agent, single-agent, and canonical pathway-control code.
- Frozen quantitative model artifacts and policy thresholds.
- Current manuscript figure/table generation scripts.
- PHI-stripped frozen prompt templates.
- De-identified Cohort D retained analysis data.

It intentionally excludes historical exploratory runs, obsolete figure sets, unrestricted validation-cohort raw data, LLM call logs, cache files, pycache files, and intermediate output directories.

## Layout

```text
AAS_Code/
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
└── manuscript_materials/        # manuscript tex and frozen prompt supplement
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

Some analysis scripts retain references to restricted external validation inputs from the original local project. Those inputs are intentionally absent here; the scripts document the exact current-generation pipeline, while external-cohort reproduction requires approved controlled-platform access.
