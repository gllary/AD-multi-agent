# llm-tool-multi-agent

Python package and scripts for the frozen safety-governed multi-agent pathway
controller used in the accompanying acute aortic dissection manuscript.

The package is designed for source-tree or editable installation so that the
scripts can access the frozen artifacts under `pipeline/artifacts/`.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements.txt
python -m pip install -e pipeline
```

## Environment

Live LLM calls are optional. Without `LLM_API_KEY`, the client runs in stub mode
where supported by the pathway code. For live calls:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="http://127.0.0.1:8003/v1/"
export LLM_MODEL="Qwen3-235B-A22B-Instruct"
```

The phase-1 Qwen runner also understands `QWEN_API_KEY`, `QWEN_BASE_URL`, and
`QWEN_MODEL_NAME`.

## Example

```bash
cd pipeline
python scripts/run_pathway.py --cohort datasetA --limit 10
```

Outputs are written under `pipeline/outputs/` by default and are ignored by git.

## Data Boundary

The public release includes retained de-identified Cohort D tables and frozen
model artifacts. Raw external validation inputs, V2 model-input tables, LLM call
logs, and unrestricted hospital data are intentionally absent.

This package is for research transparency only. It is not clinical-use software.
