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
export LLM_MODEL="provider-model-name"
```

The precomputed input runner accepts the same live-LLM environment variables.

## Example

```bash
cd pipeline
python scripts/run_pathway.py --cohort cohort_D --limit 10
```

Outputs are written under `pipeline/outputs/` by default and are ignored by git.

For controlled-access external data already transformed into CP input tables:

```bash
python scripts/run_precomputed_bundle.py --limit 10
```

## Data Boundary

The public release includes retained de-identified Cohort D tables and frozen
model artifacts. Raw external validation inputs, V2 model-input tables, LLM call
logs, and unrestricted hospital data are intentionally absent.

This package is for research transparency only. It is not clinical-use software.
