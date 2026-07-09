# Contributing

Thank you for your interest in this research code release.

This repository is a manuscript-aligned release bundle rather than a general
clinical product. Contributions should preserve the frozen analysis boundary
unless a change is explicitly marked as a separate extension.

## Scope

Appropriate contributions include:

- Reproducibility fixes for installation, packaging, documentation, and scripts.
- Clear bug fixes that do not change frozen manuscript results unexpectedly.
- Additional tests or checks for public release hygiene.
- Documentation clarifying data boundaries, ethics, or non-clinical use.

Out-of-scope contributions include:

- Clinical deployment claims or prospective triage protocols.
- Pull requests containing protected health information, raw hospital data,
  credentials, or unrestricted validation-cohort inputs.
- Silent changes to frozen thresholds, prompts, labels, model weights, or
  manuscript metrics.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements.txt
python -m pip install -e pipeline
```

Live LLM calls require environment variables. Copy `.env.example` locally if
useful, but do not commit real credentials.

## Before Submitting Changes

Run lightweight checks from the repository root:

```bash
python -m compileall pipeline/src pipeline/scripts analysis
git diff --check
rg -n '\bsk-[A-Za-z0-9_\-]{20,}' .
```

If a change affects manuscript-facing values, document the source of the new
numbers and update the supplementary files in the same pull request.
