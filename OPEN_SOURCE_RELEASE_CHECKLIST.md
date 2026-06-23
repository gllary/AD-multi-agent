# Open Source Release Checklist

Use this checklist before making a public release.

## Required Checks

- Confirm the manuscript-aligned values remain current:
  - Cohort V2 final analysis set: n = 15,109.
  - Cohort V2 AD-positive cases: n = 4,067.
  - Multi-agent residual AD-positive clinician-overseen reassessment cases: n = 123.
  - Negative labels include physician record review plus May 2026 telephone follow-up.
- Confirm no live credentials are present:

```bash
rg -n '\bsk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]+' .
```

- Confirm no macOS or local cache artifacts are tracked:

```bash
git ls-files | rg '(^|/)\.DS_Store$|__pycache__|\.pyc$|(^|/)\.env$'
```

- Confirm Python syntax:

```bash
python -m compileall pipeline/src pipeline/scripts analysis
```

- Confirm diff whitespace:

```bash
git diff --check
```

## Packaging Notes

- The release is source-tree oriented. Use `python -m pip install -e pipeline`
  so scripts can access `pipeline/artifacts/`.
- Raw external validation data are intentionally absent and cannot be released
  without the relevant institutional approvals.
- The repository is for research transparency only and must not be presented as
  clinical-use software.
