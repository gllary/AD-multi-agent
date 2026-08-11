# Publicly Releasable Development-Cohort Structured Inputs

This directory contains the de-identified CP1, CP2, and CP4 structured input
tables permitted for public release for the development-cohort analysis
denominator (n=1,010).

The `ID` column contains release-only pseudonymous identifiers (`D000001`,
`D000002`, and so on).

- `development_CP1_demo_history_exam.csv`
- `development_CP2_demo_history_exam_lab.csv`
- `development_CP4_demo_history_exam_lab_echo.csv`

The three tables contain the same 1,010 release-only IDs. Patient-level CP3 ECG
concept and measurement inputs remain under institutional governance. These
files are the public CP1, CP2, and CP4 structured-data subset.

Unavailable binary concept fields are encoded as the literal string `unknown`.
Otherwise, `1` denotes present and `0` denotes absent; unavailable values were
not imputed.
