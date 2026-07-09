# Security Policy

## Supported Scope

This repository is a research code release. It is not clinical-use software,
not a medical device, and not a prospective triage protocol.

Security reports should focus on:

- Exposed credentials or tokens.
- Accidental inclusion of protected health information or raw restricted data.
- Vulnerabilities in scripts that process local research files.
- Packaging issues that could mislead users about clinical readiness.

## Reporting

Please report security issues privately to the repository maintainers or the
corresponding study contact listed with the manuscript. Do not open a public
issue containing credentials, patient-level identifiers, or restricted data.

## Credential Handling

The code expects API keys to be supplied through environment variables such as
`LLM_API_KEY`. Real keys must not be committed. The `.env.example` file is
intentionally blank.

## Data Handling

Only the de-identified Cohort D release tables are included. The public Cohort D
`ID` column contains release-only pseudonymous identifiers, not hospital,
encounter, medical-record, or platform identifiers. Cohort V1 and V2
raw validation inputs, LLM call logs, cache files, and unrestricted hospital
data are intentionally excluded.
