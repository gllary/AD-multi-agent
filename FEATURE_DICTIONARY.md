# Feature Dictionary

This dictionary describes the de-identified, pre-CTA fields used to generate
the reported stage scores and the role-restricted evidence views. `ID` is a
release-only pseudonym. `AD` is an evaluation-only reference label and is not a
risk-model or LLM input. CP1-CP4 scores provide the quantitative inputs to the
public pathway package.

Structured-concept fields from prespecified narrative sources use
Qwen3-235B-A22B with fixed schema-constrained prompts. PHI-stripped prompt
templates and exact output schemas are provided in
`TEXT_FEATURE_EXTRACTION_PROMPTS.md`.

Missing values remain missing. Binary concept indicators use `1` for present,
`0` for absent, and a missing value when the source record is insufficient.

## CP1: History and Examination

| Field | Definition | Coding or unit |
|---|---|---|
| `Age` | Age at index presentation | years |
| `Sex` | Recorded sex | male=1, female=0 |
| `history__sudden_onset_pain` | Abrupt pain onset | binary concept |
| `history__severe_pain` | Severe pain | binary concept |
| `history__tearing_pain` | Tearing or ripping pain | binary concept |
| `history__migrating_pain` | Migrating pain | binary concept |
| `history__trauma_related` | Trauma-related presentation | binary concept |
| `history__marfan_or_ctd` | Marfan syndrome or connective-tissue disorder | binary concept |
| `history__aortic_disease_history` | Prior aortic disease history other than excluded prior AD | binary concept |
| `exam__pulse_deficit` | Pulse deficit | binary concept |
| `exam__bp_difference` | Clinically documented inter-limb blood-pressure difference | binary concept |
| `exam__new_aortic_regurgitation_murmur` | New aortic-regurgitation murmur | binary concept |
| `exam__neurologic_deficit` | Focal neurologic deficit | binary concept |
| `exam__hypotension_or_shock` | Hypotension or shock | binary concept |

## CP2: Laboratory Evidence

CP2 contains all CP1 fields plus:

| Field | Definition | Coding or unit |
|---|---|---|
| `troponin_abnormal` | Troponin I or T above the local reference limit | binary |
| `D_D_abnormal` | D-dimer above the local reference limit | binary |
| `D_D_log` | Harmonized D-dimer value transformed as natural log(1+x) | transformed continuous value |
| `NT_proBNP_log` | NT-proBNP transformed as natural log(1+x) | transformed continuous value |
| `Mb_log` | Myoglobin transformed as natural log(1+x) | transformed continuous value |
| `CK_MB_log` | Creatine kinase-MB transformed as natural log(1+x) | transformed continuous value |

## CP3: ECG Concepts and Measurements

The CP3 evidence interface contains concept indicators and structured
measurements only; raw ECG waveforms and narrative report text are not
risk-model or LLM inputs.

Concept fields are `ecg_text_st_elevation`, `ecg_text_st_depression`,
`ecg_text_arrhythmia`, and `ecg_text_acs_like_ecg`. The
`ecg_pattern_risk_context` category is a deterministic summary of those ECG
patterns: high for ST-elevation or atrial-fibrillation patterns, medium for
ST-depression, T-wave-change, or other arrhythmia patterns, low when a report
contains none of those concepts, and unknown when no eligible report is
available.

The `kw_*` fields are fixed binary concept indicators for abnormal ECG,
arrhythmia, bradycardia, ventricular hypertrophy, QT prolongation, abnormal Q
waves, ST depression, ST elevation, tachycardia, and T-wave change.
`kw_stmt_count` and `kw_text_len` contain report-structure counts without the
report text.

Structured measurements are:

| Field | Unit |
|---|---|
| `ecg_ventricularrate`, `ecg_atrialrate` | beats/min |
| `ecg_printerval`, `ecg_qrsduration`, `ecg_qtinterval`, `ecg_qtcbazett`, `ecg_rrinterval` | ms |
| `ecg_paxis`, `ecg_raxis`, `ecg_taxis` | degrees |
| `ecg_sv1`, `ecg_rv5`, `ecg_sv1rv5` | source-system voltage unit |

## CP4: Echocardiography

CP4 contains the CP1 and CP2 fields plus structural concepts from the
first eligible pre-CTA transthoracic echocardiography report:

| Field | Definition | Coding |
|---|---|---|
| `echo__ascending_aorta_dilated` | Ascending-aortic dilatation | binary concept |
| `echo__aortic_valve_disease` | Aortic-valve disease | binary concept |
| `echo__pericardial_effusion` | Pericardial effusion | binary concept |
| `echo__suspected_intimal_flap` | Reported intimal-flap, double-lumen, or true/false-lumen finding | binary concept |

Clinical diagnostic-impression fields, explicit prospective trigger phrases,
confirmatory CTA/MRA information, final diagnoses, discharge or procedure
records, Stanford classification, and follow-up outcomes are outside the
framework input boundary.

## Diagnostic-Summary Extraction Outputs

The following diagnostic-summary fields are documentation-only fields. They are
outside specialist, single-agent, and coordinator evidence packets.

| Extraction module | Narrative source | Diagnostic-summary output |
|---|---|---|
| History | Chief complaint, present illness, and other relevant history | `text_suggests_ad` |
| Examination | Physical-examination text | `text_suggests_ad` |
| ECG | First eligible ECG diagnostic text | `text_suggests_ad` |
| Echocardiography | First eligible echocardiography conclusion and findings | `suggest_ad_on_echo`, `text_suggests_ad` |

Module-prefixed forms such as `history__text_suggests_ad`,
`exam__text_suggests_ad`, `ecg__text_suggests_ad`,
`echo__text_suggests_ad`, and `echo__suggest_ad_on_echo` have the same
documentation-only status. Legacy or normalized diagnostic-summary aliases
retained in source schemas are also outside downstream agent evidence. Allowed
agent column sets are listed in
`evidence_views.py`; rendered evidence packets use the public evidence
interface in `curated_evidence.py`.
