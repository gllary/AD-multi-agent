# Manuscript-to-Repository Mapping

This table maps the method components described in the associated JAMA Network Open
manuscript to the corresponding files in this public release. It is a file
crosswalk, not a claim that controlled-access validation data are public.

| Manuscript method component | Public release file or symbol | Public scope |
|---|---|---|
| Role-restricted specialist, coordinator, and single-agent prompts | `pipeline/src/llm_tool_multi_agent/prompts.py` | Static pathway-agent system prompts |
| Structured text-feature extraction prompts | `TEXT_FEATURE_EXTRACTION_PROMPTS.md` | PHI-stripped AD-focused extraction templates and output fields |
| Structured-output contracts | `pipeline/src/llm_tool_multi_agent/schemas.py` | Specialist, grounded single-agent, coordinator, and safety-review JSON schemas |
| Output parsing and contract validation | `pipeline/src/llm_tool_multi_agent/llm_client.py` | Single-object JSON parsing plus schema, role/stage, risk-state, evidence-reference, missing-field, and action validation with PHI-free failure-category audit records |
| CP1-CP4 evidence definitions | `FEATURE_DICTIONARY.md`; `pipeline/src/llm_tool_multi_agent/evidence_views.py` | Field definitions and role-specific allowlists |
| Evidence assembly and diagnostic-summary exclusion | `pipeline/src/llm_tool_multi_agent/curated_evidence.py` | Role-bounded evidence rendering, explicit unknown handling, and diagnostic-summary filtering |
| Fixed-threshold comparator | `pipeline/src/llm_tool_multi_agent/fixed_threshold_engine.py`; `pipeline/artifacts/policy/frozen_policy_thresholds.json` | Deterministic checkpoint routing and released thresholds |
| Single-agent comparator | `pipeline/src/llm_tool_multi_agent/single_agent_engine.py`; `pipeline/src/llm_tool_multi_agent/evidence_views.py` | Current-checkpoint field union, bounded prior-controller summary, grounded structured output, and pathway control |
| Safety-governed multi-agent pathway | `pipeline/src/llm_tool_multi_agent/pathway_engine.py`; `pipeline/src/llm_tool_multi_agent/deliberation.py` | Specialist coordination and accepted-action pathway state |
| Stage legality, overrides, and fallback | `pipeline/src/llm_tool_multi_agent/safety_layer.py` | Deterministic governance applied after proposed actions |
| Runtime configuration | `pipeline/src/llm_tool_multi_agent/config.py`; `pipeline/artifacts/runtime/frozen_llm_runtime.json` | Non-sensitive model-serving and inference settings |
| Evaluation entry point | `pipeline/scripts/run_pathway.py` | Fixed-threshold, single-agent, and multi-agent execution interface |
| Released development inputs | `data/raw_data/development_cohort/` | De-identified CP1, CP2, and CP4 structured tables for the 1,010-patient development cohort |

External validation inputs, protected health information, raw narratives,
patient-level LLM traces, and controlled-access hospital data are not included.
