# Manuscript-to-Repository Mapping

This table maps the method components described in the associated manuscript,
*A Safety-Governed Multi-Agent Framework for Sequential Pathway Allocation in Suspected Aortic Dissection*, to the corresponding files in this public release. It is a file
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
| Code-action to clinical-action terminology | `README.md` section “Clinical Terminology and Output Interpretation”; `pipeline/src/llm_tool_multi_agent/safety_layer.py` | Maps the 6 code actions to clinician-overseen reassessment, checkpoint continuation, direct CTA, or urgent specialist/transfer escalation and defines the binary assigned-escalation output |
| Runtime configuration | `pipeline/src/llm_tool_multi_agent/config.py`; `pipeline/artifacts/runtime/frozen_llm_runtime.json` | Non-sensitive model-serving and inference settings |
| Evaluation entry point | `pipeline/scripts/run_pathway.py` | Fixed-threshold, single-agent, and multi-agent execution interface with pathway-allocation outputs |
| Released development inputs | `data/raw_data/development_cohort/` | De-identified CP1, CP2, and CP4 structured tables for the 1,010-patient development cohort |
| Clinical-workflow reference standard | Not released as patient-level data | Pragmatic adjudicated clinical reference based on available clinical records and 30-day index-event outcome verification. Two physicians independently reviewed deidentified clinical records and available follow-up information; agreement was 98.5%, and discordant cases were adjudicated by a senior physician. Reviewers were blinded to quantitative risk signals, agent outputs, and pathway assignments. Seventeen of 11,059 provisional AD-negative patients were reclassified because subsequent CTA performed within 30 days confirmed AD, leaving 11,042 final AD-negative patients |

External validation inputs, protected health information, raw narratives,
patient-level LLM traces, and controlled-access hospital data are not included.
