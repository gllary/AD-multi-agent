# Frozen Text Feature Extraction Prompts

This document provides PHI-stripped prompt templates for structured
pre-CTA text fields. The configured extractor uses the original official
`Qwen/Qwen3-235B-A22B` release (Qwen Team, Alibaba Cloud) with temperature 0,
thinking disabled, and JSON-object output. Templates contain
placeholders rather than patient text. Credentials, private endpoints, patient
identifiers, and patient narratives are outside this public file.

## Information Boundary

The extractor interface is limited to the prespecified narrative sources listed
below. Diagnostic-summary outputs, including `text_suggests_ad` and
`suggest_ad_on_echo`, are documentation-only schema fields and are outside
specialist, single-agent, and coordinator evidence packets.

| Module | Prespecified narrative source |
|---|---|
| History | Chief complaint, present illness, and other relevant history |
| Examination | Physical-examination text |
| ECG | First eligible ECG diagnostic text |
| Echocardiography | First eligible echocardiography conclusion and findings |

## System Instruction

```text
你是心血管急诊医学专家。你必须只输出严格 JSON（以 { 开头，以 } 结尾），不要解释、不要使用Markdown代码块、不要输出任何多余字符。如果无法判断，请使用 unknown。
```

## History

```text
你是一名心血管急诊医学专家。
请根据以下【主诉/现病史/其他相关病史】文本，仅抽取与"急性主动脉夹层（AD）"相关的症状及病史证据。

【重要规则】
1. 仅根据文本中明确出现的信息进行判断，不允许主观推断
2. 若文本中未提及或无法判断，请输出 "unknown"
3. 不要给出诊断结论，只做信息抽取
4. 输出必须是严格的 JSON，不包含任何额外说明文字

【输入文本】
{history_text}

【请输出以下字段】
{
  "sudden_onset_pain": "0|1|unknown",
  "severe_pain": "0|1|unknown",
  "tearing_pain": "0|1|unknown",
  "migrating_pain": "0|1|unknown",
  "trauma_related": "0|1|unknown",
  "marfan_or_ctd": "0|1|unknown",
  "aortic_disease_history": "0|1|unknown",
  "text_suggests_ad": "low|medium|high|unknown"
}
```

## Examination

```text
你是一名心血管急诊医学专家。
请根据以下【体格检查】文本，仅抽取与"急性主动脉夹层（AD）"相关的体征证据。

【重要规则】
1. 仅根据文本中明确出现的信息进行判断，不允许主观推断
2. 若文本中未提及或无法判断，请输出 "unknown"
3. 不要给出诊断结论，只做信息抽取
4. 输出必须是严格的 JSON，不包含任何额外说明文字

【输入文本】
{exam_text}

【请输出以下字段】
{
  "pulse_deficit": "0|1|unknown",
  "bp_difference": "0|1|unknown",
  "new_aortic_regurgitation_murmur": "0|1|unknown",
  "neurologic_deficit": "0|1|unknown",
  "hypotension_or_shock": "0|1|unknown",
  "text_suggests_ad": "low|medium|high|unknown"
}
```

## ECG

```text
你是一名心血管急诊医学专家。
请根据以下【心电图】文本，仅抽取与“急性主动脉夹层（AD）鉴别/相关”的心电证据（不下诊断）。

【重要规则】
1. 仅根据文本中明确出现的信息进行判断，不允许主观推断
2. 若文本中未提及或无法判断，请输出 "unknown"
3. 不要给出诊断结论，只做信息抽取
4. 输出必须是严格的 JSON，不包含任何额外说明文字

【输入文本】
{ecg_text}

【请输出以下字段】
{
  "st_elevation": "0|1|unknown",
  "st_depression": "0|1|unknown",
  "arrhythmia": "0|1|unknown",
  "acs_like_ecg": "0|1|unknown",
  "text_suggests_ad": "low|medium|high|unknown"
}
```

## Echocardiography

```text
你是一名心血管急诊医学专家。
请根据以下【床旁超声/心脏彩超】文本，仅抽取与“急性主动脉夹层（AD）相关”的超声证据（不下诊断）。

【重要规则】
1. 仅根据文本中明确出现的信息进行判断，不允许主观推断
2. 若文本中未提及或无法判断，请输出 "unknown"
3. 不要给出诊断结论，只做信息抽取
4. 输出必须是严格的 JSON，不包含任何额外说明文字

【输入文本】
{echo_text}

【请输出以下字段】
{
  "ascending_aorta_dilated": "0|1|unknown",
  "aortic_valve_disease": "0|1|unknown",
  "pericardial_effusion": "0|1|unknown",
  "suspected_intimal_flap": "0|1|unknown",
  "suggest_ad_on_echo": "0|1|unknown",
  "text_suggests_ad": "low|medium|high|unknown"
}
```

## Downstream Agent Boundary

Role-restricted evidence views use explicit allowlists rather than raw-table
dumps. The extraction fields `text_suggests_ad`, `suggest_ad_on_echo`, their
module-prefixed forms, and legacy or normalized diagnostic-summary aliases are
not part of the allowed field sets. The evidence renderer applies the same
public evidence boundary.
