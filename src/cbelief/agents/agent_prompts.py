# 功能说明：
# 1. 定义 Agentic C-BELIEF 各智能体的提示词。
# 2. 所有提示词围绕 query-time boundary、retrospective evidence misuse 和 delayed reattribution 设计。
# 3. 加入 C-BELIEF-Stream benchmark 的 operational label policy，避免 future evidence 反向污染 initial labels。

from __future__ import annotations

import json
from typing import Any, Dict, List


def _format_events(events: List[Any], prefix: str) -> str:
    if not events:
        return f"{prefix}: []"

    lines = [f"{prefix}:"]
    for i, ev in enumerate(events):
        if isinstance(ev, dict):
            ev_id = ev.get("evidence_id") or ev.get("id") or f"{prefix}_{i}"
            text = ev.get("text") or ev.get("event_text") or json.dumps(ev, ensure_ascii=False)
            lines.append(f"- {ev_id}: {text}")
        else:
            lines.append(f"- {prefix}_{i}: {str(ev)}")

    return "\n".join(lines)


def sample_context(sample: Dict[str, Any], include_labels: bool = False) -> str:
    visible = sample.get("visible_stream") or sample.get("visible_summary") or []
    future = sample.get("future_stream") or sample.get("future_summary") or []

    target_claim_initial = sample.get("target_claim_initial", "")
    target_claim_final = sample.get("target_claim_final", "")

    parts = [
        f"sample_id: {sample.get('sample_id', '')}",
        f"query_time: {sample.get('query_time', '')}",
        "",
        "Target claims to evaluate:",
        f"- target_claim_initial: {target_claim_initial}",
        f"- target_claim_final: {target_claim_final}",
        "",
        "Candidate hypotheses:",
        "- acute_renal_deterioration",
        "- chronic_renal_dysfunction",
        "- uncertain_or_transient_abnormality",
        "",
        _format_events(visible, "visible_evidence"),
        "",
        _format_events(future, "future_or_retrospective_evidence"),
    ]

    # 默认不应打开 include_labels。该选项仅用于本地调试，不应进入真实 LLM 输入。
    if include_labels:
        label = sample.get("weak_label_v2_2") or sample.get("weak_label_v2_1") or {}
        parts.append("")
        parts.append("Gold/Silver label for local debugging only. Do not use in model input:")
        parts.append(json.dumps(label, ensure_ascii=False, indent=2))

    return "\n".join(parts)


SYSTEM = (
    "You are a clinical temporal-reasoning assistant for longitudinal EHR evidence attribution. "
    "You must distinguish evidence visible at query time from future or retrospective evidence. "
    "A source can be real, clinically relevant, and semantically supportive, but still invalid as support "
    "for a query-time claim if it was unavailable at query time. "
    "Return strict JSON only. Do not include markdown, explanations outside JSON, or extra text."
)


BENCHMARK_LABEL_POLICY = """
Benchmark-specific operational label policy:

A. Evidence-access boundary
1. Initial labels must be judged only from visible_evidence.
2. Future or retrospective evidence must never upgrade initial_claim_status.
3. Future confirmation does not make an initial query-time claim supported.
4. Discharge diagnoses, future CKD codes, future AKI diagnoses, later RRT/dialysis, and retrospective notes may support final labels, but they are invalid as support for initial labels.

B. initial_claim_status
1. supported:
   Use supported only when visible_evidence directly supports the target initial claim.
   Examples include:
   - visible creatinine rise event, such as increase >= 0.3 mg/dL within 48h;
   - visible 7-day creatinine trend meeting acute deterioration criteria;
   - visible AKI or acute renal failure diagnosis;
   - visible RRT/dialysis event;
   - visible CKD/chronic renal evidence when the initial claim is chronic renal dysfunction.
2. partially_supported:
   Use partially_supported when visible_evidence is suggestive but not definitive.
   Examples include:
   - stable high creatinine without clear baseline or prior CKD documentation;
   - low-confidence renal abnormality;
   - incomplete trend evidence;
   - abnormal creatinine without enough temporal context.
3. insufficient:
   Use insufficient when visible_evidence does not directly support the target initial claim and the main support comes only from future or retrospective evidence.
   Typical insufficient cases include:
   - acute_by_future_diagnosis_or_rrt_only, when AKI/RRT evidence appears only after query_time;
   - chronic_by_future_ckd_code, when CKD support appears only in future diagnosis codes or retrospective documentation;
   - comorbid AKI/CKD cases where visible evidence alone cannot adjudicate the initial claim.

C. final_claim_status
1. Final labels may use visible + future_or_retrospective_evidence.
2. supported means the full record directly supports the final claim.
3. partially_supported means the full record suggests the final claim but remains weak, low-confidence, or incomplete.
4. insufficient means the full record does not support the final claim.

D. acute vs chronic hypothesis
1. Acute renal deterioration requires evidence of worsening or acute intervention:
   - creatinine rise;
   - AKI or acute renal failure diagnosis;
   - RRT/dialysis;
   - clear acute trend.
2. Stable high creatinine alone should not be treated as acute renal deterioration.
3. Chronic renal dysfunction is favored when evidence shows:
   - CKD diagnosis;
   - long-standing renal impairment;
   - stable high creatinine;
   - retrospective CKD documentation;
   - persistent renal dysfunction without clear acute worsening.
4. If visible evidence only shows stable high creatinine, the initial_primary_hypothesis may be chronic_renal_dysfunction, but initial_claim_status should often be partially_supported unless visible chronic evidence is direct and clear.

E. mixed acute-on-chronic phenotype
1. Use mixed_acute_on_chronic_renal_dysfunction only when both are present:
   - chronic renal background, CKD, or stable chronic impairment;
   - acute worsening, AKI diagnosis, RRT/dialysis, or acute creatinine trend.
2. Do not output mixed acute-on-chronic for stable CKD or stable high creatinine alone.
3. If the record mainly shows chronic renal dysfunction without acute worsening, final_clinical_phenotype should be chronic_renal_dysfunction.
4. If visible evidence supports acute deterioration and later evidence reveals CKD/chronic background, final_clinical_phenotype may be mixed_acute_on_chronic_renal_dysfunction.

F. delayed reattribution
1. requires_delayed_reattribution = true only when future or retrospective evidence changes, clarifies, or reassigns the final interpretation beyond what was visible at query time.
2. Do not set requires_delayed_reattribution = true merely because future evidence repeats the same interpretation already supported at query time.
3. If future CKD evidence changes an initially acute-looking case into mixed acute-on-chronic, requires_delayed_reattribution should usually be true.
4. If future evidence is the first real support for the claim, the initial label should remain insufficient or partially_supported; do not make it supported.

""".strip()


TEMPORAL_SAFETY_POLICY = """
Temporal safety policy:
1. initial_supporting_evidence_ids may contain only visible_evidence IDs.
2. final_supporting_evidence_ids may contain visible_evidence IDs and future_or_retrospective_evidence IDs.
3. Any future_or_retrospective_evidence ID used for initial support must be reported under invalid_initial_support_ids.
4. If there is uncertainty about whether an evidence item was visible at query time, do not use it as initial support.
5. Retrospective evidence can explain final interpretation but cannot prove what was known at query_time.
6. Keep support lists concise. Do not enumerate every repeated laboratory value; select the most diagnostic evidence IDs only.
""".strip()


def temporal_gatekeeper_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    user = f"""
Role: Temporal Gatekeeper.

Task:
Classify evidence by temporal validity for query-time reasoning.

Rules:
1. Only visible_evidence may support initial/query-time labels.
2. future_or_retrospective_evidence may support final retrospective interpretation but must not support initial labels.
3. Retrospective/discharge-level documentation is not contemporaneous support for query-time claims.
4. Evidence after query_time is not valid initial support, even if it is clinically correct.

Temporal safety policy:
{TEMPORAL_SAFETY_POLICY}

Input:
{sample_context(sample)}

Return strict JSON with keys:
- visible_evidence_ids: list[str]
- future_evidence_ids: list[str]
- retrospective_evidence_ids: list[str]
- query_time_support_rule: string
- warnings: list[str]
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def hypothesis_panel_prompt(sample: Dict[str, Any], gatekeeper: Dict[str, Any]) -> List[Dict[str, str]]:
    user = f"""
Role: Hypothesis Panel.

Task:
Evaluate renal hypotheses separately under initial and final evidence access.

Clinical hypotheses:
- acute_renal_deterioration
- chronic_renal_dysfunction
- uncertain_or_transient_abnormality

Rules:
1. initial_primary_hypothesis must use only visible evidence.
2. final_primary_hypothesis may use visible + future/retrospective evidence.
3. Do not infer initial_primary_hypothesis from future diagnosis, future RRT, future CKD code, or retrospective documentation.
4. Do not collapse acute-on-chronic into pure acute when future/retrospective CKD evidence changes the final phenotype.
5. Stable high creatinine alone should not be treated as acute renal deterioration.
6. Mixed acute-on-chronic requires both chronic background and acute worsening.

Benchmark-specific label policy:
{BENCHMARK_LABEL_POLICY}

Input:
{sample_context(sample)}

Temporal Gatekeeper output:
{json.dumps(gatekeeper, ensure_ascii=False, indent=2)}

Return strict JSON with keys:
- initial_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- final_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- hypothesis_scores: object
- rationale: string
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def evidence_adjudicator_prompt(
    sample: Dict[str, Any],
    gatekeeper: Dict[str, Any],
    panel: Dict[str, Any],
) -> List[Dict[str, str]]:
    user = f"""
Role: Evidence Adjudicator.

Task:
Select valid supporting evidence and reject temporally invalid support.

Rules:
1. initial_supporting_evidence_ids must contain only visible evidence IDs.
2. final_supporting_evidence_ids may contain visible, future, or retrospective evidence IDs.
3. If any initial support uses future/retrospective evidence, list it under invalid_initial_support_ids.
4. Do not use future diagnosis/RRT/CKD code/discharge summary to support initial_claim_status.
5. If visible evidence is weak, choose fewer initial supporting evidence IDs rather than using invalid future support.

Temporal safety policy:
{TEMPORAL_SAFETY_POLICY}

Input:
{sample_context(sample)}

Gatekeeper:
{json.dumps(gatekeeper, ensure_ascii=False, indent=2)}

Hypothesis Panel:
{json.dumps(panel, ensure_ascii=False, indent=2)}

Return strict JSON with keys:
- initial_supporting_evidence_ids: list[str]
- final_supporting_evidence_ids: list[str]
- invalid_initial_support_ids: list[str]
- rationale: string

Length constraints:
- initial_supporting_evidence_ids: at most 8 IDs.
- final_supporting_evidence_ids: at most 8 IDs.
- rationale: one short sentence.
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def delayed_reattribution_prompt(
    sample: Dict[str, Any],
    gatekeeper: Dict[str, Any],
    panel: Dict[str, Any],
    adjudicator: Dict[str, Any],
) -> List[Dict[str, str]]:
    user = f"""
Role: Delayed Reattribution Agent.

Task:
Decide whether future/retrospective evidence changes or clarifies final interpretation.

Rules:
1. Delayed reattribution is allowed for final retrospective labels.
2. Delayed evidence must not be treated as query-time support.
3. If visible evidence supports acute deterioration and later evidence indicates CKD/chronic background, final_clinical_phenotype may be mixed_acute_on_chronic_renal_dysfunction.
4. Stable high creatinine plus CKD without acute worsening should be chronic_renal_dysfunction, not mixed acute-on-chronic.
5. Do not set requires_delayed_reattribution to true if future evidence merely repeats the same chronic pattern already visible.

Benchmark-specific label policy:
{BENCHMARK_LABEL_POLICY}

Input:
{sample_context(sample)}

Gatekeeper:
{json.dumps(gatekeeper, ensure_ascii=False, indent=2)}

Hypothesis Panel:
{json.dumps(panel, ensure_ascii=False, indent=2)}

Evidence Adjudicator:
{json.dumps(adjudicator, ensure_ascii=False, indent=2)}

Return strict JSON with keys:
- requires_delayed_reattribution: bool
- final_clinical_phenotype: acute_renal_deterioration | chronic_renal_dysfunction | mixed_acute_on_chronic_renal_dysfunction | uncertain_or_transient_abnormality
- rationale: string
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def coordinator_prompt(
    sample: Dict[str, Any],
    gatekeeper: Dict[str, Any],
    panel: Dict[str, Any],
    adjudicator: Dict[str, Any],
    reattribution: Dict[str, Any],
) -> List[Dict[str, str]]:
    user = f"""
Role: Clinical Assessment Coordinator.

Task:
Produce the final structured C-BELIEF prediction by integrating all agent outputs.

Critical rules:
1. Initial labels must only use query-time visible evidence.
2. Final labels may use full evidence.
3. Never use future or retrospective evidence as support for initial_claim_status.
4. Future-confirmed renal disease does not imply initial_claim_status is supported.
5. If the only direct evidence for AKI, RRT, CKD, or renal failure appears after query_time, initial_claim_status should usually be insufficient.
6. If visible evidence is suggestive but weak, initial_claim_status should be partially_supported, not supported.
7. Stable high creatinine alone should not be classified as acute renal deterioration.
8. Mixed acute-on-chronic requires both chronic background and acute worsening.
9. final_primary_hypothesis must be exactly one of the three candidate hypotheses. Do not output mixed_acute_on_chronic_renal_dysfunction in final_primary_hypothesis; use it only in final_clinical_phenotype.
10. If the final phenotype is mixed acute-on-chronic, set final_primary_hypothesis to acute_renal_deterioration when acute worsening is present, otherwise chronic_renal_dysfunction.
11. Keep support evidence concise. Do not enumerate every repeated lab; select the most diagnostic IDs only.
12. Return strict JSON only.



Benchmark-specific label policy:
{BENCHMARK_LABEL_POLICY}

Temporal safety policy:
{TEMPORAL_SAFETY_POLICY}

Input:
{sample_context(sample)}

Temporal Gatekeeper:
{json.dumps(gatekeeper, ensure_ascii=False, indent=2)}

Hypothesis Panel:
{json.dumps(panel, ensure_ascii=False, indent=2)}

Evidence Adjudicator:
{json.dumps(adjudicator, ensure_ascii=False, indent=2)}

Delayed Reattribution:
{json.dumps(reattribution, ensure_ascii=False, indent=2)}

Return strict JSON with keys:
- initial_claim_status: supported | partially_supported | insufficient
- final_claim_status: supported | partially_supported | insufficient
- initial_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- final_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- final_clinical_phenotype: acute_renal_deterioration | chronic_renal_dysfunction | mixed_acute_on_chronic_renal_dysfunction | uncertain_or_transient_abnormality
- requires_delayed_reattribution: bool
- initial_supporting_evidence_ids: list[str]
- final_supporting_evidence_ids: list[str]
- rationale: string

Length constraints:
- initial_supporting_evidence_ids: at most 8 IDs.
- final_supporting_evidence_ids: at most 8 IDs.
- rationale: one short sentence.
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
