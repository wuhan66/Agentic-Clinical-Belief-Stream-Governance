from typing import Dict, Any, List


ACUTE = "acute_renal_deterioration"
CHRONIC = "chronic_renal_dysfunction"
UNCERTAIN = "uncertain_or_transient_abnormality"

CLAIM_STATUS = ["supported", "partially_supported", "insufficient"]
HYPOTHESES = [ACUTE, CHRONIC, UNCERTAIN]


def clean_text(x: Any, max_len: int = 260) -> str:
    text = str(x or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def event_to_line(
    e: Dict[str, Any],
    visibility: str = "",
    include_visibility: bool = False,
    hide_visibility: bool = False,
) -> str:
    eid = clean_text(e.get("event_id"), 80)
    etime = clean_text(e.get("event_time"), 80)
    etype = clean_text(e.get("event_type"), 80)
    ename = clean_text(e.get("event_name"), 120)
    value = e.get("value_num", None)
    unit = clean_text(e.get("unit"), 40)
    text = clean_text(e.get("event_text"), 260)

    value_part = ""
    if value is not None:
        value_part = f" | value={value} {unit}".strip()

    if include_visibility and not hide_visibility:
        return (
            f"- evidence_id={eid} | visibility={visibility} | time={etime} | "
            f"type={etype} | name={ename}{value_part} | text={text}"
        )

    return (
        f"- evidence_id={eid} | time={etime} | "
        f"type={etype} | name={ename}{value_part} | text={text}"
    )


def get_target_claim(sample: Dict[str, Any]) -> str:
    return (
        sample.get("target_claim_initial")
        or sample.get("target_claim")
        or "Assess whether the patient has renal deterioration at the query time."
    )


def system_instruction() -> str:
    return """You are a careful clinical reasoning assistant.

You must answer using only the allowed evidence according to the task instruction.
You must not invent evidence.
You must output valid JSON only.
Do not include markdown.
"""


def output_schema_instruction() -> str:
    return f"""Return exactly one JSON object with this schema:

{{
  "sample_id": "<sample_id>",
  "claim_status": "supported | partially_supported | insufficient",
  "primary_hypothesis": "acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality",
  "supporting_evidence_ids": ["<evidence_id>", "..."],
  "retrospective_only_evidence_ids": ["<evidence_id>", "..."],
  "rationale": "<brief explanation>"
}}

Definitions:
- claim_status = "supported" if the query-time claim is supported by evidence available at the query time.
- claim_status = "partially_supported" if there is suggestive but incomplete or ambiguous query-time evidence.
- claim_status = "insufficient" if query-time evidence is insufficient.
- primary_hypothesis must be one of: {", ".join(HYPOTHESES)}.
- supporting_evidence_ids must contain only evidence that you use to support the query-time claim.
"""


def common_case_header(sample: Dict[str, Any]) -> str:
    sample_id = sample.get("sample_id")
    query_time = sample.get("query_time")
    claim = get_target_claim(sample)

    return f"""Case:
sample_id: {sample_id}
query_time: {query_time}
target_claim: {claim}

Candidate hypotheses:
1. {ACUTE}
2. {CHRONIC}
3. {UNCERTAIN}
"""


def visible_only_prompt(sample: Dict[str, Any]) -> str:
    visible = sample.get("visible_events", []) or []

    evidence_lines = [
        event_to_line(e, visibility="visible", include_visibility=False)
        for e in visible
    ]

    return f"""{system_instruction()}

Task:
You are given only evidence available at the query time.
Assess the target claim using only these visible events.

{common_case_header(sample)}

Visible evidence:
{chr(10).join(evidence_lines) if evidence_lines else "- none"}

{output_schema_instruction()}
"""


def full_context_unmarked_prompt(sample: Dict[str, Any]) -> str:
    visible = sample.get("visible_events", []) or []
    future = sample.get("future_observed_events", []) or []
    retro = sample.get("retrospective_events", []) or []

    # Intentionally no visibility labels. This condition tests whether the model
    # misuses post-query or retrospective evidence when given full context.
    all_events = visible + future + retro

    evidence_lines = [
        event_to_line(e, include_visibility=False, hide_visibility=True)
        for e in all_events
    ]

    return f"""{system_instruction()}

Task:
You are given a longitudinal EHR context.
Assess the target claim at the query time.
Select evidence that supports your conclusion.

{common_case_header(sample)}

EHR evidence:
{chr(10).join(evidence_lines) if evidence_lines else "- none"}

{output_schema_instruction()}
"""


def full_context_with_provenance_prompt(sample: Dict[str, Any]) -> str:
    visible = sample.get("visible_events", []) or []
    future = sample.get("future_observed_events", []) or []
    retro = sample.get("retrospective_events", []) or []

    evidence_lines = []

    for e in visible:
        evidence_lines.append(event_to_line(e, visibility="visible_at_query_time", include_visibility=True))

    for e in future:
        evidence_lines.append(event_to_line(e, visibility="future_observed_after_query_time", include_visibility=True))

    for e in retro:
        evidence_lines.append(event_to_line(e, visibility="retrospective_or_discharge_level", include_visibility=True))

    return f"""{system_instruction()}

Task:
You are given longitudinal EHR evidence with visibility labels.
Assess the target claim at the query time.

Rules:
- You may use only visibility=visible_at_query_time evidence as supporting evidence for the query-time claim.
- You must not use future_observed_after_query_time evidence as query-time support.
- You must not use retrospective_or_discharge_level evidence as query-time support.
- You may list retrospective evidence in retrospective_only_evidence_ids if it is relevant but unavailable at query time.

{common_case_header(sample)}

EHR evidence:
{chr(10).join(evidence_lines) if evidence_lines else "- none"}

{output_schema_instruction()}
"""


def cbelief_prompted_prompt(sample: Dict[str, Any]) -> str:
    visible = sample.get("visible_events", []) or []
    future = sample.get("future_observed_events", []) or []
    retro = sample.get("retrospective_events", []) or []

    evidence_lines = []

    for e in visible:
        evidence_lines.append(event_to_line(e, visibility="visible_at_query_time", include_visibility=True))

    for e in future:
        evidence_lines.append(event_to_line(e, visibility="future_observed_after_query_time", include_visibility=True))

    for e in retro:
        evidence_lines.append(event_to_line(e, visibility="retrospective_or_discharge_level", include_visibility=True))

    return f"""{system_instruction()}

Task:
Use the C-BELIEF temporal evidence protocol to assess the target claim at the query time.

Protocol:
1. Separate evidence by visibility:
   - visible_at_query_time
   - future_observed_after_query_time
   - retrospective_or_discharge_level

2. Determine which evidence can support the query-time claim.
   Only visible_at_query_time evidence can be used as supporting evidence.

3. Maintain competing renal hypotheses:
   - acute_renal_deterioration
   - chronic_renal_dysfunction
   - uncertain_or_transient_abnormality

4. Decide claim_status:
   - supported: query-time visible evidence clearly supports the claim.
   - partially_supported: visible evidence is suggestive but incomplete, ambiguous, or confounded.
   - insufficient: visible evidence does not support the claim.

5. Identify retrospective-only evidence:
   Retrospective or discharge-level evidence may be clinically relevant, but it must not be used as query-time support.

Important:
A correct final diagnosis is not enough. The supporting evidence must be available at query time.

{common_case_header(sample)}

EHR evidence:
{chr(10).join(evidence_lines) if evidence_lines else "- none"}

{output_schema_instruction()}
"""