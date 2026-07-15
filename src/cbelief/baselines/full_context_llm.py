from __future__ import annotations

# Placeholder: replace call_llm() with your provider/API client.
# Keep this baseline intentionally vulnerable to retrospective evidence misuse.


def predict(sample: dict) -> dict:
    all_text = " ".join(sample.get("visible_stream", []) + sample.get("future_stream", []) + sample.get("retrospective_stream", [])).lower()
    selected = []
    selected += sample.get("visible_event_ids", [])[:2]
    selected += sample.get("future_event_ids", [])[:2]
    selected += sample.get("retrospective_event_ids", [])[:2]
    if "aki" in all_text or "acute" in all_text or "creatinine" in all_text:
        primary = "acute_renal_deterioration"
        status = "supported"
    elif "ckd" in all_text or "chronic" in all_text:
        primary = "chronic_renal_dysfunction"
        status = "supported"
    else:
        primary = "uncertain_or_transient_abnormality"
        status = "insufficient"
    belief = {"acute_renal_deterioration": 0.2, "chronic_renal_dysfunction": 0.2, "uncertain_or_transient_abnormality": 0.2}
    belief[primary] = 0.6
    return {
        "sample_id": sample["sample_id"],
        "method_name": "full_context_llm_placeholder",
        "claim_status": status,
        "primary_hypothesis": primary,
        "belief_distribution": belief,
        "supporting_evidence_ids": selected,
        "contradictory_evidence_ids": [],
        "retrospective_only_evidence_ids": [],
        "temporally_invalid_evidence_ids": [],
        "misused_future_evidence_ids": [],
        "missing_evidence": [],
        "rationale": "Placeholder full-context baseline; replace with actual LLM output.",
    }
