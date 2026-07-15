from __future__ import annotations


def predict(sample: dict) -> dict:
    visible_text = " ".join(sample.get("visible_stream", [])).lower()
    retro_text = " ".join(sample.get("retrospective_stream", [])).lower()
    if "creatinine_rise" in visible_text or "increased" in visible_text or "dialysis" in visible_text or "rrt" in visible_text:
        primary = "acute_renal_deterioration"
        status = "supported"
    elif "ckd" in visible_text or "chronic kidney" in visible_text:
        primary = "chronic_renal_dysfunction"
        status = "supported"
    elif "aki" in retro_text or "ckd" in retro_text:
        primary = "acute_renal_deterioration" if "aki" in retro_text else "chronic_renal_dysfunction"
        status = "insufficient"
    else:
        primary = "uncertain_or_transient_abnormality"
        status = "insufficient"
    belief = {
        "acute_renal_deterioration": 0.1,
        "chronic_renal_dysfunction": 0.1,
        "uncertain_or_transient_abnormality": 0.1,
    }
    belief[primary] = 0.8
    return {
        "sample_id": sample["sample_id"],
        "method_name": "rule_based",
        "claim_status": status,
        "primary_hypothesis": primary,
        "belief_distribution": belief,
        "supporting_evidence_ids": sample.get("visible_event_ids", [])[:3] if status == "supported" else [],
        "contradictory_evidence_ids": [],
        "retrospective_only_evidence_ids": sample.get("retrospective_event_ids", []) if status == "insufficient" else [],
        "temporally_invalid_evidence_ids": [],
        "misused_future_evidence_ids": [],
        "missing_evidence": [],
        "rationale": "Rule baseline based on visible creatinine trends and CKD terms.",
    }
