import json
from pathlib import Path
from typing import Dict, Any, List


HYPOTHESES = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


ACUTE_TERMS = [
    "creatinine_rise_48h",
    "creatinine_rise_7d",
    "acute kidney injury",
    "acute renal",
    " aki ",
    "dialysis",
    "renal replacement",
    "rrt",
]


CHRONIC_TERMS = [
    "ckd",
    "chronic kidney",
    "chronic renal",
    "chronic renal dysfunction",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_of(event: Dict[str, Any]) -> str:
    fields = [
        str(event.get("event_type", "")),
        str(event.get("event_name", "")),
        str(event.get("event_text", "")),
        str(event.get("source_table", "")),
        str(event.get("quality_flag", "")),
    ]
    return " ".join(fields).lower()


def supports_acute(event: Dict[str, Any]) -> bool:
    t = text_of(event)
    return any(term in t for term in ACUTE_TERMS)


def supports_chronic(event: Dict[str, Any]) -> bool:
    t = text_of(event)
    return any(term in t for term in CHRONIC_TERMS)


def get_quality(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample.get("quality_flags", {}) or {}


def get_subtype(sample: Dict[str, Any]) -> str:
    return str(sample.get("sample_subtype", "") or "")


def bool_flag(quality: Dict[str, Any], key: str) -> bool:
    value = quality.get(key, False)

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1", "present", "positive", "high", "moderate", "mild"}:
            return True
        if v in {"false", "no", "n", "0", "none", "null", "na", "nan", "absent", "negative"}:
            return False
        return False

    return False


def stable_high_creatinine_flag(quality: Dict[str, Any]) -> bool:
    value = quality.get("stable_high_creatinine_level", False)

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value > 0

    if isinstance(value, str):
        v = value.strip().lower()
        return v not in {"", "none", "false", "no", "0", "null", "na", "nan"}

    return False



def score_visible_evidence(sample: Dict[str, Any]) -> Dict[str, Any]:
    quality = get_quality(sample)
    subtype = get_subtype(sample)

    acute_score = 0.0
    chronic_score = 0.0
    uncertain_score = 0.5

    acute_ids = []
    chronic_ids = []
    partial_reasons = []

    # 1. Event-level evidence from visible events only
    for event in sample["visible_events"]:
        if supports_acute(event):
            acute_score += 1.0
            acute_ids.append(event["event_id"])

        if supports_chronic(event):
            chronic_score += 1.0
            chronic_ids.append(event["event_id"])

    # 2. Quality flags from your v2.1 dataset
    if bool_flag(quality, "has_visible_trend"):
        acute_score += 1.0
        partial_reasons.append("visible_creatinine_trend")

    trend_conf = str(quality.get("trend_confidence", "")).lower()
    if trend_conf in {"high", "strong"}:
        acute_score += 0.75
    elif trend_conf in {"low", "weak", "moderate"}:
        acute_score += 0.25
        partial_reasons.append("low_or_moderate_trend_confidence")

    if bool_flag(quality, "low_confidence_ratio_trigger"):
        acute_score += 0.35
        uncertain_score += 0.75
        partial_reasons.append("low_confidence_ratio_trigger")

    if stable_high_creatinine_flag(quality):
        chronic_score += 1.25
        partial_reasons.append("stable_high_creatinine_level")

    if bool_flag(quality, "has_aki_ckd_comorbidity"):
        acute_score += 0.5
        chronic_score += 0.5
        uncertain_score += 0.5
        partial_reasons.append("aki_ckd_comorbidity")

    # 3. Subtype-level conservative interpretation
    if subtype == "acute_by_visible_trend":
        acute_score += 1.0

    elif subtype == "low_confidence_ratio_trigger":
        acute_score += 0.35
        uncertain_score += 0.75
        partial_reasons.append("subtype_low_confidence_ratio_trigger")

    elif subtype == "chronic_by_stable_high_creatinine":
        chronic_score += 1.5

    elif subtype == "comorbid_aki_ckd":
        acute_score += 0.75
        chronic_score += 0.75
        uncertain_score += 0.5
        partial_reasons.append("subtype_comorbid_aki_ckd")

    elif subtype == "chronic_by_future_ckd_code":
        # Important: future CKD code cannot be used for initial query-time support.
        # It can only be considered in delayed reattribution.
        uncertain_score += 1.25
        partial_reasons.append("future_ckd_code_not_visible_initially")

    elif subtype == "acute_by_future_diagnosis_or_rrt_only":
        # Important: future diagnosis/RRT cannot support initial claim.
        uncertain_score += 1.25
        partial_reasons.append("future_acute_evidence_not_visible_initially")

    elif subtype == "uncertain_stable_creatinine":
        uncertain_score += 1.5

    # 4. No real visible evidence -> uncertainty dominates
    if acute_score == 0 and chronic_score == 0:
        uncertain_score += 1.5

    return {
        "acute_score": acute_score,
        "chronic_score": chronic_score,
        "uncertain_score": uncertain_score,
        "acute_supporting_ids": acute_ids,
        "chronic_supporting_ids": chronic_ids,
        "partial_reasons": partial_reasons,
    }


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = {
        "acute_renal_deterioration": max(scores["acute_score"], 0.01),
        "chronic_renal_dysfunction": max(scores["chronic_score"], 0.01),
        "uncertain_or_transient_abnormality": max(scores["uncertain_score"], 0.01),
    }
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def determine_claim_status(scores: Dict[str, Any], primary: str) -> str:
    acute = scores["acute_score"]
    chronic = scores["chronic_score"]
    uncertain = scores["uncertain_score"]
    partial_reasons = scores["partial_reasons"]

    top_non_uncertain = max(acute, chronic)
    margin = abs(acute - chronic)

    # No visible support: insufficient
    if top_non_uncertain < 1.0:
        return "insufficient"

    # Uncertainty dominates and non-uncertain evidence is weak
    if uncertain > top_non_uncertain and top_non_uncertain < 1.5:
        return "insufficient"

    # Future-only patterns cannot support query-time claim
    if any("future_" in r for r in partial_reasons):
        return "insufficient"

    # Low-confidence ratio should be partial, not supported
    if any("low_confidence" in r for r in partial_reasons):
        return "partially_supported"

    # Mixed acute/chronic only becomes partial if scores are close
    if acute > 0 and chronic > 0 and margin < 0.75:
        return "partially_supported"

    # Clear acute or chronic dominance
    if primary in {
        "acute_renal_deterioration",
        "chronic_renal_dysfunction",
    }:
        return "supported"

    return "insufficient"


def verify_claim(sample: Dict[str, Any]) -> Dict[str, Any]:
    scores = score_visible_evidence(sample)
    belief = normalize_scores(scores)
    primary = max(belief, key=belief.get)

    claim_status = determine_claim_status(scores, primary)

    supporting_ids = []
    if primary == "acute_renal_deterioration":
        supporting_ids = scores["acute_supporting_ids"]
    elif primary == "chronic_renal_dysfunction":
        supporting_ids = scores["chronic_supporting_ids"]

    prediction = {
        "sample_id": sample["sample_id"],
        "method_name": "cbelief_v0_2_1_flagfix",
        "claim_status": claim_status,
        "primary_hypothesis": primary,
        "belief_distribution": belief,

        # Query-time supporting evidence only
        "supporting_evidence_ids": supporting_ids,
        "contradictory_evidence_ids": [],

        # Explicitly separated evidence
        "retrospective_only_evidence_ids": sample["retrospective_event_ids"],
        "future_only_evidence_ids": sample["future_event_ids"],
        "misused_future_evidence_ids": [],

        "missing_evidence": [],
        "partial_reasons": scores["partial_reasons"],

        "rationale": (
            "C-BELIEF v0.1 used only query-time visible evidence for claim verification. "
            "Future and retrospective evidence were excluded from query-time support and "
            "reserved for delayed reattribution."
        ),
    }

    return prediction


def run(input_path: str, output_path: str) -> None:
    samples = load_jsonl(input_path)
    preds = [verify_claim(s) for s in samples]
    save_jsonl(preds, output_path)
    print(f"Saved {len(preds)} predictions to {output_path}")


# if __name__ == "__main__":
#     run("data/adapted/val.jsonl", "data/predictions/cbelief/val_predictions_v0_1.jsonl")

if __name__ == "__main__":
    run(
        "data/adapted/test.jsonl",
        "data/predictions/cbelief/test_predictions_v0_2_1_flagfix.jsonl",
    )