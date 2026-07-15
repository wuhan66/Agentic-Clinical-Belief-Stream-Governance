import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METHOD_NAME = "cbelief_v0_3_clean_visible"


ACUTE = "acute_renal_deterioration"
CHRONIC = "chronic_renal_dysfunction"
UNCERTAIN = "uncertain_or_transient_abnormality"


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


def event_text(e: Dict[str, Any]) -> str:
    parts = [
        e.get("event_type", ""),
        e.get("event_name", ""),
        e.get("event_text", ""),
        e.get("source_table", ""),
    ]
    return " ".join(str(x or "").lower() for x in parts)


def event_id(e: Dict[str, Any]) -> Optional[str]:
    x = e.get("event_id")
    if x is None:
        return None
    return str(x)


def get_numeric_value(e: Dict[str, Any]) -> Optional[float]:
    for key in ["value_num", "value", "numeric_value"]:
        v = e.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def is_creatinine_event(e: Dict[str, Any]) -> bool:
    t = event_text(e)
    return "creatinine" in t or "肌酐" in t


def is_aki_text_event(e: Dict[str, Any]) -> bool:
    t = event_text(e)
    terms = [
        "aki",
        "acute kidney injury",
        "acute renal failure",
        "acute renal insufficiency",
    ]
    return any(x in t for x in terms)


def is_ckd_text_event(e: Dict[str, Any]) -> bool:
    t = event_text(e)
    terms = [
        "ckd",
        "chronic kidney disease",
        "chronic renal disease",
        "chronic renal insufficiency",
        "chronic renal failure",
    ]
    return any(x in t for x in terms)


def build_visible_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    events = sample.get("visible_events", []) or []

    creatinine_values: List[float] = []
    creatinine_ids: List[str] = []
    aki_text_ids: List[str] = []
    ckd_text_ids: List[str] = []

    for e in events:
        eid = event_id(e)

        if is_creatinine_event(e):
            v = get_numeric_value(e)
            if v is not None:
                creatinine_values.append(v)
                if eid:
                    creatinine_ids.append(eid)

        if is_aki_text_event(e) and eid:
            aki_text_ids.append(eid)

        if is_ckd_text_event(e) and eid:
            ckd_text_ids.append(eid)

    min_cr = min(creatinine_values) if creatinine_values else None
    max_cr = max(creatinine_values) if creatinine_values else None
    first_cr = creatinine_values[0] if creatinine_values else None
    last_cr = creatinine_values[-1] if creatinine_values else None

    cr_range = 0.0
    if min_cr is not None and max_cr is not None:
        cr_range = max_cr - min_cr

    acute_delta = 0.0
    acute_ratio = 1.0

    if first_cr is not None and last_cr is not None:
        acute_delta = last_cr - first_cr
        if first_cr > 0:
            acute_ratio = last_cr / first_cr

    has_clear_acute_trend = False
    has_low_confidence_acute_trend = False

    if len(creatinine_values) >= 2:
        has_clear_acute_trend = acute_delta >= 0.3 or acute_ratio >= 1.5
        has_low_confidence_acute_trend = (
            not has_clear_acute_trend
            and acute_delta > 0
            and acute_ratio >= 1.3
        )

    has_stable_high_creatinine = False
    if len(creatinine_values) >= 2:
        has_stable_high_creatinine = (
            min_cr is not None
            and min_cr >= 1.3
            and cr_range <= 0.5
        )

    return {
        "n_visible_events": len(events),
        "creatinine_values": creatinine_values,
        "creatinine_ids": creatinine_ids,
        "aki_text_ids": aki_text_ids,
        "ckd_text_ids": ckd_text_ids,
        "min_cr": min_cr,
        "max_cr": max_cr,
        "first_cr": first_cr,
        "last_cr": last_cr,
        "cr_range": cr_range,
        "acute_delta": acute_delta,
        "acute_ratio": acute_ratio,
        "has_clear_acute_trend": has_clear_acute_trend,
        "has_low_confidence_acute_trend": has_low_confidence_acute_trend,
        "has_stable_high_creatinine": has_stable_high_creatinine,
        "has_aki_text": len(aki_text_ids) > 0,
        "has_ckd_text": len(ckd_text_ids) > 0,
    }


def score_visible_evidence(sample: Dict[str, Any]) -> Dict[str, Any]:
    f = build_visible_features(sample)

    acute_score = 0.0
    chronic_score = 0.0
    uncertain_score = 0.0

    acute_ids: List[str] = []
    chronic_ids: List[str] = []
    partial_reasons: List[str] = []

    if f["has_clear_acute_trend"]:
        acute_score += 2.0
        acute_ids.extend(f["creatinine_ids"])
        partial_reasons.append("visible_creatinine_trend")

    if f["has_low_confidence_acute_trend"]:
        acute_score += 1.0
        acute_ids.extend(f["creatinine_ids"])
        partial_reasons.append("low_confidence_visible_creatinine_trend")

    if f["has_aki_text"]:
        acute_score += 1.5
        acute_ids.extend(f["aki_text_ids"])
        partial_reasons.append("visible_aki_text")

    if f["has_stable_high_creatinine"]:
        chronic_score += 1.25
        chronic_ids.extend(f["creatinine_ids"])
        partial_reasons.append("stable_high_creatinine_level")

    if f["has_ckd_text"]:
        chronic_score += 2.0
        chronic_ids.extend(f["ckd_text_ids"])
        partial_reasons.append("visible_ckd_text")

    if acute_score == 0.0 and chronic_score == 0.0:
        uncertain_score = 1.0

    acute_ids = sorted(set(acute_ids))
    chronic_ids = sorted(set(chronic_ids))

    return {
        "acute_score": acute_score,
        "chronic_score": chronic_score,
        "uncertain_score": uncertain_score,
        "acute_supporting_ids": acute_ids,
        "chronic_supporting_ids": chronic_ids,
        "partial_reasons": sorted(set(partial_reasons)),
        "visible_features": f,
    }


def determine_primary_hypothesis(scores: Dict[str, Any]) -> str:
    acute = scores["acute_score"]
    chronic = scores["chronic_score"]
    uncertain = scores["uncertain_score"]

    if acute == 0.0 and chronic == 0.0:
        return UNCERTAIN

    if acute >= chronic and acute > 0:
        return ACUTE

    if chronic > acute and chronic > 0:
        return CHRONIC

    if uncertain > 0:
        return UNCERTAIN

    return UNCERTAIN


def determine_claim_status(scores: Dict[str, Any], primary: str) -> str:
    acute = scores["acute_score"]
    chronic = scores["chronic_score"]
    partial_reasons = scores["partial_reasons"]

    has_low_confidence = "low_confidence_visible_creatinine_trend" in partial_reasons
    has_stable_chronic = "stable_high_creatinine_level" in partial_reasons
    has_visible_ckd_text = "visible_ckd_text" in partial_reasons

    if primary == UNCERTAIN:
        return "insufficient"

    if has_low_confidence:
        return "partially_supported"

    if primary == CHRONIC:
        if has_stable_chronic and not has_visible_ckd_text:
            return "partially_supported"
        if chronic >= 1.0:
            return "supported"

    if primary == ACUTE:
        if acute >= 1.5:
            return "supported"
        if acute >= 1.0:
            return "partially_supported"

    return "insufficient"


def belief_distribution(scores: Dict[str, Any]) -> Dict[str, float]:
    acute = max(float(scores["acute_score"]), 0.0)
    chronic = max(float(scores["chronic_score"]), 0.0)
    uncertain = max(float(scores["uncertain_score"]), 0.0)

    total = acute + chronic + uncertain
    if total <= 0:
        return {
            ACUTE: 0.0,
            CHRONIC: 0.0,
            UNCERTAIN: 1.0,
        }

    return {
        ACUTE: acute / total,
        CHRONIC: chronic / total,
        UNCERTAIN: uncertain / total,
    }


def make_prediction(sample: Dict[str, Any]) -> Dict[str, Any]:
    scores = score_visible_evidence(sample)
    primary = determine_primary_hypothesis(scores)
    claim_status = determine_claim_status(scores, primary)

    support_ids: List[str] = []
    if primary == ACUTE:
        support_ids = scores["acute_supporting_ids"]
    elif primary == CHRONIC:
        support_ids = scores["chronic_supporting_ids"]

    return {
        "sample_id": sample.get("sample_id"),
        "method_name": METHOD_NAME,
        "claim_status": claim_status,
        "primary_hypothesis": primary,
        "belief_distribution": belief_distribution(scores),
        "supporting_evidence_ids": sorted(set(support_ids)),
        "retrospective_only_evidence_ids": [],
        "partial_reasons": scores["partial_reasons"],
        "debug_visible_features": {
            "n_visible_events": scores["visible_features"]["n_visible_events"],
            "min_cr": scores["visible_features"]["min_cr"],
            "max_cr": scores["visible_features"]["max_cr"],
            "first_cr": scores["visible_features"]["first_cr"],
            "last_cr": scores["visible_features"]["last_cr"],
            "cr_range": scores["visible_features"]["cr_range"],
            "acute_delta": scores["visible_features"]["acute_delta"],
            "acute_ratio": scores["visible_features"]["acute_ratio"],
            "has_clear_acute_trend": scores["visible_features"]["has_clear_acute_trend"],
            "has_low_confidence_acute_trend": scores["visible_features"]["has_low_confidence_acute_trend"],
            "has_stable_high_creatinine": scores["visible_features"]["has_stable_high_creatinine"],
            "has_aki_text": scores["visible_features"]["has_aki_text"],
            "has_ckd_text": scores["visible_features"]["has_ckd_text"],
        },
    }


def run(samples_path: str, out_path: str) -> None:
    samples = load_jsonl(samples_path)
    preds = [make_prediction(s) for s in samples]
    save_jsonl(preds, out_path)
    print(f"Saved {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    run(
        "data/adapted/test.jsonl",
        "data/predictions/cbelief/test_predictions_v0_3_clean_visible.jsonl",
    )