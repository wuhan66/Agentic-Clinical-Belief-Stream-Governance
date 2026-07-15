import json
from pathlib import Path
from typing import Dict, Any, List


HYPOTHESES = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
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


def supports_acute(event: Dict[str, Any]) -> bool:
    name = str(event.get("event_name", "")).lower()
    text = str(event.get("event_text", "")).lower()

    acute_terms = [
        "creatinine_rise_48h",
        "creatinine_rise_7d",
        "acute kidney injury",
        "acute renal",
        "aki",
        "dialysis",
        "renal replacement",
        "rrt",
    ]

    return any(term in name or term in text for term in acute_terms)


def supports_chronic(event: Dict[str, Any]) -> bool:
    name = str(event.get("event_name", "")).lower()
    text = str(event.get("event_text", "")).lower()

    chronic_terms = [
        "ckd",
        "chronic kidney",
        "chronic renal",
        "chronic renal dysfunction",
    ]

    return any(term in name or term in text for term in chronic_terms)


def score_visible_evidence(sample: Dict[str, Any]) -> Dict[str, float]:
    acute = 0.0
    chronic = 0.0

    supporting_ids = []
    chronic_ids = []

    for event in sample["visible_events"]:
        if supports_acute(event):
            acute += 1.0
            supporting_ids.append(event["event_id"])

        if supports_chronic(event):
            chronic += 1.0
            chronic_ids.append(event["event_id"])

    # uncertainty gets weight when evidence is weak or mixed
    uncertain = 0.5
    if acute == 0 and chronic == 0:
        uncertain = 2.0
    elif acute > 0 and chronic > 0:
        uncertain = 1.0

    return {
        "acute_score": acute,
        "chronic_score": chronic,
        "uncertain_score": uncertain,
        "acute_supporting_ids": supporting_ids,
        "chronic_supporting_ids": chronic_ids,
    }


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = {
        "acute_renal_deterioration": max(scores["acute_score"], 0.01),
        "chronic_renal_dysfunction": max(scores["chronic_score"], 0.01),
        "uncertain_or_transient_abnormality": max(scores["uncertain_score"], 0.01),
    }
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def verify_claim(sample: Dict[str, Any]) -> Dict[str, Any]:
    scores = score_visible_evidence(sample)
    belief = normalize_scores(scores)

    primary = max(belief, key=belief.get)

    acute = scores["acute_score"]
    chronic = scores["chronic_score"]

    if acute > 0 and acute >= chronic:
        claim_status = "supported"
        supporting_ids = scores["acute_supporting_ids"]
    elif acute > 0 and chronic > acute:
        claim_status = "partially_supported"
        supporting_ids = scores["acute_supporting_ids"]
    elif acute == 0 and chronic > 0:
        claim_status = "contradicted"
        supporting_ids = scores["chronic_supporting_ids"]
    else:
        claim_status = "insufficient"
        supporting_ids = []

    retrospective_ids = sample["retrospective_event_ids"]
    future_ids = sample["future_event_ids"]

    prediction = {
        "sample_id": sample["sample_id"],
        "method_name": "cbelief_v0_rule_skill",
        "claim_status": claim_status,
        "primary_hypothesis": primary,
        "belief_distribution": belief,
        "supporting_evidence_ids": supporting_ids,
        "contradictory_evidence_ids": [],
        "retrospective_only_evidence_ids": retrospective_ids,
        "future_only_evidence_ids": future_ids,
        "misused_future_evidence_ids": [],
        "missing_evidence": [],
        "rationale": "C-BELIEF v0 used only visible evidence for query-time claim verification; retrospective and future evidence were excluded from query-time support.",
    }

    return prediction


def run(input_path: str, output_path: str) -> None:
    samples = load_jsonl(input_path)
    preds = [verify_claim(s) for s in samples]
    save_jsonl(preds, output_path)
    print(f"Saved {len(preds)} predictions to {output_path}")


if __name__ == "__main__":
    run("data/adapted/val.jsonl", "data/predictions/cbelief/val_predictions.jsonl")