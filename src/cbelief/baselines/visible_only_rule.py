import json
from pathlib import Path
from typing import Dict, Any, List


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
    return " ".join([
        str(event.get("event_type", "")),
        str(event.get("event_name", "")),
        str(event.get("event_text", "")),
        str(event.get("source_table", "")),
        str(event.get("visibility", "")),
    ]).lower()


def supports_acute(event: Dict[str, Any]) -> bool:
    t = text_of(event)
    return any(term in t for term in ACUTE_TERMS)


def supports_chronic(event: Dict[str, Any]) -> bool:
    t = text_of(event)
    return any(term in t for term in CHRONIC_TERMS)


def normalize_scores(acute: float, chronic: float, uncertain: float) -> Dict[str, float]:
    vals = {
        "acute_renal_deterioration": max(acute, 0.01),
        "chronic_renal_dysfunction": max(chronic, 0.01),
        "uncertain_or_transient_abnormality": max(uncertain, 0.01),
    }
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}


def predict(sample: Dict[str, Any]) -> Dict[str, Any]:
    visible_events = sample.get("visible_events", [])

    acute_ids = []
    chronic_ids = []

    for event in visible_events:
        if supports_acute(event):
            acute_ids.append(event["event_id"])
        if supports_chronic(event):
            chronic_ids.append(event["event_id"])

    acute_score = float(len(acute_ids))
    chronic_score = float(len(chronic_ids))
    uncertain_score = 1.0 if acute_score == 0 and chronic_score == 0 else 0.25

    belief = normalize_scores(acute_score, chronic_score, uncertain_score)
    primary = max(belief, key=belief.get)

    if acute_score > 0 or chronic_score > 0:
        claim_status = "supported"
    else:
        claim_status = "insufficient"

    if primary == "acute_renal_deterioration":
        supporting_ids = acute_ids
    elif primary == "chronic_renal_dysfunction":
        supporting_ids = chronic_ids
    else:
        supporting_ids = []

    return {
        "sample_id": sample["sample_id"],
        "method_name": "visible_only_rule",
        "claim_status": claim_status,
        "primary_hypothesis": primary,
        "belief_distribution": belief,
        "supporting_evidence_ids": supporting_ids,
        "contradictory_evidence_ids": [],
        "retrospective_only_evidence_ids": [],
        "future_only_evidence_ids": [],
        "misused_future_evidence_ids": [],
        "missing_evidence": [],
        "rationale": "Visible-only rule baseline used only query-time visible evidence.",
    }


def run(input_path: str, output_path: str) -> None:
    samples = load_jsonl(input_path)
    preds = [predict(s) for s in samples]
    save_jsonl(preds, output_path)
    print(f"Saved {len(preds)} predictions to {output_path}")


if __name__ == "__main__":
    run(
        "data/adapted/test.jsonl",
        "data/predictions/visible_only_rule/test_predictions.jsonl",
    )