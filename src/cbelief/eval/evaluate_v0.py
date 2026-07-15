import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def temporal_leakage_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        selected = pred.get("supporting_evidence_ids", [])
        future = set(sample.get("future_event_ids", []))
        denominator += len(selected)
        numerator += len([x for x in selected if x in future])

    return numerator / max(denominator, 1)


def retrospective_misuse_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        selected = pred.get("supporting_evidence_ids", [])
        retro = set(sample.get("retrospective_event_ids", []))
        denominator += len(selected)
        numerator += len([x for x in selected if x in retro])

    return numerator / max(denominator, 1)


def retrospective_recognition_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        gold_retro = set(sample.get("retrospective_event_ids", []))
        pred_retro = set(pred.get("retrospective_only_evidence_ids", []))

        denominator += len(gold_retro)
        numerator += len(gold_retro & pred_retro)

    return numerator / max(denominator, 1)


def evaluate(samples_path: str, preds_path: str) -> Dict[str, Any]:
    samples = load_jsonl(samples_path)
    preds = load_jsonl(preds_path)

    assert len(samples) == len(preds), f"samples={len(samples)}, preds={len(preds)}"

    gold_claim = [s["silver_label"]["initial_claim_status"] for s in samples]
    pred_claim = [p["claim_status"] for p in preds]

    gold_hyp = [s["silver_label"]["initial_primary_hypothesis"] for s in samples]
    pred_hyp = [p["primary_hypothesis"] for p in preds]

    return {
        "n": len(samples),
        "claim_status_accuracy": accuracy_score(gold_claim, pred_claim),
        "claim_status_macro_f1": f1_score(gold_claim, pred_claim, average="macro"),
        "primary_hypothesis_accuracy": accuracy_score(gold_hyp, pred_hyp),
        "primary_hypothesis_macro_f1": f1_score(gold_hyp, pred_hyp, average="macro"),
        "temporal_leakage_rate": temporal_leakage_rate(preds, samples),
        "retrospective_misuse_rate": retrospective_misuse_rate(preds, samples),
        "retrospective_recognition_rate": retrospective_recognition_rate(preds, samples),
        "gold_claim_distribution": dict(Counter(gold_claim)),
        "pred_claim_distribution": dict(Counter(pred_claim)),
        "gold_hyp_distribution": dict(Counter(gold_hyp)),
        "pred_hyp_distribution": dict(Counter(pred_hyp)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics = evaluate(args.samples, args.preds)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()