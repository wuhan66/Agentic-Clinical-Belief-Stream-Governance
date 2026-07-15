import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score


SAMPLES_PATH = "data/adapted/test.jsonl"

METHODS = {
    "A_visible_only_rule": "data/predictions/visible_only_rule/test_predictions.jsonl",
    "B_full_context_leaky_rule": "data/predictions/full_context_leaky_rule/test_predictions.jsonl",
    "C_full_context_with_provenance_rule": "data/predictions/full_context_with_provenance_rule/test_predictions.jsonl",
    "D_cbelief_v0_2_margin": "data/predictions/cbelief/test_predictions_v0_2_margin.jsonl",
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def temporal_leakage_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        selected = pred.get("supporting_evidence_ids", [])
        future = set(sample.get("future_event_ids", []))

        denominator += len(selected)
        numerator += len([x for x in selected if x in future])

    return safe_div(numerator, denominator)


def retrospective_misuse_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        selected = pred.get("supporting_evidence_ids", [])
        retro = set(sample.get("retrospective_event_ids", []))

        denominator += len(selected)
        numerator += len([x for x in selected if x in retro])

    return safe_div(numerator, denominator)


def retrospective_recognition_rate(preds, samples) -> float:
    numerator = 0
    denominator = 0

    for pred, sample in zip(preds, samples):
        gold_retro = set(sample.get("retrospective_event_ids", []))
        pred_retro = set(pred.get("retrospective_only_evidence_ids", []))

        denominator += len(gold_retro)
        numerator += len(gold_retro & pred_retro)

    return safe_div(numerator, denominator)


def majority_label(labels: List[str]) -> str:
    if not labels:
        return "NA"
    return Counter(labels).most_common(1)[0][0]


def label_recall(gold: List[str], pred: List[str], label: str) -> float:
    denom = sum(1 for g in gold if g == label)
    num = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    return safe_div(num, denom)


def label_precision(gold: List[str], pred: List[str], label: str) -> float:
    denom = sum(1 for p in pred if p == label)
    num = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    return safe_div(num, denom)


def compute_subgroup_metrics(samples, preds) -> Dict[str, Any]:
    gold_claim = [s["silver_label"]["initial_claim_status"] for s in samples]
    pred_claim = [p["claim_status"] for p in preds]

    gold_hyp = [s["silver_label"]["initial_primary_hypothesis"] for s in samples]
    pred_hyp = [p["primary_hypothesis"] for p in preds]

    expected_claim = majority_label(gold_claim)
    expected_hyp = majority_label(gold_hyp)

    return {
        "n": len(samples),

        "claim_status_accuracy": accuracy_score(gold_claim, pred_claim),
        "primary_hypothesis_accuracy": accuracy_score(gold_hyp, pred_hyp),

        "expected_claim_label": expected_claim,
        "expected_claim_recall": label_recall(gold_claim, pred_claim, expected_claim),
        "expected_claim_precision": label_precision(gold_claim, pred_claim, expected_claim),

        "expected_hypothesis_label": expected_hyp,
        "expected_hypothesis_recall": label_recall(gold_hyp, pred_hyp, expected_hyp),
        "expected_hypothesis_precision": label_precision(gold_hyp, pred_hyp, expected_hyp),

        "temporal_leakage_rate": temporal_leakage_rate(preds, samples),
        "retrospective_misuse_rate": retrospective_misuse_rate(preds, samples),
        "retrospective_recognition_rate": retrospective_recognition_rate(preds, samples),

        "gold_claim_distribution": dict(Counter(gold_claim)),
        "pred_claim_distribution": dict(Counter(pred_claim)),
        "gold_hyp_distribution": dict(Counter(gold_hyp)),
        "pred_hyp_distribution": dict(Counter(pred_hyp)),
    }


def main():
    samples = load_jsonl(SAMPLES_PATH)

    rows = []

    for method_name, pred_path in METHODS.items():
        preds = load_jsonl(pred_path)
        assert len(samples) == len(preds), f"{method_name}: samples and predictions length mismatch"

        grouped_samples = defaultdict(list)
        grouped_preds = defaultdict(list)

        for s, p in zip(samples, preds):
            subtype = s.get("sample_subtype", "unknown")
            grouped_samples[subtype].append(s)
            grouped_preds[subtype].append(p)

        for subtype in sorted(grouped_samples.keys()):
            sub_samples = grouped_samples[subtype]
            sub_preds = grouped_preds[subtype]

            m = compute_subgroup_metrics(sub_samples, sub_preds)

            rows.append({
                "method": method_name,
                "subtype": subtype,
                "n": m["n"],

                "claim_status_accuracy": m["claim_status_accuracy"],
                "primary_hypothesis_accuracy": m["primary_hypothesis_accuracy"],

                "expected_claim_label": m["expected_claim_label"],
                "expected_claim_recall": m["expected_claim_recall"],
                "expected_claim_precision": m["expected_claim_precision"],

                "expected_hypothesis_label": m["expected_hypothesis_label"],
                "expected_hypothesis_recall": m["expected_hypothesis_recall"],
                "expected_hypothesis_precision": m["expected_hypothesis_precision"],

                "temporal_leakage_rate": m["temporal_leakage_rate"],
                "retrospective_misuse_rate": m["retrospective_misuse_rate"],
                "retrospective_recognition_rate": m["retrospective_recognition_rate"],

                "gold_claim_distribution": json.dumps(m["gold_claim_distribution"], ensure_ascii=False),
                "pred_claim_distribution": json.dumps(m["pred_claim_distribution"], ensure_ascii=False),
                "gold_hyp_distribution": json.dumps(m["gold_hyp_distribution"], ensure_ascii=False),
                "pred_hyp_distribution": json.dumps(m["pred_hyp_distribution"], ensure_ascii=False),
            })

    df = pd.DataFrame(rows)

    out = Path("data/metrics/subgroup_results_test_v2.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    show_cols = [
        "method",
        "subtype",
        "n",
        "claim_status_accuracy",
        "primary_hypothesis_accuracy",
        "expected_claim_label",
        "expected_claim_recall",
        "expected_hypothesis_label",
        "expected_hypothesis_recall",
        "temporal_leakage_rate",
        "retrospective_misuse_rate",
        "retrospective_recognition_rate",
    ]

    print("\nSubgroup results v2:")
    print(df[show_cols].to_string(index=False))

    print(f"\nSaved to {out}")

    # Compact table: focus on visible-only vs C-BELIEF
    compact = df[df["method"].isin(["A_visible_only_rule", "D_cbelief_v0_2_margin"])].copy()

    compact_out = Path("data/metrics/subgroup_visible_vs_cbelief_test_v2.csv")
    compact.to_csv(compact_out, index=False)

    print(f"Saved compact comparison to {compact_out}")

    print("\nCompact comparison: A_visible_only_rule vs D_cbelief_v0_2_margin")
    print(compact[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()