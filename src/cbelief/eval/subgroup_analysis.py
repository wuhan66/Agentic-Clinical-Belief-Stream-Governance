import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


METHODS = {
    "A_visible_only_rule": "data/predictions/visible_only_rule/test_predictions.jsonl",
    "B_full_context_leaky_rule": "data/predictions/full_context_leaky_rule/test_predictions.jsonl",
    "C_full_context_with_provenance_rule": "data/predictions/full_context_with_provenance_rule/test_predictions.jsonl",
    "D_cbelief_v0_2_margin": "data/predictions/cbelief/test_predictions_v0_2_margin.jsonl",
}

SAMPLES_PATH = "data/adapted/test.jsonl"

CLAIM_LABELS = ["supported", "partially_supported", "insufficient"]

HYP_LABELS = [
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


def compute_metrics(samples, preds) -> Dict[str, Any]:
    gold_claim = [s["silver_label"]["initial_claim_status"] for s in samples]
    pred_claim = [p["claim_status"] for p in preds]

    gold_hyp = [s["silver_label"]["initial_primary_hypothesis"] for s in samples]
    pred_hyp = [p["primary_hypothesis"] for p in preds]

    return {
        "n": len(samples),
        "claim_status_accuracy": accuracy_score(gold_claim, pred_claim),
        "claim_status_macro_f1": f1_score(
            gold_claim,
            pred_claim,
            labels=CLAIM_LABELS,
            average="macro",
            zero_division=0,
        ),
        "primary_hypothesis_accuracy": accuracy_score(gold_hyp, pred_hyp),
        "primary_hypothesis_macro_f1": f1_score(
            gold_hyp,
            pred_hyp,
            labels=HYP_LABELS,
            average="macro",
            zero_division=0,
        ),
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

    all_rows = []

    for method_name, pred_path in METHODS.items():
        preds = load_jsonl(pred_path)
        assert len(samples) == len(preds), f"{method_name}: sample/pred length mismatch"

        by_subtype_samples = defaultdict(list)
        by_subtype_preds = defaultdict(list)

        for s, p in zip(samples, preds):
            subtype = s.get("sample_subtype", "unknown")
            by_subtype_samples[subtype].append(s)
            by_subtype_preds[subtype].append(p)

        for subtype in sorted(by_subtype_samples.keys()):
            sub_samples = by_subtype_samples[subtype]
            sub_preds = by_subtype_preds[subtype]

            m = compute_metrics(sub_samples, sub_preds)
            row = {
                "method": method_name,
                "subtype": subtype,
                "n": m["n"],
                "claim_status_accuracy": m["claim_status_accuracy"],
                "claim_status_macro_f1": m["claim_status_macro_f1"],
                "primary_hypothesis_accuracy": m["primary_hypothesis_accuracy"],
                "primary_hypothesis_macro_f1": m["primary_hypothesis_macro_f1"],
                "temporal_leakage_rate": m["temporal_leakage_rate"],
                "retrospective_misuse_rate": m["retrospective_misuse_rate"],
                "retrospective_recognition_rate": m["retrospective_recognition_rate"],
                "gold_claim_distribution": json.dumps(m["gold_claim_distribution"], ensure_ascii=False),
                "pred_claim_distribution": json.dumps(m["pred_claim_distribution"], ensure_ascii=False),
                "gold_hyp_distribution": json.dumps(m["gold_hyp_distribution"], ensure_ascii=False),
                "pred_hyp_distribution": json.dumps(m["pred_hyp_distribution"], ensure_ascii=False),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    out = Path("data/metrics/subgroup_results_test.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df[
        [
            "method",
            "subtype",
            "n",
            "claim_status_macro_f1",
            "primary_hypothesis_macro_f1",
            "temporal_leakage_rate",
            "retrospective_misuse_rate",
            "retrospective_recognition_rate",
        ]
    ].to_string(index=False))

    print(f"\nSaved to {out}")

    # Also print a compact comparison between visible-only and C-BELIEF
    print("\nCompact comparison: A_visible_only_rule vs D_cbelief_v0_2_margin")
    compact = df[df["method"].isin(["A_visible_only_rule", "D_cbelief_v0_2_margin"])]
    pivot = compact.pivot_table(
        index="subtype",
        columns="method",
        values=[
            "claim_status_macro_f1",
            "primary_hypothesis_macro_f1",
            "temporal_leakage_rate",
            "retrospective_misuse_rate",
        ],
        aggfunc="first",
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()