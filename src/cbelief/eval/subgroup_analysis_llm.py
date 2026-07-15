import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


SAMPLES_PATH = "data/llm/val_subset_100.jsonl"

METHODS = {
    "LLM_visible_only_raw": "data/predictions/llm/llm_visible_only_n100_predictions.jsonl",
    "LLM_full_context_unmarked_raw": "data/predictions/llm/llm_full_context_unmarked_n100_predictions.jsonl",
    "LLM_full_context_with_provenance_raw": "data/predictions/llm/llm_full_context_with_provenance_n100_predictions.jsonl",
    "LLM_CBELIEF_prompted_raw": "data/predictions/llm/llm_cbelief_prompted_n100_predictions.jsonl",
    "LLM_CBELIEF_prompted_postprocessed": "data/predictions/llm/llm_cbelief_prompted_n100_postprocessed_predictions.jsonl",
}

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


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def event_id_sets(sample: Dict[str, Any]):
    visible = {
        str(e["event_id"])
        for e in sample.get("visible_events", []) or []
        if e.get("event_id") is not None
    }

    future = {
        str(e["event_id"])
        for e in sample.get("future_observed_events", []) or []
        if e.get("event_id") is not None
    }

    retro = {
        str(e["event_id"])
        for e in sample.get("retrospective_events", []) or []
        if e.get("event_id") is not None
    }

    return visible, future, retro


def temporal_leakage_rate(samples, preds) -> float:
    num = 0
    den = 0

    for s, p in zip(samples, preds):
        _, future, _ = event_id_sets(s)
        support = [str(x) for x in p.get("supporting_evidence_ids", []) or []]
        den += len(support)
        num += sum(1 for x in support if x in future)

    return safe_div(num, den)


def retrospective_misuse_rate(samples, preds) -> float:
    num = 0
    den = 0

    for s, p in zip(samples, preds):
        _, _, retro = event_id_sets(s)
        support = [str(x) for x in p.get("supporting_evidence_ids", []) or []]
        den += len(support)
        num += sum(1 for x in support if x in retro)

    return safe_div(num, den)


def retrospective_recognition_rate(samples, preds) -> float:
    num = 0
    den = 0

    for s, p in zip(samples, preds):
        _, _, retro = event_id_sets(s)
        pred_retro = {str(x) for x in p.get("retrospective_only_evidence_ids", []) or []}

        den += len(retro)
        num += len(retro & pred_retro)

    return safe_div(num, den)


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
        "temporal_leakage_rate": temporal_leakage_rate(samples, preds),
        "retrospective_misuse_rate": retrospective_misuse_rate(samples, preds),
        "retrospective_recognition_rate": retrospective_recognition_rate(samples, preds),
        "gold_claim_distribution": dict(Counter(gold_claim)),
        "pred_claim_distribution": dict(Counter(pred_claim)),
        "gold_hyp_distribution": dict(Counter(gold_hyp)),
        "pred_hyp_distribution": dict(Counter(pred_hyp)),
    }


def main():
    samples = load_jsonl(SAMPLES_PATH)
    rows = []

    for method, pred_path in METHODS.items():
        preds = load_jsonl(pred_path)
        assert len(samples) == len(preds), f"{method}: samples/preds mismatch"

        grouped_samples = defaultdict(list)
        grouped_preds = defaultdict(list)

        for s, p in zip(samples, preds):
            subtype = s.get("sample_subtype", "unknown")
            grouped_samples[subtype].append(s)
            grouped_preds[subtype].append(p)

        for subtype in sorted(grouped_samples.keys()):
            sub_samples = grouped_samples[subtype]
            sub_preds = grouped_preds[subtype]

            m = compute_metrics(sub_samples, sub_preds)

            rows.append({
                "method": method,
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
            })

    df = pd.DataFrame(rows)

    out = Path("data/metrics/paper_tables/table4_llm_subgroup_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    show_cols = [
        "method",
        "subtype",
        "n",
        "claim_status_accuracy",
        "claim_status_macro_f1",
        "primary_hypothesis_accuracy",
        "primary_hypothesis_macro_f1",
        "temporal_leakage_rate",
        "retrospective_misuse_rate",
        "retrospective_recognition_rate",
    ]

    print(df[show_cols].to_string(index=False))
    print(f"\nSaved to {out}")

    compact = df[df["method"].isin([
        "LLM_visible_only_raw",
        "LLM_full_context_unmarked_raw",
        "LLM_full_context_with_provenance_raw",
        "LLM_CBELIEF_prompted_postprocessed",
    ])].copy()

    compact_out = Path("data/metrics/paper_tables/table4b_llm_subgroup_compact.csv")
    compact.to_csv(compact_out, index=False, encoding="utf-8-sig")
    print(f"Saved compact table to {compact_out}")


if __name__ == "__main__":
    main()