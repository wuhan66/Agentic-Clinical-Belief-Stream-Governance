import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


SAMPLES_PATH = "data/adapted/test.jsonl"

# PRED_FILES = {
#     "A_visible_only_rule": "data/predictions/visible_only_rule/test_predictions.jsonl",
#     "D_cbelief_v0_2_margin": "data/predictions/cbelief/test_predictions_v0_2_margin.jsonl",
# }

PRED_FILES = {
    "A_visible_only_rule": "data/predictions/visible_only_rule/test_predictions.jsonl",
    "D_cbelief_v0_2_2_chronic_claimfix": "data/predictions/cbelief/test_predictions_v0_2_2_chronic_claimfix.jsonl",
}

KEY_SUBGROUPS = [
    "comorbid_aki_ckd",
    "chronic_by_stable_high_creatinine",
    "low_confidence_ratio_trigger",
]

CLAIM_LABELS = ["supported", "partially_supported", "insufficient"]

HYP_LABELS = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_event(event):
    return {
        "event_id": event.get("event_id"),
        "event_time": event.get("event_time"),
        "event_type": event.get("event_type"),
        "event_name": event.get("event_name"),
        "value_num": event.get("value_num"),
        "unit": event.get("unit"),
        "event_text": event.get("event_text"),
        "visibility": event.get("visibility"),
        "source_table": event.get("source_table"),
        "quality_flag": event.get("quality_flag"),
    }


def label_of(sample):
    return sample["silver_label"]


def analyze_method(samples, preds, method_name):
    rows = []
    error_examples = []

    for subgroup in KEY_SUBGROUPS:
        sub_items = [
            (s, p)
            for s, p in zip(samples, preds)
            if s.get("sample_subtype") == subgroup
        ]

        if not sub_items:
            continue

        sub_samples = [x[0] for x in sub_items]
        sub_preds = [x[1] for x in sub_items]

        gold_claim = [label_of(s)["initial_claim_status"] for s in sub_samples]
        pred_claim = [p["claim_status"] for p in sub_preds]

        gold_hyp = [label_of(s)["initial_primary_hypothesis"] for s in sub_samples]
        pred_hyp = [p["primary_hypothesis"] for p in sub_preds]

        claim_cm = confusion_matrix(
            gold_claim,
            pred_claim,
            labels=CLAIM_LABELS,
        )

        hyp_cm = confusion_matrix(
            gold_hyp,
            pred_hyp,
            labels=HYP_LABELS,
        )

        print("\n" + "=" * 100)
        print(f"Method: {method_name}")
        print(f"Subgroup: {subgroup}")
        print(f"n = {len(sub_items)}")

        print("\nClaim labels:", CLAIM_LABELS)
        print(claim_cm)

        print("\nHypothesis labels:", HYP_LABELS)
        print(hyp_cm)

        print("\nClaim classification report:")
        print(classification_report(
            gold_claim,
            pred_claim,
            labels=CLAIM_LABELS,
            zero_division=0,
        ))

        print("\nHypothesis classification report:")
        print(classification_report(
            gold_hyp,
            pred_hyp,
            labels=HYP_LABELS,
            zero_division=0,
        ))

        claim_errors = sum(g != p for g, p in zip(gold_claim, pred_claim))
        hyp_errors = sum(g != p for g, p in zip(gold_hyp, pred_hyp))

        rows.append({
            "method": method_name,
            "subgroup": subgroup,
            "n": len(sub_items),
            "claim_errors": claim_errors,
            "claim_error_rate": claim_errors / len(sub_items),
            "hypothesis_errors": hyp_errors,
            "hypothesis_error_rate": hyp_errors / len(sub_items),
            "gold_claim_distribution": json.dumps(dict(Counter(gold_claim)), ensure_ascii=False),
            "pred_claim_distribution": json.dumps(dict(Counter(pred_claim)), ensure_ascii=False),
            "gold_hyp_distribution": json.dumps(dict(Counter(gold_hyp)), ensure_ascii=False),
            "pred_hyp_distribution": json.dumps(dict(Counter(pred_hyp)), ensure_ascii=False),
        })

        # Collect representative errors.
        for s, p, gc, pc, gh, ph in zip(
            sub_samples,
            sub_preds,
            gold_claim,
            pred_claim,
            gold_hyp,
            pred_hyp,
        ):
            if gc != pc or gh != ph:
                q = s.get("quality_flags", {})
                error_examples.append({
                    "method": method_name,
                    "subgroup": subgroup,
                    "sample_id": s.get("sample_id"),
                    "subject_id": s.get("subject_id"),
                    "hadm_id": s.get("hadm_id"),
                    "query_time": s.get("query_time"),

                    "gold_claim": gc,
                    "pred_claim": pc,
                    "gold_hypothesis": gh,
                    "pred_hypothesis": ph,

                    "target_claim_initial": s.get("target_claim_initial"),
                    "target_claim_final": s.get("target_claim_final"),

                    "quality_flags": {
                        "has_visible_trend": q.get("has_visible_trend"),
                        "trend_confidence": q.get("trend_confidence"),
                        "low_confidence_ratio_trigger": q.get("low_confidence_ratio_trigger"),
                        "has_aki_ckd_comorbidity": q.get("has_aki_ckd_comorbidity"),
                        "stable_high_creatinine_level": q.get("stable_high_creatinine_level"),
                        "visible_context_length_group": q.get("visible_context_length_group"),
                        "n_visible_events": q.get("n_visible_events"),
                        "n_future_events": q.get("n_future_events"),
                        "min_visible_creatinine": q.get("min_visible_creatinine"),
                        "max_visible_creatinine": q.get("max_visible_creatinine"),
                        "last_visible_creatinine": q.get("last_visible_creatinine"),
                        "visible_creatinine_range": q.get("visible_creatinine_range"),
                        "has_future_aki_evidence": q.get("has_future_aki_evidence"),
                        "has_future_ckd_evidence": q.get("has_future_ckd_evidence"),
                        "has_future_rrt_evidence": q.get("has_future_rrt_evidence"),
                        "has_future_note_aki": q.get("has_future_note_aki"),
                        "has_future_note_ckd": q.get("has_future_note_ckd"),
                    },

                    "prediction_partial_reasons": p.get("partial_reasons", []),
                    "prediction_belief_distribution": p.get("belief_distribution", {}),
                    "supporting_evidence_ids": p.get("supporting_evidence_ids", []),

                    "visible_events": [short_event(e) for e in s.get("visible_events", [])],
                    "future_observed_events": [short_event(e) for e in s.get("future_observed_events", [])[:5]],
                    "retrospective_events": [short_event(e) for e in s.get("retrospective_events", [])[:5]],
                })

    return rows, error_examples


def summarize_error_patterns(error_examples):
    pattern_counter = Counter()

    for ex in error_examples:
        key = (
            ex["method"],
            ex["subgroup"],
            ex["gold_claim"],
            ex["pred_claim"],
            ex["gold_hypothesis"],
            ex["pred_hypothesis"],
        )
        pattern_counter[key] += 1

    print("\n" + "=" * 100)
    print("Top error patterns:")
    for key, count in pattern_counter.most_common(30):
        method, subgroup, gc, pc, gh, ph = key
        print(
            f"{count:4d} | {method} | {subgroup} | "
            f"claim {gc} -> {pc} | hyp {gh} -> {ph}"
        )


def main():
    samples = load_jsonl(SAMPLES_PATH)

    all_summary_rows = []
    all_error_examples = []

    for method_name, pred_path in PRED_FILES.items():
        preds = load_jsonl(pred_path)
        assert len(samples) == len(preds), f"{method_name}: samples/preds length mismatch"

        rows, errors = analyze_method(samples, preds, method_name)
        all_summary_rows.extend(rows)
        all_error_examples.extend(errors)

    summarize_error_patterns(all_error_examples)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_out = Path("data/metrics/key_subgroup_error_summary.csv")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    errors_out = "data/metrics/key_subgroup_error_examples.jsonl"
    save_jsonl(all_error_examples, errors_out)

    print("\nSaved:")
    print(summary_out)
    print(errors_out)

    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()