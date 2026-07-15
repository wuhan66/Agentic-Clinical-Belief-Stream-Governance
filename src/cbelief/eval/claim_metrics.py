from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def claim_metrics(samples: list[dict], preds: list[dict]) -> dict:
    gold_status = [s.get("silver_label", {}).get("claim_status", "insufficient") for s in samples]
    pred_status = [p.get("claim_status", "insufficient") for p in preds]
    gold_h = [s.get("silver_label", {}).get("primary_hypothesis", "uncertain_or_transient_abnormality") for s in samples]
    pred_h = [p.get("primary_hypothesis", "uncertain_or_transient_abnormality") for p in preds]
    return {
        "claim_status_accuracy": accuracy_score(gold_status, pred_status),
        "claim_status_macro_f1": f1_score(gold_status, pred_status, average="macro", zero_division=0),
        "primary_hypothesis_accuracy": accuracy_score(gold_h, pred_h),
        "primary_hypothesis_macro_f1": f1_score(gold_h, pred_h, average="macro", zero_division=0),
    }
