from __future__ import annotations

from sklearn.metrics import brier_score_loss


def brier_for_hypothesis(samples: list[dict], preds: list[dict], hypothesis: str) -> float:
    y = [1 if s.get("silver_label", {}).get("primary_hypothesis") == hypothesis else 0 for s in samples]
    prob = [float(p.get("belief_distribution", {}).get(hypothesis, 0.0)) for p in preds]
    return brier_score_loss(y, prob)
