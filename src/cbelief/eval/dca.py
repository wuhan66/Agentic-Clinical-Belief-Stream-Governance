from __future__ import annotations


def net_benefit(y_true: list[int], y_prob: list[float], threshold: float) -> float:
    y_pred = [p >= threshold for p in y_prob]
    tp = sum(1 for yp, yt in zip(y_pred, y_true) if yp and yt == 1)
    fp = sum(1 for yp, yt in zip(y_pred, y_true) if yp and yt == 0)
    n = max(len(y_true), 1)
    if threshold >= 1:
        return 0.0
    return tp / n - fp / n * (threshold / (1 - threshold))
