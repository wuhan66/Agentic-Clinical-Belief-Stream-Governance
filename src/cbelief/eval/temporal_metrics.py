from __future__ import annotations


def temporal_leakage_rate(samples: list[dict], preds: list[dict]) -> float:
    num = den = 0
    for s, p in zip(samples, preds):
        selected = p.get("supporting_evidence_ids", [])
        future = set(s.get("future_event_ids", []))
        den += len(selected)
        num += sum(1 for e in selected if e in future)
    return num / max(den, 1)


def retrospective_evidence_misuse_rate(samples: list[dict], preds: list[dict]) -> float:
    num = den = 0
    for s, p in zip(samples, preds):
        selected = p.get("supporting_evidence_ids", [])
        retro = set(s.get("retrospective_event_ids", []))
        den += len(selected)
        num += sum(1 for e in selected if e in retro)
    return num / max(den, 1)


def retrospective_recognition_rate(samples: list[dict], preds: list[dict]) -> float:
    num = den = 0
    for s, p in zip(samples, preds):
        gold = set(s.get("retrospective_event_ids", []))
        pred = set(p.get("retrospective_only_evidence_ids", []))
        den += len(gold)
        num += len(gold & pred)
    return num / max(den, 1)


def temporal_metrics(samples: list[dict], preds: list[dict]) -> dict:
    return {
        "temporal_leakage_rate": temporal_leakage_rate(samples, preds),
        "retrospective_evidence_misuse_rate": retrospective_evidence_misuse_rate(samples, preds),
        "retrospective_recognition_rate": retrospective_recognition_rate(samples, preds),
    }
