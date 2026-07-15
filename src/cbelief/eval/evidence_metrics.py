from __future__ import annotations


def support_precision_at_k(samples: list[dict], preds: list[dict], k: int = 3) -> float:
    # Uses weak/silver evidence_relations if present. If absent, returns NaN-like None.
    vals = []
    for s, p in zip(samples, preds):
        rels = s.get("silver_label", {}).get("evidence_relations", [])
        gold_support = {r["event_id"] for r in rels if r.get("relation") in {"support", "supports"}}
        if not gold_support:
            continue
        selected = p.get("supporting_evidence_ids", [])[:k]
        vals.append(len(set(selected) & gold_support) / max(len(selected), 1))
    return sum(vals) / len(vals) if vals else None


def evidence_metrics(samples: list[dict], preds: list[dict]) -> dict:
    return {"support_precision_at_3": support_precision_at_k(samples, preds, k=3)}
