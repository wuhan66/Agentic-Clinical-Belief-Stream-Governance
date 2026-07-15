from __future__ import annotations

import re

KEYWORDS = {
    "acute": ["creatinine_rise", "aki", "acute kidney", "acute renal", "dialysis", "rrt"],
    "chronic": ["ckd", "chronic kidney", "chronic renal"],
    "uncertain": ["single", "isolated", "transient", "improved"],
}


def _parse_stream_item(text: str) -> dict:
    # Expected: "EVT_ID: event text". Fallback keeps whole text.
    if ":" in text:
        eid, body = text.split(":", 1)
        return {"event_id": eid.strip(), "event_text": body.strip()}
    return {"event_id": text[:40], "event_text": text}


class EvidenceRetriever:
    def __init__(self, top_k: int = 8):
        self.top_k = top_k

    def retrieve(self, sample: dict) -> list[dict]:
        rows = []
        for group, label in [
            (sample.get("visible_stream", []), "visible"),
            (sample.get("retrospective_stream", []), "retrospective"),
            (sample.get("future_stream", []), "future"),
        ]:
            for item in group:
                event = _parse_stream_item(item)
                event["stream_group"] = label
                event["score"] = self.score(event["event_text"], sample.get("target_claim", ""))
                rows.append(event)
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        return rows[: self.top_k]

    def score(self, text: str, claim: str) -> float:
        t = text.lower()
        score = 0.0
        for kws in KEYWORDS.values():
            score += sum(1.0 for kw in kws if kw in t)
        # simple claim overlap
        claim_terms = set(re.findall(r"[a-zA-Z_]+", claim.lower()))
        text_terms = set(re.findall(r"[a-zA-Z_]+", t))
        score += len(claim_terms & text_terms) / max(len(claim_terms), 1)
        return score
