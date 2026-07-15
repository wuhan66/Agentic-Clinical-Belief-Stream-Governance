from __future__ import annotations


class EvidenceAdjudicator:
    def adjudicate(self, sample: dict, events: list[dict], temporal: dict[str, str]) -> list[dict]:
        outputs = []
        for e in events:
            eid = e["event_id"]
            tstat = temporal.get(eid, "unknown")
            relation, hyp = self.infer_relation(e.get("event_text", ""), tstat)
            outputs.append({
                "event_id": eid,
                "event_text": e.get("event_text", ""),
                "relation": relation,
                "temporal_status": tstat,
                "target_hypothesis": hyp,
                "strength": self.strength(relation, tstat),
            })
        return outputs

    def infer_relation(self, text: str, temporal_status: str) -> tuple[str, str]:
        lower = text.lower()
        if temporal_status in {"future_not_visible", "retrospective_only"}:
            if any(k in lower for k in ["aki", "acute kidney", "creatinine", "ckd", "dialysis", "rrt"]):
                hyp = "acute_renal_deterioration" if any(k in lower for k in ["aki", "acute", "rise", "dialysis", "rrt"]) else "chronic_renal_dysfunction"
                return "temporally_invalid", hyp
        if any(k in lower for k in ["creatinine_rise", "increased", "aki", "acute kidney", "dialysis", "rrt"]):
            return "support", "acute_renal_deterioration"
        if any(k in lower for k in ["ckd", "chronic kidney", "chronic renal"]):
            return "support", "chronic_renal_dysfunction"
        if any(k in lower for k in ["improved", "transient", "isolated"]):
            return "support", "uncertain_or_transient_abnormality"
        return "irrelevant", "uncertain_or_transient_abnormality"

    def strength(self, relation: str, temporal_status: str) -> float:
        if relation == "support" and temporal_status == "visible_before_query":
            return 1.0
        if relation == "temporally_invalid":
            return 0.0
        if relation == "irrelevant":
            return 0.0
        return 0.25
