from __future__ import annotations

import math

HYPOTHESES = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


class ClaimVerifier:
    def verify(self, sample: dict, adjudication: list[dict]) -> dict:
        scores = {h: 0.0 for h in HYPOTHESES}
        support_visible = []
        retro_only = []
        invalid = []
        for a in adjudication:
            if a["temporal_status"] == "visible_before_query" and a["relation"] == "support":
                scores[a["target_hypothesis"]] += a.get("strength", 1.0)
                support_visible.append(a["event_id"])
            elif a["temporal_status"] == "retrospective_only":
                retro_only.append(a["event_id"])
            elif a["temporal_status"] == "future_not_visible" or a["relation"] == "temporally_invalid":
                invalid.append(a["event_id"])
        if sum(scores.values()) == 0:
            scores["uncertain_or_transient_abnormality"] = 1.0
        belief = self.softmax(scores)
        primary = max(belief, key=belief.get)
        if support_visible and primary != "uncertain_or_transient_abnormality":
            status = "supported"
        elif retro_only and not support_visible:
            status = "insufficient"
        else:
            status = "insufficient"
        return {
            "claim_status": status,
            "primary_hypothesis": primary,
            "belief_distribution": belief,
            "supporting_evidence_ids": support_visible,
            "retrospective_only_evidence_ids": retro_only,
            "temporally_invalid_evidence_ids": invalid,
        }

    def softmax(self, scores: dict[str, float]) -> dict[str, float]:
        mx = max(scores.values())
        exp = {k: math.exp(v - mx) for k, v in scores.items()}
        denom = sum(exp.values())
        return {k: v / denom for k, v in exp.items()}
