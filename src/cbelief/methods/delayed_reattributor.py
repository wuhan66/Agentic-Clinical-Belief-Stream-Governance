from __future__ import annotations


class DelayedReattributor:
    def reattribute(self, sample: dict, verification: dict) -> list[dict]:
        records = []
        retro_ids = set(sample.get("retrospective_event_ids", []))
        for eid in verification.get("retrospective_only_evidence_ids", []):
            if eid in retro_ids:
                records.append({
                    "triggering_evidence_id": eid,
                    "new_interpretation": verification.get("primary_hypothesis"),
                    "valid_for_query_time": False,
                    "reason": "Retrospective evidence can update post-hoc interpretation but cannot count as query-time support.",
                })
        return records
