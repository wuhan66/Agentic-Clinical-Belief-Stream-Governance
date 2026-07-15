from __future__ import annotations

RETRO_VIS = {"retrospective_or_discharge_level", "retrospective_note"}


class TemporalValidator:
    def validate(self, sample: dict, events: list[dict]) -> dict[str, str]:
        visible = set(sample.get("visible_event_ids", []))
        future = set(sample.get("future_event_ids", []))
        retro = set(sample.get("retrospective_event_ids", []))
        out = {}
        for e in events:
            eid = e["event_id"]
            if eid in visible:
                out[eid] = "visible_before_query"
            elif eid in retro or e.get("visibility") in RETRO_VIS:
                out[eid] = "retrospective_only"
            elif eid in future:
                out[eid] = "future_not_visible"
            else:
                out[eid] = "unknown"
        return out
