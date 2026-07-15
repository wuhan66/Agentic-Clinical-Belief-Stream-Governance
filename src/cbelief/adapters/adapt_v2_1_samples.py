import json
from pathlib import Path
from typing import Any, Dict, List, Optional


RETROSPECTIVE_EVENT_TYPES = {
    "diagnosis",
    "note",
    "discharge_note",
    "discharge_summary",
}

RETROSPECTIVE_VISIBILITY = {
    "retrospective_or_discharge_level",
    "retrospective_note",
    "discharge_level",
}

DEFAULT_HYPOTHESES = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_event_text(event: Dict[str, Any]) -> str:
    event_type = event.get("event_type", "event")
    event_time = event.get("event_time", "")
    visibility = event.get("visibility", "unknown")
    text = event.get("event_text", "")

    return f"[{event_type.upper()}][{event_time}][{visibility}] {text}"


def is_retrospective_event(event: Dict[str, Any]) -> bool:
    event_type = str(event.get("event_type", "")).lower()
    event_name = str(event.get("event_name", "")).lower()
    visibility = str(event.get("visibility", "")).lower()
    source_table = str(event.get("source_table", "")).lower()
    text = str(event.get("event_text", "")).lower()

    if event_type in RETROSPECTIVE_EVENT_TYPES:
        return True

    if visibility in RETROSPECTIVE_VISIBILITY:
        return True

    if "diagnoses_icd" in source_table:
        return True

    if "discharge" in source_table:
        return True

    if "discharge diagnosis" in text:
        return True

    if "discharge note" in text:
        return True

    return False


def partition_future_events(
    future_events: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    future_observed = []
    retrospective = []

    for event in future_events:
        if is_retrospective_event(event):
            retrospective.append(event)
        else:
            future_observed.append(event)

    return future_observed, retrospective


def event_ids(events: List[Dict[str, Any]]) -> List[str]:
    return [e["event_id"] for e in events if "event_id" in e]


def adapt_one(raw: Dict[str, Any]) -> Dict[str, Any]:
    weak = raw.get("weak_label_v2_1") or raw.get("weak_label_v2") or raw.get("weak_label") or {}
    quality = raw.get("quality_flags_v2_1") or raw.get("quality_flags") or {}

    visible_events = raw.get("visible_events", [])
    future_events_all = raw.get("future_events", [])

    future_observed_events, retrospective_events = partition_future_events(future_events_all)

    visible_stream = raw.get("visible_stream") or [
        normalize_event_text(e) for e in visible_events
    ]

    future_observed_stream = [
        normalize_event_text(e) for e in future_observed_events
    ]

    retrospective_stream = [
        normalize_event_text(e) for e in retrospective_events
    ]

    adapted = {
        "sample_id": raw["sample_id"],
        "subject_id": raw.get("subject_id"),
        "hadm_id": raw.get("hadm_id"),
        "query_time": raw.get("query_time"),
        "sample_group": raw.get("sample_group"),
        "sample_subtype": raw.get("sample_subtype_v2_1") or raw.get("sample_subtype"),
        "clinical_process": "renal_deterioration",

        "target_claim": raw.get("target_claim"),
        "target_claim_initial": raw.get("target_claim_initial") or raw.get("target_claim"),
        "target_claim_final": raw.get("target_claim_final") or raw.get("target_claim"),

        "candidate_hypotheses": raw.get("candidate_hypotheses", DEFAULT_HYPOTHESES),

        "visible_events": visible_events,
        "future_observed_events": future_observed_events,
        "retrospective_events": retrospective_events,

        "visible_event_ids": event_ids(visible_events),
        "future_event_ids": event_ids(future_observed_events),
        "retrospective_event_ids": event_ids(retrospective_events),

        "visible_stream": visible_stream,
        "future_stream": future_observed_stream,
        "retrospective_stream": retrospective_stream,

        "all_candidate_events": visible_events + future_observed_events + retrospective_events,

        "silver_label": {
            "initial_claim_status": weak.get("initial_claim_status"),
            "final_claim_status": weak.get("final_claim_status"),
            "initial_primary_hypothesis": weak.get("initial_primary_hypothesis"),
            "final_primary_hypothesis": weak.get("final_primary_hypothesis"),
            "initial_hypothesis_distribution": weak.get("initial_hypothesis_distribution"),
            "final_hypothesis_distribution": weak.get("final_hypothesis_distribution"),
            "requires_delayed_reattribution": weak.get("requires_delayed_reattribution"),
            "temporal_interpretation": weak.get("temporal_interpretation"),
            "final_clinical_phenotype": weak.get("final_clinical_phenotype"),
            "subgroup_for_analysis": weak.get("subgroup_for_analysis"),
            "evidence_phase": weak.get("evidence_phase"),
        },

        "quality_flags": quality,
        "metadata": raw.get("metadata", {}),
        "split": raw.get("split"),
    }

    return adapted


def adapt_file(input_path: str, output_path: str) -> None:
    rows = load_jsonl(input_path)
    adapted = [adapt_one(row) for row in rows]
    save_jsonl(adapted, output_path)

    n_retro = sum(len(x["retrospective_event_ids"]) for x in adapted)
    n_future = sum(len(x["future_event_ids"]) for x in adapted)
    n_visible = sum(len(x["visible_event_ids"]) for x in adapted)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(adapted)}")
    print(f"Visible events: {n_visible}")
    print(f"Future observed events: {n_future}")
    print(f"Retrospective events: {n_retro}")


if __name__ == "__main__":
    adapt_file("data/stream/train_v2_1.jsonl", "data/adapted/train.jsonl")
    adapt_file("data/stream/val_v2_1.jsonl", "data/adapted/val.jsonl")
    adapt_file("data/stream/test_v2_1.jsonl", "data/adapted/test.jsonl")