from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT = Path(
    r"D:\下载\npj_digital_medicine\outputs\local_ehr_pilot\local_cbelief_stream_300.jsonl"
)
DEFAULT_OUTPUT = Path("data/local_ehr_pilot/local_cbelief_stream_300_eval_compatible.jsonl")

DEFAULT_HYPOTHESES = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def event_id(event: Dict[str, Any], prefix: str, index: int) -> str:
    return str(event.get("event_id") or event.get("evidence_id") or f"{prefix}_{index:03d}")


def event_text(event: Dict[str, Any]) -> str:
    return str(
        event.get("event_text")
        or event.get("evidence_line")
        or event.get("text")
        or json.dumps(event, ensure_ascii=False)
    )


def normalize_event(event: Dict[str, Any], prefix: str, index: int) -> Dict[str, Any]:
    ev_id = event_id(event, prefix, index)
    text = event_text(event)
    return {
        "event_id": ev_id,
        "evidence_id": event.get("evidence_id", ev_id),
        "event_time": event.get("event_time"),
        "event_type": event.get("event_type") or "event",
        "event_name": event.get("item_name") or event.get("event_name") or event.get("event_type") or "event",
        "value_num": event.get("value"),
        "unit": event.get("unit"),
        "event_text": text,
        "source_table": event.get("source") or event.get("source_table"),
        "source_row_id": event.get("source_row_id"),
        "visibility": event.get("visibility"),
        "quality_flag": event.get("quality_flag") or event.get("derived_from") or "local_pilot",
        "local_evidence": event,
    }


def stream_line(event: Dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "event").upper()
    event_time = event.get("event_time") or ""
    visibility = event.get("visibility") or "unknown"
    text = event.get("event_text") or ""
    return f"[{event_type}][{event_time}][{visibility}] {text}"


def normalize_events(events: Any, prefix: str) -> List[Dict[str, Any]]:
    if not isinstance(events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for i, event in enumerate(events):
        if isinstance(event, dict):
            normalized.append(normalize_event(event, prefix, i))
        else:
            normalized.append(
                {
                    "event_id": f"{prefix}_{i:03d}",
                    "event_type": "event",
                    "event_name": "event",
                    "event_text": str(event),
                    "visibility": "unknown",
                    "quality_flag": "local_pilot",
                    "local_evidence": event,
                }
            )
    return normalized


def ids(events: List[Dict[str, Any]]) -> List[str]:
    return [str(event["event_id"]) for event in events if event.get("event_id")]


def hypothesis_distribution(primary: Any) -> Dict[str, float]:
    primary = str(primary or "uncertain_or_transient_abnormality")
    if primary not in DEFAULT_HYPOTHESES:
        primary = "uncertain_or_transient_abnormality"
    return {hyp: (1.0 if hyp == primary else 0.0) for hyp in DEFAULT_HYPOTHESES}


def build_label(raw: Dict[str, Any], sample_subtype: str) -> Dict[str, Any]:
    gold = dict(raw.get("gold_labels") or {})
    initial_primary = gold.get("initial_primary_hypothesis")
    final_primary = gold.get("final_primary_hypothesis")
    return {
        "initial_claim_status": gold.get("initial_claim_status"),
        "final_claim_status": gold.get("final_claim_status"),
        "initial_primary_hypothesis": initial_primary,
        "final_primary_hypothesis": final_primary,
        "initial_hypothesis_distribution": hypothesis_distribution(initial_primary),
        "final_hypothesis_distribution": hypothesis_distribution(final_primary),
        "requires_delayed_reattribution": bool(gold.get("requires_delayed_reattribution")),
        "temporal_interpretation": (
            "Initial assessment uses only visible_stream. Future_stream contains post-query "
            "events; retrospective_stream contains discharge-level or retrospective evidence."
        ),
        "final_clinical_phenotype": gold.get("final_clinical_phenotype"),
        "subgroup_for_analysis": sample_subtype,
        "evidence_phase": "visible_only_to_final_update",
        "initial_supporting_evidence_ids": gold.get("initial_supporting_evidence_ids", []),
        "final_supporting_evidence_ids": gold.get("final_supporting_evidence_ids", []),
        "retrospective_only_evidence_ids": gold.get("retrospective_only_evidence_ids", []),
    }


def build_quality_flags(
    raw: Dict[str, Any],
    visible_events: List[Dict[str, Any]],
    future_events: List[Dict[str, Any]],
    retrospective_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence_counts = raw.get("evidence_counts") or {}
    needs_review = raw.get("needs_review_reasons") or []
    return {
        "label_status": raw.get("label_status"),
        "needs_human_audit": bool(raw.get("needs_human_audit")),
        "needs_review_reasons": needs_review if isinstance(needs_review, list) else [str(needs_review)],
        "n_visible_events": len(visible_events),
        "n_future_events": len(future_events),
        "n_retrospective_events": len(retrospective_events),
        "n_total_events": evidence_counts.get(
            "total", len(visible_events) + len(future_events) + len(retrospective_events)
        ),
        "visible_context_length_group": "empty" if not visible_events else "normal",
        "has_future_evidence": bool(future_events),
        "has_retrospective_evidence": bool(retrospective_events),
        "local_schema_version": raw.get("schema_version"),
    }


def adapt_one(raw: Dict[str, Any]) -> Dict[str, Any]:
    visible_events = normalize_events(raw.get("visible_evidence"), "visible_evidence")
    future_events = normalize_events(raw.get("future_evidence"), "future_evidence")
    retrospective_events = normalize_events(raw.get("retrospective_evidence"), "retrospective_evidence")

    sample_subtype = str(raw.get("temporal_trap_subtype") or raw.get("sample_subtype") or "local_ehr_pilot")
    target_claim = (
        raw.get("target_claim")
        or raw.get("target_claim_initial")
        or "Assess whether the patient has renal deterioration at the query time."
    )

    adapted = {
        "sample_id": raw["sample_id"],
        "subject_id": raw.get("patient_id"),
        "hadm_id": raw.get("visit_id"),
        "patient_id": raw.get("patient_id"),
        "visit_id": raw.get("visit_id"),
        "query_time": raw.get("query_time"),
        "sample_group": raw.get("original_local_outcome", {}).get("outcome_name"),
        "sample_subtype": sample_subtype,
        "clinical_process": "renal_deterioration",
        "target_claim": target_claim,
        "target_claim_initial": raw.get("target_claim_initial") or target_claim,
        "target_claim_final": raw.get("target_claim_final") or target_claim,
        "candidate_hypotheses": DEFAULT_HYPOTHESES,
        "visible_events": visible_events,
        "future_observed_events": future_events,
        "retrospective_events": retrospective_events,
        "visible_event_ids": ids(visible_events),
        "future_event_ids": ids(future_events),
        "retrospective_event_ids": ids(retrospective_events),
        "visible_stream": [stream_line(event) for event in visible_events],
        "future_stream": [stream_line(event) for event in future_events],
        "retrospective_stream": [stream_line(event) for event in retrospective_events],
        "all_candidate_events": visible_events + future_events + retrospective_events,
        "weak_label_v2_2": build_label(raw, sample_subtype),
        "silver_label": build_label(raw, sample_subtype),
        "gold_labels": raw.get("gold_labels", {}),
        "quality_flags": build_quality_flags(raw, visible_events, future_events, retrospective_events),
        "metadata": {
            "source_dataset": raw.get("source_dataset"),
            "schema_version": raw.get("schema_version"),
            "label_status": raw.get("label_status"),
            "admission_time": raw.get("admission_time"),
            "discharge_time": raw.get("discharge_time"),
            "age": raw.get("age"),
            "sex": raw.get("sex"),
            "original_local_outcome": raw.get("original_local_outcome"),
            "audit_fields": raw.get("audit_fields"),
        },
        "split": "local_ehr_pilot",
    }
    return adapted


def adapt_file(input_path: Path, output_path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(input_path)
    adapted = [adapt_one(row) for row in rows]
    write_jsonl(output_path, adapted)
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local EHR pilot C-BELIEF-like JSONL into eval-compatible C-BELIEF samples."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    adapted = adapt_file(args.input_jsonl, args.output_jsonl)

    print(f"[INFO] Input: {args.input_jsonl}")
    print(f"[INFO] Output: {args.output_jsonl}")
    print(f"[INFO] Samples: {len(adapted)}")
    print(f"[INFO] Visible events: {sum(len(row['visible_event_ids']) for row in adapted)}")
    print(f"[INFO] Future observed events: {sum(len(row['future_event_ids']) for row in adapted)}")
    print(f"[INFO] Retrospective events: {sum(len(row['retrospective_event_ids']) for row in adapted)}")
    print(f"[WARN] Labels are provisional local silver labels and still require human audit.")


if __name__ == "__main__":
    main()
