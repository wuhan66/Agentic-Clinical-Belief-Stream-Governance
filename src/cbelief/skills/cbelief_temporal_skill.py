import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


CLAIM_LABELS = {
    "supported",
    "partially_supported",
    "insufficient",
}

HYP_LABELS = {
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
}

UNCERTAIN_HYP = "uncertain_or_transient_abnormality"


def canonicalize_claim_status(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", "", s)

    mapping = {
        "supported": "supported",
        "partially_supported": "partially_supported",
        "partiallysupported": "partially_supported",
        "partial": "partially_supported",
        "insufficient": "insufficient",
        "insufficent": "insufficient",
        "insufficien": "insufficient",
    }

    return mapping.get(s, "insufficient")


def canonicalize_hypothesis(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", "", s)

    mapping = {
        "acute_renal_deterioration": "acute_renal_deterioration",
        "acuterenaldeterioration": "acute_renal_deterioration",
        "acute": "acute_renal_deterioration",

        "chronic_renal_dysfunction": "chronic_renal_dysfunction",
        "chronicrenaldysfunction": "chronic_renal_dysfunction",
        "chronic": "chronic_renal_dysfunction",

        "uncertain_or_transient_abnormality": "uncertain_or_transient_abnormality",
        "uncertainortransientabnormality": "uncertain_or_transient_abnormality",
        "uncertain": "uncertain_or_transient_abnormality",
        "transient": "uncertain_or_transient_abnormality",
    }

    return mapping.get(s, UNCERTAIN_HYP)


def normalize_id_list(x: Any) -> List[str]:
    if x is None:
        return []

    if not isinstance(x, list):
        return []

    out = []
    for item in x:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)

    # preserve order, remove duplicates
    seen = set()
    deduped = []
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def evidence_id_sets(sample: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    visible_ids = {
        str(e["event_id"])
        for e in sample.get("visible_events", []) or []
        if e.get("event_id") is not None
    }

    future_ids = {
        str(e["event_id"])
        for e in sample.get("future_observed_events", []) or []
        if e.get("event_id") is not None
    }

    retrospective_ids = {
        str(e["event_id"])
        for e in sample.get("retrospective_events", []) or []
        if e.get("event_id") is not None
    }

    return visible_ids, future_ids, retrospective_ids


def repair_prediction_with_cbelief_skill(
    sample: Dict[str, Any],
    pred: Dict[str, Any],
) -> Dict[str, Any]:
    """
    C-BELIEF Temporal Reasoning Skill.

    This is not a clinical rule model.
    It is a runtime constraint layer that enforces temporal validity:

    1. supporting_evidence_ids may contain only query-time visible evidence.
    2. retrospective evidence is moved to retrospective_only_evidence_ids.
    3. future evidence is removed from support and recorded as invalid.
    4. uncertain hypothesis forces insufficient claim and empty support.
    5. supported / partially_supported claims without valid support are downgraded.
    """

    visible_ids, future_ids, retrospective_ids = evidence_id_sets(sample)
    all_known_ids = visible_ids | future_ids | retrospective_ids

    repaired = dict(pred)

    original_claim = canonicalize_claim_status(pred.get("claim_status"))
    original_hyp = canonicalize_hypothesis(pred.get("primary_hypothesis"))

    support_ids = normalize_id_list(pred.get("supporting_evidence_ids"))
    retro_only_ids = normalize_id_list(pred.get("retrospective_only_evidence_ids"))

    visible_support = [eid for eid in support_ids if eid in visible_ids]
    future_support = [eid for eid in support_ids if eid in future_ids]
    retrospective_support = [eid for eid in support_ids if eid in retrospective_ids]
    unknown_support = [eid for eid in support_ids if eid not in all_known_ids]

    repairs = []

    if future_support:
        repairs.append("removed_future_evidence_from_support")

    if retrospective_support:
        repairs.append("moved_retrospective_support_to_retrospective_only")

    if unknown_support:
        repairs.append("removed_unknown_support_ids")

    # Retrospective evidence can be retained only as retrospective-only evidence.
    merged_retro_only = retro_only_ids + retrospective_support
    merged_retro_only = list(dict.fromkeys(merged_retro_only))

    claim = original_claim
    hyp = original_hyp

    # Invariant: uncertain/transient hypothesis cannot support a positive renal pathology claim.
    if hyp == UNCERTAIN_HYP:
        if claim != "insufficient" or visible_support:
            repairs.append("uncertain_hypothesis_forces_insufficient_claim")
        claim = "insufficient"
        visible_support = []

    # Invariant: positive claim requires valid query-time support.
    if claim in {"supported", "partially_supported"} and len(visible_support) == 0:
        repairs.append("positive_claim_without_visible_support_forces_insufficient")
        claim = "insufficient"

    repaired["claim_status"] = claim
    repaired["primary_hypothesis"] = hyp
    repaired["supporting_evidence_ids"] = visible_support
    repaired["retrospective_only_evidence_ids"] = merged_retro_only

    # Extra diagnostic fields. Existing evaluator will ignore these.
    repaired["future_only_evidence_ids"] = list(dict.fromkeys(future_support))
    repaired["invalid_support_ids"] = list(dict.fromkeys(future_support + unknown_support))
    repaired["cbelief_skill_trace"] = {
        "original_claim_status": original_claim,
        "original_primary_hypothesis": original_hyp,
        "original_supporting_evidence_ids": support_ids,
        "visible_support_ids_kept": visible_support,
        "future_support_ids_removed": future_support,
        "retrospective_support_ids_moved": retrospective_support,
        "unknown_support_ids_removed": unknown_support,
        "repairs": repairs,
    }

    return repaired


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