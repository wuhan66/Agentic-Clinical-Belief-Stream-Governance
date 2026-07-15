# 功能说明：
# 1. 对 Agentic C-BELIEF 输出进行确定性 temporal skill verification。
# 2. 检查 initial supporting evidence 是否错误跨越 query-time boundary。

from __future__ import annotations

from typing import Any, Dict, List, Set


def _event_ids(events: List[Any], prefix: str) -> Set[str]:
    ids: Set[str] = set()
    for i, ev in enumerate(events or []):
        if isinstance(ev, dict):
            ids.add(str(ev.get("evidence_id") or ev.get("id") or f"{prefix}_{i}"))
        else:
            ids.add(f"{prefix}_{i}")
    return ids


def verify_temporal_support(sample: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    visible = sample.get("visible_stream") or sample.get("visible_summary") or []
    future = sample.get("future_stream") or sample.get("future_summary") or []

    visible_ids = _event_ids(visible, "visible_evidence")
    future_ids = _event_ids(future, "future_or_retrospective_evidence")

    initial_support = set(str(x) for x in prediction.get("initial_supporting_evidence_ids", []) or [])
    invalid_future = sorted(initial_support & future_ids)
    unknown = sorted(x for x in initial_support if x not in visible_ids and x not in future_ids)

    return {
        "n_initial_support": len(initial_support),
        "n_invalid_future_or_retrospective_support": len(invalid_future),
        "n_unknown_support": len(unknown),
        "invalid_future_or_retrospective_support_ids": invalid_future,
        "unknown_support_ids": unknown,
        "has_temporal_misuse": bool(invalid_future),
        "has_unknown_support": bool(unknown),
    }
