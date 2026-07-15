# 功能说明：
# 1. 定义 Agentic C-BELIEF 多智能体输出的标准字段和允许标签。
# 2. 提供轻量校验函数，避免 LLM 输出越界标签影响后续评估。

from __future__ import annotations

from typing import Any, Dict, Iterable

CLAIM_STATUS = {"supported", "partially_supported", "insufficient"}
PRIMARY_HYPOTHESIS = {
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
}
FINAL_PHENOTYPE = PRIMARY_HYPOTHESIS | {"mixed_acute_on_chronic_renal_dysfunction"}
EVIDENCE_TIME_CLASS = {"visible", "future", "retrospective", "unknown"}
MIXED_PHENOTYPE = "mixed_acute_on_chronic_renal_dysfunction"

DEFAULT_PREDICTION: Dict[str, Any] = {
    "initial_claim_status": "insufficient",
    "final_claim_status": "insufficient",
    "initial_primary_hypothesis": "uncertain_or_transient_abnormality",
    "final_primary_hypothesis": "uncertain_or_transient_abnormality",
    "final_clinical_phenotype": "uncertain_or_transient_abnormality",
    "requires_delayed_reattribution": False,
    "initial_supporting_evidence_ids": [],
    "final_supporting_evidence_ids": [],
    "rationale": "",
}


def normalize_label(value: Any, allowed: Iterable[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in set(allowed) else default


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize final coordinator prediction to the expected schema."""
    out = dict(DEFAULT_PREDICTION)
    if isinstance(pred, dict):
        out.update(pred)

    out["initial_claim_status"] = normalize_label(
        out.get("initial_claim_status"), CLAIM_STATUS, DEFAULT_PREDICTION["initial_claim_status"]
    )
    out["final_claim_status"] = normalize_label(
        out.get("final_claim_status"), CLAIM_STATUS, DEFAULT_PREDICTION["final_claim_status"]
    )
    out["initial_primary_hypothesis"] = normalize_label(
        out.get("initial_primary_hypothesis"), PRIMARY_HYPOTHESIS, DEFAULT_PREDICTION["initial_primary_hypothesis"]
    )
    final_primary_raw = str(out.get("final_primary_hypothesis") or "").strip()
    if final_primary_raw == MIXED_PHENOTYPE:
        final_primary_raw = "acute_renal_deterioration"
    out["final_primary_hypothesis"] = normalize_label(
        final_primary_raw, PRIMARY_HYPOTHESIS, DEFAULT_PREDICTION["final_primary_hypothesis"]
    )
    out["final_clinical_phenotype"] = normalize_label(
        out.get("final_clinical_phenotype"), FINAL_PHENOTYPE, DEFAULT_PREDICTION["final_clinical_phenotype"]
    )
    out["requires_delayed_reattribution"] = normalize_bool(out.get("requires_delayed_reattribution"))

    for key in ["initial_supporting_evidence_ids", "final_supporting_evidence_ids"]:
        value = out.get(key, [])
        if isinstance(value, list):
            out[key] = [str(x).strip() for x in value if str(x).strip()]
        elif isinstance(value, str) and value.strip():
            out[key] = [value.strip()]
        else:
            out[key] = []

    out["rationale"] = str(out.get("rationale", ""))
    return out
