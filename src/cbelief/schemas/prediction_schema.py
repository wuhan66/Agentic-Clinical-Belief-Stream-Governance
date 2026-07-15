from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

ClaimStatus = Literal["supported", "partially_supported", "contradicted", "insufficient"]
Hypothesis = Literal[
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


class CBeliefPrediction(BaseModel):
    sample_id: str
    method_name: str
    claim_status: ClaimStatus
    primary_hypothesis: Hypothesis
    belief_distribution: dict[str, float]

    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    retrospective_only_evidence_ids: list[str] = Field(default_factory=list)
    temporally_invalid_evidence_ids: list[str] = Field(default_factory=list)
    misused_future_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    temporal_validity_summary: str | None = None
    rationale: str | None = None
    raw_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
