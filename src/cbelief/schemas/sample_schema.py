from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

Hypothesis = Literal[
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]


class CBeliefSample(BaseModel):
    sample_id: str
    subject_id: int
    hadm_id: int | None = None
    query_time: str
    clinical_process: str = "renal_deterioration"

    visible_event_ids: list[str] = Field(default_factory=list)
    future_event_ids: list[str] = Field(default_factory=list)
    retrospective_event_ids: list[str] = Field(default_factory=list)

    visible_stream: list[str] = Field(default_factory=list)
    future_stream: list[str] = Field(default_factory=list)
    retrospective_stream: list[str] = Field(default_factory=list)

    target_claim: str
    candidate_hypotheses: list[Hypothesis]
    silver_label: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
