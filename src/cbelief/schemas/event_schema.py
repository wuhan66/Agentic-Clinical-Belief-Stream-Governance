from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


Visibility = Literal[
    "observed_at_time",
    "retrospective_or_discharge_level",
    "retrospective_note",
    "future_after_query",
    "unknown",
]

EventType = Literal["lab", "trend", "diagnosis", "procedure", "note", "state", "other"]

QualityFlag = Literal["clean", "weak", "retrospective", "temporal_uncertain", "low_confidence"]


class RenalEvent(BaseModel):
    event_id: str
    subject_id: int
    hadm_id: int | None = None
    event_time: str | None = None
    event_type: EventType
    event_name: str
    event_value: str | None = None
    event_text: str
    visibility: Visibility = "unknown"
    quality_flag: QualityFlag = "clean"
    source_table: str | None = None
    source_row_id: str | None = None
    ai_rule_confidence: float | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
