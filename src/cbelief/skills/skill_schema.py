from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SkillSpec(BaseModel):
    skill_id: str
    name: str | None = None
    version: str = "1.0.0"
    status: str = "locked"
    task: str
    description: str | None = None
    trigger: dict[str, Any] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
