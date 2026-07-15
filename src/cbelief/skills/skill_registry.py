from __future__ import annotations

from pathlib import Path
import yaml
from cbelief.skills.skill_schema import SkillSpec


class SkillRegistry:
    def __init__(self, library_dir: str | Path):
        self.library_dir = Path(library_dir)
        self.skills: dict[str, SkillSpec] = {}
        self.load()

    def load(self) -> None:
        if not self.library_dir.exists():
            return
        for p in self.library_dir.glob("*.yaml"):
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            spec = SkillSpec(**data)
            self.skills[spec.skill_id] = spec

    def select(self, event: dict, claim: str) -> list[SkillSpec]:
        selected = []
        for spec in self.skills.values():
            trig = spec.trigger or {}
            ok = True
            for key, val in trig.items():
                if key not in event:
                    continue
                if isinstance(val, list):
                    ok &= event[key] in val
                else:
                    ok &= event[key] == val
            if ok:
                selected.append(spec)
        return selected
