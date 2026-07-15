# 功能说明：
# 1. 在原有 C-BELIEF 架构的 src/cbelief/agents 下实现 training-free 多智能体推理流程。
# 2. 串联 Temporal Gatekeeper、Hypothesis Panel、Evidence Adjudicator、Delayed Reattribution、Coordinator。
# 3. 输出结构化预测和 temporal skill verifier 结果，供后续评估脚本使用。

from __future__ import annotations

from typing import Any, Dict

from .agent_prompts import (
    coordinator_prompt,
    delayed_reattribution_prompt,
    evidence_adjudicator_prompt,
    hypothesis_panel_prompt,
    temporal_gatekeeper_prompt,
)
from .agent_schemas import normalize_prediction
from .json_utils import extract_json_object
from .llm_client import ChatLLMClient
from .temporal_skill_verifier import verify_temporal_support


class AgenticCBeliefPipeline:
    def __init__(self, llm: ChatLLMClient) -> None:
        self.llm = llm

    def _run_agent(self, messages, agent_name: str) -> Dict[str, Any]:
        raw = self.llm.chat(messages)
        parsed = extract_json_object(raw)
        if not parsed:
            tail = raw.strip()[-120:] if raw else ""
            raise ValueError(
                f"{agent_name} returned unparsable JSON. "
                f"Response length={len(raw or '')}; tail={tail!r}"
            )
        parsed["_raw_response"] = raw
        return parsed

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        gatekeeper = self._run_agent(temporal_gatekeeper_prompt(sample), "temporal_gatekeeper")
        panel = self._run_agent(hypothesis_panel_prompt(sample, gatekeeper), "hypothesis_panel")
        adjudicator = self._run_agent(evidence_adjudicator_prompt(sample, gatekeeper, panel), "evidence_adjudicator")
        reattribution = self._run_agent(
            delayed_reattribution_prompt(sample, gatekeeper, panel, adjudicator),
            "delayed_reattribution",
        )
        coordinator = self._run_agent(
            coordinator_prompt(sample, gatekeeper, panel, adjudicator, reattribution),
            "clinical_coordinator",
        )

        prediction = normalize_prediction(coordinator)
        temporal_verification = verify_temporal_support(sample, prediction)

        return {
            "sample_id": sample.get("sample_id", ""),
            "method": "agentic_cbelief_v0",
            "prediction": prediction,
            "agents": {
                "temporal_gatekeeper": gatekeeper,
                "hypothesis_panel": panel,
                "evidence_adjudicator": adjudicator,
                "delayed_reattribution": reattribution,
                "clinical_coordinator": coordinator,
            },
            "temporal_skill_verification": temporal_verification,
        }
