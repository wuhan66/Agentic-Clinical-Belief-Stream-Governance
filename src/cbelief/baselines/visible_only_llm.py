from __future__ import annotations

from cbelief.baselines.rule_based import predict as rule_predict


def predict(sample: dict) -> dict:
    # Placeholder visible-only LLM: uses only visible stream; replace with provider/API call.
    pred = rule_predict(sample)
    pred["method_name"] = "visible_only_llm_placeholder"
    pred["rationale"] = "Placeholder visible-only LLM; uses only query-time visible evidence."
    return pred
