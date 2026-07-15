from __future__ import annotations

from cbelief.methods.evidence_retriever import EvidenceRetriever
from cbelief.methods.temporal_validator import TemporalValidator
from cbelief.methods.evidence_adjudicator import EvidenceAdjudicator
from cbelief.methods.claim_verifier import ClaimVerifier
from cbelief.methods.delayed_reattributor import DelayedReattributor


class CBeliefStream:
    def __init__(self, top_k: int = 8):
        self.retriever = EvidenceRetriever(top_k=top_k)
        self.temporal_validator = TemporalValidator()
        self.adjudicator = EvidenceAdjudicator()
        self.verifier = ClaimVerifier()
        self.reattributor = DelayedReattributor()

    def predict(self, sample: dict) -> dict:
        events = self.retriever.retrieve(sample)
        temporal = self.temporal_validator.validate(sample, events)
        adjudication = self.adjudicator.adjudicate(sample, events, temporal)
        verification = self.verifier.verify(sample, adjudication)
        reattr = self.reattributor.reattribute(sample, verification)
        return {
            "sample_id": sample["sample_id"],
            "method_name": "cbelief_stream_v0_1",
            **verification,
            "contradictory_evidence_ids": [],
            "misused_future_evidence_ids": [],
            "missing_evidence": self.infer_missing_evidence(sample),
            "temporal_validity_summary": "Only visible_before_query evidence contributes to query-time support; retrospective evidence is separated.",
            "rationale": "Structured C-BELIEF-Stream inference with temporal gatekeeping and delayed reattribution.",
            "metadata": {"adjudication": adjudication, "reattribution_records": reattr},
        }

    def infer_missing_evidence(self, sample: dict) -> list[str]:
        visible = " ".join(sample.get("visible_stream", [])).lower()
        missing = []
        if "creatinine" not in visible:
            missing.append("serum creatinine timeline")
        if "urine" not in visible:
            missing.append("urine output")
        if "baseline" not in visible:
            missing.append("pre-admission baseline creatinine")
        return missing
