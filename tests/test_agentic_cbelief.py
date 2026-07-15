import unittest

from cbelief.agents.agent_schemas import normalize_prediction
from cbelief.agents.agentic_cbelief import AgenticCBeliefPipeline
from cbelief.agents.json_utils import extract_json_object


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages):
        if not self.responses:
            raise RuntimeError("No more mock responses")
        return self.responses.pop(0)


class AgenticCBeliefTests(unittest.TestCase):
    def test_extract_json_from_think_markdown_block(self):
        raw = (
            "reasoning before final answer</think>\n\n"
            "```json\n"
            '{"initial_primary_hypothesis":"uncertain_or_transient_abnormality",'
            '"hypothesis_scores":{"acute_renal_deterioration":0}}\n'
            "```"
        )

        parsed = extract_json_object(raw)

        self.assertEqual(parsed["initial_primary_hypothesis"], "uncertain_or_transient_abnormality")
        self.assertEqual(parsed["hypothesis_scores"]["acute_renal_deterioration"], 0)

    def test_extract_json_from_escaped_markdown_block(self):
        raw = (
            r"model text</think>\n\n```json\n"
            r"{\"initial_primary_hypothesis\":\"acute_renal_deterioration\","
            r"\"rationale\":\"valid JSON but escaped\"}\n```"
        )

        parsed = extract_json_object(raw)

        self.assertEqual(parsed["initial_primary_hypothesis"], "acute_renal_deterioration")
        self.assertEqual(parsed["rationale"], "valid JSON but escaped")

    def test_mixed_final_primary_maps_to_acute(self):
        pred = normalize_prediction(
            {
                "initial_claim_status": "supported",
                "final_claim_status": "supported",
                "initial_primary_hypothesis": "acute_renal_deterioration",
                "final_primary_hypothesis": "mixed_acute_on_chronic_renal_dysfunction",
                "final_clinical_phenotype": "mixed_acute_on_chronic_renal_dysfunction",
                "requires_delayed_reattribution": True,
            }
        )

        self.assertEqual(pred["final_primary_hypothesis"], "acute_renal_deterioration")
        self.assertEqual(pred["final_clinical_phenotype"], "mixed_acute_on_chronic_renal_dysfunction")

    def test_pipeline_rejects_unparsable_agent_json(self):
        valid_gatekeeper = (
            '{"visible_evidence_ids":["visible_evidence_0"],'
            '"future_evidence_ids":[],"retrospective_evidence_ids":[],'
            '"query_time_support_rule":"visible only","warnings":[]}'
        )
        valid_panel = (
            '{"initial_primary_hypothesis":"acute_renal_deterioration",'
            '"final_primary_hypothesis":"acute_renal_deterioration",'
            '"hypothesis_scores":{"acute_renal_deterioration":1},'
            '"rationale":"visible creatinine trend"}'
        )
        valid_adjudicator = (
            '{"initial_supporting_evidence_ids":["visible_evidence_0"],'
            '"final_supporting_evidence_ids":["visible_evidence_0"],'
            '"invalid_initial_support_ids":[],"rationale":"valid visible support"}'
        )
        valid_reattribution = (
            '{"requires_delayed_reattribution":false,'
            '"final_clinical_phenotype":"acute_renal_deterioration",'
            '"rationale":"no change"}'
        )

        llm = SequenceLLM(
            [
                valid_gatekeeper,
                valid_panel,
                valid_adjudicator,
                valid_reattribution,
                '{"initial_claim_status":"supported","final_claim_status"',
            ]
        )
        pipeline = AgenticCBeliefPipeline(llm)

        sample = {
            "sample_id": "s1",
            "query_time": "2150-01-01",
            "visible_stream": ["creatinine increased by at least 0.3 mg/dL within 48h"],
            "future_stream": [],
        }

        with self.assertRaisesRegex(ValueError, "clinical_coordinator returned unparsable JSON"):
            pipeline.run_one(sample)


if __name__ == "__main__":
    unittest.main()
