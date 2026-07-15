import argparse
from collections import Counter

from cbelief.skills.cbelief_temporal_skill import (
    load_jsonl,
    save_jsonl,
    repair_prediction_with_cbelief_skill,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.samples)
    preds = load_jsonl(args.preds)

    assert len(samples) == len(preds), (
        f"samples/preds length mismatch: {len(samples)} vs {len(preds)}"
    )

    repaired = []
    repair_counter = Counter()

    changed_claim = 0
    changed_support = 0
    changed_retro_only = 0

    for sample, pred in zip(samples, preds):
        new_pred = repair_prediction_with_cbelief_skill(sample, pred)
        repaired.append(new_pred)

        if pred.get("claim_status") != new_pred.get("claim_status"):
            changed_claim += 1

        if pred.get("supporting_evidence_ids") != new_pred.get("supporting_evidence_ids"):
            changed_support += 1

        if pred.get("retrospective_only_evidence_ids") != new_pred.get("retrospective_only_evidence_ids"):
            changed_retro_only += 1

        trace = new_pred.get("cbelief_skill_trace", {})
        for r in trace.get("repairs", []):
            repair_counter[r] += 1

    save_jsonl(repaired, args.out)

    print("Loaded samples:", len(samples))
    print("Loaded predictions:", len(preds))
    print("Saved:", args.out)
    print()
    print("Changed claim_status:", changed_claim)
    print("Changed supporting_evidence_ids:", changed_support)
    print("Changed retrospective_only_evidence_ids:", changed_retro_only)
    print()
    print("Repair counts:")
    for k, v in repair_counter.most_common():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()