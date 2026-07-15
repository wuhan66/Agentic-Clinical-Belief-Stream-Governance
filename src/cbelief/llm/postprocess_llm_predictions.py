import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


UNCERTAIN = "uncertain_or_transient_abnormality"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def postprocess(pred: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(pred)

    hyp = p.get("primary_hypothesis")
    claim = p.get("claim_status")

    # Minimal consistency rule:
    # If the model's own best hypothesis is uncertain/transient,
    # then it should not claim positive renal pathology is supported.
    if hyp == UNCERTAIN:
        p["claim_status"] = "insufficient"
        p["supporting_evidence_ids"] = []
        reasons = list(p.get("postprocess_reasons", []))
        reasons.append("uncertain_hypothesis_forces_insufficient_claim")
        p["postprocess_reasons"] = reasons

    # If no support ids are provided, supported should be downgraded.
    support_ids = p.get("supporting_evidence_ids", []) or []
    if p.get("claim_status") == "supported" and len(support_ids) == 0:
        p["claim_status"] = "insufficient"
        reasons = list(p.get("postprocess_reasons", []))
        reasons.append("supported_without_supporting_evidence_ids")
        p["postprocess_reasons"] = reasons

    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    preds = load_jsonl(args.infile)
    out = [postprocess(p) for p in preds]

    save_jsonl(out, args.outfile)

    changed = sum(
        1 for a, b in zip(preds, out)
        if a.get("claim_status") != b.get("claim_status")
    )

    print("Loaded:", len(preds))
    print("Changed claim_status:", changed)
    print("Saved:", args.outfile)


if __name__ == "__main__":
    main()