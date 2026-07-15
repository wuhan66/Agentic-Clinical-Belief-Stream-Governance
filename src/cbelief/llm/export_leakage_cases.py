import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_events(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    event_map = {}

    for group_name in ["visible_events", "future_observed_events", "retrospective_events"]:
        for e in sample.get(group_name, []) or []:
            eid = e.get("event_id")
            if eid is None:
                continue

            e2 = dict(e)
            e2["_group"] = group_name
            event_map[str(eid)] = e2

    return event_map


def clean_text(x: Any, max_len: int = 500) -> str:
    text = str(x or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def event_summary(e: Dict[str, Any]) -> str:
    return (
        f"{e.get('_group')} | "
        f"{e.get('event_id')} | "
        f"{e.get('event_time')} | "
        f"{e.get('event_type')} | "
        f"{e.get('event_name')} | "
        f"{e.get('value_num')} {e.get('unit', '')} | "
        f"{clean_text(e.get('event_text'), 240)}"
    )


def gold_claim(sample: Dict[str, Any]) -> str:
    return sample["silver_label"]["initial_claim_status"]


def gold_hyp(sample: Dict[str, Any]) -> str:
    return sample["silver_label"]["initial_primary_hypothesis"]


def collect_leakage_rows(
    samples: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    method: str,
) -> List[Dict[str, Any]]:
    assert len(samples) == len(preds), "samples and predictions length mismatch"

    rows = []

    for sample, pred in zip(samples, preds):
        event_map = index_events(sample)

        support_ids = [str(x) for x in pred.get("supporting_evidence_ids", []) or []]
        retro_only_ids = [str(x) for x in pred.get("retrospective_only_evidence_ids", []) or []]

        for eid in support_ids:
            e = event_map.get(eid)

            if e is None:
                rows.append({
                    "method": method,
                    "sample_id": sample.get("sample_id"),
                    "subtype": sample.get("sample_subtype"),
                    "query_time": sample.get("query_time"),
                    "gold_claim": gold_claim(sample),
                    "pred_claim": pred.get("claim_status"),
                    "gold_hypothesis": gold_hyp(sample),
                    "pred_hypothesis": pred.get("primary_hypothesis"),
                    "misuse_type": "unknown_support_id",
                    "misused_evidence_id": eid,
                    "misused_evidence_group": "unknown",
                    "misused_evidence_time": "",
                    "misused_evidence_type": "",
                    "misused_evidence_name": "",
                    "misused_evidence_value": "",
                    "misused_evidence_text": "",
                    "visible_support_ids": json.dumps(
                        [x for x in support_ids if event_map.get(x, {}).get("_group") == "visible_events"],
                        ensure_ascii=False,
                    ),
                    "future_support_ids": json.dumps(
                        [x for x in support_ids if event_map.get(x, {}).get("_group") == "future_observed_events"],
                        ensure_ascii=False,
                    ),
                    "retrospective_support_ids": json.dumps(
                        [x for x in support_ids if event_map.get(x, {}).get("_group") == "retrospective_events"],
                        ensure_ascii=False,
                    ),
                    "retrospective_only_evidence_ids": json.dumps(retro_only_ids, ensure_ascii=False),
                    "rationale": clean_text(pred.get("rationale"), 1000),
                })
                continue

            group = e.get("_group")

            if group not in {"future_observed_events", "retrospective_events"}:
                continue

            if group == "future_observed_events":
                misuse_type = "future_evidence_used_as_support"
            elif group == "retrospective_events":
                misuse_type = "retrospective_evidence_used_as_support"
            else:
                misuse_type = "unknown"

            rows.append({
                "method": method,
                "sample_id": sample.get("sample_id"),
                "subtype": sample.get("sample_subtype"),
                "query_time": sample.get("query_time"),
                "gold_claim": gold_claim(sample),
                "pred_claim": pred.get("claim_status"),
                "gold_hypothesis": gold_hyp(sample),
                "pred_hypothesis": pred.get("primary_hypothesis"),
                "misuse_type": misuse_type,
                "misused_evidence_id": eid,
                "misused_evidence_group": group,
                "misused_evidence_time": e.get("event_time"),
                "misused_evidence_type": e.get("event_type"),
                "misused_evidence_name": e.get("event_name"),
                "misused_evidence_value": f"{e.get('value_num')} {e.get('unit', '')}",
                "misused_evidence_text": clean_text(e.get("event_text"), 500),
                "visible_support_ids": json.dumps(
                    [x for x in support_ids if event_map.get(x, {}).get("_group") == "visible_events"],
                    ensure_ascii=False,
                ),
                "future_support_ids": json.dumps(
                    [x for x in support_ids if event_map.get(x, {}).get("_group") == "future_observed_events"],
                    ensure_ascii=False,
                ),
                "retrospective_support_ids": json.dumps(
                    [x for x in support_ids if event_map.get(x, {}).get("_group") == "retrospective_events"],
                    ensure_ascii=False,
                ),
                "retrospective_only_evidence_ids": json.dumps(retro_only_ids, ensure_ascii=False),
                "rationale": clean_text(pred.get("rationale"), 1000),
            })

    return rows


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: List[Dict[str, Any]], path: str, max_cases: int = 20) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# LLM Leakage Case Export")
    lines.append("")
    lines.append(f"Total misuse evidence rows: {len(rows)}")
    lines.append("")

    subtype_counts = {}
    misuse_counts = {}

    for r in rows:
        subtype_counts[r["subtype"]] = subtype_counts.get(r["subtype"], 0) + 1
        misuse_counts[r["misuse_type"]] = misuse_counts.get(r["misuse_type"], 0) + 1

    lines.append("## Misuse type counts")
    lines.append("")
    for k, v in sorted(misuse_counts.items()):
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Subtype counts")
    lines.append("")
    for k, v in sorted(subtype_counts.items()):
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Representative cases")
    lines.append("")

    for i, r in enumerate(rows[:max_cases], start=1):
        lines.append(f"### Case {i}: {r['sample_id']}")
        lines.append("")
        lines.append(f"- subtype: `{r['subtype']}`")
        lines.append(f"- query_time: `{r['query_time']}`")
        lines.append(f"- gold claim / pred claim: `{r['gold_claim']}` / `{r['pred_claim']}`")
        lines.append(f"- gold hypothesis / pred hypothesis: `{r['gold_hypothesis']}` / `{r['pred_hypothesis']}`")
        lines.append(f"- misuse type: `{r['misuse_type']}`")
        lines.append(f"- misused evidence id: `{r['misused_evidence_id']}`")
        lines.append(f"- misused evidence group: `{r['misused_evidence_group']}`")
        lines.append(f"- misused evidence time: `{r['misused_evidence_time']}`")
        lines.append(f"- misused evidence type/name: `{r['misused_evidence_type']}` / `{r['misused_evidence_name']}`")
        lines.append("")
        lines.append("Misused evidence text:")
        lines.append("")
        lines.append(f"> {r['misused_evidence_text']}")
        lines.append("")
        lines.append("Model rationale:")
        lines.append("")
        lines.append(f"> {r['rationale']}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--method", default="llm_full_context_unmarked_raw")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--max_md_cases", type=int, default=20)
    args = parser.parse_args()

    samples = load_jsonl(args.samples)
    preds = load_jsonl(args.preds)

    rows = collect_leakage_rows(samples, preds, args.method)

    save_csv(rows, args.out_csv)
    save_markdown(rows, args.out_md, max_cases=args.max_md_cases)

    print("Samples:", len(samples))
    print("Predictions:", len(preds))
    print("Misused evidence rows:", len(rows))
    print("Saved CSV:", args.out_csv)
    print("Saved Markdown:", args.out_md)

    misuse_counts = {}
    subtype_counts = {}

    for r in rows:
        misuse_counts[r["misuse_type"]] = misuse_counts.get(r["misuse_type"], 0) + 1
        subtype_counts[r["subtype"]] = subtype_counts.get(r["subtype"], 0) + 1

    print("\nMisuse type counts:")
    for k, v in sorted(misuse_counts.items()):
        print(f"{k}: {v}")

    print("\nSubtype counts:")
    for k, v in sorted(subtype_counts.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()