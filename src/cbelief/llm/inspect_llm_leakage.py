import json
import argparse
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
            e = dict(e)
            e["_group"] = group_name
            event_map[str(eid)] = e

    return event_map


def short_event(e: Dict[str, Any]) -> str:
    return (
        f"group={e.get('_group')} | "
        f"id={e.get('event_id')} | "
        f"time={e.get('event_time')} | "
        f"type={e.get('event_type')} | "
        f"name={e.get('event_name')} | "
        f"value={e.get('value_num')} {e.get('unit', '')} | "
        f"text={str(e.get('event_text', ''))[:180]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--max_cases", type=int, default=20)
    args = parser.parse_args()

    samples = load_jsonl(args.samples)
    preds = load_jsonl(args.preds)

    assert len(samples) == len(preds), "samples and preds length mismatch"

    leakage_cases = []
    retro_misuse_cases = []

    total_support = 0
    future_support = 0
    retro_support = 0

    for sample, pred in zip(samples, preds):
        event_map = index_events(sample)

        support_ids = [str(x) for x in pred.get("supporting_evidence_ids", [])]
        total_support += len(support_ids)

        future_ids = []
        retro_ids = []
        visible_ids = []
        unknown_ids = []

        for eid in support_ids:
            e = event_map.get(eid)

            if e is None:
                unknown_ids.append(eid)
                continue

            group = e.get("_group")

            if group == "future_observed_events":
                future_ids.append(eid)
                future_support += 1
            elif group == "retrospective_events":
                retro_ids.append(eid)
                retro_support += 1
            elif group == "visible_events":
                visible_ids.append(eid)

        if future_ids:
            leakage_cases.append((sample, pred, future_ids, retro_ids, visible_ids, unknown_ids))

        if retro_ids:
            retro_misuse_cases.append((sample, pred, future_ids, retro_ids, visible_ids, unknown_ids))

    print("Total samples:", len(samples))
    print("Total supporting evidence ids:", total_support)
    print("Future evidence used as support:", future_support)
    print("Retrospective evidence used as support:", retro_support)

    if total_support:
        print("Temporal leakage rate:", future_support / total_support)
        print("Retrospective misuse rate:", retro_support / total_support)
    else:
        print("Temporal leakage rate: NA, no support ids")
        print("Retrospective misuse rate: NA, no support ids")

    print("\nLeakage cases:", len(leakage_cases))
    print("Retrospective misuse cases:", len(retro_misuse_cases))

    print("\n" + "=" * 100)
    print("Examples with future or retrospective support")
    print("=" * 100)

    shown = 0

    for sample, pred, future_ids, retro_ids, visible_ids, unknown_ids in leakage_cases + retro_misuse_cases:
        if shown >= args.max_cases:
            break

        event_map = index_events(sample)

        print("\n" + "-" * 100)
        print("sample_id:", sample.get("sample_id"))
        print("query_time:", sample.get("query_time"))
        print("subtype:", sample.get("sample_subtype"))
        print("gold_claim:", sample["silver_label"]["initial_claim_status"])
        print("pred_claim:", pred.get("claim_status"))
        print("gold_hyp:", sample["silver_label"]["initial_primary_hypothesis"])
        print("pred_hyp:", pred.get("primary_hypothesis"))
        print("future_support_ids:", future_ids)
        print("retrospective_support_ids:", retro_ids)
        print("visible_support_ids:", visible_ids)
        print("unknown_support_ids:", unknown_ids)
        print("rationale:", pred.get("rationale", "")[:600])

        if future_ids:
            print("\nFuture evidence used as support:")
            for eid in future_ids:
                print(" ", short_event(event_map[eid]))

        if retro_ids:
            print("\nRetrospective evidence used as support:")
            for eid in retro_ids:
                print(" ", short_event(event_map[eid]))

        shown += 1


if __name__ == "__main__":
    main()