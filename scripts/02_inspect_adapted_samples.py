import json
from pathlib import Path
from collections import Counter


def main(path: str = "data/adapted/train.jsonl", n: int = 3):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} samples from {path}")

    claim_counter = Counter()
    hyp_counter = Counter()
    subtype_counter = Counter()

    n_visible = 0
    n_future = 0
    n_retro = 0
    n_requires_reattr = 0

    for row in rows:
        label = row["silver_label"]
        claim_counter[label.get("initial_claim_status")] += 1
        hyp_counter[label.get("initial_primary_hypothesis")] += 1
        subtype_counter[row.get("sample_subtype")] += 1

        n_visible += len(row["visible_event_ids"])
        n_future += len(row["future_event_ids"])
        n_retro += len(row["retrospective_event_ids"])

        if label.get("requires_delayed_reattribution"):
            n_requires_reattr += 1

    print("\nInitial claim status:")
    print(claim_counter)

    print("\nInitial primary hypothesis:")
    print(hyp_counter)

    print("\nSample subtype:")
    print(subtype_counter.most_common(20))

    print("\nEvent counts:")
    print(f"visible: {n_visible}")
    print(f"future observed: {n_future}")
    print(f"retrospective: {n_retro}")
    print(f"requires delayed reattribution: {n_requires_reattr}")

    print("\nExamples:")
    for row in rows[:n]:
        print("=" * 80)
        print("sample_id:", row["sample_id"])
        print("query_time:", row["query_time"])
        print("target_claim_initial:", row["target_claim_initial"])
        print("label:", row["silver_label"])
        print("visible_events:", len(row["visible_events"]))
        print("future_events:", len(row["future_observed_events"]))
        print("retrospective_events:", len(row["retrospective_events"]))

        print("\nFirst visible:")
        if row["visible_events"]:
            print(row["visible_events"][0])

        print("\nFirst retrospective:")
        if row["retrospective_events"]:
            print(row["retrospective_events"][0])


if __name__ == "__main__":
    main()