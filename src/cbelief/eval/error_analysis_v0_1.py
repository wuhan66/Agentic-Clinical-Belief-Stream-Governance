import json
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report


LABELS = ["supported", "partially_supported", "insufficient"]


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    samples = load_jsonl("data/adapted/val.jsonl")
    preds = load_jsonl("data/predictions/cbelief/val_predictions_v0_1.jsonl")

    gold_claim = [s["silver_label"]["initial_claim_status"] for s in samples]
    pred_claim = [p["claim_status"] for p in preds]

    print("Claim status classification report:")
    print(classification_report(gold_claim, pred_claim, labels=LABELS, zero_division=0))

    print("Confusion matrix rows=gold, cols=pred:")
    print(LABELS)
    print(confusion_matrix(gold_claim, pred_claim, labels=LABELS))

    by_subtype = defaultdict(lambda: Counter())
    by_subtype_pred = defaultdict(lambda: Counter())

    for s, p in zip(samples, preds):
        subtype = s.get("sample_subtype", "unknown")
        by_subtype[subtype][s["silver_label"]["initial_claim_status"]] += 1
        by_subtype_pred[subtype][p["claim_status"]] += 1

    print("\nBy subtype:")
    for subtype in sorted(by_subtype.keys()):
        print("=" * 80)
        print("Subtype:", subtype)
        print("Gold:", dict(by_subtype[subtype]))
        print("Pred:", dict(by_subtype_pred[subtype]))

    # show typical errors
    print("\nTypical supported -> partial/insufficient errors:")
    n = 0
    for s, p in zip(samples, preds):
        g = s["silver_label"]["initial_claim_status"]
        pred = p["claim_status"]
        if g == "supported" and pred != "supported":
            print("-" * 80)
            print("sample_id:", s["sample_id"])
            print("subtype:", s.get("sample_subtype"))
            print("gold:", g, "pred:", pred)
            print("gold hyp:", s["silver_label"]["initial_primary_hypothesis"])
            print("pred hyp:", p["primary_hypothesis"])
            print("quality:", s.get("quality_flags", {}))
            print("partial_reasons:", p.get("partial_reasons"))
            print("visible events:")
            for e in s["visible_events"]:
                print(" ", e.get("event_type"), e.get("event_name"), e.get("event_text"))
            n += 1
            if n >= 5:
                break


if __name__ == "__main__":
    main()