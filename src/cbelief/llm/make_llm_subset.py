import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List


INPUT_PATH = "data/adapted/val.jsonl"
OUT_PATH = "data/llm/val_subset_100.jsonl"
SEED = 42
TARGET_N = 100


KEY_SUBTYPES = [
    "acute_by_visible_trend",
    "chronic_by_stable_high_creatinine",
    "chronic_by_future_ckd_code",
    "acute_by_future_diagnosis_or_rrt_only",
    "comorbid_aki_ckd",
    "uncertain_stable_creatinine",
    "low_confidence_ratio_trigger",
]


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


def main():
    random.seed(SEED)
    samples = load_jsonl(INPUT_PATH)

    grouped = defaultdict(list)
    for s in samples:
        subtype = s.get("sample_subtype", "unknown")
        grouped[subtype].append(s)

    print("Validation subtype distribution:")
    for subtype, rows in sorted(grouped.items()):
        print(f"{subtype:40s} {len(rows)}")

    selected = []

    # Balanced first pass.
    base_quota = TARGET_N // len(KEY_SUBTYPES)

    for subtype in KEY_SUBTYPES:
        rows = grouped.get(subtype, [])
        random.shuffle(rows)
        take = min(base_quota, len(rows))
        selected.extend(rows[:take])

    # Fill remaining slots from all not-yet-selected samples.
    selected_ids = {s["sample_id"] for s in selected}
    remaining = [s for s in samples if s["sample_id"] not in selected_ids]
    random.shuffle(remaining)

    while len(selected) < TARGET_N and remaining:
        selected.append(remaining.pop())

    # Trim if slightly over.
    selected = selected[:TARGET_N]

    # Add LLM subset metadata only. This is for bookkeeping, not for model input.
    for i, s in enumerate(selected):
        s["llm_subset_id"] = f"val_llm_{i:04d}"

    save_jsonl(selected, OUT_PATH)

    print("\nSaved:", OUT_PATH)
    print("n:", len(selected))
    print("\nSelected subtype distribution:")
    print(dict(Counter(s.get("sample_subtype", "unknown") for s in selected)))

    print("\nGold claim distribution:")
    print(dict(Counter(s["silver_label"]["initial_claim_status"] for s in selected)))

    print("\nGold hypothesis distribution:")
    print(dict(Counter(s["silver_label"]["initial_primary_hypothesis"] for s in selected)))


if __name__ == "__main__":
    main()