import json
from pathlib import Path
from collections import Counter, defaultdict


def inspect_jsonl(path: str, n: int = 5):
    path = Path(path)
    print(f"Inspecting: {path}")
    print(f"Exists: {path.exists()}")
    print(f"Size MB: {path.stat().st_size / 1024 / 1024:.2f}")

    key_counter = Counter()
    type_map = defaultdict(Counter)

    examples = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            obj = json.loads(line)
            key_counter.update(obj.keys())

            for k, v in obj.items():
                type_map[k][type(v).__name__] += 1

            if len(examples) < n:
                examples.append(obj)

    print("\nTop-level keys:")
    for k, c in key_counter.most_common():
        print(f"  {k}: {c}")

    print("\nField types:")
    for k, counter in type_map.items():
        print(f"  {k}: {dict(counter)}")

    print("\nExamples:")
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        for k, v in ex.items():
            if isinstance(v, str):
                print(f"{k}: {v[:300]}")
            elif isinstance(v, list):
                print(f"{k}: list(len={len(v)})")
                if len(v) > 0:
                    print(f"  first item: {str(v[0])[:300]}")
            elif isinstance(v, dict):
                print(f"{k}: dict(keys={list(v.keys())[:20]})")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    inspect_jsonl("data/stream/train_v2_1.jsonl", n=3)