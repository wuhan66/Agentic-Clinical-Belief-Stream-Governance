from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="last",
        help="Which duplicate sample_id record to keep.",
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    counts = Counter(str(row.get("sample_id", "")).strip() for row in rows)
    duplicate_ids = {sid for sid, count in counts.items() if sid and count > 1}

    deduped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            raise ValueError("Found row without sample_id; refusing to deduplicate.")
        if args.keep == "first" and sid in deduped:
            continue
        deduped[sid] = row

    output_jsonl = args.output_jsonl
    backup_path = None
    if args.in_place:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.input_jsonl.with_suffix(args.input_jsonl.suffix + f".bak_{timestamp}")
        shutil.copy2(args.input_jsonl, backup_path)
        output_jsonl = args.input_jsonl
    elif output_jsonl is None:
        raise ValueError("Use --in-place or pass --output-jsonl.")

    assert output_jsonl is not None
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, list(deduped.values()))

    print(f"input_jsonl={args.input_jsonl}")
    print(f"output_jsonl={output_jsonl}")
    if backup_path is not None:
        print(f"backup_jsonl={backup_path}")
    print(f"rows_before={len(rows)}")
    print(f"rows_after={len(deduped)}")
    print(f"duplicate_sample_ids={len(duplicate_ids)}")
    print(f"duplicate_extra_rows={len(rows) - len(deduped)}")
    print(f"keep={args.keep}")


if __name__ == "__main__":
    main()
