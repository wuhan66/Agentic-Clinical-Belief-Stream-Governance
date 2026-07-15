# 功能说明：
# 1. 将 C-BELIEF_STREAM 数据集 JSONL 按 sample_id / 行号切分成多个不重叠 shard。
# 2. 自动检查 shard 之间是否存在重复 sample_id，以及合并后是否覆盖原始数据。
# 3. 适合多 GPU 数据并行推理，每张 GPU 独立处理一个 shard 文件，避免 offset/limit 手写错误。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_sample_id(row: Dict, idx: int) -> str:
    sid = str(row.get("sample_id", "")).strip()
    if not sid:
        raise ValueError(f"Missing sample_id at row index {idx}")
    return sid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["contiguous", "round_robin"],
        default="round_robin",
        help="contiguous: 连续切片；round_robin: 按行号取模切片，通常负载更均衡。",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    n = len(rows)

    original_ids = [get_sample_id(row, i) for i, row in enumerate(rows)]
    if len(original_ids) != len(set(original_ids)):
        seen: Set[str] = set()
        dup = []
        for sid in original_ids:
            if sid in seen:
                dup.append(sid)
            seen.add(sid)
        raise ValueError(f"Original dataset has duplicated sample_id, examples: {dup[:10]}")

    shards: List[List[Dict]] = [[] for _ in range(args.num_shards)]

    if args.mode == "round_robin":
        for i, row in enumerate(rows):
            shard_id = i % args.num_shards
            shards[shard_id].append(row)
    else:
        shard_size = (n + args.num_shards - 1) // args.num_shards
        for shard_id in range(args.num_shards):
            start = shard_id * shard_size
            end = min((shard_id + 1) * shard_size, n)
            shards[shard_id] = rows[start:end]

    all_shard_ids: List[str] = []

    for shard_id, shard_rows in enumerate(shards):
        out_path = args.output_dir / f"shard{shard_id}.jsonl"
        write_jsonl(out_path, shard_rows)

        shard_ids = [get_sample_id(row, i) for i, row in enumerate(shard_rows)]
        all_shard_ids.extend(shard_ids)

        print(f"[INFO] shard{shard_id}: {len(shard_rows)} rows -> {out_path}")

    original_set = set(original_ids)
    shard_set = set(all_shard_ids)

    duplicated_in_shards = len(all_shard_ids) - len(shard_set)
    missing = original_set - shard_set
    extra = shard_set - original_set

    print(f"[INFO] original rows: {len(original_ids)}")
    print(f"[INFO] total shard rows: {len(all_shard_ids)}")
    print(f"[INFO] duplicated sample_ids in shards: {duplicated_in_shards}")
    print(f"[INFO] missing sample_ids: {len(missing)}")
    print(f"[INFO] extra sample_ids: {len(extra)}")

    if duplicated_in_shards != 0 or missing or extra:
        raise RuntimeError("Shard validation failed.")

    print("[OK] Shards are non-overlapping and fully cover the original dataset.")


if __name__ == "__main__":
    main()