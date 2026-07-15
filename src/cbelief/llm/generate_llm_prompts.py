import json
from pathlib import Path
from typing import Dict, Any, List, Callable

from cbelief.llm.prompts import (
    visible_only_prompt,
    full_context_unmarked_prompt,
    full_context_with_provenance_prompt,
    cbelief_prompted_prompt,
)


SUBSET_PATH = "data/llm/val_subset_100.jsonl"
OUT_DIR = "data/llm/prompts"


CONDITIONS = {
    "llm_visible_only": visible_only_prompt,
    "llm_full_context_unmarked": full_context_unmarked_prompt,
    "llm_full_context_with_provenance": full_context_with_provenance_prompt,
    "llm_cbelief_prompted": cbelief_prompted_prompt,
}


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


def make_prompt_rows(
    samples: List[Dict[str, Any]],
    condition: str,
    prompt_fn: Callable[[Dict[str, Any]], str],
) -> List[Dict[str, Any]]:
    rows = []

    for s in samples:
        rows.append({
            "sample_id": s.get("sample_id"),
            "llm_subset_id": s.get("llm_subset_id"),
            "condition": condition,
            "prompt": prompt_fn(s),
            # Metadata below is for analysis only. Do not put it into the prompt.
            "sample_subtype": s.get("sample_subtype"),
            "gold_claim_status": s["silver_label"]["initial_claim_status"],
            "gold_primary_hypothesis": s["silver_label"]["initial_primary_hypothesis"],
        })

    return rows


def main():
    samples = load_jsonl(SUBSET_PATH)
    print("Loaded subset:", len(samples))

    for condition, prompt_fn in CONDITIONS.items():
        rows = make_prompt_rows(samples, condition, prompt_fn)
        out_path = f"{OUT_DIR}/{condition}_prompts.jsonl"
        save_jsonl(rows, out_path)
        print(f"Saved {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()