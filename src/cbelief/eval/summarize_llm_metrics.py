import json
from pathlib import Path

import pandas as pd


# METRIC_FILES = {
#     "LLM_visible_only_raw": "data/metrics/llm_visible_only_n100.json",
#     "LLM_visible_only_postprocessed": "data/metrics/llm_visible_only_n100_postprocessed.json",
#
#     "LLM_full_context_unmarked_raw": "data/metrics/llm_full_context_unmarked_n100.json",
#     "LLM_full_context_unmarked_postprocessed": "data/metrics/llm_full_context_unmarked_n100_postprocessed.json",
#
#     "LLM_full_context_with_provenance_raw": "data/metrics/llm_full_context_with_provenance_n100.json",
#     "LLM_full_context_with_provenance_postprocessed": "data/metrics/llm_full_context_with_provenance_n100_postprocessed.json",
#
#     "LLM_CBELIEF_prompted_raw": "data/metrics/llm_cbelief_prompted_n100.json",
#     "LLM_CBELIEF_prompted_postprocessed": "data/metrics/llm_cbelief_prompted_n100_postprocessed.json",
# }

METRIC_FILES = {
    "LLM_visible_only_raw": "data/metrics/llm_visible_only_n100.json",
    "LLM_visible_only_skill": "data/metrics/llm_visible_only_n100_skill.json",

    "LLM_full_context_unmarked_raw": "data/metrics/llm_full_context_unmarked_n100.json",
    "LLM_full_context_unmarked_skill": "data/metrics/llm_full_context_unmarked_n100_skill.json",

    "LLM_full_context_with_provenance_raw": "data/metrics/llm_full_context_with_provenance_n100.json",
    "LLM_full_context_with_provenance_skill": "data/metrics/llm_full_context_with_provenance_n100_skill.json",

    "LLM_CBELIEF_prompted_raw": "data/metrics/llm_cbelief_prompted_n100.json",
    "LLM_CBELIEF_prompted_skill": "data/metrics/llm_cbelief_prompted_n100_skill.json",
}



MAIN_COLUMNS = [
    "method",
    "n",
    "claim_status_accuracy",
    "claim_status_macro_f1",
    "primary_hypothesis_accuracy",
    "primary_hypothesis_macro_f1",
    "temporal_leakage_rate",
    "retrospective_misuse_rate",
    "retrospective_recognition_rate",
]


def load_json(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    for method, path in METRIC_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"Missing: {path}")
            continue

        m = load_json(path)
        row = {"method": method}
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)

    out = Path("data/metrics/llm_n100_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df[MAIN_COLUMNS].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()