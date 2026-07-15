import json
from pathlib import Path
import pandas as pd


METRIC_FILES = {
    "A_visible_only_rule": "data/metrics/visible_only_rule_test.json",
    "B_full_context_leaky_rule": "data/metrics/full_context_leaky_rule_test.json",
    "C_full_context_with_provenance_rule": "data/metrics/full_context_with_provenance_rule_test.json",
    #"D_cbelief_v0_2_margin": "data/metrics/cbelief_v0_2_margin_test.json",
    "D_cbelief_v0_2_1_flagfix": "data/metrics/cbelief_v0_2_1_flagfix_test.json",
}


KEEP = [
    "n",
    "claim_status_accuracy",
    "claim_status_macro_f1",
    "primary_hypothesis_accuracy",
    "primary_hypothesis_macro_f1",
    "temporal_leakage_rate",
    "retrospective_misuse_rate",
    "retrospective_recognition_rate",
]


def main():
    rows = []

    for method, path in METRIC_FILES.items():
        path = Path(path)
        if not path.exists():
            print(f"Missing: {path}")
            continue

        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)

        row = {"method": method}
        for k in KEEP:
            row[k] = m.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path("data/metrics/temporal_evidence_trap_test.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()