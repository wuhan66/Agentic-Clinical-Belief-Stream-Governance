import json
from pathlib import Path

import pandas as pd


LLM_SUMMARY_PATH = "data/metrics/llm_n100_summary.csv"
LEAKAGE_CASES_PATH = "data/metrics/llm_full_context_unmarked_leakage_cases.csv"

OUT_DIR = Path("data/metrics/paper_tables")


# METHOD_RENAME = {
#     "LLM_visible_only_raw": "Visible-only LLM",
#     "LLM_visible_only_postprocessed": "Visible-only LLM + verifier",
#     "LLM_full_context_unmarked_raw": "Full-context LLM, unmarked",
#     "LLM_full_context_unmarked_postprocessed": "Full-context LLM, unmarked + verifier",
#     "LLM_full_context_with_provenance_raw": "Full-context LLM with provenance",
#     "LLM_full_context_with_provenance_postprocessed": "Full-context LLM with provenance + verifier",
#     "LLM_CBELIEF_prompted_raw": "C-BELIEF-prompted LLM",
#     "LLM_CBELIEF_prompted_postprocessed": "C-BELIEF-prompted LLM + verifier",
# }

METHOD_RENAME = {
    "LLM_visible_only_raw": "Visible-only LLM",
    "LLM_visible_only_skill": "Visible-only LLM + temporal skill",

    "LLM_full_context_unmarked_raw": "Full-context LLM, unmarked",
    "LLM_full_context_unmarked_skill": "Full-context LLM, unmarked + temporal skill",

    "LLM_full_context_with_provenance_raw": "Full-context LLM with provenance",
    "LLM_full_context_with_provenance_skill": "Full-context LLM with provenance + temporal skill",

    "LLM_CBELIEF_prompted_raw": "C-BELIEF-prompted LLM",
    "LLM_CBELIEF_prompted_skill": "C-BELIEF-prompted LLM + temporal skill",
}



# MAIN_METHOD_ORDER = [
#     "LLM_visible_only_raw",
#     "LLM_full_context_unmarked_raw",
#     "LLM_full_context_with_provenance_raw",
#     "LLM_CBELIEF_prompted_raw",
#     "LLM_CBELIEF_prompted_postprocessed",
# ]

MAIN_METHOD_ORDER = [
    "LLM_visible_only_raw",
    "LLM_visible_only_skill",
    "LLM_full_context_unmarked_raw",
    "LLM_full_context_unmarked_skill",
    "LLM_full_context_with_provenance_raw",
    "LLM_full_context_with_provenance_skill",
    "LLM_CBELIEF_prompted_raw",
    "LLM_CBELIEF_prompted_skill",
]

def round_cols(df, cols, ndigits=3):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).round(ndigits)
    return out


def make_llm_main_table():
    df = pd.read_csv(LLM_SUMMARY_PATH)

    df = df[df["method"].isin(MAIN_METHOD_ORDER)].copy()
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(MAIN_METHOD_ORDER)})
    df = df.sort_values("method_order")

    df["Method"] = df["method"].map(METHOD_RENAME)

    out = df[
        [
            "Method",
            "n",
            "claim_status_accuracy",
            "claim_status_macro_f1",
            "primary_hypothesis_accuracy",
            "primary_hypothesis_macro_f1",
            "temporal_leakage_rate",
            "retrospective_misuse_rate",
            "retrospective_recognition_rate",
        ]
    ].rename(
        columns={
            "n": "N",
            "claim_status_accuracy": "Claim accuracy",
            "claim_status_macro_f1": "Claim macro-F1",
            "primary_hypothesis_accuracy": "Hypothesis accuracy",
            "primary_hypothesis_macro_f1": "Hypothesis macro-F1",
            "temporal_leakage_rate": "Temporal leakage",
            "retrospective_misuse_rate": "Retrospective misuse",
            "retrospective_recognition_rate": "Retrospective recognition",
        }
    )

    out = round_cols(
        out,
        [
            "Claim accuracy",
            "Claim macro-F1",
            "Hypothesis accuracy",
            "Hypothesis macro-F1",
            "Temporal leakage",
            "Retrospective misuse",
            "Retrospective recognition",
        ],
    )

    path = OUT_DIR / "table1_llm_main_results.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return out, path


def make_llm_all_table():
    df = pd.read_csv(LLM_SUMMARY_PATH)
    df["Method"] = df["method"].map(METHOD_RENAME).fillna(df["method"])

    out = df[
        [
            "Method",
            "n",
            "claim_status_accuracy",
            "claim_status_macro_f1",
            "primary_hypothesis_accuracy",
            "primary_hypothesis_macro_f1",
            "temporal_leakage_rate",
            "retrospective_misuse_rate",
            "retrospective_recognition_rate",
        ]
    ].rename(
        columns={
            "n": "N",
            "claim_status_accuracy": "Claim accuracy",
            "claim_status_macro_f1": "Claim macro-F1",
            "primary_hypothesis_accuracy": "Hypothesis accuracy",
            "primary_hypothesis_macro_f1": "Hypothesis macro-F1",
            "temporal_leakage_rate": "Temporal leakage",
            "retrospective_misuse_rate": "Retrospective misuse",
            "retrospective_recognition_rate": "Retrospective recognition",
        }
    )

    out = round_cols(
        out,
        [
            "Claim accuracy",
            "Claim macro-F1",
            "Hypothesis accuracy",
            "Hypothesis macro-F1",
            "Temporal leakage",
            "Retrospective misuse",
            "Retrospective recognition",
        ],
    )

    path = OUT_DIR / "table1b_llm_all_results_with_postprocessing.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return out, path


def make_leakage_distribution_tables():
    df = pd.read_csv(LEAKAGE_CASES_PATH)

    by_type = (
        df.groupby("misuse_type")
        .size()
        .reset_index(name="n_misused_evidence")
        .sort_values("n_misused_evidence", ascending=False)
    )

    by_subtype = (
        df.groupby(["subtype", "misuse_type"])
        .size()
        .reset_index(name="n_misused_evidence")
        .sort_values(["n_misused_evidence", "subtype"], ascending=[False, True])
    )

    by_subtype_wide = by_subtype.pivot_table(
        index="subtype",
        columns="misuse_type",
        values="n_misused_evidence",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    by_subtype_wide["total_misused_evidence"] = by_subtype_wide.drop(columns=["subtype"]).sum(axis=1)
    by_subtype_wide = by_subtype_wide.sort_values("total_misused_evidence", ascending=False)

    type_path = OUT_DIR / "table2a_leakage_by_misuse_type.csv"
    subtype_path = OUT_DIR / "table2b_leakage_by_subtype.csv"

    by_type.to_csv(type_path, index=False, encoding="utf-8-sig")
    by_subtype_wide.to_csv(subtype_path, index=False, encoding="utf-8-sig")

    return by_type, type_path, by_subtype_wide, subtype_path


def make_representative_cases():
    df = pd.read_csv(LEAKAGE_CASES_PATH)

    # Prefer representative rows with clear retrospective/future misuse.
    priority_types = [
        "future_evidence_used_as_support",
        "retrospective_evidence_used_as_support",
    ]

    rows = []

    for misuse_type in priority_types:
        sub = df[df["misuse_type"] == misuse_type].copy()
        if not sub.empty:
            rows.append(sub.iloc[0])

    # Add representative cases from key subtypes.
    key_subtypes = [
        "chronic_by_stable_high_creatinine",
        "chronic_by_future_ckd_code",
        "acute_by_future_diagnosis_or_rrt_only",
        "comorbid_aki_ckd",
    ]

    for subtype in key_subtypes:
        sub = df[
            (df["subtype"] == subtype)
            & (df["misuse_type"] == "retrospective_evidence_used_as_support")
        ].copy()

        if not sub.empty:
            rows.append(sub.iloc[0])

    rep = pd.DataFrame(rows).drop_duplicates(subset=["sample_id", "misused_evidence_id"])

    keep_cols = [
        "sample_id",
        "subtype",
        "query_time",
        "gold_claim",
        "pred_claim",
        "gold_hypothesis",
        "pred_hypothesis",
        "misuse_type",
        "misused_evidence_id",
        "misused_evidence_group",
        "misused_evidence_time",
        "misused_evidence_type",
        "misused_evidence_name",
        "misused_evidence_text",
        "rationale",
    ]

    rep = rep[keep_cols]

    path = OUT_DIR / "table3_representative_leakage_cases.csv"
    rep.to_csv(path, index=False, encoding="utf-8-sig")

    return rep, path


def make_markdown_report(table1, by_type, by_subtype, rep):
    lines = []

    lines.append("# Paper-ready LLM Results")
    lines.append("")

    lines.append("## Table 1. Main LLM results on stratified validation subset, n=100")
    lines.append("")
    lines.append(table1.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 2A. Misused evidence by misuse type")
    lines.append("")
    lines.append(by_type.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 2B. Misused evidence by subtype")
    lines.append("")
    lines.append(by_subtype.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 3. Representative leakage cases")
    lines.append("")
    lines.append(rep.to_markdown(index=False))
    lines.append("")

    lines.append("## Draft result interpretation")
    lines.append("")
    lines.append(
        "Full-context LLM reasoning without explicit temporal provenance produced "
        "post-query evidence misuse, including retrospective/discharge-level note spans "
        "and a small number of future laboratory events selected as supporting evidence. "
        "Adding provenance labels eliminated temporal and retrospective misuse but did not "
        "substantially improve hypothesis-level reasoning. C-BELIEF prompting achieved the "
        "strongest hypothesis macro-F1 and retrospective-evidence recognition while maintaining "
        "zero temporal leakage and zero retrospective misuse. A lightweight consistency verifier "
        "substantially improved claim-status calibration without changing hypothesis predictions."
    )
    lines.append("")

    path = OUT_DIR / "paper_ready_llm_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table1, table1_path = make_llm_main_table()
    table_all, table_all_path = make_llm_all_table()
    by_type, type_path, by_subtype, subtype_path = make_leakage_distribution_tables()
    rep, rep_path = make_representative_cases()
    md_path = make_markdown_report(table1, by_type, by_subtype, rep)

    print("Saved:")
    print(table1_path)
    print(table_all_path)
    print(type_path)
    print(subtype_path)
    print(rep_path)
    print(md_path)

    print("\nMain LLM table:")
    print(table1.to_string(index=False))

    print("\nLeakage by type:")
    print(by_type.to_string(index=False))

    print("\nLeakage by subtype:")
    print(by_subtype.to_string(index=False))


if __name__ == "__main__":
    main()