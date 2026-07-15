# 功能说明：
# 1. 评估 Agentic C-BELIEF 输出与 v2.2 标签的一致性。
# 2. 输出总体 accuracy、macro-F1、temporal safety、support validity、subtype-level metrics 和错误案例表。
# 3. 该脚本只基于已有 prediction JSONL 和 dataset JSONL 做比对，不重新调用 LLM，不修改数据集。

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# =========================
# Label spaces
# =========================

CLAIM_STATUS_LABELS = [
    "supported",
    "partially_supported",
    "insufficient",
]

PRIMARY_HYPOTHESIS_LABELS = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]

FINAL_PHENOTYPE_LABELS = [
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "mixed_acute_on_chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
]

FIELDS = [
    "initial_claim_status",
    "final_claim_status",
    "initial_primary_hypothesis",
    "final_primary_hypothesis",
    "final_clinical_phenotype",
    "requires_delayed_reattribution",
]

FIELD_LABELS = {
    "initial_claim_status": CLAIM_STATUS_LABELS,
    "final_claim_status": CLAIM_STATUS_LABELS,
    "initial_primary_hypothesis": PRIMARY_HYPOTHESIS_LABELS,
    "final_primary_hypothesis": PRIMARY_HYPOTHESIS_LABELS,
    "final_clinical_phenotype": FINAL_PHENOTYPE_LABELS,
    "requires_delayed_reattribution": [False, True],
}


# =========================
# IO helpers
# =========================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def label_of(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample.get("weak_label_v2_2") or sample.get("weak_label_v2_1") or {}


def safe_get(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def parse_bool_like(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    return bool(x)


def normalize_value(field: str, x: Any) -> Any:
    if field == "requires_delayed_reattribution":
        return parse_bool_like(x)
    if x is None:
        return ""
    return str(x).strip()


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, tuple):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
        if "," in s:
            return [v.strip() for v in s.split(",") if v.strip()]
        return [s]
    return [str(x).strip()]


# =========================
# Metric helpers
# =========================

def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def f1_for_label(y_true: Sequence[Any], y_pred: Sequence[Any], label: Any) -> float:
    tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred))
    fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred))

    if tp == 0 and fp == 0 and fn == 0:
        return float("nan")
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float:
    scores = []
    for lab in labels:
        score = f1_for_label(y_true, y_pred, lab)
        if not pd.isna(score):
            scores.append(score)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def positive_f1(y_true: Sequence[Any], y_pred: Sequence[Any], positive_label: Any = True) -> float:
    score = f1_for_label(y_true, y_pred, positive_label)
    if pd.isna(score):
        return 0.0
    return float(score)


# =========================
# Evidence / support helpers
# =========================

def event_id_and_text(ev: Any, prefix: str, i: int) -> Tuple[str, str]:
    if isinstance(ev, dict):
        ev_id = ev.get("evidence_id") or ev.get("id") or f"{prefix}_{i}"
        text = ev.get("text") or ev.get("event_text") or json.dumps(ev, ensure_ascii=False)
        return str(ev_id), str(text)
    return f"{prefix}_{i}", str(ev)


def build_evidence_maps(sample: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    visible = sample.get("visible_stream") or sample.get("visible_summary") or []
    future = sample.get("future_stream") or sample.get("future_summary") or []

    if not isinstance(visible, list):
        visible = [visible]
    if not isinstance(future, list):
        future = [future]

    visible_map: Dict[str, str] = {}
    future_map: Dict[str, str] = {}

    for i, ev in enumerate(visible):
        ev_id, text = event_id_and_text(ev, "visible_evidence", i)
        visible_map[ev_id] = text

    for i, ev in enumerate(future):
        ev_id, text = event_id_and_text(ev, "future_or_retrospective_evidence", i)
        future_map[ev_id] = text

    return visible_map, future_map


def looks_retrospective(text: str) -> bool:
    s = text.lower()
    keywords = [
        "retrospective",
        "discharge",
        "discharge summary",
        "hospital course",
        "final diagnosis",
        "diagnosis code",
        "diagnoses",
        "icd",
        "ckd code",
        "problem list",
        "assessment",
        "after incorporating",
        "after review",
    ]
    return any(k in s for k in keywords)


def extract_gatekeeper(pred_row: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        safe_get(pred_row, "agent_outputs", "temporal_gatekeeper", default={}),
        safe_get(pred_row, "agent_outputs", "gatekeeper", default={}),
        safe_get(pred_row, "agents", "temporal_gatekeeper", default={}),
        safe_get(pred_row, "agents", "gatekeeper", default={}),
        pred_row.get("temporal_gatekeeper", {}),
        pred_row.get("gatekeeper", {}),
    ]
    for c in candidates:
        if isinstance(c, dict) and c:
            return c
    return {}


def extract_prediction(pred_row: Dict[str, Any]) -> Dict[str, Any]:
    pred = pred_row.get("prediction", {})
    if isinstance(pred, dict) and pred:
        return pred

    # 兼容极端情况：如果 prediction 没包起来，但字段直接在 pred_row 顶层
    direct = {}
    for field in FIELDS:
        if field in pred_row:
            direct[field] = pred_row.get(field)
    for field in ["initial_supporting_evidence_ids", "final_supporting_evidence_ids", "rationale"]:
        if field in pred_row:
            direct[field] = pred_row.get(field)
    return direct


def derive_support_safety(sample: Dict[str, Any], pred_row: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    visible_map, future_map = build_evidence_maps(sample)

    visible_ids = set(visible_map.keys())
    all_future_ids = set(future_map.keys())

    gatekeeper = extract_gatekeeper(pred_row)
    gatekeeper_future_ids = set(as_str_list(gatekeeper.get("future_evidence_ids", [])))
    gatekeeper_retro_ids = set(as_str_list(gatekeeper.get("retrospective_evidence_ids", [])))

    if gatekeeper_future_ids or gatekeeper_retro_ids:
        retrospective_ids = gatekeeper_retro_ids
        future_only_ids = gatekeeper_future_ids - retrospective_ids
        # 保底：如果 gatekeeper 漏了部分 future evidence，仍然把 sample future stream 视为非 visible
        future_or_retro_ids = all_future_ids | future_only_ids | retrospective_ids
    else:
        retrospective_ids = {ev_id for ev_id, text in future_map.items() if looks_retrospective(text)}
        future_only_ids = all_future_ids - retrospective_ids
        future_or_retro_ids = all_future_ids

    initial_support_ids = as_str_list(pred.get("initial_supporting_evidence_ids", []))
    final_support_ids = as_str_list(pred.get("final_supporting_evidence_ids", []))

    initial_support_set = set(initial_support_ids)

    valid_initial_ids = sorted(initial_support_set & visible_ids)
    future_leak_ids = sorted(initial_support_set & future_only_ids)
    retrospective_misuse_ids = sorted(initial_support_set & retrospective_ids)

    known_ids = visible_ids | future_or_retro_ids
    unknown_ids = sorted([x for x in initial_support_ids if x not in known_ids])

    invalid_temporal_ids = sorted(set(future_leak_ids) | set(retrospective_misuse_ids))
    invalid_or_unknown_ids = sorted(set(invalid_temporal_ids) | set(unknown_ids))

    temporal = pred_row.get("temporal_skill_verification", {}) or {}

    # 如果 pipeline 自带 verifier 标记，也纳入最终判断
    verifier_has_temporal = parse_bool_like(temporal.get("has_temporal_misuse", False))
    verifier_has_unknown = parse_bool_like(temporal.get("has_unknown_support", False))

    has_future_leakage = len(future_leak_ids) > 0
    has_retrospective_misuse = len(retrospective_misuse_ids) > 0
    has_temporal_misuse = has_future_leakage or has_retrospective_misuse or verifier_has_temporal
    has_unknown_support = len(unknown_ids) > 0 or verifier_has_unknown

    n_initial_support = len(initial_support_ids)
    n_valid_initial_support = len(valid_initial_ids)
    valid_initial_support_ratio = (
        n_valid_initial_support / n_initial_support if n_initial_support > 0 else None
    )

    return {
        "n_initial_support_ids": n_initial_support,
        "n_final_support_ids": len(final_support_ids),
        "n_valid_initial_support_ids": n_valid_initial_support,
        "n_future_leakage_ids": len(future_leak_ids),
        "n_retrospective_misuse_ids": len(retrospective_misuse_ids),
        "n_unknown_initial_support_ids": len(unknown_ids),
        "n_invalid_temporal_initial_support_ids": len(invalid_temporal_ids),
        "n_invalid_or_unknown_initial_support_ids": len(invalid_or_unknown_ids),
        "valid_initial_support_ratio": valid_initial_support_ratio,
        "has_future_leakage": has_future_leakage,
        "has_retrospective_misuse": has_retrospective_misuse,
        "has_temporal_misuse": has_temporal_misuse,
        "has_unknown_support": has_unknown_support,
        "initial_supporting_evidence_ids": "|".join(initial_support_ids),
        "final_supporting_evidence_ids": "|".join(final_support_ids),
        "valid_initial_support_ids": "|".join(valid_initial_ids),
        "future_leakage_ids": "|".join(future_leak_ids),
        "retrospective_misuse_ids": "|".join(retrospective_misuse_ids),
        "unknown_initial_support_ids": "|".join(unknown_ids),
        "invalid_temporal_initial_support_ids": "|".join(invalid_temporal_ids),
        "invalid_or_unknown_initial_support_ids": "|".join(invalid_or_unknown_ids),
    }


# =========================
# Reporting helpers
# =========================

def add_field_metrics(metrics: List[Dict[str, Any]], df: pd.DataFrame, prefix: str = "") -> None:
    for field in FIELDS:
        gold_col = f"gold_{field}"
        pred_col = f"pred_{field}"
        correct_col = f"correct_{field}"

        y_true = df[gold_col].tolist()
        y_pred = df[pred_col].tolist()
        labels = FIELD_LABELS[field]

        metrics.append({
            "metric": f"{prefix}accuracy_{field}",
            "value": float(df[correct_col].mean()),
        })

        metrics.append({
            "metric": f"{prefix}macro_f1_{field}",
            "value": macro_f1(y_true, y_pred, labels),
        })

        if field == "requires_delayed_reattribution":
            metrics.append({
                "metric": f"{prefix}positive_f1_{field}",
                "value": positive_f1(y_true, y_pred, True),
            })


def build_confusion_long(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for field in FIELDS:
        gold_col = f"gold_{field}"
        pred_col = f"pred_{field}"

        counts = (
            df.groupby([gold_col, pred_col], dropna=False)
            .size()
            .reset_index(name="count")
        )

        for _, r in counts.iterrows():
            rows.append({
                "field": field,
                "gold": r[gold_col],
                "pred": r[pred_col],
                "count": int(r["count"]),
            })

    return rows


def build_subtype_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for subtype, gdf in df.groupby("sample_subtype", dropna=False):
        row: Dict[str, Any] = {
            "sample_subtype": subtype,
            "n": len(gdf),
        }

        for field in FIELDS:
            gold_col = f"gold_{field}"
            pred_col = f"pred_{field}"
            correct_col = f"correct_{field}"
            row[f"accuracy_{field}"] = float(gdf[correct_col].mean())
            row[f"macro_f1_{field}"] = macro_f1(
                gdf[gold_col].tolist(),
                gdf[pred_col].tolist(),
                FIELD_LABELS[field],
            )

        row["future_leakage_rate"] = float(gdf["has_future_leakage"].mean())
        row["retrospective_misuse_rate"] = float(gdf["has_retrospective_misuse"].mean())
        row["temporal_misuse_rate"] = float(gdf["has_temporal_misuse"].mean())
        row["unknown_support_rate"] = float(gdf["has_unknown_support"].mean())
        row["invalid_initial_support_per_case"] = float(gdf["n_invalid_temporal_initial_support_ids"].mean())
        row["unknown_initial_support_per_case"] = float(gdf["n_unknown_initial_support_ids"].mean())

        total_support = int(gdf["n_initial_support_ids"].sum())
        total_valid = int(gdf["n_valid_initial_support_ids"].sum())
        row["valid_initial_support_ratio"] = (
            total_valid / total_support if total_support > 0 else None
        )

        rows.append(row)

    return rows


def build_support_safety_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    total_cases = len(df)
    total_initial_support = int(df["n_initial_support_ids"].sum())
    total_valid_initial_support = int(df["n_valid_initial_support_ids"].sum())
    total_future_leakage_ids = int(df["n_future_leakage_ids"].sum())
    total_retrospective_misuse_ids = int(df["n_retrospective_misuse_ids"].sum())
    total_unknown_support_ids = int(df["n_unknown_initial_support_ids"].sum())
    total_invalid_temporal_ids = int(df["n_invalid_temporal_initial_support_ids"].sum())
    total_invalid_or_unknown_ids = int(df["n_invalid_or_unknown_initial_support_ids"].sum())

    valid_ratio = (
        total_valid_initial_support / total_initial_support
        if total_initial_support > 0
        else None
    )

    rows = [
        {"metric": "n_cases", "value": total_cases},
        {"metric": "total_initial_support_ids", "value": total_initial_support},
        {"metric": "total_valid_initial_support_ids", "value": total_valid_initial_support},
        {"metric": "total_future_leakage_ids", "value": total_future_leakage_ids},
        {"metric": "total_retrospective_misuse_ids", "value": total_retrospective_misuse_ids},
        {"metric": "total_unknown_initial_support_ids", "value": total_unknown_support_ids},
        {"metric": "total_invalid_temporal_initial_support_ids", "value": total_invalid_temporal_ids},
        {"metric": "total_invalid_or_unknown_initial_support_ids", "value": total_invalid_or_unknown_ids},
        {"metric": "valid_initial_support_ratio", "value": valid_ratio},
        {"metric": "future_leakage_rate", "value": float(df["has_future_leakage"].mean())},
        {"metric": "retrospective_misuse_rate", "value": float(df["has_retrospective_misuse"].mean())},
        {"metric": "temporal_misuse_rate", "value": float(df["has_temporal_misuse"].mean())},
        {"metric": "unknown_support_rate", "value": float(df["has_unknown_support"].mean())},
        {"metric": "invalid_initial_support_per_case", "value": float(df["n_invalid_temporal_initial_support_ids"].mean())},
        {"metric": "unknown_initial_support_per_case", "value": float(df["n_unknown_initial_support_ids"].mean())},
        {"metric": "invalid_or_unknown_initial_support_per_case", "value": float(df["n_invalid_or_unknown_initial_support_ids"].mean())},
        {"metric": "mean_initial_support_ids_per_case", "value": float(df["n_initial_support_ids"].mean())},
    ]
    return rows


def infer_default_path(metrics_csv: Path, suffix: str) -> Path:
    stem = metrics_csv.stem
    if stem.endswith("_metrics"):
        base = stem[: -len("_metrics")]
    elif stem.endswith("_metrics_extended"):
        base = stem[: -len("_metrics_extended")]
    else:
        base = stem
    return metrics_csv.with_name(f"{base}_{suffix}")


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--errors-csv", type=Path, required=True)

    parser.add_argument("--per-sample-csv", type=Path, default=None)
    parser.add_argument("--subtype-metrics-csv", type=Path, default=None)
    parser.add_argument("--support-safety-csv", type=Path, default=None)
    parser.add_argument("--confusion-csv", type=Path, default=None)

    args = parser.parse_args()

    per_sample_csv = args.per_sample_csv or infer_default_path(args.metrics_csv, "per_sample.csv")
    subtype_metrics_csv = args.subtype_metrics_csv or infer_default_path(args.metrics_csv, "subtype_metrics.csv")
    support_safety_csv = args.support_safety_csv or infer_default_path(args.metrics_csv, "support_safety.csv")
    confusion_csv = args.confusion_csv or infer_default_path(args.metrics_csv, "confusion_matrices_long.csv")

    dataset_rows = read_jsonl(args.dataset_jsonl)
    pred_rows_all = read_jsonl(args.pred_jsonl)

    dataset = {str(x.get("sample_id", "")): x for x in dataset_rows if str(x.get("sample_id", ""))}
    preds_ok = [x for x in pred_rows_all if x.get("status") == "ok"]
    status_counts = Counter(str(x.get("status", "missing")) for x in pred_rows_all)

    rows: List[Dict[str, Any]] = []
    unmatched_pred_ids: List[str] = []

    for pred_row in preds_ok:
        sid = str(pred_row.get("sample_id", "")).strip()
        sample = dataset.get(sid)

        if not sample:
            unmatched_pred_ids.append(sid)
            continue

        gold = label_of(sample)
        pred = extract_prediction(pred_row)

        sample_subtype = (
            gold.get("sample_subtype")
            or safe_get(sample, "quality_flags_v2_2", "sample_subtype", default="")
            or safe_get(sample, "quality_flags_v2_1", "sample_subtype", default="")
            or sample.get("sample_subtype", "")
        )

        record: Dict[str, Any] = {
            "sample_id": sid,
            "sample_subtype": sample_subtype,
            "status": pred_row.get("status", ""),
        }

        for field in FIELDS:
            g = normalize_value(field, gold.get(field, ""))
            p = normalize_value(field, pred.get(field, ""))
            record[f"gold_{field}"] = g
            record[f"pred_{field}"] = p
            record[f"correct_{field}"] = int(g == p)

        safety = derive_support_safety(sample, pred_row, pred)
        record.update(safety)

        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matched prediction rows found. Check sample_id alignment and status == 'ok'.")

    # -------------------------
    # Overall metrics
    # -------------------------
    metrics: List[Dict[str, Any]] = [
        {"metric": "n_dataset_rows", "value": len(dataset_rows)},
        {"metric": "n_prediction_rows", "value": len(pred_rows_all)},
        {"metric": "n_prediction_status_ok", "value": len(preds_ok)},
        {"metric": "n_prediction_status_error", "value": status_counts.get("error", 0)},
        {"metric": "n_prediction_status_missing", "value": status_counts.get("missing", 0)},
        {"metric": "n_unmatched_prediction_ids", "value": len(unmatched_pred_ids)},
        {"metric": "n_evaluated", "value": len(df)},
        {"metric": "status_ok_rate", "value": len(preds_ok) / len(pred_rows_all) if pred_rows_all else 0.0},
    ]

    add_field_metrics(metrics, df)

    # Temporal safety decomposition
    metrics.extend(build_support_safety_metrics(df))

    # -------------------------
    # Error rows
    # -------------------------
    error_mask = pd.Series(False, index=df.index)
    for field in FIELDS:
        error_mask = error_mask | (df[f"correct_{field}"] == 0)

    error_mask = (
        error_mask
        | df["has_future_leakage"].astype(bool)
        | df["has_retrospective_misuse"].astype(bool)
        | df["has_temporal_misuse"].astype(bool)
        | df["has_unknown_support"].astype(bool)
    )

    errors_df = df[error_mask].copy()

    # -------------------------
    # Extra reports
    # -------------------------
    subtype_metrics = build_subtype_metrics(df)
    confusion_rows = build_confusion_long(df)
    support_safety_rows = build_support_safety_metrics(df)

    # -------------------------
    # Write outputs
    # -------------------------
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(args.metrics_csv, index=False, encoding="utf-8-sig")
    errors_df.to_csv(args.errors_csv, index=False, encoding="utf-8-sig")
    df.to_csv(per_sample_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(subtype_metrics).to_csv(subtype_metrics_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(confusion_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(support_safety_rows).to_csv(support_safety_csv, index=False, encoding="utf-8-sig")

    # -------------------------
    # Console summary
    # -------------------------
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    print(f"[INFO] Metrics: {args.metrics_csv}")
    print(f"[INFO] Errors: {args.errors_csv}")
    print(f"[INFO] Per-sample details: {per_sample_csv}")
    print(f"[INFO] Subtype metrics: {subtype_metrics_csv}")
    print(f"[INFO] Confusion matrices long table: {confusion_csv}")
    print(f"[INFO] Support safety metrics: {support_safety_csv}")

    if unmatched_pred_ids:
        print(f"[WARN] Unmatched prediction IDs: {len(unmatched_pred_ids)}")
        print(unmatched_pred_ids[:20])


if __name__ == "__main__":
    main()