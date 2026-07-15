from __future__ import annotations

from pathlib import Path
import json
import typer
import pandas as pd
from rich import print

from cbelief.io.config import load_yaml
from cbelief.io.load_data import read_table, write_table, read_jsonl, write_jsonl
from cbelief.build.standardize_events import standardize_events
from cbelief.build.build_streams import build_patient_streams
from cbelief.build.build_samples import build_samples
from cbelief.build.split_data import split_by_patient
from cbelief.methods.cbelief_stream import CBeliefStream
from cbelief.eval.claim_metrics import claim_metrics
from cbelief.eval.evidence_metrics import evidence_metrics
from cbelief.eval.temporal_metrics import temporal_metrics

app = typer.Typer(add_completion=False)


def _path(cfg: dict, key: str) -> str:
    return cfg[key]


@app.command("validate-data")
def validate_data(config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    p = Path(_path(cfg, "input_events"))
    if not p.exists():
        raise typer.BadParameter(f"Missing input_events: {p}")
    df = read_table(p)
    report = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "event_id_unique": bool(df["event_id"].is_unique) if "event_id" in df else False,
        "missing_subject_id": int(df["subject_id"].isna().sum()) if "subject_id" in df else None,
    }
    out = Path(cfg["metrics_dir"]) / "data_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)


@app.command("standardize-events")
def standardize_events_cmd(config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    df = read_table(cfg["input_events"])
    out = standardize_events(df)
    write_table(out, cfg["standardized_events"])
    print(f"Wrote {len(out)} events to {cfg['standardized_events']}")


@app.command("build-streams")
def build_streams_cmd(config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    df = read_table(cfg["standardized_events"])
    streams = build_patient_streams(df, cfg["patient_streams"])
    print(f"Wrote {len(streams)} patient/admission streams to {cfg['patient_streams']}")


@app.command("build-samples")
def build_samples_cmd(config: str = "configs/paths.yaml", experiment: str = "configs/experiment.yaml"):
    cfg = load_yaml(config)
    exp = load_yaml(experiment)
    claim = exp.get("claim_templates", {}).get("acute_vs_chronic") or "The renal abnormality is more consistent with acute renal deterioration than chronic renal dysfunction."
    samples = build_samples(cfg["patient_streams"], cfg["samples"], claim=claim)
    print(f"Wrote {len(samples)} samples to {cfg['samples']}")


@app.command("split-data")
def split_data_cmd(config: str = "configs/paths.yaml", experiment: str = "configs/experiment.yaml"):
    cfg = load_yaml(config)
    exp = load_yaml(experiment)
    split = exp.get("split", {})
    report = split_by_patient(
        cfg["samples"], cfg["train_samples"], cfg["val_samples"], cfg["test_samples"],
        seed=exp.get("seed", 42), train=split.get("train", 0.70), val=split.get("val", 0.15)
    )
    print(report)


def _split_path(cfg: dict, split: str) -> str:
    return cfg[f"{split}_samples"]


@app.command("run-method")
def run_method(method: str = "cbelief", split: str = "val", config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    samples = read_jsonl(_split_path(cfg, split))
    if method != "cbelief":
        raise typer.BadParameter("Only method=cbelief is implemented here.")
    model = CBeliefStream(top_k=8)
    preds = [model.predict(s) for s in samples]
    out = Path(cfg["predictions_dir"]) / method / f"{split}.jsonl"
    write_jsonl(preds, out)
    print(f"Wrote {len(preds)} predictions to {out}")


@app.command("run-baseline")
def run_baseline(method: str = "rule_based", split: str = "val", config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    samples = read_jsonl(_split_path(cfg, split))
    if method == "rule_based":
        from cbelief.baselines.rule_based import predict
    elif method == "full_context_llm":
        from cbelief.baselines.full_context_llm import predict
    elif method == "visible_only_llm":
        from cbelief.baselines.visible_only_llm import predict
    else:
        raise typer.BadParameter(f"Unknown baseline: {method}")
    preds = [predict(s) for s in samples]
    out = Path(cfg["predictions_dir"]) / method / f"{split}.jsonl"
    write_jsonl(preds, out)
    print(f"Wrote {len(preds)} predictions to {out}")


@app.command("evaluate")
def evaluate(split: str = "val", config: str = "configs/paths.yaml"):
    cfg = load_yaml(config)
    samples = read_jsonl(_split_path(cfg, split))
    rows = []
    for pred_file in Path(cfg["predictions_dir"]).glob(f"*/{split}.jsonl"):
        preds = read_jsonl(pred_file)
        m = {}
        m.update(claim_metrics(samples, preds))
        m.update(evidence_metrics(samples, preds))
        m.update(temporal_metrics(samples, preds))
        m["method"] = pred_file.parent.name
        rows.append(m)
    df = pd.DataFrame(rows)
    out = Path(cfg["metrics_dir"]) / f"main_results_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df)
    print(f"Wrote {out}")


if __name__ == "__main__":
    app()
