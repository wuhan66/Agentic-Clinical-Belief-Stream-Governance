from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapt_local_ehr_pilot_to_eval import adapt_file  # noqa: E402


DEFAULT_RAW_JSONL = Path(
    r"D:\下载\npj_digital_medicine\outputs\local_ehr_pilot\local_cbelief_stream_300.jsonl"
)
DEFAULT_DATASET_JSONL = Path("data/local_ehr_pilot/local_cbelief_stream_300_eval_compatible.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/local_ehr_pilot")


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(cmd: List[str], env: Optional[dict[str, str]] = None) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_inference_command(args: argparse.Namespace, dataset_jsonl: Path, pred_jsonl: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/05_run_agentic_cbelief.py",
        "--input-jsonl",
        str(dataset_jsonl),
        "--output-jsonl",
        str(pred_jsonl),
        "--provider",
        args.provider,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--sleep",
        str(args.sleep),
        "--offset",
        str(args.offset),
    ]

    if args.config is not None:
        cmd.extend(["--config", str(resolve_project_path(args.config))])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.resume:
        cmd.append("--resume")
    if args.resume_ok_only:
        cmd.append("--resume-ok-only")

    optional_pairs = [
        ("--api-key", args.api_key),
        ("--api-base", args.api_base),
        ("--model", args.model),
        ("--hf-model", args.hf_model),
        ("--hf-dtype", args.hf_dtype),
        ("--hf-device-map", args.hf_device_map),
        ("--hf-max-new-tokens", args.hf_max_new_tokens),
        ("--hf-do-sample", args.hf_do_sample),
        ("--hf-use-chat-template", args.hf_use_chat_template),
        ("--hf-trust-remote-code", args.hf_trust_remote_code),
        ("--cuda-visible-devices", args.cuda_visible_devices),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            cmd.extend([flag, str(value)])

    return cmd


def build_eval_command(dataset_jsonl: Path, pred_jsonl: Path, output_dir: Path, run_name: str) -> List[str]:
    return [
        sys.executable,
        "scripts/06_evaluate_agentic_cbelief_extended.py",
        "--dataset-jsonl",
        str(dataset_jsonl),
        "--pred-jsonl",
        str(pred_jsonl),
        "--metrics-csv",
        str(output_dir / f"{run_name}_metrics.csv"),
        "--errors-csv",
        str(output_dir / f"{run_name}_errors.csv"),
        "--per-sample-csv",
        str(output_dir / f"{run_name}_per_sample.csv"),
        "--subtype-metrics-csv",
        str(output_dir / f"{run_name}_subtype_metrics.csv"),
        "--support-safety-csv",
        str(output_dir / f"{run_name}_support_safety.csv"),
        "--confusion-csv",
        str(output_dir / f"{run_name}_confusion_matrices_long.csv"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local EHR pilot experiment end to end: adapt local data, run Agentic C-BELIEF, "
            "and evaluate predictions with the extended evaluator."
        )
    )

    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--skip-adapt", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this run's existing prediction and report files before running.",
    )

    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--provider",
        choices=["mock", "deepseek", "openai_compatible", "api", "hf_local"],
        default="mock",
    )

    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)

    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-dtype", default=None)
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-new-tokens", type=int, default=None)
    parser.add_argument("--hf-do-sample", default=None)
    parser.add_argument("--hf-use-chat-template", default=None)
    parser.add_argument("--hf-trust-remote-code", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-ok-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_jsonl = resolve_project_path(args.raw_jsonl)
    dataset_jsonl = resolve_project_path(args.dataset_jsonl)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    if run_name is None:
        limit_part = "full" if args.limit is None else f"n{args.limit}"
        run_name = f"local_ehr_pilot_agentic_{args.provider}_{limit_part}"

    pred_jsonl = output_dir / f"{run_name}_predictions.jsonl"
    report_paths = [
        pred_jsonl,
        output_dir / f"{run_name}_metrics.csv",
        output_dir / f"{run_name}_errors.csv",
        output_dir / f"{run_name}_per_sample.csv",
        output_dir / f"{run_name}_subtype_metrics.csv",
        output_dir / f"{run_name}_support_safety.csv",
        output_dir / f"{run_name}_confusion_matrices_long.csv",
    ]

    if args.overwrite:
        for path in report_paths:
            remove_if_exists(path)

    if not args.skip_adapt:
        adapted = adapt_file(raw_jsonl, dataset_jsonl)
        print(f"[INFO] Adapted dataset: {dataset_jsonl}", flush=True)
        print(f"[INFO] Samples: {len(adapted)}", flush=True)
    elif not dataset_jsonl.exists():
        raise FileNotFoundError(f"--skip-adapt was set, but dataset JSONL does not exist: {dataset_jsonl}")

    if not args.skip_inference:
        env = os.environ.copy()
        run_command(build_inference_command(args, dataset_jsonl, pred_jsonl), env=env)
    elif not pred_jsonl.exists():
        raise FileNotFoundError(f"--skip-inference was set, but prediction JSONL does not exist: {pred_jsonl}")

    if not args.skip_eval:
        run_command(build_eval_command(dataset_jsonl, pred_jsonl, output_dir, run_name))

    print("[DONE] Local EHR pilot experiment finished.", flush=True)
    print(f"[INFO] Dataset: {dataset_jsonl}", flush=True)
    print(f"[INFO] Predictions: {pred_jsonl}", flush=True)
    if not args.skip_eval:
        print(f"[INFO] Metrics: {output_dir / f'{run_name}_metrics.csv'}", flush=True)
        print(f"[INFO] Per-sample: {output_dir / f'{run_name}_per_sample.csv'}", flush=True)


if __name__ == "__main__":
    main()
