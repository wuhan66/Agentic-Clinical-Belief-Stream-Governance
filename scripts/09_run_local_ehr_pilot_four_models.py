from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS: Dict[str, str] = {
    "qwen3_4b": "/home/wh/hf_models/Qwen3-4B",
    "qwen3_8b": "/home/wh/hf_models/Qwen3-8B",
    "minicpm3_4b": "/home/wh/hf_models/MiniCPM3-4B",
    "deepseek_r1_qwen7b": "/home/wh/hf_models/DeepSeek-R1-Distill-Qwen-7B",
}


def run_command(cmd: List[str]) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_command(
    model_key: str,
    model_path: str,
    args: argparse.Namespace,
) -> List[str]:
    run_name = f"{args.run_prefix}_{model_key}"
    cmd = [
        sys.executable,
        "scripts/08_run_local_ehr_pilot_experiment.py",
        "--provider",
        "hf_local",
        "--hf-model",
        model_path,
        "--hf-dtype",
        args.hf_dtype,
        "--hf-device-map",
        args.hf_device_map,
        "--hf-max-new-tokens",
        str(args.hf_max_new_tokens),
        "--hf-do-sample",
        str(args.hf_do_sample).lower(),
        "--hf-use-chat-template",
        str(args.hf_use_chat_template).lower(),
        "--hf-trust-remote-code",
        str(args.hf_trust_remote_code).lower(),
        "--run-name",
        run_name,
        "--resume",
        "--resume-ok-only",
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--sleep",
        str(args.sleep),
    ]

    if args.raw_jsonl is not None:
        cmd.extend(["--raw-jsonl", str(args.raw_jsonl)])
    if args.dataset_jsonl is not None:
        cmd.extend(["--dataset-jsonl", str(args.dataset_jsonl)])
    if args.output_dir is not None:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.offset:
        cmd.extend(["--offset", str(args.offset)])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])

    return cmd


def parse_model_overrides(values: Optional[List[str]]) -> Dict[str, str]:
    if not values:
        return {}

    overrides: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model override must be name=path, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Model override must be name=path, got: {value}")
        overrides[name] = path
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local EHR pilot Agentic C-BELIEF experiment for four HF local models."
    )
    parser.add_argument("--run-prefix", default="local_ehr_pilot_agentic")
    parser.add_argument("--raw-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DEFAULT_MODELS.keys()),
        default=None,
        help="Run only selected model keys.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Override or add a model as key=/path/to/model. Can be repeated.",
    )

    parser.add_argument("--hf-dtype", default="float16")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-max-new-tokens", type=int, default=2048)
    parser.add_argument("--hf-do-sample", default=False)
    parser.add_argument("--hf-use-chat-template", default=True)
    parser.add_argument("--hf-trust-remote-code", default=True)
    parser.add_argument("--cuda-visible-devices", default=None)

    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = dict(DEFAULT_MODELS)
    models.update(parse_model_overrides(args.model))

    selected_keys = args.only or list(DEFAULT_MODELS.keys())
    for key in selected_keys:
        if key not in models:
            raise KeyError(f"Unknown model key: {key}")
        run_command(build_command(key, models[key], args))

    print("[DONE] Four-model local EHR pilot batch finished.", flush=True)


if __name__ == "__main__":
    main()
