# 功能说明：
# 1. 在原有 C-BELIEF_STREAM 项目结构上运行 Agentic C-BELIEF 多智能体推理。
# 2. 支持 mock / deepseek / openai_compatible / hf_local provider，兼容 API 与 HuggingFace 本地模型。
# 3. 支持 config YAML、limit、offset 和 resume，输出每条样本的 prediction、agent 中间结果和 temporal skill verification。

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbelief.agents import AgenticCBeliefPipeline  # noqa: E402
from cbelief.agents.llm_client import ChatLLMClient  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path, ok_only: bool = False) -> Set[str]:
    if not path.exists():
        return set()

    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = str(obj.get("sample_id", "")).strip()
                if not sid:
                    continue
                if ok_only and obj.get("status") != "ok":
                    continue
                done.add(sid)
            except Exception:
                continue
    return done


def iter_samples(
    rows: List[Dict[str, Any]],
    limit: Optional[int],
    offset: int = 0,
) -> Iterable[Dict[str, Any]]:
    sliced = rows[offset:]
    if limit is not None and limit > 0:
        sliced = sliced[:limit]
    return sliced


def read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)

    import yaml

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}

    if not isinstance(obj, dict):
        raise ValueError(f"Config YAML must be a dict: {path}")

    return obj


def cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def bool_from_any(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def set_cuda_visible_devices(cuda_visible_devices: Optional[str]) -> None:
    if cuda_visible_devices is None:
        return
    if str(cuda_visible_devices).strip() == "":
        return
    # 必须在 torch/transformers 真正加载模型前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()


def build_llm(args: argparse.Namespace, cfg: Dict[str, Any]) -> ChatLLMClient:
    provider = first_not_none(
        args.provider,
        cfg_get(cfg, ["llm", "backend"]),
        "mock",
    )

    # API 参数
    api_key = first_not_none(
        args.api_key,
        cfg_get(cfg, ["llm", "api", "api_key"]),
        os.getenv(str(cfg_get(cfg, ["llm", "api", "api_key_env"], "DEEPSEEK_API_KEY")), None),
    )
    api_base = first_not_none(
        args.api_base,
        cfg_get(cfg, ["llm", "api", "api_base"]),
        os.getenv(str(cfg_get(cfg, ["llm", "api", "api_base_env"], "DEEPSEEK_API_BASE")), None),
        cfg_get(cfg, ["llm", "api", "default_api_base"]),
    )
    model = first_not_none(
        args.model,
        cfg_get(cfg, ["llm", "api", "model"]),
        os.getenv(str(cfg_get(cfg, ["llm", "api", "model_env"], "DEEPSEEK_MODEL")), None),
        cfg_get(cfg, ["llm", "api", "default_model"]),
    )

    # HF local 参数
    hf_model = first_not_none(
        args.hf_model,
        cfg_get(cfg, ["llm", "hf_local", "model_name_or_path"]),
        os.getenv("CBELIEF_HF_MODEL", None),
    )
    hf_dtype = first_not_none(
        args.hf_dtype,
        cfg_get(cfg, ["llm", "hf_local", "torch_dtype"]),
        os.getenv("CBELIEF_HF_DTYPE", None),
        "float16",
    )
    hf_device_map = first_not_none(
        args.hf_device_map,
        cfg_get(cfg, ["llm", "hf_local", "device_map"]),
        os.getenv("CBELIEF_HF_DEVICE_MAP", None),
        "auto",
    )
    hf_max_new_tokens = first_not_none(
        args.hf_max_new_tokens,
        cfg_get(cfg, ["llm", "hf_local", "max_new_tokens"]),
        os.getenv("CBELIEF_HF_MAX_NEW_TOKENS", None),
        args.max_tokens,
    )
    hf_do_sample = bool_from_any(
        first_not_none(
            args.hf_do_sample,
            cfg_get(cfg, ["llm", "hf_local", "do_sample"]),
            os.getenv("CBELIEF_HF_DO_SAMPLE", None),
        ),
        default=False,
    )
    hf_use_chat_template = bool_from_any(
        first_not_none(
            args.hf_use_chat_template,
            cfg_get(cfg, ["llm", "hf_local", "use_chat_template"]),
            os.getenv("CBELIEF_HF_USE_CHAT_TEMPLATE", None),
        ),
        default=True,
    )
    hf_trust_remote_code = bool_from_any(
        first_not_none(
            args.hf_trust_remote_code,
            cfg_get(cfg, ["llm", "hf_local", "trust_remote_code"]),
            os.getenv("CBELIEF_HF_TRUST_REMOTE_CODE", None),
        ),
        default=True,
    )

    cuda_visible_devices = first_not_none(
        args.cuda_visible_devices,
        cfg_get(cfg, ["llm", "hf_local", "cuda_visible_devices"]),
        os.getenv("CUDA_VISIBLE_DEVICES", None),
    )
    if str(provider).lower().strip() == "hf_local":
        set_cuda_visible_devices(cuda_visible_devices)

    return ChatLLMClient(
        provider=str(provider),
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep=args.sleep,
        timeout=int(cfg_get(cfg, ["llm", "api", "timeout"], 120)),
        max_retries=int(cfg_get(cfg, ["llm", "api", "max_retries"], 3)),
        hf_model_name_or_path=hf_model,
        hf_torch_dtype=str(hf_dtype),
        hf_device_map=str(hf_device_map),
        hf_trust_remote_code=hf_trust_remote_code,
        hf_max_new_tokens=int(hf_max_new_tokens),
        hf_do_sample=hf_do_sample,
        hf_use_chat_template=hf_use_chat_template,
        strip_think=bool_from_any(cfg_get(cfg, ["generation", "strip_think_tags"], True), True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)

    parser.add_argument(
        "--provider",
        choices=["mock", "deepseek", "openai_compatible", "api", "hf_local"],
        default=None,
    )

    # API / OpenAI-compatible
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)

    # HF local
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-dtype", default=None)
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-new-tokens", type=int, default=None)
    parser.add_argument("--hf-do-sample", default=None)
    parser.add_argument("--hf-use-chat-template", default=None)
    parser.add_argument("--hf-trust-remote-code", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)

    # Generation / run control
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-ok-only",
        action="store_true",
        help="Resume only skips rows with status == ok. Failed rows will be retried.",
    )

    args = parser.parse_args()
    cfg = read_yaml(args.config)

    input_jsonl = args.input_jsonl or cfg_get(cfg, ["data", "input_jsonl"])
    output_jsonl = args.output_jsonl or cfg_get(cfg, ["experiment", "output_jsonl"])

    if input_jsonl is None:
        raise ValueError("Missing --input-jsonl or data.input_jsonl in config.")
    if output_jsonl is None:
        raise ValueError("Missing --output-jsonl or experiment.output_jsonl in config.")

    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)

    if args.limit is None:
        cfg_limit = cfg_get(cfg, ["data", "limit"], None)
        args.limit = int(cfg_limit) if cfg_limit is not None else None

    cfg_offset = cfg_get(cfg, ["data", "offset"], None)
    if args.offset == 0 and cfg_offset is not None:
        args.offset = int(cfg_offset)

    cfg_resume = cfg_get(cfg, ["experiment", "resume"], None)
    if cfg_resume is not None and not args.resume:
        args.resume = bool_from_any(cfg_resume, default=False)

    cfg_sleep = cfg_get(cfg, ["experiment", "sleep"], None)
    if cfg_sleep is not None and args.sleep == 0.0:
        args.sleep = float(cfg_sleep)

    rows = read_jsonl(input_jsonl)
    done_ids = load_done_ids(output_jsonl, ok_only=args.resume_ok_only) if args.resume else set()

    llm = build_llm(args, cfg)
    pipeline = AgenticCBeliefPipeline(llm)

    selected = list(iter_samples(rows, args.limit, args.offset))

    n_skipped = 0
    n_ok = 0
    n_error = 0

    for sample in tqdm(selected, desc="Agentic C-BELIEF"):
        sid = str(sample.get("sample_id", "")).strip()

        if args.resume and sid in done_ids:
            n_skipped += 1
            continue

        try:
            out = pipeline.run_one(sample)
            out["status"] = "ok"
            n_ok += 1
        except Exception as e:
            out = {
                "sample_id": sid,
                "method": "agentic_cbelief_v0",
                "status": "error",
                "error": repr(e),
            }
            n_error += 1

        append_jsonl(output_jsonl, out)

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Selected rows: {len(selected)}")
    print(f"[INFO] Skipped rows: {n_skipped}")
    print(f"[INFO] New ok rows: {n_ok}")
    print(f"[INFO] New error rows: {n_error}")
    print(f"[INFO] Output: {output_jsonl}")


if __name__ == "__main__":
    main()