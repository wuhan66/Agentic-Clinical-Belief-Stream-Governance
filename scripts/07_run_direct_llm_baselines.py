# 功能说明：
# 1. Run reusable direct-LLM baselines for C-BELIEF-Stream without using the multi-agent pipeline.
# 2. Supports visible_only, full_context, full_context_provenance, cbelief_prompted, or all four modes.
# 3. Supports model selection by provider + API model name or HF local model path, with automatic output naming.
# 4. Writes prediction JSONL in the same outer shape expected by scripts/06_evaluate_agentic_cbelief_extended.py.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbelief.agents.agent_prompts import BENCHMARK_LABEL_POLICY, TEMPORAL_SAFETY_POLICY  # noqa: E402
from cbelief.agents.agent_schemas import normalize_prediction  # noqa: E402
from cbelief.agents.json_utils import extract_json_object  # noqa: E402
from cbelief.agents.llm_client import ChatLLMClient  # noqa: E402
from cbelief.agents.temporal_skill_verifier import verify_temporal_support  # noqa: E402


CLAIM_BASELINE_MODES = [
    "visible_only",
    "full_context",
    "full_context_provenance",
    "cbelief_prompted",
]
NO_CLAIM_BASELINE_MODES = [f"{mode}_no_claim" for mode in CLAIM_BASELINE_MODES]
BASELINE_MODES = CLAIM_BASELINE_MODES + NO_CLAIM_BASELINE_MODES
MODE_CHOICES = BASELINE_MODES + ["all", "all_no_claim", "all_plus_no_claim"]


SYSTEM = (
    "You are a clinical reasoning assistant for longitudinal EHR evidence attribution. "
    "Return strict JSON only. Do not include markdown, explanations outside JSON, or extra text."
)


REQUIRED_JSON_SCHEMA = """
Return strict JSON with exactly these keys:
- initial_claim_status: supported | partially_supported | insufficient
- final_claim_status: supported | partially_supported | insufficient
- initial_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- final_primary_hypothesis: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality
- final_clinical_phenotype: acute_renal_deterioration | chronic_renal_dysfunction | uncertain_or_transient_abnormality | mixed_acute_on_chronic_renal_dysfunction
- requires_delayed_reattribution: boolean
- initial_supporting_evidence_ids: list[str]
- final_supporting_evidence_ids: list[str]
- rationale: string
""".strip()


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
            except Exception:
                continue
            sid = str(obj.get("sample_id", "")).strip()
            if not sid:
                continue
            if ok_only and obj.get("status") != "ok":
                continue
            done.add(sid)
    return done


def iter_samples(rows: List[Dict[str, Any]], limit: Optional[int], offset: int) -> Iterable[Dict[str, Any]]:
    selected = rows[offset:]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


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
    text = str(cuda_visible_devices).strip()
    if text:
        os.environ["CUDA_VISIBLE_DEVICES"] = text


def as_event_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def collect_events(sample: Dict[str, Any], keys: List[str]) -> List[Any]:
    events: List[Any] = []
    seen_ids: Set[str] = set()
    for key in keys:
        for ev in as_event_list(sample.get(key)):
            ev_id = None
            if isinstance(ev, dict):
                ev_id = str(ev.get("evidence_id") or ev.get("event_id") or ev.get("id") or "")
            fingerprint = ev_id or json.dumps(ev, ensure_ascii=False, sort_keys=True) if isinstance(ev, dict) else str(ev)
            if fingerprint in seen_ids:
                continue
            seen_ids.add(fingerprint)
            events.append(ev)
    return events


def event_id_and_text(event: Any, prefix: str, index: int) -> tuple[str, str]:
    if isinstance(event, dict):
        ev_id = str(event.get("evidence_id") or event.get("event_id") or event.get("id") or f"{prefix}_{index}")
        text = event.get("text") or event.get("event_text") or json.dumps(event, ensure_ascii=False)
        return ev_id, str(text)
    return f"{prefix}_{index}", str(event)


def format_events(events: List[Any], title: str, time_label: Optional[str] = None) -> str:
    if not events:
        return f"{title}: []"

    lines = [f"{title}:"]
    for i, ev in enumerate(events):
        ev_id, text = event_id_and_text(ev, title.lower().replace(" ", "_"), i)
        if time_label:
            lines.append(f"- {ev_id} [{time_label}]: {text}")
        else:
            lines.append(f"- {ev_id}: {text}")
    return "\n".join(lines)


def get_visible_events(sample: Dict[str, Any]) -> List[Any]:
    return collect_events(sample, ["visible_stream", "visible_summary", "visible_events"])


def get_future_events(sample: Dict[str, Any]) -> List[Any]:
    return collect_events(
        sample,
        [
            "future_stream",
            "future_summary",
            "future_observed_events",
            "retrospective_stream",
            "retrospective_summary",
            "retrospective_events",
        ],
    )


def build_prompt(sample: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    if mode not in BASELINE_MODES:
        raise ValueError(f"Unknown baseline mode: {mode}")

    visible = get_visible_events(sample)
    future = get_future_events(sample)
    no_claim = mode.endswith("_no_claim")
    base_mode = mode.removesuffix("_no_claim")

    header_parts = [
        f"""
sample_id: {sample.get('sample_id', '')}
query_time: {sample.get('query_time', '')}
""".strip()
    ]
    if not no_claim:
        header_parts.append(
            f"""
target_claim_initial: {sample.get('target_claim_initial', '')}
target_claim_final: {sample.get('target_claim_final', '')}
""".strip()
        )

    header_parts.append(
        """
Candidate hypotheses:
- acute_renal_deterioration
- chronic_renal_dysfunction
- uncertain_or_transient_abnormality
""".strip()
    )
    header = "\n\n".join(header_parts)

    if base_mode == "visible_only":
        task = """
Baseline setting: visible_only_direct_llm.
You receive only evidence visible at query time. Make both initial and final predictions from the visible evidence available in this prompt. Do not assume later diagnoses or later procedures unless explicitly visible here.
""".strip()
        evidence = format_events(visible, "visible_evidence", "visible")
        extra_rules = "Initial supporting evidence IDs must come from visible_evidence."

    elif base_mode == "full_context":
        task = """
Baseline setting: full_context_direct_llm_unmarked.
You receive all evidence together as ordinary clinical context. Make the best clinical prediction from the complete record.
This baseline is intentionally not given a strict temporal gatekeeping instruction or provenance labels.
""".strip()
        evidence = format_events(visible + future, "all_available_evidence", None)
        extra_rules = "Select support IDs from the evidence provided in the prompt."

    elif base_mode == "full_context_provenance":
        task = """
Baseline setting: full_context_direct_llm_with_provenance.
You receive all evidence, and each evidence item is marked as visible or future_or_retrospective.
Use visible evidence for initial/query-time labels. Use visible + future_or_retrospective evidence for final labels.
""".strip()
        evidence = "\n\n".join(
            [
                format_events(visible, "visible_evidence", "visible"),
                format_events(future, "future_or_retrospective_evidence", "future_or_retrospective"),
            ]
        )
        extra_rules = f"""
Temporal safety policy:
{TEMPORAL_SAFETY_POLICY}

Benchmark-specific label policy:
{BENCHMARK_LABEL_POLICY}
""".strip()

    else:  # cbelief_prompted
        task = """
Baseline setting: cbelief_prompted_direct_llm.
You are a single LLM prompted with the C-BELIEF temporal evidence-access policy, but you do not have a multi-agent decomposition.
Apply the benchmark label policy and temporal safety policy directly in one pass.
Use visible evidence only for initial/query-time labels. Use visible + future_or_retrospective evidence for final labels.
This baseline tests whether prompt-level temporal instruction alone can replace the full Agentic C-BELIEF architecture.
""".strip()
        evidence = "\n\n".join(
            [
                format_events(visible, "visible_evidence", "visible"),
                format_events(future, "future_or_retrospective_evidence", "future_or_retrospective"),
            ]
        )
        extra_rules = f"""
C-BELIEF temporal evidence-access policy:
{TEMPORAL_SAFETY_POLICY}

C-BELIEF benchmark-specific operational label policy:
{BENCHMARK_LABEL_POLICY}

Important distinction:
- This is a single-pass prompted LLM baseline.
- Do not simulate multiple agents.
- Do not report hidden chain-of-thought.
- Return only the final structured JSON.
""".strip()

    user = f"""
{task}

{header}

{evidence}

Rules:
{extra_rules}

{REQUIRED_JSON_SCHEMA}
""".strip()

    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def sanitize_tag(text: str) -> str:
    text = str(text or "").strip().replace("\\", "/")
    text = text.rstrip("/").split("/")[-1]
    text = text.lower()
    text = re.sub(r"[^a-z0-9._+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def infer_model_tag(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    if args.model_tag:
        return sanitize_tag(args.model_tag)
    if args.hf_model:
        return sanitize_tag(args.hf_model)
    cfg_hf = cfg_get(cfg, ["llm", "hf_local", "model_name_or_path"])
    provider = first_not_none(args.provider, cfg_get(cfg, ["llm", "backend"]), "mock")
    if provider == "hf_local" and cfg_hf:
        return sanitize_tag(str(cfg_hf))
    if args.model:
        return sanitize_tag(args.model)
    cfg_model = cfg_get(cfg, ["llm", "api", "model"])
    if cfg_model:
        return sanitize_tag(str(cfg_model))
    return sanitize_tag(str(provider))


def resolve_modes(mode: str) -> List[str]:
    if mode == "all":
        return list(CLAIM_BASELINE_MODES)
    if mode == "all_no_claim":
        return list(NO_CLAIM_BASELINE_MODES)
    if mode == "all_plus_no_claim":
        return list(BASELINE_MODES)
    return [mode]


def resolve_output_path(args: argparse.Namespace, mode: str, model_tag: str) -> Path:
    if args.output_jsonl is not None:
        if len(resolve_modes(args.mode)) > 1:
            raise ValueError("--output-jsonl can only be used when --mode is a single baseline mode. Use --output-dir with --mode all.")
        return args.output_jsonl

    output_dir = args.output_dir or Path("outputs/direct_baselines")
    limit_tag = "full" if args.limit is None or args.limit <= 0 else f"n{args.limit}"
    offset_tag = "" if args.offset == 0 else f"_offset{args.offset}"
    return output_dir / f"direct_llm_{model_tag}_{mode}_{limit_tag}{offset_tag}.jsonl"


def build_llm(args: argparse.Namespace, cfg: Dict[str, Any]) -> ChatLLMClient:
    provider = first_not_none(args.provider, cfg_get(cfg, ["llm", "backend"]), "mock")

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

    hf_model = first_not_none(args.hf_model, cfg_get(cfg, ["llm", "hf_local", "model_name_or_path"]), os.getenv("CBELIEF_HF_MODEL", None))
    hf_dtype = first_not_none(args.hf_dtype, cfg_get(cfg, ["llm", "hf_local", "torch_dtype"]), os.getenv("CBELIEF_HF_DTYPE", None), "float16")
    hf_device_map = first_not_none(args.hf_device_map, cfg_get(cfg, ["llm", "hf_local", "device_map"]), os.getenv("CBELIEF_HF_DEVICE_MAP", None), "auto")
    hf_max_new_tokens = first_not_none(args.hf_max_new_tokens, cfg_get(cfg, ["llm", "hf_local", "max_new_tokens"]), os.getenv("CBELIEF_HF_MAX_NEW_TOKENS", None), args.max_tokens)
    hf_do_sample = bool_from_any(first_not_none(args.hf_do_sample, cfg_get(cfg, ["llm", "hf_local", "do_sample"]), os.getenv("CBELIEF_HF_DO_SAMPLE", None)), False)
    hf_use_chat_template = bool_from_any(first_not_none(args.hf_use_chat_template, cfg_get(cfg, ["llm", "hf_local", "use_chat_template"]), os.getenv("CBELIEF_HF_USE_CHAT_TEMPLATE", None)), True)
    hf_trust_remote_code = bool_from_any(first_not_none(args.hf_trust_remote_code, cfg_get(cfg, ["llm", "hf_local", "trust_remote_code"]), os.getenv("CBELIEF_HF_TRUST_REMOTE_CODE", None)), True)

    cuda_visible_devices = first_not_none(args.cuda_visible_devices, cfg_get(cfg, ["llm", "hf_local", "cuda_visible_devices"]), os.getenv("CUDA_VISIBLE_DEVICES", None))
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


def run_one(llm: ChatLLMClient, sample: Dict[str, Any], mode: str, model_tag: str) -> Dict[str, Any]:
    messages = build_prompt(sample, mode)
    raw = llm.chat(messages)
    parsed = extract_json_object(raw)
    prediction = normalize_prediction(parsed)
    temporal_verification = verify_temporal_support(sample, prediction)

    return {
        "sample_id": sample.get("sample_id", ""),
        "method": f"direct_llm_{mode}",
        "baseline_mode": mode,
        "model_tag": model_tag,
        "prediction": prediction,
        "raw_prediction": parsed,
        "_raw_response": raw,
        "temporal_skill_verification": temporal_verification,
    }


def run_mode(
    llm: ChatLLMClient,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    mode: str,
    model_tag: str,
) -> Dict[str, Any]:
    output_jsonl = resolve_output_path(args, mode, model_tag)
    selected = list(iter_samples(rows, args.limit, args.offset))
    done_ids = load_done_ids(output_jsonl, ok_only=args.resume_ok_only) if args.resume else set()

    n_skipped = 0
    n_ok = 0
    n_error = 0

    for sample in tqdm(selected, desc=f"Direct LLM baseline: {mode}"):
        sid = str(sample.get("sample_id", "")).strip()
        if args.resume and sid in done_ids:
            n_skipped += 1
            continue

        try:
            out = run_one(llm, sample, mode, model_tag)
            out["status"] = "ok"
            n_ok += 1
        except Exception as e:
            out = {
                "sample_id": sid,
                "method": f"direct_llm_{mode}",
                "baseline_mode": mode,
                "model_tag": model_tag,
                "status": "error",
                "error": repr(e),
            }
            n_error += 1

        append_jsonl(output_jsonl, out)
        if args.sleep > 0:
            time.sleep(args.sleep)

    return {
        "mode": mode,
        "output_jsonl": str(output_jsonl),
        "selected": len(selected),
        "skipped": n_skipped,
        "new_ok": n_ok,
        "new_error": n_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/direct_baselines"))
    parser.add_argument("--mode", choices=MODE_CHOICES, required=True)
    parser.add_argument("--model-tag", default=None, help="Short name used in output filenames, e.g. qwen3_4b or phi3_5_mini.")

    parser.add_argument("--provider", choices=["mock", "deepseek", "openai_compatible", "api", "hf_local"], default=None)
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

    args = parser.parse_args()
    cfg = read_yaml(args.config)

    input_jsonl = args.input_jsonl or cfg_get(cfg, ["data", "input_jsonl"])
    if input_jsonl is None:
        raise ValueError("Missing --input-jsonl or data.input_jsonl in config.")
    input_jsonl = Path(input_jsonl)

    if args.limit is None:
        cfg_limit = cfg_get(cfg, ["data", "limit"], None)
        args.limit = int(cfg_limit) if cfg_limit is not None else None

    cfg_offset = cfg_get(cfg, ["data", "offset"], None)
    if args.offset == 0 and cfg_offset is not None:
        args.offset = int(cfg_offset)

    cfg_sleep = cfg_get(cfg, ["experiment", "sleep"], None)
    if cfg_sleep is not None and args.sleep == 0.0:
        args.sleep = float(cfg_sleep)

    rows = read_jsonl(input_jsonl)
    modes = resolve_modes(args.mode)
    model_tag = infer_model_tag(args, cfg)
    llm = build_llm(args, cfg)

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Modes: {', '.join(modes)}")
    print(f"[INFO] Model tag: {model_tag}")

    summaries = []
    for mode in modes:
        summaries.append(run_mode(llm, rows, args, mode, model_tag))

    for summary in summaries:
        print("[INFO] " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
