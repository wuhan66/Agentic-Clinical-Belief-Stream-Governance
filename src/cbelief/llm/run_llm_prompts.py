import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

CLAIM_LABELS = {"supported", "partially_supported", "insufficient"}

HYP_LABELS = {
    "acute_renal_deterioration",
    "chronic_renal_dysfunction",
    "uncertain_or_transient_abnormality",
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


def strip_code_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    raw = text or ""
    cleaned = strip_code_fence(raw)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, ""
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, ""
        except Exception as e:
            return None, f"json_parse_error: {e}"

    return None, "no_json_object_found"

import re


def canonicalize_claim_status(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", "", s)

    mapping = {
        "supported": "supported",
        "partially_supported": "partially_supported",
        "partiallysupported": "partially_supported",
        "partial": "partially_supported",
        "insufficient": "insufficient",
        "insufficent": "insufficient",
        "insufficien": "insufficient",
    }

    return mapping.get(s, "insufficient")


def canonicalize_hypothesis(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", "", s)

    mapping = {
        "acute_renal_deterioration": "acute_renal_deterioration",
        "acuterenaldeterioration": "acute_renal_deterioration",
        "acute": "acute_renal_deterioration",

        "chronic_renal_dysfunction": "chronic_renal_dysfunction",
        "chronicrenaldysfunction": "chronic_renal_dysfunction",
        "chronic": "chronic_renal_dysfunction",

        "uncertain_or_transient_abnormality": "uncertain_or_transient_abnormality",
        "uncertainortransientabnormality": "uncertain_or_transient_abnormality",
        "uncertain": "uncertain_or_transient_abnormality",
        "transient": "uncertain_or_transient_abnormality",
    }

    return mapping.get(s, "uncertain_or_transient_abnormality")

def normalize_string_list(x: Any) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        return [str(v) for v in x if v is not None]

    if isinstance(x, str):
        if not x.strip():
            return []
        return [x.strip()]

    return []


def normalize_prediction(
    parsed: Optional[Dict[str, Any]],
    sample_id: str,
    condition: str,
    response_text: str,
    parse_error: str,
) -> Dict[str, Any]:
    if parsed is None:
        return {
            "sample_id": sample_id,
            "method_name": condition,
            "claim_status": "insufficient",
            "primary_hypothesis": "uncertain_or_transient_abnormality",
            "supporting_evidence_ids": [],
            "retrospective_only_evidence_ids": [],
            "belief_distribution": {},
            "rationale": "",
            "parse_error": parse_error,
            "raw_response_text": response_text,
        }

    claim_status = canonicalize_claim_status(parsed.get("claim_status", ""))
    primary_hypothesis = canonicalize_hypothesis(parsed.get("primary_hypothesis", ""))

    supporting_ids = normalize_string_list(parsed.get("supporting_evidence_ids", []))
    retro_ids = normalize_string_list(parsed.get("retrospective_only_evidence_ids", []))

    return {
        "sample_id": sample_id,
        "method_name": condition,
        "claim_status": claim_status,
        "primary_hypothesis": primary_hypothesis,
        "supporting_evidence_ids": supporting_ids,
        "retrospective_only_evidence_ids": retro_ids,
        "belief_distribution": parsed.get("belief_distribution", {}),
        "rationale": str(parsed.get("rationale", "")),
        "parse_error": parse_error,
        "raw_response_text": response_text,
    }

def build_chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")

    if base_url.endswith("/chat/completions"):
        return base_url

    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"

    return base_url + "/chat/completions"


def call_chat_completion(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        api_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}") from e

    obj = json.loads(body)

    try:
        return obj["choices"][0]["message"]["content"]
    except Exception:
        pass

    try:
        return obj["choices"][0]["text"]
    except Exception:
        pass

    return json.dumps(obj, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--subset_path", default="data/llm/val_subset_100.jsonl")
    args = parser.parse_args()

    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip()
    api_key = os.environ.get("DEEPSEEK_API_KEY", DEFAULT_API_KEY).strip()
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()

    api_url = build_chat_completions_url(base_url)

    if not api_key:
        raise RuntimeError(
            "Missing DeepSeek API key. Please set environment variable DEEPSEEK_API_KEY."
        )

    if not model:
        raise RuntimeError("Missing model name. Please set DEEPSEEK_MODEL.")

    condition = args.condition

    prompt_path = f"data/llm/prompts/{condition}_prompts.jsonl"
    prompt_rows = load_jsonl(prompt_path)

    selected_prompt_rows = prompt_rows[args.start: args.start + args.limit]

    subset_samples = load_jsonl(args.subset_path)
    sample_by_id = {str(s["sample_id"]): s for s in subset_samples}

    selected_samples = []
    raw_rows = []
    pred_rows = []

    print(f"Condition: {condition}")
    print(f"Prompt path: {prompt_path}")
    print(f"Start: {args.start}")
    print(f"Limit: {args.limit}")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"API URL: {api_url}")
    print()

    for i, row in enumerate(selected_prompt_rows):
        sample_id = str(row["sample_id"])
        prompt = row["prompt"]

        print(f"[{i + 1}/{len(selected_prompt_rows)}] Calling LLM for sample_id={sample_id}")

        response_text = call_chat_completion(
            prompt=prompt,
            api_url=api_url,
            api_key=api_key,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

        parsed, parse_error = extract_json_object(response_text)

        pred = normalize_prediction(
            parsed=parsed,
            sample_id=sample_id,
            condition=condition,
            response_text=response_text,
            parse_error=parse_error,
        )

        raw_rows.append({
            "sample_id": sample_id,
            "condition": condition,
            "response_text": response_text,
            "parsed": parsed,
            "parse_error": parse_error,
        })

        pred_rows.append(pred)

        if sample_id not in sample_by_id:
            raise RuntimeError(f"sample_id not found in subset: {sample_id}")
        selected_samples.append(sample_by_id[sample_id])

        time.sleep(args.sleep)

    n_tag = f"n{len(pred_rows)}"

    raw_out = f"data/llm/raw/{condition}_{n_tag}_raw.jsonl"
    pred_out = f"data/predictions/llm/{condition}_{n_tag}_predictions.jsonl"
    sample_out = f"data/llm/eval_samples/{condition}_{n_tag}_samples.jsonl"

    save_jsonl(raw_rows, raw_out)
    save_jsonl(pred_rows, pred_out)
    save_jsonl(selected_samples, sample_out)

    print()
    print("Saved raw responses to:", raw_out)
    print("Saved predictions to:", pred_out)
    print("Saved aligned eval samples to:", sample_out)

    n_parse_errors = sum(1 for p in pred_rows if p.get("parse_error"))
    print("Parse errors:", n_parse_errors)


if __name__ == "__main__":
    main()
