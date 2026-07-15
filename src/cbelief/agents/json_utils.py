from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable


def _loads_dict(candidate: str) -> Dict[str, Any]:
    try:
        obj = json.loads(candidate.strip())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _balanced_json_candidates(text: str) -> Iterable[str]:
    in_string = False
    escape = False
    start = None
    depth = 0
    candidates = []

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escape = False
            continue

        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start : i + 1])
                start = None

    return candidates


def _decode_escaped_text(text: str) -> str:
    stripped = text.strip()
    if not any(token in stripped for token in (r"\n", r"\"", r"\\", r"\t")):
        return stripped

    return (
        stripped.replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def _candidate_texts(text: str) -> Iterable[str]:
    yield text

    # R1-style models often emit long reasoning before the final JSON block.
    if "</think>" in text:
        yield text.rsplit("</think>", 1)[-1]

    for block in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I):
        yield block.group(1)

    decoded = _decode_escaped_text(text)
    if decoded != text:
        yield decoded
        if "</think>" in decoded:
            yield decoded.rsplit("</think>", 1)[-1]
        for block in re.finditer(r"```(?:json)?\s*(.*?)\s*```", decoded, flags=re.S | re.I):
            yield block.group(1)


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    for candidate in _candidate_texts(text.strip()):
        obj = _loads_dict(candidate)
        if obj:
            return obj

        decoded_candidate = _decode_escaped_text(candidate)
        if decoded_candidate != candidate:
            obj = _loads_dict(decoded_candidate)
            if obj:
                return obj

            for json_text in reversed(list(_balanced_json_candidates(decoded_candidate))):
                obj = _loads_dict(json_text)
                if obj:
                    return obj

        for json_text in reversed(list(_balanced_json_candidates(candidate))):
            obj = _loads_dict(json_text)
            if obj:
                return obj

    return {}
