# 功能说明：
# 1. 封装 Agentic C-BELIEF 的 LLM 调用接口。
# 2. 支持 mock、deepseek/openai_compatible API、HuggingFace 本地模型 hf_local 三种模式。
# 3. hf_local 模式会在服务器本地加载模型，避免 API 成本，适合开源模型批量实验。

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


def strip_think_tags(text: str) -> str:
    """去除 reasoning 模型可能输出的 <think>...</think> 内容。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def mock_response(messages: List[Dict[str, str]]) -> str:
    """mock provider 的固定输出，只用于测试链路。"""
    return json.dumps(
        {
            "initial_claim_status": "supported",
            "final_claim_status": "supported",
            "initial_primary_hypothesis": "acute_renal_deterioration",
            "final_primary_hypothesis": "acute_renal_deterioration",
            "final_clinical_phenotype": "acute_renal_deterioration",
            "requires_delayed_reattribution": False,
            "initial_supporting_evidence_ids": [],
            "final_supporting_evidence_ids": [],
            "rationale": "mock response",
        },
        ensure_ascii=False,
    )


class ChatLLMClient:
    def __init__(
        self,
        provider: str = "mock",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        sleep: float = 0.0,
        timeout: int = 120,
        max_retries: int = 3,
        hf_model_name_or_path: Optional[str] = None,
        hf_torch_dtype: str = "float16",
        hf_device_map: str = "auto",
        hf_trust_remote_code: bool = False,
        hf_max_new_tokens: Optional[int] = None,
        hf_do_sample: bool = False,
        hf_use_chat_template: bool = True,
        hf_use_cache: bool = True,
        strip_think: bool = True,
    ) -> None:
        self.provider = provider.lower().strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep = sleep
        self.timeout = timeout
        self.max_retries = max_retries
        self.strip_think = strip_think

        # API 参数
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.api_base = api_base or os.getenv(
            "DEEPSEEK_API_BASE",
            "https://api.deepseek.com/chat/completions",
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # HF local 参数
        self.hf_model_name_or_path = (
            hf_model_name_or_path
            or os.getenv("CBELIEF_HF_MODEL", "")
        )
        self.hf_torch_dtype = os.getenv("CBELIEF_HF_DTYPE", hf_torch_dtype)
        self.hf_device_map = os.getenv("CBELIEF_HF_DEVICE_MAP", hf_device_map)
        self.hf_trust_remote_code = (
            os.getenv("CBELIEF_HF_TRUST_REMOTE_CODE", str(hf_trust_remote_code))
            .lower()
            .strip()
            in {"true", "1", "yes", "y"}
        )
        self.hf_max_new_tokens = int(
            os.getenv(
                "CBELIEF_HF_MAX_NEW_TOKENS",
                str(hf_max_new_tokens or max_tokens),
            )
        )
        self.hf_do_sample = (
            os.getenv("CBELIEF_HF_DO_SAMPLE", str(hf_do_sample))
            .lower()
            .strip()
            in {"true", "1", "yes", "y"}
        )
        self.hf_use_chat_template = (
            os.getenv("CBELIEF_HF_USE_CHAT_TEMPLATE", str(hf_use_chat_template))
            .lower()
            .strip()
            in {"true", "1", "yes", "y"}
        )
        self.hf_use_cache = (
            os.getenv("CBELIEF_HF_USE_CACHE", str(hf_use_cache))
            .lower()
            .strip()
            in {"true", "1", "yes", "y"}
        )

        self._tokenizer = None
        self._model = None

        if self.provider == "hf_local":
            self._load_hf_model()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "mock":
            return mock_response(messages)

        if self.provider in {"deepseek", "api", "openai_compatible"}:
            return self._chat_api(messages)

        if self.provider == "hf_local":
            return self._chat_hf_local(messages)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_api(self, messages: List[Dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]

                if self.strip_think:
                    text = strip_think_tags(text)

                if self.sleep > 0:
                    time.sleep(self.sleep)

                return text.strip()

            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 8)
                print(f"[WARN] API call failed at attempt {attempt + 1}: {repr(e)}; sleep {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"API call failed after retries: {repr(last_err)}")

    def _load_hf_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.hf_model_name_or_path:
            raise ValueError(
                "hf_local requires model path. Set --hf-model or export CBELIEF_HF_MODEL."
            )

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(self.hf_torch_dtype.lower(), torch.float16)

        print(f"[INFO] Loading tokenizer: {self.hf_model_name_or_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name_or_path,
            trust_remote_code=self.hf_trust_remote_code,
        )

        print(f"[INFO] Loading model: {self.hf_model_name_or_path}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=self.hf_device_map,
            trust_remote_code=self.hf_trust_remote_code,
        )

        self._model.eval()
        print("[INFO] HF local model loaded.")

    def _chat_hf_local(self, messages: List[Dict[str, str]]) -> str:
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("HF local model is not loaded.")

        tokenizer = self._tokenizer
        model = self._model

        if self.hf_use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
    )
        else:
            prompt = self._messages_to_plain_prompt(messages)

        inputs = tokenizer(prompt, return_tensors="pt")

        # device_map=auto 时 model.device 不一定可靠，取第一个参数所在 device
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.hf_max_new_tokens,
            "do_sample": self.hf_do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": self.hf_use_cache,
        }

        if self.hf_do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if self.strip_think:
            text = strip_think_tags(text)

        if self.sleep > 0:
            time.sleep(self.sleep)

        return text.strip()

    @staticmethod
    def _messages_to_plain_prompt(messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)


def build_llm_client(provider: str) -> LLMClient:
    return LLMClient.from_env(provider=provider)
