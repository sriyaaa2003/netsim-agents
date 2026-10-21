"""LLM backends behind one small interface, with call, token and time accounting."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Callable


class LLM:
    """Base class: subclasses implement _generate(messages) -> str."""

    name = "llm"

    def __init__(self) -> None:
        self.calls = 0
        self.seconds = 0.0
        self.prompt_chars = 0
        self.output_chars = 0

    def generate(self, messages: list[dict]) -> str:
        start = time.perf_counter()
        out = self._generate(messages)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.prompt_chars += sum(len(m["content"]) for m in messages)
        self.output_chars += len(out)
        return out

    def _generate(self, messages: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError

    def snapshot(self) -> dict:
        return {"calls": self.calls, "seconds": self.seconds,
                "prompt_chars": self.prompt_chars, "output_chars": self.output_chars}


class ScriptedLLM(LLM):
    """Test double: a function (messages) -> str. Used only in unit tests, never for reported results."""

    name = "scripted"

    def __init__(self, fn: Callable[[list[dict]], str]) -> None:
        super().__init__()
        self.fn = fn

    def _generate(self, messages: list[dict]) -> str:
        return self.fn(messages)


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible chat endpoint, e.g. a local Ollama server (http://localhost:11434/v1)."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", max_tokens: int = 512) -> None:
        super().__init__()
        self.base_url, self.model, self.api_key, self.max_tokens = base_url.rstrip("/"), model, api_key, max_tokens
        self.name = f"openai-compat:{model}"

    def _generate(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]


class HFLocalLLM(LLM):
    """A Hugging Face causal LM run in-process with greedy decoding (used on Kaggle GPUs)."""

    def __init__(self, model_id: str, max_new_tokens: int = 512) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def _generate(self, messages: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    """Parse the first JSON object in an LLM reply (handles code fences and chatter). Raises ValueError."""
    candidates = [m for m in _FENCE.findall(text)] + [text]
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(cand)):
                depth += cand[i] == "{"
                depth -= cand[i] == "}"
                if depth == 0:
                    try:
                        return json.loads(cand[start:i + 1])
                    except json.JSONDecodeError:
                        break
            start = cand.find("{", start + 1)
    raise ValueError("no valid JSON object found in the reply")
