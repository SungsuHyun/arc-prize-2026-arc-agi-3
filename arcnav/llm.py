"""Minimal OpenAI-compatible chat client (tool calling, reasoning field),
dependency-free so it runs inside the Kaggle rerun container."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatResult:
    message: dict
    usage: dict = field(default_factory=dict)
    reasoning: str = ""
    latency: float = 0.0
    finish_reason: str = ""


class ContextLengthError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "local-qwen", api_key: str = "x",
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4096, timeout: float = 600.0,
                 extra_body: Optional[dict] = None):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.top_p, self.max_tokens, self.timeout = temperature, top_p, max_tokens, timeout
        self.extra_body = extra_body or {}
        self.prompt_tokens = self.completion_tokens = 0

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, *, tool_choice: Any = "auto", retries: int = 3,
             override: Optional[dict] = None) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature, "top_p": self.top_p,
                                "max_tokens": self.max_tokens, **self.extra_body, **(override or {})}
        if tools:
            body["tools"] = tools; body["tool_choice"] = tool_choice
        data = json.dumps(body).encode()
        last: Exception = RuntimeError("no attempt")
        for attempt in range(retries):
            t0 = time.time()
            req = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code == 400 and ("context length" in text or "maximum context" in text or "too long" in text):
                    raise ContextLengthError(text[:300])
                last = RuntimeError(f"HTTP {e.code}: {text[:300]}")
            except Exception as e:  # network / timeout
                last = e
            else:
                choice = out["choices"][0]; msg = choice["message"]
                usage = out.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0)); self.completion_tokens += int(usage.get("completion_tokens", 0))
                reasoning = msg.pop("reasoning_content", None) or msg.pop("reasoning", None) or ""
                msg.setdefault("content", "")
                if msg.get("content") is None:
                    msg["content"] = ""
                return ChatResult(message=msg, usage=usage, reasoning=reasoning, latency=time.time() - t0, finish_reason=choice.get("finish_reason", ""))
            time.sleep(min(30, 2 ** attempt))
        raise last
