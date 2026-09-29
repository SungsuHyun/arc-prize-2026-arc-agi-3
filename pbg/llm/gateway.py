"""llm/gateway.py — LLM Gateway (spec §14): tiered models, retries, 60 s timeout, cost metering, response cache.
Vendor-neutral: any OpenAI-compatible /chat/completions endpoint. Dependency-free (urllib)."""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

from .cache import ResponseCache

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
JSON_RE = re.compile(r"```json\s*\n(.*?)```", re.S)


def _dotenv() -> dict:
    """KEY=value pairs of the repository's .env (never printed, never committed)."""
    root = Path(__file__).resolve().parents[2]
    out = {}
    p = root / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_llm_config(path: Optional[Path] = None) -> dict:
    """llm.yaml by default; PBG_LLM_CONFIG selects another profile (e.g. pbg/llm/llm-opus.yaml). `api_key: ${VAR}` is
    expanded from the environment or the repository .env."""
    p = path or Path(os.environ.get("PBG_LLM_CONFIG") or Path(__file__).resolve().parent / "llm.yaml")
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    key = str(cfg.get("api_key", ""))
    if key.startswith("${") and key.endswith("}"):
        var = key[2:-1]
        cfg["api_key"] = os.environ.get(var) or _dotenv().get(var) or ""
    return cfg


def extract_code(text: str) -> Optional[str]:
    """The single python code block of a reply (the first one if several; None if none). A block whose closing fence
    was cut off by the token limit is salvaged as-is (the sandbox rejects it if it does not parse)."""
    if not text:
        return None
    if "```python" in text and text.count("```") % 2 == 1:
        text = text + "\n```"
    blocks = [b for b in CODE_RE.findall(text) if "def build_model" in b or "def build_goal" in b]
    if not blocks:
        blocks = CODE_RE.findall(text)
    return blocks[0].strip() if blocks else None


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = JSON_RE.search(text)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(cand)
    except Exception:
        return None


class LLMGateway:
    def __init__(self, config: Optional[dict] = None, *, cache_dir: Optional[Path] = None, log=None, max_calls: int = 60):
        self.cfg = config or load_llm_config()
        self.log = log or (lambda *a, **k: None)
        self.cache = ResponseCache(cache_dir or self.cfg.get("cache_dir"))
        self.calls = self.failures = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.latency_total = 0.0
        self.max_calls = max_calls
        self.history: list[dict] = []

    def tier(self, purpose: str) -> dict:
        name = (self.cfg.get("purposes") or {}).get(purpose, "reasoning")
        return dict((self.cfg.get("tiers") or {}).get(name) or {"model": "local-qwen", "temperature": 0.6, "top_p": 0.95, "max_tokens": 4000})

    def exhausted(self) -> bool:
        return self.calls >= self.max_calls

    def chat(self, messages: list[dict], *, purpose: str = "world_model", image: Optional[bytes] = None, use_cache: bool = True,
             override: Optional[dict] = None) -> str:
        tier = self.tier(purpose)
        if override:
            tier.update(override)
        if tier.get("no_sampling"):
            # hosted models that reject sampling parameters: candidates differ by prompt, not by temperature
            tier.pop("temperature", None); tier.pop("top_p", None)
        msgs = [dict(m) for m in messages]
        if image is not None:
            b64 = base64.b64encode(image).decode()
            last = msgs[-1]
            last["content"] = [{"type": "text", "text": last["content"]}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        params = {k: tier[k] for k in ("temperature", "top_p", "max_tokens") if k in tier}
        key = self.cache.key(msgs, tier["model"], params)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.history.append({"purpose": purpose, "cached": True, "chars": len(hit)})
                return hit
        if self.exhausted():
            raise RuntimeError(f"LLM call cap reached ({self.max_calls})")
        body = {"model": tier["model"], "messages": msgs, **params, **(tier.get("extra_body") or {})}
        data = json.dumps(body).encode()
        timeout = float(self.cfg.get("timeout", 60)); retries = int(self.cfg.get("retries", 2))
        last_err: Exception = RuntimeError("no attempt")
        for attempt in range(retries + 1):
            t0 = time.time(); self.calls += 1
            req = urllib.request.Request(self.cfg.get("base_url", "http://127.0.0.1:1234/v1").rstrip("/") + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.get('api_key', 'x')}"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    out = json.loads(resp.read())
                msg = out["choices"][0]["message"]
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not text.strip() and reasoning:
                    # the model spent its budget thinking: salvage a code block from the reasoning if there is one
                    text = reasoning if "```" in reasoning else ""
                    self.log(f"llm {purpose}: empty content, reasoning {len(reasoning)} chars ({'code found' if text else 'no code'})")
                usage = out.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0)); self.completion_tokens += int(usage.get("completion_tokens", 0))
                self.latency_total += time.time() - t0
                self.history.append({"purpose": purpose, "cached": False, "latency": round(time.time() - t0, 1), "chars": len(text), "usage": usage})
                if text.strip():
                    self.cache.put(key, text, {"purpose": purpose, "usage": usage})   # never cache an empty reply
                return text
            except urllib.error.HTTPError as e:
                last_err = RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
            except Exception as e:
                last_err = e
            self.failures += 1
            self.log(f"llm {purpose} attempt {attempt + 1} failed: {last_err!r}")
            if self.exhausted():
                break
            time.sleep(min(10, 2 ** attempt))
        raise last_err

    @staticmethod
    def extract_code(text: str) -> Optional[str]:
        return extract_code(text)

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        return extract_json(text)

    def stats(self) -> dict:
        return {"calls": self.calls, "failures": self.failures, "cache_hits": self.cache.hits, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_total": round(self.latency_total, 1)}
