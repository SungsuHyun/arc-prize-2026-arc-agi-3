"""llm/cache.py — response cache keyed by the context hash (spec §14: identical context -> cached reply)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


class ResponseCache:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, str] = {}
        self.hits = 0

    @staticmethod
    def key(messages: list[dict], model: str, params: dict) -> str:
        return hashlib.sha256(json.dumps({"m": messages, "model": model, "p": params}, sort_keys=True, default=str).encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[str]:
        if key in self.mem:
            self.hits += 1
            return self.mem[key]
        if self.path and (self.path / f"{key}.json").exists():
            try:
                v = json.loads((self.path / f"{key}.json").read_text())["text"]
                self.mem[key] = v; self.hits += 1
                return v
            except Exception:
                return None
        return None

    def put(self, key: str, text: str, meta: Optional[dict] = None) -> None:
        self.mem[key] = text
        if self.path:
            (self.path / f"{key}.json").write_text(json.dumps({"text": text, **(meta or {})}))
