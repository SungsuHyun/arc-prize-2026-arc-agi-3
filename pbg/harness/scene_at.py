"""harness/scene_at.py — pull the Scene at any (game, level, step) of a recorded log for module-level tests (spec §16)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..core.types import Scene, Transition
from .replay_runner import replay_transitions


class Harness:
    def __init__(self, logs_root: Path):
        self.root = Path(logs_root)
        self._cache: dict[str, list[Transition]] = {}

    def raw_path(self, game: str) -> Path:
        p = self.root / game / "raw.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def transitions(self, game: str) -> list[Transition]:
        if game not in self._cache:
            records = [json.loads(l) for l in self.raw_path(game).read_text().splitlines() if l.strip()]
            self._cache[game], _, _ = replay_transitions(records, game)
        return self._cache[game]

    def scene_at(self, game: str, level: int, step: int) -> Optional[Scene]:
        """Scene after `step` actions on `level` (step 0 = the level's first scene)."""
        ts = [t for t in self.transitions(game) if t.level == level]
        if not ts:
            return None
        if step <= 0:
            return ts[0].before
        return ts[min(step, len(ts)) - 1].after
