"""planner/stuck.py — stuck detector (spec §10): <= 5 distinct frame hashes in the last 20 steps, or no top-goal
progress increase in 15 steps."""
from __future__ import annotations

from collections import deque
from typing import Optional


class StuckDetector:
    def __init__(self, window: int = 20, min_distinct: int = 5, progress_window: int = 15):
        self.hashes: deque = deque(maxlen=window)
        self.progress: deque = deque(maxlen=progress_window)
        self.min_distinct = min_distinct
        self.visited: set[str] = set()

    def reset(self) -> None:
        self.hashes.clear(); self.progress.clear()

    def check(self, frame_hash: str, progress: Optional[float] = None) -> Optional[str]:
        self.hashes.append(frame_hash); self.visited.add(frame_hash)
        if progress is not None:
            self.progress.append(progress)
        if len(self.hashes) == self.hashes.maxlen and len(set(self.hashes)) <= self.min_distinct:
            return "few_distinct_states"
        if len(self.progress) == self.progress.maxlen and max(self.progress) <= self.progress[0] + 1e-9:
            return "no_progress"
        return None
