"""planner/beam.py — beam search (width 32) for large deterministic state spaces (spec §10)."""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from ..core.types import Action, Scene
from .astar import plan_key


def beam_search(start: Scene, successors: Callable[[Scene], Iterable[tuple[Action, Scene]]], is_goal: Callable[[Scene], bool],
                progress: Callable[[Scene], float], *, depth: int = 60, width: int = 32, time_limit: float = 2.0) -> Optional[list[Action]]:
    import time
    t0 = time.perf_counter()
    if is_goal(start):
        return []
    beam = [(progress(start), start, [])]
    seen = {plan_key(start)}
    for _ in range(depth):
        if time.perf_counter() - t0 > time_limit:
            return None
        cand = []
        for _, scene, path in beam:
            for a, nxt in successors(scene):
                k = plan_key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                p2 = path + [a]
                if is_goal(nxt):
                    return p2
                cand.append((progress(nxt), nxt, p2))
        if not cand:
            return None
        cand.sort(key=lambda x: -x[0])
        beam = cand[:width]
    return None
