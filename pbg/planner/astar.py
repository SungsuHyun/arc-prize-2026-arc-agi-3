"""planner/astar.py — A* over model-predicted scenes (spec §10). Heuristic = weight * (1 - progress); revisiting a
state already seen during execution costs +5."""
from __future__ import annotations

import heapq
import itertools
import time
from typing import Callable, Iterable, Optional

from ..core.types import Action, Scene


def plan_key(scene: Scene) -> tuple:
    return scene.key()


def astar(start: Scene, successors: Callable[[Scene], Iterable[tuple[Action, Scene]]], is_goal: Callable[[Scene], bool],
          progress: Callable[[Scene], float], *, depth: int = 60, weight: float = 8.0, max_nodes: int = 100_000,
          visited_penalty: Optional[set] = None, time_limit: float = 0.5) -> Optional[list[Action]]:
    counter = itertools.count()
    start_key = plan_key(start)
    if is_goal(start):
        return []
    open_heap = [(weight * (1.0 - progress(start)), next(counter), 0.0, 0, start_key)]
    nodes = {start_key: (start, None, None, 0.0)}   # key -> (scene, parent_key, action, g)
    closed: set = set()
    expanded = 0; t0 = time.perf_counter()
    while open_heap:
        f, _, g, d, key = heapq.heappop(open_heap)
        if key in closed:
            continue
        closed.add(key)
        scene = nodes[key][0]
        if is_goal(scene):
            return _path(nodes, key)
        if d >= depth:
            continue
        expanded += 1
        if expanded > max_nodes or (expanded % 200 == 0 and time.perf_counter() - t0 > time_limit):
            return None
        for a, nxt in successors(scene):
            k = plan_key(nxt)
            if k in closed:
                continue
            step = 1.0 + (5.0 if visited_penalty and k in visited_penalty else 0.0)
            g2 = g + step
            old = nodes.get(k)
            if old is not None and old[3] <= g2:
                continue
            nodes[k] = (nxt, key, a, g2)
            h = weight * (1.0 - progress(nxt))
            heapq.heappush(open_heap, (g2 + h, next(counter), g2, d + 1, k))
    return None


def _path(nodes: dict, key) -> list[Action]:
    out = []
    while True:
        scene, parent, action, _ = nodes[key]
        if parent is None:
            break
        out.append(action); key = parent
    out.reverse()
    return out
