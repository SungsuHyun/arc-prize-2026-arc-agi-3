"""perception/track.py — object tracking across frames (spec §6 step 4): Hungarian matching with
cost = 10 * colour mismatch + 5 * shape mismatch + centroid distance (per cell); above track_cost_max -> unmatched.
Tie-break: the previous displacement direction (spec §15 'tracking id swap')."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Object


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular cost matrix (rows <= cols after padding). Returns (row, col) pairs."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T; n, m = m, n; transposed = True
    INF = float("inf")
    u = [0.0] * (n + 1); v = [0.0] * (m + 1); p = [0] * (m + 1); way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = [INF] * (m + 1); used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = INF; j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, m + 1):
        if p[j]:
            pairs.append((p[j] - 1, j - 1))
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return pairs


class Tracker:
    def __init__(self, cost_max: float = 20.0):
        self.cost_max = cost_max
        self.next_id = 0
        self.last_disp: dict[int, tuple[int, int]] = {}

    def match(self, prev: Optional[list[Object]], cur: list[Object]) -> dict[int, int]:
        """Returns {current provisional index -> tracking id}; unmatched current objects get fresh ids."""
        assign: dict[int, int] = {}
        if not prev:
            for i, _ in enumerate(cur):
                assign[i] = self.next_id; self.next_id += 1
            return assign
        n, m = len(prev), len(cur)
        cost = np.zeros((n, m))
        for i, a in enumerate(prev):
            ar, ac = a.center
            for j, b in enumerate(cur):
                br, bc = b.center
                c = 10.0 * (a.color != b.color) + 5.0 * (a.shape_sig != b.shape_sig) + abs(ar - br) + abs(ac - bc)
                if a.color != b.color and abs(ar - br) + abs(ac - bc) > 2:
                    c = 1e6          # a different colour somewhere else is another object (a button never becomes the panel next to it)
                if a.colors != b.colors and a.color == b.color:
                    c += 2.0
                d = self.last_disp.get(a.id)
                if d is not None and (d[0] or d[1]):
                    # tie-break: prefer continuing the previous displacement direction
                    dr, dc = br - ar, bc - ac
                    if (dr, dc) != (0, 0) and (np.sign(dr) != np.sign(d[0]) or np.sign(dc) != np.sign(d[1])):
                        c += 0.5
                cost[i, j] = c
        used_ids = set()
        for i, j in hungarian(cost):
            if cost[i, j] <= self.cost_max:
                assign[j] = prev[i].id; used_ids.add(prev[i].id)
                ar, ac = prev[i].center; br, bc = cur[j].center
                self.last_disp[prev[i].id] = (br - ar, bc - ac)
        for j in range(m):
            if j not in assign:
                assign[j] = self.next_id; self.next_id += 1
        return assign
