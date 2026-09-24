"""Observation encoding: grid -> ascii, connected-component segmentation,
object hashes, containment and adjacency. Pure python, no dependencies."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# 16 ARC colours as single characters (hex digits keep the mapping obvious)
COLOR_CHARS = "0123456789abcdef"
COLOR_NAMES = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "grey", 6: "magenta",
               7: "orange", 8: "sky", 9: "brown", 10: "white", 11: "purple", 12: "cyan",
               13: "lime", 14: "pink", 15: "teal"}


def grid_to_ascii(grid: list[list[int]]) -> str:
    return "\n".join("".join(COLOR_CHARS[v & 15] for v in row) for row in grid)


def _shape_hash(cells: list[tuple[int, int]], color: int) -> str:
    y0 = min(r for r, _ in cells); x0 = min(c for _, c in cells)
    norm = tuple(sorted((r - y0, c - x0) for r, c in cells))
    return f"{color:x}:{abs(hash(norm)) % 10**8:08d}"


@dataclass
class Frame:
    grid: list[list[int]]
    step: int = 0
    level: int = 1
    _ascii: Optional[str] = field(default=None, repr=False)
    _seg: Optional[dict] = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.grid), len(self.grid[0]) if self.grid else 0)

    @property
    def ascii(self) -> str:
        if self._ascii is None:
            self._ascii = grid_to_ascii(self.grid)
        return self._ascii

    @property
    def background(self) -> int:
        return Counter(v for row in self.grid for v in row).most_common(1)[0][0]

    @property
    def segmentation(self) -> dict:
        """{'nodes': [...], 'adjacency': [[i, j], ...], 'background': color}.
        Node: id, color, pixels, bbox (r0, c0, r1, c1), center (row, col), hash,
        children (ids fully inside this node's bbox), hud (edge strip heuristic)."""
        if self._seg is None:
            self._seg = segment(self.grid)
        return self._seg

    def to_payload(self) -> dict:
        return {"grid": self.grid, "step": self.step, "level": self.level}

    @staticmethod
    def from_payload(p: dict) -> "Frame":
        return Frame(grid=[list(map(int, r)) for r in p["grid"]], step=int(p.get("step", 0)), level=int(p.get("level", 1)))


def segment(grid: list[list[int]], max_nodes: int = 60) -> dict:
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    comp = [[-1] * w for _ in range(h)]
    nodes: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            if comp[sr][sc] >= 0 or grid[sr][sc] == bg:
                continue
            color = grid[sr][sc]; nid = len(nodes)
            stack, cells = [(sr, sc)], []
            comp[sr][sc] = nid
            while stack:
                r, c = stack.pop(); cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and comp[nr][nc] < 0 and grid[nr][nc] == color:
                        comp[nr][nc] = nid; stack.append((nr, nc))
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            hud = ((r1 - r0 <= 2 and c1 - c0 >= 24 and (r0 <= 1 or r1 >= h - 2)) or
                   (c1 - c0 <= 2 and r1 - r0 >= 24 and (c0 <= 1 or c1 >= w - 2)))
            nodes.append({"id": nid, "color": color, "pixels": len(cells), "bbox": (r0, c0, r1, c1),
                          "center": ((r0 + r1) // 2, (c0 + c1) // 2), "hash": _shape_hash(cells, color),
                          "children": [], "hud": hud})
    # adjacency (edge-sharing components, ignoring background)
    adj: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            a = comp[r][c]
            if a < 0:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr < h and nc < w:
                    b = comp[nr][nc]
                    if b >= 0 and b != a:
                        adj.add((min(a, b), max(a, b)))
    # containment by bbox (small inside larger)
    for n in nodes:
        r0, c0, r1, c1 = n["bbox"]
        for m in nodes:
            if m is n:
                continue
            mr0, mc0, mr1, mc1 = m["bbox"]
            if r0 < mr0 and c0 < mc0 and mr1 < r1 and mc1 < c1:
                n["children"].append(m["id"])
    keep = sorted(nodes, key=lambda n: -n["pixels"])[:max_nodes]
    keep_ids = {n["id"] for n in keep}
    return {"nodes": sorted(keep, key=lambda n: n["id"]),
            "adjacency": sorted([list(p) for p in adj if p[0] in keep_ids and p[1] in keep_ids]),
            "background": bg}


def summarize_diff(before: Frame, after: Frame, max_items: int = 6) -> dict:
    """Compact description of what changed between two frames."""
    changed = [(r, c) for r in range(len(before.grid)) for c in range(len(before.grid[0]))
               if before.grid[r][c] != after.grid[r][c]]
    out: dict[str, Any] = {"changed_cells": len(changed)}
    if not changed:
        return out
    rs = [r for r, _ in changed]; cs = [c for _, c in changed]
    out["bbox"] = (min(rs), min(cs), max(rs), max(cs))
    bs = {n["hash"]: n for n in before.segmentation["nodes"]}
    as_ = {n["hash"]: n for n in after.segmentation["nodes"]}
    moved, appeared, gone = [], [], []
    for hsh, n in as_.items():
        if hsh in bs:
            b = bs[hsh]
            if b["center"] != n["center"]:
                moved.append({"color": n["color"], "pixels": n["pixels"], "from": b["center"], "to": n["center"]})
        else:
            appeared.append({"color": n["color"], "pixels": n["pixels"], "center": n["center"]})
    for hsh, b in bs.items():
        if hsh not in as_:
            gone.append({"color": b["color"], "pixels": b["pixels"], "center": b["center"]})
    out["moved"] = moved[:max_items]; out["appeared"] = appeared[:max_items]; out["disappeared"] = gone[:max_items]
    return out


def masked_ascii(frame: "Frame") -> str:
    """Board text with HUD strips (edge counters/gauges) blanked out, for change and cycle detection."""
    rows = [list(r) for r in frame.ascii.splitlines()]
    for n in frame.segmentation["nodes"]:
        if n.get("hud"):
            r0, c0, r1, c1 = n["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    rows[r][c] = "."
    return "\n".join("".join(r) for r in rows)
