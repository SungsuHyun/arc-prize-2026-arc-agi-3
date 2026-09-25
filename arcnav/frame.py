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


def infer_cell_grid(grid: list[list[int]], min_n: int = 6, max_n: int = 32, tolerance: float = 0.06):
    """Recover the game's logical N x N cell grid from a 64 x 64 nearest-neighbour render.
    Returns (n, cells) with cells[r][c] = the dominant colour of block (r, c), or None if no N fits."""
    h = len(grid)
    best = None
    for n in range(min_n, max_n + 1):
        bounds = [int(i * h / n) for i in range(n + 1)]
        bad = 0; cells = []
        for r in range(n):
            row = []
            for c in range(n):
                vals = Counter(grid[i][j] for i in range(bounds[r], bounds[r + 1]) for j in range(bounds[c], bounds[c + 1]))
                colour, cnt = vals.most_common(1)[0]
                total = sum(vals.values())
                if cnt < total:
                    bad += 1
                row.append(colour)
            cells.append(row)
        if bad <= tolerance * n * n:
            return n, cells
    return None


def cell_bounds(n: int, size: int = 64) -> list[int]:
    return [int(i * size / n) for i in range(n + 1)]


def cell_lattice(grid: list[list[int]], hud_rows: set | None = None, hud_cols: set | None = None, min_votes: int = 2):
    """Recover the cell lattice of a rendered game grid from colour-change positions.
    Returns (row_bounds, col_bounds): boundary indices (a cell spans [b[k], b[k+1])). Works for any scaling."""
    h, w = len(grid), len(grid[0])
    hud_rows = hud_rows or set(); hud_cols = hud_cols or set()
    col_votes = [0] * (w + 1); row_votes = [0] * (h + 1)
    for i in range(h):
        if i in hud_rows:
            continue
        for j in range(1, w):
            if grid[i][j] != grid[i][j - 1]:
                col_votes[j] += 1
    for j in range(w):
        if j in hud_cols:
            continue
        for i in range(1, h):
            if grid[i][j] != grid[i - 1][j]:
                row_votes[i] += 1
    col_b = [0] + [j for j in range(1, w) if col_votes[j] >= min_votes] + [w]
    row_b = [0] + [i for i in range(1, h) if row_votes[i] >= min_votes] + [h]
    return row_b, col_b


def to_cells(grid: list[list[int]], row_b: list[int], col_b: list[int]) -> list[list[int]]:
    """Dominant colour per lattice cell."""
    out = []
    for r in range(len(row_b) - 1):
        row = []
        for c in range(len(col_b) - 1):
            vals = Counter(grid[i][j] for i in range(row_b[r], row_b[r + 1]) for j in range(col_b[c], col_b[c + 1]))
            row.append(vals.most_common(1)[0][0])
        out.append(row)
    return out


def hud_lines(frame) -> tuple[set, set]:
    rows, cols = set(), set()
    for n in frame.segmentation["nodes"]:
        if n["hud"]:
            r0, c0, r1, c1 = n["bbox"]
            if r1 - r0 <= 2:
                rows |= set(range(r0, r1 + 1))
            if c1 - c0 <= 2:
                cols |= set(range(c0, c1 + 1))
    return rows, cols


def texture_colors(grid: list[list[int]]) -> set:
    """Colours that occur mostly as isolated 1-px runs (checkerboard textures inside cells)."""
    stats: dict = {}
    for row in grid:
        prev = row[0]; k = 0
        for v in row + [None]:
            if v == prev:
                k += 1
            else:
                s = stats.setdefault(prev, [0, 0]); s[0] += 1; s[1] += (k == 1)
                prev = v; k = 1
    return {c for c, (n, ones) in stats.items() if n >= 8 and ones / n > 0.8}


def find_lattice(grid: list[list[int]], ignore: set | None = None, n_range=(6, 24), origins=(0, 1, 2, 3, 4)):
    """Find (n, origin, bounds) such that colour-change positions (ignoring texture colours) fall on the lattice
    round(origin + k*(64-2*origin)/n). Returns (coverage, n, origin, bounds) or None."""
    ignore = ignore or set()
    h, w = len(grid), len(grid[0])
    col_changes: Counter = Counter(); row_changes: Counter = Counter()
    for i in range(h):
        for j in range(1, w):
            if grid[i][j] != grid[i][j - 1] and grid[i][j] not in ignore and grid[i][j - 1] not in ignore:
                col_changes[j] += 1
    for j in range(w):
        for i in range(1, h):
            if grid[i][j] != grid[i - 1][j] and grid[i][j] not in ignore and grid[i - 1][j] not in ignore:
                row_changes[i] += 1
    total = sum(col_changes.values()) + sum(row_changes.values())
    if not total:
        return None
    best = None
    for n in range(n_range[0], n_range[1] + 1):
        for o in origins:
            step = (w - 2 * o) / n
            bounds = [round(o + k * step) for k in range(n + 1)]
            bs = set(bounds)
            hit = sum(v for j, v in col_changes.items() if j in bs) + sum(v for i, v in row_changes.items() if i in bs)
            cov = hit / total
            if best is None or cov > best[0] + 1e-9 or (abs(cov - best[0]) < 1e-9 and n < best[1]):
                best = (cov, n, o, bounds)
    return best


def lattice_cells(grid: list[list[int]], bounds: list[int]) -> list[list[Counter]]:
    """Colour histogram per lattice cell (cells outside the lattice margin are ignored)."""
    out = []
    for r in range(len(bounds) - 1):
        row = []
        for c in range(len(bounds) - 1):
            row.append(Counter(grid[i][j] for i in range(bounds[r], bounds[r + 1]) for j in range(bounds[c], bounds[c + 1])))
        out.append(row)
    return out
