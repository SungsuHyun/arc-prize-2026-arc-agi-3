"""Rule-based planners that run on confirmed rules instead of the model's arithmetic.

two_body_merge: joint-state BFS for games where a second body moves with the avatar under a fixed
transform (mirror), walls stop each body separately, hazards reset, and the goal is to bring both
bodies onto the same cell (or adjacent-swap). Cells are the movement lattice (cell size from nav).
"""
from __future__ import annotations

from collections import deque
from typing import Optional


def _cell_color(grid, r, c):
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
        return grid[r][c]
    return None


def _body_blocked(grid, r, c, size, wall_colors, body_colors):
    """Would a size x size body with top-left (r, c) overlap a wall colour or leave the grid?"""
    h, w = len(grid), len(grid[0])
    if r < 0 or c < 0 or r + size > h or c + size > w:
        return True
    for i in range(r, r + size):
        for j in range(c, c + size):
            v = grid[i][j]
            if v in wall_colors and v not in body_colors:
                return True
    return False


def _touches(grid, r, c, size, colors):
    for i in range(r, r + size):
        for j in range(c, c + size):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] in colors:
                return True
    return False


def two_body_merge(grid, body_a, body_b, moves: dict, transform: str, wall_colors, hazard_colors=(), body_colors=(),
                   max_nodes: int = 200000) -> Optional[list[str]]:
    """body_a/body_b: (top_row, left_col, size). moves: action -> (dx, dy) in pixels. transform: mirror_x|mirror_y|mirror_xy|same.
    Returns the action list that brings both bodies to the same top-left, avoiding hazards, or None."""
    (ra, ca, sa), (rb, cb, sb) = body_a, body_b
    walls = set(wall_colors); hazards = set(hazard_colors); bcol = set(body_colors)
    tf = {"same": (1, 1), "mirror_x": (-1, 1), "mirror_y": (1, -1), "mirror_xy": (-1, -1)}[transform]
    start = (ra, ca, rb, cb)
    prev = {start: None}; q = deque([start]); n = 0
    while q and n < max_nodes:
        s = q.popleft(); n += 1
        r1, c1, r2, c2 = s
        if (r1, c1) == (r2, c2):
            path = []
            while prev[s] is not None:
                s, a = prev[s]; path.append(a)
            return path[::-1]
        for a, (dx, dy) in moves.items():
            n1 = (r1 + dy, c1 + dx); n2 = (r2 + tf[1] * dy, c2 + tf[0] * dx)
            if _body_blocked(grid, n1[0], n1[1], sa, walls, bcol):
                n1 = (r1, c1)
            if _body_blocked(grid, n2[0], n2[1], sb, walls, bcol):
                n2 = (r2, c2)
            if n1 == (r1, c1) and n2 == (r2, c2):
                continue
            if hazards and (_touches(grid, n1[0], n1[1], sa, hazards) or _touches(grid, n2[0], n2[1], sb, hazards)):
                continue
            # swap across adjacent cells also counts as a merge
            if n1 == (r2, c2) and n2 == (r1, c1):
                path = [a]
                t = s
                while prev[t] is not None:
                    t, b = prev[t]; path.append(b)
                return path[::-1]
            ns = (n1[0], n1[1], n2[0], n2[1])
            if ns not in prev:
                prev[ns] = (s, a); q.append(ns)
    return None


# ---------------------------------------------------------------- cell-space version (lattice from the body itself)
def lattice_from_body(bbox_xyxy, size: int = 64):
    """A one-cell body at pixel bbox (x0, y0, x1, y1) fixes the lattice: step = body width, origin = x0 mod step."""
    x0, y0, x1, y1 = bbox_xyxy
    step = max(x1 - x0 + 1, y1 - y0 + 1)
    ox, oy = x0 % step, y0 % step
    cols = [ox + k * step for k in range(0, (size - ox) // step + 1)]
    rows = [oy + k * step for k in range(0, (size - oy) // step + 1)]
    return rows, cols


def cell_of(bbox_xyxy, rows, cols):
    x0, y0, x1, y1 = bbox_xyxy
    r = max(i for i, b in enumerate(rows) if b <= y0) if any(b <= y0 for b in rows) else 0
    c = max(i for i, b in enumerate(cols) if b <= x0) if any(b <= x0 for b in cols) else 0
    return r, c


def cell_colors(grid, rows, cols):
    """Per cell: (dominant colour, set of colours)."""
    out = []
    for r in range(len(rows) - 1):
        row = []
        for c in range(len(cols) - 1):
            vals = {}
            for i in range(rows[r], min(rows[r + 1], len(grid))):
                for j in range(cols[c], min(cols[c + 1], len(grid[0]))):
                    vals[grid[i][j]] = vals.get(grid[i][j], 0) + 1
            dom = max(vals, key=vals.get) if vals else None
            row.append((dom, set(vals)))
        out.append(row)
    return out


def two_body_merge_cells(cells, a, b, transform: str, passable, hazard_colors=(), max_nodes: int = 200000, forbidden=()) -> Optional[list[str]]:
    """cells[r][c] = (dominant, colour set). a, b = (row, col) cell coords. Moves are one cell.
    A move is blocked for a body if the target cell's dominant colour is not passable (or off-grid).
    Hazard cells (containing a hazard colour) are never entered."""
    H, W = len(cells), len(cells[0])
    MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    tf = {"same": (1, 1), "mirror_x": (1, -1), "mirror_y": (-1, 1), "mirror_xy": (-1, -1)}[transform]   # (row sign, col sign)
    hz = set(hazard_colors); fb = set(forbidden)

    def step(pos, dr, dc):
        r, c = pos[0] + dr, pos[1] + dc
        if not (0 <= r < H and 0 <= c < W):
            return pos
        dom, cols = cells[r][c]
        if cols & hz:            # hazard-textured cells are enterable (and deadly), never walls
            return (r, c)
        if dom not in passable:
            return pos
        return (r, c)

    start = (a, b); prev = {start: None}; q = deque([start]); n = 0
    while q and n < max_nodes:
        s = q.popleft(); n += 1
        pa, pb = s
        if pa == pb:
            path = []
            while prev[s] is not None:
                s, act = prev[s]; path.append(act)
            return path[::-1]
        for act, (dr, dc) in MOVES.items():
            na = step(pa, dr, dc); nb = step(pb, tf[0] * dr, tf[1] * dc)
            if na == pa and nb == pb:
                continue
            if hz and (cells[na[0]][na[1]][1] & hz or cells[nb[0]][nb[1]][1] & hz):
                continue
            if fb and (na in fb or nb in fb):   # cells that caused a reset before
                continue
            if na == pb and nb == pa:   # swap counts as a merge
                path = [act]; t = s
                while prev[t] is not None:
                    t, x = prev[t]; path.append(x)
                return path[::-1]
            ns = (na, nb)
            if ns not in prev:
                prev[ns] = (s, act); q.append(ns)
    return None
