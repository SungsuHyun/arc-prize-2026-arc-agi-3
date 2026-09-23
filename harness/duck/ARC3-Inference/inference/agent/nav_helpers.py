"""Navigation helper exposed to the duck's python tool as ``nav``.

Stateless: rebuilt on every refresh from ``transitions`` (action, before/after
frames) so it survives the ephemeral sandbox. Ported from our v006–v011 agents
(avatar detection from co-moving objects, cell-size estimate, coarse map,
floor/wall learning from actual avatar-block moves, BFS path, targets,
frontier, action-gauge detection). Coordinates are (row, col) like MOUSE.

Typical use inside the tool:
    print(nav.summary())                      # avatar, moves, targets, gauge
    seq = nav.path_to(row, col)               # list of action names or None
    if seq: action(seq)
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Optional

HEX = "0123456789abcdef"
_BIG = 400          # objects larger than this are never the avatar
_MAX_HISTORY = 300  # transitions considered (recent)


def _grid_of(frame: Any) -> Optional[list[list[int]]]:
    if frame is None:
        return None
    g = getattr(frame, "_grid", None)
    if g is None and isinstance(frame, dict):
        g = frame.get("grid")
    if g is None:
        g = getattr(frame, "grid", None)
    if not g:
        return None
    return [list(map(int, r)) for r in g]


def extract_objects(grid: list[list[int]], max_objects: int = 40) -> list[dict]:
    h, w = len(grid), len(grid[0])
    background = Counter(c for row in grid for c in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    objects = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy][sx] or grid[sy][sx] == background:
                continue
            color = grid[sy][sx]
            stack, cells = [(sy, sx)], []
            seen[sy][sx] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            objects.append({"color": color, "size": len(cells),
                            "bbox": (min(xs), min(ys), max(xs), max(ys)),          # x0,y0,x1,y1
                            "center": (sum(xs) // len(xs), sum(ys) // len(ys))})    # x,y
    objects.sort(key=lambda o: -o["size"])
    return objects[:max_objects], background


def _moved(prev: list[dict], cur: list[dict]) -> list[tuple[dict, int, int]]:
    """(new object, dx, dy) for objects with identical color+size that shifted."""
    pool = list(prev)
    out = []
    for c in cur:
        best, best_d = None, None
        for i, p in enumerate(pool):
            if p is None or p["color"] != c["color"] or p["size"] != c["size"]:
                continue
            d = abs(p["center"][0] - c["center"][0]) + abs(p["center"][1] - c["center"][1])
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None:
            p = pool[best]; pool[best] = None
            if best_d:
                out.append((c, c["center"][0] - p["center"][0], c["center"][1] - p["center"][1]))
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


class NavHelper:
    def __init__(self, transitions: list, current_frame: Any):
        self.grid = _grid_of(current_frame)
        self.objects, self.background = extract_objects(self.grid) if self.grid else ([], None)
        self.move_log: dict[str, list[tuple[int, int]]] = {}
        self.player_votes: Counter = Counter()
        self.player_parts: set[tuple[int, int]] = set()
        self.gauge_hist: dict[int, list[int]] = {}
        self._steps: list[tuple[str, list[list[int]], list[list[int]]]] = []
        for t in (transitions or [])[-_MAX_HISTORY:]:
            b, a = _grid_of(getattr(t, "before_frame", None)), _grid_of(getattr(t, "after_frame", None))
            name = str(getattr(t, "action", "") or "").strip().upper()
            if b is None or a is None or not name:
                continue
            self._steps.append((name, b, a))
        self._learn()
        self.md = self._map_data()
        self.floor_votes: Counter = Counter()
        self.walls: set[tuple[int, int]] = set()
        self.visited: set[tuple[int, int]] = set()
        self._learn_terrain()

    # ── learning ───────────────────────────────────────────────────────────
    def _learn(self) -> None:
        prev_objs = None
        for name, b, a in self._steps:
            po, _ = extract_objects(b); co, _ = extract_objects(a)
            moved = [m for m in _moved(po, co) if m[0]["size"] <= _BIG]
            if moved and not name.startswith("MOUSE"):
                c, dx, dy = max(moved, key=lambda m: (abs(m[1]) + abs(m[2]), m[0]["size"]))
                self.move_log.setdefault(name, []).append((dx, dy))
                self.player_votes[(c["color"], c["size"])] += 1
                self.player_parts = {(m[0]["color"], m[0]["size"]) for m in moved if (m[1], m[2]) == (dx, dy)}
            # gauge: same-color object whose size changed a little near the same spot
            for o in co:
                for p in po:
                    if p["color"] == o["color"] and p["size"] != o["size"] and \
                       abs(p["center"][0] - o["center"][0]) + abs(p["center"][1] - o["center"][1]) <= 3 and \
                       o["size"] > 8:
                        h = self.gauge_hist.setdefault(o["color"], [])
                        if not h:
                            h.append(p["size"])
                        h.append(o["size"])
                        break

    def move_delta(self, name: str) -> Optional[tuple[int, int]]:
        log = self.move_log.get(name.upper()) or []
        if not log:
            return None
        if len(log) == 1:
            d = log[0]
            return d if abs(d[0]) + abs(d[1]) <= 8 else None
        (delta, n), = Counter(log).most_common(1)
        return delta if n >= 2 and n * 2 > len(log) else None

    @property
    def moves(self) -> dict[str, tuple[int, int]]:
        """action name -> (dx, dy) in pixels (x=col, y=row)."""
        out = {}
        for a in self.move_log:
            d = self.move_delta(a)
            if d:
                out[a] = d
        return out

    def avatar(self) -> Optional[dict]:
        main = None
        for (color, size), _ in self.player_votes.most_common(3):
            for o in self.objects:
                if o["color"] == color and o["size"] == size:
                    main = o; break
            if main:
                break
        if main is None:
            return None
        x0, y0, x1, y1 = main["bbox"]; size = main["size"]; colors = {main["color"]}
        for o in self.objects:
            if o is main or (o["color"], o["size"]) not in self.player_parts:
                continue
            bx0, by0, bx1, by1 = o["bbox"]
            if bx0 > x1 + 1 or bx1 < x0 - 1 or by0 > y1 + 1 or by1 < y0 - 1:
                continue
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
            size += o["size"]; colors.add(o["color"])
        return {"colors": sorted(colors), "size": size, "bbox_xyxy": (x0, y0, x1, y1),
                "row": (y0 + y1) // 2, "col": (x0 + x1) // 2}

    def cell_size(self) -> int:
        g = 0
        for d in self.moves.values():
            for v in (abs(d[0]), abs(d[1])):
                if v:
                    g = _gcd(g, v)
        return g if 2 <= g <= 8 else 4

    # ── coarse map ─────────────────────────────────────────────────────────
    def _map_data(self) -> Optional[dict]:
        if not self.grid:
            return None
        cell = self.cell_size(); av = self.avatar()
        ox = oy = 0
        if av:
            ox, oy = av["bbox_xyxy"][0] % cell, av["bbox_xyxy"][1] % cell
        colors, y = [], oy
        while y < 64 and len(colors) < 16:
            row, x = [], ox
            while x < 64 and len(row) < 16:
                block = [self.grid[yy][xx] for yy in range(y, min(64, y + cell)) for xx in range(x, min(64, x + cell))]
                row.append(Counter(block).most_common(1)[0][0]); x += cell
            colors.append(row); y += cell
        pb = ((av["bbox_xyxy"][0] - ox) // cell, (av["bbox_xyxy"][1] - oy) // cell) if av else None
        return {"cell": cell, "ox": ox, "oy": oy, "colors": colors, "player": pb,
                "ncols": len(colors[0]), "nrows": len(colors)}

    def block_of(self, row: int, col: int) -> tuple[int, int]:
        md = self.md
        return (max(0, min(md["ncols"] - 1, (col - md["ox"]) // md["cell"])),
                max(0, min(md["nrows"] - 1, (row - md["oy"]) // md["cell"])))

    def block_moves(self) -> dict[str, tuple[int, int]]:
        md = self.md; out = {}
        for a, (dx, dy) in self.moves.items():
            if dx % md["cell"] == 0 and dy % md["cell"] == 0:
                out[a] = (dx // md["cell"], dy // md["cell"])
        return out

    def _learn_terrain(self) -> None:
        """Replay the history through the current map geometry: blocks the avatar
        actually moved from are floor; moves that did not move it mark walls."""
        if not self.md:
            return
        md = self.md; bm = self.block_moves()
        if not bm:
            return
        prev_block = None
        sig = None
        for name, b, a in self._steps:
            objs_a, _ = extract_objects(a)
            # locate avatar in the after-frame by main color/size votes
            av = None
            for (color, size), _ in self.player_votes.most_common(3):
                for o in objs_a:
                    if o["color"] == color and o["size"] == size:
                        av = o; break
                if av:
                    break
            blk = ((av["bbox"][0] - md["ox"]) // md["cell"], (av["bbox"][1] - md["oy"]) // md["cell"]) if av else None
            if blk is not None:
                self.visited.add(blk)
            mv = bm.get(name)
            if prev_block is not None and blk is not None and mv is not None:
                if blk == prev_block:
                    self.walls.add((prev_block[0] + mv[0], prev_block[1] + mv[1]))
                elif blk == (prev_block[0] + mv[0], prev_block[1] + mv[1]):
                    c, r = prev_block
                    if 0 <= r < md["nrows"] and 0 <= c < md["ncols"]:
                        # colour of the block we just left, in the after-frame
                        cell = md["cell"]
                        x0, y0 = md["ox"] + c * cell, md["oy"] + r * cell
                        block = [a[yy][xx] for yy in range(y0, min(64, y0 + cell)) for xx in range(x0, min(64, x0 + cell))]
                        self.floor_votes[Counter(block).most_common(1)[0][0]] += 1
            prev_block = blk
        if self.md["player"] is not None:
            self.visited.add(self.md["player"])

    @property
    def floor_colors(self) -> set[int]:
        strong = {c for c, n in self.floor_votes.items() if n >= 2}
        if strong:
            return strong
        return {self.floor_votes.most_common(1)[0][0]} if self.floor_votes else set()

    def walkable(self, b: tuple[int, int]) -> bool:
        md = self.md; c, r = b
        if not (0 <= r < md["nrows"] and 0 <= c < md["ncols"]) or b in self.walls:
            return False
        return md["colors"][r][c] in self.floor_colors or b == md["player"]

    # ── planning ───────────────────────────────────────────────────────────
    def path_to(self, row: int, col: int, max_len: int = 40) -> Optional[list[str]]:
        """BFS over known-walkable blocks to the block containing pixel (row, col).
        Returns action names (e.g. ['LEFT','LEFT','UP']) or None if unreachable
        / navigation not learned yet. The final step may enter any colour."""
        if not self.md or self.md["player"] is None:
            return None
        bm = self.block_moves()
        if not bm:
            return None
        start, target = self.md["player"], self.block_of(row, col)
        if start == target:
            return []
        if target in self.walls:
            return None
        prev = {start: (start, "")}
        q = deque([start])
        while q:
            cur = q.popleft()
            for a, (dc, dr) in bm.items():
                nb = (cur[0] + dc, cur[1] + dr)
                if nb in prev:
                    continue
                if nb == target or self.walkable(nb):
                    prev[nb] = (cur, a)
                    if nb == target:
                        q.clear(); break
                    q.append(nb)
        if target not in prev:
            return None
        acts, b = [], target
        while b != start:
            b, a = prev[b]; acts.append(a)
        acts.reverse()
        return acts[:max_len]

    def targets(self, max_n: int = 10) -> list[dict]:
        """Non-floor, non-background small objects with (row, col) and path length."""
        if not self.md or self.md["player"] is None:
            return []
        av = self.avatar(); pb = av["bbox_xyxy"] if av else None
        out, seen = [], set()
        for o in self.objects:
            if o["size"] > 120 or o["color"] == self.background or o["color"] in self.floor_colors:
                continue
            x0, y0, x1, y1 = o["bbox"]
            if pb and not (x0 > pb[2] or x1 < pb[0] or y0 > pb[3] or y1 < pb[1]):
                continue
            b = self.block_of(o["center"][1], o["center"][0])
            if b in seen or b == self.md["player"]:
                continue
            seen.add(b)
            p = self.path_to(o["center"][1], o["center"][0])
            out.append({"color": o["color"], "size": o["size"], "row": o["center"][1], "col": o["center"][0],
                        "path_len": None if p is None else len(p), "visited": b in self.visited or b in self.walls})
        out.sort(key=lambda t: (t["path_len"] is None, t["path_len"] or 0))
        return out[:max_n]

    def frontier(self) -> Optional[tuple[int, int]]:
        """Nearest unvisited walkable block as (row, col) pixel centre, else an
        unknown-colour block adjacent to the reachable area."""
        if not self.md or self.md["player"] is None:
            return None
        bm = self.block_moves()
        if not bm:
            return None
        md = self.md; start = md["player"]
        seen, q, unknown = {start}, deque([start]), []
        while q:
            cur = q.popleft()
            for dc, dr in bm.values():
                nb = (cur[0] + dc, cur[1] + dr)
                if nb in seen:
                    continue
                seen.add(nb)
                if self.walkable(nb):
                    if nb not in self.visited:
                        return self._block_center(nb)
                    q.append(nb)
                elif nb not in self.walls and 0 <= nb[1] < md["nrows"] and 0 <= nb[0] < md["ncols"]:
                    unknown.append(nb)
        return self._block_center(unknown[0]) if unknown else None

    def _block_center(self, b: tuple[int, int]) -> tuple[int, int]:
        md = self.md; c, r = b
        return (md["oy"] + r * md["cell"] + md["cell"] // 2, md["ox"] + c * md["cell"] + md["cell"] // 2)

    def gauge(self) -> Optional[dict]:
        for color, hist in self.gauge_hist.items():
            if len(hist) < 3:
                continue
            steps = [b - a for a, b in zip(hist, hist[1:])]
            s = steps[-1]; k = 0
            while k < len(steps) and steps[-1 - k] == s:
                k += 1
            if s != 0 and k >= 2:
                return {"color": color, "size": hist[-1], "per_action": s,
                        "actions_left": hist[-1] // abs(s) if s < 0 else None}
        return None

    # ── text ───────────────────────────────────────────────────────────────
    def map(self) -> str:
        if not self.md:
            return ""
        md = self.md; pb = md["player"]
        av = self.avatar(); pw = ph = 1
        if av:
            pw = max(1, (av["bbox_xyxy"][2] - av["bbox_xyxy"][0] + md["cell"]) // md["cell"])
            ph = max(1, (av["bbox_xyxy"][3] - av["bbox_xyxy"][1] + md["cell"]) // md["cell"])
        rows = []
        for r, row in enumerate(md["colors"]):
            line = []
            for c, col in enumerate(row):
                if pb and pb[0] <= c < pb[0] + pw and pb[1] <= r < pb[1] + ph:
                    line.append("P")
                elif (c, r) in self.walls:
                    line.append("#")
                else:
                    line.append(HEX[col])
            rows.append(f"{r:2d} {''.join(line)}")
        return (f"cell={md['cell']}px origin(row={md['oy']},col={md['ox']}); char=(col,row) block majority colour hex, "
                f"P=avatar, #=known wall; pixel row=r*{md['cell']}+{md['oy']}, col=c*{md['cell']}+{md['ox']}\n" + "\n".join(rows))

    def summary(self) -> str:
        av = self.avatar(); mv = self.moves
        lines = []
        if av:
            lines.append(f"avatar: colours {av['colors']} size {av['size']} at row {av['row']}, col {av['col']}")
        else:
            lines.append("avatar: not identified yet (take each movement action once or twice)")
        if mv:
            lines.append("moves (dx=cols,dy=rows): " + ", ".join(f"{a}=({dx:+d},{dy:+d})" for a, (dx, dy) in mv.items()))
            lines.append(f"floor colours {sorted(self.floor_colors) or '?'}, walls known {len(self.walls)}, blocks visited {len(self.visited)}")
        g = self.gauge()
        if g:
            lines.append(f"gauge: colour {g['color']} {g['size']} cells, {g['per_action']:+d}/action"
                         + (f", ~{g['actions_left']} actions left" if g["actions_left"] is not None else ""))
        ts = self.targets()
        if ts:
            lines.append("targets (row,col colour size path visited):")
            for t in ts:
                lines.append(f"  ({t['row']},{t['col']}) c{t['color']} s{t['size']} "
                             + (f"{t['path_len']} moves" if t["path_len"] is not None else "unreachable")
                             + (" visited" if t["visited"] else ""))
        fr = self.frontier()
        if fr:
            lines.append(f"frontier: nearest unexplored block at row {fr[0]}, col {fr[1]}")
        return "\n".join(lines)


def build_nav(transitions: list, current_frame: Any) -> Optional[NavHelper]:
    try:
        return NavHelper(transitions, current_frame)
    except Exception:
        return None
