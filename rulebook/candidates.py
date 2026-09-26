"""Candidate actions for one decision: every valid key, a run of a key, clicks on distinct objects, walks to targets.
Each candidate gets its deterministic prediction (per first step, and for walks the simulated end position)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from arcnav.frame import masked_ascii

from .env import Game, MOVE_KEYS, INTERACT_KEYS
from .predict import Evidence, Prediction


@dataclass
class Candidate:
    label: str
    kind: str                     # key | key-run | interact | click | goto | frontier
    actions: list                 # engine actions (str or dict)
    prediction: str
    color: Optional[int] = None
    tested: bool = False          # tried before in this board state (masked) on this level
    priority: int = 50            # fallback policy order (lower first)
    pred_kind: str = "unknown"    # predictor kind of the first step

    def line(self) -> str:
        return f"{self.label:<26} -> {self.prediction}" + ("  [tried here]" if self.tested else "")


def state_key(game: Game, ev: Optional[Evidence] = None) -> str:
    """Level + world inventory (non-HUD objects away from the edge, avatar excluded) + avatar position. Edge counters/gauges are
    ignored so that an action that only ticks a counter does not make every candidate look new again."""
    skip = set(ev.avatar["colors"]) if ev and ev.avatar else set()
    items = []
    for n in game.frame.segmentation["nodes"]:
        if n["hud"] or n["color"] in skip:
            continue
        r0, c0, r1, c1 = n["bbox"]
        if (r1 - r0 <= 2 or c1 - c0 <= 2) and (r0 <= 2 or r1 >= 61 or c0 <= 2 or c1 >= 61):
            continue
        items.append((n["color"], n["pixels"], r0, c0, r1, c1))
    av = tuple(ev.avatar["bbox_xyxy"]) if ev and ev.avatar else ()
    return f"L{game.level}:{hash((tuple(sorted(items)), av))}"


def build(game: Game, ev: Evidence, tried: set, *, max_clicks: int = 24) -> list[Candidate]:
    valid = list(game.valid_actions); out: list[Candidate] = []
    st = state_key(game, ev)
    keys = [a for a in MOVE_KEYS if a in valid]; inter = [a for a in INTERACT_KEYS if a in valid]
    nav = ev.nav
    movement = bool(nav and ev.moves and ev.avatar)
    # walks to targets (movement games with a known avatar)
    if movement:
        tg = [t for t in nav.targets(max_n=14) if t.get("path_len") is not None]
        tg.sort(key=lambda t: (bool(t.get("visited")), t["path_len"]))
        for i, t in enumerate(tg[:8]):
            path = nav.path_to(t["row"], t["col"])
            if path is None or len(path) > 30:
                continue
            if not path:
                continue
            lab = f"goto({t['color']}@{t['row']},{t['col']})"
            pred = _walk_prediction(ev, path, t)
            out.append(Candidate(lab, "goto", list(path), pred, color=t["color"], priority=10 + i + (30 if t.get("visited") else 0), pred_kind="move"))
        fr = nav.frontier()
        if fr is not None:
            path = nav.path_to(fr[0], fr[1])
            if path:
                out.append(Candidate(f"frontier({fr[0]},{fr[1]})", "frontier", list(path), f"walk {len(path)} steps into unexplored floor at ({fr[0]},{fr[1]})", priority=35))
    for a in keys:
        p = ev.predict(a)
        out.append(Candidate(a, "key", [a], p.text, priority=40 if p.kind == "unknown" else 45, pred_kind=p.kind))
    if movement:
        for a in keys:
            p = ev.predict(a)
            if p.kind == "move":
                out.append(Candidate(f"{a}x8", "key-run", [a] * 8, f"repeat {a} until blocked (up to 8): first step {p.text}", priority=48))
    for a in inter:
        p = ev.predict(a)
        out.append(Candidate(a, "interact", [a], p.text, priority=20 if p.kind == "unknown" else 42, pred_kind=p.kind))
    if "MOUSE" in valid and game.frame is not None:
        nodes = []
        for n in game.frame.segmentation["nodes"]:
            if n["hud"]:
                continue
            r0, c0, r1, c1 = n["bbox"]
            thin = (r1 - r0 <= 2) or (c1 - c0 <= 2)
            if thin and (r0 <= 1 or r1 >= 62 or c0 <= 1 or c1 >= 62):
                continue
            if n["pixels"] <= 2 and keys:
                continue
            nodes.append(n)
        nodes.sort(key=lambda n: n["pixels"])
        # round-robin over colours (smallest object of every colour first) so that no colour is crowded out by a big grid of tiles
        by_colour: dict[int, list] = {}
        for n in nodes:
            by_colour.setdefault(n["color"], []).append(n)
        ordered = []
        for k in range(max((len(v) for v in by_colour.values()), default=0)):
            for c in sorted(by_colour, key=lambda c: by_colour[c][0]["pixels"]):
                if k < len(by_colour[c]) and k < 8:
                    ordered.append(by_colour[c][k])
        nodes = ordered
        seen_cells: set = set(); seen_colors: Counter = Counter()
        base = 25 if not movement else 60
        for n in nodes:
            cell = (n["center"][0] // 4, n["center"][1] // 4)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            act = {"action": "MOUSE", "row": n["center"][0], "col": n["center"][1]}
            p = ev.predict(act)
            lab = f"click({n['color']}@{n['center'][0]},{n['center'][1]})"
            out.append(Candidate(lab, "click", [act], f"[{n['pixels']}px] {p.text}", color=n["color"],
                                 priority=base + (0 if seen_colors[n["color"]] == 0 else 8) + min(seen_colors[n["color"]], 6) + (10 if p.kind in ("noop", "hud") else 0), pred_kind=p.kind))
            seen_colors[n["color"]] += 1
            if len([c for c in out if c.kind == "click"]) >= max_clicks:
                break
    for c in out:
        c.tested = (st, c.label) in tried
    out.sort(key=lambda c: (c.tested, c.priority))
    return out


def _walk_prediction(ev: Evidence, path: list, t: dict) -> str:
    av = ev.avatar; x0, y0, x1, y1 = av["bbox_xyxy"]
    for a in path:
        dx, dy = ev.moves.get(a, (0, 0)); x0 += dx; y0 += dy
    gauge = f"; gauge {ev.gauge['per_action'] * len(path):+}" if ev.gauge else ""
    return f"walk {len(path)} steps ({''.join(a[0] for a in path)}) to colour {t['color']} object at ({t['row']},{t['col']}); avatar ends near ({y0},{x0}){gauge}"


def resolve(label: str, cands: list, game: Game, ev: Evidence) -> Optional[Candidate]:
    """Map a model-written label to a candidate: exact label first; then click(c@r,c) / goto(c@r,c) with coordinates the
    model read off the board (any cell of that colour is accepted as an ad-hoc click; goto goes to the nearest listed target of
    that colour); bare key names."""
    import re
    label = label.strip()
    by = {c.label: c for c in cands}
    if label in by:
        return by[label]
    m = re.match(r"(click|goto)\((\d+)@(\d+),(\d+)\)", label)
    if m:
        kind, col, r, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        same = [x for x in cands if x.kind == kind and x.color == col]
        if kind == "click" and 0 <= r < 64 and 0 <= c < 64 and game.frame.grid[r][c] == col and "MOUSE" in game.valid_actions:
            act = {"action": "MOUSE", "row": r, "col": c}; p = ev.predict(act)
            return Candidate(f"click({col}@{r},{c})", "click", [act], p.text, color=col, pred_kind=p.kind)
        if same:
            return min(same, key=lambda x: abs(_coord(x.label)[0] - r) + abs(_coord(x.label)[1] - c))
        return None
    up = label.upper()
    return by.get(up) or next((x for x in cands if x.label.upper() == up), None)


def _coord(label: str) -> tuple:
    import re
    m = re.search(r"@(\d+),(\d+)", label)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
