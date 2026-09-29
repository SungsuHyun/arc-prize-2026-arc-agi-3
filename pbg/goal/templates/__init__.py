"""goal/templates — the initial goal template library (spec §9). Each template instantiates GoalInstances on a scene:
is_goal(scene) -> bool, progress(scene) -> [0,1], plus a clue bonus from scene structure."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ...core.types import Action, Scene


@dataclass
class GoalInstance:
    name: str                     # e.g. "reach(agent,target)"
    template: str
    params: dict
    is_goal_fn: Callable[[Scene], bool]
    progress_fn: Callable[[Scene], float]
    confidence: float = 0.5
    clue: float = 0.0
    origin: str = "template"      # template | llm
    code: str = ""
    progress_weight: float = 1.0  # lowered to 0.5 when progress is not monotone during execution
    history: list = field(default_factory=list)
    estimate_fn: Optional[Callable[[Scene], Optional[float]]] = None   # remaining actions estimate (A* heuristic), if the template can compute one

    def estimate(self, scene: Scene) -> Optional[float]:
        if self.estimate_fn is None:
            return None
        try:
            return self.estimate_fn(scene)
        except Exception:
            return None

    def is_goal(self, scene: Scene) -> bool:
        try:
            return bool(self.is_goal_fn(scene))
        except Exception:
            return False

    def progress(self, scene: Scene) -> float:
        try:
            return max(0.0, min(1.0, float(self.progress_fn(scene))))
        except Exception:
            return 0.0

    def to_json(self) -> dict:
        return {"name": self.name, "template": self.template, "params": self.params, "confidence": round(self.confidence, 3), "clue": self.clue, "origin": self.origin}


# ── helpers ──
def _dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _max_dist(scene: Scene) -> int:
    return scene.grid_shape[0] + scene.grid_shape[1]


def _roles(scene: Scene) -> set[str]:
    return {o.role for o in scene.objects if o.role and o.role != "unknown"}


def _agents(s: Scene) -> list:
    """Objects whose role is 'agent' or a custom variant containing 'agent' (agent_left, custom:agent2 ...)."""
    return [o for o in s.objects if o.role and "agent" in o.role]


def _shape_multiset(scene: Scene, region: str) -> Counter:
    return Counter((o.color, o.shape_sig) for o in scene.in_region(region))


# ── templates ──
def _targets(s: Scene, b: str) -> list:
    """Objects of role b, or of colour c when b == 'color:c' (unknown-role objects grouped by colour)."""
    if b.startswith("color:"):
        c = int(b[6:])
        strips = {r.id for r in s.regions if r.kind_hint == "ui_strip"}
        return [o for o in s.objects if o.color == c and (o.role in (None, "unknown")) and o.region not in strips]
    return s.by_role(b)


def t_reach(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    roles = _roles(scene)
    if not _agents(scene):
        return out
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    targets = sorted(r for r in roles - {"wall", "decoration", "indicator"} if "agent" not in r)
    targets += sorted({f"color:{o.color}" for o in scene.objects if o.role in (None, "unknown") and o.region not in strips})
    a = "agent"
    for b in targets:
        def is_goal(s, b=b):
            return any(x.overlaps(y) or _adjacent(x, y) for x in _agents(s) for y in _targets(s, b))
        def progress(s, b=b):
            xs, ys = _agents(s), _targets(s, b)
            if not xs or not ys:
                return 0.0
            d = min(_dist(x.center, y.center) for x in xs for y in ys)
            return 1.0 - d / _max_dist(s)
        conf_penalty = 0.15 if b.startswith("color:") else 0.0
        out.append(GoalInstance(f"reach({a},{b})", "reach", {"a": a, "b": b}, is_goal, progress, clue=-conf_penalty))
    return out


def t_match_shapes(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    regs = [r for r in scene.regions if r.kind_hint != "ui_strip" and r.id != "R0"]
    if len(regs) < 2:
        return out
    for src in regs:
        for dst in regs:
            if src.id == dst.id:
                continue
            ms_src, ms_dst = _shape_multiset(scene, src.id), _shape_multiset(scene, dst.id)
            if not ms_dst or not ms_src:
                continue
            shared = {k[1] for k in ms_src if k[1] != "empty"} & {k[1] for k in ms_dst}
            clue = 0.2 if len(shared) >= 2 else (0.1 if shared else 0.0)
            def is_goal(s, src=src.id, dst=dst.id):
                a, b = _shape_multiset(s, src), _shape_multiset(s, dst)
                return bool(b) and a == b
            def progress(s, src=src.id, dst=dst.id):
                a, b = _shape_multiset(s, src), _shape_multiset(s, dst)
                if not b:
                    return 0.0
                return sum((a & b).values()) / sum(b.values())
            out.append(GoalInstance(f"match_shapes({src.id},{dst.id})", "match_shapes", {"src": src.id, "dst": dst.id}, is_goal, progress, clue=clue))
    return out


def t_all_removed(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    counts = Counter(o.color for o in scene.objects if o.region not in strips)
    for color, n in counts.items():
        if n == 0:
            continue
        def is_goal(s, color=color):
            return not any(o.color == color for o in s.objects if o.region not in strips)
        def progress(s, color=color, n=n):
            k = sum(1 for o in s.objects if o.color == color and o.region not in strips)
            return 1.0 - min(k, n) / n
        out.append(GoalInstance(f"all_removed({color})", "all_removed", {"color": color}, is_goal, progress))
    return out


def t_all_collected(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    for role in ("collectible", "target", "key"):
        n = len(scene.by_role(role))
        if n == 0:
            continue
        def is_goal(s, role=role):
            return len(s.by_role(role)) == 0
        def progress(s, role=role, n=n):
            return 1.0 - min(len(s.by_role(role)), n) / n
        out.append(GoalInstance(f"all_collected({role})", "all_collected", {"role": role}, is_goal, progress))
    return out


def t_fill_region(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    for reg in scene.regions:
        if reg.kind_hint == "ui_strip" or reg.id == "R0":
            continue
        objs = scene.in_region(reg.id)
        if not objs:
            continue
        for color in {o.color for o in objs}:
            def is_goal(s, rid=reg.id, color=color):
                objs = s.in_region(rid)
                return bool(objs) and all(o.color == color for o in objs) and sum(o.area for o in objs) >= 0.9 * (s.region(rid).area if s.region(rid) else 1)
            def progress(s, rid=reg.id, color=color):
                reg2 = s.region(rid)
                if reg2 is None:
                    return 0.0
                return min(1.0, sum(o.area for o in s.in_region(rid) if o.color == color) / max(1, reg2.area))
            out.append(GoalInstance(f"fill_region({reg.id},{color})", "fill_region", {"region": reg.id, "color": color}, is_goal, progress))
    return out


def t_sort_by(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    objs = [o for o in scene.objects if o.region not in strips]
    if len(objs) < 3:
        return out
    for axis in ("row", "col"):
        for key in ("color", "size"):
            def order(s, axis=axis, key=key):
                os_ = [o for o in s.objects if o.region not in strips]
                os_.sort(key=lambda o: o.center[0] if axis == "row" else o.center[1])
                return [o.color if key == "color" else o.area for o in os_]
            def is_goal(s, order=order):
                v = order(s)
                return len(v) >= 3 and v == sorted(v)
            def progress(s, order=order):
                v = order(s)
                if len(v) < 2:
                    return 0.0
                return sum(1 for i in range(1, len(v)) if v[i] >= v[i - 1]) / (len(v) - 1)
            out.append(GoalInstance(f"sort_by({axis},{key})", "sort_by", {"axis": axis, "key": key}, is_goal, progress))
    return out


def t_count_equals(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    for reg in scene.regions:
        if reg.kind_hint != "ui_strip":
            continue
        n_now = sum(o.area for o in scene.in_region(reg.id))
        for n in (0,):
            def is_goal(s, rid=reg.id, n=n):
                return sum(o.area for o in s.in_region(rid)) == n
            def progress(s, rid=reg.id, n=n, n0=n_now):
                k = sum(o.area for o in s.in_region(rid))
                return 1.0 - abs(k - n) / max(1, abs(n0 - n))
            out.append(GoalInstance(f"count_equals({reg.id},{n})", "count_equals", {"indicator": reg.id, "n": n}, is_goal, progress, confidence=0.2))
    return out


def t_align(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    roles = sorted(_roles(scene) - {"wall", "decoration", "indicator"})
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            for axis in ("row", "col"):
                def is_goal(s, a=a, b=b, axis=axis):
                    xs, ys = s.by_role(a), s.by_role(b)
                    k = 0 if axis == "row" else 1
                    return bool(xs and ys) and all(any(x.center[k] == y.center[k] for y in ys) for x in xs)
                def progress(s, a=a, b=b, axis=axis):
                    xs, ys = s.by_role(a), s.by_role(b)
                    if not xs or not ys:
                        return 0.0
                    k = 0 if axis == "row" else 1
                    d = sum(min(abs(x.center[k] - y.center[k]) for y in ys) for x in xs) / len(xs)
                    return 1.0 - d / s.grid_shape[k]
                out.append(GoalInstance(f"align({a},{b},{axis})", "align", {"a": a, "b": b, "axis": axis}, is_goal, progress))
    return out


def t_enclose(scene: Scene, ctx: dict) -> list[GoalInstance]:
    out = []
    roles = _roles(scene)
    if "target" not in roles:
        return out
    for wall_color in {o.color for o in scene.objects if o.role == "wall"}:
        def is_goal(s, wc=wall_color):
            g = s.render()
            for t in s.by_role("target"):
                r0, c0, r1, c1 = t.bbox
                ring = []
                for r in range(r0 - 1, r1 + 1):
                    for c in (c0 - 1, c1):
                        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                            ring.append(g[r, c])
                for c in range(c0, c1):
                    for r in (r0 - 1, r1):
                        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                            ring.append(g[r, c])
                if not ring or any(v != wc for v in ring):
                    return False
            return True
        def progress(s, wc=wall_color):
            g = s.render(); tot = ok = 0
            for t in s.by_role("target"):
                r0, c0, r1, c1 = t.bbox
                for r in range(r0 - 1, r1 + 1):
                    for c in (c0 - 1, c1):
                        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                            tot += 1; ok += g[r, c] == wc
            return ok / tot if tot else 0.0
        out.append(GoalInstance(f"enclose(target,{wall_color})", "enclose", {"role": "target", "wall_color": wall_color}, is_goal, progress))
    return out


def t_sequence(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Puzzle-type: a specific action sequence (aux 'seq' reaches len(order)); instantiated only with a hint in ctx."""
    order = ctx.get("sequence")
    if not order:
        return []
    def is_goal(s, n=len(order)):
        return int(s.aux.get("seq", 0)) >= n
    def progress(s, n=len(order)):
        return min(1.0, int(s.aux.get("seq", 0)) / n)
    return [GoalInstance(f"sequence({len(order)})", "sequence", {"actions": list(order)}, is_goal, progress)]


def _frames(scene: Scene) -> list:
    """Frame-like objects: hollow (mask has an interior hole) and larger than 3x3."""
    out = []
    for o in scene.objects:
        if o.height >= 3 and o.width >= 3 and o.area < o.height * o.width * 0.8:
            inner = o.mask[1:-1, 1:-1]
            if inner.size and not inner.all():
                out.append(o)
    return out


def t_inside_frame(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Every piece of colour c sits inside a frame (a hollow object whose bbox contains the piece). Pieces = objects of
    a colour that also has at least one frame with a matching colour or a frame of any colour."""
    frames = _frames(scene)
    if not frames:
        return []
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    frame_ids = {f.id for f in frames}
    pieces_by_color: dict = {}
    for o in scene.objects:
        if o.id in frame_ids or o.region in strips or o.area < 2 or o.height >= 12:
            continue
        pieces_by_color.setdefault(o.color, []).append(o)
    out = []
    frame_colors = {f.color for f in frames}
    for color, pieces in pieces_by_color.items():
        if len(pieces) > 16:
            continue
        def inside(p, f):
            return f.bbox[0] <= p.bbox[0] and f.bbox[1] <= p.bbox[1] and p.bbox[2] <= f.bbox[2] and p.bbox[3] <= f.bbox[3]
        def is_goal(s, color=color):
            fs = _frames(s); ps = [o for o in s.objects if o.color == color and o.id not in {f.id for f in fs} and o.height < 12]
            return bool(ps) and all(any(inside(p, f) for f in fs) for p in ps)
        def progress(s, color=color):
            fs = _frames(s); ps = [o for o in s.objects if o.color == color and o.id not in {f.id for f in fs} and o.height < 12]
            if not ps or not fs:
                return 0.0
            done = sum(1 for p in ps if any(inside(p, f) for f in fs))
            near = sum(min(_dist(p.center, f.center) for f in fs) for p in ps if not any(inside(p, f) for f in fs))
            return (done + max(0.0, 1.0 - near / (_max_dist(s) * max(1, len(ps) - done))) * 0.5) / len(ps)
        clue = 0.15 if color in frame_colors else 0.0
        out.append(GoalInstance(f"inside_frame({color})", "inside_frame", {"color": color}, is_goal, progress, clue=clue))
    return out


def _mark_groups(scene: Scene) -> dict:
    """(colour, size) -> {(top, left)} of small square marks (1x1 .. 3x3) that occur at least four times."""
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    marks: dict = {}
    for o in scene.objects:
        if o.height == o.width and o.height <= 3 and o.area == o.height * o.width and o.region not in strips:
            marks.setdefault((o.color, o.height), set()).add((o.bbox[0], o.bbox[1]))
    return {k: v for k, v in marks.items() if len(v) >= 4}


def _marked_slots(scene: Scene) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Slots marked by four small square corner marks of one colour: returns [(colour, interior bbox)]."""
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    marks = _mark_groups(scene)
    mark_keys = set(marks)
    piece_dims = {(o.height, o.width) for o in scene.objects if 2 <= o.area <= 64 and o.region not in strips and (o.color, o.height) not in mark_keys}
    slots = []
    for (color, k), pts in marks.items():
        if len(pts) < 4 or len(pts) > 64:
            continue
        rows = sorted({r for r, _ in pts}); cols = sorted({c for _, c in pts})
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1:]:
                if r1 - r0 < k + 1 or r1 - r0 > 16:
                    continue
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1:]:
                        if c1 - c0 < k + 1 or c1 - c0 > 16:
                            continue
                        if {(r0, c0), (r0, c1), (r1, c0), (r1, c1)} <= pts:
                            inner = (r0 + k, c0 + k, r1, c1)
                            if (inner[2] - inner[0], inner[3] - inner[1]) in piece_dims:
                                slots.append((color, inner))   # interior sized like a piece
    return slots


def t_fill_marked_slots(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Every corner-marked slot holds a piece (of the marker colour when such pieces exist) — lp85-style targets."""
    slots = _marked_slots(scene)
    if not slots:
        return []
    mark_sizes = set()
    for (c, k), pts in ((k_, v) for k_, v in _mark_groups(scene).items()):
        mark_sizes.add((c, k))
    mark_keys = mark_sizes
    def pieces(s):
        strips = {r.id for r in s.regions if r.kind_hint == "ui_strip"}
        return [o for o in s.objects if 2 <= o.area <= 64 and o.region not in strips
                and not (o.height == o.width and o.height <= 3 and (o.color, o.height) in mark_keys)]
    def inside(p, b):
        return b[0] <= p.bbox[0] and b[1] <= p.bbox[1] and p.bbox[2] <= b[2] and p.bbox[3] <= b[3]
    colors = {c for c, _ in slots}
    piece_colors = {o.color for o in pieces(scene)}
    by_color = bool(colors & piece_colors)
    def filled(s, slot):
        color, b = slot
        return any(inside(p, b) and (p.color == color or not by_color) for p in pieces(s))
    def is_goal(s):
        sl = _marked_slots(s)
        return bool(sl) and all(filled(s, x) for x in sl)
    dist_cache: dict = {}

    def graph_distance(pos, target, graph) -> Optional[int]:
        """Fewest trigger clicks moving a piece from top-left `pos` to `target` under the learned permutations
        (BFS over positions, other pieces ignored: a relaxed plan)."""
        key = (pos, target)
        if key in dist_cache:
            return dist_cache[key]
        from collections import deque
        seen = {pos}; q = deque([(pos, 0)])
        while q:
            cur, d = q.popleft()
            if cur == target:
                dist_cache[key] = d; return d
            for mapping in graph.values():
                nxt = mapping.get(cur)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt); q.append((nxt, d + 1))
        dist_cache[key] = None
        return None

    def progress(s):
        sl = _marked_slots(s)
        if not sl:
            return 0.0
        done = sum(1 for x in sl if filled(s, x))
        ps = pieces(s); near = 0.0
        graph = getattr(getattr(inst, "model_hint", None), "position_graph", None)
        for color, b in sl:
            if filled(s, (color, b)):
                continue
            cands = [p for p in ps if p.color == color] if by_color else ps
            if not cands:
                continue
            if graph:
                ds = [graph_distance((p.bbox[0], p.bbox[1]), (b[0], b[1]), graph) for p in cands]
                ds = [d for d in ds if d is not None]
                if ds:
                    near += 1.0 / (1.0 + min(ds)); continue
            cr, cc = (b[0] + b[2] - 1) // 2, (b[1] + b[3] - 1) // 2
            near += 1.0 - min(_dist(p.center, (cr, cc)) for p in cands) / _max_dist(s)
        return (done + 0.9 * near) / len(sl)

    def estimate(s):
        """Relaxed plan length: sum over unfilled slots of the fewest clicks that bring a matching piece there."""
        graph = getattr(getattr(inst, "model_hint", None), "position_graph", None)
        if not graph:
            return None
        total = 0
        for color, b in _marked_slots(s):
            if filled(s, (color, b)):
                continue
            cands = [p for p in pieces(s) if p.color == color] if by_color else pieces(s)
            ds = [graph_distance((p.bbox[0], p.bbox[1]), (b[0], b[1]), graph) for p in cands]
            ds = [d for d in ds if d is not None]
            if not ds:
                return None
            total += min(ds)
        return float(total)
    def abstract_plan(s, model):
        """Exact plan over the goal-relevant pieces only: BFS on the joint top-left positions of the pieces that can fill
        the slots (other pieces ride along), under the learned per-trigger permutations. Returns trigger top-lefts."""
        graph = getattr(model, "position_graph", None)
        if not graph:
            return None
        from collections import deque
        sl = _marked_slots(s)
        open_slots = [(c, b) for c, b in sl if not filled(s, (c, b))]
        if not open_slots:
            return []
        targets = [(b[0], b[1]) for _, b in open_slots]
        cands = [p for p in pieces(s) if (p.color in {c for c, _ in open_slots})] if by_color else pieces(s)
        cands = cands[:4]
        if not cands:
            return None
        start = tuple((p.bbox[0], p.bbox[1]) for p in cands)
        def done(state):
            return all(any(pos == tg for pos in state) for tg in targets)
        triggers = list(graph.items())
        seen = {start: None}; q = deque([start]); parent = {}
        while q:
            cur = q.popleft()
            if done(cur):
                path = []
                while cur in parent:
                    cur, trig = parent[cur]; path.append(trig)
                return list(reversed(path))
            for tkey, mapping in triggers:
                nxt = tuple(mapping.get(pos, pos) for pos in cur)
                if nxt not in seen:
                    seen[nxt] = True; parent[nxt] = (cur, tkey); q.append(nxt)
            if len(seen) > 200_000:
                return None
        return None
    inst = GoalInstance("fill_marked_slots", "fill_marked_slots", {"slots": len(slots), "by_color": by_color}, is_goal, progress, clue=0.2, estimate_fn=estimate)
    inst.abstract_plan = abstract_plan
    return [inst]


def t_same_cell(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Two agent-like objects (or two objects of one colour that both move) end on the same relative position."""
    ag = _agents(scene)
    if len(ag) != 2:
        return []
    def rel(o, s):
        reg = s.region(o.region)
        r0, c0 = (reg.bbox[0], reg.bbox[1]) if reg else (0, 0)
        return (o.bbox[0] - r0, o.bbox[1] - c0)
    def is_goal(s):
        a = _agents(s)
        return len(a) == 2 and (a[0].overlaps(a[1]) or _adjacent(a[0], a[1]))
    def progress(s):
        a = _agents(s)
        if len(a) != 2:
            return 0.0
        return 1.0 - _dist(a[0].center, a[1].center) / _max_dist(s)
    return [GoalInstance("same_cell(agents)", "same_cell", {}, is_goal, progress, clue=0.1)]


def _layout(scene: Scene, rid: str) -> frozenset:
    """Colour layout of a region's objects relative to the region origin (for pattern matching between regions)."""
    reg = scene.region(rid)
    if reg is None:
        return frozenset()
    r0, c0 = reg.bbox[0], reg.bbox[1]
    return frozenset((o.color, o.bbox[0] - r0, o.bbox[1] - c0, o.shape_sig) for o in scene.in_region(rid))


def t_pattern_match(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """The objects of one region reproduce the colour/shape layout of another region (board copies the panel)."""
    regs = [r for r in scene.regions if r.kind_hint in ("board", "panel") and r.id != "R0"]
    out = []
    for src in regs:
        for dst in regs:
            if src.id == dst.id or not scene.in_region(src.id) or not scene.in_region(dst.id):
                continue
            def is_goal(s, a=src.id, b=dst.id):
                la, lb = _layout(s, a), _layout(s, b)
                return bool(lb) and la == lb
            def progress(s, a=src.id, b=dst.id):
                la, lb = _layout(s, a), _layout(s, b)
                return len(la & lb) / len(lb) if lb else 0.0
            same_size = abs(src.area - dst.area) < 0.2 * max(src.area, dst.area)
            out.append(GoalInstance(f"pattern_match({src.id},{dst.id})", "pattern_match", {"src": src.id, "dst": dst.id}, is_goal, progress, clue=0.15 if same_size else 0.0))
    return out


def _color_units(s: Scene, c: int) -> list:
    """Bounding boxes (r0,c0,r1,c1) of colour c per object outside ui strips: a plain object, or the colour-c pixels of a
    multi-colour object (a mover's coloured tip)."""
    strips = {r.id for r in s.regions if r.kind_hint == "ui_strip"}
    out = []
    for o in s.objects:
        if o.region in strips or c not in o.colors:
            continue
        if o.color_mask is None or len(o.colors) == 1:
            out.append(tuple(o.bbox)); continue
        ys, xs = np.nonzero(o.color_mask == c)
        if len(ys):
            out.append((o.bbox[0] + int(ys.min()), o.bbox[1] + int(xs.min()), o.bbox[0] + int(ys.max()) + 1, o.bbox[1] + int(xs.max()) + 1))
    return out


def t_align_color(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Everything of one colour lines up: a mover's coloured tip must sit in the same columns (or rows) as the marker of
    that colour (slider / pointer games). One colour, two or three units, at least one of them part of a larger object."""
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    colors = {c for o in scene.objects if o.region not in strips for c in o.colors}
    pairs = [c for c in colors if _marker_pair(_color_units(scene, c))]
    if len(pairs) >= 2:
        return []             # several marker/tip pairs: the win is their conjunction (align_all_colors), not one of them
    out = []
    for c in sorted(colors):
        units = _color_units(scene, c)
        if not 2 <= len(units) <= 3 or any((u[2] - u[0]) * (u[3] - u[1]) > 100 for u in units):
            continue          # a big unit is a bar or panel, not a marker
        if len({(u[2] - u[0], u[3] - u[1]) for u in units}) == 1:
            continue          # identical units (two buttons of one colour) are a set of controls, not a marker and a tip
        owners = [o for o in scene.objects if o.region not in strips and c in o.colors]
        if len(owners) < 2:
            continue
        for axis in ("col", "row"):
            k = (1, 3) if axis == "col" else (0, 2)
            def is_goal(s, c=c, k=k):
                us = _color_units(s, c)
                return len(us) >= 2 and len({(u[k[0]], u[k[1]]) for u in us}) == 1
            def progress(s, c=c, k=k):
                us = _color_units(s, c)
                if len(us) < 2:
                    return 0.0
                lo = [u[k[0]] for u in us]
                return 1.0 - (max(lo) - min(lo)) / s.grid_shape[k[0]]
            clue = 0.15 if any(len(o.colors) > 1 for o in owners) else 0.0
            out.append(GoalInstance(f"align_color({c},{axis})", "align_color", {"color": c, "axis": axis}, is_goal, progress, clue=clue))
    return out


def _marker_pair(units: list) -> bool:
    """Two small units of one colour that differ in size: a marker and a mover's tip (two identical buttons are not)."""
    return (len(units) == 2 and all((u[2] - u[0]) * (u[3] - u[1]) <= 100 for u in units)
            and (units[0][2] - units[0][0], units[0][3] - units[0][1]) != (units[1][2] - units[1][0], units[1][3] - units[1][1]))


def t_align_all_colors(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Every colour that has exactly two small units (a marker and a mover's tip) lines up on the same axis: the
    conjunction of align_color over colours (three movers, three markers). One colour -> same as align_color."""
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    colors = sorted({c for o in scene.objects if o.region not in strips for c in o.colors})
    pairs = [c for c in colors if _marker_pair(_color_units(scene, c))]
    if len(pairs) < 2:
        return []
    out = []
    for axis in ("col", "row"):
        k = (1, 3) if axis == "col" else (0, 2)
        def aligned(s, c, k=k):
            us = _color_units(s, c)
            return len(us) >= 2 and len({(u[k[0]], u[k[1]]) for u in us}) == 1
        def is_goal(s, cs=tuple(pairs), k=k):
            return all(aligned(s, c, k) for c in cs)
        def progress(s, cs=tuple(pairs), k=k):
            tot = 0.0
            for c in cs:
                us = _color_units(s, c)
                if len(us) < 2:
                    continue
                lo = [u[k[0]] for u in us]
                tot += 1.0 - (max(lo) - min(lo)) / s.grid_shape[k[0]]
            return tot / len(cs)
        def estimate(s, cs=tuple(pairs), k=k):
            tot = 0.0
            for c in cs:
                us = _color_units(s, c)
                if len(us) >= 2:
                    lo = [u[k[0]] for u in us]; tot += (max(lo) - min(lo)) / 2.0
            return tot
        out.append(GoalInstance(f"align_all_colors({axis})", "align_all_colors", {"axis": axis, "colors": list(pairs)}, is_goal, progress, clue=0.2, estimate_fn=estimate))
    return out


def _layout_of(o) -> "np.ndarray":
    return o.color_mask if o.color_mask is not None else np.where(o.mask, o.color, -1).astype(np.int8)


def _find_by_id_or_bbox(s: Scene, oid: int, bbox):
    o = s.get(oid)
    if o is not None and (o.height, o.width) == (bbox[2] - bbox[0], bbox[3] - bbox[1]):
        return o
    return next((x for x in s.objects if tuple(x.bbox) == tuple(bbox)), None)


def t_match_pattern(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """A multi-colour rectangular object (the target pattern) must be reproduced by another object of the same size (the canvas):
    stamp / paint / colour-fill games. Both directions are instantiated; the level-up filter and the clue (the canvas starts
    uniform) decide which is the target."""
    pats = [o for o in scene.objects if o.color_mask is not None and len(o.colors) >= 2 and o.area == o.height * o.width
            and min(o.height, o.width) >= 3 and scene.region(o.region) is not None and scene.region(o.region).kind_hint != "ui_strip"]
    out = []
    for tgt in pats:
        for cv in scene.objects:
            if cv.id == tgt.id or (cv.height, cv.width) != (tgt.height, tgt.width) or cv.area != tgt.area:
                continue
            if scene.region(cv.region) is not None and scene.region(cv.region).kind_hint == "ui_strip":
                continue
            def is_goal(s, a=tgt.id, ab=tgt.bbox, b=cv.id, bb=cv.bbox):
                A, B = _find_by_id_or_bbox(s, a, ab), _find_by_id_or_bbox(s, b, bb)
                return A is not None and B is not None and A.area == B.area and bool(np.array_equal(_layout_of(A), _layout_of(B)))
            def progress(s, a=tgt.id, ab=tgt.bbox, b=cv.id, bb=cv.bbox):
                A, B = _find_by_id_or_bbox(s, a, ab), _find_by_id_or_bbox(s, b, bb)
                if A is None or B is None or A.area != B.area:
                    return 0.0
                la, lb = _layout_of(A), _layout_of(B)
                return float((la == lb).mean()) if la.shape == lb.shape else 0.0
            def estimate(s, a=tgt.id, ab=tgt.bbox, b=cv.id, bb=cv.bbox):
                A, B = _find_by_id_or_bbox(s, a, ab), _find_by_id_or_bbox(s, b, bb)
                if A is None or B is None or A.area != B.area:
                    return None
                la, lb = _layout_of(A), _layout_of(B)
                return float(len({int(v) for v in la[la != lb]})) if la.shape == lb.shape else None   # >= one action per colour still missing
            # the canvas is painted with the target's colours (blank or partly done); the name follows the canvas POSITION so the
            # goal keeps its confidence when the tracker re-numbers a fully repainted canvas
            clue = 0.2 if set(cv.colors) < set(tgt.colors) else (0.1 if set(cv.colors) == set(tgt.colors) else 0.0)
            out.append(GoalInstance(f"match_pattern({tgt.id}->@{cv.bbox[0]},{cv.bbox[1]})", "match_pattern", {"target": tgt.id, "canvas": cv.id, "canvas_bbox": list(cv.bbox)},
                                    is_goal, progress, clue=clue, estimate_fn=estimate))
    return out


def t_explore(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Curiosity fallback (not a win condition): bring the agent next to an object it has not touched yet. `ctx['touched']`
    is the set of object identities already reached. Used by the orchestrator when no real goal yields a plan."""
    if not _agents(scene):
        return []
    touched = set(ctx.get("touched", ()))
    def targets(s):
        return [o for o in s.objects if not (o.role and "agent" in o.role) and o.role not in ("wall", "indicator", "decoration") and o.identity() not in touched
                and s.region(o.region) is not None and s.region(o.region).kind_hint != "ui_strip"]
    def is_goal(s):
        ag = _agents(s)
        return any(a.overlaps(o) or _adjacent(a, o) for a in ag for o in targets(s))
    def progress(s):
        ag, ts = _agents(s), targets(s)
        if not ag or not ts:
            return 0.0
        return 1.0 - min(_dist(a.center, o.center) for a in ag for o in ts) / _max_dist(s)
    return [GoalInstance("explore(agent)", "explore", {}, is_goal, progress, confidence=0.0)]


def _adjacent(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox; br0, bc0, br1, bc1 = b.bbox
    touch_r = (ar1 == br0 or br1 == ar0) and not (ac1 <= bc0 or bc1 <= ac0)
    touch_c = (ac1 == bc0 or bc1 == ac0) and not (ar1 <= br0 or br1 <= ar0)
    return touch_r or touch_c


# confidence tiers: role-based templates are more specific than colour/geometry ones (spec §9 usage-stats prior 0.5)
BASE_CONFIDENCE = {"reach": 0.5, "all_collected": 0.5, "inside_frame": 0.5, "fill_marked_slots": 0.5, "same_cell": 0.5, "pattern_match": 0.45, "match_pattern": 0.5, "align_color": 0.5, "align_all_colors": 0.5, "match_shapes": 0.45,
                   "enclose": 0.4, "align": 0.35, "all_removed": 0.3, "fill_region": 0.3, "count_equals": 0.2, "sort_by": 0.2, "sequence": 0.3, "explore": 0.0}

TEMPLATES: dict[str, Callable[[Scene, dict], list[GoalInstance]]] = {
    "reach": t_reach, "match_shapes": t_match_shapes, "all_removed": t_all_removed, "all_collected": t_all_collected,
    "fill_region": t_fill_region, "sort_by": t_sort_by, "count_equals": t_count_equals, "align": t_align, "enclose": t_enclose,
    "sequence": t_sequence, "inside_frame": t_inside_frame, "fill_marked_slots": t_fill_marked_slots, "same_cell": t_same_cell, "pattern_match": t_pattern_match, "match_pattern": t_match_pattern, "align_color": t_align_color, "align_all_colors": t_align_all_colors, "explore": t_explore}


def instantiate_all(scene: Scene, ctx: Optional[dict] = None, usage_stats: Optional[dict] = None) -> list[GoalInstance]:
    """All template instances that are NOT already true on the scene (a goal achieved at the start is not a goal)."""
    ctx = ctx or {}
    out: list[GoalInstance] = []
    for name, fn in TEMPLATES.items():
        if name == "explore" and not ctx.get("include_explore"):
            continue
        try:
            insts = fn(scene, ctx)
        except Exception:
            continue
        stats = (usage_stats or {}).get(name, {})
        base = stats.get("games_verified", 0) / stats["games_used"] if stats.get("games_used") else BASE_CONFIDENCE.get(name, 0.5)
        for g in insts:
            if g.is_goal(scene):
                continue
            g.confidence = min(1.0, base + g.clue)
            out.append(g)
    return out
