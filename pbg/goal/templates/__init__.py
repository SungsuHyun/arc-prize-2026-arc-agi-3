"""goal/templates — the initial goal template library (spec §9). Each template instantiates GoalInstances on a scene:
is_goal(scene) -> bool, progress(scene) -> [0,1], plus a clue bonus from scene structure."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

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
    if "agent" not in roles:
        return out
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    targets = sorted(roles - {"agent", "wall", "decoration", "indicator"})
    targets += sorted({f"color:{o.color}" for o in scene.objects if o.role in (None, "unknown") and o.region not in strips})
    a = "agent"
    for b in targets:
        def is_goal(s, b=b):
            return any(x.overlaps(y) or _adjacent(x, y) for x in s.by_role(a) for y in _targets(s, b))
        def progress(s, b=b):
            xs, ys = s.by_role(a), _targets(s, b)
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


def t_explore(scene: Scene, ctx: dict) -> list[GoalInstance]:
    """Curiosity fallback (not a win condition): bring the agent next to an object it has not touched yet. `ctx['touched']`
    is the set of object identities already reached. Used by the orchestrator when no real goal yields a plan."""
    if "agent" not in _roles(scene):
        return []
    touched = set(ctx.get("touched", ()))
    def targets(s):
        return [o for o in s.objects if o.role not in ("agent", "wall", "indicator", "decoration") and o.identity() not in touched
                and s.region(o.region) is not None and s.region(o.region).kind_hint != "ui_strip"]
    def is_goal(s):
        ag = s.by_role("agent")
        return any(a.overlaps(o) or _adjacent(a, o) for a in ag for o in targets(s))
    def progress(s):
        ag, ts = s.by_role("agent"), targets(s)
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
BASE_CONFIDENCE = {"reach": 0.5, "all_collected": 0.5, "match_shapes": 0.45, "enclose": 0.4, "align": 0.35, "all_removed": 0.3,
                   "fill_region": 0.3, "count_equals": 0.2, "sort_by": 0.2, "sequence": 0.3, "explore": 0.0}

TEMPLATES: dict[str, Callable[[Scene, dict], list[GoalInstance]]] = {
    "reach": t_reach, "match_shapes": t_match_shapes, "all_removed": t_all_removed, "all_collected": t_all_collected,
    "fill_region": t_fill_region, "sort_by": t_sort_by, "count_equals": t_count_equals, "align": t_align, "enclose": t_enclose,
    "sequence": t_sequence, "explore": t_explore}


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
