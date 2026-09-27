"""Relational win predicates learned from a completed level and evaluated live on later levels.

At level completion the harness looks at the board just before the completing action (and, when the effect model can
predict it, the board just after) and keeps every predicate of a small family that holds there but did not hold when the
level started: gone(colour in a region), inside(objects of colour a in frames of colour f), equal(two panels' marks),
pose(an object has the same shape as a target object), pressed(the final click was a submit button). On later levels
each predicate is re-grounded on the new board and reports whether it holds and how far it is from holding.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from arcnav.frame import Frame

from .entities import Obj, Scene, build_scene

FRAME_KINDS = ("corners", "outline")


@dataclass
class FrameShape:
    color: int
    kind: str
    inner: tuple            # (r0, c0, r1, c1) the area a block must occupy
    dots: list              # object ids forming the frame

    def contains(self, o: Obj) -> bool:
        r0, c0, r1, c1 = self.inner
        tol = 1 if self.kind == "corners" else 2
        ctol = 2 if self.kind == "corners" else 3
        return o.bbox[0] >= r0 - tol and o.bbox[1] >= c0 - tol and o.bbox[2] <= r1 + tol and o.bbox[3] <= c1 + tol and \
            abs((o.bbox[0] + o.bbox[2]) / 2 - (r0 + r1) / 2) <= ctol and abs((o.bbox[1] + o.bbox[3]) / 2 - (c0 + c1) / 2) <= ctol


def find_frames(sc: Scene) -> list:
    """Frames drawn with small marks: four corner marks of one colour forming a rectangle, or a ring of dots."""
    out: list[FrameShape] = []
    small: dict = defaultdict(list)
    for o in sc.objs:
        if not o.hud and o.size <= 4:
            small[o.color].append(o)
    for c, objs in small.items():
        used: set = set()
        # dotted outlines first: chain dots (<= 2 px) within distance 3
        dots = [o for o in objs if o.size <= 2]
        seen: set = set()
        for o in dots:
            if o.id in seen:
                continue
            cluster = [o]; seen.add(o.id); i = 0
            while i < len(cluster):
                cur = cluster[i]; i += 1
                for x in dots:
                    if x.id not in seen and abs(x.center[0] - cur.center[0]) + abs(x.center[1] - cur.center[1]) <= 3:
                        seen.add(x.id); cluster.append(x)
            if len(cluster) >= 6:
                rs = [x.center[0] for x in cluster]; cs = [x.center[1] for x in cluster]
                r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
                # a ring, not a line: dots on at least three sides of the bbox
                sides = sum(1 for cond in (any(x.center[0] == r0 for x in cluster), any(x.center[0] == r1 for x in cluster),
                                           any(x.center[1] == c0 for x in cluster), any(x.center[1] == c1 for x in cluster)) if cond)
                if r1 - r0 >= 4 and c1 - c0 >= 4 and sides >= 3 and len(cluster) <= 40:
                    out.append(FrameShape(c, "outline", (r0, c0, r1, c1), [x.id for x in cluster])); used |= {x.id for x in cluster}
        # corner frames: four equal marks at the corners of a rectangle
        marks = [o for o in objs if o.id not in used]
        for a in marks:
            for b in marks:
                if b.id == a.id or abs(b.bbox[0] - a.bbox[0]) > 1 or b.bbox[1] <= a.bbox[3] + 1 or b.size != a.size:
                    continue
                for d in marks:
                    if d.id in (a.id, b.id) or abs(d.bbox[1] - a.bbox[1]) > 1 or d.bbox[0] <= a.bbox[2] + 1 or d.size != a.size:
                        continue
                    e = next((x for x in marks if x.id not in (a.id, b.id, d.id) and abs(x.bbox[0] - d.bbox[0]) <= 1 and abs(x.bbox[1] - b.bbox[1]) <= 1), None)
                    if e is None or a.id in used:
                        continue
                    inner = (a.bbox[2] + 1, a.bbox[3] + 1, d.bbox[0] - 1, b.bbox[1] - 1)
                    if inner[2] - inner[0] >= 1 and inner[3] - inner[1] >= 1 and inner[2] - inner[0] <= 12 and inner[3] - inner[1] <= 12:
                        out.append(FrameShape(c, "corners", inner, [a.id, b.id, d.id, e.id])); used |= {a.id, b.id, d.id, e.id}
    return out


def region_color_map(sc: Scene) -> dict:
    """region id -> (colour, index among regions of that colour ordered by position)"""
    by: dict = defaultdict(list)
    for rg in sc.regions:
        by[rg.color].append(rg)
    out = {}
    for c, rgs in by.items():
        rgs.sort(key=lambda r: (r.bbox[0], r.bbox[1]))
        for i, rg in enumerate(rgs):
            out[rg.id] = (c, i)
    return out


def _region_by_spec(sc: Scene, spec: tuple) -> Optional[int]:
    inv = {v: k for k, v in region_color_map(sc).items()}
    return inv.get(tuple(spec))


# ── predicates ──────────────────────────────────────────────────────────

class Predicate:
    kind = "pred"

    def fact(self) -> dict:
        raise NotImplementedError

    def evaluate(self, sc: Scene) -> tuple:
        """(holds, progress text, progress number: 0 = far, 1 = holds)"""
        raise NotImplementedError

    @property
    def needs_submit(self) -> bool:
        return False


@dataclass
class Gone(Predicate):
    color: int
    region: tuple           # (region colour, index)
    kind = "gone"

    def fact(self) -> dict:
        return {"kind": "gone", "params": {"color": self.color, "region": list(self.region)}, "support": 1, "counter": 0,
                "text": f"no colour-{self.color} object is left in the colour-{self.region[0]} area"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        rid = _region_by_spec(sc, self.region)
        n = sum(1 for o in sc.objs if o.color == self.color and not o.hud and (rid is None or o.region == rid))
        return n == 0, (f"holds (no colour-{self.color} objects left)" if n == 0 else f"{n} colour-{self.color} objects still there"), 1.0 if n == 0 else 1.0 / (1 + n)


@dataclass
class Inside(Predicate):
    color: int              # block colour
    frame: int              # frame colour
    fkind: str
    kind = "inside"

    def fact(self) -> dict:
        same = self.color == self.frame
        return {"kind": "inside", "params": {"color": self.color, "frame": self.frame}, "support": 1, "counter": 0,
                "text": (f"every block sits inside the frame of its own colour ({'corner marks' if self.fkind == 'corners' else 'dotted outline'}; seen with colour {self.color})" if same
                         else f"every colour-{self.color} block sits inside a frame drawn in colour {self.frame} ({'corner marks' if self.fkind == 'corners' else 'dotted outline'})")}

    def pairs(self, sc: Scene, frames: Optional[list] = None) -> list:
        """[(colour of blocks, colour of frames)] this predicate covers on the board: one pair, or every same-colour pair."""
        allf = frames if frames is not None else find_frames(sc)
        if self.color != self.frame:
            return [(self.color, self.frame)]
        return sorted({f.color for f in allf})

    def evaluate(self, sc: Scene, ev=None, frames: Optional[list] = None) -> tuple:
        allf = frames if frames is not None else find_frames(sc)
        same = self.color == self.frame
        pairs = [(c, c) for c in sorted({f.color for f in allf})] if same else [(self.color, self.frame)]
        tot_k = tot_n = 0; notes = []
        for bc, fc in pairs:
            frames_c = [f for f in allf if f.color == fc]
            dot_ids = {i for f in frames_c for i in f.dots}
            blocks = [o for o in sc.objs if o.color == bc and not o.hud and o.size >= 4 and o.id not in dot_ids]
            if not frames_c or not blocks:
                continue
            inside = [o for o in blocks if any(f.contains(o) for f in frames_c)]
            k, n = len(inside), min(len(blocks), len(frames_c)); tot_k += k; tot_n += n
            if k < n:
                notes.append(f"colour {bc}: {k}/{n} in frames; " + ", ".join(f"#{o.id}@({o.center[0]},{o.center[1]})" for o in blocks if o not in inside) + " outside, frames at " + ", ".join(f"({f.inner[0]},{f.inner[1]})" for f in frames_c))
        if tot_n == 0:
            return False, f"no colour-{self.frame} frame / colour-{self.color} block on this board" if not same else "no frame with same-colour blocks on this board", 0.0
        holds = tot_k >= tot_n
        return holds, (f"holds ({tot_k} block(s) in frames)" if holds else "; ".join(notes)), tot_k / tot_n


@dataclass
class Equal(Predicate):
    r1: tuple               # (region colour, index) editable
    r2: tuple               # template
    kind = "equal"

    def fact(self) -> dict:
        return {"kind": "equal", "params": {"r1": list(self.r1), "r2": list(self.r2)}, "support": 1, "counter": 0,
                "text": f"the marks in the colour-{self.r1[0]} area match the marks in the colour-{self.r2[0]} area (same pattern)"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        a, b = _region_by_spec(sc, self.r1), _region_by_spec(sc, self.r2)
        if a is None or b is None:
            return False, "one of the areas is missing on this board", 0.0
        ma, mb = _mask(sc, a), _mask(sc, b)
        diff = set(ma.items()) ^ set(mb.items())
        cells = len({k for k, v in diff})
        return cells == 0, ("holds (patterns identical)" if cells == 0 else f"{cells} cells differ between the two areas"), 1.0 / (1 + cells)


@dataclass
class Pose(Predicate):
    color: int
    target: int
    kind = "pose"

    def fact(self) -> dict:
        return {"kind": "pose", "params": {"color": self.color, "target": self.target}, "support": 1, "counter": 0,
                "text": f"a colour-{self.color} object has exactly the shape of a colour-{self.target} object (the target shape)"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        a = [o for o in sc.objs if o.color == self.color and not o.hud and o.size >= 6]
        t = [o for o in sc.objs if o.color == self.target and not o.hud and o.size >= 6]
        ok = any(x.shape == y.shape for x in a for y in t)
        return ok, ("holds (shapes match)" if ok else f"no colour-{self.color} object matches a colour-{self.target} shape yet"), 1.0 if ok else 0.0


@dataclass
class Pressed(Predicate):
    color: int
    region: tuple
    size: int
    kind = "pressed"

    def fact(self) -> dict:
        return {"kind": "pressed", "params": {"color": self.color, "region": list(self.region)}, "support": 1, "counter": 0,
                "text": f"the level is submitted by clicking the colour-{self.color} button ({self.size}px) in the colour-{self.region[0]} area once the other conditions hold"}

    def button(self, sc: Scene) -> Optional[Obj]:
        rid = _region_by_spec(sc, self.region)
        cands = [o for o in sc.objs if o.color == self.color and not o.hud and o.size >= 8 and (rid is None or o.region == rid)]
        return min(cands, key=lambda o: abs(o.size - self.size)) if cands else None

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        b = self.button(sc)
        return b is not None, (f"submit button = #{b.id} at ({b.center[0]},{b.center[1]})" if b else "no such button on this board"), 1.0 if b else 0.0

    @property
    def needs_submit(self) -> bool:
        return True


# ── movement-game predicates (built from rulebook entries: model-written or harness win facts) ──────────

def _avatar_box(ev):
    av = getattr(ev, "avatar", None) if ev is not None else None
    if not av:
        return None
    x0, y0, x1, y1 = av["bbox_xyxy"]
    return (y0, x0, y1, x1)


def _touching(box, o: Obj, gap: int = 1) -> bool:
    r0, c0, r1, c1 = box
    return not (o.bbox[0] > r1 + gap or o.bbox[2] < r0 - gap or o.bbox[1] > c1 + gap or o.bbox[3] < c0 - gap)


@dataclass
class CollectAll(Predicate):
    color: int
    kind = "collect_all"

    def fact(self) -> dict:
        return {"kind": "collect_all", "params": {"collect": self.color}, "support": 1, "counter": 0, "text": f"every colour-{self.color} object has been collected"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        box = _avatar_box(ev)
        left = [o for o in sc.objs if o.color == self.color and not o.hud and o.size <= 80 and not (box and _touching(box, o, 0))]
        n = len(left)
        return n == 0, (f"holds (no colour-{self.color} objects left)" if n == 0 else f"{n} colour-{self.color} objects left: " + ", ".join(f"#{o.id}@({o.center[0]},{o.center[1]})" for o in left[:5])), 1.0 / (1 + n)


@dataclass
class Reach(Predicate):
    color: int
    kind = "reach"

    def fact(self) -> dict:
        return {"kind": "reach", "params": {"reach": self.color}, "support": 1, "counter": 0, "text": f"the avatar stands on / touches a colour-{self.color} object"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        box = _avatar_box(ev)
        targets = [o for o in sc.objs if o.color == self.color and not o.hud]
        if box is None:
            return False, "avatar not identified yet", 0.0
        if not targets:
            return False, f"no colour-{self.color} object on the board", 0.0
        hit = [o for o in targets if _touching(box, o)]
        return bool(hit), (f"holds (avatar touches #{hit[0].id})" if hit else f"avatar at ({box[0]},{box[1]}); nearest colour-{self.color} object at " + ", ".join(f"#{o.id}@({o.center[0]},{o.center[1]})" for o in targets[:3])), 1.0 if hit else 0.0


@dataclass
class CollectReach(Predicate):
    color: int
    reach: int
    kind = "collect_reach"

    def fact(self) -> dict:
        return {"kind": "collect_reach", "params": {"collect": self.color, "reach": self.reach}, "support": 1, "counter": 0,
                "text": f"collect every colour-{self.color} object, then touch a colour-{self.reach} object"}

    def evaluate(self, sc: Scene, ev=None) -> tuple:
        ok1, t1, p1 = CollectAll(self.color).evaluate(sc, ev)
        ok2, t2, p2 = Reach(self.reach).evaluate(sc, ev)
        return ok1 and ok2, ("holds" if ok1 and ok2 else ("collected; " + t2 if ok1 else t1)), (p1 + (p2 if ok1 else 0)) / 2


def from_entry(entry) -> Optional[Predicate]:
    """A predicate object for a rulebook win entry of a known kind (model-written or harness-written), else None."""
    k, p = entry.kind, entry.params or {}
    try:
        if k == "collect_all" and p.get("collect") is not None:
            return CollectAll(int(p["collect"]))
        if k == "reach" and p.get("reach") is not None:
            return Reach(int(p["reach"]))
        if k == "collect_reach" and p.get("collect") is not None and p.get("reach") is not None:
            return CollectReach(int(p["collect"]), int(p["reach"]))
        if k == "gone" and p.get("color") is not None and p.get("region"):
            return Gone(int(p["color"]), tuple(p["region"]))
        if k == "inside" and p.get("color") is not None and p.get("frame") is not None:
            return Inside(int(p["color"]), int(p["frame"]), "corners")
        if k == "pressed" and p.get("color") is not None and p.get("region"):
            return Pressed(int(p["color"]), tuple(p["region"]), int(p.get("size", 20)))
    except (TypeError, ValueError):
        return None
    return None


def _mask(sc: Scene, rid: int) -> dict:
    rg = sc.region_named(rid); r0, c0 = rg.bbox[0], rg.bbox[1]
    out = {}
    for o in sc.objs:
        if o.region == rid and not o.hud and o.size <= 16 and o.cells:
            for r, c in o.cells:
                out[(r - r0, c - c0)] = o.color
    return out


# ── inference at level completion ───────────────────────────────────────

def infer(start: Frame, pre: Frame, post_board: Optional[list], final_action, ev) -> list:
    """Predicates that hold on the (predicted) completing board but not on the level's first board."""
    s0, s1 = build_scene(start), build_scene(pre)
    s2 = build_scene(Frame(post_board, step=pre.step, level=pre.level)) if post_board is not None else None
    preds: list[Predicate] = []
    rc1 = region_color_map(s1)
    # gone(colour, region)
    for rid, spec in rc1.items():
        cols0 = {o.color for o in s0.objs if o.region == _region_by_spec(s0, spec) and not o.hud} if _region_by_spec(s0, spec) is not None else set()
        cols1 = {o.color for o in s1.objs if o.region == rid and not o.hud}
        for c in cols0 - cols1:
            preds.append(Gone(c, spec))
    # inside(colour, frame colour): on the predicted post board (what the final action produced) when available, else on pre;
    # a predicate the final action made true (false on pre, true on post) is the strongest evidence
    submit_like = isinstance(final_action, dict) and ev is not None and (lambda o: o is not None and ev.cs.alive.get(o.key, 0) == 0)(s1.obj_around(final_action["row"], final_action["col"]))
    boards = ([("post", s2)] if s2 is not None else []) + [("pre", s1)]
    pre_frames = find_frames(s1)
    # marker colours (the selection highlight that jumps to wherever one clicks) are never the block a frame is for
    marker_cols = set()
    if ev is not None:
        for tbl in (ev.cs.cursor, ev.cs_all.cursor):
            for key, cnt in tbl.items():
                k, n = cnt.most_common(1)[0]
                if k != -1 and n * 2 > sum(cnt.values()):
                    marker_cols.add(k)
    for tag, sc in boards:
        frames = pre_frames   # frames are static drawings; a block arriving on the post board may hide some of their dots
        if not frames:
            continue
        dot_ids = {i for f in frames for i in f.dots}
        for fc in {f.color for f in frames}:
            fs = [f for f in frames if f.color == fc]
            for bc in {o.color for o in sc.objs if o.size >= 4 and not o.hud and o.id not in dot_ids and o.color not in marker_cols}:
                blocks = [o for o in sc.objs if o.color == bc and o.size >= 4 and not o.hud and o.id not in dot_ids]
                k = sum(1 for o in blocks if any(f.contains(o) for f in fs))
                if k >= min(len(blocks), len(fs)) and k >= 1:
                    p = Inside(bc, fc, fs[0].kind)
                    if tag == "post" and p.evaluate(s1, None, pre_frames)[0] and not submit_like:
                        continue   # already true before the final action, which was not a submit: not what the action achieved
                    if tag == "pre" and s2 is not None and not submit_like:
                        continue   # the final action changed the board: judge on the post board only
                    if not p.evaluate(s0)[0] and not any(isinstance(q, Inside) and q.color == bc and q.frame == fc for q in preds):
                        preds.append(p)
    # equal(two areas of the same size)
    regs = [rg for rg in s1.regions if not rg.edge or rg.size < 1500]
    for a in regs:
        for b in regs:
            if a.id >= b.id or abs((a.bbox[2] - a.bbox[0]) - (b.bbox[2] - b.bbox[0])) > 1 or abs((a.bbox[3] - a.bbox[1]) - (b.bbox[3] - b.bbox[1])) > 1:
                continue
            ma, mb = _mask(s1, a.id), _mask(s1, b.id)
            if ma and ma == mb:
                p = Equal(rc1[a.id], rc1[b.id])
                if not p.evaluate(s0)[0]:
                    preds.append(p)
    # pose(colour a matches a colour-t shape)
    shapes: dict = defaultdict(set)
    for o in s1.objs:
        if not o.hud and o.size >= 6:
            shapes[o.shape].add(o.color)
    for sh, cols in shapes.items():
        if len(cols) >= 2:
            cols = sorted(cols)
            for a in cols:
                for t in cols:
                    if a != t:
                        p = Pose(a, t)
                        if not p.evaluate(s0)[0] and not any(isinstance(q, Pose) and q.color == a and q.target == t for q in preds):
                            preds.append(p)
    # pressed(submit button): the final click hit a sizeable object whose earlier clicks did nothing in the playfield
    if isinstance(final_action, dict):
        o = s1.obj_around(final_action["row"], final_action["col"])
        if o is not None and o.size >= 8 and o.region in rc1:
            dead_before = ev.cs.dead.get(o.key, 0) if ev is not None else 0
            alive_before = ev.cs.alive.get(o.key, 0) if ev is not None else 0
            same = [x for x in s1.objs if x.color == o.color and not x.hud and x.size >= 8]
            if alive_before == 0 and (dead_before >= 1 or len(same) == 1):
                preds.append(Pressed(o.color, rc1[o.region], o.size))
    return preds


def essential_sequence(attempt: list) -> list:
    """The winning attempt's actions minus those that changed nothing in the playfield (the last one is always kept),
    described at object level."""
    from .clicks import record
    out = []
    for i, t in enumerate(attempt):
        last = i == len(attempt) - 1
        if isinstance(t.action, dict):
            rec = record(t)
            if rec.cls != "world" and not last:
                continue
            o = rec.obj
            what = "level completed" if last else ("the object turned into colour %d" % rec.vanish_to if rec.vanish_to is not None else
                                                     ("the colour-%d marker moved here" % rec.marker.color if rec.marker else f"{len(rec.moved)} objects moved"))
            out.append(f"click {('#%d colour %d %dpx' % (o.id, o.color, o.size)) if o else 'floor colour %d' % rec.color}" + (f" in P{rec.region}" if rec.region >= 0 else "") + f" @({rec.row},{rec.col}) -> {what}")
        else:
            if t.before_frame.ascii == t.after_frame.ascii and not last:
                continue
            out.append(f"{t.action}" + (" -> level completed" if last else ""))
    return out
