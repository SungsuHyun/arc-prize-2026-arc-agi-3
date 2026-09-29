"""probe/clickmap.py — click response map: every click in the log clustered by its TARGET CLASS (colour + shape of the
object under the cursor, or the background cell), with how often that class reacted. Shared by the walk probe, the
information-gain experiments and the planner so that none of them keeps clicking things already known to be inert.

The class key is level-independent (an arrow button of colour 9 stays an arrow button on the next level), so the map is
a knowledge asset that accumulates across levels: known-responsive classes are tried first on a new level, known-inert
classes at most once."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from ..core.types import Action, Scene, Transition
from .semantics import core_diff

BG_CELL = 8   # background clicks are clustered by 8x8 cell


def click_key(scene: Scene, row: int, col: int) -> str:
    """Class of a click target: 'c<colour>:<shape>' for an object, 'bg:<region>:<r8>,<c8>' for empty space."""
    for o in sorted(scene.objects, key=lambda o: o.area):          # smallest object first (a marker on a bar)
        r0, c0, r1, c1 = o.bbox
        if r0 <= row < r1 and c0 <= col < c1:
            return object_key(o)
    reg = next((r.id for r in scene.regions if r.id != "R0" and r.bbox[0] <= row < r.bbox[2] and r.bbox[1] <= col < r.bbox[3]), "R0")
    return f"bg:{reg}:{row // BG_CELL},{col // BG_CELL}"


def object_key(o) -> str:
    """Class of an object for the click map: colour + shape, but a solid rectangle is keyed by colour + thickness so a
    bar whose length changes stays one class."""
    if o.mask is not None and min(o.height, o.width) >= 2 and float(o.mask.mean()) >= 0.75:
        return f"c{o.color}:rect"
    return f"c{o.color}:{o.shape_sig[:8]}"


class ClickMap:
    def __init__(self) -> None:
        self.n: Counter = Counter()             # class -> clicks
        self.responsive: Counter = Counter()    # class -> clicks that changed something outside ui strips
        self.effects: dict[str, Counter] = {}   # class -> Counter of (moved colour, displacement) / 'recolor' / 'remove' / 'spawn'
        self.seen_ids: set[str] = set()         # transition ids already counted
        self.tried_level: dict[int, set[str]] = {}   # level -> classes clicked on that level

    # ── building ──
    def add(self, t: Transition) -> None:
        if t.action.type != "CLICK" or t.id in self.seen_ids:
            return
        self.seen_ids.add(t.id)
        key = click_key(t.before, t.action.row, t.action.col)
        self.n[key] += 1
        self.tried_level.setdefault(t.level, set()).add(key)
        c = core_diff(t)
        if c["moved"] or c["recolored"] or c["reshaped"] or c["appeared"] or c["disappeared"]:
            self.responsive[key] += 1
            eff = self.effects.setdefault(key, Counter())
            for i, v in c["moved"]:
                o = t.before.get(i)
                eff[("move", o.color if o else -1, tuple(v))] += 1
            for x in c["recolored"]:
                eff[("recolor", x[1], x[2])] += 1
            for x in c["reshaped"]:
                eff[("reshape",)] += 1
            if c["appeared"]:
                eff[("spawn",)] += 1
            if c["disappeared"]:
                eff[("remove",)] += 1

    def extend(self, ts: Iterable[Transition]) -> None:
        for t in ts:
            self.add(t)

    @classmethod
    def from_log(cls, ts: Iterable[Transition]) -> "ClickMap":
        m = cls(); m.extend(ts); return m

    # ── queries ──
    def status(self, key: str) -> str:
        """'untried' | 'responsive' | 'inert'."""
        if self.n[key] == 0:
            return "untried"
        return "responsive" if self.responsive[key] > 0 else "inert"

    def total(self) -> int:
        return sum(self.n.values())

    def rank(self, scene: Scene, *, level: Optional[int] = None, include_inert: bool = False, limit: int = 64,
             extra: Iterable[tuple[int, int]] = ()) -> list[Action]:
        """Click candidates on this scene in priority order: untried classes, then responsive classes (one object per
        class, largest first), then background cells never tried; inert classes only with `include_inert`, and then
        only if not yet clicked on this level (one re-check per level: a submit button may wake up)."""
        strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
        tried_here = self.tried_level.get(level, set()) if level is not None else set()
        untried: list[Action] = []; resp: list[Action] = []; inert: list[Action] = []
        seen_keys: set[str] = set()
        for o in sorted(scene.objects, key=lambda o: (-o.area, o.id)):
            if o.region in strips or o.area < 2:
                continue
            key = object_key(o)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            a = Action.click(*o.center)
            st = self.status(key)
            if st == "untried":
                untried.append(a)
            elif st == "responsive":
                resp.append(a)
            elif include_inert and key not in tried_here:
                inert.append(a)
        # responsive classes with several objects (four arrow buttons of one colour): every object of the class matters
        for o in sorted(scene.objects, key=lambda o: (-o.area, o.id)):
            key = object_key(o)
            if o.region in strips or o.area < 2 or self.status(key) != "responsive":
                continue
            a = Action.click(*o.center)
            if a not in resp:
                resp.append(a)
        bg: list[Action] = []
        for r in scene.regions:
            if r.kind_hint == "ui_strip":
                continue
            a = Action.click(*r.center)
            if self.status(click_key(scene, a.row, a.col)) == "untried" and not any(x == a for x in untried + resp):
                bg.append(a)
        for rc in extra:
            a = Action.click(int(rc[0]), int(rc[1]))
            if a not in resp and a not in untried:
                resp.append(a)
        return (untried + resp + bg + inert)[:limit]

    # ── persistence ──
    def to_json(self) -> dict:
        return {"n": dict(self.n), "responsive": dict(self.responsive),
                "effects": {k: [[list(e) if isinstance(e, tuple) else e, c] for e, c in v.items()] for k, v in self.effects.items()},
                "tried_level": {str(l): sorted(v) for l, v in self.tried_level.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "ClickMap":
        m = cls()
        m.n = Counter(d.get("n", {})); m.responsive = Counter(d.get("responsive", {}))
        m.effects = {k: Counter({tuple(_tuplify(e)): c for e, c in v}) for k, v in d.get("effects", {}).items()}
        m.tried_level = {int(l): set(v) for l, v in d.get("tried_level", {}).items()}
        return m

    def summary(self) -> str:
        parts = []
        for k in sorted(self.n, key=lambda k: (-self.responsive[k], -self.n[k])):
            parts.append(f"{k}:{self.responsive[k]}/{self.n[k]}")
        return " ".join(parts[:12])


def _tuplify(x):
    return tuple(_tuplify(y) for y in x) if isinstance(x, list) else x
