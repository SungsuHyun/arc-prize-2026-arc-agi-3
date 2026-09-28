"""Click records and statistics keyed by (colour, region) and by button object.

One ClickRecord per click transition (cached on the transition): what was clicked (object / region), the change class,
whether the clicked object vanished, which object moved onto the click point (marker / selection), and the displacement of
every matched object. ClickStats aggregates records into the tables the predictor uses:
  hist[(colour, region)]     change classes            vanish[(colour, region)]  colour the object turned into
  cursor[(colour, region)]   marker colour             succ[button key]          old bbox -> new bbox of objects that moved
  coupled[(marker, colour)]  displacement ratio        dead[object key]          clicks that changed nothing / HUD only
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from arcnav.frame import Frame, masked_ascii

from .entities import Obj, Scene, build_scene, match


def _edge_only_diff(a, b, margin: int = 3) -> bool:
    for i in range(64):
        for j in range(64):
            if a[i][j] != b[i][j] and margin <= i < 64 - margin and margin <= j < 64 - margin:
                return False
    return True


def change_class(before: Frame, after: Frame) -> str:
    """'world' = something inside the playfield changed; 'hud' = only cells within 3px of the board edge or HUD strips changed; 'none'."""
    if before.ascii == after.ascii:
        return "none"
    if masked_ascii(before) == masked_ascii(after) or _edge_only_diff(before.grid, after.grid):
        return "hud"
    return "world"


@dataclass
class ClickRecord:
    row: int
    col: int
    color: int
    region: int
    obj: Optional[Obj]                 # clicked object in the before scene (None = clicked a region's floor)
    cls: str                           # none | hud | world
    vanish_to: Optional[int]           # colour the clicked object's cells became (all of them, nothing else in the playfield changed)
    marker: Optional[Obj]              # object that moved onto the click point (before-scene object)
    marker_to: Optional[tuple] = None  # its new bbox
    marker_sel: bool = False           # it landed on the clicked object (selection) rather than on floor
    moved: list = field(default_factory=list)      # [(oa, ob, dr, dc)]
    appeared: list = field(default_factory=list)
    disappeared: list = field(default_factory=list)
    scene_before: object = None                    # before scene (for whole-colour shift rules)
    effect: Optional[tuple] = None                 # effect summary (see effect_summary), world clicks only
    occ_before: Optional[dict] = None              # bbox -> (colour, shape) of every non-HUD object (buttons only)
    occ_after: Optional[dict] = None


def effect_summary(sa: Scene, sb: Scene, m: Optional[dict] = None) -> tuple:
    """Coarse, position-free summary of what a transition did to the playfield: per colour, whether its
    total non-HUD pixel count grew/shrank and the dominant displacement of its matched objects.
    Tuple of (colour, sign, (dr, dc) | None), sorted. Empty tuple = nothing changed outside the HUD."""
    if m is None:
        m = match(sa, sb)
    px_b = Counter(); px_a = Counter()
    for o in sa.objs:
        if not o.hud and o.size >= 2:
            px_b[o.color] += o.size
    for o in sb.objs:
        if not o.hud and o.size >= 2:
            px_a[o.color] += o.size
    mv = defaultdict(Counter)
    for oa, ob, dr, dc in m["moved"]:
        if oa.size >= 2 and not oa.hud:
            mv[oa.color][(dr, dc)] += 1
    out = []
    for k in set(px_b) | set(px_a) | set(mv):
        d = px_a.get(k, 0) - px_b.get(k, 0)
        sign = 1 if d > 0 else -1 if d < 0 else 0
        move = mv[k].most_common(1)[0][0] if mv.get(k) else None
        if sign or move:
            out.append((k, sign, move))
    return tuple(sorted(out, key=lambda x: (x[0], x[1], x[2] or (0, 0))))


def effect_text(eff: tuple) -> str:
    parts = []
    for k, sign, move in eff:
        p = f"colour {k} " + ("grows" if sign > 0 else "shrinks" if sign < 0 else "")
        if move:
            p = (p + " and " if sign else p) + f"moves ({move[0]:+},{move[1]:+})"
        parts.append(p.strip())
    return ", ".join(parts) if parts else "nothing in the playfield changes"


def key_effect(t) -> Optional[tuple]:
    """Effect summary of a key transition (cached on the transition); None when it changed nothing / HUD only."""
    if hasattr(t, "_eff"):
        return t._eff
    eff = None
    if change_class(t.before_frame, t.after_frame) == "world":
        eff = effect_summary(build_scene(t.before_frame), build_scene(t.after_frame))
    t._eff = eff
    return eff


def record(t) -> ClickRecord:
    """Compute (and cache on the transition) the click record."""
    rec = getattr(t, "_rec", None)
    if rec is not None:
        return rec
    r, c = t.action["row"], t.action["col"]
    sa, sb = build_scene(t.before_frame), build_scene(t.after_frame)
    colour = t.before_frame.grid[r][c]
    obj = sa.obj_around(r, c); region = obj.region if obj else sa.label[r][c]
    cls = change_class(t.before_frame, t.after_frame)
    m = match(sa, sb) if cls == "world" else {"pairs": [], "moved": [], "appeared": [], "disappeared": []}
    vanish = None
    hit = sa.obj_at(r, c)
    if cls == "world" and hit is not None and hit.cells and any(o.id == hit.id for o in m["disappeared"]):
        obj_v = hit
        after = t.after_frame.grid; cs = set(obj_v.cells)
        cols = {after[rr][cc] for rr, cc in obj_v.cells}
        if len(cols) == 1 and all(t.before_frame.grid[i][j] == after[i][j] or (i, j) in cs for i in range(3, 61) for j in range(3, 61)):
            vanish = cols.pop()
    marker = None; marker_to = None; sel = False
    for oa, ob, dr, dc in sorted(m["moved"], key=lambda x: -x[0].size):
        if oa.size >= 4 and ob.bbox[0] <= r <= ob.bbox[2] and ob.bbox[1] <= c <= ob.bbox[3]:
            marker, marker_to = oa, ob.bbox
            sel = obj is not None and obj.size >= 4 and obj.id != oa.id
            break
    occ_b = occ_a = None
    def _occ(sc: Scene, skip: int = -1) -> dict:
        return {o.bbox: (o.color, o.shape) for o in sc.objs
                if not o.hud and o.id != skip and o.size >= 2 and 1 < o.center[0] < 62 and 1 < o.center[1] < 62}
    if cls == "world" and obj is not None and obj.size >= 4 and marker is None and m["moved"]:
        occ_b = _occ(sa, obj.id); occ_a = _occ(sb)
    eff = effect_summary(sa, sb, m) if cls == "world" else None
    rec = ClickRecord(r, c, colour, region, obj, cls, vanish, marker, marker_to, sel, m["moved"], m["appeared"], m["disappeared"], sa, eff, occ_b, occ_a)
    try:
        t._rec = rec
    except Exception:
        pass
    return rec


def _ratio(a: int, b: int) -> Optional[Fraction]:
    if b == 0:
        return None if a != 0 else Fraction(0)
    return Fraction(a, b)


class ClickStats:
    def __init__(self):
        self.hist: dict = defaultdict(Counter)
        self.vanish: dict = defaultdict(Counter)
        self.cursor: dict = defaultdict(Counter)        # (colour, region) -> Counter(marker colour | -1)
        self.select: dict = defaultdict(Counter)        # (colour, region) -> Counter(True/False): marker landed on the clicked object
        self.presses: dict = defaultdict(list)          # button key -> [(occ_before, occ_after)] with occ = {bbox: (colour, shape)}
        self.succ_n: Counter = Counter()                # button key -> clicks observed
        self._sigma_cache: dict = {}
        self.coupled: dict = defaultdict(Counter)       # (marker colour, other colour) -> Counter((ratio_r, ratio_c))
        self.coupled_obs: dict = defaultdict(list)      # (marker colour, other colour) -> [((marker dr, dc), (other dr, dc))]
        self.shift: dict = defaultdict(Counter)         # (colour, region) -> Counter((moved colour, dr, dc)) for clicks that moved things (not marker clicks)
        self.shift_n: Counter = Counter()               # (colour, region) -> world-changing clicks observed
        self.effects: dict = defaultdict(Counter)       # effect key -> Counter(effect summary); keys: ("BTN", bbox) / ("CELL", colour, region) / ("KEY", name)
        self.effects_n: Counter = Counter()             # effect key -> observations (world-changing or not)
        self.dead: Counter = Counter()                  # object key -> none/hud clicks
        self.clicked_at: Counter = Counter()            # (region, bbox) -> clicks on this level, whatever the object's colour was
        self.alive: Counter = Counter()                 # object key -> clicks that changed the playfield
        self.n = 0

    def add(self, rec: ClickRecord) -> None:
        self.n += 1
        key = (rec.color, rec.region)
        self.hist[key][rec.cls] += 1
        self.vanish[key][rec.vanish_to if rec.vanish_to is not None else -1] += 1
        self.cursor[key][rec.marker.color if rec.marker else -1] += 1
        if rec.marker:
            self.select[key][rec.marker_sel] += 1
        if rec.obj is not None:
            (self.dead if rec.cls != "world" else self.alive)[rec.obj.key] += 1
            self.clicked_at[(rec.obj.region, rec.obj.bbox)] += 1
        ek = ("BTN", rec.obj.bbox) if rec.obj is not None else ("CELL", rec.color, rec.region)
        self.add_effect(ek, rec.effect if rec.cls == "world" else ())
        if rec.cls == "world" and rec.marker is None and rec.obj is not None:
            # a "shift" click: EVERY non-HUD object of colour k (>= 4 px) moved by the same (dr, dc)
            self.shift_n[key] += 1
            present = Counter(o.color for o in rec.scene_before.objs if not o.hud and o.size >= 4) if rec.scene_before is not None else Counter()
            per = defaultdict(Counter)
            for oa, ob, dr, dc in rec.moved:
                if oa.size >= 4:
                    per[oa.color][(dr, dc)] += 1
            for k, cnt in per.items():
                (dr, dc), m = cnt.most_common(1)[0]
                if m == present.get(k, 0) and m >= 1:
                    self.shift[key][(k, dr, dc)] += 1
        if rec.obj is not None and rec.moved and rec.marker is None and rec.occ_before is not None:
            b = rec.obj.key; self.succ_n[b] += 1
            self.presses[b].append((rec.occ_before, rec.occ_after)); self._sigma_cache.pop(b, None)
        if rec.marker is not None and not rec.marker_sel:
            mdr, mdc = rec.marker_to[0] - rec.marker.bbox[0], rec.marker_to[1] - rec.marker.bbox[1]
            for oa, ob, dr, dc in rec.moved:
                if oa.id == rec.marker.id or oa.size < 4 or oa.shape == rec.marker.shape or min(oa.bbox[2] - oa.bbox[0], oa.bbox[3] - oa.bbox[1]) < 2:
                    continue
                self.coupled_obs[(rec.marker.color, oa.color)].append(((mdr, mdc), (dr, dc)))
                rr, rc = _ratio(dr, mdr), _ratio(dc, mdc)
                if rr is not None and rc is not None:
                    self.coupled[(rec.marker.color, oa.color)][(rr, rc)] += 1

    # ── lookups with colour-only fallback ────────────────────────────────
    def _get(self, table: dict, colour: int, region: int):
        v = table.get((colour, region))
        if v:
            return v, True
        agg: Counter = Counter()
        for (c, rg), cnt in table.items():
            if c == colour:
                agg.update(cnt)
        return (agg if agg else None), False

    def hist_of(self, colour: int, region: int):
        return self._get(self.hist, colour, region)

    def vanish_of(self, colour: int, region: int):
        return self._get(self.vanish, colour, region)

    def cursor_of(self, colour: int, region: int):
        return self._get(self.cursor, colour, region)

    def coupling(self, marker_colour: int, other_colour: int):
        obs = self.coupled_obs.get((marker_colour, other_colour))
        if not obs or len(obs) < 3:
            return None
        best = None
        for ratio in self.coupled[(marker_colour, other_colour)]:
            rr, rc = ratio
            n = sum(1 for (mdr, mdc), (dr, dc) in obs if abs(round(float(rr) * mdr) - dr) <= 1 and abs(round(float(rc) * mdc) - dc) <= 1)
            if best is None or n > best[1]:
                best = (ratio, n)
        return best if best and best[1] * 10 >= 6 * len(obs) else None

    def sigma(self, button_key) -> dict:
        """Slot permutation of a button: slot -> next slot, solved from every recorded press by constraint propagation.
        Constraints: an occupied slot p sends its occupant to sigma(p) (same colour+shape there afterwards); an empty slot
        sends 'empty'. Identical blocks are no problem because only occupancy is used, not object identity."""
        if button_key in self._sigma_cache:
            return self._sigma_cache[button_key]
        presses = self.presses.get(button_key) or []
        slots: set = set()
        for occ_b, occ_a in presses:
            slots |= set(occ_b) | set(occ_a)
        cand = {}
        for p in slots:
            cs = set()
            for q in slots:
                ok = True
                for occ_b, occ_a in presses:
                    if occ_b.get(p) != occ_a.get(q):
                        ok = False; break
                if ok:
                    cs.add(q)
            cand[p] = cs
        changed = True; fixed = {}; self._tentative = getattr(self, "_tentative", {})
        while changed:
            changed = False
            for p, cs in cand.items():
                if p in fixed:
                    continue
                if len(cs) == 1:
                    q = next(iter(cs)); fixed[p] = q; changed = True
                    for p2, cs2 in cand.items():
                        if p2 != p and q in cs2 and p2 not in fixed:
                            cs2.discard(q)
        self._sigma_cache[button_key] = fixed
        # tie-break (tentative, never used for verdicts): among the remaining candidates the nearest slot when it is unique and close
        tentative = {}
        for p, cs in cand.items():
            if p in fixed or not cs:
                continue
            free = [q for q in cs if q not in fixed.values()]
            if not free:
                continue
            free.sort(key=lambda q: abs(q[0] - p[0]) + abs(q[1] - p[1]))
            d0 = abs(free[0][0] - p[0]) + abs(free[0][1] - p[1])
            if d0 <= 8 and (len(free) == 1 or abs(free[1][0] - p[0]) + abs(free[1][1] - p[1]) > d0):
                tentative[p] = free[0]
        self._sigma_cache[button_key] = fixed
        self._tentative[button_key] = tentative
        return fixed

    def tentative(self, button_key) -> dict:
        self.sigma(button_key)
        return self._tentative.get(button_key, {})

    def shifts_of(self, colour: int, region: int) -> list:
        """[(moved colour, dr, dc)] that happened in at least 60% of the world-changing clicks on (colour, region), >= 2 clicks."""
        n = self.shift_n.get((colour, region), 0)
        if n < 2:
            return []
        return [(k, dr, dc) for (k, dr, dc), m in self.shift[(colour, region)].items() if m * 10 >= 6 * n]

    def add_effect(self, ek, eff: Optional[tuple]) -> None:
        self.effects_n[ek] += 1
        if eff is not None:
            self.effects[ek][eff] += 1

    def effect_of(self, ek):
        """(effect summary, support, observations) when one summary explains >= 60% of >= 2 observations, else None."""
        n = self.effects_n.get(ek, 0)
        if n < 2 or not self.effects.get(ek):
            return None
        eff, m = self.effects[ek].most_common(1)[0]
        if m >= 2 and m * 10 >= 6 * n:
            return eff, m, n
        return None

    # (2) keys as buttons: a key that moves several objects (no avatar) gets the same occupancy/permutation treatment as a button
    def add_key_press(self, name: str, occ_b: dict, occ_a: dict) -> None:
        b = ("KEY", name)
        self.succ_n[b] += 1
        self.presses[b].append((occ_b, occ_a)); self._sigma_cache.pop(b, None)

    def is_dead(self, key) -> bool:
        return self.dead.get(key, 0) >= 2 and self.alive.get(key, 0) == 0

    def successor(self, button_key, colour: int, bbox: tuple):
        sg = self.sigma(button_key)
        return sg.get(bbox)

    def slots(self, button_key) -> set:
        out: set = set()
        for occ_b, occ_a in self.presses.get(button_key) or []:
            out |= set(occ_b) | set(occ_a)
        return out
