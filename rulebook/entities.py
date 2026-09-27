"""Entities: regions (panels / floors / walls = large connected components of one colour), objects (the other components)
with the region that encloses them, and object matching between two frames (displacements, appearances, disappearances).

Everything the rulebook says about a click is keyed by (colour, region) instead of colour alone, so that the same colour
playing two roles in two panels (tn36's read-only template marks vs editable marks) yields two rules, not one contradiction.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from arcnav.frame import Frame

REGION_MIN = 150     # px: components at least this big are regions, smaller ones are objects
EDGE = 3             # px: a thin strip within this distance of the board edge is HUD


@dataclass
class Obj:
    id: int
    color: int
    size: int
    bbox: tuple            # (r0, c0, r1, c1)
    center: tuple          # (r, c)
    shape: int             # hash of the cell pattern relative to bbox
    region: int            # enclosing region id, -1 if none
    hud: bool
    cells: Optional[list] = None

    @property
    def key(self) -> tuple:
        """Identity within a level while the object does not move: (colour, region, bbox)."""
        return (self.color, self.region, self.bbox)

    def label(self) -> str:
        return f"#{self.id} colour {self.color} {self.size}px @({self.center[0]},{self.center[1]})" + (f" in P{self.region}" if self.region >= 0 else "")


@dataclass
class Region:
    id: int
    color: int
    size: int
    bbox: tuple
    edge: bool             # touches the board edge (border / frame)

    def label(self) -> str:
        r0, c0, r1, c1 = self.bbox
        return f"P{self.id} (colour {self.color} area rows {r0}-{r1} cols {c0}-{c1})"


@dataclass
class Scene:
    frame: Frame
    objs: list
    regions: list
    label: list            # 64x64 region id per cell (-1 = object / none)
    comp: list             # 64x64 component index per cell (objects only, -1 otherwise)

    def obj_at(self, r: int, c: int) -> Optional[Obj]:
        i = self.comp[r][c]
        return self.objs[i] if i >= 0 else None

    def obj_around(self, r: int, c: int, min_size: int = 4) -> Optional[Obj]:
        """The clicked object, or — when the click hit a speck (<= 2 px) — the smallest real object whose bbox contains it."""
        o = self.obj_at(r, c)
        if o is not None and o.size > 2:
            return o
        best = None
        for x in self.objs:
            if x.hud or x.size < min_size or (o is not None and x.id == o.id):
                continue
            r0, c0, r1, c1 = x.bbox
            if r0 <= r <= r1 and c0 <= c <= c1 and (best is None or x.size < best.size):
                best = x
        return best or o

    def region_at(self, r: int, c: int) -> int:
        o = self.obj_at(r, c)
        return o.region if o else self.label[r][c]

    def region_named(self, rid: int) -> Optional[Region]:
        return next((x for x in self.regions if x.id == rid), None)

    def by_region(self) -> dict:
        out: dict = defaultdict(list)
        for o in self.objs:
            out[o.region].append(o)
        return out


def _shape_hash(cells, r0, c0) -> int:
    return hash(tuple(sorted((r - r0, c - c0) for r, c in cells)))


def build_scene(frame: Frame) -> Scene:
    cached = getattr(frame, "_scene", None)
    if cached is not None:
        return cached
    grid = frame.grid; h, w = len(grid), len(grid[0])
    seen = [[-1] * w for _ in range(h)]
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] >= 0:
                continue
            color = grid[sr][sc]; cid = len(comps); stack = [(sr, sc)]; cells = []
            seen[sr][sc] = cid
            while stack:
                r, c = stack.pop(); cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and seen[nr][nc] < 0 and grid[nr][nc] == color:
                        seen[nr][nc] = cid; stack.append((nr, nc))
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            comps.append((color, cells, (min(rs), min(cs), max(rs), max(cs))))
    label = [[-1] * w for _ in range(h)]; comp = [[-1] * w for _ in range(h)]
    regions: list[Region] = []; objs: list[Obj] = []
    for color, cells, bbox in comps:
        r0, c0, r1, c1 = bbox
        thin = (r1 - r0 <= 2) or (c1 - c0 <= 2)
        hud = thin and (r0 < EDGE or r1 >= h - EDGE or c0 < EDGE or c1 >= w - EDGE) and len(cells) >= 8
        if len(cells) >= REGION_MIN and not hud:
            rid = len(regions)
            regions.append(Region(rid, color, len(cells), bbox, r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1))
            for r, c in cells:
                label[r][c] = rid
        else:
            oid = len(objs)
            objs.append(Obj(oid, color, len(cells), bbox, ((r0 + r1) // 2, (c0 + c1) // 2), _shape_hash(cells, r0, c0), -1, hud,
                            cells if len(cells) <= 400 else None))
            for r, c in cells:
                comp[r][c] = oid
    # enclosing region of an object: majority region label on the 1-px ring around its bbox
    for o in objs:
        r0, c0, r1, c1 = o.bbox; votes: Counter = Counter()
        for r in range(r0 - 1, r1 + 2):
            for c in (c0 - 1, c1 + 1):
                if 0 <= r < h and 0 <= c < w and label[r][c] >= 0:
                    votes[label[r][c]] += 1
        for c in range(c0, c1 + 1):
            for r in (r0 - 1, r1 + 1):
                if 0 <= r < h and 0 <= c < w and label[r][c] >= 0:
                    votes[label[r][c]] += 1
        if votes:
            o.region = votes.most_common(1)[0][0]
        else:   # ring is all objects (e.g. a mark inside a bigger object): inherit the neighbouring object's region later
            o.region = -1
    for o in objs:
        if o.region == -1:
            r0, c0, r1, c1 = o.bbox; votes = Counter()
            for r in range(r0 - 1, r1 + 2):
                for c in range(c0 - 1, c1 + 2):
                    if 0 <= r < h and 0 <= c < w and comp[r][c] >= 0 and comp[r][c] != o.id and objs[comp[r][c]].region >= 0:
                        votes[objs[comp[r][c]].region] += 1
            if votes:
                o.region = votes.most_common(1)[0][0]
    sc = Scene(frame, objs, regions, label, comp)
    try:
        frame._scene = sc
    except Exception:
        pass
    return sc


def match(a: Scene, b: Scene, max_dist: int = 64) -> dict:
    """Pair objects of scene a with objects of scene b: same colour and shape, nearest first (greedy on distance), so that
    identical blocks that all shift one slot are each paired with their nearest successor. Returns
    {'pairs': [(oa, ob)], 'moved': [(oa, ob, dr, dc)], 'appeared': [ob], 'disappeared': [oa]}."""
    groups_b: dict = defaultdict(list)
    for o in b.objs:
        if not o.hud:
            groups_b[(o.color, o.size, o.shape)].append(o)
    cands = []
    for oa in a.objs:
        if oa.hud:
            continue
        for ob in groups_b.get((oa.color, oa.size, oa.shape), []):
            d = abs(oa.center[0] - ob.center[0]) + abs(oa.center[1] - ob.center[1])
            if d <= max_dist:
                cands.append((d, oa.id, ob.id))
    cands.sort()
    used_a: set = set(); used_b: set = set(); pairs = []
    for d, ia, ib in cands:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia); used_b.add(ib); pairs.append((a.objs[ia], b.objs[ib]))
    moved = [(oa, ob, ob.bbox[0] - oa.bbox[0], ob.bbox[1] - oa.bbox[1]) for oa, ob in pairs if oa.bbox != ob.bbox]
    appeared = [o for o in b.objs if not o.hud and o.id not in used_b]
    disappeared = [o for o in a.objs if not o.hud and o.id not in used_a]
    return {"pairs": pairs, "moved": moved, "appeared": appeared, "disappeared": disappeared}


def scene_text(sc: Scene, limit_per_region: int = 14) -> str:
    """Compact entity table for prompts: regions, then objects grouped by region and colour."""
    lines = []
    for rg in sc.regions:
        lines.append(f"  {rg.label()}" + (" [border]" if rg.edge else ""))
    by = sc.by_region()
    for rid in sorted(by, key=lambda k: (k < 0, k)):
        objs = [o for o in by[rid] if not o.hud]
        if not objs:
            continue
        cols: dict = defaultdict(list)
        for o in objs:
            cols[o.color].append(o)
        head = f"  in P{rid}:" if rid >= 0 else "  (no region):"
        parts = []
        for c, os_ in sorted(cols.items(), key=lambda kv: -len(kv[1])):
            os_.sort(key=lambda o: (o.size, o.center))
            pos = ", ".join(f"#{o.id}@({o.center[0]},{o.center[1]}){o.size}px" for o in os_[:limit_per_region])
            parts.append(f"colour {c} x{len(os_)}: {pos}{' ...' if len(os_) > limit_per_region else ''}")
        lines.append(head + " " + "; ".join(parts))
    huds = [o for o in sc.objs if o.hud]
    if huds:
        lines.append("  HUD strips: " + ", ".join(f"colour {o.color} {o.size}px rows {o.bbox[0]}-{o.bbox[2]} cols {o.bbox[1]}-{o.bbox[3]}" for o in huds[:6]))
    return "\n".join(lines)
