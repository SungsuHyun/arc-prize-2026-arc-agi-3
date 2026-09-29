"""perception/segment.py — object segmentation (spec §6 steps 2-3): foreground pixels -> per-colour 4-connected
components; adjacent multi-colour components are kept as merge candidates; confirmed merges (from track.py) fuse
components into one multi-colour object."""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..core.types import Object, Region, shape_signature
from .regions import component_stats, label_components, smallest_region_for_bbox


def foreground_mask(grid: np.ndarray, regions: list[Region], global_bg: int, region_masks: dict, *, min_region_area: int = 0) -> np.ndarray:
    """A pixel is foreground when its colour differs from the background of the smallest region containing it.
    Global-background-coloured components inside a region that touch that region's bbox border are holes, not objects;
    enclosed ones are objects when small and background (a nested board) when at least min_region_area."""
    h, w = grid.shape
    fg = grid != global_bg
    # per non-global region: pixels of the region colour are background; global-bg pixels inside are holes unless enclosed
    for reg in regions:
        if reg.id == "R0" or reg.kind_hint == "ui_strip":
            continue
        r0, c0, r1, c1 = reg.bbox
        sub = grid[r0:r1, c0:c1]
        fg[r0:r1, c0:c1] &= sub != reg.bg_color
        if reg.bg_color == global_bg:
            continue          # nested floor of the background colour: its pixels are already background
        holes = sub == global_bg
        if holes.any():
            labels, n = label_components(sub, holes)
            if n:
                border = np.zeros(n, dtype=bool)
                for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
                    e = edge[edge >= 0]
                    border[e] = True
                if min_region_area > 0:
                    sizes = np.bincount(labels[labels >= 0].ravel(), minlength=n)
                    border |= sizes >= min_region_area          # a large enclosed area is a nested board, not an object
                enclosed = holes & ~border[np.clip(labels, 0, None)] & (labels >= 0)
                fg[r0:r1, c0:c1] |= enclosed
    return fg


def checker_mask(grid: np.ndarray) -> np.ndarray:
    """Pixels that belong to a 2x2-periodic two-colour texture (checkerboard hazards, dotted floors): every 2x2 block
    whose diagonals agree and whose rows differ marks its four pixels."""
    h, w = grid.shape
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    blk = (a == d) & (b == c) & (a != b)
    m = np.zeros((h, w), dtype=bool)
    m[:-1, :-1] |= blk; m[:-1, 1:] |= blk; m[1:, :-1] |= blk; m[1:, 1:] |= blk
    return m


def segment_objects(grid: np.ndarray, regions: list[Region], global_bg: int, region_masks: dict, *, connectivity: int = 4,
                    max_objects: int = 200, min_region_area: int = 0) -> tuple[list[Object], list[tuple[int, int]]]:
    """Returns (objects with provisional ids 0..n-1, adjacency pairs between different-colour components)."""
    h, w = grid.shape
    fg = foreground_mask(grid, regions, global_bg, region_masks, min_region_area=min_region_area)
    objs: list[Object] = []
    # 2x2-periodic textures become ONE multi-colour object each (a checkerboard is not 100 one-pixel objects)
    tex = checker_mask(grid) & (fg | (grid != global_bg))
    tex_labels, tn = label_components(np.zeros_like(grid), tex, 8)
    tex_stats = component_stats(tex_labels, tn) if tn else []
    tex_used = np.zeros((h, w), dtype=bool)
    a_, b_, c_, d_ = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    blk = (a_ == d_) & (b_ == c_) & (a_ != b_)
    for s in tex_stats:
        r0, c0, r1, c1 = s["bbox"]
        # a real texture is dense, at least 3x3, and MOST of its 2x2 windows are checker blocks (a piece inside four
        # corner marks only has checker blocks at the corners)
        if s["area"] < 12 or (r1 - r0) < 3 or (c1 - c0) < 3 or s["area"] < 0.75 * (r1 - r0) * (c1 - c0):
            continue
        windows = blk[r0:r1 - 1, c0:c1 - 1]
        if windows.size == 0 or windows.mean() < 0.6:
            continue
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool); m[s["rows"] - r0, s["cols"] - c0] = True
        cm = np.where(m, grid[r0:r1, c0:c1], -1).astype(np.int8)
        cols = sorted({int(v) for v in grid[s["rows"], s["cols"]]})
        color = max(cols, key=lambda cc: int((cm == cc).sum()))
        reg = smallest_region_for_bbox(regions, (r0, c0, r1, c1))
        objs.append(Object(len(objs), color, tuple(cols), (r0, c0, r1, c1), m, s["area"], shape_signature(m), reg.id, color_mask=cm))
        tex_used[s["rows"], s["cols"]] = True
    fg = fg & ~tex_used
    labels, n = label_components(grid, fg, connectivity)
    stats = component_stats(labels, n)
    base = len(objs)
    for s in stats:
        r0, c0, r1, c1 = s["bbox"]
        color = int(grid[s["rows"][0], s["cols"][0]])
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool); m[s["rows"] - r0, s["cols"] - c0] = True
        reg = smallest_region_for_bbox(regions, (r0, c0, r1, c1))
        objs.append(Object(base + s["id"], color, (color,), (r0, c0, r1, c1), m, s["area"], shape_signature(m), reg.id))
    # adjacency between different-colour components (merge candidates)
    adj: set[tuple[int, int]] = set()
    a = labels[:, :-1]; b = labels[:, 1:]
    sel = (a >= 0) & (b >= 0) & (a != b)
    for x, y in zip(a[sel].tolist(), b[sel].tolist()):
        adj.add((base + min(x, y), base + max(x, y)))
    a = labels[:-1, :]; b = labels[1:, :]
    sel = (a >= 0) & (b >= 0) & (a != b)
    for x, y in zip(a[sel].tolist(), b[sel].tolist()):
        adj.add((base + min(x, y), base + max(x, y)))
    if len(objs) > max_objects:
        objs.sort(key=lambda o: -o.area); objs = objs[:max_objects]
        keep = {o.id for o in objs}
        adj = {p for p in adj if p[0] in keep and p[1] in keep}
    return objs, sorted(adj)


def fuse(objects: list[Object], grid, regions: list[Region]) -> Object:
    """Fuse several components into one multi-colour object (keeps the smallest id)."""
    r0 = min(o.bbox[0] for o in objects); c0 = min(o.bbox[1] for o in objects)
    r1 = max(o.bbox[2] for o in objects); c1 = max(o.bbox[3] for o in objects)
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    cmask = np.full((r1 - r0, c1 - c0), -1, dtype=np.int8)
    for o in objects:
        a0, b0, a1, b1 = o.bbox
        sub = mask[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
        sub |= o.mask
        cm = cmask[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
        cm[o.mask] = o.color if o.color_mask is None else o.color_mask[o.mask]
    counts = Counter()
    for o in objects:
        counts[o.color] += o.area
    color = counts.most_common(1)[0][0]
    colors = tuple(sorted(counts))
    reg = smallest_region_for_bbox(regions, (r0, c0, r1, c1))
    parts = [p for o in objects for p in (o.parts or [o])]
    return Object(min(o.id for o in objects), int(color), colors, (r0, c0, r1, c1), mask, int(mask.sum()), shape_signature(mask), reg.id,
                  color_mask=cmask, parts=parts)


def parts_adjacent(a: Object, b: Object) -> bool:
    """Do two components touch (4-neighbourhood) anywhere?"""
    r0 = max(a.bbox[0], b.bbox[0]) - 1; c0 = max(a.bbox[1], b.bbox[1]) - 1
    r1 = min(a.bbox[2], b.bbox[2]) + 1; c1 = min(a.bbox[3], b.bbox[3]) + 1
    if r1 <= r0 or c1 <= c0:
        return False
    def paste(o):
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        ar0, ac0 = max(o.bbox[0], r0), max(o.bbox[1], c0); ar1, ac1 = min(o.bbox[2], r1), min(o.bbox[3], c1)
        if ar1 > ar0 and ac1 > ac0:
            m[ar0 - r0:ar1 - r0, ac0 - c0:ac1 - c0] = o.mask[ar0 - o.bbox[0]:ar1 - o.bbox[0], ac0 - o.bbox[1]:ac1 - o.bbox[1]]
        return m
    ma, mb = paste(a), paste(b)
    grown = ma.copy()
    grown[1:, :] |= ma[:-1, :]; grown[:-1, :] |= ma[1:, :]; grown[:, 1:] |= ma[:, :-1]; grown[:, :-1] |= ma[:, 1:]
    return bool((grown & mb).any())


def rect_composites(objs: list[Object], adj: Optional[list[tuple[int, int]]], regions: list[Region], excluded=(), *,
                    min_side: int = 2, min_area: int = 6, max_parts: int = 6, min_share: float = 0.2) -> list[list[Object]]:
    """Adjacent different-colour components whose union is a FILLED rectangle: a two-colour target pattern, a half-stamped
    canvas, a framed button. Thin parts (1 px lines, corner marks), parts under `min_share` of the union (a marker on a bar)
    and ui strips never take part; `excluded` are parts seen moving on their own (an agent standing next to a wall is not a pattern). Adjacency is taken from the segmentation
    when given, else from the masks."""
    by = {o.id: o for o in objs}
    strip = {r.id for r in regions if r.kind_hint == "ui_strip"}
    def ok(o: Object) -> bool:
        return o.id not in excluded and o.parts is None and o.area >= min_area and min(o.height, o.width) >= min_side and o.region not in strip
    elig = [o for o in objs if ok(o)]
    nbr: dict[int, set[int]] = {}
    if adj is None:
        pairs = [(a.id, b.id) for i, a in enumerate(elig) for b in elig[i + 1:] if parts_adjacent(a, b)]
    else:
        pairs = [(a, b) for a, b in adj if a in by and b in by and ok(by[a]) and ok(by[b])]
    for a, b in pairs:
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    def is_rect(ids) -> bool:
        os_ = [by[i] for i in ids]
        r0 = min(o.bbox[0] for o in os_); c0 = min(o.bbox[1] for o in os_); r1 = max(o.bbox[2] for o in os_); c1 = max(o.bbox[3] for o in os_)
        total = (r1 - r0) * (c1 - c0)
        # a small end-cap (a marker on a slider bar, a tab on a panel) is its own object, not a part of a pattern
        return sum(o.area for o in os_) == total and all(o.area >= min_share * total for o in os_)
    seen: set[int] = set(); out: list[list[int]] = []
    for start in sorted(nbr):
        if start in seen:
            continue
        stack = [start]; cl: set[int] = set()
        while stack:
            x = stack.pop()
            if x in cl:
                continue
            cl.add(x); stack.extend(nbr[x] - cl)
        seen |= cl
        if 2 <= len(cl) <= max_parts and is_rect(cl):
            out.append(sorted(cl)); continue
        used: set[int] = set()
        for a in sorted(cl):
            if a in used:
                continue
            for b in sorted(nbr[a]):
                if b not in used and is_rect({a, b}):
                    out.append([a, b]); used |= {a, b}; break
    return [[by[i] for i in ids] for ids in out]
