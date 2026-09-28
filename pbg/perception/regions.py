"""perception/regions.py — region segmentation (spec §6 step 1).

Regions are large single-colour blocks: every 4-connected component whose area is >= region_min_area_ratio of the grid
becomes a Region over its bounding box (R0 is always the global background covering the whole grid). Thin periodic
strips (thickness <= ui_strip_max_thickness) get their own Region with kind_hint 'ui_strip'."""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..core.types import Region


def label_components(grid: np.ndarray, mask: Optional[np.ndarray] = None, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label same-colour connected components (colour-uniform). Returns (labels (-1 = unlabelled), n)."""
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    use = np.ones((h, w), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    g = grid
    n = 0
    nbrs4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    nbrs = nbrs4 if connectivity == 4 else nbrs4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    rows, cols = np.nonzero(use)
    for sr, sc in zip(rows.tolist(), cols.tolist()):
        if labels[sr, sc] >= 0:
            continue
        color = g[sr, sc]
        labels[sr, sc] = n
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] < 0 and use[nr, nc] and g[nr, nc] == color:
                    labels[nr, nc] = n
                    stack.append((nr, nc))
        n += 1
    return labels, n


def component_stats(labels: np.ndarray, n: int) -> list[dict]:
    """Per component: bbox (exclusive), area, and the pixel index arrays."""
    out = []
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.searchsorted(sorted_labels, np.arange(n))
    ends = np.searchsorted(sorted_labels, np.arange(n), side="right")
    w = labels.shape[1]
    for i in range(n):
        idx = order[starts[i]:ends[i]]
        rr, cc = idx // w, idx % w
        out.append({"id": i, "area": int(len(idx)), "bbox": (int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1), "rows": rr, "cols": cc})
    return out


def _is_periodic(strip: np.ndarray) -> bool:
    """A 1-D colour sequence is periodic if some period p in 2..len/2 repeats at least twice (with tolerance at the end)."""
    seq = strip.tolist()
    n = len(seq)
    for p in range(2, n // 2 + 1):
        ok = all(seq[i] == seq[i - p] for i in range(p, n - (n % p)))
        if ok and n // p >= 2 and len(set(seq[:p])) > 1:
            return True
    return False


def find_regions(grid: np.ndarray, *, min_area_ratio: float = 0.05, ui_strip_max_thickness: int = 3) -> tuple[list[Region], int, dict]:
    """Returns (regions, global_bg, meta). meta has 'region_masks' {region_id: (bbox, mask)} for non-global regions."""
    h, w = grid.shape
    counts = Counter(grid.ravel().tolist())
    # the global background is the colour that dominates the grid border (a framed board keeps its frame colour as
    # background even when the board colour covers more pixels); ties -> overall count
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]).tolist()
    bc = Counter(border)
    global_bg = int(max(bc, key=lambda c: (bc[c], counts[c])))
    labels, n = label_components(grid)
    stats = component_stats(labels, n)
    min_area = max(4, int(min_area_ratio * h * w))
    regions: list[Region] = [Region("R0", global_bg, (0, 0, h, w), "board")]
    masks: dict[str, tuple] = {}
    big = [s for s in stats if s["area"] >= min_area]
    big.sort(key=lambda s: -s["area"])
    rid = 1
    for s in big:
        color = int(grid[s["rows"][0], s["cols"][0]])
        r0, c0, r1, c1 = s["bbox"]
        if color == global_bg and (r1 - r0) * (c1 - c0) >= 0.5 * h * w:
            continue  # the global background itself (a same-colour sub-block smaller than half the grid stays a panel)
        fill = s["area"] / ((r1 - r0) * (c1 - c0))
        if fill < 0.25:
            continue  # sparse (a frame/outline, not a block): treat as an object instead
        kind = "board" if (r1 - r0) * (c1 - c0) >= 0.5 * h * w else "panel"
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool); m[s["rows"] - r0, s["cols"] - c0] = True
        m |= grid[r0:r1, c0:c1] == color            # small same-colour gaps inside the bbox belong to the region too
        reg = Region(f"R{rid}", color, (r0, c0, r1, c1), kind, None if fill >= 0.98 else m)
        masks[reg.id] = ((r0, c0, r1, c1), m, fill)
        regions.append(reg); rid += 1
    # nested boards: a large area of the global background colour enclosed inside a region (the floor inside a framed
    # panel) is a region of its own, so rendering and floor rules see the floor colour there
    nested = []
    for s in stats:
        color = int(grid[s["rows"][0], s["cols"][0]])
        if color != global_bg or s["area"] < min_area:
            continue
        r0, c0, r1, c1 = s["bbox"]
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue          # touches the grid border: this is the outer background itself
        m = np.zeros((r1 - r0, c1 - c0), dtype=bool); m[s["rows"] - r0, s["cols"] - c0] = True
        fill = s["area"] / ((r1 - r0) * (c1 - c0))
        reg = Region(f"R{rid}", global_bg, (r0, c0, r1, c1), "board", None if fill >= 0.98 else m)
        masks[reg.id] = ((r0, c0, r1, c1), m, fill)
        nested.append(reg); rid += 1
    regions += nested
    # ui strips: thin, long components with a periodic pattern along their axis (any colour but the containing bg)
    for s in stats:
        r0, c0, r1, c1 = s["bbox"]
        th, ln = min(r1 - r0, c1 - c0), max(r1 - r0, c1 - c0)
        color = int(grid[s["rows"][0], s["cols"][0]])
        if color == global_bg:
            continue
        # anything lying entirely in the outermost row/column is a HUD element (counters that start at 1 px and grow)
        outermost = (r1 - r0 == 1 and (r0 == 0 or r1 == h)) or (c1 - c0 == 1 and (c0 == 0 or c1 == w))
        if not outermost and (th > ui_strip_max_thickness or ln < 12 or s["area"] < 6):
            continue
        line = grid[r0, c0:c1] if (c1 - c0) >= (r1 - r0) else grid[r0:r1, c0]
        at_edge = r0 <= 2 or c0 <= 2 or r1 >= h - 2 or c1 >= w - 2
        # periodic along the axis (tick marks), or a solid bar hugging a grid edge (gauge / timer)
        if outermost or _is_periodic(line) or at_edge:
            # a strip region spans the whole edge band so every piece of the gauge (bar, ticks, remainder) shares it
            if (c1 - c0) >= (r1 - r0):
                bbox = (r0, 0, r1, w)
            else:
                bbox = (0, c0, h, c1)
            if any(reg.kind_hint == "ui_strip" and reg.bbox == bbox for reg in regions):
                continue
            regions.append(Region(f"R{rid}", global_bg, bbox, "ui_strip")); rid += 1
    # merge overlapping strip bands into one
    strips = [r for r in regions if r.kind_hint == "ui_strip"]
    merged: list[Region] = []
    for r in sorted(strips, key=lambda r: r.bbox):
        for m in merged:
            if _bands_overlap(m.bbox, r.bbox):
                m.bbox = (min(m.bbox[0], r.bbox[0]), min(m.bbox[1], r.bbox[1]), max(m.bbox[2], r.bbox[2]), max(m.bbox[3], r.bbox[3]))
                break
        else:
            merged.append(r)
    regions = [r for r in regions if r.kind_hint != "ui_strip"] + merged
    remap = {}
    for i, r in enumerate(regions):
        remap[r.id] = f"R{i}"; r.id = f"R{i}"
    masks = {remap.get(k, k): v for k, v in masks.items()}
    return regions, global_bg, {"region_masks": masks, "labels": labels, "stats": stats}


def _bands_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def smallest_region_for_bbox(regions: list[Region], bbox: tuple[int, int, int, int]) -> Region:
    r0, c0, r1, c1 = bbox
    best = regions[0]
    for reg in regions:
        if reg.kind_hint == "ui_strip":
            R0, C0, R1, C1 = reg.bbox
            if R0 <= r0 and C0 <= c0 and r1 <= R1 and c1 <= C1 and reg.area < best.area:
                best = reg
            continue
        R0, C0, R1, C1 = reg.bbox
        cr, cc = (r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2
        if R0 <= cr < R1 and C0 <= cc < C1 and reg.area < best.area:
            best = reg
    return best


def stabilize_regions(cur: list[Region], prev: list[Region]) -> list[Region]:
    """Temporal stability (spec §15 occlusion): a region of the same colour seen in the previous scene keeps at least its
    previous extent (an object crossing a board can split its background component). Regions never shrink within a level."""
    out = []
    used = set()
    for r in cur:
        if r.id == "R0" or r.kind_hint == "ui_strip":
            out.append(r); continue
        best = None
        for q in prev:
            if q.id == "R0" or q.kind_hint == "ui_strip" or q.bg_color != r.bg_color or q.id in used:
                continue
            if _bands_overlap(q.bbox, r.bbox):
                best = q; break
        if best is not None:
            used.add(best.id)
            bb = (min(r.bbox[0], best.bbox[0]), min(r.bbox[1], best.bbox[1]), max(r.bbox[2], best.bbox[2]), max(r.bbox[3], best.bbox[3]))
            r = Region(r.id, r.bg_color, bb, "board" if best.kind_hint == "board" else r.kind_hint, _union_mask(bb, r, best))
        out.append(r)
    # previous regions with no counterpart this frame (fully occluded / split into small parts) are kept
    for q in prev:
        if q.id == "R0" or q.kind_hint == "ui_strip" or q.id in used:
            continue
        if not any(o.bg_color == q.bg_color and _bands_overlap(o.bbox, q.bbox) for o in out if o.id != "R0"):
            out.append(Region(q.id, q.bg_color, q.bbox, q.kind_hint, q.mask))
    # re-number non-strip regions by size, strips last (ids must be stable-ish: sort by bbox for determinism)
    strips = [r for r in out if r.kind_hint == "ui_strip"]; rest = [r for r in out if r.kind_hint != "ui_strip" and r.id != "R0"]
    rest.sort(key=lambda r: (-r.area, r.bbox))
    final = [out[0]] + rest + strips
    for i, r in enumerate(final):
        r.id = f"R{i}"
    return final


def _union_mask(bb, a: Region, b: Region):
    """Union of two regions' pixel masks inside the union bbox (None when both are full rectangles and equal to bb)."""
    if a.mask is None and b.mask is None and a.bbox == bb and b.bbox == bb:
        return None
    m = np.zeros((bb[2] - bb[0], bb[3] - bb[1]), dtype=bool)
    for reg in (a, b):
        r0, c0, r1, c1 = reg.bbox
        sub = m[r0 - bb[0]:r1 - bb[0], c0 - bb[1]:c1 - bb[1]]
        sub |= True if reg.mask is None else reg.mask
    return m
