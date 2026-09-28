"""perception/diff.py — Diff between two tracked scenes (spec §6 step 5)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Diff, Scene


def compute_diff(before: Scene, after: Scene, *, before_grid: Optional[np.ndarray] = None, after_grid: Optional[np.ndarray] = None,
                 noop_pixel_threshold: int = 0) -> Diff:
    b = {o.id: o for o in before.objects}; a = {o.id: o for o in after.objects}
    d = Diff()
    for oid, ob in b.items():
        oa = a.get(oid)
        if oa is None:
            d.disappeared.append(ob); continue
        if oa.bbox != ob.bbox and oa.shape_sig == ob.shape_sig:
            d.moved.append((oid, (oa.bbox[0] - ob.bbox[0], oa.bbox[1] - ob.bbox[1])))
        elif oa.bbox != ob.bbox or oa.shape_sig != ob.shape_sig:
            if oa.shape_sig != ob.shape_sig:
                d.reshaped.append((oid, ob.shape_sig, oa.shape_sig))
            if oa.bbox[:2] != ob.bbox[:2]:
                d.moved.append((oid, (oa.bbox[0] - ob.bbox[0], oa.bbox[1] - ob.bbox[1])))
        if oa.color != ob.color or oa.colors != ob.colors:
            d.recolored.append((oid, ob.color, oa.color))
    for oid, oa in a.items():
        if oid not in b:
            d.appeared.append(oa)
    if before_grid is not None and after_grid is not None:
        d.pixel_changes = int((np.asarray(before_grid) != np.asarray(after_grid)).sum())
    else:
        d.pixel_changes = int((before.render() != after.render()).sum())
    d.is_noop = d.pixel_changes <= noop_pixel_threshold and not (d.moved or d.appeared or d.disappeared or d.recolored or d.reshaped)
    return d
