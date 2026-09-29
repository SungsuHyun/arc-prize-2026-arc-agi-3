"""wml/effects.py — effect-table world model: every trigger (a click target object, or a button) is described by the
JOINT change it produced in the frame (movers shifted, bars lengthened/shortened, objects appearing/vanishing),
learned from the log and applied as one rule per trigger. Resizes carry a built-in guard (a bar cannot shrink below
zero), which is what makes "feed the middle bar before pulling from it" plannable: the game refuses the click, and
so does the model.

Components of an effect (all level-specific positions, class-level signature in `meta` for the knowledge asset):
  ("move",   color, sig, dr, dc)                   every object of that colour+shape shifts
  ("resize", color, axis, lo, hi, anchor, delta)   the rectangle of `color` whose fixed-axis extent is (lo, hi) grows /
                                                   shrinks by `delta` along the other axis; `anchor` = the fixed edge
                                                   coordinate (a bar attached to the left border keeps c0 = anchor)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from ..core.contracts import Rule, RuleModel
from ..core.types import Action, Object, Scene, Transition, shape_signature
from ..env.calibrate import object_under
from ..probe.semantics import core_diff


def _is_rect(o: Object, scene: Optional[Scene] = None) -> bool:
    """A bar/panel: fills most of its bbox, or fills it together with the objects drawn over it (buttons on a bar)."""
    if o.mask is None or o.height < 2 or o.width < 2:
        return False
    fill = float(o.mask.mean())
    if fill >= 0.75:
        return True
    if fill < 0.4 or scene is None:
        return False
    r0, c0, r1, c1 = o.bbox
    cov = o.mask.copy()
    for q in scene.objects:
        if q.id == o.id:
            continue
        a0, b0, a1, b1 = q.bbox
        rr0, cc0, rr1, cc1 = max(r0, a0), max(c0, b0), min(r1, a1), min(c1, b1)
        if rr1 > rr0 and cc1 > cc0:
            cov[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] |= q.mask[rr0 - a0:rr1 - a0, cc0 - b0:cc1 - b0]
    return float(cov.mean()) >= 0.95


def _effective_bbox(o: Object, scene: Optional[Scene]) -> tuple:
    """A bar's bbox extended over objects drawn on top of its ends (a button that covers the first rows of a column):
    the bar geometrically continues under them."""
    r0, c0, r1, c1 = o.bbox
    if scene is None:
        return (r0, c0, r1, c1)
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    for q in scene.objects:
        if q.id == o.id or q.region in strips or q.area > 0.25 * o.area or max(q.height, q.width) > 8:
            continue          # a cap is a small object (button, marker); a wall spanning the bar's width is not
        a0, b0, a1, b1 = q.bbox
        # a cap drawn over the bar's end spans the bar's full thickness (a button on a column); a mover parked at the
        # bar's end does not, and must not be counted as bar
        cap_rows = (a1 == r0 or a0 == r1) and (b0, b1) == (c0, c1)
        cap_cols = (b1 == c0 or b0 == c1) and (a0, a1) == (r0, r1)
        if cap_rows or cap_cols:
            r0, c0, r1, c1 = min(r0, a0), min(c0, b0), max(r1, a1), max(c1, b1)
    return (r0, c0, r1, c1)


def _resize_component(ob: Object, oa: Optional[Object], sb: Optional[Scene] = None, sa: Optional[Scene] = None):
    """Describe a one-edge resize of a rectangle (or its disappearance) as a component, else None."""
    if not _is_rect(ob, sb):
        return None
    r0, c0, r1, c1 = _effective_bbox(ob, sb)
    if oa is None:
        # vanished: shrunk to zero along the free axis -- we do not know the axis; caller resolves with other observations
        return ("vanish", ob.color, (r0, c0, r1, c1))
    if not _is_rect(oa, sa) or oa.color != ob.color:
        return None
    s0, d0, s1, d1 = _effective_bbox(oa, sa)
    # delta is always the LENGTH change (positive = grows); anchor is the edge that stayed
    if (r0, r1) == (s0, s1) and (c0, c1) != (d0, d1):
        if c0 == d0:
            return ("resize", ob.color, "row", r0, r1, c0, (d1 - d0) - (c1 - c0))       # right edge moved
        if c1 == d1:
            return ("resize", ob.color, "row", r0, r1, c1, (d1 - d0) - (c1 - c0))       # left edge moved
    if (c0, c1) == (d0, d1) and (r0, r1) != (s0, s1):
        if r0 == s0:
            return ("resize", ob.color, "col", c0, c1, r0, (s1 - s0) - (r1 - r0))
        if r1 == s1:
            return ("resize", ob.color, "col", c0, c1, r1, (s1 - s0) - (r1 - r0))
    return None


def observe_effect(t: Transition) -> Optional[list[tuple]]:
    """Components of the change in one transition (None when nothing outside ui strips changed)."""
    c = core_diff(t)
    if not (c["moved"] or c["recolored"] or c["reshaped"] or c["appeared"] or c["disappeared"]):
        return None
    b = {o.id: o for o in t.before.objects}; a = {o.id: o for o in t.after.objects}
    comps: list[tuple] = []
    reshaped = {x[0] for x in c["reshaped"]}
    for oid, (dr, dc) in c["moved"]:
        if oid in reshaped:
            continue
        ob = b.get(oid)
        if ob is not None:
            comps.append(("move", ob.color, ob.shape_sig[:8], dr, dc))
    for oid in reshaped:
        ob, oa = b.get(oid), a.get(oid)
        if ob is None:
            continue
        rc = _resize_component(ob, oa, t.before, t.after)
        if rc is not None:
            comps.append(rc)
    for o in c["disappeared"]:
        rc = _resize_component(o, None, t.before, None)
        if rc is not None:
            comps.append(rc)
    for o in c["appeared"]:
        if _is_rect(o, t.after):
            comps.append(("appear", o.color, _effective_bbox(o, t.after)))
    return comps


def _trigger_key(t: Transition):
    a = t.action
    if a.type == "BUTTON":
        return ("BUTTON", a.id)
    if a.type == "CLICK":
        oid = object_under(t.before, a.row, a.col)
        o = t.before.get(oid) if oid is not None else None
        if o is None:
            return None
        return ("CLICK", o.color, o.shape_sig[:8], tuple(o.bbox))
    return None


def _resolve_vanish_appear(comps_by_obs: list[list[tuple]]) -> list[list[tuple]]:
    """A bar that vanished / appeared is the extreme case of a resize seen in the trigger's other observations: rewrite
    it with that resize's axis, extent and delta (its full length must equal |delta|)."""
    resizes = [c for obs in comps_by_obs for c in obs if c[0] == "resize"]
    out = []
    for obs in comps_by_obs:
        new = []
        for c in obs:
            if c[0] in ("vanish", "appear"):
                color, bbox = c[1], c[2]
                r0, c0, r1, c1 = bbox
                match = None
                for rz in resizes:
                    _, col, axis, lo, hi, anchor, delta = rz
                    if col != color:
                        continue
                    if axis == "row" and (lo, hi) == (r0, r1) and (c1 - c0) == abs(delta) and anchor in (c0, c1):
                        match = ("resize", color, "row", lo, hi, anchor, -abs(delta) if c[0] == "vanish" else abs(delta))
                    if axis == "col" and (lo, hi) == (c0, c1) and (r1 - r0) == abs(delta) and anchor in (r0, r1):
                        match = ("resize", color, "col", lo, hi, anchor, -abs(delta) if c[0] == "vanish" else abs(delta))
                    if match:
                        break
                if match:
                    new.append(match)
            else:
                new.append(c)
        out.append(new)
    return out


def learn_effect_table(log: list[Transition], min_obs: int = 2, prior_classes: Optional[dict] = None) -> dict:
    """trigger key -> list of components (the consistent joint effect), plus per-trigger evidence counts."""
    obs: dict = defaultdict(list); noops: Counter = Counter()
    for t in log:
        if t.action.type == "RESET" or t.status_change:
            continue
        k = _trigger_key(t)
        if k is None:
            continue
        comps = observe_effect(t)
        if comps is None:
            noops[k] += 1
        else:
            obs[k].append(comps)
    table: dict = {}
    for k, lst in obs.items():
        lst = _resolve_vanish_appear(lst)
        need = min_obs
        if prior_classes and k[0] == "CLICK" and f"c{k[1]}:{k[2]}" in prior_classes:
            need = 1
        if len(lst) < need:
            continue
        cnt: Counter = Counter()
        for comps in lst:
            for c in set(comps):
                cnt[c] += 1
        # components present in at least 80% of the observations (and in >= `need` of them)
        keep = [c for c, n in cnt.items() if n >= need and n >= 0.5 * len(lst)]   # majority: partial effects are guard-blocked cases (a mover at the edge)
        if not keep:
            continue
        table[k] = {"components": sorted(keep, key=str), "n": len(lst), "noops": noops.get(k, 0)}
    return table


def _apply_components(scene: Scene, comps: list[tuple]) -> Optional[Scene]:
    """Apply a joint effect. The guard: a resize that would shrink a bar below zero (or a bar that is not there to
    shrink) blocks the WHOLE effect -> the scene is returned unchanged (the game refuses the click)."""
    objs = list(scene.objects)
    # guard first
    for c in comps:
        if c[0] != "resize" or c[6] >= 0:
            continue
        _, color, axis, lo, hi, anchor, delta = c
        tgt = _find_bar(objs, color, axis, lo, hi, scene)
        if tgt is None or (tgt.width if axis == "row" else tgt.height) < abs(delta):
            return scene
    new_objs: list[Object] = []; consumed: set[int] = set(); extra: list[Object] = []
    others = list(objs)
    for c in comps:
        if c[0] == "move":
            _, color, sig, dr, dc = c
            for o in objs:
                if o.id in consumed or o.color != color or o.shape_sig[:8] != sig:
                    continue
                new_objs.append(o.moved(dr, dc)); consumed.add(o.id)
        elif c[0] == "resize":
            _, color, axis, lo, hi, anchor, delta = c
            tgt = _find_bar(objs, color, axis, lo, hi, scene)
            if tgt is None:
                if delta > 0:
                    extra.append(_make_bar(color, axis, lo, hi, anchor, delta, scene))
                continue
            consumed.add(tgt.id)
            r0, c0, r1, c1 = _effective_bbox(tgt, scene)
            if axis == "row":
                nb = (r0, c0, r1, c1 + delta) if anchor == c0 else (r0, c0 - delta, r1, c1)
            else:
                nb = (r0, c0, r1 + delta, c1) if anchor == r0 else (r0 - delta, c0, r1, c1)
            if nb[2] > nb[0] and nb[3] > nb[1]:
                res = _trim(tgt.with_mask(_bar_mask(nb, [q for q in others if q.id != tgt.id]), None, (nb[0], nb[1])))
                if res is not None:
                    new_objs.append(res)
    rest = [o for o in objs if o.id not in consumed]
    return scene.copy(objects=sorted(rest + new_objs + extra, key=lambda o: o.id))


def _bar_mask(nb: tuple, covering: list[Object]) -> np.ndarray:
    """Full rectangle minus the cells of objects drawn over it (buttons, markers)."""
    m = np.ones((nb[2] - nb[0], nb[3] - nb[1]), dtype=bool)
    for q in covering:
        a0, b0, a1, b1 = q.bbox
        rr0, cc0, rr1, cc1 = max(nb[0], a0), max(nb[1], b0), min(nb[2], a1), min(nb[3], b1)
        if rr1 > rr0 and cc1 > cc0:
            m[rr0 - nb[0]:rr1 - nb[0], cc0 - nb[1]:cc1 - nb[1]] &= ~q.mask[rr0 - a0:rr1 - a0, cc0 - b0:cc1 - b0]
    return m


def _trim(o: Object) -> Optional[Object]:
    """Tight bbox around the mask (an empty mask means the bar is gone)."""
    if not o.mask.any():
        return None
    ys, xs = np.nonzero(o.mask)
    r0, c0 = int(ys.min()), int(xs.min()); r1, c1 = int(ys.max()) + 1, int(xs.max()) + 1
    if (r0, c0, r1, c1) == (0, 0, o.mask.shape[0], o.mask.shape[1]):
        return o
    return o.with_mask(o.mask[r0:r1, c0:c1], None, (o.bbox[0] + r0, o.bbox[1] + c0))


def _resized_mask(o: Object, nb: tuple) -> np.ndarray:
    """The object's mask cropped / extended (with True cells) to the new bbox: holes it already has stay where they are."""
    r0, c0, r1, c1 = o.bbox
    m = np.ones((nb[2] - nb[0], nb[3] - nb[1]), dtype=bool)
    rr0, cc0 = max(r0, nb[0]), max(c0, nb[1]); rr1, cc1 = min(r1, nb[2]), min(c1, nb[3])
    if rr1 > rr0 and cc1 > cc0:
        m[rr0 - nb[0]:rr1 - nb[0], cc0 - nb[1]:cc1 - nb[1]] = o.mask[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    return m


def _find_bar(objs: list[Object], color: int, axis: str, lo: int, hi: int, scene: Optional[Scene] = None) -> Optional[Object]:
    for o in objs:
        if o.color != color or o.height < 2 or o.width < 2:
            continue
        r0, c0, r1, c1 = _effective_bbox(o, scene) if scene is not None else o.bbox
        if axis == "row" and (r0, r1) == (lo, hi):
            return o
        if axis == "col" and (c0, c1) == (lo, hi):
            return o
    return None


def _make_bar(color: int, axis: str, lo: int, hi: int, anchor: int, length: int, scene: Scene) -> Object:
    if axis == "row":
        bbox = (lo, anchor, hi, anchor + length) if anchor + length <= scene.grid_shape[1] else (lo, anchor - length, hi, anchor)
    else:
        bbox = (anchor, lo, anchor + length, hi) if anchor + length <= scene.grid_shape[0] else (anchor - length, lo, anchor, hi)
    m = np.ones((bbox[2] - bbox[0], bbox[3] - bbox[1]), dtype=bool)
    nid = max([o.id for o in scene.objects] + [0]) + 1000
    return Object(nid, color, (color,), bbox, m, int(m.sum()), shape_signature(m), "R0")


def class_signature(comps: list[tuple]) -> list:
    """Level-independent description of an effect: kinds, colours and magnitudes without positions."""
    out = []
    for c in comps:
        if c[0] == "move":
            out.append(["move", c[1], abs(c[3]) + abs(c[4])])
        elif c[0] == "resize":
            out.append(["resize", c[1], abs(c[6])])
    return sorted(out)


def induce_effect_model(log: list[Transition], semantics: dict, available: list[Action], knowledge=None,
                        click_map=None) -> list[tuple[str, RuleModel]]:
    prior = getattr(knowledge, "effect_rules", None) if knowledge is not None else None
    table = learn_effect_table(log, prior_classes=prior)
    if not table:
        return []
    cmap = click_map if click_map is not None else getattr(knowledge, "clicks", None)
    rules: list[Rule] = []
    for k, entry in table.items():
        comps = entry["components"]
        if k[0] == "BUTTON":
            bid = k[1]
            applies = (lambda s, a, bid=bid: a.type == "BUTTON" and a.id == bid)
            name = f"effect_ACTION{bid}"
        else:
            _, color, sig, bbox = k
            def applies(s, a, bbox=bbox):
                if a.type != "CLICK":
                    return False
                r0, c0, r1, c1 = bbox
                return r0 <= a.row < r1 and c0 <= a.col < c1
            name = f"effect_click_at_{bbox[0]}_{bbox[1]}"
        r = Rule(name, applies, (lambda s, a, comps=comps: _apply_components(s, comps)), source="induced")
        r.meta = {"kind": "effect", "trigger_class": (f"c{k[1]}:{k[2]}" if k[0] == "CLICK" else f"ACTION{k[1]}"), "signature": class_signature(comps),
                  "components": [list(c) for c in comps]}
        rules.append(r)
    explained_boxes = {k[3] for k in table if k[0] == "CLICK"}
    explained_buttons = {k[1] for k in table if k[0] == "BUTTON"}

    def unknown(scene: Scene, action: Action):
        # a trigger the click map knows to react but the table does not explain -> UNKNOWN; everything else is a no-op
        if action.type == "BUTTON":
            return None if (action.id not in explained_buttons and any(a.type == "BUTTON" and a.id == action.id for a in available)) else scene
        if action.type == "CLICK" and cmap is not None:
            from ..probe.clickmap import click_key
            oid = object_under(scene, action.row, action.col)
            o = scene.get(oid) if oid is not None else None
            if o is not None and tuple(o.bbox) in explained_boxes:
                return scene
            if cmap.status(click_key(scene, action.row, action.col)) == "responsive":
                return None
        return scene
    rules.append(Rule("effect_unknown", lambda s, a: True, unknown, source="induced"))

    def role_fn(scene: Scene) -> dict[int, str]:
        roles = {}
        for o in scene.objects:
            if tuple(o.bbox) in explained_boxes:
                roles[o.id] = f"custom:trigger_at_{o.bbox[0]}_{o.bbox[1]}"
        return roles
    model = RuleModel(rules, role_fn, default="noop", name="induced:effects")
    model.level_scoped = True          # positions of triggers and bars are level-specific
    return [("induced:effects", model)]
