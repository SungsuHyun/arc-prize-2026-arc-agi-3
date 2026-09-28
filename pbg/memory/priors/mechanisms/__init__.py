"""priors/mechanisms — reusable rule *effects* over Scene (spec §11, initial 30 mechanisms).

Every mechanism is a builder returning `effect(scene, action) -> Scene | None`; rules reference objects by ROLE, never
by id. LLM-written models import these by name (`from core.mechanisms import *` inside the sandbox maps here).
Direction convention: DIRS maps a button id to (dr, dc)."""
from __future__ import annotations

import inspect
from typing import Callable, Optional

from ....core.types import Action, Object, Scene

Effect = Callable[[Scene, Action], Optional[Scene]]
DIRS4 = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}      # ARC-AGI-3 convention: ACTION1 up, 2 down, 3 left, 4 right


def _roleset(x) -> frozenset:
    """A role argument may be one role or a collection of roles."""
    if x is None:
        return frozenset()
    if isinstance(x, str):
        return frozenset([x])
    return frozenset(x)


def _by_roles(scene: Scene, roles) -> list[Object]:
    rs = _roleset(roles)
    return [o for o in scene.objects if o.role in rs]


def _replace(scene: Scene, new: dict[int, Object]) -> Scene:
    return scene.copy(objects=[new.get(o.id, o) for o in scene.objects])


def _inside(scene: Scene, o: Object) -> bool:
    h, w = scene.grid_shape
    return 0 <= o.bbox[0] and 0 <= o.bbox[1] and o.bbox[2] <= h and o.bbox[3] <= w


def _region_of(scene: Scene, o: Object):
    return scene.region(o.region)


def _within_region(scene: Scene, o: Object) -> bool:
    reg = _region_of(scene, o)
    if reg is None or reg.id == "R0":
        return _inside(scene, o)
    r0, c0, r1, c1 = reg.bbox
    return o.bbox[0] >= r0 and o.bbox[1] >= c0 and o.bbox[2] <= r1 and o.bbox[3] <= c1


# ── movement ──
def move_role(role: str, dirs: dict[int, tuple[int, int]] = DIRS4, step: int = 1) -> Effect:
    """Shift every object of `role` by step * dirs[action.id] on a BUTTON action."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id not in dirs:
            return scene
        dr, dc = dirs[action.id]
        new = {o.id: o.moved(dr * step, dc * step) for o in _by_roles(scene, role)}
        s = _replace(scene, new)
        s.aux["_last_move"] = (dr * step, dc * step, role)
        return s
    return effect


def move_role_diagonal(role: str, dirs: dict[int, tuple[int, int]], step: int = 1) -> Effect:
    """Same as move_role with an explicit (possibly diagonal) direction table."""
    return move_role(role, dirs, step)


def rotate_role(role: str, button: int, k: int = 1) -> Effect:
    """Rotate the mask of every `role` object by k*90 degrees (about its bbox) on `button`."""
    import numpy as np
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id != button:
            return scene
        from ....core.types import shape_signature
        new = {}
        for o in _by_roles(scene, role):
            m = np.rot90(o.mask, k); r0, c0 = o.bbox[0], o.bbox[1]
            new[o.id] = Object(o.id, o.color, o.colors, (r0, c0, r0 + m.shape[0], c0 + m.shape[1]), m, o.area, shape_signature(m), o.region, o.role)
        return _replace(scene, new)
    return effect


def teleport_role(role: str, button: int, to_role: str) -> Effect:
    """On `button`, move each `role` object onto the first `to_role` object's position."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id != button:
            return scene
        tgt = _by_roles(scene, to_role)
        if not tgt:
            return scene
        tr, tc = tgt[0].bbox[0], tgt[0].bbox[1]
        new = {o.id: o.moved(tr - o.bbox[0], tc - o.bbox[1]) for o in _by_roles(scene, role)}
        return _replace(scene, new)
    return effect


def slide_until_blocked(role: str, blockers: tuple[str, ...] = ("wall",), dirs: dict[int, tuple[int, int]] = DIRS4, step: int = 1, max_steps: int = 64) -> Effect:
    """Inertial movement: keep moving `role` in the pressed direction until a blocker or the region edge (ice)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id not in dirs:
            return scene
        dr, dc = dirs[action.id]
        new = {}
        for o in _by_roles(scene, role):
            cur = o
            for _ in range(max_steps):
                nxt = cur.moved(dr * step, dc * step)
                if not _within_region(scene, nxt) or any(nxt.overlaps(b) for b in scene.objects if b.role in _roleset(blockers) and b.id != o.id):
                    break
                cur = nxt
            new[o.id] = cur
        return _replace(scene, new)
    return effect


# ── interaction ──
def cancel_move_if_overlap(mover: str, blocker: str) -> Effect:
    """Undo the last move of `mover` objects that now overlap a `blocker` object (wall collision)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if before is None:
            return scene
        blockers = _by_roles(scene, blocker)
        new = {}
        for o in _by_roles(scene, mover):
            if any(o.overlaps(b) for b in blockers if b.id != o.id):
                prev = before.get(o.id)
                if prev is not None:
                    p = prev.moved(0, 0); p.role = o.role; new[o.id] = p
        return _replace(scene, new)
    return effect


def cancel_move_if_outside(mover: str) -> Effect:
    """Undo the last move of `mover` objects that left their region (or the grid)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if before is None:
            return scene
        new = {}
        for o in _by_roles(scene, mover):
            if not _within_region(scene, o):
                prev = before.get(o.id)
                if prev is not None:
                    p = prev.moved(0, 0); p.role = o.role; new[o.id] = p
        return _replace(scene, new)
    return effect


def cancel_move_if_off_floor(mover: str, floor_colors: tuple[int, ...]) -> Effect:
    """Undo the move when any pixel under the mover (in the rendered *before* scene minus the mover) is not a floor colour."""
    import numpy as np
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if before is None:
            return scene
        movers = _by_roles(scene, mover)
        if not movers:
            return scene
        ids = {o.id for o in movers}
        floor = before.copy(objects=[o for o in before.objects if o.id not in ids]).render()
        h, w = floor.shape
        new = {}
        for o in movers:
            r0, c0, r1, c1 = o.bbox
            if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                bad = True
            else:
                under = floor[r0:r1, c0:c1][o.mask]
                bad = not np.isin(under, list(floor_colors)).all()
            if bad:
                prev = before.get(o.id)
                if prev is not None:
                    p = prev.moved(0, 0); p.role = o.role; new[o.id] = p
        return _replace(scene, new)
    return effect


def push_role(mover: str, pushable: str, blockers: tuple[str, ...] = ("wall",)) -> Effect:
    """A mover overlapping a pushable object after its move pushes it by the same displacement, unless the pushable
    would hit a blocker or another pushable (then the move is cancelled)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        lm = scene.aux.get("_last_move"); before = scene.aux.get("_before")
        if not lm or before is None:
            return scene
        dr, dc, _ = lm
        new = {}
        for o in _by_roles(scene, mover):
            for p in _by_roles(scene, pushable):
                if o.overlaps(p):
                    q = p.moved(dr, dc)
                    blocked = (not _within_region(scene, q)) or any(q.overlaps(b) for b in scene.objects if b.id not in (p.id, o.id) and (b.role in _roleset(blockers) or b.role in _roleset(pushable)))
                    if blocked:
                        prev = before.get(o.id)
                        if prev is not None:
                            pv = prev.moved(0, 0); pv.role = o.role; new[o.id] = pv
                    else:
                        new[p.id] = q
        return _replace(scene, new)
    return effect


def pull_role(mover: str, pullable: str) -> Effect:
    """After a move, a pullable object that was adjacent behind the mover follows it."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        lm = scene.aux.get("_last_move"); before = scene.aux.get("_before")
        if not lm or before is None:
            return scene
        dr, dc, _ = lm
        new = {}
        for o in _by_roles(scene, mover):
            prev = before.get(o.id)
            if prev is None:
                continue
            behind = prev.moved(-dr, -dc)
            for p in _by_roles(scene, pullable):
                if p.overlaps(behind):
                    new[p.id] = p.moved(dr, dc)
        return _replace(scene, new)
    return effect


def gravity(role: str, blockers: tuple[str, ...] = ("wall", "pushable"), step: int = 1, direction: tuple[int, int] = (1, 0)) -> Effect:
    """After every action, `role` objects fall in `direction` until blocked or at the region edge."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        dr, dc = direction
        new = {}
        for o in _by_roles(scene, role):
            cur = o
            for _ in range(64):
                nxt = cur.moved(dr * step, dc * step)
                if not _within_region(scene, nxt) or any(nxt.overlaps(b) for b in scene.objects if b.id != o.id and (b.role in _roleset(blockers) or b.role in _roleset(role))):
                    break
                cur = nxt
            new[o.id] = cur
        return _replace(scene, new)
    return effect


# ── state ──
def toggle_color(role: str, button: int, a: int, b: int) -> Effect:
    """On `button`, objects of `role` switch colour a<->b."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id != button:
            return scene
        new = {o.id: o.recolored(b if o.color == a else a) for o in _by_roles(scene, role) if o.color in (a, b)}
        return _replace(scene, new)
    return effect


def recolor_on_click(role: str, cycle: tuple[int, ...]) -> Effect:
    """Clicking a `role` object advances its colour along `cycle`."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK":
            return scene
        new = {}
        for o in _by_roles(scene, role):
            r0, c0, r1, c1 = o.bbox
            if r0 <= action.row < r1 and c0 <= action.col < c1 and o.mask[action.row - r0, action.col - c0] and o.color in cycle:
                new[o.id] = o.recolored(cycle[(cycle.index(o.color) + 1) % len(cycle)])
        return _replace(scene, new)
    return effect


def move_on_click(trigger_role: str, mover_role: str, dr: int, dc: int) -> Effect:
    """Clicking a `trigger_role` object shifts every `mover_role` object by (dr, dc) (arrow buttons, conveyor controls)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK":
            return scene
        hit = False
        for o in _by_roles(scene, trigger_role):
            r0, c0, r1, c1 = o.bbox
            if r0 <= action.row < r1 and c0 <= action.col < c1 and o.mask[action.row - r0, action.col - c0]:
                hit = True; break
        if not hit:
            return scene
        new = {o.id: o.moved(dr, dc) for o in _by_roles(scene, mover_role)}
        return _replace(scene, new)
    return effect


def set_flag_on_click(role: str, key: str = "selected") -> Effect:
    """Clicking a `role` object records its id in scene.aux[key] (selection state)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK":
            return scene
        for o in _by_roles(scene, role):
            r0, c0, r1, c1 = o.bbox
            if r0 <= action.row < r1 and c0 <= action.col < c1 and o.mask[action.row - r0, action.col - c0]:
                s = scene.copy(); s.aux[key] = o.id
                return s
        return scene
    return effect


def move_selected_on_click(key: str = "selected") -> Effect:
    """Clicking empty space moves the selected object (aux[key]) so that its top-left lands on the click."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK" or key not in scene.aux:
            return scene
        o = scene.get(scene.aux[key])
        if o is None:
            return scene
        s = _replace(scene, {o.id: o.moved(action.row - o.bbox[0], action.col - o.bbox[1])})
        s.aux.pop(key, None)
        return s
    return effect


def counter_step(key: str = "counter", delta: int = -1, only_when_changed: bool = True) -> Effect:
    """Hidden counter in aux[key] that changes by delta on every action (optionally only on non-noop ones)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if only_when_changed and before is not None and sorted(o.identity() for o in before.objects) == sorted(o.identity() for o in scene.objects):
            return scene
        s = scene.copy(); s.aux[key] = int(s.aux.get(key, 0)) + delta
        return s
    return effect


def shrink_strip(region_id: str, per_action: int = 1, axis: str = "col", only_when_changed: bool = False) -> Effect:
    """A ui_strip bar loses `per_action` cells per action (gauge / timer): the largest object inside the strip shrinks
    from its end (col axis: right end; row axis: bottom end)."""
    import numpy as np
    from ....core.types import shape_signature
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if only_when_changed and before is not None and sorted(o.identity() for o in before.objects) == sorted(o.identity() for o in scene.objects):
            return scene
        bars = [o for o in scene.objects if o.region == region_id]
        if not bars:
            return scene
        bar = max(bars, key=lambda o: o.area)
        r0, c0, r1, c1 = bar.bbox
        if axis == "col":
            if c1 - c0 <= per_action:
                return scene.copy(objects=[o for o in scene.objects if o.id != bar.id])
            m = bar.mask[:, :c1 - c0 - per_action]; bbox = (r0, c0, r1, c1 - per_action)
        else:
            if r1 - r0 <= per_action:
                return scene.copy(objects=[o for o in scene.objects if o.id != bar.id])
            m = bar.mask[:r1 - r0 - per_action, :]; bbox = (r0, c0, r1 - per_action, c1)
        nb = Object(bar.id, bar.color, bar.colors, bbox, m, int(m.sum()), shape_signature(m), bar.region, bar.role)
        return _replace(scene, {bar.id: nb})
    return effect


# ── creation / removal ──
def collect_on_overlap(mover: str, collectible: str, counter_key: Optional[str] = "collected") -> Effect:
    """A collectible overlapped by the mover disappears (and aux[counter_key] increments)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        movers = _by_roles(scene, mover)
        gone = {c.id for c in _by_roles(scene, collectible) if any(m.overlaps(c) for m in movers)}
        if not gone:
            return scene
        s = scene.copy(objects=[o for o in scene.objects if o.id not in gone])
        if counter_key:
            s.aux[counter_key] = int(s.aux.get(counter_key, 0)) + len(gone)
        return s
    return effect


def remove_on_click(role: str) -> Effect:
    """Clicking a `role` object removes it."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK":
            return scene
        keep = []
        for o in scene.objects:
            r0, c0, r1, c1 = o.bbox
            if o.role in _roleset(role) and r0 <= action.row < r1 and c0 <= action.col < c1 and o.mask[action.row - r0, action.col - c0]:
                continue
            keep.append(o)
        return scene.copy(objects=keep)
    return effect


def spawn_on_button(template_role: str, button: int, offset: tuple[int, int], new_role: str = "spawned") -> Effect:
    """On `button`, a copy of each `template_role` object appears at `offset` from it."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id != button:
            return scene
        nid = max([o.id for o in scene.objects] + [-1]) + 1
        added = []
        for o in _by_roles(scene, template_role):
            c = o.moved(*offset); c.id = nid; c.role = new_role; nid += 1; added.append(c)
        return scene.copy(objects=scene.objects + added)
    return effect


def hazard_kills(mover: str, hazard: str, key: str = "dead") -> Effect:
    """Overlapping a hazard marks aux[key] = True (the planner treats it as a dead end)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if any(m.overlaps(h) for m in _by_roles(scene, mover) for h in _by_roles(scene, hazard)):
            s = scene.copy(); s.aux[key] = True
            return s
        return scene
    return effect


# ── conditions ──
def key_opens_door(mover: str, key_role: str, door_role: str, flag: str = "has_key") -> Effect:
    """Collecting a key sets aux[flag]; with the flag, doors are removed when the mover touches them."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        movers = _by_roles(scene, mover)
        s = scene
        keys = [k for k in _by_roles(scene, key_role) if any(m.overlaps(k) for m in movers)]
        if keys:
            s = s.copy(objects=[o for o in s.objects if o.id not in {k.id for k in keys}]); s.aux[flag] = True
        if s.aux.get(flag):
            doors = [d for d in _by_roles(s, door_role) if any(m.overlaps(d) for m in movers)]
            if doors:
                s = s.copy(objects=[o for o in s.objects if o.id not in {d.id for d in doors}])
        return s
    return effect


def door_blocks_without_key(mover: str, door_role: str, flag: str = "has_key") -> Effect:
    """Doors act as walls until aux[flag] is set."""
    inner = cancel_move_if_overlap(mover, door_role)
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if scene.aux.get(flag):
            return scene
        return inner(scene, action)
    return effect


def switch_toggles_role(mover: str, switch_role: str, target_role: str, a: int, b: int) -> Effect:
    """Stepping on a switch recolours every target object a<->b."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        movers = _by_roles(scene, mover)
        on = [sw for sw in _by_roles(scene, switch_role) if any(m.overlaps(sw) for m in movers)]
        if not on:
            return scene
        if before is not None:
            was_on = [sw for sw in _by_roles(before, switch_role) if any(m.overlaps(sw) for m in _by_roles(before, mover))]
            if was_on:
                return scene
        new = {o.id: o.recolored(b if o.color == a else a) for o in _by_roles(scene, target_role) if o.color in (a, b)}
        return _replace(scene, new)
    return effect


def color_match_pass(mover: str, gate_role: str) -> Effect:
    """A gate blocks the mover unless they share a colour."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        before = scene.aux.get("_before")
        if before is None:
            return scene
        new = {}
        for o in _by_roles(scene, mover):
            for g in _by_roles(scene, gate_role):
                if o.overlaps(g) and g.color != o.color:
                    prev = before.get(o.id)
                    if prev is not None:
                        p = prev.moved(0, 0); p.role = o.role; new[o.id] = p
        return _replace(scene, new)
    return effect


def sequence_lock(role: str, order: tuple[int, ...], key: str = "seq") -> Effect:
    """Clicking `role` objects in colour `order` advances aux[key]; a wrong click resets it."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "CLICK":
            return scene
        for o in _by_roles(scene, role):
            r0, c0, r1, c1 = o.bbox
            if r0 <= action.row < r1 and c0 <= action.col < c1 and o.mask[action.row - r0, action.col - c0]:
                s = scene.copy(); i = int(s.aux.get(key, 0))
                s.aux[key] = i + 1 if i < len(order) and o.color == order[i] else 0
                return s
        return scene
    return effect


# ── ui ──
def lives_decrement_on_flag(flag: str = "dead", key: str = "lives", start: int = 3) -> Effect:
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if scene.aux.get(flag):
            s = scene.copy(); s.aux[key] = int(s.aux.get(key, start)) - 1; s.aux.pop(flag, None)
            return s
        return scene
    return effect


def highlight_selected(key: str = "selected", color: int = 10) -> Effect:
    """The selected object (aux[key]) is drawn in `color`."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        oid = scene.aux.get(key)
        o = scene.get(oid) if oid is not None else None
        if o is None or o.color == color:
            return scene
        s = _replace(scene, {o.id: o.recolored(color)}); s.aux[f"_{key}_orig_color"] = o.color
        return s
    return effect


def timer_tick(key: str = "timer", start: int = 100) -> Effect:
    return counter_step(key, -1, only_when_changed=False)


def noop_for(buttons: tuple[int, ...]) -> Effect:
    """Explicit no-op for the given buttons (keeps the prediction covered instead of UNKNOWN)."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        return scene
    return effect


def unknown_for(buttons: tuple[int, ...]) -> Effect:
    """Explicit UNKNOWN for buttons the model does not explain."""
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        return None if action.type == "BUTTON" and action.id in buttons else scene
    return effect


def split_on_button(role: str, button: int) -> Effect:
    """On `button`, each `role` object splits into its left/top and right/bottom halves."""
    import numpy as np
    from ....core.types import shape_signature
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        if action.type != "BUTTON" or action.id != button:
            return scene
        nid = max([o.id for o in scene.objects] + [-1]) + 1
        out = []
        for o in scene.objects:
            if o.role not in _roleset(role) or o.width < 2:
                out.append(o); continue
            r0, c0, r1, c1 = o.bbox; mid = (c1 - c0) // 2
            for j, (a, b) in enumerate(((0, mid), (mid, c1 - c0))):
                m = o.mask[:, a:b]
                if m.any():
                    out.append(Object(o.id if j == 0 else nid, o.color, o.colors, (r0, c0 + a, r1, c0 + b), m, int(m.sum()), shape_signature(m), o.region, o.role))
                    if j: nid += 1
        return scene.copy(objects=out)
    return effect


def merge_touching(role: str) -> Effect:
    """Objects of `role` that touch after an action become one object (id = smallest)."""
    import numpy as np
    from ....core.types import shape_signature
    def effect(scene: Scene, action: Action) -> Optional[Scene]:
        objs = _by_roles(scene, role)
        if len(objs) < 2:
            return scene
        groups: list[list[Object]] = []
        for o in objs:
            placed = None
            for g in groups:
                if any(o.moved(0, 0).overlaps(x.moved(1, 0)) or o.overlaps(x.moved(-1, 0)) or o.overlaps(x.moved(0, 1)) or o.overlaps(x.moved(0, -1)) or o.overlaps(x) for x in g):
                    if placed is None:
                        g.append(o); placed = g
                    else:
                        placed.extend(g); g.clear()
            if placed is None:
                groups.append([o])
        groups = [g for g in groups if g]
        if all(len(g) == 1 for g in groups):
            return scene
        out = [o for o in scene.objects if o.role not in _roleset(role)]
        for g in groups:
            if len(g) == 1:
                out.append(g[0]); continue
            r0 = min(o.bbox[0] for o in g); c0 = min(o.bbox[1] for o in g); r1 = max(o.bbox[2] for o in g); c1 = max(o.bbox[3] for o in g)
            m = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            for o in g:
                m[o.bbox[0] - r0:o.bbox[2] - r0, o.bbox[1] - c0:o.bbox[3] - c0] |= o.mask
            out.append(Object(min(o.id for o in g), g[0].color, g[0].colors, (r0, c0, r1, c1), m, int(m.sum()), shape_signature(m), g[0].region, role))
        return scene.copy(objects=out)
    return effect


MECHANISMS: dict[str, Callable] = {f.__name__: f for f in (
    move_role, move_role_diagonal, rotate_role, teleport_role, slide_until_blocked,
    cancel_move_if_overlap, cancel_move_if_outside, cancel_move_if_off_floor, push_role, pull_role, gravity,
    toggle_color, recolor_on_click, move_on_click, set_flag_on_click, move_selected_on_click, counter_step, shrink_strip,
    collect_on_overlap, remove_on_click, spawn_on_button, hazard_kills, split_on_button, merge_touching,
    key_opens_door, door_blocks_without_key, switch_toggles_role, color_match_pass, sequence_lock,
    lives_decrement_on_flag, highlight_selected, timer_tick, noop_for, unknown_for)}


def mechanism_signatures() -> str:
    """Signatures + one-line descriptions for prompts (bodies excluded, spec §14)."""
    lines = []
    for name, f in MECHANISMS.items():
        doc = ((inspect.getdoc(f) or "").splitlines() or [""])[0]
        lines.append(f"{name}{inspect.signature(f)}  # {doc}")
    return "\n".join(lines)


__all__ = list(MECHANISMS) + ["MECHANISMS", "DIRS4", "Effect", "mechanism_signatures"]
