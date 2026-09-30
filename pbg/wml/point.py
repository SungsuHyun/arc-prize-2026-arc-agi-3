"""wml/point.py — POINT-select games (docs/029 priority 3): a click is a destination, the piece steps toward it.

Neither the LLM nor the click inducers wrote r11l's rule ("click a cell -> the piece moves one step toward it"); the
mechanism `step_toward_click` existed but nothing instantiated it. This inducer reads the click transitions of the
log: for every object that moved on a click, does its displacement point from its centre toward the click? A colour/
shape class that does so on >= 2 clicks and on most of the clicks that moved it becomes `mover`; the step length and
the axis rule (4-connected vs diagonal) are read off the displacements. Two variants are returned so evaluate() can
pick: free movement, and static objects as walls (the step is cancelled on overlap)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from ..core.contracts import Rule, RuleModel
from ..core.types import Action, Scene, Transition
from ..memory.priors.mechanisms import jump_to_click, step_toward_click
from ..probe.clickmap import object_key
from ..probe.semantics import core_diff

MIN_OBS = 2
MIN_SHARE = 0.6


MODES = ("4", "diag", "jump")     # step along the major axis / step on both axes / the centre lands on the click


def _expected(o, action: Action, step: int, mode: str) -> tuple[int, int]:
    if mode == "jump":
        cr = (o.bbox[0] + o.bbox[2] - 1) // 2; cc = (o.bbox[1] + o.bbox[3] - 1) // 2
        return (int(action.row) - cr, int(action.col) - cc)
    cr = (o.bbox[0] + o.bbox[2] - 1) / 2.0; cc = (o.bbox[1] + o.bbox[3] - 1) / 2.0
    drr = action.row - cr; dcc = action.col - cc
    sr = (drr > 0) - (drr < 0); sc = (dcc > 0) - (dcc < 0)
    if mode == "diag":
        return (sr * step, sc * step)
    return (sr * step, 0) if abs(drr) >= abs(dcc) else (0, sc * step)


def induce_point_hypotheses(log: list[Transition], semantics: dict, available: list[Action], knowledge=None) -> list[tuple[str, RuleModel]]:
    clicks = [t for t in log if t.action.type == "CLICK"]
    if len(clicks) < MIN_OBS:
        return []
    moved_by_class: dict[str, int] = Counter()
    match: dict[tuple[str, int, str], int] = Counter()        # (class, step, mode) -> clicks where the displacement matched the mode
    for t in clicks:
        before = {o.id: o for o in t.before.objects}
        cd = core_diff(t)
        moves = [(oid, d, before.get(oid)) for oid, d in cd["moved"]]
        # a jump longer than the tracker's matching radius shows up as one object of a class vanishing and one appearing
        gone = {}; came = {}
        for o in cd["disappeared"]:
            gone.setdefault(object_key(o), []).append(o)
        for o in cd["appeared"]:
            came.setdefault(object_key(o), []).append(o)
        for kk in gone:
            if len(gone[kk]) == 1 and len(came.get(kk, [])) == 1:
                g, c = gone[kk][0], came[kk][0]
                moves.append((g.id, (c.bbox[0] - g.bbox[0], c.bbox[1] - g.bbox[1]), g))
        for oid, (dr, dc), o in moves:
            if o is None:
                continue
            k = object_key(o); moved_by_class[k] += 1
            step = max(abs(dr), abs(dc))
            for mode in MODES:
                if (dr, dc) == _expected(o, t.action, step, mode):
                    match[(k, 0 if mode == "jump" else step, mode)] += 1
    out: list[tuple[str, RuleModel]] = []
    for (k, step, mode), n in sorted(match.items(), key=lambda kv: -kv[1]):
        if n < MIN_OBS or n < MIN_SHARE * moved_by_class[k]:
            continue
        static = {object_key(o) for t in log for o in t.before.objects} - {k}      # classes that never were the mover
        never_moved = static - {object_key(before_o) for t in log for i, _ in core_diff(t)["moved"] for before_o in [dict((o.id, o) for o in t.before.objects).get(i)] if before_o is not None}

        def role_fn_factory(walls: frozenset):
            def role_fn(scene: Scene) -> dict[int, str]:
                roles = {}
                strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
                for o in scene.objects:
                    kk = object_key(o)
                    roles[o.id] = "mover" if kk == k else ("wall" if kk in walls and o.region not in strips else "unknown")
                return roles
            return role_fn
        for variant, walls in (("free", frozenset()), ("walls", frozenset(never_moved))):
            if variant == "walls" and not walls:
                continue
            effect = jump_to_click("mover", ("wall",)) if mode == "jump" else step_toward_click("mover", ("wall",), step=step, diagonal=(mode == "diag"))
            rule = Rule(f"point_{mode}_{variant}", lambda s, a: a.type == "CLICK", effect, source="induced",
                        meta={"class": k, "step": step, "mode": mode, "walls": sorted(walls)})
            model = RuleModel([rule], role_fn_factory(walls), default="noop", name=f"point_{mode}_s{step}_{variant}")
            model.point_game = True; model.point_step = max(1, step); model.point_mode = mode
            out.append((model.name, model))
        break            # the best-supported (class, step, axis rule) only; its two variants compete in evaluate()
    return out


def point_click_candidates(scene: Scene, step: int = 1, reach: int = 6) -> list[Action]:
    """Clicks that steer each mover: a cell `reach` away in each of the four directions (and the diagonals). For a jump
    model the planner also needs the centres of the other objects (the places a piece can be put)."""
    out: list[Action] = []
    for o in scene.objects:
        if o.role != "mover":
            continue
        r, c = o.center
        for dr, dc in ((-reach, 0), (reach, 0), (0, -reach), (0, reach), (-reach, -reach), (-reach, reach), (reach, -reach), (reach, reach)):
            rr, cc = max(0, min(63, r + dr)), max(0, min(63, c + dc))
            a = Action.click(rr, cc)
            if a not in out:
                out.append(a)
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    for o in scene.objects:
        if o.role != "mover" and o.region not in strips and o.area >= 2:
            a = Action.click(*o.center)
            if a not in out:
                out.append(a)
    return out[:40]
