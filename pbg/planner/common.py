"""planner/common.py — shared search utilities: action candidates, successor generation, progress heuristic."""
from __future__ import annotations

from typing import Callable, Optional

from ..core.types import Action, Scene


PASSIVE_ROLES = ("wall", "indicator", "decoration", "unknown", None)


def click_candidates(scene: Scene, extra: list[tuple[int, int]] = (), limit: int = 40) -> list[Action]:
    """Click coordinates: object centres (objects with an active role first) + region centres + previously responsive
    coordinates (spec §10)."""
    seen: set[tuple[int, int]] = set(); out: list[Action] = []
    strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
    triggers = [o for o in scene.objects if o.role and "trigger" in o.role]
    if triggers:
        # the model knows which objects react to clicks: plan over those (plus known responsive coordinates) only
        for o in triggers:
            if o.center not in seen:
                seen.add(o.center); out.append(Action.click(*o.center))
        for rc in extra:
            rc = (int(rc[0]), int(rc[1]))
            if rc not in seen:
                seen.add(rc); out.append(Action.click(*rc))
        return out[:limit]
    for o in sorted(scene.objects, key=lambda o: (o.role in PASSIVE_ROLES, -o.area, o.id)):
        if o.region in strips:
            continue
        c = o.center
        if c not in seen:
            seen.add(c); out.append(Action.click(*c))
    for r in scene.regions:
        if r.kind_hint != "ui_strip" and r.center not in seen:
            seen.add(r.center); out.append(Action.click(*r.center))
    for rc in extra:
        rc = (int(rc[0]), int(rc[1]))
        if rc not in seen:
            seen.add(rc); out.append(Action.click(*rc))
    return out[:limit]


def action_set(scene: Scene, available: list[Action], *, semantics: Optional[dict] = None, responsive: list[tuple[int, int]] = (),
               max_clicks: int = 40) -> list[Action]:
    noop = set()
    if semantics:
        noop = {k for k, v in semantics.items() if not k.startswith("_") and isinstance(v, dict) and v.get("class") == "NOOP" and v.get("n", 0) >= 2}
    out: list[Action] = []
    for a in available:
        if a.type == "BUTTON" and a.key not in noop:
            out.append(a)
        elif a.type == "CLICK":
            out.extend(click_candidates(scene, responsive, max_clicks))
    return out


def dead(scene: Scene) -> bool:
    return bool(scene.aux.get("dead"))


def make_successor(model, actions_fn: Callable[[Scene], list[Action]]):
    def successors(scene: Scene):
        for a in actions_fn(scene):
            try:
                nxt = model.predict(scene, a)
            except Exception:
                nxt = None
            if nxt is None or dead(nxt):
                continue
            yield a, nxt
    return successors
