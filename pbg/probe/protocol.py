"""probe/protocol.py — Probe (spec §7): a fixed exploration protocol, not random search, within a budget cap.

Initial protocol (game start):
  1. every button once in id order; 2. every non-noop button once more (consistency); 3. clicks: each object centre once,
  each region background centre once; 4. if nothing changed: triple-taps and ordered pairs; 5. RESET once if available
  (does it cost budget, what does it restore).
Local protocol (new level / after a prediction error): move next to novel objects and press each direction once;
reproduce the mismatching (state, action) once. Cap: 5% of the total budget."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.types import Action, Scene, Transition
from .semantics import ActionSemantics, classify_actions

BUTTON_ORDER = (1, 2, 3, 4, 5, 7)


@dataclass
class ProbeResult:
    transitions: list[Transition] = field(default_factory=list)
    semantics: ActionSemantics = field(default_factory=ActionSemantics)
    actions_used: int = 0
    stochastic: bool = False
    reset_costs_action: Optional[bool] = None
    reset_restores: Optional[bool] = None
    stopped_early: bool = False
    steps_done: list[str] = field(default_factory=list)


class Probe:
    """`session` must provide: act(action, kind) -> Transition | None (None = game ended / error), scene (current Scene),
    available_actions() -> list[Action], level, and status(). Budget caps are passed per run."""

    def __init__(self, session):
        self.s = session

    # ── helpers ──
    def _buttons(self) -> list[Action]:
        avail = {a.id for a in self.s.available_actions() if a.type == "BUTTON"}
        return [Action.button(i) for i in BUTTON_ORDER if i in avail]

    def _has_click(self) -> bool:
        return any(a.type == "CLICK" for a in self.s.available_actions())

    def _has_reset(self) -> bool:
        return getattr(self.s, "reset_allowed", True)

    def _click_targets(self, scene: Scene, limit: int) -> list[Action]:
        """Click targets in priority order: one object per (colour, shape) class (largest first), then region centres,
        then the remaining objects of already-seen classes. Independent of the budget size: the cheap, informative
        clicks always come first and `limit` only cuts the tail."""
        seen: set[tuple] = set(); first: list[Action] = []; rest: list[Action] = []
        strips = {r.id for r in scene.regions if r.kind_hint == "ui_strip"}
        for o in sorted(scene.objects, key=lambda o: (-o.area, o.id)):
            if o.region in strips or o.area < 2:
                continue
            key = (o.color, o.shape_sig)
            (rest if key in seen else first).append(Action.click(*o.center))
            seen.add(key)
        regions = [Action.click(*r.center) for r in scene.regions if r.kind_hint != "ui_strip"]
        return (first + regions + rest)[:limit]

    # ── protocols ──
    def run_initial(self, budget_cap: int, *, do_reset_probe: bool = True) -> ProbeResult:
        res = ProbeResult()
        done_level = self.s.level

        def act(a: Action, why: str) -> Optional[Transition]:
            if res.actions_used >= budget_cap or self.s.level != done_level or self.s.finished():
                res.stopped_early = True
                return None
            t = self.s.act(a, "initial_probe")
            if t is None:
                res.stopped_early = True
                return None
            res.actions_used += 1; res.transitions.append(t)
            return t

        buttons = self._buttons()
        # 1. each button once
        first: dict[str, Transition] = {}
        for a in buttons:
            t = act(a, "button-once")
            if t is None:
                break
            first[a.key] = t
        res.steps_done.append("buttons")
        # 2. every button once more (the state has changed since step 1, so a no-op may now act and vice versa)
        for a in buttons:
            if act(a, "button-repeat") is None:
                break
        res.steps_done.append("repeat")
        # 3. clicks (per-level absolute cap on top of the budget fraction: a board with 47 objects must not cost 47 clicks)
        if self._has_click():
            click_cap = min(budget_cap - res.actions_used - 2, int(getattr(self.s, "initial_click_cap", 16)))
            for a in self._click_targets(self.s.scene, max(1, click_cap)):
                if act(a, "click-target") is None:
                    break
            res.steps_done.append("clicks")
        # 4. nothing changed so far -> accumulation / combination mechanisms
        if res.transitions and all(t.diff.is_noop for t in res.transitions):
            for a in buttons:
                for _ in range(3):
                    if act(a, "triple") is None:
                        break
            for a in buttons:
                for b in buttons:
                    if a.key != b.key:
                        if act(a, "pair") is None or act(b, "pair") is None:
                            break
            res.steps_done.append("combos")
        # 5. reset once
        if do_reset_probe and self._has_reset() and not res.stopped_early and res.actions_used < budget_cap and self.s.level == done_level:
            before_hash = res.transitions[0].before.frame_hash if res.transitions else None
            used_before = self.s.actions_used()
            t = self.s.act(Action.reset(), "initial_probe")
            if t is not None:
                res.actions_used += 1
                res.reset_costs_action = self.s.actions_used() > used_before
                res.reset_restores = (before_hash is not None and t.after.frame_hash == before_hash)
                res.steps_done.append("reset")
        res.semantics = classify_actions(res.transitions)
        res.stochastic = self.s.status().stochasticity_score > 0.0
        return res

    def run_walk(self, budget_cap: int, *, max_per_dir: int = 12, agent_ids: set = frozenset(), dirs: Optional[dict] = None,
                 bumped: Optional[set] = None) -> ProbeResult:
        """Re-exploration after STUCK / no plan (spec §7 budget row 'reprobe'): first bump every object adjacent to the agent
        once (passability may have changed: doors, keys), then press each move button repeatedly until the board stops
        changing, then untried clicks. Reveals walls, reach and side effects."""
        res = ProbeResult()
        level = self.s.level
        bumped = bumped if bumped is not None else set()

        def act(a: Action) -> Optional[Transition]:
            if res.actions_used >= budget_cap or self.s.level != level or self.s.finished():
                res.stopped_early = True
                return None
            t = self.s.act(a, "reprobe")
            if t is None:
                res.stopped_early = True
                return None
            res.actions_used += 1; res.transitions.append(t)
            return t

        buttons = self._buttons()
        # 1. bump: step once toward each adjacent object not bumped before on this level (a door that opened, a key)
        if agent_ids and dirs:
            scene = self.s.scene
            agents = [o for o in scene.objects if o.id in agent_ids]
            for a in buttons:
                if a.id not in dirs:
                    continue
                dr, dc = dirs[a.id]
                for ag in agents:
                    ghost = ag.moved(dr, dc)
                    for o in scene.objects:
                        if o.id == ag.id or o.id in agent_ids or not ghost.overlaps(o):
                            continue
                        key = (level, o.identity())
                        if key in bumped:
                            continue
                        bumped.add(key)
                        if act(a) is None:
                            break
        for a in buttons:
            for _ in range(max_per_dir):
                t = act(a)
                if t is None or t.diff.is_noop or t.status_change:
                    break
        if self._has_click():
            for a in self._click_targets(self.s.scene, max(0, budget_cap - res.actions_used)):
                if act(a) is None:
                    break
        res.steps_done.append("walk")
        res.semantics = classify_actions(res.transitions)
        return res

    def run_local(self, budget_cap: int, *, novel_ids: list[int] = (), mismatch: Optional[Transition] = None,
                  navigate: Optional[Callable[[int], list[Action]]] = None) -> ProbeResult:
        res = ProbeResult()
        level = self.s.level

        def act(a: Action) -> Optional[Transition]:
            if res.actions_used >= budget_cap or self.s.level != level or self.s.finished():
                res.stopped_early = True
                return None
            t = self.s.act(a, "level_probe")
            if t is None:
                res.stopped_early = True
                return None
            res.actions_used += 1; res.transitions.append(t)
            return t

        buttons = self._buttons()
        for oid in list(novel_ids)[:3]:
            if navigate is not None:
                path = navigate(oid) or []
                for a in path[:max(0, budget_cap - res.actions_used - len(buttons))]:
                    if act(a) is None:
                        break
            for a in buttons:
                if act(a) is None:
                    break
        if mismatch is not None and self.s.scene.frame_hash == mismatch.before.frame_hash:
            act(mismatch.action)
        res.steps_done.append("local")
        res.semantics = classify_actions(res.transitions)
        return res
